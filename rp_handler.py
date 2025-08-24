import os, json, time, tempfile
import runpod
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, HfApi, login
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# ----------------------------
# Config
# ----------------------------
MAX_SEQ_LEN   = 2048
DTYPE         = None
LOAD_IN_4BIT  = True
BASE_MODEL    = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"

# ----------------------------
# Live JSON logger for stdout
# ----------------------------
class LiveJSONLogger(TrainerCallback):
    """Emit compact JSON logs every logging step & eval step."""
    def __init__(self, seq_len: int, per_device_bs: int, gas: int):
        self.seq_len = seq_len
        self.per_device_bs = per_device_bs
        self.gas = gas
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self._last_t = None
        self._last_step = 0
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        step = int(state.global_step or 0)
        sps = None
        tps = None
        if self._last_t is not None and step > self._last_step:
            dt = max(1e-6, now - self._last_t)
            dstep = step - self._last_step
            sps = dstep / dt
            eff_bs = self.per_device_bs * self.gas * self.world
            tps = sps * eff_bs * self.seq_len

        payload = {
            "event": "train_log",
            "step": step,
            "epoch": float(state.epoch or 0.0),
            "loss": float(logs.get("loss")) if logs and "loss" in logs else None,
            "eval_loss": float(logs.get("eval_loss")) if logs and "eval_loss" in logs else None,
            "lr": float(logs.get("learning_rate")) if logs and "learning_rate" in logs else None,
            "steps_per_sec": round(sps, 3) if sps is not None else None,
            "tokens_per_sec_est": int(tps) if tps is not None else None,
            "elapsed_sec": int(now - self._t0),
        }
        print(json.dumps(payload), flush=True)
        self._last_t = now
        self._last_step = step

# ----------------------------
# Helpers
# ----------------------------
def hf_login_if_available():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok, add_to_git_credential=False)

def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    revision: str | None = None,
    input_field: str = "input",
    output_field: str = "output",
    num_proc: int = 2,
):
    """
    Load dataset from HF Hub and map to a single 'text' column:
    '### Input: ...\\n### Output: ...<|endoftext|>'
    """
    ds = load_dataset(dataset_id, split=split, revision=revision)

    def _fmt(example):
        inp = example[input_field]
        out = example[output_field]
        try:
            out_str = json.dumps(out, ensure_ascii=False)
        except Exception:
            out_str = json.dumps(str(out), ensure_ascii=False)
        return {"text": f"### Input: {inp}\n### Output: {out_str}<|endoftext|>"}

    ds = ds.map(_fmt, remove_columns=ds.column_names, num_proc=num_proc)
    return ds

def upload_to_hf(folder: str, repo_name: str | None):
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN env var not set.")
    login(token=tok, add_to_git_credential=False)
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{repo_name or 'runpod_model'}"
    create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
    return repo_id

def train_one_run(
    dataset_train: Dataset,
    dataset_eval: Dataset,
    output_dir: str,
    per_device_bs: int,
    gas: int,
    epochs: float,
    max_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_strategy: str,
):
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Build TrainingArguments (conditionally include save_steps)
    args_kwargs = dict(
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=gas,
        warmup_steps=10,
        num_train_epochs=epochs,
        max_steps=max_steps,  # -1 => use epochs
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=False,
    )
    if save_strategy == "steps" or save_steps > 0:
        args_kwargs["save_strategy"] = "steps"
        args_kwargs["save_steps"] = max(1, int(save_steps))

    args = TrainingArguments(**args_kwargs)

    live = LiveJSONLogger(seq_len=MAX_SEQ_LEN, per_device_bs=per_device_bs, gas=gas)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        args=args,
        callbacks=[live],
    )

    # GPU stats + train
    gpu = torch.cuda.get_device_properties(0)
    start_reserved = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    res = trainer.train()
    used = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    lora_used = round(used - start_reserved, 3)
    pct = round(used / (gpu.total_memory/1024/1024/1024) * 100, 3)
    lpct = round(lora_used / (gpu.total_memory/1024/1024/1024) * 100, 3)

    # Save adapter + tokenizer
    save_dir = os.path.join(output_dir, "finetuned_model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    stats = {
        "train_runtime_s": float(res.metrics.get("train_runtime", 0.0)),
        "train_runtime_min": round(float(res.metrics.get("train_runtime", 0.0)) / 60, 2),
        "gpu_name": gpu.name,
        "peak_reserved_gb": used,
        "peak_reserved_for_training_gb": lora_used,
        "peak_reserved_pct": pct,
        "peak_reserved_training_pct": lpct,
        "best_eval_loss": float(res.metrics.get("eval_loss", 0.0)) if "eval_loss" in res.metrics else None,
        "final_step": int(getattr(res, "global_step", 0)),
    }
    return stats, save_dir

# ----------------------------
# Serverless handler
# ----------------------------
def handler(event):
    """
    Inputs:
      - dataset_id (str): e.g. "sbussiso/4chan-pairs"
      - split (str): default "train"
      - revision (str): optional HF commit/tag
      - input_field (str): default "input"
      - output_field (str): default "output"
      - repo_name (str): HF repo suffix (default "runpod_model")
      - max_samples (int): cap training samples for quick tests (optional)

      Training knobs (optional):
      - num_train_epochs (float)  default 3
      - max_steps (int)           default -1 (use epochs)
      - per_device_train_batch_size (int) default 2
      - gradient_accumulation_steps (int) default 4
      - logging_steps (int)       default 5
      - eval_ratio (float)        default 0.01
      - eval_steps (int)          default 50
      - save_steps (int)          default 0 (disabled unless save_strategy="steps")
      - save_strategy (str)       "epoch" | "steps" (default "epoch")
    """
    inp = event.get("input") or {}

    dataset_id  = inp.get("dataset_id")
    split       = inp.get("split", "train")
    revision    = inp.get("revision")
    in_col      = inp.get("input_field", "input")
    out_col     = inp.get("output_field", "output")
    repo_name   = inp.get("repo_name")
    max_samples = int(inp.get("max_samples", -1))

    # Training knobs
    epochs      = float(inp.get("num_train_epochs", 3))
    max_steps   = int(inp.get("max_steps", -1))
    bsz         = int(inp.get("per_device_train_batch_size", 2))
    gas         = int(inp.get("gradient_accumulation_steps", 4))
    logging_steps = int(inp.get("logging_steps", 5))
    eval_ratio    = float(inp.get("eval_ratio", 0.01))
    eval_steps    = int(inp.get("eval_steps", 50))
    save_steps    = int(inp.get("save_steps", 0))
    save_strategy = inp.get("save_strategy", "epoch")

    if not dataset_id:
        raise ValueError("dataset_id is required, e.g. 'sbussiso/4chan-pairs'.")

    # Auth for private datasets & pushing models
    hf_login_if_available()

    # Workspace
    workdir = tempfile.mkdtemp(prefix="train_")
    out_dir = os.path.join(workdir, "outputs")

    # Load dataset
    ds = load_hf_dataset(
        dataset_id=dataset_id,
        split=split,
        revision=revision,
        input_field=in_col,
        output_field=out_col,
        num_proc=2,
    )
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    # Train/eval split
    splits = ds.train_test_split(test_size=eval_ratio, seed=3407)
    train_ds, eval_ds = splits["train"], splits["test"]

    # Train + upload
    stats, save_dir = train_one_run(
        dataset_train=train_ds,
        dataset_eval=eval_ds,
        output_dir=out_dir,
        per_device_bs=bsz,
        gas=gas,
        epochs=epochs,
        max_steps=max_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_strategy=save_strategy,
    )
    repo_id = upload_to_hf(save_dir, repo_name)

    return {"ok": True, "repo_id": repo_id, "stats": stats}

# Start serverless loop
runpod.serverless.start({"handler": handler})
