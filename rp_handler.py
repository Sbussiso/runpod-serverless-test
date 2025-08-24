import os, json, time, tempfile
import runpod
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, HfApi, login
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True
BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"

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
    Loads a dataset from the HF Hub and maps it to a single 'text' column:
    '### Input: ...\n### Output: ...<|endoftext|>'
    """
    # Assumes HF_TOKEN is set for private datasets
    ds = load_dataset(dataset_id, split=split, revision=revision)

    def _fmt(example):
        inp = example[input_field]
        out = example[output_field]
        # Ensure both are JSON-safe strings
        try:
            out_str = json.dumps(out, ensure_ascii=False)
        except Exception:
            out_str = json.dumps(str(out), ensure_ascii=False)
        return {"text": f"### Input: {inp}\n### Output: {out_str}<|endoftext|>"}

    # Map in place without materializing a giant Python list
    ds = ds.map(_fmt, remove_columns=ds.column_names, num_proc=num_proc)
    return ds

def train_one_run(dataset: Dataset, output_dir: str):
    # load model
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_pin_memory=False,
            report_to="none",
        ),
    )

    # GPU stats
    gpu = torch.cuda.get_device_properties(0)
    start_reserved = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    res = trainer.train()
    used = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    lora_used = round(used - start_reserved, 3)
    pct = round(used / (gpu.total_memory/1024/1024/1024) * 100, 3)
    lpct = round(lora_used / (gpu.total_memory/1024/1024/1024) * 100, 3)

    # save LoRA adapter + tokenizer
    model.save_pretrained(os.path.join(output_dir, "finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "finetuned_model"))

    stats = {
        "train_runtime_s": float(res.metrics["train_runtime"]),
        "train_runtime_min": round(res.metrics["train_runtime"] / 60, 2),
        "gpu_name": gpu.name,
        "peak_reserved_gb": used,
        "peak_reserved_for_training_gb": lora_used,
        "peak_reserved_pct": pct,
        "peak_reserved_training_pct": lpct,
    }
    return stats

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

def handler(event):
    """
    Inputs:
      - dataset_id (str): e.g. "sbussiso/4chan-pairs"
      - split (str, optional): e.g. "train" (default)
      - revision (str, optional): HF commit SHA / tag for reproducibility
      - input_field (str, optional): default "input"
      - output_field (str, optional): default "output"
      - repo_name (str, optional): HF model repo name suffix (default "runpod_model")
    """
    inp = event.get("input") or {}
    dataset_id  = inp.get("dataset_id")             # REQUIRED
    split       = inp.get("split", "train")
    revision    = inp.get("revision")               # optional pin
    in_col      = inp.get("input_field", "input")
    out_col     = inp.get("output_field", "output")
    repo_name   = inp.get("repo_name")

    if not dataset_id:
        raise ValueError("dataset_id is required, e.g. 'sbussiso/4chan-pairs'.")

    # Login once so we can read private datasets and later push the model
    hf_login_if_available()

    # Workspace
    workdir = tempfile.mkdtemp(prefix="train_")
    out_dir = os.path.join(workdir, "outputs")

    # Load HF dataset by name
    ds = load_hf_dataset(
        dataset_id=dataset_id,
        split=split,
        revision=revision,
        input_field=in_col,
        output_field=out_col,
        num_proc=2,
    )

    # Train + upload
    stats = train_one_run(ds, out_dir)
    repo_id = upload_to_hf(os.path.join(out_dir, "finetuned_model"), repo_name)

    return {"ok": True, "repo_id": repo_id, "stats": stats}

runpod.serverless.start({"handler": handler})
