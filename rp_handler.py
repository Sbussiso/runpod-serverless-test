import os, json, time, tempfile, pathlib, requests
import runpod
import torch
from datasets import Dataset
from huggingface_hub import create_repo, HfApi, login
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True
BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"

def load_dataset(local_path: str | None, dataset_url: str | None):
    if dataset_url:
        # Download into a temp file (good for large sets)
        r = requests.get(dataset_url, timeout=60)
        r.raise_for_status()
        data = r.json()
    else:
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError("scraped_training_data.json not found and no dataset_url provided.")
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Your format
    def fmt(ex): return f"### Input: {ex['input']}\n### Output: {json.dumps(ex['output'])}<|endoftext|>"
    formatted = [fmt(x) for x in data]
    return Dataset.from_dict({"text": formatted})

def train_one_run(dataset: Dataset, output_dir: str):
    # load model
    t0 = time.time()
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

    # GPU stats before train
    gpu = torch.cuda.get_device_properties(0)
    start_reserved = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    res = trainer.train()
    # after
    used = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    lora_used = round(used - start_reserved, 3)
    pct = round(used / (gpu.total_memory/1024/1024/1024) * 100, 3)
    lpct = round(lora_used / (gpu.total_memory/1024/1024/1024) * 100, 3)

    # save LoRA adapter + tokenizer
    model.save_pretrained(output_dir + "/finetuned_model")
    tokenizer.save_pretrained(output_dir + "/finetuned_model")

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
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set.")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{repo_name or 'runpod_model'}"
    create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
    return repo_id

def handler(event):
    """
    Inputs (optional):
      - dataset_url: HTTP(S) link to scraped_training_data.json if not baked into the image
      - repo_name: Hugging Face repo suffix, defaults to 'runpod_model'
    """
    inp = event.get("input", {}) or {}
    dataset_url = inp.get("dataset_url")
    repo_name = inp.get("repo_name")

    # Prepare workspace
    workdir = tempfile.mkdtemp(prefix="train_")
    out_dir = os.path.join(workdir, "outputs")

    # Load dataset (from URL or local file)
    ds = load_dataset("/app/scraped_training_data.json", dataset_url)

    # Train
    stats = train_one_run(ds, out_dir)

    # Upload (push only the saved model folder)
    repo_id = upload_to_hf(os.path.join(out_dir, "finetuned_model"), repo_name)

    return {"ok": True, "repo_id": repo_id, "stats": stats}

runpod.serverless.start({"handler": handler})
