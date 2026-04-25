"""
train.py  — Fine-tuning with Unsloth + checkpoint save/resume support.

Usage:
    # Fresh training
    python scripts/train.py --config configs/train.yaml

    # Resume from last checkpoint
    python scripts/train.py --config configs/train.yaml --resume

    # Resume from specific checkpoint folder
    python scripts/train.py --config configs/train.yaml --resume --checkpoint_dir outputs/checkpoint-epoch-2
"""

import os, json, argparse, time
import yaml, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import EarlyStoppingCallback, TrainerCallback


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def df_to_dataset(df):
    return Dataset.from_pandas(df[["prompt"]].rename(columns={"prompt": "text"}))

def is_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# ──────────────────────────────────────────────────────────────────────────────
# Progress Bar Callback — works in Kaggle commit (background) log mode
# ──────────────────────────────────────────────────────────────────────────────
class TextProgressCallback(TrainerCallback):
    """
    Prints a Unicode block progress bar on every logging step.
    Example output:
      [Epoch 1/8 | Step  138/ 552] ████████████░░░░░░░░░░░░  50% | Loss: 1.2345 | ⏱ 00:12:34
    """
    BAR_LEN = 24

    def on_train_begin(self, args, state, control, **kwargs):
        self._total_steps = state.max_steps
        self._start_time  = time.time()
        print(f"\n{'─'*62}")
        print(f"  🚀 Training started | Total steps: {self._total_steps}")
        print(f"{'─'*62}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.max_steps == 0:
            return

        step      = state.global_step
        total     = state.max_steps
        epoch     = state.epoch or 0
        num_epochs = args.num_train_epochs

        loss      = logs.get("loss", logs.get("train_loss", None))
        lr        = logs.get("learning_rate", None)

        filled    = int(self.BAR_LEN * step / total)
        bar       = "█" * filled + "░" * (self.BAR_LEN - filled)
        pct       = step / total * 100

        elapsed   = time.time() - self._start_time
        h, rem    = divmod(int(elapsed), 3600)
        m, s      = divmod(rem, 60)
        time_str  = f"{h:02d}:{m:02d}:{s:02d}"

        loss_str  = f"Loss: {loss:.4f}" if loss is not None else ""
        lr_str    = f"| LR: {lr:.2e}" if lr is not None else ""

        print(
            f"  [Epoch {epoch:.0f}/{num_epochs:.0f} | Step {step:4d}/{total}] "
            f"{bar} {pct:5.1f}% | {loss_str} {lr_str} | ⏱ {time_str}",
            flush=True,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            val_loss = metrics.get("eval_loss", None)
            if val_loss:
                print(f"  {'─'*58}")
                print(f"  📊 Validation Loss: {val_loss:.4f} (Epoch {state.epoch:.0f} done)")
                print(f"  {'─'*58}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._start_time
        h, rem  = divmod(int(elapsed), 3600)
        m, s    = divmod(rem, 60)
        print(f"\n{'─'*62}")
        print(f"  ✅ Training finished! Total time: {h:02d}:{m:02d}:{s:02d}")
        print(f"{'─'*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def build_system_prompt(intent_list: list) -> str:
    """Build the system prompt containing all valid intent labels."""
    intents_str = "\n".join(f"- {intent}" for intent in intent_list)
    return (
        f"You are a banking intent classifier. "
        f"Given a customer query, classify it into exactly one of the "
        f"following {len(intent_list)} intent categories:\n"
        f"{intents_str}\n\n"
        f"Respond with ONLY the intent name, nothing else."
    )


def evaluate(model, tokenizer, test_df, max_seq_length, id2label):
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    model.eval()

    intent_list   = [id2label[str(i)] for i in range(len(id2label))]
    system_prompt = build_system_prompt(intent_list)
    intent_set    = set(intent_list)

    def make_inference_prompt(text: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def match_intent(raw: str) -> str:
        """Robust intent matching: exact → contains → fuzzy."""
        raw = raw.strip().lower()
        # 1. Exact match
        if raw in intent_set:
            return raw
        # 2. Starts-with or contains match
        for lbl in intent_set:
            if raw.startswith(lbl) or lbl in raw:
                return lbl
        # 3. Partial token overlap (pick best)
        raw_tokens = set(raw.replace("_", " ").split())
        best, best_score = raw, 0
        for lbl in intent_set:
            lbl_tokens = set(lbl.replace("_", " ").split())
            score = len(raw_tokens & lbl_tokens)
            if score > best_score:
                best, best_score = lbl, score
        return best

    true_labels, pred_labels = [], []
    print("\n[Eval] Generating predictions on test set...")
    for _, row in test_df.iterrows():
        prompt = make_inference_prompt(row["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        raw  = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = match_intent(raw)
        true_labels.append(row["label_name"].lower())
        pred_labels.append(pred)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n[Eval] Test Accuracy : {acc*100:.2f}%")
    report = classification_report(true_labels, pred_labels, zero_division=0)
    print("[Eval] Classification Report:")
    print(report)
    return acc, true_labels, pred_labels

# ──────────────────────────────────────────────────────────────────────────────
# Find latest checkpoint helper
# ──────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(output_dir):
    """Return path to the latest Hugging Face checkpoint folder, or None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by modification time — latest last
    checkpoints.sort(key=lambda p: os.path.getmtime(p))
    return checkpoints[-1]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(config_path, resume=False, checkpoint_dir=None):
    cfg = load_config(config_path)

    # Paths
    train_path     = cfg["data"]["train_path"]
    val_path       = cfg["data"].get("val_path", "sample_data/val.csv")
    test_path      = cfg["data"]["test_path"]
    label_map_path = cfg["data"]["label_map_path"]
    ckpt_save_path = cfg["checkpoint"]["save_path"]
    output_dir     = cfg["training"]["output_dir"]

    # Model
    model_name     = cfg["model"]["name"]
    max_seq_len    = cfg["model"]["max_seq_length"]
    load_4bit      = cfg["model"]["load_in_4bit"]

    # LoRA
    lora_r         = cfg["lora"]["r"]
    lora_alpha     = cfg["lora"]["lora_alpha"]
    lora_dropout   = cfg["lora"]["lora_dropout"]
    lora_bias      = cfg["lora"]["bias"]
    target_mods    = cfg["lora"]["target_modules"]

    # Training
    batch_size     = cfg["training"]["per_device_train_batch_size"]
    grad_accum     = cfg["training"]["gradient_accumulation_steps"]
    num_epochs     = cfg["training"]["num_train_epochs"]
    lr             = cfg["training"]["learning_rate"]
    warmup_ratio   = cfg["training"].get("warmup_ratio", 0.05)
    weight_decay   = cfg["training"]["weight_decay"]
    optimizer      = cfg["training"]["optimizer"]
    lr_scheduler   = cfg["training"].get("lr_scheduler_type", "cosine")
    logging_steps  = cfg["training"]["logging_steps"]
    save_strategy  = cfg["training"]["save_strategy"]

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("[1/6] Loading data...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df  = pd.read_csv(test_path)
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_labels = len(label_map["id2label"])
    has_val = val_df is not None
    print(f"  Train={len(train_df)} | Val={len(val_df) if has_val else 0} | Test={len(test_df)} | Intents={num_labels}")
    if has_val:
        print(f"  ✅ Validation set found — Early Stopping enabled")
    else:
        print(f"  ⚠️  No val.csv found — run preprocess_data.py first")

    # ── 2. Determine load path (fresh vs resume) ──────────────────────────
    from unsloth import FastLanguageModel

    if resume:
        load_path = checkpoint_dir or find_latest_checkpoint(output_dir)
        if load_path and os.path.isdir(load_path):
            print(f"\n[2/6] RESUMING from checkpoint: {load_path}")
        else:
            print(f"\n[2/6] No checkpoint found in '{output_dir}'. Starting fresh.")
            load_path = model_name
    else:
        load_path = model_name
        print(f"\n[2/6] Loading base model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_4bit,
    )
    print("  Model loaded.")

    # ── 3. LoRA (skip if resuming — adapters are already merged/saved) ────
    # When resuming from a PEFT checkpoint, get_peft_model re-applies adapters.
    # When resuming from outputs/checkpoint-N, HF Trainer handles it automatically.
    if not (resume and load_path != model_name):
        print(f"\n[3/6] Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_mods,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        model.print_trainable_parameters()
    else:
        print("\n[3/6] LoRA adapters loaded from checkpoint — skipping get_peft_model.")

    # ── 4. Dataset ────────────────────────────────────────────────────────
    print("\n[4/6] Preparing dataset...")
    train_dataset = df_to_dataset(train_df)
    val_dataset   = df_to_dataset(val_df) if has_val else None

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\n[5/6] Starting training...")
    from trl import SFTTrainer, SFTConfig

    bf16 = is_bf16()
    fp16 = not bf16

    # Determine resume_from_checkpoint value for Trainer
    resume_from = None
    if resume:
        candidate = checkpoint_dir or find_latest_checkpoint(output_dir)
        if candidate and os.path.isdir(candidate):
            resume_from = candidate

    # Evaluation strategy: only enable if we have a val set
    eval_strategy  = cfg["training"].get("eval_strategy", "epoch") if has_val else "no"
    load_best      = cfg["training"].get("load_best_model_at_end", True) if has_val else False
    metric_best    = cfg["training"].get("metric_for_best_model", "eval_loss") if has_val else None
    patience       = cfg["checkpoint"].get("early_stopping_patience", 2)

    sft_cfg = SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        optim=optimizer,
        lr_scheduler_type=lr_scheduler,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        output_dir=output_dir,
        save_strategy=save_strategy,
        save_total_limit=3,
        eval_strategy=eval_strategy,          # đánh giá mỗi epoch
        load_best_model_at_end=load_best,    # giữ model tốt nhất trên val
        metric_for_best_model=metric_best,   # dùng val_loss để so sánh
        report_to="none",
        seed=42,
        dataset_num_proc=2,
    )

    # Callbacks: TextProgressBar (always) + EarlyStopping (if val set exists)
    callbacks = [TextProgressCallback()]
    if has_val:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        print(f"  EarlyStopping enabled (patience={patience} epochs)")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,            # validation set
        args=sft_cfg,
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name} | VRAM: {gpu.total_memory/1024**3:.1f}GB")

    stats = trainer.train(resume_from_checkpoint=resume_from)
    train_loss = stats.training_loss
    print(f"\n  Training loss: {train_loss:.4f}")
    print(f"  Runtime      : {stats.metrics['train_runtime']:.1f}s")

    # ── 6. Evaluate + Save ────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set...")
    acc, true_labels, pred_labels = evaluate(model, tokenizer, test_df, max_seq_len, label_map["id2label"])

    # Save final checkpoint
    os.makedirs(ckpt_save_path, exist_ok=True)
    print(f"\nSaving final checkpoint to '{ckpt_save_path}' ...")
    model.save_pretrained(ckpt_save_path)
    tokenizer.save_pretrained(ckpt_save_path)

    # Save training metadata
    meta = {
        "model_name"      : model_name,
        "num_intents"     : num_labels,
        "max_seq_length"  : max_seq_len,
        "test_accuracy_pct": round(acc * 100, 2),
        "train_loss"      : round(train_loss, 4),
        "resume_used"     : resume,
        "hyperparameters" : {
            "lora_r"                   : lora_r,
            "lora_alpha"               : lora_alpha,
            "learning_rate"            : lr,
            "num_train_epochs"         : num_epochs,
            "batch_size"               : batch_size,
            "gradient_accumulation"    : grad_accum,
            "effective_batch_size"     : batch_size * grad_accum,
            "optimizer"                : optimizer,
            "lr_scheduler_type"        : lr_scheduler,
            "max_seq_length"           : max_seq_len,
        },
    }
    with open(os.path.join(ckpt_save_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  ✅ Training complete!")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Checkpoint    : {ckpt_save_path}/")
    print(f"  Intermediate  : {output_dir}/checkpoint-*/")
    print(f"{'='*50}")
    print(f"\nTo resume training, run:")
    print(f"  python scripts/train.py --config {config_path} --resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, default="configs/train.yaml")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Resume from a specific checkpoint folder (overrides auto-detect)")
    args = parser.parse_args()
    main(args.config, resume=args.resume, checkpoint_dir=args.checkpoint_dir)
