"""
inference.py
------------
Standalone inference module for the fine-tuned banking intent classifier.

Required by project spec — interface:
    class IntentClassification:
        __init__(self, model_path): load config, tokenizer, model
        __call__(self, message):    return predicted intent label

Usage:
    # From Python
    from scripts.inference import IntentClassification
    clf = IntentClassification("configs/inference.yaml")
    label = clf("I lost my card and need a replacement")
    print(label)  # e.g. "card_arrival"

    # From CLI (via inference.sh)
    python scripts/inference.py --config configs/inference.yaml --input "your message here"
"""

import os
import json
import argparse

import yaml
import torch


class IntentClassification:
    """
    Banking intent classifier backed by a fine-tuned Llama 3.1 model (Unsloth).

    Parameters
    ----------
    model_path : str
        Path to the inference YAML config file. The config must contain at least:
            model.checkpoint_path  — path to the saved LoRA checkpoint directory
            model.max_seq_length   — max token length (must match training)
            model.load_in_4bit     — whether to load in 4-bit quantisation
            data.label_map_path    — path to label_map.json
    """

    def __init__(self, model_path: str):
        # ── 1. Load config ──────────────────────────────────────────────────
        with open(model_path, "r") as f:
            cfg = yaml.safe_load(f)

        checkpoint_path = cfg["model"]["checkpoint_path"]
        max_seq_length  = cfg["model"].get("max_seq_length", 512)
        load_in_4bit    = cfg["model"].get("load_in_4bit", True)
        label_map_path  = cfg["data"]["label_map_path"]

        gen_cfg             = cfg.get("generation", {})
        self._max_new_tokens    = gen_cfg.get("max_new_tokens", 15)
        self._do_sample         = gen_cfg.get("do_sample", False)
        self._repetition_penalty = gen_cfg.get("repetition_penalty", 1.1)

        # ── 2. Load label map ───────────────────────────────────────────────
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        # id2label keys may be stored as strings in JSON
        self._id2label  = {int(k): v for k, v in label_map["id2label"].items()}
        self._intent_set = set(self._id2label.values())
        self._intent_list = [self._id2label[i] for i in range(len(self._id2label))]

        # ── 3. Build system prompt (contains list of all valid intents) ──────
        intents_str = "\n".join(f"- {intent}" for intent in self._intent_list)
        self._system_prompt = (
            f"You are a banking intent classifier. "
            f"Given a customer query, classify it into exactly one of the "
            f"following {len(self._intent_list)} intent categories:\n"
            f"{intents_str}\n\n"
            f"Respond with ONLY the intent name, nothing else."
        )

        # ── 4. Load model & tokenizer ───────────────────────────────────────
        print(f"Loading checkpoint from: {checkpoint_path}")
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self._model)
        self._model.eval()
        print("✅ Model loaded and ready for inference.")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, message: str) -> str:
        """Format input using Llama 3.1 ChatML template."""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self._system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{message.lower().strip()}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _match_intent(self, raw: str) -> str:
        """
        Robust matching: exact → starts-with/contains → best token overlap.
        Ensures the output is always one of the known intent labels.
        """
        raw = raw.strip().lower()
        # 1. Exact match
        if raw in self._intent_set:
            return raw
        # 2. Starts-with or contains
        for lbl in self._intent_set:
            if raw.startswith(lbl) or lbl in raw:
                return lbl
        # 3. Token overlap (best effort)
        raw_tokens = set(raw.replace("_", " ").split())
        best, best_score = raw, 0
        for lbl in self._intent_set:
            lbl_tokens = set(lbl.replace("_", " ").split())
            score = len(raw_tokens & lbl_tokens)
            if score > best_score:
                best, best_score = lbl, score
        return best

    # ── Public interface (required by project spec) ──────────────────────────

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single banking query.

        Parameters
        ----------
        message : str   Input customer message.

        Returns
        -------
        str             Predicted intent label (one of the trained categories).
        """
        prompt  = self._build_prompt(message)
        inputs  = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                repetition_penalty=self._repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        raw_pred   = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._match_intent(raw_pred)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run banking intent inference."
    )
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml",
        help="Path to the inference YAML config file."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input banking query message to classify (optional). If not provided, starts interactive mode."
    )
    args = parser.parse_args()

    clf = IntentClassification(model_path=args.config)

    if args.input:
        # Single execution mode
        result = clf(args.input)
        print(f"\n{'='*50}")
        print(f"  Input  : {args.input}")
        print(f"  Intent : {result}")
        print(f"{'='*50}")
    else:
        # Interactive chat mode
        print(f"\n{'='*50}")
        print("  🤖 Banking Intent Classification - Interactive Mode")
        print("  Type 'quit' or 'exit' to stop.")
        print(f"{'='*50}")
        while True:
            try:
                user_input = input("\n🧑 Customer: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input.strip():
                    continue
                
                result = clf(user_input)
                print(f"🤖 Intent  : \033[92m{result}\033[0m")
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
