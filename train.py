#!/usr/bin/env python3
"""
Global Model Trainer (`train.py`)
Trains all Matrix (coding) and Delphi (conversational) AI models across
Generation 1.0, Generation 1.1, Generation 2.0, and Generation 2.1 MoE (12-Layer Sparse-MoE Models):
  - matrix/1.0/flash, matrix/1.0/pro
  - delphi/1.0/flash, delphi/1.0/pro
  - matrix/1.1/flash, matrix/1.1/pro
  - delphi/1.1/flash, delphi/1.1/pro
  - matrix/2.0/flash, matrix/2.0/pro
  - delphi/2.0/flash, delphi/2.0/pro
  - matrix/2.1/flash, matrix/2.1/pro, matrix/2.1/moe
  - delphi/2.1/flash, delphi/2.1/pro, delphi/2.1/moe

Can also run the anti-memorization fine-tuner (`finetune.py`) automatically
with `--with-finetune` so every trained model is verified not to copy verbatim
from its training files.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent

ALL_MODELS: List[Dict[str, Any]] = [
    {
        "key": "matrix-1.0-flash",
        "gen": "1.0",
        "family": "matrix",
        "tier": "flash",
        "name": "Matrix 1.0 Flash",
        "script": ROOT_DIR / "matrix" / "1.0" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "write a python function to reverse a list",
    },
    {
        "key": "matrix-1.0-pro",
        "gen": "1.0",
        "family": "matrix",
        "tier": "pro",
        "name": "Matrix 1.0 Pro",
        "script": ROOT_DIR / "matrix" / "1.0" / "pro" / "model.py",
        "default_epochs": 45,
        "sample_prompt": "implement an LRU cache class in python",
    },
    {
        "key": "delphi-1.0-flash",
        "gen": "1.0",
        "family": "delphi",
        "tier": "flash",
        "name": "Delphi 1.0 Flash",
        "script": ROOT_DIR / "delphi" / "1.0" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "hello how can I stay creative today",
    },
    {
        "key": "delphi-1.0-pro",
        "gen": "1.0",
        "family": "delphi",
        "tier": "pro",
        "name": "Delphi 1.0 Pro",
        "script": ROOT_DIR / "delphi" / "1.0" / "pro" / "model.py",
        "default_epochs": 45,
        "sample_prompt": "explain how complex systems emerge from simple rules",
    },
    {
        "key": "matrix-1.1-flash",
        "gen": "1.1",
        "family": "matrix",
        "tier": "flash",
        "name": "Matrix 1.1 Flash",
        "script": ROOT_DIR / "matrix" / "1.1" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "write a python function to normalize rows of a matrix in numpy",
    },
    {
        "key": "matrix-1.1-pro",
        "gen": "1.1",
        "family": "matrix",
        "tier": "pro",
        "name": "Matrix 1.1 Pro",
        "script": ROOT_DIR / "matrix" / "1.1" / "pro" / "model.py",
        "default_epochs": 45,
        "sample_prompt": "build a sparse mixture of experts layer in pytorch",
    },
    {
        "key": "delphi-1.1-flash",
        "gen": "1.1",
        "family": "delphi",
        "tier": "flash",
        "name": "Delphi 1.1 Flash",
        "script": ROOT_DIR / "delphi" / "1.1" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "how do I turn a new idea into a real project",
    },
    {
        "key": "delphi-1.1-pro",
        "gen": "1.1",
        "family": "delphi",
        "tier": "pro",
        "name": "Delphi 1.1 Pro",
        "script": ROOT_DIR / "delphi" / "1.1" / "pro" / "model.py",
        "default_epochs": 45,
        "sample_prompt": "how does information theory connect thermodynamics and language",
    },
    {
        "key": "matrix-2.0-flash",
        "gen": "2.0",
        "family": "matrix",
        "tier": "flash",
        "name": "Matrix 2.0 Flash",
        "script": ROOT_DIR / "matrix" / "2.0" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "write a python decorator that retries a failing function with exponential backoff",
    },
    {
        "key": "matrix-2.0-pro",
        "gen": "2.0",
        "family": "matrix",
        "tier": "pro",
        "name": "Matrix 2.0 Pro",
        "script": ROOT_DIR / "matrix" / "2.0" / "pro" / "model.py",
        "default_epochs": 40,
        "sample_prompt": "implement grouped query attention GQA with rotary position embeddings in pytorch",
    },
    {
        "key": "delphi-2.0-flash",
        "gen": "2.0",
        "family": "delphi",
        "tier": "flash",
        "name": "Delphi 2.0 Flash",
        "script": ROOT_DIR / "delphi" / "2.0" / "flash" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "how can I build a habit of deep focus in a distracted world",
    },
    {
        "key": "delphi-2.0-pro",
        "gen": "2.0",
        "family": "delphi",
        "tier": "pro",
        "name": "Delphi 2.0 Pro",
        "script": ROOT_DIR / "delphi" / "2.0" / "pro" / "model.py",
        "default_epochs": 40,
        "sample_prompt": "how do symmetry principles shape the laws of physics and mathematics",
    },
    {
        "key": "matrix-2.1-flash",
        "gen": "2.1",
        "family": "matrix",
        "tier": "flash",
        "name": "Matrix 2.1 Flash MoE",
        "script": ROOT_DIR / "matrix" / "2.1" / "flash" / "model.py",
        "default_epochs": 30,
        "sample_prompt": "write a fast top-2 sparse MoE router function in pytorch",
    },
    {
        "key": "matrix-2.1-pro",
        "gen": "2.1",
        "family": "matrix",
        "tier": "pro",
        "name": "Matrix 2.1 Pro MoE",
        "script": ROOT_DIR / "matrix" / "2.1" / "pro" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
    },
    {
        "key": "matrix-2.1-moe",
        "gen": "2.1",
        "family": "matrix",
        "tier": "moe",
        "name": "Matrix 2.1 MoE",
        "script": ROOT_DIR / "matrix" / "2.1" / "moe" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
    },
    {
        "key": "delphi-2.1-flash",
        "gen": "2.1",
        "family": "delphi",
        "tier": "flash",
        "name": "Delphi 2.1 Flash MoE",
        "script": ROOT_DIR / "delphi" / "2.1" / "flash" / "model.py",
        "default_epochs": 30,
        "sample_prompt": "how can switching between different mental modes help me solve creative problems",
    },
    {
        "key": "delphi-2.1-pro",
        "gen": "2.1",
        "family": "delphi",
        "tier": "pro",
        "name": "Delphi 2.1 Pro MoE",
        "script": ROOT_DIR / "delphi" / "2.1" / "pro" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
    },
    {
        "key": "delphi-2.1-moe",
        "gen": "2.1",
        "family": "delphi",
        "tier": "moe",
        "name": "Delphi 2.1 MoE",
        "script": ROOT_DIR / "delphi" / "2.1" / "moe" / "model.py",
        "default_epochs": 35,
        "sample_prompt": "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
    },
]


def load_script_module(module_name: str, script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def train_entry(entry: Dict[str, Any], epochs_override: int | None = None) -> None:
    script_path: Path = entry["script"]
    if not script_path.exists():
        print(f"[SKIP] {entry['name']} — script not found at {script_path}")
        return

    epochs = epochs_override if epochs_override is not None else entry["default_epochs"]
    print(f"\n==================================================================")
    print(f" Training {entry['name']} ({entry['key']}) — {epochs} epochs")
    print(f"==================================================================")

    start_t = time.perf_counter()
    mod = load_script_module(f"train_{entry['key'].replace('-', '_').replace('.', '_')}", script_path)
    model, tokenizer = mod.train_model(epochs=epochs)
    elapsed = time.perf_counter() - start_t

    param_count = sum(p.numel() for p in model.parameters())
    num_layers = len(model.blocks) if hasattr(model, "blocks") else 0

    # Run a quick sample generation check
    prompt = entry["sample_prompt"]
    if hasattr(mod, "answer_prompt"):
        sample_out = mod.answer_prompt(prompt, model, tokenizer)
    else:
        sample_out = mod.reply_to_user(prompt, model, tokenizer)

    print(f"[{entry['name']}] Architecture: {num_layers} hidden layers | {param_count:,} parameters")
    print(f"[{entry['name']}] Trained in {elapsed:.2f}s")
    print(f"[{entry['name']}] Sample Prompt: {prompt!r}")
    print(f"[{entry['name']}] Sample Response:\n{sample_out}\n")


def main() -> None:
    valid_keys = ["all"] + [m["key"] for m in ALL_MODELS]
    parser = argparse.ArgumentParser(
        description="Train all Matrix (coding) and Delphi (conversational) AI models across Gen 1.0, 1.1, 2.0, and 2.1 MoE"
    )
    parser.add_argument(
        "--gen",
        choices=["all", "1.0", "1.1", "2.0", "2.1"],
        default="all",
        help="Which generation of models to train (default: all)",
    )
    parser.add_argument(
        "--family",
        choices=["all", "matrix", "delphi"],
        default="all",
        help="Filter by model family: matrix (coding) or delphi (talking)",
    )
    parser.add_argument(
        "--tier",
        choices=["all", "flash", "pro", "moe"],
        default="all",
        help="Filter by model tier: flash (fast), pro (heavy), or moe (flagship sparse MoE)",
    )
    parser.add_argument(
        "--target",
        choices=valid_keys,
        default="all",
        help="Train a specific model key or all (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs for selected models",
    )
    parser.add_argument(
        "--with-finetune",
        action="store_true",
        help="Run global anti-memorization fine-tuning (finetune.py) immediately after training",
    )
    args = parser.parse_args()

    selected: List[Dict[str, Any]] = []
    for item in ALL_MODELS:
        if args.target != "all" and item["key"] != args.target:
            continue
        if args.gen != "all" and item["gen"] != args.gen:
            continue
        if args.family != "all" and item["family"] != args.family:
            continue
        if args.tier != "all" and item["tier"] != args.tier:
            continue
        selected.append(item)

    if not selected:
        print("No models matched the given filters.")
        return

    print(f"Preparing to train {len(selected)} model(s): {', '.join(m['name'] for m in selected)}")
    overall_start = time.perf_counter()

    for entry in selected:
        train_entry(entry, epochs_override=args.epochs)

    if args.with_finetune:
        finetune_path = ROOT_DIR / "finetune.py"
        if finetune_path.exists():
            print("\n>>> Running Global Anti-Memorization Fine-Tuning stage (`finetune.py`)...")
            ft_mod = load_script_module("global_finetune", finetune_path)
            for entry in selected:
                if entry["key"] in ft_mod.MODEL_SPECS:
                    ft_mod.finetune_single_model(entry["key"])

    total_elapsed = time.perf_counter() - overall_start
    print(f"\nAll {len(selected)} model(s) finished training in {total_elapsed:.2f}s!")


if __name__ == "__main__":
    main()
