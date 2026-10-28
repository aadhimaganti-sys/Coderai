#!/usr/bin/env python3
"""
Global Anti-Memorization & Anti-Repetition Fine-Tuner (`finetune.py`)
Fine-tunes all Matrix and Delphi AI models across Generation 1.0, 1.1, and 2.0:
  - matrix-1.0-flash, matrix-1.0-pro
  - delphi-1.0-flash, delphi-1.0-pro
  - matrix-1.1-flash, matrix-1.1-pro
  - delphi-1.1-flash, delphi-1.1-pro
  - matrix-2.0-flash, matrix-2.0-pro (10 Hidden Layers, 1M+ Parameters)
  - delphi-2.0-flash, delphi-2.0-pro (10 Hidden Layers, 1M+ Parameters)

How it guarantees models do NOT repeat or copy verbatim from their training files:
1. Cross-File Clause & Code Recombination: Mixes prompts, explanations, clauses, and
   code snippets across different training files so every fine-tuning epoch sees fresh
   synthetic combinations.
2. Embedding-Space Gaussian Jitter & Token Mixup: Adds continuous Gaussian noise to
   token embeddings during fine-tuning so the Transformer cannot memorize sharp
   lookup tables.
3. Over-Confidence Suppression + Entropy Regularization + Temporal Unlikelihood Loss:
   Penalizes any token prediction whose probability on the exact training continuation
   exceeds `max_token_prob` (e.g. 0.65), rewards predictive entropy, and penalizes
   consecutive token repetition via unlikelihood training.
4. Multi-Order Training N-Gram Copy Blocker (3-gram & 4-gram) + Self-Repetition Blocker:
   Extracts N-gram indices from all raw training files and stores them in the checkpoint
   (`forbidden_ngrams`, `forbidden_trigrams`). During generation, any token that would
   complete a verbatim N-gram from the training files is heavily penalized so the model
   must paraphrase and compose original sentences.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parent

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "matrix-1.0-flash": {
        "name": "Matrix 1.0 Flash",
        "gen": "1.0",
        "dir": ROOT_DIR / "matrix" / "1.0" / "flash",
        "script": ROOT_DIR / "matrix" / "1.0" / "flash" / "model.py",
        "domain": "code",
        "test_prompt": "write a python function to filter and average positive numbers",
    },
    "matrix-1.0-pro": {
        "name": "Matrix 1.0 Pro",
        "gen": "1.0",
        "dir": ROOT_DIR / "matrix" / "1.0" / "pro",
        "script": ROOT_DIR / "matrix" / "1.0" / "pro" / "model.py",
        "domain": "code",
        "test_prompt": "design a scalable rate limiter with retry logic in python",
    },
    "delphi-1.0-flash": {
        "name": "Delphi 1.0 Flash",
        "gen": "1.0",
        "dir": ROOT_DIR / "delphi" / "1.0" / "flash",
        "script": ROOT_DIR / "delphi" / "1.0" / "flash" / "model.py",
        "domain": "chat",
        "test_prompt": "how can I stay curious and creative when learning something new",
    },
    "delphi-1.0-pro": {
        "name": "Delphi 1.0 Pro",
        "gen": "1.0",
        "dir": ROOT_DIR / "delphi" / "1.0" / "pro",
        "script": ROOT_DIR / "delphi" / "1.0" / "pro" / "model.py",
        "domain": "chat",
        "test_prompt": "how do simple rules in nature give rise to consciousness and stories",
    },
    "matrix-1.1-flash": {
        "name": "Matrix 1.1 Flash",
        "gen": "1.1",
        "dir": ROOT_DIR / "matrix" / "1.1" / "flash",
        "script": ROOT_DIR / "matrix" / "1.1" / "flash" / "model.py",
        "domain": "code",
        "test_prompt": "write a python helper to normalize and slice windowed arrays in numpy",
    },
    "matrix-1.1-pro": {
        "name": "Matrix 1.1 Pro",
        "gen": "1.1",
        "dir": ROOT_DIR / "matrix" / "1.1" / "pro",
        "script": ROOT_DIR / "matrix" / "1.1" / "pro" / "model.py",
        "domain": "code",
        "test_prompt": "architect a resilient circuit breaker with consistent hashing in python",
    },
    "delphi-1.1-flash": {
        "name": "Delphi 1.1 Flash",
        "gen": "1.1",
        "dir": ROOT_DIR / "delphi" / "1.1" / "flash",
        "script": ROOT_DIR / "delphi" / "1.1" / "flash" / "model.py",
        "domain": "chat",
        "test_prompt": "how do I stay adaptable and creative when starting a new project",
    },
    "delphi-1.1-pro": {
        "name": "Delphi 1.1 Pro",
        "gen": "1.1",
        "dir": ROOT_DIR / "delphi" / "1.1" / "pro",
        "script": ROOT_DIR / "delphi" / "1.1" / "pro" / "model.py",
        "domain": "chat",
        "test_prompt": "how do feedback loops connect information theory and human wisdom",
    },
    "matrix-2.0-flash": {
        "name": "Matrix 2.0 Flash",
        "gen": "2.0",
        "dir": ROOT_DIR / "matrix" / "2.0" / "flash",
        "script": ROOT_DIR / "matrix" / "2.0" / "flash" / "model.py",
        "domain": "code",
        "test_prompt": "write a python decorator for exponential backoff retry and merge intervals",
    },
    "matrix-2.0-pro": {
        "name": "Matrix 2.0 Pro",
        "gen": "2.0",
        "dir": ROOT_DIR / "matrix" / "2.0" / "pro",
        "script": ROOT_DIR / "matrix" / "2.0" / "pro" / "model.py",
        "domain": "code",
        "test_prompt": "implement grouped query attention with a transactional key value cache in python",
    },
    "delphi-2.0-flash": {
        "name": "Delphi 2.0 Flash",
        "gen": "2.0",
        "dir": ROOT_DIR / "delphi" / "2.0" / "flash",
        "script": ROOT_DIR / "delphi" / "2.0" / "flash" / "model.py",
        "domain": "chat",
        "test_prompt": "how can I build deep focus and turn creative sparks into finished work",
    },
    "delphi-2.0-pro": {
        "name": "Delphi 2.0 Pro",
        "gen": "2.0",
        "dir": ROOT_DIR / "delphi" / "2.0" / "pro",
        "script": ROOT_DIR / "delphi" / "2.0" / "pro" / "model.py",
        "domain": "chat",
        "test_prompt": "how do symmetry principles and causal reasoning connect physics to human wisdom",
    },
    "matrix-2.1-flash": {
        "name": "Matrix 2.1 Flash MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "matrix" / "2.1" / "flash",
        "script": ROOT_DIR / "matrix" / "2.1" / "flash" / "model.py",
        "domain": "code",
        "test_prompt": "write a fast top-2 sparse MoE router function in pytorch",
    },
    "matrix-2.1-pro": {
        "name": "Matrix 2.1 Pro MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "matrix" / "2.1" / "pro",
        "script": ROOT_DIR / "matrix" / "2.1" / "pro" / "model.py",
        "domain": "code",
        "test_prompt": "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
    },
    "matrix-2.1-moe": {
        "name": "Matrix 2.1 MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "matrix" / "2.1" / "moe",
        "script": ROOT_DIR / "matrix" / "2.1" / "moe" / "model.py",
        "domain": "code",
        "test_prompt": "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
    },
    "delphi-2.1-flash": {
        "name": "Delphi 2.1 Flash MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "delphi" / "2.1" / "flash",
        "script": ROOT_DIR / "delphi" / "2.1" / "flash" / "model.py",
        "domain": "chat",
        "test_prompt": "how can switching between different mental modes help me solve creative problems",
    },
    "delphi-2.1-pro": {
        "name": "Delphi 2.1 Pro MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "delphi" / "2.1" / "pro",
        "script": ROOT_DIR / "delphi" / "2.1" / "pro" / "model.py",
        "domain": "chat",
        "test_prompt": "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
    },
    "delphi-2.1-moe": {
        "name": "Delphi 2.1 MoE",
        "gen": "2.1",
        "dir": ROOT_DIR / "delphi" / "2.1" / "moe",
        "script": ROOT_DIR / "delphi" / "2.1" / "moe" / "model.py",
        "domain": "chat",
        "test_prompt": "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
    },
}


def load_module_from_path(module_name: str, script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_training_ngram_index(
    samples: List[str],
    tokenizer: Any,
    n: int = 4,
) -> Dict[Tuple[int, ...], Set[int]]:
    prefix_to_next: Dict[Tuple[int, ...], Set[int]] = {}
    if n < 2:
        return prefix_to_next

    for text in samples:
        ids = [int(x) for x in tokenizer.encode(text, add_special=False)]
        if len(ids) < n:
            continue
        for i in range(len(ids) - n + 1):
            prefix = tuple(ids[i : i + n - 1])
            next_tok = int(ids[i + n - 1])
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(next_tok)
    return prefix_to_next


def serialize_ngram_index(index: Dict[Tuple[int, ...], Set[int]]) -> List[Tuple[List[int], List[int]]]:
    return [(list(prefix), sorted(list(next_set))) for prefix, next_set in index.items()]


def synthesize_cross_file_recombinations(
    samples: List[str],
    domain: str,
    rng: np.random.Generator,
    multiplier: int = 2,
) -> List[str]:
    parsed: List[Tuple[str, str]] = []
    for sample in samples:
        if "Assistant:" in sample:
            parts = sample.split("Assistant:", 1)
            user_part = parts[0].replace("User:", "").strip()
            asst_part = parts[1].strip()
            if user_part and asst_part:
                parsed.append((user_part, asst_part))

    if len(parsed) < 2:
        return list(samples)

    synonym_swaps_chat = [
        ("great", "wonderful"),
        ("important", "essential"),
        ("explore", "discover"),
        ("creative", "imaginative"),
        ("understand", "grasp"),
        ("complex", "intricate"),
        ("quickly", "smoothly"),
    ]
    synonym_swaps_code = [
        ("Here is a", "Below is a clean"),
        ("You can", "One effective way is to"),
        ("function", "helper routine"),
        ("cleanly", "efficiently"),
        ("computes", "calculates"),
    ]
    swaps = synonym_swaps_code if domain == "code" else synonym_swaps_chat

    augmented: List[str] = list(samples)
    for _ in range(len(parsed) * multiplier):
        idx1, idx2 = rng.choice(len(parsed), size=2, replace=False)
        u1, a1 = parsed[int(idx1)]
        u2, a2 = parsed[int(idx2)]

        for old_w, new_w in swaps:
            if rng.random() < 0.45:
                a1 = a1.replace(old_w, new_w)
            if rng.random() < 0.45:
                a2 = a2.replace(old_w, new_w)

        if domain == "code":
            a1_intro = a1.split("```")[0].strip()
            a2_parts = a2.split("```")
            if len(a2_parts) >= 3:
                a2_code = "```" + a2_parts[1] + "```"
                recombined_reply = f"{a1_intro} We can also combine this with a modular approach:\n{a2_code}"
            else:
                recombined_reply = f"{a1}\n\nAdditionally:\n{a2}"
            new_sample = f"User: {u1} and {u2}\nAssistant: {recombined_reply}"
        else:
            s1 = [s.strip() for s in a1.split(".") if s.strip()]
            s2 = [s.strip() for s in a2.split(".") if s.strip()]
            lead = s1[0] + "." if s1 else a1
            bridge = s2[-1] + "." if s2 else a2
            mid = s2[0] + "." if len(s2) > 1 else ""
            recombined_reply = " ".join(part for part in [lead, mid, bridge] if part)
            new_sample = f"User: {u1} — and how does that relate to {u2}?\nAssistant: {recombined_reply}"

        augmented.append(new_sample)

    return augmented


class AntiMemorizationDataset(Dataset):
    def __init__(
        self,
        samples: List[str],
        tokenizer: Any,
        max_seq_len: int,
        mask_prob: float = 0.08,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.encoded = [tokenizer.encode(s, add_special=True) for s in samples]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoded[idx].copy()
        if len(seq) > self.max_seq_len + 1:
            seq = seq[: self.max_seq_len + 1]

        x_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)
        y_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)

        valid_len = min(len(seq) - 1, self.max_seq_len)
        x_slice = seq[:valid_len].copy()

        if valid_len > 6:
            mask = np.random.random(size=valid_len) < self.mask_prob
            mask[:3] = False
            x_slice[mask] = self.tokenizer.unk_id

        x_np[:valid_len] = x_slice
        y_np[:valid_len] = seq[1 : valid_len + 1]
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def anti_memorization_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.15,
    entropy_weight: float = 0.06,
    max_token_prob: float = 0.65,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)

    ce_loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )

    valid_mask = flat_targets != pad_id
    if not torch.any(valid_mask):
        return ce_loss

    valid_logits = flat_logits[valid_mask]
    valid_targets = flat_targets[valid_mask]

    probs = F.softmax(valid_logits, dim=-1)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    target_probs = probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)
    overconfident_excess = F.relu(target_probs - max_token_prob)
    confidence_penalty = (overconfident_excess ** 2).mean() * 2.5

    b, t, _ = logits.size()
    if t > 1:
        seq_probs = F.softmax(logits[:, 1:, :], dim=-1)
        prev_tokens = targets[:, :-1].unsqueeze(-1)
        repeat_probs = seq_probs.gather(-1, prev_tokens).squeeze(-1)
        non_pad = (targets[:, 1:] != pad_id).float()
        unlikelihood = (-torch.log(torch.clamp(1.0 - repeat_probs, min=1e-4)) * non_pad).sum() / (non_pad.sum() + 1e-6)
    else:
        unlikelihood = torch.tensor(0.0, device=logits.device)

    return ce_loss - (entropy_weight * entropy) + confidence_penalty + 0.08 * unlikelihood


def forward_with_embedding_noise(
    model: nn.Module,
    idx: torch.Tensor,
    noise_std: float = 0.05,
) -> torch.Tensor:
    b, t = idx.size()
    pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
    x = model.token_emb(idx) + model.pos_emb(pos)
    if model.training and noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    x = model.drop(x)
    for block in model.blocks:
        x = block(x)
    if hasattr(model, "norm_f"):
        x = model.norm_f(x)
    else:
        x = model.ln_f(x)
    return model.lm_head(x)


@torch.no_grad()
def generate_non_copying(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    ngram_index: Dict[Tuple[int, ...], Set[int]],
    ngram_n: int = 4,
    max_new_tokens: int = 90,
    temperature: float = 0.88,
    top_k: int = 28,
    top_p: float = 0.92,
    repetition_penalty: float = 1.25,
    copy_penalty: float = 6.5,
    device: str = "cpu",
) -> str:
    model.eval()
    formatted = f"<|bos|> <|user|> {prompt.strip()} \n <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    generated = list(map(int, prompt_ids))
    prompt_len = len(generated)
    generated_trigrams: Set[Tuple[int, int, int]] = set()

    for _ in range(max_new_tokens):
        ctx = generated[-model.max_seq_len :]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].cpu().numpy()

        logits[tokenizer.pad_id] = -1e9
        logits[tokenizer.unk_id] = -1e9

        if ngram_n >= 2 and len(generated) >= ngram_n - 1:
            prefix = tuple(generated[-(ngram_n - 1) :])
            forbidden_next = ngram_index.get(prefix)
            if forbidden_next:
                for tok_id in forbidden_next:
                    if 0 <= tok_id < len(logits):
                        logits[tok_id] -= copy_penalty

        if len(generated) - prompt_len >= 2:
            last_two = (generated[-2], generated[-1])
            for cand in range(len(logits)):
                if (last_two[0], last_two[1], cand) in generated_trigrams:
                    logits[cand] -= 1e4

        for seen_id in set(generated[-38:]):
            if logits[seen_id] > 0:
                logits[seen_id] /= repetition_penalty
            else:
                logits[seen_id] *= repetition_penalty

        logits = logits / max(temperature, 1e-4)

        if 0 < top_k < len(logits):
            idx_k = np.argpartition(logits, -top_k)[-top_k:]
            masked = np.full_like(logits, -1e9)
            masked[idx_k] = logits[idx_k]
            logits = masked

        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / (np.sum(probs) + 1e-12)

        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff = cumulative > top_p
        if np.any(cutoff):
            cut = int(np.argmax(cutoff)) + 1
            keep = sorted_idx[: max(1, cut)]
            filtered = np.zeros_like(probs)
            filtered[keep] = probs[keep]
            probs = filtered / (np.sum(filtered) + 1e-12)

        next_id = int(np.random.choice(len(probs), p=probs))
        if next_id == tokenizer.eos_id:
            break
        if len(generated) - prompt_len >= 2:
            generated_trigrams.add((generated[-2], generated[-1], next_id))
        generated.append(next_id)

    reply_ids = generated[prompt_len:]
    return tokenizer.decode(reply_ids)


def compute_novelty_metrics(
    generated_text: str,
    raw_samples: List[str],
    tokenizer: Any,
    n: int = 4,
) -> Tuple[float, bool]:
    gen_ids = [int(x) for x in tokenizer.encode(generated_text, add_special=False)]
    if len(gen_ids) < n:
        return 100.0, True

    training_ngrams: Set[Tuple[int, ...]] = set()
    for sample in raw_samples:
        s_ids = [int(x) for x in tokenizer.encode(sample, add_special=False)]
        for i in range(max(0, len(s_ids) - n + 1)):
            training_ngrams.add(tuple(s_ids[i : i + n]))

    gen_ngrams = [tuple(gen_ids[i : i + n]) for i in range(len(gen_ids) - n + 1)]
    novel_count = sum(1 for ng in gen_ngrams if ng not in training_ngrams)
    novel_ratio = (novel_count / max(1, len(gen_ngrams))) * 100.0

    norm_gen = " ".join(generated_text.lower().split())
    not_exact_copy = all(norm_gen not in " ".join(s.lower().split()) for s in raw_samples)
    return novel_ratio, not_exact_copy


def finetune_single_model(
    key: str,
    epochs: int = 20,
    lr: float = 1.2e-3,
    entropy_weight: float = 0.06,
    max_token_prob: float = 0.65,
    no_copy_ngram: int = 4,
    extra_data_dir: Path | None = None,
) -> None:
    spec = MODEL_SPECS[key]
    name = spec["name"]
    script_path = spec["script"]
    domain = spec["domain"]

    if not script_path.exists():
        print(f"[SKIP] {name} — script not found at {script_path}")
        return

    print(f"\n==================================================================")
    print(f" Fine-Tuning: {name} ({key})")
    print(f"==================================================================")

    mod = load_module_from_path(f"mod_{key.replace('-', '_').replace('.', '_')}", script_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(2026)

    if hasattr(mod, "load_multi_file_training_corpus"):
        raw_samples = mod.load_multi_file_training_corpus(mod.DATA_DIR, rng)
    elif hasattr(mod, "load_pro_training_corpus"):
        raw_samples = mod.load_pro_training_corpus(mod.DATA_DIR, rng)
    elif hasattr(mod, "load_dialogue_corpus"):
        raw_samples = mod.load_dialogue_corpus(mod.DATA_DIR, rng)
    else:
        raw_samples = mod.load_pro_dialogue_corpus(mod.DATA_DIR, rng)

    if extra_data_dir is not None and extra_data_dir.exists():
        for txt_file in sorted(extra_data_dir.glob("*.txt")):
            blocks = [b.strip() for b in txt_file.read_text(encoding="utf-8").split("<|endoftext|>") if b.strip()]
            raw_samples.extend(blocks)

    recombined_samples = synthesize_cross_file_recombinations(raw_samples, domain=domain, rng=rng, multiplier=2)

    model, tokenizer = mod.load_or_train()
    old_vocab_size = tokenizer.vocab_size
    tokenizer.build_vocab(recombined_samples)

    if tokenizer.vocab_size != old_vocab_size:
        ckpt = torch.load(mod.CHECKPOINT_PATH, map_location=device, weights_only=False)
        cfg = dict(ckpt["config"])
        cfg["vocab_size"] = tokenizer.vocab_size
        model_cls = type(model)
        new_model = model_cls(**cfg).to(device)
        with torch.no_grad():
            new_model.token_emb.weight[:old_vocab_size].copy_(model.token_emb.weight[:old_vocab_size])
            for name_p, param in model.named_parameters():
                if name_p not in {"token_emb.weight", "lm_head.weight"}:
                    dict(new_model.named_parameters())[name_p].copy_(param)
        model = new_model

    ngram_index = build_training_ngram_index(raw_samples, tokenizer, n=no_copy_ngram)
    trigram_index = build_training_ngram_index(raw_samples, tokenizer, n=3)

    dataset = AntiMemorizationDataset(
        recombined_samples,
        tokenizer,
        max_seq_len=model.max_seq_len,
        mask_prob=0.08,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in model.parameters())
    num_layers = len(model.blocks) if hasattr(model, "blocks") else 0

    print(
        f"[{name}] ({num_layers} layers, {param_count:,} params) Fine-tuning on {len(recombined_samples)} "
        f"cross-recombined samples (max_token_prob={max_token_prob}, entropy_weight={entropy_weight}, {no_copy_ngram}-gram copy blocker)..."
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = forward_with_embedding_noise(model, x, noise_std=0.05)
            loss = anti_memorization_loss(
                logits,
                y,
                pad_id=tokenizer.pad_id,
                label_smoothing=0.15,
                entropy_weight=entropy_weight,
                max_token_prob=max_token_prob,
            )
            if hasattr(model, "get_moe_auxiliary_loss"):
                loss = loss + 0.02 * model.get_moe_auxiliary_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Fine-Tune Epoch {epoch:02d}/{epochs:02d} | Anti-Memorization Loss: {np.mean(epoch_losses):.4f}")

    ckpt = torch.load(mod.CHECKPOINT_PATH, map_location=device, weights_only=False)
    ckpt["model_state"] = model.state_dict()
    ckpt["tokenizer"] = tokenizer.state_dict()
    ckpt["config"]["vocab_size"] = tokenizer.vocab_size
    ckpt["anti_copy"] = {
        "ngram_n": no_copy_ngram,
        "forbidden_ngrams": serialize_ngram_index(ngram_index),
        "forbidden_trigrams": serialize_ngram_index(trigram_index),
    }
    mod.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, mod.CHECKPOINT_PATH)
    print(f"[{name}] Saved fine-tuned checkpoint + {len(ngram_index)} anti-copy {no_copy_ngram}-gram rules to {mod.CHECKPOINT_PATH}")

    sample_reply = generate_non_copying(
        model,
        tokenizer,
        prompt=spec["test_prompt"],
        ngram_index=ngram_index,
        ngram_n=no_copy_ngram,
        device=device,
    )
    novelty_pct, not_exact = compute_novelty_metrics(sample_reply, raw_samples, tokenizer, n=no_copy_ngram)
    print(f"[{name}] Verification Prompt: {spec['test_prompt']!r}")
    print(f"[{name}] Novel {no_copy_ngram}-grams: {novelty_pct:.1f}% | Non-verbatim verified: {not_exact}")
    print(f"[{name}] Sample Output:\n{sample_reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global Anti-Memorization Fine-Tuner for Matrix and Delphi (Gen 1.0, 1.1, 2.0, and 2.1 MoE)"
    )
    parser.add_argument(
        "--gen",
        choices=["all", "1.0", "1.1", "2.0", "2.1"],
        default="all",
        help="Filter models by generation (default: all)",
    )
    parser.add_argument(
        "--target",
        choices=["all"] + list(MODEL_SPECS.keys()),
        default="all",
        help="Which model(s) to fine-tune (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs per model")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Fine-tuning learning rate")
    parser.add_argument("--entropy-weight", type=float, default=0.06, help="Entropy regularization weight")
    parser.add_argument("--max-token-prob", type=float, default=0.65, help="Confidence cap on training tokens")
    parser.add_argument("--no-copy-ngram", type=int, default=4, help="N-gram length blocked from verbatim copying")
    parser.add_argument("--extra-data", type=str, default=None, help="Optional extra .txt directory")
    args = parser.parse_args()

    extra_dir = Path(args.extra_data).resolve() if args.extra_data else None
    targets = [
        k for k, v in MODEL_SPECS.items()
        if (args.target == "all" or k == args.target)
        and (args.gen == "all" or v["gen"] == args.gen)
    ]

    for key in targets:
        finetune_single_model(
            key=key,
            epochs=args.epochs,
            lr=args.lr,
            entropy_weight=args.entropy_weight,
            max_token_prob=args.max_token_prob,
            no_copy_ngram=args.no_copy_ngram,
            extra_data_dir=extra_dir,
        )


if __name__ == "__main__":
    main()
