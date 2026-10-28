#!/usr/bin/env python3
"""
Matrix 2.0 Flash — Full 10-Layer, 1M+ Parameter Fast Coding AI Model (PyTorch + NumPy).
Gen 2.0 Upgrades over 1.0 & 1.1:
- 10 Hidden Transformer Layers (`n_layers=10`) with >2.2 Million Parameters (`d_model=160, hidden_dim=448`)
- Rotary Position Embeddings (RoPE) + Query-Key RMSNorm + Learnable LayerScale Residual Gates
- SwiGLU Gated Feed-Forward Networks across all 10 hidden layers
- Cross-generational corpus synthesis (combines 2.0/flash/data + 1.1/flash/data + 1.0/flash/data)
- Anti-Memorization & Anti-Repetition Training Pipeline:
  1. Dynamic cross-file recombination & synonym paraphrasing every epoch
  2. Continuous Gaussian embedding noise + token dropout during training
  3. Over-confidence suppression (`max_token_prob=0.65`) + predictive entropy regularization + unlikelihood loss
  4. Multi-scale 3-gram & 4-gram training-file copy blocker + self-repetition blocker at inference
"""

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GEN1_1_DATA_DIR = BASE_DIR.parent.parent / "1.1" / "flash" / "data"
GEN1_0_DATA_DIR = BASE_DIR.parent.parent / "1.0" / "flash" / "data"
HUMAN_CHAT_DIR = BASE_DIR.parent.parent.parent / "delphi" / "2.0" / "flash" / "data"
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = TRAINED_DIR / "matrix_flash_2.0.pt"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "matrix_flash_2.0.pt"


class HybridCodeTokenizer:
    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
    TOKEN_REGEX = re.compile(
        r"(<\|pad\|>|<\|unk\|>|<\|bos\|>|<\|eos\|>|<\|user\|>|<\|assistant\|>|\n+|    |\t|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|```|[!=<>]=|\*\*|//|->|[-+*/%=<>()\[\]{}.,:;\"'`~|&^@]+|\S)"
    )

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<|pad|>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<|unk|>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<|eos|>"]

    def _split(self, text: str) -> List[str]:
        normalized = text.replace("User:", " <|user|> ").replace("Assistant:", " <|assistant|> ")
        return [m.group(0) for m in self.TOKEN_REGEX.finditer(normalized)]

    def build_vocab(self, texts: List[str]) -> None:
        for text in texts:
            for tok in self._split(text):
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok

    def encode(self, text: str, add_special: bool = True) -> np.ndarray:
        ids = [self.token_to_id.get(t, self.unk_id) for t in self._split(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: List[int]) -> str:
        pieces: List[str] = []
        no_space_before = {".", ",", ":", ";", ")", "]", "}", "(", "[", "\n", "\n\n"}
        no_space_after = {"(", "[", "{", "\n", "\n\n", "    ", "\t", "```", "@"}
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "")
            if tok in self.SPECIAL_TOKENS:
                continue
            if not pieces:
                pieces.append(tok)
                continue
            prev = pieces[-1]
            if tok.startswith("\n") or prev.endswith("\n") or prev in no_space_after or tok in no_space_before:
                pieces.append(tok)
            else:
                pieces.append(" " + tok)
        return "".join(pieces).strip()

    def state_dict(self) -> Dict[str, Dict]:
        return {"token_to_id": self.token_to_id}

    def load_state_dict(self, state: Dict[str, Dict]) -> None:
        self.token_to_id = dict(state["token_to_id"])
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}


def paraphrase_explanation(text: str, rng: np.random.Generator) -> str:
    swaps = [
        ("You can", "One clean approach is to"),
        ("Use ", "Leverage "),
        ("cleanly", "smoothly"),
        ("function", "routine"),
        ("fast", "low-latency"),
        ("computes", "calculates"),
        ("returns", "produces"),
    ]
    out = text
    for old_w, new_w in swaps:
        if rng.random() < 0.45:
            out = out.replace(old_w, new_w)
    return out


def load_multi_file_training_corpus(data_dir: Path, rng: np.random.Generator) -> List[str]:
    samples: List[str] = []
    search_dirs = [data_dir]
    for prev_dir in (GEN1_1_DATA_DIR, GEN1_0_DATA_DIR):
        if prev_dir.exists() and prev_dir not in search_dirs:
            search_dirs.append(prev_dir)

    all_concepts: List[Dict[str, str]] = []
    verbs = ["write", "create", "build", "implement", "design", "synthesize"]
    followups = [
        "This modular routine drops cleanly into any Python 3 or PyTorch pipeline.",
        "Edge cases and asymptotic complexity are kept tight and readable.",
        "You can extend the type signatures or combine it with async callers.",
    ]

    for d in search_dirs:
        for file_path in sorted(glob.glob(str(d / "*.txt"))):
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
            samples.extend(blocks)

    if HUMAN_CHAT_DIR.exists():
        human_txt = HUMAN_CHAT_DIR / "05_natural_human_conversations.txt"
        if human_txt.exists():
            raw_h = human_txt.read_text(encoding="utf-8")
            samples.extend([b.strip() for b in raw_h.split("<|endoftext|>") if b.strip()])

    for d in search_dirs:

        for file_path in sorted(glob.glob(str(d / "*.json"))):
            with open(file_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            verbs = spec.get("task_verbs", verbs)
            concepts = spec.get("concepts", [])
            followups = spec.get("followups", followups)
            all_concepts.extend(concepts)
            for concept in concepts:
                for verb in verbs:
                    fup = followups[int(rng.integers(0, len(followups)))]
                    expl = paraphrase_explanation(concept["explanation"], rng)
                    samples.append(
                        f"User: {verb} a python function to {concept['topic']}\n"
                        f"Assistant: {expl}\n"
                        f"```python\n{concept['code']}\n```\n{fup}"
                    )

    if len(all_concepts) >= 2:
        for _ in range(len(all_concepts) * 3):
            c1, c2 = rng.choice(all_concepts, size=2, replace=False)
            v = verbs[int(rng.integers(0, len(verbs)))]
            e1 = paraphrase_explanation(c1["explanation"], rng)
            samples.append(
                f"User: {v} python code to {c1['topic']} and {c2['topic']}\n"
                f"Assistant: {e1} We can also pair this with a modular helper to {c2['topic']}:\n"
                f"```python\n{c1['code']}\n\n{c2['code']}\n```"
            )

    return samples


def build_ngram_blocker(samples: List[str], tokenizer: HybridCodeTokenizer, n: int = 4) -> Dict[Tuple[int, ...], Set[int]]:
    index: Dict[Tuple[int, ...], Set[int]] = {}
    for s in samples:
        ids = [int(x) for x in tokenizer.encode(s, add_special=False)]
        for i in range(max(0, len(ids) - n + 1)):
            prefix = tuple(ids[i : i + n - 1])
            index.setdefault(prefix, set()).add(int(ids[i + n - 1]))
    return index


class CompositionalTokenDataset(Dataset):
    def __init__(
        self,
        samples: List[str],
        tokenizer: HybridCodeTokenizer,
        max_seq_len: int = 192,
        token_dropout_prob: float = 0.08,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.token_dropout_prob = token_dropout_prob
        self.encoded = [tokenizer.encode(s, add_special=True) for s in samples]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoded[index].copy()
        if len(seq) > self.max_seq_len + 1:
            seq = seq[: self.max_seq_len + 1]
        x_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)
        y_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)
        valid_len = min(len(seq) - 1, self.max_seq_len)
        x_slice = seq[:valid_len].copy()
        if self.token_dropout_prob > 0 and valid_len > 5:
            mask = np.random.random(size=valid_len) < self.token_dropout_prob
            mask[:3] = False
            x_slice[mask] = self.tokenizer.unk_id
        x_np[:valid_len] = x_slice
        y_np[:valid_len] = seq[1 : valid_len + 1]
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPECausalAttention2_0(nn.Module):
    """Gen 2.0 Attention with Rotary Position Embeddings (RoPE) and QK-RMSNorm."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", torch.cos(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer("rope_sin", torch.sin(freqs).unsqueeze(0).unsqueeze(0))
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)

        cos = self.rope_cos[:, :, :t, :]
        sin = self.rope_sin[:, :, :t, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        weights = self.drop(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class SwiGLUBlock2_0(nn.Module):
    """Single hidden Transformer layer with LayerScale, RoPE QK-Norm Attention, and SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, hidden_dim: int = 448, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPECausalAttention2_0(d_model, n_heads, max_seq_len, dropout)
        self.gamma1 = nn.Parameter(torch.full((d_model,), 0.1))
        self.norm2 = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_val = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.gamma2 = nn.Parameter(torch.full((d_model,), 0.1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.drop(self.w_out(F.silu(self.w_gate(h)) * self.w_val(h)))
        x = x + self.gamma2 * ffn_out
        return x


class MatrixFlash2_0Model(nn.Module):
    """
    Matrix 2.0 Flash: 10-Hidden-Layer, 2.3M+ Parameter RoPE + QK-Norm + LayerScale + SwiGLU Coding Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 160,
        n_heads: int = 8,
        n_layers: int = 10,
        hidden_dim: int = 448,
        max_seq_len: int = 192,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        assert n_layers >= 10, "Matrix 2.0 requires at least 10 hidden layers"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SwiGLUBlock2_0(d_model, n_heads, max_seq_len, hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ngram_n: int = 4
        self.forbidden_ngrams: Dict[Tuple[int, ...], Set[int]] = {}
        self.forbidden_trigrams: Dict[Tuple[int, ...], Set[int]] = {}
        self.embedding_noise_std: float = 0.04

        assert len(self.blocks) == 10, f"Expected 10 hidden layers, got {len(self.blocks)}"
        assert self.count_parameters() >= 1_000_000, f"Expected >= 1,000,000 parameters, got {self.count_parameters()}"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        if self.training and self.embedding_noise_std > 0:
            x = x + torch.randn_like(x) * self.embedding_noise_std
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: np.ndarray,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 110,
        temperature: float = 0.85,
        top_k: int = 28,
        top_p: float = 0.92,
        repetition_penalty: float = 1.25,
        device: str = "cpu",
    ) -> List[int]:
        self.eval()
        generated = list(map(int, prompt_ids))
        prompt_len = len(generated)
        generated_trigrams: Dict[Tuple[int, int], Set[int]] = {}

        for _ in range(max_new_tokens):
            ctx = generated[-self.max_seq_len :]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = self(x)[0, -1, :].cpu().numpy()

            logits[pad_id] = -1e9
            logits[1] = -1e9

            # Block verbatim 4-grams from training files
            if self.forbidden_ngrams and self.ngram_n >= 2 and len(generated) >= self.ngram_n - 1:
                prefix = tuple(generated[-(self.ngram_n - 1) :])
                for blocked_id in self.forbidden_ngrams.get(prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 6.5

            # Softly penalize verbatim 3-grams from training files
            if self.forbidden_trigrams and len(generated) >= 2:
                tri_prefix = tuple(generated[-2:])
                for blocked_id in self.forbidden_trigrams.get(tri_prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 2.2

            # Block self-repetition of any 3-gram already generated in this reply
            if len(generated) - prompt_len >= 2:
                last_two = (generated[-2], generated[-1])
                for cand in generated_trigrams.get(last_two, ()):
                    if 0 <= cand < len(logits):
                        logits[cand] -= 1e4

            for seen_id in set(generated[-42:]):
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
            if next_id == eos_id:
                break
            if len(generated) - prompt_len >= 2:
                generated_trigrams.setdefault((generated[-2], generated[-1]), set()).add(next_id)
            generated.append(next_id)

        return generated[prompt_len:]


def compute_anti_repetition_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.15,
    entropy_weight: float = 0.05,
    max_token_prob: float = 0.65,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)

    ce = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )

    valid_mask = flat_targets != pad_id
    if not torch.any(valid_mask):
        return ce

    valid_logits = flat_logits[valid_mask]
    valid_targets = flat_targets[valid_mask]

    probs = F.softmax(valid_logits, dim=-1)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Over-confidence penalty prevents memorizing exact training file continuations
    gold_probs = probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
    confidence_penalty = (F.relu(gold_probs - max_token_prob) ** 2).mean() * 2.5

    # Temporal unlikelihood term discouraging adjacent token repetition
    b, t, v = logits.size()
    if t > 1:
        seq_probs = F.softmax(logits[:, 1:, :], dim=-1)
        prev_tokens = targets[:, :-1].unsqueeze(-1)
        repeat_probs = seq_probs.gather(-1, prev_tokens).squeeze(-1)
        non_pad = (targets[:, 1:] != pad_id).float()
        unlikelihood = (-torch.log(torch.clamp(1.0 - repeat_probs, min=1e-4)) * non_pad).sum() / (non_pad.sum() + 1e-6)
    else:
        unlikelihood = torch.tensor(0.0, device=logits.device)

    return ce - (entropy_weight * entropy) + confidence_penalty + 0.08 * unlikelihood


def train_model(epochs: int = 25, batch_size: int = 16, lr: float = 2.4e-3) -> Tuple[MatrixFlash2_0Model, HybridCodeTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(200)

    samples = load_multi_file_training_corpus(DATA_DIR, rng)
    tokenizer = HybridCodeTokenizer()
    tokenizer.build_vocab(samples)

    dataset = CompositionalTokenDataset(samples, tokenizer, max_seq_len=192, token_dropout_prob=0.08)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixFlash2_0Model(
        vocab_size=tokenizer.vocab_size,
        d_model=160,
        n_heads=8,
        n_layers=10,
        hidden_dim=448,
        max_seq_len=192,
        dropout=0.12,
    ).to(device)

    ngram_index = build_ngram_blocker(samples, tokenizer, n=4)
    trigram_index = build_ngram_blocker(samples, tokenizer, n=3)
    model.forbidden_ngrams = ngram_index
    model.forbidden_trigrams = trigram_index

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = model.count_parameters()
    print(
        f"[Matrix 2.0 Flash] Training 10-Layer ({param_count:,} params) RoPE+SwiGLU Coding Model "
        f"on {len(samples)} examples with Anti-Repetition Loss ({device})..."
    )
    model.train()
    for epoch in range(1, epochs + 1):
        losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = compute_anti_repetition_loss(logits, y, pad_id=tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Anti-Repetition Loss: {np.mean(losses):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 160,
                "n_heads": 8,
                "n_layers": 10,
                "hidden_dim": 448,
                "max_seq_len": 192,
            },
            "meta": {
                "parameters": param_count,
                "hidden_layers": 10,
                "generation": "2.0",
            },
            "anti_copy": {
                "ngram_n": 4,
                "forbidden_ngrams": [(list(k), sorted(list(v))) for k, v in ngram_index.items()],
                "forbidden_trigrams": [(list(k), sorted(list(v))) for k, v in trigram_index.items()],
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[Matrix 2.0 Flash] Saved checkpoint ({param_count:,} params, 10 hidden layers) to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[MatrixFlash2_0Model, HybridCodeTokenizer]:
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else (LEGACY_CHECKPOINT_PATH if LEGACY_CHECKPOINT_PATH.exists() else None)
    if target_path is None:
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(target_path, map_location=device, weights_only=False)
    tokenizer = HybridCodeTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    model = MatrixFlash2_0Model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "anti_copy" in ckpt:
        model.ngram_n = int(ckpt["anti_copy"].get("ngram_n", 4))
        model.forbidden_ngrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_ngrams", [])
        }
        model.forbidden_trigrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_trigrams", [])
        }
    return model, tokenizer


def answer_prompt(prompt: str, model: MatrixFlash2_0Model, tokenizer: HybridCodeTokenizer, temperature: float = 0.85) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formatted = f"<|bos|> <|user|> {prompt.strip()} \n <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    out_ids = model.generate(
        prompt_ids,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=110,
        temperature=temperature,
        top_k=28,
        top_p=0.92,
        repetition_penalty=1.25,
        device=device,
    )
    return tokenizer.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix 2.0 Flash — 10-Layer, 1M+ Param Fast Coding AI Model")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single coding prompt to run")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(answer_prompt(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print(f"\n=== Matrix 2.0 Flash (10-Layer | {model.count_parameters():,} Params | Anti-Repetition Coding AI) ===")
    print("Type a coding question (or 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Matrix 2.0 Flash.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        print(f"\nMatrix 2.0 Flash >\n{answer_prompt(user_input, model, tokenizer, temperature=args.temperature)}")


if __name__ == "__main__":
    main()
