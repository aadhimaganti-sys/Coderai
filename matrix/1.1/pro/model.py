#!/usr/bin/env python3
"""
Matrix 1.1 Pro — Next-Generation Heavy Coding, Distributed Systems & ML Architecture AI Model.
Gen 1.1 Upgrades over 1.0 Pro:
- Rotary Position Embeddings (RoPE) + Query-Key RMSNorm attention
- Top-2 Sparse Mixture-of-Experts (MoE) SwiGLU feed-forward layers (4 specialized experts per block)
- Cross-generational multi-file corpus loader (combines 1.1/pro/data + 1.0/pro/data)
- Built-in Anti-Memorization Objective (entropy regularization + confidence cap + N-gram copy blocker)
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
GEN1_0_DATA_DIR = BASE_DIR.parent.parent / "1.0" / "pro" / "data"
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = TRAINED_DIR / "matrix_pro_1.1.pt"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "matrix_pro_1.1.pt"


class ProCodeTokenizer:
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

    def _tokenize(self, text: str) -> List[str]:
        normalized = text.replace("User:", " <|user|> ").replace("Assistant:", " <|assistant|> ")
        return [m.group(0) for m in self.TOKEN_REGEX.finditer(normalized)]

    def build_vocab(self, texts: List[str]) -> None:
        for text in texts:
            for tok in self._tokenize(text):
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok

    def encode(self, text: str, add_special: bool = True) -> np.ndarray:
        ids = [self.token_to_id.get(t, self.unk_id) for t in self._tokenize(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: List[int]) -> str:
        pieces: List[str] = []
        no_space_before = {".", ",", ":", ";", ")", "]", "}", "(", "[", "\n", "\n\n"}
        no_space_after = {"(", "[", "{", "\n", "\n\n", "    ", "\t", "```", "_", "@"}
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


def load_pro_training_corpus(data_dir: Path, rng: np.random.Generator) -> List[str]:
    samples: List[str] = []
    search_dirs = [data_dir]
    if GEN1_0_DATA_DIR.exists() and GEN1_0_DATA_DIR != data_dir:
        search_dirs.append(GEN1_0_DATA_DIR)

    all_modules: List[Dict[str, str]] = []
    prefixes = ["design and implement", "architect a scalable solution to", "build a production module to"]
    notes = ["This architecture scales cleanly and guards edge cases explicitly."]

    for d in search_dirs:
        for txt_path in sorted(glob.glob(str(d / "*.txt"))):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw = f.read()
            blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
            samples.extend(blocks)

        for json_path in sorted(glob.glob(str(d / "*.json"))):
            with open(json_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            prefixes = spec.get("architectural_prefixes", prefixes)
            modules = spec.get("modules", [])
            notes = spec.get("engineering_notes", notes)
            all_modules.extend(modules)

            for mod in modules:
                for prefix in prefixes:
                    note = notes[int(rng.integers(0, len(notes)))]
                    samples.append(
                        f"User: {prefix} {mod['goal']}\n"
                        f"Assistant: {mod['rationale']}\n"
                        f"```python\n{mod['code']}\n```\n{note}"
                    )

    if len(all_modules) >= 2:
        for _ in range(len(all_modules) * 3):
            m1, m2 = rng.choice(all_modules, size=2, replace=False)
            prefix = prefixes[int(rng.integers(0, len(prefixes)))]
            note = notes[int(rng.integers(0, len(notes)))]
            samples.append(
                f"User: {prefix} {m1['goal']} combined with {m2['goal']}\n"
                f"Assistant: {m1['rationale']} In addition, {m2['rationale'].lower()}\n"
                f"```python\n{m1['code']}\n\n{m2['code']}\n```\n{note}"
            )

    return samples


def build_ngram_blocker(samples: List[str], tokenizer: ProCodeTokenizer, n: int = 4) -> Dict[Tuple[int, ...], Set[int]]:
    index: Dict[Tuple[int, ...], Set[int]] = {}
    for s in samples:
        ids = [int(x) for x in tokenizer.encode(s, add_special=False)]
        for i in range(max(0, len(ids) - n + 1)):
            prefix = tuple(ids[i : i + n - 1])
            index.setdefault(prefix, set()).add(int(ids[i + n - 1]))
    return index


class ProSequenceDataset(Dataset):
    def __init__(
        self,
        samples: List[str],
        tokenizer: ProCodeTokenizer,
        max_seq_len: int = 256,
        jitter_prob: float = 0.06,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.jitter_prob = jitter_prob
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
        if self.jitter_prob > 0 and valid_len > 6:
            drop_mask = np.random.random(size=valid_len) < self.jitter_prob
            drop_mask[:4] = False
            x_slice[drop_mask] = self.tokenizer.unk_id
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


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_val = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


class SparseMoESwiGLU(nn.Module):
    """Gen 1.1 Pro Top-2 Sparse Mixture-of-Experts SwiGLU Layer."""

    def __init__(self, d_model: int, hidden_dim: int, n_experts: int = 4, top_k: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, hidden_dim, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_probs = F.softmax(self.router(x), dim=-1)
        weights, indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.zeros_like(x)
        for k_i in range(self.top_k):
            e_idx = indices[..., k_i]
            w = weights[..., k_i].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = (e_idx == expert_id).unsqueeze(-1)
                if mask.any():
                    out = out + mask * w * expert(x)
        return out


class RoPEProAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.12) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

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
        probs = self.attn_drop(F.softmax(scores, dim=-1))
        merged = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out_proj(merged))


class ProMoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPEProAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.moe = SparseMoESwiGLU(d_model, int(d_model * 2), n_experts=4, top_k=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class MatrixPro1_1Model(nn.Module):
    """
    Matrix 1.1 Pro: 6-layer, 8-head RoPE + QK-Norm + Top-2 Sparse MoE Coding Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 256,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ProMoETransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ngram_n: int = 4
        self.forbidden_ngrams: Dict[Tuple[int, ...], Set[int]] = {}

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
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
        max_new_tokens: int = 160,
        temperature: float = 0.82,
        top_k: int = 30,
        top_p: float = 0.92,
        repetition_penalty: float = 1.2,
        device: str = "cpu",
    ) -> List[int]:
        self.eval()
        generated = list(map(int, prompt_ids))
        for _ in range(max_new_tokens):
            ctx = generated[-self.max_seq_len :]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = self(x)[0, -1, :].cpu().numpy()

            logits[pad_id] = -1e9
            logits[1] = -1e9

            if self.forbidden_ngrams and self.ngram_n >= 2 and len(generated) >= self.ngram_n - 1:
                prefix = tuple(generated[-(self.ngram_n - 1) :])
                for blocked_id in self.forbidden_ngrams.get(prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 5.8

            for seen in set(generated[-48:]):
                if logits[seen] > 0:
                    logits[seen] /= repetition_penalty
                else:
                    logits[seen] *= repetition_penalty

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
            generated.append(next_id)

        return generated[len(prompt_ids) :]


def train_model(epochs: int = 45, batch_size: int = 8, lr: float = 2e-3) -> Tuple[MatrixPro1_1Model, ProCodeTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(111)

    samples = load_pro_training_corpus(DATA_DIR, rng)
    tokenizer = ProCodeTokenizer()
    tokenizer.build_vocab(samples)

    dataset = ProSequenceDataset(samples, tokenizer, max_seq_len=256, jitter_prob=0.06)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixPro1_1Model(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=256,
        dropout=0.12,
    ).to(device)

    ngram_index = build_ngram_blocker(samples, tokenizer, n=4)
    model.forbidden_ngrams = ngram_index

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[Matrix 1.1 Pro] Training 6-layer RoPE + Sparse-MoE Coding Transformer on {len(samples)} examples ({device})...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            ce = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
                ignore_index=tokenizer.pad_id,
                label_smoothing=0.14,
            )
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = ce - 0.04 * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {np.mean(epoch_losses):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "max_seq_len": 256,
            },
            "anti_copy": {
                "ngram_n": 4,
                "forbidden_ngrams": [(list(k), sorted(list(v))) for k, v in ngram_index.items()],
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[Matrix 1.1 Pro] Saved checkpoint to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[MatrixPro1_1Model, ProCodeTokenizer]:
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else (LEGACY_CHECKPOINT_PATH if LEGACY_CHECKPOINT_PATH.exists() else None)
    if target_path is None:
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(target_path, map_location=device, weights_only=False)
    tokenizer = ProCodeTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    model = MatrixPro1_1Model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "anti_copy" in ckpt:
        model.ngram_n = int(ckpt["anti_copy"].get("ngram_n", 4))
        model.forbidden_ngrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_ngrams", [])
        }
    return model, tokenizer


def answer_prompt(prompt: str, model: MatrixPro1_1Model, tokenizer: ProCodeTokenizer, temperature: float = 0.82) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formatted = f"<|bos|> <|user|> {prompt.strip()} \n <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    out_ids = model.generate(
        prompt_ids,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=160,
        temperature=temperature,
        top_k=30,
        top_p=0.92,
        repetition_penalty=1.2,
        device=device,
    )
    return tokenizer.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix 1.1 Pro — RoPE + Sparse-MoE Heavy Coding AI Model")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single coding prompt to run")
    parser.add_argument("--temperature", type=float, default=0.82, help="Sampling temperature")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(answer_prompt(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print("\n=== Matrix 1.1 Pro (Next-Gen Sparse-MoE Coding & Architecture AI) ===")
    print("Type a complex coding or architecture question (or 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Matrix 1.1 Pro.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        print(f"\nMatrix 1.1 Pro >\n{answer_prompt(user_input, model, tokenizer, temperature=args.temperature)}")


if __name__ == "__main__":
    main()
