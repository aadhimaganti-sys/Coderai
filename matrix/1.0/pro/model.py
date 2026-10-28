#!/usr/bin/env python3
"""
Matrix 1.0 Pro — Deep, High-Capacity PyTorch + NumPy Coding & Architecture AI Model.
Designed for heavy engineering tasks: complex algorithms, software architecture,
PyTorch/NumPy pipelines, and multi-module synthesis.
Features:
- 6-layer Deep Causal Transformer with RMSNorm and SwiGLU feed-forward layers
- Multi-file training corpus loader with compositional blueprint recombination
- Token-jitter regularization + label smoothing so the model synthesizes novel
  code and explanations rather than parroting training files verbatim
"""

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = TRAINED_DIR / "matrix_pro_1.0.pt"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "matrix_pro_1.0.pt"


class ProCodeTokenizer:
    """
    Fine-grained code + natural language tokenizer backed by NumPy arrays.
    Preserves indentation, operators, identifiers, and punctuation so the model
    can dynamically assemble unseen code structures and explanations.
    """

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
    TOKEN_REGEX = re.compile(
        r"(<\|pad\|>|<\|unk\|>|<\|bos\|>|<\|eos\|>|<\|user\|>|<\|assistant\|>|\n+|    |\t|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|```|[!=<>]=|\*\*|//|->|[-+*/%=<>()\[\]{}.,:;\"'`~|&^]+|\S)"
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
        no_space_after = {"(", "[", "{", "\n", "\n\n", "    ", "\t", "```", "_"}
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
    """
    Loads all training files from data_dir (.txt and .json) and synthesizes
    multi-module architectural combinations so Matrix Pro learns deep compositional
    coding patterns rather than memorizing single files.
    """
    samples: List[str] = []

    for txt_path in sorted(glob.glob(str(data_dir / "*.txt"))):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
        samples.extend(blocks)

    for json_path in sorted(glob.glob(str(data_dir / "*.json"))):
        with open(json_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        prefixes = spec.get("architectural_prefixes", ["implement"])
        modules = spec.get("modules", [])
        notes = spec.get("engineering_notes", [""])

        for mod in modules:
            for prefix in prefixes:
                note = notes[int(rng.integers(0, len(notes)))]
                sample = (
                    f"User: {prefix} {mod['goal']}\n"
                    f"Assistant: {mod['rationale']}\n"
                    f"```python\n{mod['code']}\n```\n"
                    f"{note}"
                ).strip()
                samples.append(sample)

        # Synthesize multi-component systems combining pairs of modules
        if len(modules) >= 2:
            for _ in range(len(modules) * 3):
                m1, m2 = rng.choice(modules, size=2, replace=False)
                prefix = prefixes[int(rng.integers(0, len(prefixes)))]
                note = notes[int(rng.integers(0, len(notes)))]
                combined = (
                    f"User: {prefix} {m1['goal']} integrated with {m2['goal']}\n"
                    f"Assistant: {m1['rationale']} Furthermore, {m2['rationale'].lower()}\n"
                    f"```python\n{m1['code']}\n\n{m2['code']}\n```\n"
                    f"{note}"
                ).strip()
                samples.append(combined)

    return samples


class ProSequenceDataset(Dataset):
    """NumPy-backed sequence dataset with span-jitter augmentation."""

    def __init__(
        self,
        samples: List[str],
        tokenizer: ProCodeTokenizer,
        max_seq_len: int = 256,
        jitter_prob: float = 0.05,
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
    """Root Mean Square Layer Normalization used in modern heavy LLMs."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU gated feed-forward network for high-capacity reasoning."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_val = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


class ProCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.12) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        probs = self.attn_drop(F.softmax(scores, dim=-1))
        merged = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out_proj(merged))


class ProTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = ProCausalAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, int(d_model * 8 / 3), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MatrixProModel(nn.Module):
    """
    Matrix 1.0 Pro: Deep 6-layer, 8-head causal Transformer with RMSNorm and SwiGLU
    for heavy coding, algorithms, and software architecture tasks.
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
            [ProTransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ngram_n: int = 4
        self.forbidden_ngrams: Dict[Tuple[int, ...], set] = {}

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

            # Block verbatim N-gram continuations copied from training files
            if self.forbidden_ngrams and self.ngram_n >= 2 and len(generated) >= self.ngram_n - 1:
                prefix = tuple(generated[-(self.ngram_n - 1) :])
                for blocked_id in self.forbidden_ngrams.get(prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 5.5

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


def train_model(epochs: int = 45, batch_size: int = 8, lr: float = 2e-3) -> Tuple[MatrixProModel, ProCodeTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(101)

    samples = load_pro_training_corpus(DATA_DIR, rng)
    tokenizer = ProCodeTokenizer()
    tokenizer.build_vocab(samples)

    dataset = ProSequenceDataset(samples, tokenizer, max_seq_len=256, jitter_prob=0.05)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixProModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=256,
        dropout=0.12,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    print(f"[Matrix 1.0 Pro] Training 6-layer Deep Coding Transformer on {len(samples)} examples ({device})...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
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
        },
        CHECKPOINT_PATH,
    )
    print(f"[Matrix 1.0 Pro] Saved checkpoint to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[MatrixProModel, ProCodeTokenizer]:
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else (LEGACY_CHECKPOINT_PATH if LEGACY_CHECKPOINT_PATH.exists() else None)
    if target_path is None:
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(target_path, map_location=device, weights_only=False)
    tokenizer = ProCodeTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    model = MatrixProModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "anti_copy" in ckpt:
        model.ngram_n = int(ckpt["anti_copy"].get("ngram_n", 4))
        model.forbidden_ngrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_ngrams", [])
        }
    return model, tokenizer


def answer_prompt(prompt: str, model: MatrixProModel, tokenizer: ProCodeTokenizer, temperature: float = 0.82) -> str:
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
    parser = argparse.ArgumentParser(description="Matrix 1.0 Pro — Deep Coding & Architecture AI Model")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model from data/ files")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single coding prompt to run")
    parser.add_argument("--temperature", type=float, default=0.82, help="Sampling temperature for creative output")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(answer_prompt(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print("\n=== Matrix 1.0 Pro (Heavy Coding & Architecture AI) ===")
    print("Type a complex coding or architecture question (or 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Matrix 1.0 Pro.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        reply = answer_prompt(user_input, model, tokenizer, temperature=args.temperature)
        print(f"\nMatrix Pro >\n{reply}")


if __name__ == "__main__":
    main()
