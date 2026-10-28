#!/usr/bin/env python3
"""
Matrix 1.0 Flash — Fast, lightweight PyTorch + NumPy Coding AI Model.
Designed for quick code generation, syntax help, and fast training across multiple training files.
Uses a causal Transformer architecture with compositional data augmentation, label smoothing,
and nucleus (top-p) + top-k sampling with repetition penalty so it generates novel responses
beyond verbatim training files.
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
CHECKPOINT_PATH = TRAINED_DIR / "matrix_flash_1.0.pt"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "matrix_flash_1.0.pt"


class HybridCodeTokenizer:
    """
    Word + punctuation + code-symbol tokenizer powered by NumPy.
    Splits text into fine-grained words, indentation, and code symbols so the model
    can recombine tokens into novel sentences and code blocks.
    """

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
    TOKEN_PATTERN = re.compile(r"\n+|    |\t|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|```|[!=<>]=|[-+*/%=<>()\[\]{}.,:;\"'`~|&^]+|\S")

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
        text = text.replace("User:", " <|user|> ").replace("Assistant:", " <|assistant|> ")
        tokens: List[str] = []
        for part in text.split():
            if part in self.SPECIAL_TOKENS:
                tokens.append(part)
            else:
                tokens.extend(self.TOKEN_PATTERN.findall(part))
        # Preserve newlines and indentation from raw text
        raw_tokens: List[str] = []
        pos = 0
        raw = text
        special_regex = re.compile(
            r"(<\|pad\|>|<\|unk\|>|<\|bos\|>|<\|eos\|>|<\|user\|>|<\|assistant\|>|\n+|    |\t|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|```|[!=<>]=|[-+*/%=<>()\[\]{}.,:;\"'`~|&^]+|\S)"
        )
        for match in special_regex.finditer(raw):
            raw_tokens.append(match.group(0))
        return raw_tokens

    def build_vocab(self, texts: List[str]) -> None:
        for text in texts:
            for tok in self._split(text):
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok

    def encode(self, text: str, add_special: bool = True) -> np.ndarray:
        tokens = self._split(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: List[int]) -> str:
        pieces: List[str] = []
        no_space_before = {".", ",", ":", ";", ")", "]", "}", "(", "[", "\n", "\n\n"}
        no_space_after = {"(", "[", "{", "\n", "\n\n", "    ", "\t", "```"}
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


def load_multi_file_training_corpus(data_dir: Path, rng: np.random.Generator) -> List[str]:
    """
    Loads all .txt and .json training files from data_dir and synthesizes
    recombined compositional dialogues so the model generalizes and composes
    new responses rather than memorizing single files verbatim.
    """
    samples: List[str] = []

    txt_files = sorted(glob.glob(str(data_dir / "*.txt")))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
        samples.extend(blocks)

    json_files = sorted(glob.glob(str(data_dir / "*.json")))
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        verbs = spec.get("task_verbs", ["write", "create"])
        concepts = spec.get("concepts", [])
        followups = spec.get("followups", [""])

        for concept in concepts:
            topic = concept["topic"]
            explanation = concept["explanation"]
            code = concept["code"]
            for verb in verbs:
                followup = followups[int(rng.integers(0, len(followups)))]
                sample = (
                    f"User: {verb} a python function to {topic}\n"
                    f"Assistant: {explanation}\n"
                    f"```python\n{code}\n```\n"
                    f"{followup}"
                ).strip()
                samples.append(sample)

        # Multi-concept composition: combine two coding tasks in one turn so the
        # network learns to blend explanations and code constructs dynamically.
        if len(concepts) >= 2:
            for _ in range(len(concepts) * 2):
                c1, c2 = rng.choice(concepts, size=2, replace=False)
                v1 = verbs[int(rng.integers(0, len(verbs)))]
                combined = (
                    f"User: {v1} code to {c1['topic']} and {c2['topic']}\n"
                    f"Assistant: {c1['explanation']} We can also include a helper to {c2['topic']}:\n"
                    f"```python\n{c1['code']}\n\n{c2['code']}\n```"
                )
                samples.append(combined)

    return samples


class CompositionalTokenDataset(Dataset):
    """
    PyTorch Dataset backed by NumPy arrays with stochastic token dropout/synonym jitter
    during training to discourage rote memorization and encourage generalization.
    """

    def __init__(
        self,
        samples: List[str],
        tokenizer: HybridCodeTokenizer,
        max_seq_len: int = 128,
        token_dropout_prob: float = 0.04,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.token_dropout_prob = token_dropout_prob
        self.encoded_samples: List[np.ndarray] = [
            tokenizer.encode(sample, add_special=True) for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoded_samples[index].copy()
        if len(seq) > self.max_seq_len + 1:
            seq = seq[: self.max_seq_len + 1]

        x_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)
        y_np = np.full(self.max_seq_len, self.tokenizer.pad_id, dtype=np.int64)

        valid_len = min(len(seq) - 1, self.max_seq_len)
        x_slice = seq[:valid_len].copy()

        # Stochastic input token jitter using NumPy so the model predicts from context
        if self.token_dropout_prob > 0 and valid_len > 4:
            mask = np.random.random(size=valid_len) < self.token_dropout_prob
            # Never mask special tokens at the start
            mask[:3] = False
            x_slice[mask] = self.tokenizer.unk_id

        x_np[:valid_len] = x_slice
        y_np[:valid_len] = seq[1 : valid_len + 1]

        return torch.from_numpy(x_np), torch.from_numpy(y_np)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention for fast autoregressive generation."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.size()
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.proj(out)


class FlashTransformerBlock(nn.Module):
    """Lightweight pre-norm Transformer block optimized for speed (Flash tier)."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 3 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MatrixFlashModel(nn.Module):
    """
    Matrix 1.0 Flash: Compact, low-latency causal Transformer for rapid coding tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 160,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [FlashTransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ngram_n: int = 4
        self.forbidden_ngrams: Dict[Tuple[int, ...], set] = {}

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.size()
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: np.ndarray,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 90,
        temperature: float = 0.85,
        top_k: int = 24,
        top_p: float = 0.92,
        repetition_penalty: float = 1.18,
        device: str = "cpu",
    ) -> List[int]:
        """
        Generates novel responses using temperature scaling, repetition penalty,
        anti-copy N-gram blocking, top-k, and top-p (nucleus) sampling via PyTorch and NumPy.
        """
        self.eval()
        generated = list(map(int, prompt_ids))
        for _ in range(max_new_tokens):
            context = generated[-self.max_seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(x)[0, -1, :].cpu().numpy()

            # Prevent generating pad or unk tokens
            logits[pad_id] = -1e9
            logits[1] = -1e9

            # Block verbatim N-gram continuations copied from training files
            if self.forbidden_ngrams and self.ngram_n >= 2 and len(generated) >= self.ngram_n - 1:
                prefix = tuple(generated[-(self.ngram_n - 1) :])
                for blocked_id in self.forbidden_ngrams.get(prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 5.5

            # Apply repetition penalty over recent tokens so output stays creative
            for seen_id in set(generated[-32:]):
                if logits[seen_id] > 0:
                    logits[seen_id] /= repetition_penalty
                else:
                    logits[seen_id] *= repetition_penalty

            logits = logits / max(temperature, 1e-4)

            # Top-k filtering in NumPy
            if top_k > 0 and top_k < len(logits):
                top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.full_like(logits, -1e9)
                mask[top_k_indices] = logits[top_k_indices]
                logits = mask

            # Softmax in NumPy
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / (np.sum(probs) + 1e-12)

            # Top-p (nucleus) filtering in NumPy
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumulative = np.cumsum(sorted_probs)
            cutoff = cumulative > top_p
            if np.any(cutoff):
                first_cut = int(np.argmax(cutoff)) + 1
                keep_idx = sorted_idx[: max(1, first_cut)]
                filtered = np.zeros_like(probs)
                filtered[keep_idx] = probs[keep_idx]
                probs = filtered / (np.sum(filtered) + 1e-12)

            next_id = int(np.random.choice(len(probs), p=probs))
            if next_id == eos_id:
                break
            generated.append(next_id)

        return generated[len(prompt_ids) :]


def train_model(epochs: int = 35, batch_size: int = 8, lr: float = 3e-3) -> Tuple[MatrixFlashModel, HybridCodeTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    samples = load_multi_file_training_corpus(DATA_DIR, rng)
    tokenizer = HybridCodeTokenizer()
    tokenizer.build_vocab(samples)

    dataset = CompositionalTokenDataset(samples, tokenizer, max_seq_len=160, token_dropout_prob=0.04)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixFlashModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=160,
        dropout=0.12,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Label smoothing prevents memorizing exact one-hot targets from training files
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    print(f"[Matrix 1.0 Flash] Training on {len(samples)} multi-file & recombined coding examples ({device})...")
    model.train()
    for epoch in range(1, epochs + 1):
        losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {np.mean(losses):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "max_seq_len": 160,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[Matrix 1.0 Flash] Saved checkpoint to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[MatrixFlashModel, HybridCodeTokenizer]:
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else (LEGACY_CHECKPOINT_PATH if LEGACY_CHECKPOINT_PATH.exists() else None)
    if target_path is None:
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(target_path, map_location=device, weights_only=False)
    tokenizer = HybridCodeTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    cfg = ckpt["config"]
    model = MatrixFlashModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "anti_copy" in ckpt:
        model.ngram_n = int(ckpt["anti_copy"].get("ngram_n", 4))
        model.forbidden_ngrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_ngrams", [])
        }
    return model, tokenizer


def answer_prompt(prompt: str, model: MatrixFlashModel, tokenizer: HybridCodeTokenizer, temperature: float = 0.85) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formatted = f"<|bos|> <|user|> {prompt.strip()} \n <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    out_ids = model.generate(
        prompt_ids,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=90,
        temperature=temperature,
        top_k=24,
        top_p=0.92,
        repetition_penalty=1.18,
        device=device,
    )
    return tokenizer.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix 1.0 Flash — Fast Coding AI Model")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model from data/ files")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single coding prompt to run")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature for creative output")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(answer_prompt(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print("\n=== Matrix 1.0 Flash (Fast Coding AI) ===")
    print("Type a coding question (or 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Matrix 1.0 Flash.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        reply = answer_prompt(user_input, model, tokenizer, temperature=args.temperature)
        print(f"\nMatrix Flash >\n{reply}")


if __name__ == "__main__":
    main()
