#!/usr/bin/env python3
"""
Delphi 1.0 Pro — Deep, High-Capacity Conversational & Reasoning PyTorch + NumPy AI Model.
Designed for rich conversations, deep science & philosophy discussions, creative storytelling,
and multi-perspective synthesis.
Features:
- 6-layer Deep Causal Transformer with RMSNorm and SwiGLU feed-forward network
- Multi-file dialogue loader with cross-topic perspective recombination
- Token-dropout regularization + label smoothing + nucleus (top-p) sampling
  so Delphi Pro composes thoughtful, original responses beyond verbatim training text
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
CHECKPOINT_PATH = BASE_DIR / "delphi_pro_1.0.pt"


class DeepDialogueTokenizer:
    """
    Expressive word + punctuation tokenizer backed by NumPy arrays for rich natural language.
    """

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
    TOKEN_REGEX = re.compile(
        r"(<\|pad\|>|<\|unk\|>|<\|bos\|>|<\|eos\|>|<\|user\|>|<\|assistant\|>|[A-Za-z]+(?:[-'][A-Za-z]+)*|[0-9]+(?:\.[0-9]+)?|[.,!?;:—\-()\"']|\S)"
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
        words: List[str] = []
        no_space_before = {".", ",", "!", "?", ";", ":", ")", "—", "'"}
        no_space_after = {"(", "\"", "—"}
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "")
            if tok in self.SPECIAL_TOKENS:
                continue
            if not words:
                words.append(tok)
                continue
            if tok in no_space_before or words[-1] in no_space_after:
                words.append(tok)
            else:
                words.append(" " + tok)
        return "".join(words).strip()

    def state_dict(self) -> Dict[str, Dict]:
        return {"token_to_id": self.token_to_id}

    def load_state_dict(self, state: Dict[str, Dict]) -> None:
        self.token_to_id = dict(state["token_to_id"])
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}


def load_pro_dialogue_corpus(data_dir: Path, rng: np.random.Generator) -> List[str]:
    """
    Loads all .txt and .json conversational files from data_dir and synthesizes
    cross-perspective dialogues so Delphi Pro learns to weave ideas together
    rather than repeating static paragraphs.
    """
    dialogues: List[str] = []

    for txt_path in sorted(glob.glob(str(data_dir / "*.txt"))):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
        dialogues.extend(blocks)

    for json_path in sorted(glob.glob(str(data_dir / "*.json"))):
        with open(json_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        framings = spec.get("framings", [""])
        perspectives = spec.get("perspectives", [])
        reflections = spec.get("reflections", [""])

        for p in perspectives:
            for framing in framings:
                reflection = reflections[int(rng.integers(0, len(reflections)))]
                turn = (
                    f"User: {p['query']}\n"
                    f"Assistant: {framing} {p['insight']} {reflection}"
                ).strip()
                dialogues.append(turn)

        # Multi-perspective synthesis: combine pairs of philosophical/scientific insights
        if len(perspectives) >= 2:
            for _ in range(len(perspectives) * 3):
                p1, p2 = rng.choice(perspectives, size=2, replace=False)
                framing = framings[int(rng.integers(0, len(framings)))]
                reflection = reflections[int(rng.integers(0, len(reflections)))]
                synthesized = (
                    f"User: {p1['query']} and {p2['query']}\n"
                    f"Assistant: {framing} {p1['insight']} Building on that idea, {p2['insight']} {reflection}"
                ).strip()
                dialogues.append(synthesized)

    return dialogues


class DeepDialogueDataset(Dataset):
    def __init__(
        self,
        dialogues: List[str],
        tokenizer: DeepDialogueTokenizer,
        max_seq_len: int = 224,
        word_dropout: float = 0.05,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.word_dropout = word_dropout
        self.encoded = [tokenizer.encode(d, add_special=True) for d in dialogues]

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
        if self.word_dropout > 0 and valid_len > 6:
            mask = np.random.random(size=valid_len) < self.word_dropout
            mask[:4] = False
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
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_val = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


class ProDialogueAttention(nn.Module):
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


class DelphiProBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = ProDialogueAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * 8 / 3), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DelphiProModel(nn.Module):
    """
    Delphi 1.0 Pro: 6-layer, 8-head Deep Conversational & Reasoning Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 224,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DelphiProBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
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
        max_new_tokens: int = 120,
        temperature: float = 0.88,
        top_k: int = 32,
        top_p: float = 0.93,
        repetition_penalty: float = 1.22,
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

            for seen in set(generated[-40:]):
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


def train_model(epochs: int = 45, batch_size: int = 8, lr: float = 2e-3) -> Tuple[DelphiProModel, DeepDialogueTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(2026)

    dialogues = load_pro_dialogue_corpus(DATA_DIR, rng)
    tokenizer = DeepDialogueTokenizer()
    tokenizer.build_vocab(dialogues)

    dataset = DeepDialogueDataset(dialogues, tokenizer, max_seq_len=224, word_dropout=0.05)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DelphiProModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=224,
        dropout=0.12,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.12)

    print(f"[Delphi 1.0 Pro] Training 6-layer Deep Conversational Transformer on {len(dialogues)} dialogues ({device})...")
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
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {np.mean(losses):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "max_seq_len": 224,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[Delphi 1.0 Pro] Saved checkpoint to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[DelphiProModel, DeepDialogueTokenizer]:
    if not CHECKPOINT_PATH.exists():
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    tokenizer = DeepDialogueTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    model = DelphiProModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "anti_copy" in ckpt:
        model.ngram_n = int(ckpt["anti_copy"].get("ngram_n", 4))
        model.forbidden_ngrams = {
            tuple(prefix): set(next_ids)
            for prefix, next_ids in ckpt["anti_copy"].get("forbidden_ngrams", [])
        }
    return model, tokenizer


def reply_to_user(prompt: str, model: DelphiProModel, tokenizer: DeepDialogueTokenizer, temperature: float = 0.88) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formatted = f"<|bos|> <|user|> {prompt.strip()} <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    out_ids = model.generate(
        prompt_ids,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=120,
        temperature=temperature,
        top_k=32,
        top_p=0.93,
        repetition_penalty=1.22,
        device=device,
    )
    return tokenizer.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delphi 1.0 Pro — Deep Conversational & Reasoning AI Model")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model from data/ files")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single conversational message to send")
    parser.add_argument("--temperature", type=float, default=0.88, help="Sampling temperature for creative output")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(reply_to_user(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print("\n=== Delphi 1.0 Pro (Deep Conversational & Reasoning AI) ===")
    print("Ask a deep question or start a conversation (type 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Delphi 1.0 Pro.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        reply = reply_to_user(user_input, model, tokenizer, temperature=args.temperature)
        print(f"\nDelphi Pro > {reply}")


if __name__ == "__main__":
    main()
