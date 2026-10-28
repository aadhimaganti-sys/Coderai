#!/usr/bin/env python3
"""
Delphi 2.1 MoE — Full 12-Layer, 8M+ Parameter Fine-Grained Shared + Sparse Mixture-of-Experts Conversational AI Model.
Gen 2.1 MoE Upgrades over Gen 2.0:
- 12 Hidden Transformer Layers (`n_layers=12`, `d_model=224`) with Grouped-Query Attention (GQA, 8Q / 2KV heads)
- Rotary Position Embeddings (RoPE) + Query-Key RMSNorm + LayerScale Residual Gates
- Fine-Grained Shared-Expert + Top-2 Routed 6-Expert Sparse Mixture-of-Experts (MoE) SwiGLU across all 12 layers
- Switch Load-Balancing Auxiliary Loss + Router Z-Loss + Dynamic Expert Bias Balancing & Live Utilization Telemetry
- Cross-generational corpus synthesis (combines 2.1/moe/data + 2.0/pro/data + 1.1/pro/data + 1.0/pro/data)
- Anti-Memorization & Anti-Repetition Training Pipeline with 3-gram & 4-gram copy blocking
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
GEN2_0_DATA_DIR = BASE_DIR.parent.parent / "2.0" / "pro" / "data"
GEN1_1_DATA_DIR = BASE_DIR.parent.parent / "1.1" / "pro" / "data"
GEN1_0_DATA_DIR = BASE_DIR.parent.parent / "1.0" / "pro" / "data"
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = TRAINED_DIR / "delphi_moe_2.1.pt"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "delphi_moe_2.1.pt"


class DeepDialogueTokenizer:
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


def paraphrase_insight(text: str, rng: np.random.Generator) -> str:
    swaps = [
        ("emerges", "arises"),
        ("fundamental", "foundational"),
        ("patterns", "structures"),
        ("remarkable", "extraordinary"),
        ("connects", "bridges"),
        ("understanding", "comprehension"),
    ]
    out = text
    for old_w, new_w in swaps:
        if rng.random() < 0.45:
            out = out.replace(old_w, new_w)
    return out


def load_pro_dialogue_corpus(data_dir: Path, rng: np.random.Generator) -> List[str]:
    dialogues: List[str] = []
    search_dirs = [data_dir]
    for prev_dir in (GEN2_0_DATA_DIR, GEN1_1_DATA_DIR, GEN1_0_DATA_DIR):
        if prev_dir.exists() and prev_dir not in search_dirs:
            search_dirs.append(prev_dir)

    all_perspectives: List[Dict[str, str]] = []
    framings = ["Routing that question across our scientific, philosophical, and human lenses:"]
    reflections = ["Which facet of that synthesis sparks the most intriguing question for you?"]

    for d in search_dirs:
        for txt_path in sorted(glob.glob(str(d / "*.txt"))):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw = f.read()
            blocks = [b.strip() for b in raw.split("<|endoftext|>") if b.strip()]
            dialogues.extend(blocks)

        for json_path in sorted(glob.glob(str(d / "*.json"))):
            with open(json_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            framings = spec.get("framings", framings)
            perspectives = spec.get("perspectives", [])
            reflections = spec.get("reflections", reflections)
            all_perspectives.extend(perspectives)

            for p in perspectives:
                for framing in framings:
                    reflection = reflections[int(rng.integers(0, len(reflections)))]
                    ins = paraphrase_insight(p["insight"], rng)
                    dialogues.append(
                        f"User: {p['query']}\n"
                        f"Assistant: {framing} {ins} {reflection}"
                    )

    if len(all_perspectives) >= 2:
        for _ in range(len(all_perspectives) * 3):
            p1, p2 = rng.choice(all_perspectives, size=2, replace=False)
            framing = framings[int(rng.integers(0, len(framings)))]
            reflection = reflections[int(rng.integers(0, len(reflections)))]
            i1 = paraphrase_insight(p1["insight"], rng)
            dialogues.append(
                f"User: {p1['query']} and {p2['query']}\n"
                f"Assistant: {framing} {i1} Building on that principle, {p2['insight']} {reflection}"
            )

    return dialogues


def build_ngram_blocker(dialogues: List[str], tokenizer: DeepDialogueTokenizer, n: int = 4) -> Dict[Tuple[int, ...], Set[int]]:
    index: Dict[Tuple[int, ...], Set[int]] = {}
    for d in dialogues:
        ids = [int(x) for x in tokenizer.encode(d, add_special=False)]
        for i in range(max(0, len(ids) - n + 1)):
            prefix = tuple(ids[i : i + n - 1])
            index.setdefault(prefix, set()).add(int(ids[i + n - 1]))
    return index


class DeepDialogueDataset(Dataset):
    def __init__(
        self,
        dialogues: List[str],
        tokenizer: DeepDialogueTokenizer,
        max_seq_len: int = 240,
        word_dropout: float = 0.08,
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


class FineGrainedSharedAndRoutedMoE2_1(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.shared_expert = SwiGLUExpert(d_model, hidden_dim // 2, dropout)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, hidden_dim, dropout) for _ in range(n_experts)])
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.register_buffer("last_expert_utilization", torch.full((n_experts,), 1.0 / n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        z_loss = (torch.logsumexp(router_logits, dim=-1) ** 2).mean() * 1e-3
        gate_probs = F.softmax(router_logits, dim=-1)

        routing_scores = gate_probs + self.expert_bias
        _, indices = torch.topk(routing_scores, k=self.top_k, dim=-1)
        weights = torch.gather(gate_probs, dim=-1, index=indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        routed_out = torch.zeros_like(x)
        dispatch_counts = torch.zeros(self.n_experts, device=x.device)
        for k_i in range(self.top_k):
            e_idx = indices[..., k_i]
            w = weights[..., k_i].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = (e_idx == expert_id).unsqueeze(-1)
                dispatch_counts[expert_id] += mask.float().sum()
                if mask.any():
                    routed_out = routed_out + mask * w * expert(x)

        frac_routed = dispatch_counts / (dispatch_counts.sum() + 1e-8)
        mean_prob = gate_probs.mean(dim=(0, 1))
        balance_loss = (frac_routed * mean_prob).sum() * self.n_experts
        self.last_aux_loss = balance_loss + z_loss

        with torch.no_grad():
            self.last_expert_utilization.copy_(frac_routed.detach())
            if self.training:
                target = 1.0 / self.n_experts
                self.expert_bias.add_(0.005 * torch.sign(target - frac_routed.detach()))

        return 0.35 * self.shared_expert(x) + 0.65 * routed_out


class RoPEGQADialogueAttention2_1(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        max_seq_len: int = 240,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
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
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos = self.rope_cos[:, :, :t, :]
        sin = self.rope_sin[:, :, :t, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        probs = self.attn_drop(F.softmax(scores, dim=-1))
        merged = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out_proj(merged))


class DelphiMoE2_1Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        n_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPEGQADialogueAttention2_1(d_model, n_heads, n_kv_heads, max_seq_len, dropout)
        self.gamma1 = nn.Parameter(torch.full((d_model,), 0.1))
        self.norm2 = RMSNorm(d_model)
        self.moe = FineGrainedSharedAndRoutedMoE2_1(
            d_model,
            int(d_model * 2),
            n_experts=n_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.gamma2 = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.moe(self.norm2(x))
        return x


class DelphiMoE2_1Model(nn.Module):
    """
    Delphi 2.1 MoE: 12-Hidden-Layer, 12M+ Parameter GQA + RoPE + QK-Norm + 6-Expert Sparse MoE Conversational Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 224,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 12,
        n_experts: int = 6,
        top_k: int = 2,
        max_seq_len: int = 240,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        assert n_layers >= 10, "Delphi 2.1 MoE requires at least 10 hidden layers"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DelphiMoE2_1Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=max_seq_len,
                    n_experts=n_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ngram_n: int = 4
        self.forbidden_ngrams: Dict[Tuple[int, ...], Set[int]] = {}
        self.forbidden_trigrams: Dict[Tuple[int, ...], Set[int]] = {}
        self.embedding_noise_std: float = 0.04

        assert self.count_parameters() >= 1_000_000, f"Expected >= 1,000,000 parameters, got {self.count_parameters()}"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_moe_auxiliary_loss(self) -> torch.Tensor:
        losses = [b.moe.last_aux_loss for b in self.blocks if hasattr(b, "moe")]
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def get_expert_utilization(self) -> List[float]:
        mats = [b.moe.last_expert_utilization.detach().cpu() for b in self.blocks if hasattr(b, "moe")]
        if not mats:
            return []
        avg = torch.stack(mats).mean(dim=0)
        return [round(float(v) * 100.0, 1) for v in avg]

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
        max_new_tokens: int = 128,
        temperature: float = 0.86,
        top_k: int = 32,
        top_p: float = 0.93,
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

            if self.forbidden_ngrams and self.ngram_n >= 2 and len(generated) >= self.ngram_n - 1:
                prefix = tuple(generated[-(self.ngram_n - 1) :])
                for blocked_id in self.forbidden_ngrams.get(prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 6.5

            if self.forbidden_trigrams and len(generated) >= 2:
                tri_prefix = tuple(generated[-2:])
                for blocked_id in self.forbidden_trigrams.get(tri_prefix, ()):
                    if 0 <= blocked_id < len(logits):
                        logits[blocked_id] -= 2.2

            if len(generated) - prompt_len >= 2:
                last_two = (generated[-2], generated[-1])
                for cand in generated_trigrams.get(last_two, ()):
                    if 0 <= cand < len(logits):
                        logits[cand] -= 1e4

            for seen in set(generated[-44:]):
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
            if len(generated) - prompt_len >= 2:
                generated_trigrams.setdefault((generated[-2], generated[-1]), set()).add(next_id)
            generated.append(next_id)

        return generated[prompt_len:]


def compute_anti_repetition_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    moe_aux_loss: torch.Tensor | None = None,
    label_smoothing: float = 0.15,
    entropy_weight: float = 0.05,
    max_token_prob: float = 0.65,
    moe_aux_weight: float = 0.02,
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

    gold_probs = probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
    confidence_penalty = (F.relu(gold_probs - max_token_prob) ** 2).mean() * 2.5

    b, t, _ = logits.size()
    if t > 1:
        seq_probs = F.softmax(logits[:, 1:, :], dim=-1)
        prev_tokens = targets[:, :-1].unsqueeze(-1)
        repeat_probs = seq_probs.gather(-1, prev_tokens).squeeze(-1)
        non_pad = (targets[:, 1:] != pad_id).float()
        unlikelihood = (-torch.log(torch.clamp(1.0 - repeat_probs, min=1e-4)) * non_pad).sum() / (non_pad.sum() + 1e-6)
    else:
        unlikelihood = torch.tensor(0.0, device=logits.device)

    total = ce - (entropy_weight * entropy) + confidence_penalty + 0.08 * unlikelihood
    if moe_aux_loss is not None:
        total = total + moe_aux_weight * moe_aux_loss
    return total


def train_model(epochs: int = 25, batch_size: int = 16, lr: float = 1.8e-3) -> Tuple[DelphiMoE2_1Model, DeepDialogueTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(213)

    dialogues = load_pro_dialogue_corpus(DATA_DIR, rng)
    tokenizer = DeepDialogueTokenizer()
    tokenizer.build_vocab(dialogues)

    dataset = DeepDialogueDataset(dialogues, tokenizer, max_seq_len=160, word_dropout=0.08)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DelphiMoE2_1Model(
        vocab_size=tokenizer.vocab_size,
        d_model=224,
        n_heads=8,
        n_kv_heads=2,
        n_layers=12,
        n_experts=6,
        top_k=2,
        max_seq_len=240,
        dropout=0.12,
    ).to(device)

    ngram_index = build_ngram_blocker(dialogues, tokenizer, n=4)
    trigram_index = build_ngram_blocker(dialogues, tokenizer, n=3)
    model.forbidden_ngrams = ngram_index
    model.forbidden_trigrams = trigram_index

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = model.count_parameters()
    print(
        f"[Delphi 2.1 MoE] Training 12-Layer ({param_count:,} params) GQA + 6-Expert Sparse-MoE Conversational Model "
        f"on {len(dialogues)} dialogues with Router Z-Loss + Anti-Repetition Loss ({device})..."
    )
    model.train()
    for epoch in range(1, epochs + 1):
        losses: List[float] = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            aux_loss = model.get_moe_auxiliary_loss()
            loss = compute_anti_repetition_loss(logits, y, pad_id=tokenizer.pad_id, moe_aux_loss=aux_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            util = model.get_expert_utilization()
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | MoE Anti-Repetition Loss: {np.mean(losses):.4f} "
                f"| Expert Load (%): {util}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 224,
                "n_heads": 8,
                "n_kv_heads": 2,
                "n_layers": 12,
                "n_experts": 6,
                "top_k": 2,
                "max_seq_len": 240,
            },
            "meta": {
                "parameters": param_count,
                "hidden_layers": 12,
                "n_experts": 6,
                "generation": "2.1",
            },
            "anti_copy": {
                "ngram_n": 4,
                "forbidden_ngrams": [(list(k), sorted(list(v))) for k, v in ngram_index.items()],
                "forbidden_trigrams": [(list(k), sorted(list(v))) for k, v in trigram_index.items()],
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[Delphi 2.1 MoE] Saved checkpoint ({param_count:,} params, 12 hidden layers) to {CHECKPOINT_PATH}")
    return model, tokenizer


def load_or_train() -> Tuple[DelphiMoE2_1Model, DeepDialogueTokenizer]:
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else (LEGACY_CHECKPOINT_PATH if LEGACY_CHECKPOINT_PATH.exists() else None)
    if target_path is None:
        return train_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(target_path, map_location=device, weights_only=False)
    tokenizer = DeepDialogueTokenizer()
    tokenizer.load_state_dict(ckpt["tokenizer"])
    model = DelphiMoE2_1Model(**ckpt["config"]).to(device)
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


def reply_to_user(prompt: str, model: DelphiMoE2_1Model, tokenizer: DeepDialogueTokenizer, temperature: float = 0.86) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formatted = f"<|bos|> <|user|> {prompt.strip()} <|assistant|>"
    prompt_ids = tokenizer.encode(formatted, add_special=False)
    out_ids = model.generate(
        prompt_ids,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=128,
        temperature=temperature,
        top_k=32,
        top_p=0.93,
        repetition_penalty=1.25,
        device=device,
    )
    return tokenizer.decode(out_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delphi 2.1 MoE — 12-Layer, 1M+ Param Fine-Grained Sparse-MoE Conversational AI")
    parser.add_argument("--train", action="store_true", help="Train or retrain the model")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--prompt", type=str, default=None, help="Single conversational message")
    parser.add_argument("--temperature", type=float, default=0.86, help="Sampling temperature")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_model(epochs=args.epochs)
    else:
        model, tokenizer = load_or_train()

    if args.prompt:
        print(reply_to_user(args.prompt, model, tokenizer, temperature=args.temperature))
        return

    print(f"\n=== Delphi 2.1 MoE (12-Layer | {model.count_parameters():,} Params | 6+1 Sparse-MoE Conversational AI) ===")
    print("Ask a deep question or start a conversation (type 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Delphi 2.1 MoE.")
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        print(f"\nDelphi 2.1 MoE > {reply_to_user(user_input, model, tokenizer, temperature=args.temperature)}")


if __name__ == "__main__":
    main()
