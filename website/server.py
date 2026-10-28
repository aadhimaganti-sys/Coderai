#!/usr/bin/env python3
"""
Matrix & Delphi AI Studio — Web Server (`website/server.py`)
Serves the interactive AI Studio web interface and provides REST API endpoints
backed by the PyTorch Matrix (coding) and Delphi (conversational) models across
Generations 1.0, 1.1, and 2.0 (10 Hidden Layers, 1,000,000+ Parameters).

Endpoints:
  GET  /api/models  -> Lists all 12 models with parameter counts, hidden layers, and checkpoint status
  POST /api/chat    -> Generates a non-repeating response from the selected AI model + novelty metrics
  POST /api/train   -> Trains or fine-tunes a selected AI model with the Anti-Repetition objective
"""

import argparse
import importlib.util
import json
import mimetypes
import subprocess
import sys
import tempfile
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple


WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "matrix-2.1-moe": {
        "key": "matrix-2.1-moe",
        "name": "Matrix 2.1 MoE",
        "family": "matrix",
        "gen": "2.1",
        "tier": "moe",
        "domain": "code",
        "tagline": "12-Layer GQA + 6-Expert Fine-Grained Shared+Sparse MoE Coding AI",
        "architecture": "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 224,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "2.1" / "moe" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_moe_2.1.pt",
        "sample_prompts": [
            "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
            "write an auxiliary-loss-free MoE router with dynamic expert bias balancing in pytorch",
            "implement distributed expert parallelism all-to-all token dispatch simulation in python",
            "implement multi-head latent attention KV compression projection in pytorch",
        ],
    },
    "matrix-2.1-pro": {
        "key": "matrix-2.1-pro",
        "name": "Matrix 2.1 Pro MoE",
        "family": "matrix",
        "gen": "2.1",
        "tier": "pro",
        "domain": "code",
        "tagline": "12-Layer GQA + 6-Expert Shared+Sparse MoE Deep Systems & ML AI",
        "architecture": "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 224,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "2.1" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_pro_2.1.pt",
        "sample_prompts": [
            "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
            "write an auxiliary-loss-free MoE router with dynamic expert bias balancing in pytorch",
            "implement grouped query attention GQA with rotary position embeddings in pytorch",
        ],
    },
    "matrix-2.1-flash": {
        "key": "matrix-2.1-flash",
        "name": "Matrix 2.1 Flash MoE",
        "family": "matrix",
        "gen": "2.1",
        "tier": "flash",
        "domain": "code",
        "tagline": "12-Layer High-Speed GQA + 4-Expert Sparse MoE Coding AI",
        "architecture": "GQA (8Q/4KV) + RoPE + QK-Norm + Top-2 Sparse MoE (4 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 176,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "2.1" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_flash_2.1.pt",
        "sample_prompts": [
            "write a fast top-2 sparse MoE router function in pytorch",
            "implement a token-choice expert load balancer monitor in python",
            "write a python decorator that retries a failing function with exponential backoff",
        ],
    },
    "delphi-2.1-moe": {
        "key": "delphi-2.1-moe",
        "name": "Delphi 2.1 MoE",
        "family": "delphi",
        "gen": "2.1",
        "tier": "moe",
        "domain": "chat",
        "tagline": "12-Layer GQA + 6-Expert Fine-Grained Shared+Sparse MoE Reasoning AI",
        "architecture": "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 224,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "2.1" / "moe" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_moe_2.1.pt",
        "sample_prompts": [
            "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
            "why does specialization combined with shared context unlock deeper intelligence",
            "how do load balancing principles appear in ecosystems and creative teams",
            "hi there how is it going",
        ],
    },
    "delphi-2.1-pro": {
        "key": "delphi-2.1-pro",
        "name": "Delphi 2.1 Pro MoE",
        "family": "delphi",
        "gen": "2.1",
        "tier": "pro",
        "domain": "chat",
        "tagline": "12-Layer GQA + 6-Expert Shared+Sparse MoE Deep Science & Philosophy AI",
        "architecture": "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 224,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "2.1" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_pro_2.1.pt",
        "sample_prompts": [
            "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
            "why does specialization combined with shared context unlock deeper intelligence",
            "how do symmetry principles shape the laws of physics and mathematics",
        ],
    },
    "delphi-2.1-flash": {
        "key": "delphi-2.1-flash",
        "name": "Delphi 2.1 Flash MoE",
        "family": "delphi",
        "gen": "2.1",
        "tier": "flash",
        "domain": "chat",
        "tagline": "12-Layer High-Speed GQA + 4-Expert Sparse MoE Conversational AI",
        "architecture": "GQA (8Q/4KV) + RoPE + QK-Norm + Top-2 Sparse MoE (4 Routed + 1 Shared + Z-Loss)",
        "hidden_layers": 12,
        "d_model": 176,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "2.1" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_flash_2.1.pt",
        "sample_prompts": [
            "how can switching between different mental modes help me solve creative problems",
            "how do I keep conversations engaging and natural",
            "how can I build a habit of deep focus in a distracted world",
        ],
    },
    "matrix-2.0-pro": {
        "key": "matrix-2.0-pro",
        "name": "Matrix 2.0 Pro",
        "family": "matrix",
        "gen": "2.0",
        "tier": "pro",
        "domain": "code",
        "tagline": "10-Layer Shared+Sparse MoE Deep Coding & Architecture AI",
        "architecture": "RoPE + QK-RMSNorm + LayerScale + Top-2 Sparse MoE (4 Routed + 1 Shared Expert)",
        "hidden_layers": 10,
        "d_model": 192,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "2.0" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_pro_2.0.pt",
        "sample_prompts": [
            "hi how is it going",
            "can we talk through my software idea in plain normal language first",
            "implement grouped query attention GQA with rotary position embeddings in pytorch",
            "implement a transactional key value store with nested rollback snapshots in python",
        ],
    },
    "matrix-2.0-flash": {
        "key": "matrix-2.0-flash",
        "name": "Matrix 2.0 Flash",
        "family": "matrix",
        "gen": "2.0",
        "tier": "flash",
        "domain": "code",
        "tagline": "10-Layer High-Speed SwiGLU + LayerScale Coding AI",
        "architecture": "RoPE + QK-RMSNorm + LayerScale + SwiGLU (d_model=160, hidden=448)",
        "hidden_layers": 10,
        "d_model": 160,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "2.0" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_flash_2.0.pt",
        "sample_prompts": [
            "hey how are you doing",
            "i am stuck on a bug and getting frustrated",
            "write a python decorator that retries a failing function with exponential backoff",
            "implement a generic ring buffer in python with O(1) append and iteration",
        ],
    },
    "delphi-2.0-pro": {
        "key": "delphi-2.0-pro",
        "name": "Delphi 2.0 Pro",
        "family": "delphi",
        "gen": "2.0",
        "tier": "pro",
        "domain": "chat",
        "tagline": "10-Layer Shared+Sparse MoE Deep Reasoning, Science & Worldbuilding AI",
        "architecture": "RoPE + QK-RMSNorm + LayerScale + Top-2 Sparse MoE (4 Routed + 1 Shared Expert)",
        "hidden_layers": 10,
        "d_model": 192,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "2.0" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_pro_2.0.pt",
        "sample_prompts": [
            "hi there how is it going",
            "can you explain big ideas in a normal human way without sounding like a textbook",
            "how do symmetry principles shape the laws of physics and mathematics",
            "craft a philosophical micro-story about an observatory at the edge of a timeless sea",
        ],
    },
    "delphi-2.0-flash": {
        "key": "delphi-2.0-flash",
        "name": "Delphi 2.0 Flash",
        "family": "delphi",
        "gen": "2.0",
        "tier": "flash",
        "domain": "chat",
        "tagline": "10-Layer High-Speed Conversational, Clarity & Creativity AI",
        "architecture": "RoPE + QK-RMSNorm + LayerScale + SwiGLU (d_model=160, hidden=448)",
        "hidden_layers": 10,
        "d_model": 160,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "2.0" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_flash_2.0.pt",
        "sample_prompts": [
            "hi",
            "hello how are you doing today",
            "can we just have a normal conversation",
            "how can I build a habit of deep focus in a distracted world",
        ],
    },
    "matrix-1.1-pro": {
        "key": "matrix-1.1-pro",
        "name": "Matrix 1.1 Pro",
        "family": "matrix",
        "gen": "1.1",
        "tier": "pro",
        "domain": "code",
        "tagline": "6-Layer Sparse-MoE Distributed Systems & ML Coding AI",
        "architecture": "RoPE + QK-Norm + Top-2 Sparse MoE (6 layers)",
        "hidden_layers": 6,
        "d_model": 256,
        "n_heads": 8,
        "script": ROOT_DIR / "matrix" / "1.1" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_pro_1.1.pt",
        "sample_prompts": [
            "implement a consistent hashing ring for distributed caching in python",
            "implement the A* pathfinding algorithm on a weighted graph",
        ],
    },
    "matrix-1.1-flash": {
        "key": "matrix-1.1-flash",
        "name": "Matrix 1.1 Flash",
        "family": "matrix",
        "gen": "1.1",
        "tier": "flash",
        "domain": "code",
        "tagline": "3-Layer RoPE + SwiGLU Fast Idiomatic Coding AI",
        "architecture": "RoPE + QK-Norm + SwiGLU (3 layers)",
        "hidden_layers": 3,
        "d_model": 144,
        "n_heads": 4,
        "script": ROOT_DIR / "matrix" / "1.1" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_flash_1.1.pt",
        "sample_prompts": [
            "write a python function to parse key value pairs from a query string",
            "how do I write a context manager using contextlib",
        ],
    },
    "delphi-1.1-pro": {
        "key": "delphi-1.1-pro",
        "name": "Delphi 1.1 Pro",
        "family": "delphi",
        "gen": "1.1",
        "tier": "pro",
        "domain": "chat",
        "tagline": "6-Layer Sparse-MoE Epistemology & Systems Dialogue AI",
        "architecture": "RoPE + QK-Norm + Top-2 Sparse MoE (6 layers)",
        "hidden_layers": 6,
        "d_model": 256,
        "n_heads": 8,
        "script": ROOT_DIR / "delphi" / "1.1" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_pro_1.1.pt",
        "sample_prompts": [
            "how does information theory connect thermodynamics and human language",
            "explain how feedback loops shape both ecosystems and human habits",
        ],
    },
    "delphi-1.1-flash": {
        "key": "delphi-1.1-flash",
        "name": "Delphi 1.1 Flash",
        "family": "delphi",
        "gen": "1.1",
        "tier": "flash",
        "domain": "chat",
        "tagline": "3-Layer RoPE + SwiGLU Warm Conversational AI",
        "architecture": "RoPE + QK-Norm + SwiGLU (3 layers)",
        "hidden_layers": 3,
        "d_model": 144,
        "n_heads": 4,
        "script": ROOT_DIR / "delphi" / "1.1" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_flash_1.1.pt",
        "sample_prompts": [
            "what is a great way to spark a conversation with someone new",
            "how do I stay adaptable when plans change",
        ],
    },
    "matrix-1.0-pro": {
        "key": "matrix-1.0-pro",
        "name": "Matrix 1.0 Pro",
        "family": "matrix",
        "gen": "1.0",
        "tier": "pro",
        "domain": "code",
        "tagline": "4-Layer Pre-LN Algorithms & OOP Coding AI",
        "architecture": "Pre-LN Causal Transformer + SwiGLU (4 layers)",
        "hidden_layers": 4,
        "d_model": 192,
        "n_heads": 6,
        "script": ROOT_DIR / "matrix" / "1.0" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_pro_1.0.pt",
        "sample_prompts": [
            "implement an LRU cache class in python",
        ],
    },
    "matrix-1.0-flash": {
        "key": "matrix-1.0-flash",
        "name": "Matrix 1.0 Flash",
        "family": "matrix",
        "gen": "1.0",
        "tier": "flash",
        "domain": "code",
        "tagline": "2-Layer Lightweight Python Basics Coding AI",
        "architecture": "Pre-LN Causal Transformer (2 layers)",
        "hidden_layers": 2,
        "d_model": 128,
        "n_heads": 4,
        "script": ROOT_DIR / "matrix" / "1.0" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "matrix_flash_1.0.pt",
        "sample_prompts": [
            "write a python function to reverse a list",
        ],
    },
    "delphi-1.0-pro": {
        "key": "delphi-1.0-pro",
        "name": "Delphi 1.0 Pro",
        "family": "delphi",
        "gen": "1.0",
        "tier": "pro",
        "domain": "chat",
        "tagline": "4-Layer Deep Science & Philosophy Conversational AI",
        "architecture": "Pre-LN Causal Transformer + Gated MLP (4 layers)",
        "hidden_layers": 4,
        "d_model": 192,
        "n_heads": 6,
        "script": ROOT_DIR / "delphi" / "1.0" / "pro" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_pro_1.0.pt",
        "sample_prompts": [
            "explain how complex systems emerge from simple rules",
        ],
    },
    "delphi-1.0-flash": {
        "key": "delphi-1.0-flash",
        "name": "Delphi 1.0 Flash",
        "family": "delphi",
        "gen": "1.0",
        "tier": "flash",
        "domain": "chat",
        "tagline": "2-Layer Everyday Chat & Curiosity AI",
        "architecture": "Pre-LN Causal Transformer (2 layers)",
        "hidden_layers": 2,
        "d_model": 128,
        "n_heads": 4,
        "script": ROOT_DIR / "delphi" / "1.0" / "flash" / "model.py",
        "checkpoint": TRAINED_DIR / "delphi_flash_1.0.pt",
        "sample_prompts": [
            "hello how can I stay creative today",
        ],
    },
}

_LOADED_MODELS: Dict[str, Tuple[Any, Any, Any, list]] = {}
_PARAM_CACHE: Dict[str, int] = {}


def _load_module(key: str) -> Any:
    spec_info = MODEL_REGISTRY[key]
    script_path = spec_info["script"]
    mod_name = f"web_mod_{key.replace('-', '_').replace('.', '_')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _extract_raw_samples(mod: Any) -> list:
    import numpy as np
    rng = np.random.default_rng(42)
    if hasattr(mod, "load_multi_file_training_corpus"):
        return mod.load_multi_file_training_corpus(mod.DATA_DIR, rng)
    if hasattr(mod, "load_pro_training_corpus"):
        return mod.load_pro_training_corpus(mod.DATA_DIR, rng)
    if hasattr(mod, "load_dialogue_corpus"):
        return mod.load_dialogue_corpus(mod.DATA_DIR, rng)
    return mod.load_pro_dialogue_corpus(mod.DATA_DIR, rng)


def get_parameter_count_fast(key: str) -> int:
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    if key in _LOADED_MODELS:
        model = _LOADED_MODELS[key][1]
        cnt = sum(p.numel() for p in model.parameters())
        _PARAM_CACHE[key] = cnt
        return cnt
    try:
        mod = _load_module(key)
        samples = _extract_raw_samples(mod)
        # Instantiate tokenizer and model without training just to count exact parameters
        for attr in ("HybridCodeTokenizer", "ProCodeTokenizer", "ConversationalTokenizer", "DeepDialogueTokenizer"):
            if hasattr(mod, attr):
                tok = getattr(mod, attr)()
                tok.build_vocab(samples)
                break
        for m_attr in (
            "MatrixMoE2_1Model",
            "MatrixPro2_1MoEModel",
            "MatrixFlash2_1MoEModel",
            "DelphiMoE2_1Model",
            "DelphiPro2_1MoEModel",
            "DelphiFlash2_1MoEModel",
            "MatrixFlash2_0Model",
            "MatrixPro2_0Model",
            "DelphiFlash2_0Model",
            "DelphiPro2_0Model",
            "MatrixFlash1_1Model",
            "MatrixPro1_1Model",
            "DelphiFlash1_1Model",
            "DelphiPro1_1Model",
            "MatrixFlashModel",
            "MatrixProModel",
            "DelphiFlashModel",
            "DelphiProModel",
        ):
            if hasattr(mod, m_attr):
                cls = getattr(mod, m_attr)
                instance = cls(vocab_size=tok.vocab_size)
                cnt = sum(p.numel() for p in instance.parameters())
                _PARAM_CACHE[key] = cnt
                return cnt
    except Exception:
        pass
    return 0


def ensure_loaded(key: str, force_retrain: bool = False, epochs: int | None = None) -> Tuple[Any, Any, Any, list]:
    if not force_retrain and key in _LOADED_MODELS:
        return _LOADED_MODELS[key]
    mod = _load_module(key)
    raw_samples = _extract_raw_samples(mod)
    if force_retrain:
        model, tokenizer = mod.train_model(epochs=epochs or 25)
    else:
        if not mod.CHECKPOINT_PATH.exists():
            # Train a fast initial pass so interactive web users get a responsive model immediately
            model, tokenizer = mod.train_model(epochs=epochs or 20)
        else:
            model, tokenizer = mod.load_or_train()
    _PARAM_CACHE[key] = sum(p.numel() for p in model.parameters())
    _LOADED_MODELS[key] = (mod, model, tokenizer, raw_samples)
    return _LOADED_MODELS[key]


def compute_novelty(reply: str, raw_samples: list, tokenizer: Any, n: int = 4) -> Tuple[float, bool]:
    gen_ids = [int(x) for x in tokenizer.encode(reply, add_special=False)]
    if len(gen_ids) < n:
        return 100.0, True
    train_ngrams = set()
    for s in raw_samples:
        s_ids = [int(x) for x in tokenizer.encode(s, add_special=False)]
        for i in range(max(0, len(s_ids) - n + 1)):
            train_ngrams.add(tuple(s_ids[i : i + n]))
    gen_ngrams = [tuple(gen_ids[i : i + n]) for i in range(len(gen_ids) - n + 1)]
    novel = sum(1 for ng in gen_ngrams if ng not in train_ngrams)
    novelty_pct = (novel / max(1, len(gen_ngrams))) * 100.0
    norm_gen = " ".join(reply.lower().split())
    not_exact = all(norm_gen not in " ".join(s.lower().split()) for s in raw_samples)
    return round(novelty_pct, 1), not_exact


def _generate_with_params(
    mod: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    domain: str,
    system_instruction: str = "",
    temperature: float = 0.85,
    top_k: int = 30,
    top_p: float = 0.92,
    max_new_tokens: int = 150,
    repetition_penalty: float = 1.25,
) -> Tuple[str, int]:
    import inspect
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    effective_prompt = prompt.strip()
    if system_instruction.strip():
        effective_prompt = f"{system_instruction.strip()}\n{effective_prompt}"

    gen_sig = inspect.signature(model.generate) if hasattr(model, "generate") else None
    if gen_sig is not None and "top_p" in gen_sig.parameters and "repetition_penalty" in gen_sig.parameters:
        if domain == "code":
            formatted = f"<|bos|> <|user|> {effective_prompt} \n <|assistant|>"
        else:
            formatted = f"<|bos|> <|user|> {effective_prompt} <|assistant|>"
        prompt_ids = tokenizer.encode(formatted, add_special=False)
        out_ids = model.generate(
            prompt_ids,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        reply = tokenizer.decode(out_ids)
        return reply, len(out_ids)

    if hasattr(mod, "answer_prompt"):
        reply = mod.answer_prompt(effective_prompt, model, tokenizer, temperature=temperature)
    else:
        reply = mod.reply_to_user(effective_prompt, model, tokenizer, temperature=temperature)
    tok_cnt = len(tokenizer.encode(reply, add_special=False))
    return reply, int(tok_cnt)


def _list_workspace_files() -> List[Dict[str, Any]]:
    files_out: List[Dict[str, Any]] = []
    interesting_roots = [
        ROOT_DIR / "matrix",
        ROOT_DIR / "delphi",
        ROOT_DIR / "train.py",
        ROOT_DIR / "finetune.py",
        ROOT_DIR / "README.md",
    ]
    for item in interesting_roots:
        if item.is_file():
            rel = item.relative_to(ROOT_DIR).as_posix()
            files_out.append({"path": rel, "size": item.stat().st_size, "category": "root"})
        elif item.is_dir():
            for sub in sorted(item.rglob("*")):
                if sub.is_file() and sub.suffix in {".py", ".txt", ".json", ".md"} and "__pycache__" not in sub.parts:
                    rel = sub.relative_to(ROOT_DIR).as_posix()
                    files_out.append({"path": rel, "size": sub.stat().st_size, "category": item.name})
    return files_out


class AIStudioRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/models":
            models_out = []
            for key, info in MODEL_REGISTRY.items():
                params = get_parameter_count_fast(key)
                expert_util: List[float] = []
                if key in _LOADED_MODELS and hasattr(_LOADED_MODELS[key][1], "get_expert_utilization"):
                    expert_util = _LOADED_MODELS[key][1].get_expert_utilization()
                models_out.append(
                    {
                        "key": info["key"],
                        "name": info["name"],
                        "family": info["family"],
                        "gen": info["gen"],
                        "tier": info["tier"],
                        "domain": info["domain"],
                        "tagline": info["tagline"],
                        "architecture": info["architecture"],
                        "hidden_layers": info["hidden_layers"],
                        "d_model": info["d_model"],
                        "n_heads": info["n_heads"],
                        "parameters": params,
                        "meets_1m_spec": params >= 1_000_000 and info["hidden_layers"] >= 10,
                        "trained": info["checkpoint"].exists(),
                        "sample_prompts": info["sample_prompts"],
                        "expert_utilization": expert_util,
                    }
                )
            self._send_json(200, {"models": models_out})
            return

        if path == "/api/files":
            self._send_json(200, {"files": _list_workspace_files()})
            return

        if path == "/api/file":
            rel_path = (query.get("path", [""])[0] or "").strip().lstrip("/")
            if not rel_path:
                self._send_json(400, {"error": "Missing path parameter"})
                return
            target_file = (ROOT_DIR / rel_path).resolve()
            if not str(target_file).startswith(str(ROOT_DIR)) or not target_file.exists() or not target_file.is_file():
                self._send_json(404, {"error": f"File not found: {rel_path}"})
                return
            try:
                content_text = target_file.read_text(encoding="utf-8", errors="replace")
                self._send_json(200, {"path": rel_path, "content": content_text, "size": target_file.stat().st_size})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        # Serve static frontend files from WEB_DIR
        if path in ("/", "/index.html"):
            file_path = WEB_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            file_path = (WEB_DIR / rel).resolve()
            if not str(file_path).startswith(str(WEB_DIR)):
                self._send_json(403, {"error": "Forbidden"})
                return

        if not file_path.exists() or not file_path.is_file():
            self._send_json(404, {"error": f"Not found: {path}"})
            return

        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if path == "/api/chat":
            key = data.get("model", "matrix-2.1-moe")
            prompt = str(data.get("prompt", "")).strip()
            system_instruction = str(data.get("system_instruction", "")).strip()
            temperature = float(data.get("temperature", 0.85))
            top_k = int(data.get("top_k", 30))
            top_p = float(data.get("top_p", 0.92))
            max_new_tokens = int(data.get("max_new_tokens", 150))
            repetition_penalty = float(data.get("repetition_penalty", 1.25))

            if key not in MODEL_REGISTRY:
                self._send_json(400, {"error": f"Unknown model key: {key}"})
                return
            if not prompt:
                self._send_json(400, {"error": "Prompt cannot be empty"})
                return

            t0 = time.perf_counter()
            mod, model, tokenizer, raw_samples = ensure_loaded(key)
            reply, tok_count = _generate_with_params(
                mod=mod,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                domain=MODEL_REGISTRY[key]["domain"],
                system_instruction=system_instruction,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            novelty_pct, non_verbatim = compute_novelty(reply, raw_samples, tokenizer, n=4)
            params = sum(p.numel() for p in model.parameters())
            layers = len(model.blocks) if hasattr(model, "blocks") else MODEL_REGISTRY[key]["hidden_layers"]
            expert_util = model.get_expert_utilization() if hasattr(model, "get_expert_utilization") else []

            self._send_json(
                200,
                {
                    "model": key,
                    "name": MODEL_REGISTRY[key]["name"],
                    "reply": reply,
                    "latency_ms": elapsed_ms,
                    "parameters": params,
                    "hidden_layers": layers,
                    "token_count": tok_count,
                    "novelty_4gram_pct": novelty_pct,
                    "non_verbatim_verified": non_verbatim,
                    "expert_utilization": expert_util,
                },
            )
            return

        if path == "/api/compare":
            key_a = data.get("model_a", "matrix-2.1-moe")
            key_b = data.get("model_b", "matrix-2.0-pro")
            prompt = str(data.get("prompt", "")).strip()
            system_instruction = str(data.get("system_instruction", "")).strip()
            temperature = float(data.get("temperature", 0.85))
            top_k = int(data.get("top_k", 30))
            top_p = float(data.get("top_p", 0.92))
            max_new_tokens = int(data.get("max_new_tokens", 150))
            repetition_penalty = float(data.get("repetition_penalty", 1.25))

            if key_a not in MODEL_REGISTRY or key_b not in MODEL_REGISTRY:
                self._send_json(400, {"error": "Invalid model keys for comparison"})
                return
            if not prompt:
                self._send_json(400, {"error": "Prompt cannot be empty"})
                return

            results = []
            for k in (key_a, key_b):
                t0 = time.perf_counter()
                mod, model, tokenizer, raw_samples = ensure_loaded(k)
                reply, tok_count = _generate_with_params(
                    mod=mod,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    domain=MODEL_REGISTRY[k]["domain"],
                    system_instruction=system_instruction,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                novelty_pct, non_verbatim = compute_novelty(reply, raw_samples, tokenizer, n=4)
                params = sum(p.numel() for p in model.parameters())
                layers = len(model.blocks) if hasattr(model, "blocks") else MODEL_REGISTRY[k]["hidden_layers"]
                expert_util = model.get_expert_utilization() if hasattr(model, "get_expert_utilization") else []
                results.append(
                    {
                        "model": k,
                        "name": MODEL_REGISTRY[k]["name"],
                        "architecture": MODEL_REGISTRY[k]["architecture"],
                        "reply": reply,
                        "latency_ms": elapsed_ms,
                        "parameters": params,
                        "hidden_layers": layers,
                        "token_count": tok_count,
                        "novelty_4gram_pct": novelty_pct,
                        "non_verbatim_verified": non_verbatim,
                        "expert_utilization": expert_util,
                    }
                )
            self._send_json(200, {"ok": True, "comparisons": results})
            return

        if path == "/api/run_code":
            code = str(data.get("code", "")).strip()
            if not code:
                self._send_json(400, {"error": "Code cannot be empty"})
                return
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = Path(tmp.name)
            try:
                proc = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    cwd=str(ROOT_DIR),
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                self._send_json(
                    200,
                    {
                        "ok": proc.returncode == 0,
                        "exit_code": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "elapsed_ms": elapsed_ms,
                    },
                )
            except subprocess.TimeoutExpired:
                self._send_json(
                    200,
                    {
                        "ok": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Execution timed out after 8.0 seconds.",
                        "elapsed_ms": 8000.0,
                    },
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            return

        if path == "/api/train":
            key = data.get("model", "matrix-2.1-moe")
            epochs = int(data.get("epochs", 20))
            mode = str(data.get("mode", "train")).lower()
            if key not in MODEL_REGISTRY:
                self._send_json(400, {"error": f"Unknown model key: {key}"})
                return

            t0 = time.perf_counter()
            if mode == "finetune":
                ft_path = ROOT_DIR / "finetune.py"
                spec = importlib.util.spec_from_file_location("web_finetuner", str(ft_path))
                if spec and spec.loader:
                    ft_mod = importlib.util.module_from_spec(spec)
                    sys.modules["web_finetuner"] = ft_mod
                    spec.loader.exec_module(ft_mod)
                    ft_mod.finetune_single_model(key=key, epochs=epochs)
                _LOADED_MODELS.pop(key, None)
                mod, model, tokenizer, raw_samples = ensure_loaded(key, force_retrain=False)
            else:
                mod, model, tokenizer, raw_samples = ensure_loaded(key, force_retrain=True, epochs=epochs)

            elapsed_s = round(time.perf_counter() - t0, 2)
            params = sum(p.numel() for p in model.parameters())
            layers = len(model.blocks) if hasattr(model, "blocks") else MODEL_REGISTRY[key]["hidden_layers"]
            expert_util = model.get_expert_utilization() if hasattr(model, "get_expert_utilization") else []

            sample_prompt = MODEL_REGISTRY[key]["sample_prompts"][0]
            if hasattr(mod, "answer_prompt"):
                sample_reply = mod.answer_prompt(sample_prompt, model, tokenizer)
            else:
                sample_reply = mod.reply_to_user(sample_prompt, model, tokenizer)
            novelty_pct, non_verbatim = compute_novelty(sample_reply, raw_samples, tokenizer, n=4)

            self._send_json(
                200,
                {
                    "ok": True,
                    "mode": mode,
                    "model": key,
                    "name": MODEL_REGISTRY[key]["name"],
                    "epochs": epochs,
                    "elapsed_s": elapsed_s,
                    "parameters": params,
                    "hidden_layers": layers,
                    "novelty_4gram_pct": novelty_pct,
                    "non_verbatim_verified": non_verbatim,
                    "expert_utilization": expert_util,
                    "sample_reply": sample_reply,
                },
            )
            return

        self._send_json(404, {"error": f"Unknown endpoint: {path}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Matrix & Delphi AI Studio Website Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AIStudioRequestHandler)
    print(f"==================================================================")
    print(f" Matrix & Delphi AI Studio Website running at http://{args.host}:{args.port}")
    print(f" Serving Gen 2.0 (10-Layer, 1M+ Param), Gen 1.1 & Gen 1.0 AIs")
    print(f"==================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AI Studio server.")
        server.server_close()


if __name__ == "__main__":
    main()
