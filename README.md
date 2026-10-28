# Gemini-3: Matrix & Delphi AI Studio

<p align="center">
  <img src="https://images.unsplash.com/photo-1526379095098-d400fd0bf935?auto=format&fit=crop&w=1400&q=80" alt="AI coding and research workspace" width="100%" />
</p>

A Gemini-3-inspired local AI lab for training, exploring, and interacting with a family of compact language models built around two themes:

- Matrix: coding, systems, algorithms, and engineering-heavy generation
- Delphi: conversation, reasoning, science, creativity, and worldbuilding

This repo combines a PyTorch model training stack, anti-memorization fine-tuning, and a browser-based AI studio UI for experimenting with different model generations and tiers.

## Highlights

- 18 models across Gen 1.0, 1.1, 2.0, and **2.1 MoE**
- Matrix and Delphi model families
- Flash, Pro, and flagship **MoE** variants
- Web dashboard for prompt testing and model switching
- Anti-repetition / anti-copy fine-tuning pipeline
- Lightweight local research workflow for experimentation

## Gallery

<p align="center">
  <img src="https://images.unsplash.com/photo-1515879218367-8466d910aaa4?auto=format&fit=crop&w=1200&q=80" alt="Developer workstation with code" width="48%" />
  <img src="https://images.unsplash.com/photo-1531482615713-2afd69097998?auto=format&fit=crop&w=1200&q=80" alt="AI and research interface" width="48%" />
</p>

## Project architecture

This repository contains:

- `train.py` — trains the available model set across generations and families
- `finetune.py` — applies anti-memorization and anti-repetition fine-tuning
- `website/server.py` — local web server for the interactive AI Studio UI
- `website/index.html` — dashboard and chat interface
- `matrix/` — Matrix model family implementations and training data
- `delphi/` — Delphi model family implementations and training data

<p align="center">
  <img src="https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&w=1400&q=80" alt="AI model training and experimentation" width="100%" />
</p>

## Model families

### Matrix
Designed for code generation, software patterns, algorithms, and structured technical writing.

### Delphi
Designed for reflective dialogue, creative writing, science communication, and nuanced conversation.

### Generations
- Gen 1.0
- Gen 1.1
- Gen 2.0 (10-Layer, 1M+ Parameters)
- **Gen 2.1 MoE** (12-Layer Grouped-Query Attention + Fine-Grained Shared & Routed Sparse Mixture-of-Experts with Router Z-Loss and Dynamic Bias Balancing across all models)

The Gen 2.1 MoE models are the most advanced in the project, featuring 12 hidden layers, Grouped-Query Attention (GQA), RoPE + QK-RMSNorm, and Shared + Top-2 Routed SwiGLU Sparse MoE layers across `moe`, `pro`, and `flash` tiers.

## Quick start

### 1) Train a model set

```bash
python train.py --gen 2.1 --family matrix
```

Train a specific model:

```bash
python train.py --target matrix-2.1-moe --epochs 10
```

Run the full training sweep:

```bash
python train.py
```

### 2) Run the anti-copy fine-tuning pass

```bash
python finetune.py --gen 2.1
```

### 3) Launch the web interface

```bash
python website/server.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Example usage

After starting the website, you can:

- switch between Matrix and Delphi models
- filter by generation (1.0 / 1.1 / 2.0 / 2.1 MoE)
- test prompts from the sample prompt chips
- adjust temperature for generation style
- train or fine-tune the active model
- review novelty and anti-copy metrics

## Anti-repetition strategy

The project includes an anti-copy pipeline designed to reduce memorization and repetition during generation. According to the code, this includes mechanisms such as:

- cross-file recombination
- embedding-space jitter
- entropy regularization
- temporal unlikelihood training
- 3-gram and 4-gram blocking
- self-repetition prevention

This makes the project useful as a research-oriented local generative AI environment rather than a simple one-off script.

## Repository layout

```text
.
├── train.py
├── finetune.py
├── matrix/
│   ├── 1.0/
│   ├── 1.1/
│   ├── 2.0/
│   └── 2.1/
├── delphi/
│   ├── 1.0/
│   ├── 1.1/
│   ├── 2.0/
│   └── 2.1/
├── website/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── server.py
├── .gitignore
└── README.md
```

## Notes

This project is intended as a local experimentation and AI studio repo. It is designed for research, prototyping, and learning about transformer-style model training, generation behavior, and anti-copy safeguards.
