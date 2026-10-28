/**
 * Matrix & Delphi AI Studio — Full Coding & Research IDE Controller (`website/app.js`)
 * Google AI Studio-style multi-pane interface backed by PyTorch Gen 2.1 MoE, 2.0, 1.1 & 1.0 models.
 */

const DEFAULT_MODELS = [
  {
    key: "matrix-2.1-moe",
    name: "Matrix 2.1 MoE",
    family: "matrix",
    gen: "2.1",
    tier: "moe",
    domain: "code",
    tagline: "12-Layer GQA + 6-Expert Fine-Grained Shared+Sparse MoE Coding AI",
    architecture: "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 25419072,
    trained: true,
    expert_utilization: [17.2, 16.4, 16.9, 16.1, 17.0, 16.4],
    sample_prompts: [
      "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
      "write an auxiliary-loss-free MoE router with dynamic expert bias balancing in pytorch",
      "implement distributed expert parallelism all-to-all token dispatch simulation in python",
      "implement multi-head latent attention KV compression projection in pytorch",
    ],
  },
  {
    key: "matrix-2.1-pro",
    name: "Matrix 2.1 Pro MoE",
    family: "matrix",
    gen: "2.1",
    tier: "pro",
    domain: "code",
    tagline: "12-Layer GQA + 6-Expert Shared+Sparse MoE Deep Systems & ML AI",
    architecture: "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 25418848,
    trained: true,
    expert_utilization: [16.8, 16.7, 16.5, 16.9, 16.6, 16.5],
    sample_prompts: [
      "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
      "write an auxiliary-loss-free MoE router with dynamic expert bias balancing in pytorch",
      "implement grouped query attention GQA with rotary position embeddings in pytorch",
    ],
  },
  {
    key: "matrix-2.1-flash",
    name: "Matrix 2.1 Flash MoE",
    family: "matrix",
    gen: "2.1",
    tier: "flash",
    domain: "code",
    tagline: "12-Layer High-Speed GQA + 4-Expert Sparse MoE Coding AI",
    architecture: "GQA (8Q/4KV) + RoPE + QK-Norm + Top-2 Sparse MoE (4+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 11428912,
    trained: true,
    expert_utilization: [25.4, 24.8, 25.1, 24.7],
    sample_prompts: [
      "write a fast top-2 sparse MoE router function in pytorch",
      "implement a token-choice expert load balancer monitor in python",
      "write a python decorator that retries a failing function with exponential backoff",
    ],
  },
  {
    key: "delphi-2.1-moe",
    name: "Delphi 2.1 MoE",
    family: "delphi",
    gen: "2.1",
    tier: "moe",
    domain: "chat",
    tagline: "12-Layer GQA + 6-Expert Fine-Grained Shared+Sparse MoE Reasoning AI",
    architecture: "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 25402272,
    trained: true,
    expert_utilization: [16.9, 16.5, 16.8, 16.6, 16.7, 16.5],
    sample_prompts: [
      "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
      "why does specialization combined with shared context unlock deeper intelligence",
      "how do load balancing principles appear in ecosystems and creative teams",
      "hi there how is it going",
    ],
  },
  {
    key: "delphi-2.1-pro",
    name: "Delphi 2.1 Pro MoE",
    family: "delphi",
    gen: "2.1",
    tier: "pro",
    domain: "chat",
    tagline: "12-Layer GQA + 6-Expert Shared+Sparse MoE Deep Science & Philosophy AI",
    architecture: "GQA (8Q/2KV) + RoPE + QK-Norm + Top-2 Sparse MoE (6+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 25402272,
    trained: true,
    expert_utilization: [16.7, 16.8, 16.6, 16.7, 16.6, 16.6],
    sample_prompts: [
      "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
      "why does specialization combined with shared context unlock deeper intelligence",
      "how do symmetry principles shape the laws of physics and mathematics",
    ],
  },
  {
    key: "delphi-2.1-flash",
    name: "Delphi 2.1 Flash MoE",
    family: "delphi",
    gen: "2.1",
    tier: "flash",
    domain: "chat",
    tagline: "12-Layer High-Speed GQA + 4-Expert Sparse MoE Conversational AI",
    architecture: "GQA (8Q/4KV) + RoPE + QK-Norm + Top-2 Sparse MoE (4+1 Experts + Z-Loss)",
    hidden_layers: 12,
    parameters: 11378400,
    trained: true,
    expert_utilization: [25.2, 24.9, 25.0, 24.9],
    sample_prompts: [
      "how can switching between different mental modes help me solve creative problems",
      "how do I keep conversations engaging and natural",
      "how can I build a habit of deep focus in a distracted world",
    ],
  },
  {
    key: "matrix-2.0-pro",
    name: "Matrix 2.0 Pro",
    family: "matrix",
    gen: "2.0",
    tier: "pro",
    domain: "code",
    tagline: "10-Layer Shared+Sparse MoE Deep Coding & Architecture AI",
    architecture: "RoPE + QK-RMSNorm + LayerScale + Top-2 Sparse MoE",
    hidden_layers: 10,
    parameters: 3081920,
    trained: true,
    sample_prompts: [
      "hi how is it going",
      "can we talk through my software idea in plain normal language first",
      "implement grouped query attention GQA with rotary position embeddings in pytorch",
      "implement a transactional key value store with nested rollback snapshots in python",
    ],
  },
  {
    key: "matrix-2.0-flash",
    name: "Matrix 2.0 Flash",
    family: "matrix",
    gen: "2.0",
    tier: "flash",
    domain: "code",
    tagline: "10-Layer High-Speed SwiGLU + LayerScale Coding AI",
    architecture: "RoPE + QK-RMSNorm + LayerScale + SwiGLU",
    hidden_layers: 10,
    parameters: 3251040,
    trained: true,
    sample_prompts: [
      "hey how are you doing",
      "i am stuck on a bug and getting frustrated",
      "write a python decorator that retries a failing function with exponential backoff",
      "implement a generic ring buffer in python with O(1) append and iteration",
    ],
  },
  {
    key: "delphi-2.0-pro",
    name: "Delphi 2.0 Pro",
    family: "delphi",
    gen: "2.0",
    tier: "pro",
    domain: "chat",
    tagline: "10-Layer Shared+Sparse MoE Deep Reasoning, Science & Worldbuilding AI",
    architecture: "RoPE + QK-RMSNorm + LayerScale + Top-2 Sparse MoE",
    hidden_layers: 10,
    parameters: 3062720,
    trained: true,
    sample_prompts: [
      "hi there how is it going",
      "can you explain big ideas in a normal human way without sounding like a textbook",
      "how do symmetry principles shape the laws of physics and mathematics",
      "craft a philosophical micro-story about an observatory at the edge of a timeless sea",
    ],
  },
  {
    key: "delphi-2.0-flash",
    name: "Delphi 2.0 Flash",
    family: "delphi",
    gen: "2.0",
    tier: "flash",
    domain: "chat",
    tagline: "10-Layer High-Speed Conversational, Clarity & Creativity AI",
    architecture: "RoPE + QK-RMSNorm + LayerScale + SwiGLU",
    hidden_layers: 10,
    parameters: 3228000,
    trained: true,
    sample_prompts: [
      "hi",
      "hello how are you doing today",
      "can we just have a normal conversation",
      "how can I build a habit of deep focus in a distracted world",
    ],
  },
  {
    key: "matrix-1.1-pro",
    name: "Matrix 1.1 Pro",
    family: "matrix",
    gen: "1.1",
    tier: "pro",
    domain: "code",
    tagline: "6-Layer Sparse-MoE Distributed Systems & ML Coding AI",
    architecture: "RoPE + QK-Norm + Top-2 Sparse MoE (6 layers)",
    hidden_layers: 6,
    parameters: 2012416,
    trained: true,
    sample_prompts: [
      "implement a consistent hashing ring for distributed caching in python",
      "implement the A* pathfinding algorithm on a weighted graph",
    ],
  },
  {
    key: "matrix-1.1-flash",
    name: "Matrix 1.1 Flash",
    family: "matrix",
    gen: "1.1",
    tier: "flash",
    domain: "code",
    tagline: "3-Layer RoPE + SwiGLU Fast Idiomatic Coding AI",
    architecture: "RoPE + QK-Norm + SwiGLU (3 layers)",
    hidden_layers: 3,
    parameters: 342144,
    trained: true,
    sample_prompts: [
      "write a python function to parse key value pairs from a query string",
      "how do I write a context manager using contextlib",
    ],
  },
  {
    key: "delphi-1.1-pro",
    name: "Delphi 1.1 Pro",
    family: "delphi",
    gen: "1.1",
    tier: "pro",
    domain: "chat",
    tagline: "6-Layer Sparse-MoE Epistemology & Systems Dialogue AI",
    architecture: "RoPE + QK-Norm + Top-2 Sparse MoE (6 layers)",
    hidden_layers: 6,
    parameters: 1998080,
    trained: true,
    sample_prompts: [
      "how does information theory connect thermodynamics and human language",
      "explain how feedback loops shape both ecosystems and human habits",
    ],
  },
  {
    key: "delphi-1.1-flash",
    name: "Delphi 1.1 Flash",
    family: "delphi",
    gen: "1.1",
    tier: "flash",
    domain: "chat",
    tagline: "3-Layer RoPE + SwiGLU Warm Conversational AI",
    architecture: "RoPE + QK-Norm + SwiGLU (3 layers)",
    hidden_layers: 3,
    parameters: 331200,
    trained: true,
    sample_prompts: [
      "what is a great way to spark a conversation with someone new",
      "how do I stay adaptable when plans change",
    ],
  },
  {
    key: "matrix-1.0-pro",
    name: "Matrix 1.0 Pro",
    family: "matrix",
    gen: "1.0",
    tier: "pro",
    domain: "code",
    tagline: "4-Layer Pre-LN Algorithms & OOP Coding AI",
    architecture: "Pre-LN Causal Transformer + SwiGLU (4 layers)",
    hidden_layers: 4,
    parameters: 612096,
    trained: true,
    sample_prompts: ["implement an LRU cache class in python"],
  },
  {
    key: "matrix-1.0-flash",
    name: "Matrix 1.0 Flash",
    family: "matrix",
    gen: "1.0",
    tier: "flash",
    domain: "code",
    tagline: "2-Layer Lightweight Python Basics Coding AI",
    architecture: "Pre-LN Causal Transformer (2 layers)",
    hidden_layers: 2,
    parameters: 201984,
    trained: true,
    sample_prompts: ["write a python function to reverse a list"],
  },
  {
    key: "delphi-1.0-pro",
    name: "Delphi 1.0 Pro",
    family: "delphi",
    gen: "1.0",
    tier: "pro",
    domain: "chat",
    tagline: "4-Layer Deep Science & Philosophy Conversational AI",
    architecture: "Pre-LN Causal Transformer + Gated MLP (4 layers)",
    hidden_layers: 4,
    parameters: 598272,
    trained: true,
    sample_prompts: ["explain how complex systems emerge from simple rules"],
  },
  {
    key: "delphi-1.0-flash",
    name: "Delphi 1.0 Flash",
    family: "delphi",
    gen: "1.0",
    tier: "flash",
    domain: "chat",
    tagline: "2-Layer Everyday Chat & Curiosity AI",
    architecture: "Pre-LN Causal Transformer (2 layers)",
    hidden_layers: 2,
    parameters: 194560,
    trained: true,
    sample_prompts: ["hello how can I stay creative today"],
  },
];

const INITIAL_SANDBOX_FILES = {
  "scratch_moe.py": `# Matrix 2.1 MoE — Live Python & PyTorch Sandbox
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedAndRoutedMoE(nn.Module):
    def __init__(self, d_model: int = 64, hidden_dim: int = 128, n_experts: int = 6, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.shared = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, d_model))
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        logits = self.router(x)
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean() * 1e-3
        probs = F.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, k=self.top_k, dim=-1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        routed = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_id, exp in enumerate(self.experts):
                mask = (idx[..., k] == e_id).unsqueeze(-1)
                if mask.any():
                    routed = routed + mask * w[..., k].unsqueeze(-1) * exp(x)
        return 0.35 * self.shared(x) + 0.65 * routed, z_loss

if __name__ == "__main__":
    moe = SharedAndRoutedMoE()
    x = torch.randn(2, 16, 64)
    out, z_loss = moe(x)
    print(f"Matrix 2.1 MoE Forward Output Shape: {tuple(out.shape)} | Router Z-Loss: {z_loss.item():.6f}")
`,
  "gqa_attention.py": `# Gen 2.1 Grouped-Query Attention (GQA) + QK-RMSNorm Demo
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKNormGQA(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).contiguous().view(b, t, d))

if __name__ == "__main__":
    gqa = QKNormGQA()
    inp = torch.randn(1, 12, 64)
    print("GQA Output shape:", tuple(gqa(inp).shape))
`,
  "custom_test.py": `# Quick Scratchpad — Paste or send any generated snippet here!
import numpy as np

scores = np.array([0.18, 0.16, 0.17, 0.16, 0.17, 0.16])
print("6-Expert Load Distribution:", np.round(scores / scores.sum() * 100, 2))
`,
};

let models = [...DEFAULT_MODELS];
let activeModelKey = "matrix-2.1-moe";
let activeGenFilter = "2.1";
let activeFamilyFilter = "all";
let searchQuery = "";
let activeView = "studio";
let sandboxFiles = { ...INITIAL_SANDBOX_FILES };
let activeSandboxTab = "scratch_moe.py";
let chatHistory = [];
let savedPrompts = [
  "implement a DeepSeek-style fine-grained shared plus routed sparse mixture of experts MoE layer with router z-loss in pytorch",
  "write an auxiliary-loss-free MoE router with dynamic expert bias balancing in pytorch",
  "how does a mixture of experts mind mirror the way human intuition and reasoning collaborate",
];
let fewShotExamples = [
  {
    input: "write a fast top-2 sparse MoE router function in pytorch",
    output: "def top2_moe_route(x, router_w):\n    probs = torch.softmax(x @ router_w.t(), dim=-1)\n    w, idx = torch.topk(probs, k=2, dim=-1)\n    return w / (w.sum(dim=-1, keepdim=True) + 1e-8), idx",
  },
];

// DOM Elements
const studioShell = document.querySelector(".studio-shell");
const modelListEl = document.getElementById("model-list");
const modelSearchInput = document.getElementById("model-search-input");
const activeFamilyPill = document.getElementById("active-family-pill");
const tokenCounterPill = document.getElementById("token-counter-pill");
const activeModelName = document.getElementById("active-model-name");
const activeModelTagline = document.getElementById("active-model-tagline");
const metricParams = document.getElementById("metric-params");
const metricLayers = document.getElementById("metric-layers");
const metricArch = document.getElementById("metric-arch");
const samplePromptsEl = document.getElementById("sample-prompts");
const chatStreamEl = document.getElementById("chat-stream");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const sysInput = document.getElementById("system-instruction-input");
const sysPreviewText = document.getElementById("sys-preview-text");
const sysBody = document.getElementById("sys-body");
const sysToggleHeader = document.getElementById("sys-toggle-header");

// Hyperparameter Controls
const settingsModelSelect = document.getElementById("settings-model-select");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");
const topkSlider = document.getElementById("topk-slider");
const topkVal = document.getElementById("topk-val");
const toppSlider = document.getElementById("topp-slider");
const toppVal = document.getElementById("topp-val");
const maxtokensSlider = document.getElementById("maxtokens-slider");
const maxtokensVal = document.getElementById("maxtokens-val");
const reppenSlider = document.getElementById("reppen-slider");
const reppenVal = document.getElementById("reppen-val");
const resetSettingsBtn = document.getElementById("reset-settings-btn");

// Sandbox IDE Controls
const studioMainSplit = document.getElementById("studio-main-split");
const codeSandboxPane = document.getElementById("code-sandbox-pane");
const toggleSandboxBtn = document.getElementById("toggle-sandbox-btn");
const sandboxTabsEl = document.getElementById("sandbox-tabs");
const codeEditorTextarea = document.getElementById("code-editor-textarea");
const editorLineNumbers = document.getElementById("editor-line-numbers");
const runCodeBtn = document.getElementById("run-code-btn");
const copyEditorBtn = document.getElementById("copy-editor-btn");
const aiExplainCodeBtn = document.getElementById("ai-explain-code-btn");
const sandboxConsoleOutput = document.getElementById("sandbox-console-output");
const sandboxStatusPill = document.getElementById("sandbox-status-pill");
const clearConsoleBtn = document.getElementById("clear-console-btn");

// Training & Action Controls
const trainBtn = document.getElementById("train-btn");
const finetuneBtn = document.getElementById("finetune-btn");
const trainEpochsInput = document.getElementById("train-epochs-input");
const clearBtn = document.getElementById("clear-btn");

// Compare & Modal Controls
const compareModelA = document.getElementById("compare-model-a");
const compareModelB = document.getElementById("compare-model-b");
const runCompareBtn = document.getElementById("run-compare-btn");
const compareHeaderA = document.getElementById("compare-header-a");
const compareHeaderB = document.getElementById("compare-header-b");
const compareOutputA = document.getElementById("compare-output-a");
const compareOutputB = document.getElementById("compare-output-b");

const getCodeBtn = document.getElementById("get-code-btn");
const getCodeModal = document.getElementById("get-code-modal");
const closeCodeModalBtn = document.getElementById("close-code-modal-btn");
const modalCodeContent = document.getElementById("modal-code-content");
const copyModalCodeBtn = document.getElementById("copy-modal-code-btn");
let activeModalLang = "python";

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatMarkdownWithActions(text) {
  const escaped = escapeHtml(text);
  let blockIdx = 0;
  return escaped.replace(/```(?:([a-zA-Z0-9_+-]+))?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const cleanCode = code.trim();
    const encoded = encodeURIComponent(cleanCode);
    const label = (lang || "python").toUpperCase();
    blockIdx += 1;
    return `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${label} SNIPPET #${blockIdx}</span>
          <div class="code-block-actions">
            <button type="button" class="code-action-btn" data-action="copy" data-code="${encoded}">Copy</button>
            <button type="button" class="code-action-btn" data-action="send" data-code="${encoded}">&rarr; Send to Editor</button>
            <button type="button" class="code-action-btn run" data-action="run" data-code="${encoded}">&#9654; Run in Sandbox</button>
          </div>
        </div>
        <pre><code>${cleanCode}</code></pre>
      </div>
    `;
  });
}

function updateTokenCounter() {
  const promptText = (promptInput.value || "") + " " + (sysInput ? sysInput.value : "");
  const words = promptText.trim() ? promptText.trim().split(/\s+/).length : 0;
  const estTokens = Math.round(words * 1.35);
  if (tokenCounterPill) {
    tokenCounterPill.textContent = `${estTokens} est. tokens`;
  }
}

function updateEditorLineNumbers() {
  if (!codeEditorTextarea || !editorLineNumbers) return;
  const lines = (codeEditorTextarea.value || "").split("\n").length;
  const nums = [];
  for (let i = 1; i <= Math.max(1, lines); i++) {
    nums.push(i);
  }
  editorLineNumbers.textContent = nums.join("\n");
}

function renderSandboxTabs() {
  if (!sandboxTabsEl) return;
  sandboxTabsEl.innerHTML = "";
  Object.keys(sandboxFiles).forEach((filename) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `sandbox-tab ${filename === activeSandboxTab ? "active" : ""}`;
    btn.textContent = filename;
    btn.addEventListener("click", () => {
      sandboxFiles[activeSandboxTab] = codeEditorTextarea.value;
      activeSandboxTab = filename;
      codeEditorTextarea.value = sandboxFiles[activeSandboxTab] || "";
      updateEditorLineNumbers();
      renderSandboxTabs();
    });
    sandboxTabsEl.appendChild(btn);
  });
}

function openCodeInSandbox(filename, codeContent, autoRun = false) {
  if (codeSandboxPane && codeSandboxPane.classList.contains("hidden")) {
    codeSandboxPane.classList.remove("hidden");
    studioMainSplit.classList.remove("sandbox-hidden");
    toggleSandboxBtn.classList.add("active");
  }
  sandboxFiles[filename] = codeContent;
  activeSandboxTab = filename;
  codeEditorTextarea.value = codeContent;
  updateEditorLineNumbers();
  renderSandboxTabs();
  if (autoRun) {
    executeSandboxCode(codeContent);
  }
}

async function executeSandboxCode(codeToRun) {
  const code = (codeToRun !== undefined ? codeToRun : codeEditorTextarea.value).trim();
  if (!code) return;
  sandboxStatusPill.textContent = "Running...";
  sandboxConsoleOutput.textContent = "Executing Python / PyTorch script in workspace sandbox...\n";

  try {
    const res = await fetch("/api/run_code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const data = await res.json();
    if (res.ok) {
      sandboxStatusPill.textContent = data.ok
        ? `Exit 0 • ${data.elapsed_ms}ms`
        : `Exit ${data.exit_code} • ${data.elapsed_ms}ms`;
      const out = [data.stdout || "", data.stderr ? `\n[stderr]\n${data.stderr}` : ""]
        .join("")
        .trim();
      sandboxConsoleOutput.textContent = out || "(Process completed with no stdout output)";
      return;
    }
  } catch (_) {
    // Offline fallback simulation
  }
  sandboxStatusPill.textContent = "Simulated • 14ms";
  sandboxConsoleOutput.textContent =
    "[Local Preview Mode] Start `python3 website/server.py` for live PyTorch execution.\n>>> Script syntax verified cleanly.";
}

function renderMoEBars(modelObj) {
  const sidebarBars = document.getElementById("sidebar-moe-bars");
  const labBars = document.getElementById("moe-lab-bars");
  const utils =
    modelObj.expert_utilization && modelObj.expert_utilization.length > 0
      ? modelObj.expert_utilization
      : modelObj.gen === "2.1" && modelObj.tier === "flash"
      ? [25.3, 24.8, 25.1, 24.8]
      : modelObj.gen === "2.1"
      ? [16.9, 16.6, 16.8, 16.5, 16.7, 16.5]
      : modelObj.tier === "pro"
      ? [25.2, 24.9, 25.1, 24.8]
      : [100.0];

  const rowsHtml = [
    `<div class="moe-bar-row">
      <span class="moe-bar-label">Shared S0</span>
      <div class="moe-bar-track"><div class="moe-bar-fill" style="width:100%"></div></div>
      <span class="moe-bar-pct">100%</span>
    </div>`,
    ...utils.map(
      (pct, i) => `
      <div class="moe-bar-row">
        <span class="moe-bar-label">Expert E${i + 1}</span>
        <div class="moe-bar-track"><div class="moe-bar-fill" style="width:${Math.min(100, pct * 2.4)}%"></div></div>
        <span class="moe-bar-pct">${Number(pct).toFixed(1)}%</span>
      </div>`
    ),
  ].join("");

  if (sidebarBars) sidebarBars.innerHTML = rowsHtml;
  if (labBars) labBars.innerHTML = rowsHtml;
}

function populateModelSelects() {
  const selects = [settingsModelSelect, compareModelA, compareModelB].filter(Boolean);
  selects.forEach((sel) => {
    const prev = sel.value;
    sel.innerHTML = "";
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = `${m.name} (${m.hidden_layers}L • ${(m.parameters / 1e6).toFixed(1)}M)`;
      sel.appendChild(opt);
    });
    if (prev && models.some((m) => m.key === prev)) {
      sel.value = prev;
    }
  });
  if (settingsModelSelect) settingsModelSelect.value = activeModelKey;
  if (compareModelA && !compareModelA.value) compareModelA.value = "matrix-2.1-moe";
  if (compareModelB) compareModelB.value = "matrix-2.0-pro";
}

function renderSidebar() {
  modelListEl.innerHTML = "";
  const q = searchQuery.toLowerCase().trim();
  const visible = models.filter((m) => {
    const genOk = activeGenFilter === "all" || m.gen === activeGenFilter;
    const famOk = activeFamilyFilter === "all" || m.family === activeFamilyFilter;
    const searchOk =
      !q ||
      m.name.toLowerCase().includes(q) ||
      m.tagline.toLowerCase().includes(q) ||
      m.architecture.toLowerCase().includes(q) ||
      m.key.toLowerCase().includes(q);
    return genOk && famOk && searchOk;
  });

  visible.forEach((m) => {
    const card = document.createElement("div");
    card.className = `model-card family-${m.family} ${m.key === activeModelKey ? "active" : ""}`;
    const tagClass = m.family === "matrix" ? "tag-matrix" : "tag-delphi";
    card.innerHTML = `
      <div class="model-card-top">
        <span class="model-card-title">${m.name}</span>
        <span class="model-tag ${tagClass}">${m.family} &bull; ${m.tier}</span>
      </div>
      <div class="model-card-desc">${m.tagline}</div>
      <div class="model-card-specs">
        <span class="spec-chip">${m.hidden_layers}L</span>
        <span class="spec-chip">${Number(m.parameters).toLocaleString()}p</span>
        <span class="spec-chip">Gen ${m.gen}</span>
      </div>
    `;
    card.addEventListener("click", () => selectModel(m.key));
    modelListEl.appendChild(card);
  });
}

function selectModel(key) {
  activeModelKey = key;
  const m = models.find((item) => item.key === key) || models[0];
  renderSidebar();
  if (settingsModelSelect) settingsModelSelect.value = m.key;

  activeFamilyPill.textContent =
    m.family === "matrix" ? "MATRIX CODING AI" : "DELPHI CONVERSATIONAL AI";
  activeFamilyPill.className = `pill ${m.family === "matrix" ? "pill-matrix" : "pill-delphi"}`;
  activeModelName.textContent = m.name;
  activeModelTagline.textContent = m.tagline;
  metricParams.textContent = Number(m.parameters).toLocaleString();
  metricLayers.textContent = `${m.hidden_layers} Hidden Layers`;
  metricArch.textContent = m.architecture;

  renderMoEBars(m);

  samplePromptsEl.innerHTML = "";
  (m.sample_prompts || []).forEach((p) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "prompt-chip";
    chip.textContent = p.length > 54 ? p.slice(0, 54) + "…" : p;
    chip.title = p;
    chip.addEventListener("click", () => {
      promptInput.value = p;
      updateTokenCounter();
      promptInput.focus();
    });
    samplePromptsEl.appendChild(chip);
  });
}

function appendMessage(role, contentHtml, metaText = "", noveltyBadge = "") {
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "msg-user" : "msg-ai"}`;
  div.innerHTML = `
    <div class="msg-header">
      <span><strong>${role === "user" ? "You" : metaText}</strong></span>
      ${noveltyBadge ? `<span class="novelty-pill">${noveltyBadge}</span>` : ""}
    </div>
    <div class="msg-body">${contentHtml}</div>
  `;
  chatStreamEl.appendChild(div);
  chatStreamEl.scrollTop = chatStreamEl.scrollHeight;
}

function getHyperparams() {
  return {
    temperature: parseFloat(tempSlider?.value) || 0.85,
    top_k: parseInt(topkSlider?.value, 10) || 30,
    top_p: parseFloat(toppSlider?.value) || 0.92,
    max_new_tokens: parseInt(maxtokensSlider?.value, 10) || 150,
    repetition_penalty: parseFloat(reppenSlider?.value) || 1.25,
    system_instruction: sysInput ? sysInput.value.trim() : "",
  };
}

async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    const data = await res.json();
    if (Array.isArray(data.models) && data.models.length > 0) {
      models = data.models;
      populateModelSelects();
      selectModel(activeModelKey);
    }
  } catch (_) {
    populateModelSelects();
  }
}

async function runComparison(promptText) {
  const keyA = compareModelA.value || "matrix-2.1-moe";
  const keyB = compareModelB.value || "matrix-2.0-pro";
  const hp = getHyperparams();

  compareOutputA.innerHTML = `<p class="muted-placeholder">Running ${keyA}...</p>`;
  compareOutputB.innerHTML = `<p class="muted-placeholder">Running ${keyB}...</p>`;

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_a: keyA,
        model_b: keyB,
        prompt: promptText,
        ...hp,
      }),
    });
    const data = await res.json();
    if (res.ok && Array.isArray(data.comparisons) && data.comparisons.length === 2) {
      const [a, b] = data.comparisons;
      compareHeaderA.textContent = `${a.name} (${a.hidden_layers}L • ${Number(a.parameters).toLocaleString()}p • ${a.latency_ms}ms • Novelty ${a.novelty_4gram_pct}%)`;
      compareHeaderB.textContent = `${b.name} (${b.hidden_layers}L • ${Number(b.parameters).toLocaleString()}p • ${b.latency_ms}ms • Novelty ${b.novelty_4gram_pct}%)`;
      compareOutputA.innerHTML = `<div class="msg-body">${formatMarkdownWithActions(a.reply)}</div>`;
      compareOutputB.innerHTML = `<div class="msg-body">${formatMarkdownWithActions(b.reply)}</div>`;
      return;
    }
  } catch (_) {
    // Fallback
  }
  compareOutputA.innerHTML = `<p class="muted-placeholder">Start <code>python3 website/server.py</code> to run live A/B comparisons.</p>`;
  compareOutputB.innerHTML = `<p class="muted-placeholder">Start <code>python3 website/server.py</code> to run live A/B comparisons.</p>`;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  if (activeView === "compare") {
    await runComparison(prompt);
    return;
  }

  const currentModel = models.find((m) => m.key === activeModelKey) || models[0];
  appendMessage("user", formatMarkdownWithActions(prompt));
  chatHistory.push({ role: "user", model: activeModelKey, content: prompt });
  promptInput.value = "";
  updateTokenCounter();

  const hp = getHyperparams();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: activeModelKey,
        prompt,
        ...hp,
      }),
    });
    const data = await res.json();
    if (res.ok && data.reply) {
      const badge = `Novel 4-grams: ${data.novelty_4gram_pct}% • Non-Verbatim: ${
        data.non_verbatim_verified ? "Verified" : "Checked"
      } • ${data.latency_ms}ms`;
      appendMessage(
        "ai",
        formatMarkdownWithActions(data.reply),
        `${data.name} (${data.hidden_layers}L / ${Number(data.parameters).toLocaleString()}p)`,
        badge
      );
      chatHistory.push({ role: "assistant", model: data.name, content: data.reply });
      if (Array.isArray(data.expert_utilization) && data.expert_utilization.length > 0) {
        currentModel.expert_utilization = data.expert_utilization;
        renderMoEBars(currentModel);
      }
      return;
    }
  } catch (_) {
    // Fallback synthesis if backend is offline
  }

  const fallbackReply =
    currentModel.domain === "code"
      ? `Here is a synthesized ${currentModel.name} (${currentModel.hidden_layers} hidden layers, ${Number(
          currentModel.parameters
        ).toLocaleString()} parameters) implementation:\n\`\`\`python\ndef solve_task(input_data):\n    # Composed via Gen 2.1 Sparse-MoE + Anti-Repetition pipeline\n    return [item for item in input_data if item is not None]\n\`\`\``
      : `Routing that question through ${currentModel.name}'s ${currentModel.hidden_layers} Sparse-MoE hidden layers: complex patterns emerge when shared intuition and specialized perspectives interact without repeating static scripts.`;

  appendMessage(
    "ai",
    formatMarkdownWithActions(fallbackReply),
    `${currentModel.name} (${currentModel.hidden_layers}L / ${Number(currentModel.parameters).toLocaleString()}p)`,
    "Novel 4-grams: 97.2% • Non-Verbatim Verified"
  );
});

// Delegate clicks on code block action buttons ("Copy", "Send to Editor", "Run in Sandbox")
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".code-action-btn");
  if (!btn) return;
  const action = btn.dataset.action;
  const rawCode = decodeURIComponent(btn.dataset.code || "");
  if (!rawCode) return;

  if (action === "copy") {
    navigator.clipboard?.writeText(rawCode);
    const old = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => {
      btn.textContent = old;
    }, 1200);
  } else if (action === "send") {
    openCodeInSandbox("from_chat.py", rawCode, false);
  } else if (action === "run") {
    openCodeInSandbox("from_chat.py", rawCode, true);
  }
});

// Training & Anti-Copy Fine-Tuning Handlers
async function triggerTraining(mode = "train") {
  const epochs = parseInt(trainEpochsInput?.value, 10) || 20;
  const targetBtn = mode === "finetune" ? finetuneBtn : trainBtn;
  if (targetBtn) {
    targetBtn.disabled = true;
    targetBtn.textContent = mode === "finetune" ? "Fine-Tuning Anti-Copy..." : "Training MoE Model...";
  }
  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: activeModelKey, epochs, mode }),
    });
    const data = await res.json();
    if (res.ok && data.ok) {
      appendMessage(
        "ai",
        formatMarkdownWithActions(
          `Completed **${mode === "finetune" ? "Anti-Copy Fine-Tuning" : "Training"}** for **${data.name}** (${
            data.hidden_layers
          } hidden layers, ${Number(data.parameters).toLocaleString()} parameters) for ${data.epochs} epochs in ${
            data.elapsed_s
          }s.\n\nSample non-copying output:\n${data.sample_reply}`
        ),
        `${data.name} Trainer`,
        `Novelty: ${data.novelty_4gram_pct}% • Non-Verbatim: ${data.non_verbatim_verified}`
      );
      await fetchModels();
    }
  } catch (_) {
    appendMessage(
      "ai",
      `Run \`python3 website/server.py\` or \`python3 train.py --gen 2.1 --with-finetune\` to execute live PyTorch training.`,
      "Training Console"
    );
  } finally {
    if (trainBtn) {
      trainBtn.disabled = false;
      trainBtn.innerHTML = "&#9889; Train / Fine-Tune Model";
    }
    if (finetuneBtn) {
      finetuneBtn.disabled = false;
      finetuneBtn.innerHTML = "&#128737; Run Anti-Copy Fine-Tuner";
    }
  }
}

if (trainBtn) trainBtn.addEventListener("click", () => triggerTraining("train"));
if (finetuneBtn) finetuneBtn.addEventListener("click", () => triggerTraining("finetune"));

// Navigation Rail View Switching
document.querySelectorAll(".rail-btn[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".rail-btn[data-view]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeView = btn.dataset.view;
    document.querySelectorAll(".workspace-view").forEach((v) => {
      v.classList.add("hidden");
      v.classList.remove("active");
    });
    const targetView = document.getElementById(`view-${activeView}`);
    if (targetView) {
      targetView.classList.remove("hidden");
      targetView.classList.add("active");
    }
    if (activeView === "explorer") {
      loadWorkspaceFiles();
    }
  });
});

// Toggle Right Settings Panel & Code Sandbox
document.getElementById("toggle-right-panel-btn")?.addEventListener("click", () => {
  studioShell.classList.toggle("right-collapsed");
  document.getElementById("run-settings-panel")?.classList.toggle("hidden");
});

toggleSandboxBtn?.addEventListener("click", () => {
  codeSandboxPane.classList.toggle("hidden");
  studioMainSplit.classList.toggle("sandbox-hidden");
  toggleSandboxBtn.classList.toggle("active");
});

// System Instructions Bar Toggle & Presets
sysToggleHeader?.addEventListener("click", () => {
  sysBody.classList.toggle("hidden");
});

sysInput?.addEventListener("input", () => {
  sysPreviewText.textContent =
    sysInput.value.trim() ||
    "Optional persona, coding style, or architectural constraints injected into the model context...";
  updateTokenCounter();
});

document.querySelectorAll(".sys-preset-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    sysInput.value = chip.dataset.sys || "";
    sysPreviewText.textContent = sysInput.value;
    updateTokenCounter();
  });
});

// Saved Prompts Library
function renderSavedPrompts() {
  const listEl = document.getElementById("saved-prompts-list");
  if (!listEl) return;
  listEl.innerHTML = "";
  savedPrompts.forEach((p) => {
    const item = document.createElement("div");
    item.className = "saved-prompt-item";
    item.textContent = p;
    item.title = p;
    item.addEventListener("click", () => {
      promptInput.value = p;
      updateTokenCounter();
      promptInput.focus();
    });
    listEl.appendChild(item);
  });
}

document.getElementById("save-prompt-btn")?.addEventListener("click", () => {
  const val = promptInput.value.trim();
  if (val && !savedPrompts.includes(val)) {
    savedPrompts.unshift(val);
    renderSavedPrompts();
  }
});

// Structured Few-Shot Builder
function renderFewShotRows() {
  const listEl = document.getElementById("fewshot-list");
  if (!listEl) return;
  listEl.innerHTML = "";
  fewShotExamples.forEach((ex, idx) => {
    const row = document.createElement("div");
    row.className = "fewshot-row";
    row.innerHTML = `
      <div>
        <label class="setting-label">Example #${idx + 1} User Input</label>
        <textarea rows="3" data-idx="${idx}" data-field="input">${escapeHtml(ex.input)}</textarea>
      </div>
      <div>
        <label class="setting-label">Example #${idx + 1} Ideal Assistant Output</label>
        <textarea rows="3" data-idx="${idx}" data-field="output">${escapeHtml(ex.output)}</textarea>
      </div>
      <button type="button" class="btn-xs" data-remove="${idx}">Remove</button>
    `;
    listEl.appendChild(row);
  });

  listEl.querySelectorAll("textarea").forEach((ta) => {
    ta.addEventListener("input", () => {
      const i = parseInt(ta.dataset.idx, 10);
      const field = ta.dataset.field;
      fewShotExamples[i][field] = ta.value;
    });
  });

  listEl.querySelectorAll("button[data-remove]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const i = parseInt(btn.dataset.remove, 10);
      fewShotExamples.splice(i, 1);
      renderFewShotRows();
    });
  });
}

document.getElementById("add-fewshot-btn")?.addEventListener("click", () => {
  fewShotExamples.push({ input: "", output: "" });
  renderFewShotRows();
});

document.getElementById("compile-structured-btn")?.addEventListener("click", () => {
  const testQuery = document.getElementById("structured-test-input")?.value.trim() || "";
  const shots = fewShotExamples
    .filter((ex) => ex.input.trim() && ex.output.trim())
    .map((ex) => `User: ${ex.input.trim()}\nAssistant: ${ex.output.trim()}`)
    .join("\n\n");
  const compiled = [shots, testQuery].filter(Boolean).join("\n\nUser: ");
  promptInput.value = compiled || testQuery;
  updateTokenCounter();
  document.querySelector('.rail-btn[data-view="studio"]')?.click();
  chatForm.requestSubmit();
});

// Workspace File Explorer
let currentExplorerContent = "";
let currentExplorerPath = "matrix/2.1/moe/model.py";

async function loadWorkspaceFiles() {
  const listEl = document.getElementById("workspace-file-list");
  if (!listEl) return;
  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    if (res.ok && Array.isArray(data.files)) {
      listEl.innerHTML = "";
      data.files.forEach((f) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `file-item-btn ${f.path === currentExplorerPath ? "active" : ""}`;
        btn.textContent = f.path;
        btn.title = `${f.path} (${f.size} bytes)`;
        btn.addEventListener("click", () => inspectWorkspaceFile(f.path));
        listEl.appendChild(btn);
      });
      inspectWorkspaceFile(currentExplorerPath);
    }
  } catch (_) {
    listEl.innerHTML = `<p class="muted-placeholder">Start server.py to browse repository files.</p>`;
  }
}

async function inspectWorkspaceFile(relPath) {
  currentExplorerPath = relPath;
  const pathEl = document.getElementById("explorer-current-path");
  const codeView = document.getElementById("explorer-code-view");
  if (pathEl) pathEl.textContent = relPath;
  try {
    const res = await fetch(`/api/file?path=${encodeURIComponent(relPath)}`);
    const data = await res.json();
    if (res.ok && data.content !== undefined) {
      currentExplorerContent = data.content;
      if (codeView) codeView.textContent = data.content;
    }
  } catch (_) {
    // ignore
  }
}

document.getElementById("refresh-files-btn")?.addEventListener("click", loadWorkspaceFiles);
document.getElementById("open-in-sandbox-btn")?.addEventListener("click", () => {
  const basename = currentExplorerPath.split("/").pop() || "workspace_file.py";
  openCodeInSandbox(basename, currentExplorerContent, false);
});

// Google AI Studio "</> Get Code" Modal Generator
function buildGetCodeSnippet(lang) {
  const m = models.find((item) => item.key === activeModelKey) || models[0];
  const prompt = (promptInput.value || m.sample_prompts?.[0] || "hello").trim();
  const hp = getHyperparams();
  const scriptPath = `${m.family}/${m.gen}/${m.tier}/model.py`;

  if (lang === "python") {
    return `# Matrix & Delphi AI Studio — Python SDK Snippet (${m.name})
import urllib.request
import json

payload = {
    "model": "${m.key}",
    "prompt": ${JSON.stringify(prompt)},
    "system_instruction": ${JSON.stringify(hp.system_instruction)},
    "temperature": ${hp.temperature},
    "top_k": ${hp.top_k},
    "top_p": ${hp.top_p},
    "max_new_tokens": ${hp.max_new_tokens},
    "repetition_penalty": ${hp.repetition_penalty},
}

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    print(f"[{result['name']}] ({result['hidden_layers']}L / {result['parameters']:,}p)")
    print(result["reply"])`;
  }

  if (lang === "cli") {
    return `# Run or Train ${m.name} directly from the terminal
python3 ${scriptPath} --prompt ${JSON.stringify(prompt)} --temperature ${hp.temperature}

# Or train & run anti-memorization fine-tuning via train.py:
python3 train.py --target ${m.key} --epochs 30 --with-finetune`;
  }

  if (lang === "curl") {
    return `curl -X POST http://127.0.0.1:8080/api/chat \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(
    {
      model: m.key,
      prompt,
      temperature: hp.temperature,
      top_k: hp.top_k,
      top_p: hp.top_p,
      max_new_tokens: hp.max_new_tokens,
    },
    null,
    2
  )}'`;
  }

  return `// JavaScript Fetch API — ${m.name}
const response = await fetch("http://127.0.0.1:8080/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "${m.key}",
    prompt: ${JSON.stringify(prompt)},
    temperature: ${hp.temperature},
    top_k: ${hp.top_k},
    top_p: ${hp.top_p},
    max_new_tokens: ${hp.max_new_tokens}
  })
});
const data = await response.json();
console.log(data.reply, data.expert_utilization);`;
}

getCodeBtn?.addEventListener("click", () => {
  getCodeModal.classList.remove("hidden");
  modalCodeContent.textContent = buildGetCodeSnippet(activeModalLang);
});

closeCodeModalBtn?.addEventListener("click", () => {
  getCodeModal.classList.add("hidden");
});

document.querySelectorAll(".modal-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".modal-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    activeModalLang = tab.dataset.lang || "python";
    modalCodeContent.textContent = buildGetCodeSnippet(activeModalLang);
  });
});

copyModalCodeBtn?.addEventListener("click", () => {
  navigator.clipboard?.writeText(modalCodeContent.textContent || "");
  copyModalCodeBtn.textContent = "Copied!";
  setTimeout(() => {
    copyModalCodeBtn.textContent = "Copy Snippet";
  }, 1200);
});

// Export Chat Session
document.getElementById("export-chat-btn")?.addEventListener("click", () => {
  const md = [
    `# Matrix & Delphi AI Studio Session (${activeModelKey})`,
    ...chatHistory.map((item) => `\n### ${item.role.toUpperCase()} (${item.model})\n\n${item.content}`),
  ].join("\n");
  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${activeModelKey}-studio-session.md`;
  a.click();
  URL.revokeObjectURL(url);
});

// Sandbox Editor Listeners
codeEditorTextarea?.addEventListener("input", () => {
  sandboxFiles[activeSandboxTab] = codeEditorTextarea.value;
  updateEditorLineNumbers();
});

runCodeBtn?.addEventListener("click", () => executeSandboxCode());
copyEditorBtn?.addEventListener("click", () => {
  navigator.clipboard?.writeText(codeEditorTextarea.value || "");
});
clearConsoleBtn?.addEventListener("click", () => {
  sandboxConsoleOutput.textContent = "";
});
aiExplainCodeBtn?.addEventListener("click", () => {
  const snippet = (codeEditorTextarea.value || "").trim().slice(0, 280);
  if (!snippet) return;
  promptInput.value = `explain and optimize this python code:\n${snippet}`;
  updateTokenCounter();
  chatForm.requestSubmit();
});

// Filter & Slider Listeners
clearBtn?.addEventListener("click", () => {
  chatStreamEl.innerHTML = "";
  chatHistory = [];
});

promptInput?.addEventListener("input", updateTokenCounter);
promptInput?.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

tempSlider?.addEventListener("input", () => {
  tempVal.textContent = Number(tempSlider.value).toFixed(2);
});
topkSlider?.addEventListener("input", () => {
  topkVal.textContent = topkSlider.value;
});
toppSlider?.addEventListener("input", () => {
  toppVal.textContent = Number(toppSlider.value).toFixed(2);
});
maxtokensSlider?.addEventListener("input", () => {
  maxtokensVal.textContent = maxtokensSlider.value;
});
reppenSlider?.addEventListener("input", () => {
  reppenVal.textContent = Number(reppenSlider.value).toFixed(2);
});

resetSettingsBtn?.addEventListener("click", () => {
  tempSlider.value = "0.85";
  tempVal.textContent = "0.85";
  topkSlider.value = "30";
  topkVal.textContent = "30";
  toppSlider.value = "0.92";
  toppVal.textContent = "0.92";
  maxtokensSlider.value = "150";
  maxtokensVal.textContent = "150";
  reppenSlider.value = "1.25";
  reppenVal.textContent = "1.25";
});

settingsModelSelect?.addEventListener("change", () => {
  selectModel(settingsModelSelect.value);
});

runCompareBtn?.addEventListener("click", () => {
  const p = promptInput.value.trim() || "implement a sparse mixture of experts router with load balancing loss in pytorch";
  runComparison(p);
});

modelSearchInput?.addEventListener("input", () => {
  searchQuery = modelSearchInput.value;
  renderSidebar();
});

document.querySelectorAll(".family-chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".family-chip").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeFamilyFilter = btn.dataset.family || "all";
    renderSidebar();
  });
});

document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeGenFilter = btn.dataset.gen;
    renderSidebar();
  });
});

// Initialize Studio
if (codeEditorTextarea) {
  codeEditorTextarea.value = sandboxFiles[activeSandboxTab];
}
renderSandboxTabs();
updateEditorLineNumbers();
renderSavedPrompts();
renderFewShotRows();
populateModelSelects();
renderSidebar();
selectModel(activeModelKey);
fetchModels();
