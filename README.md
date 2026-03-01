# Vulnerability Detection with CodeBERT 🛡️

Binary classification system that detects security vulnerabilities in C/C++ code snippets using a fine-tuned [CodeBERT](https://huggingface.co/microsoft/codebert-base) model. Includes high-transparency explainability (SHAP & LIME) and a FastAPI inference API.

---

## 🚀 Quick Start

### 1. Requirements
Ensure you have [uv](https://github.com/astral-sh/uv) installed (the ultra-fast Python package manager).

### 2. Setup
Run this once to provision the environment and install dependencies:
```bash
make setup
```

### 3. Training
To train on a small sample (default 1,000):
```bash
make train
```
To train on a specific number of samples (e.g., 50k):
```bash
make train MAX_SAMPLES=50000
```

### 4. Evaluation
Run metrics on the test set:
```bash
make evaluate
```

### 5. Run the API
Start the FastAPI inference service with auto-reload:
```bash
make dev
```

### 6. Test the API
Use `curl` to send a C++ snippet for analysis.

**Option A: LIME Explanation (Fastest)**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "code": "void foo(char *input) { char buffer[10]; strcpy(buffer, input); }",
    "include_explanation": true,
    "explainer": "lime"
  }'
```

**Option B: SHAP Explanation (Most Accurate)**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "code": "void foo(char *input) { char buffer[10]; strcpy(buffer, input); }",
    "include_explanation": true,
    "explainer": "shap"
  }'
```

**Health Check**
```bash
curl http://localhost:8000/health
```

---

## 📂 Project Structure

```text
vuln_detector/
├── src/
│   ├── api/               # FastAPI endpoints & schemas
│   ├── data/              # Data loading & preprocessing
│   ├── explainability/    # SHAP, LIME, & Visualizers
│   ├── model/             # Training & Inference logic
│   └── utils/             # Logging & Device selection
├── configs/               # Global configuration
├── model/                 # Fine-tuned model directory
├── reports/               # Metrics & Model Card
├── pyproject.toml         # Dependency management (uv)
├── Makefile               # Task automation
└── README.md
```

---

## 💻 Hardware & Performance

- **Cross-Platform Support**: Automatically detects and uses the best available hardware:
    - **NVIDIA GPU**: Uses **CUDA** (Windows / Linux).
    - **Apple Silicon**: Uses **MPS** (macOS).
    - **Fallback**: Uses **CPU** if no compatible GPU is detected.
- **Explainability**: SHAP/LIME runs are optimized for transparency without blocking inference.

## 📊 Dataset

Built using the [DiverseVul](https://github.com/wagner-group/diversevul) dataset, containing over 330,000 C/C++ functions with verified vulnerability labels.

---

## 📜 Compliance

This project includes a [Model Card](reports/model_card.md) documenting compliance with **EU AI Act** transparency obligations (Articles 11 & 14).
