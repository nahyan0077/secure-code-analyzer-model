# Model Card — Vulnerability Detection with CodeBERT

## Model Details

| Field | Value |
|---|---|
| **Base Model** | [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) |
| **Task** | Binary sequence classification (vulnerable / safe) |
| **Language** | C / C++ source code |
| **Fine-tuning Dataset** | [DiverseVul](https://github.com/wagner-group/diversevul) (2023-07-02 snapshot) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Hardware** | Apple Silicon (MPS backend) |

## Divergence from Reference Paper

This implementation is based on the methodology of [Haurogné et al. (2024)](https://doi.org/10.1016/j.mlwa.2024.100598) — *"Vulnerability detection using BERT based LLM model with transparency obligation practice towards trustworthy AI"*.

Key differences from the paper:

| Aspect | Paper | This Project | Justification |
|---|---|---|---|
| **Base model** | `bert-base-uncased` | `microsoft/codebert-base` | CodeBERT is pre-trained on 6 programming languages (incl. C/C++) and natural language, making it semantically stronger for code understanding than generic BERT. |
| **Class imbalance** | Undersampling (18,945 vs 18,945) | Weighted CrossEntropyLoss | Weighted loss preserves all training data while penalizing errors on the minority class proportionally. Undersampling discards ~94% of safe examples. |
| **Epochs** | 10 | 5 (with early stopping, patience=2) | Early stopping halts training when validation F1 plateaus, preventing overfitting without manual epoch selection. |

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 5 (with early stopping, patience=2) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max token length | 512 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Gradient clipping | max_grad_norm=1.0 |
| Optimizer | AdamW (HuggingFace default) |
| Best model selection | eval_f1 (greater is better) |
| Class weighting | Inverse-frequency weighted CrossEntropyLoss |

## Dataset Summary

- **Source**: DiverseVul — derived from CVEFixes, covering CVEs up to August 2022.
- **Total records**: ~330,000 (18,945 vulnerable + 311,547 safe)
- **Vulnerable (target=1)**: ~18,945 (5.7%), spanning 155 CWEs
- **Safe (target=0)**: ~311,547 (94.3%), from 7,514 commits
- **Split**: 80% train / 10% val / 10% test (stratified by target label)
- **Subsampling**: When `MAX_SAMPLES` is set, CWE-aware stratified subsampling preserves the CWE distribution.

### Data Collection Methodology
The DiverseVul dataset was collected by crawling security issue websites and extracting both vulnerability-fixing commits and corresponding source code from various open-source projects (Chen et al., 2023). Each entry contains: function snippet, target label, CWE identifier, project name, and commit ID.

## Class Imbalance Strategy

The DiverseVul dataset is heavily imbalanced (~5.7% vulnerable). The reference paper addressed this by undersampling the majority class to 18,945 vs 18,945.

This project uses **inverse-frequency weighted CrossEntropyLoss** instead, which:
- Preserves all training data (no information discarded)
- Assigns higher loss weight to minority class errors proportional to `total / (num_classes × count)`
- Is generally preferred in modern deep learning for imbalanced classification

## Context Window & Truncation

BERT-based models have a hard limit of **512 tokens**. This project uses the full 512-token window.

**Known limitation** (acknowledged by the paper): *"The model accepts up to 512 tokens as input, but a considerable number of code snippets exceeded this maximum size. Therefore, if the vulnerability were located after the first 512 tokens, it was not provided to the model."*

- **Truncation strategy**: Right-truncation (tokens beyond position 512 are discarded)
- **No chunking**: Long functions are not split into overlapping windows
- **Impact**: Vulnerabilities located deep within long functions may be invisible to the model

## Intended Use

Assist security researchers and developers in identifying potentially vulnerable code patterns in C/C++ functions. This model is a **screening tool** — findings should be reviewed by a human security expert.

## Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 86.42% |
| Precision | 15.98% |
| Recall | 32.14% |
| F1-Score | 21.34% |
| Total Samples | 33,050 |

### Confusion Matrix

|  | Predicted Safe | Predicted Vulnerable |
|---|---|---|
| **Actually Safe** | 27,952 (TN) | 3,203 (FP) |
| **Actually Vulnerable** | 1,286 (FN) | 609 (TP) |

> **Note**: These metrics reflect an earlier model trained with MAX_LENGTH=256 on a 50K subset. Re-evaluation with MAX_LENGTH=512 and full dataset is recommended.

## Limitations

- Trained only on C/C++ code; not suitable for other languages.
- Truncates code to 512 tokens — long functions may lose context beyond this window.
- Does not provide line-level or patch-level localisation of the vulnerability.
- Should not be used as the sole decision maker in production security pipelines.

## Bias Discussion

- **CWE coverage bias**: The DiverseVul dataset spans 155 CWEs, but the distribution is highly skewed. Common CWEs (e.g., CWE-119 buffer overflow, CWE-20 input validation) are overrepresented, while rare CWEs may have insufficient training examples.
- **Token sensitivity**: As noted in the reference paper, BERT-based models can become "overly sensitive to the meanings of words, particularly those used as function names or variable names." A function named `vulnerable()` may be flagged simply due to its name, not its logic.
- **Project distribution**: The model's performance may vary across different coding styles and project types. Functions from projects not represented in the training data may yield lower accuracy.

## Known Failure Cases

1. **Variable name bias**: Functions with security-related variable names (e.g., `is_vulnerable`, `unsafe_ptr`) may be flagged regardless of actual safety.
2. **Long functions**: Vulnerabilities past token 512 are invisible to the model.
3. **Obfuscated code**: Minified or heavily macro-expanded code may confuse the tokenizer.
4. **Safe patterns with dangerous function names**: Safe usage of `strcpy` with proper bounds checking may still trigger false positives.

## Generalization Risk

The reference paper cites Chen et al. (2023): *"the performance of all models on unseen projects decreases significantly to only 9.4%"*. This project uses random stratified splitting (not project-aware), which means train and test sets may contain functions from the same project. Real-world performance on entirely new codebases is expected to be significantly lower than reported metrics.

## EU AI Act Transparency Compliance

This model is documented in accordance with transparency obligations for AI systems (Title IV of the EU AI Act), following the three-dimension framework from Haurogné et al. (2024).

### Transparency Dimensions
1. **Data Transparency**: Dataset origin (DiverseVul/CVEFixes), size, CWE distribution, collection methodology, and known biases are documented above.
2. **Model Outcome Transparency**: All evaluation metrics, confusion matrix, and generalization risks are documented above.
3. **Explainability**: SHAP (local token attribution) and LIME (local feature importance) are integrated into the prediction API, with noise-filtered heatmap visualization.

### AI Risk Categorization
- **Category**: General Purpose AI (GPAI) for security analysis.
- **Criticality**: Low-to-Medium (Screening tool). High-risk if used autonomously for critical infrastructure access control.

### Technical Documentation (Article 11)
- **Architecture**: CodeBERT (RoBERTa-base architecture), 125M parameters, 12 layers, 768 hidden size, 12 attention heads.
- **Optimization**: Cross-Entropy with inverse-frequency class weights to mitigate majority-class bias.
- **Explainability Methods**: SHAP (Kernel SHAP, game-theoretic attribution) and LIME (local linear approximation).
- **Compute**: Multi-platform support for NVIDIA CUDA, Apple MPS, and CPU fallback.

### Human Oversight (Article 14)
- **Requirement**: This model's output MUST be reviewed by a human security analyst.
- **Interpretation**: "Vulnerable" flags should be treated as warnings, not final verdicts. Analysts should use the token-level heatmaps to verify the logic.

## Ethical Considerations

This model should augment, not replace, human code review. False negatives (missed vulnerabilities) are possible and could lead to security risks if the model is relied upon exclusively. The human-in-the-loop approach is essential for verifying predictions, especially given the model's known sensitivity to token patterns.
