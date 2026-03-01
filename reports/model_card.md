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

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max token length | 512 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Class weighting | Inverse-frequency weighted CrossEntropyLoss |

## Dataset Summary

- **Total records**: ~330,000
- **Vulnerable (target=1)**: ~18,945 (5.7%)
- **Safe (target=0)**: ~311,547 (94.3%)
- **Split**: 80% train / 10% val / 10% test (stratified)

## Intended Use

Assist security researchers and developers in identifying potentially vulnerable code patterns in C/C++ functions. This model is a **screening tool** — findings should be reviewed by a human security expert.

## Limitations

- Trained only on C/C++ code; not suitable for other languages.
- Truncates code to 512 tokens — long functions may lose context.
- Imbalanced dataset may bias toward the majority (safe) class despite class weighting.
- Does not provide line-level or patch-level localisation of the vulnerability.
- Should not be used as the sole decision maker in production security pipelines.

## Evaluation Metrics

See `reports/metrics.json` after running evaluation.

## EU AI Act Transparency Compliance

This model is documented in accordance with transparency obligations for AI systems (Title IV of the EU AI Act).

### AI Risk Categorization
- **Category**: General Purpose AI (GPAI) for security analysis.
- **Criticality**: Low-to-Medium (Screening tool). High-risk if used autonomously for critical infrastructure access control.

### Technical Documentation (Article 11)
- **Architecture**: Transformer-based encoder (RoBERTa-base).
- **Optimization**: Cross-Entropy with inverse-frequency class weights to mitigate majority-class bias.
- **Explainability Methods**: SHAP (Game theoretic attribution) and LIME (Local linear approximation).
- **Compute Optimization**: Multi-platform support for NVIDIA CUDA, Apple MPS, and CPU fallback.

### Human Oversight (Article 14)
- **Requirement**: This model's output MUST be reviewed by a human security analyst.
- **Interpretation**: "Vulnerable" flags should be treated as warnings, not final verdicts. Analysts should use the token-level heatmaps to verify the logic.

## Ethical Considerations

This model should augment, not replace, human code review. False negatives (missed vulnerabilities) are possible and could lead to security risks if the model is relied upon exclusively.
