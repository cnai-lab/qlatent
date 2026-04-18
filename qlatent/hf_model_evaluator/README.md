# QPsychometric Evaluation Pipeline

A large-scale pipeline for evaluating psychometric properties of pre-trained language models from HuggingFace. The pipeline applies standardized psychometric questionnaires (e.g., PHQ-9, GAD-7, BIG5, ASI, SOC, CS) to NLI (zero-shot-classification) and MLM (fill-mask) models, logging per-model responses and accuracy metrics.

---

## Requirements

- Python 3.9
- Linux environment
- CUDA-capable GPU (recommended; falls back to CPU if unavailable)
- HuggingFace account (optional but recommended to avoid API rate limits)

---

## Installation

```bash
conda env create -f env_qpsychometric_pipeline.yml
conda activate qpsychometric_pipeline
```

---

## HuggingFace Cache

By default, models and datasets are cached to `~/.cache/huggingface`. To override this, set the following environment variables before running:

```bash
export HF_HOME=/path/to/your/cache
export TRANSFORMERS_CACHE=/path/to/your/cache
export HF_DATASETS_CACHE=/path/to/your/cache
```

---

## Usage

All scripts must be run from the root of this directory.

### Evaluate a single NLI model

Runs one `zero-shot-classification` or `text-classification` model against all QMNLI questionnaires.

```bash
python single_model_qmnli.py --model_id <HF_MODEL_ID> --base_dir ./model_logs
```

**Example:**
```bash
python single_model_qmnli.py --model_id typeform/distilbert-base-uncased-mnli --base_dir ./model_logs
```

---

### Evaluate a single MLM model

Runs one `fill-mask` model against all QMLM and QMNLI questionnaires.

```bash
python single_model_qmlm.py --model_id <HF_MODEL_ID> --base_dir ./model_logs
```

---

### Evaluate all NLI models (per-model controller — recommended for large runs)

Iterates over all HuggingFace NLI models, spawning an isolated subprocess per model. Completed and errored models are skipped on resume.

```bash
python per_model_controller_nli.py --base_dir ./model_logs [--hf_token HF_TOKEN] [--only_zero_shot]
```

| Argument | Description | Default |
|---|---|---|
| `--base_dir` | Output directory | `./model_logs` |
| `--hf_token` | HuggingFace API token | None |
| `--chunk_file` | Path to a `.txt` file with one model ID per line (for partial runs) | None |
| `--only_zero_shot` | Restrict to `zero-shot-classification` models only | False |

---

### Evaluate all MLM models (per-model controller)

```bash
python per_model_controller_mlm.py --base_dir ./model_logs [--hf_token HF_TOKEN]
```

| Argument | Description | Default |
|---|---|---|
| `--base_dir` | Output directory | `./model_logs` |
| `--hf_token` | HuggingFace API token | None |
| `--chunk_file` | Path to a `.txt` file with one model ID per line | None |

---

## Output Structure

Results are written to `--base_dir` (default: `./model_logs`):

```
model_logs/
├── QMNLI/
│   ├── ASI.csv          # Per-question responses for each model
│   ├── BIG5.csv
│   ├── CS.csv
│   ├── GAD7.csv
│   ├── PHQ9.csv
│   └── SOC.csv
├── QMLM/
│   └── ...              # Same structure for fill-mask models
├── models_meta_data.csv # Model metadata (architecture, params, vocab size, etc.)
├── models_errors.csv    # Models that failed or were skipped
├── models_acc.csv       # MNLI accuracy scores for NLI models
└── current_model.csv    # Progress tracker (last processed model)
```

---

## Resuming Interrupted Runs

The pipeline is resume-safe. On restart, `per_model_controller_nli.py` reads `QMNLI/ASI.csv` and `models_errors.csv` to determine which models have already been processed and skips them automatically.

---

## Project Structure

```
.
├── pipeline/
│   ├── pipeline_executor.py     # Core evaluation loop
│   └── questionnaire.py         # Questionnaire preparation utilities
├── utils/
│   ├── file_utils.py            # CSV logging and file management
│   ├── logging_utils.py         # Logging configuration
│   ├── model_utils.py           # Model loading and GPU management
│   └── model_acc_evaluator.py   # MNLI accuracy evaluation
├── global_variables.py          # Shared constants
├── per_model_controller_nli.py  # NLI orchestrator (subprocess-per-model)
├── per_model_controller_mlm.py  # MLM orchestrator (subprocess-per-model)
├── single_model_qmnli.py        # Single NLI model entry point
├── single_model_qmlm.py         # Single MLM model entry point
└── env_qpsychometric_pipeline.yml
```
