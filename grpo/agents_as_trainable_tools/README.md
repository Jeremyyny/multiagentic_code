# Agents-as-Trainable-Tools: Zero-to-End-to-End Training and Evaluation Guide

This guide is written for users with no prior context.  
If you follow it step by step, you can:

1. Prepare data and splits
2. Build tool SFT data (weak or GPT-generated)
3. Train reasoning/context tool models
4. Train the manager model with GRPO
5. Optionally run self-evolution stages
6. Compare your pipeline against baseline models with a leaderboard

All commands below are shown for Windows PowerShell.

---

## 1) What this project does

This project trains a manager model that can optionally call two tools during reasoning:

- `reasoning_tool`
- `context_tool`

The manager is trained with GRPO.  
The tools are trained with SFT.  
Then you evaluate whether the full pipeline outperforms no-tool and direct-answer baselines.

Main scripts:

- Training pipeline: `agents_as_trainable_tools/agents_as_tools.py`
- Comparative evaluation: `agents_as_trainable_tools/evaluate_pipeline_vs_baselines.py`

---

## 2) Repository paths you need

From your current repo root:

- Repo root: `c:\Users\yyn07\Desktop\multi_agent_test\New_GRPO_code\trl_vllm`
- Training script: `agents_as_trainable_tools/agents_as_tools.py`
- Evaluation script: `agents_as_trainable_tools/evaluate_pipeline_vs_baselines.py`
- PubMedQA data folder: `agents_as_trainable_tools/Pubmedqa`
- MedQA data folder: `agents_as_trainable_tools/MedQA`

---

## 3) Environment setup

Open PowerShell in repo root and run:

```powershell
cd c:\Users\yyn07\Desktop\multi_agent_test\New_GRPO_code\trl_vllm

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

Recommended extras:

```powershell
pip install peft wandb requests
```

Sanity check:

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py --help
.venv\Scripts\python.exe agents_as_trainable_tools\evaluate_pipeline_vs_baselines.py --help
```

---

## 4) Data loading behavior (important)

`agents_as_tools.py` now supports:

- File path
- Directory path
- Alias name

### 4.1 `--data_path` aliases

- `--data_path pubmedqa` -> resolves to `agents_as_trainable_tools/Pubmedqa`
- `--data_path medqa` -> resolves to `agents_as_trainable_tools/MedQA`
- Empty `--data_path` -> uses task default by `--task_name`

### 4.2 Task defaults

- `task_name=pubmedqa`: labels default to `yes,no,maybe`
- `task_name=medqa`: labels default to `A,B,C,D,E`

### 4.3 MedQA invalid rows

MedQA source may contain rows with missing/invalid labels.  
The loader automatically drops invalid rows and prints a summary:

- `missing_question`
- `bad_label`
- `kept`

This is expected and not a failure.

---

## 5) Full training pipeline stages

`agents_as_tools.py` stages:

1. `make_splits`
2. `build_tool_sft`
3. `train_tool_reasoning`
4. `train_tool_context`
5. `train_manager_grpo`
6. `evolve_build_manager_sft` (optional)
7. `train_manager_sft` (optional)
8. `evolve_round` (optional one-command evolution round)

---

## 6) Recommended runbook: PubMedQA with 500 samples

This is the simplest practical workflow for your use case.

### Stage A: Build a 500-sample split

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage make_splits `
  --task_name pubmedqa `
  --data_path pubmedqa `
  --max_samples 500 `
  --test_size 100 `
  --dev_size 100 `
  --split_path agents_as_trainable_tools\splits_pubmedqa_500.json `
  --seed 42
```

Expected split sizes:

- train: 300
- dev: 100
- test: 100

### Stage B: Build tool SFT data (weak mode)

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage build_tool_sft `
  --task_name pubmedqa `
  --data_path pubmedqa `
  --split_path agents_as_trainable_tools\splits_pubmedqa_500.json `
  --tool_sft_out_dir agents_as_trainable_tools\tool_sft_pubmedqa_500 `
  --top_k 20 `
  --tool_variants_train 3 `
  --tool_variants_dev 2 `
  --tool_synth_mode weak `
  --seed 42
```

Outputs:

- `tool_reasoning_train.jsonl`
- `tool_reasoning_dev.jsonl`
- `tool_context_train.jsonl`
- `tool_context_dev.jsonl`

all under `agents_as_trainable_tools/tool_sft_pubmedqa_500`.

### Stage C: Train reasoning tool (SFT)

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage train_tool_reasoning `
  --task_name pubmedqa `
  --base_model Qwen/Qwen3-0.6B `
  --tool_sft_out_dir agents_as_trainable_tools\tool_sft_pubmedqa_500 `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --tool_use_lora `
  --tool_max_seq_len 4096 `
  --tool_lr 2e-4 `
  --tool_epochs 2 `
  --tool_bs 1 `
  --tool_grad_accum 8 `
  --seed 42
```

### Stage D: Train context tool (SFT)

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage train_tool_context `
  --task_name pubmedqa `
  --base_model Qwen/Qwen3-0.6B `
  --tool_sft_out_dir agents_as_trainable_tools\tool_sft_pubmedqa_500 `
  --context_tool_out context_lora_mvp_split `
  --tool_use_lora `
  --tool_max_seq_len 4096 `
  --tool_lr 2e-4 `
  --tool_epochs 2 `
  --tool_bs 1 `
  --tool_grad_accum 8 `
  --seed 42
```

### Stage E: Train manager with GRPO

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage train_manager_grpo `
  --task_name pubmedqa `
  --base_model Qwen/Qwen3-0.6B `
  --data_path pubmedqa `
  --split_path agents_as_trainable_tools\splits_pubmedqa_500.json `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --context_tool_out context_lora_mvp_split `
  --manager_out manager_grpo_mvp_split `
  --mgr_bs 4 `
  --mgr_max_completion_length 4096 `
  --mgr_temperature 0.9 `
  --mgr_num_generations 6 `
  --grpo_beta 0.01 `
  --fail_buffer_jsonl manager_grpo_mvp_split\fail_buffer.jsonl `
  --raw_trace_jsonl manager_grpo_mvp_split\train_raw_trace.jsonl `
  --seed 42
```

Main outputs:

- Manager checkpoint directory: `manager_grpo_mvp_split`
- Failure buffer: `manager_grpo_mvp_split/fail_buffer.jsonl`
- Raw trace: `manager_grpo_mvp_split/train_raw_trace.jsonl`

---

## 7) GPT-generated tool SFT data (optional)

If you want GPT teacher synthesis, set:

```powershell
$env:TEACHER_BASE_URL="https://api.openai.com"
$env:TEACHER_API_KEY="YOUR_API_KEY"
$env:TEACHER_MODEL="gpt-4o-mini"
$env:TEACHER_TIMEOUT="60"
```

Then run `build_tool_sft` with:

- `--tool_synth_mode gpt`

If required teacher env vars are missing, the script fails fast with a clear error.

---

## 8) Optional self-evolution stages

### 8.1 Build manager SFT data from failures

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage evolve_build_manager_sft `
  --task_name pubmedqa `
  --base_model Qwen/Qwen3-0.6B `
  --data_path pubmedqa `
  --split_path agents_as_trainable_tools\splits_pubmedqa_500.json `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --context_tool_out context_lora_mvp_split `
  --manager_out manager_grpo_mvp_split `
  --evolve_out_dir evolve_manager_sft `
  --max_fail_samples 2000 `
  --planning_mode realistic `
  --seed 42
```

Output:

- `evolve_manager_sft/manager_sft_from_failures.jsonl`

### 8.2 Train manager SFT on evolved data

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage train_manager_sft `
  --task_name pubmedqa `
  --base_model Qwen/Qwen3-0.6B `
  --evolve_out_dir evolve_manager_sft `
  --manager_sft_out manager_sft_evolved `
  --manager_sft_lr 2e-5 `
  --manager_sft_epochs 1 `
  --manager_sft_max_seq_len 4096 `
  --manager_sft_bs 1 `
  --manager_sft_grad_accum 8 `
  --manager_sft_use_lora `
  --seed 42
```

### 8.3 One-command evolution round

`--stage evolve_round` will run:

1. GRPO manager training
2. Build evolved SFT from failures
3. Manager SFT training

---

## 9) Comparative evaluation: pipeline vs baselines

Use:

- `agents_as_trainable_tools/evaluate_pipeline_vs_baselines.py`

It supports:

- `pipeline_tools` (manager + tools)
- `pipeline_no_tools` (same manager, tools disabled)
- `direct_model` baselines (comma-separated model directories)
- `random_baseline`
- `majority_baseline`

### 9.1 Example command

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\evaluate_pipeline_vs_baselines.py `
  --task_name pubmedqa `
  --data_path pubmedqa `
  --split_path agents_as_trainable_tools\splits_pubmedqa_500.json `
  --split_key test_ids `
  --max_eval_samples 100 `
  --pipeline_manager_dir manager_grpo_mvp_split `
  --pipeline_base_model_for_tools Qwen/Qwen3-0.6B `
  --pipeline_reasoning_adapter reasoning_lora_mvp_split `
  --pipeline_context_adapter context_lora_mvp_split `
  --add_pipeline_no_tools_baseline `
  --baseline_model_dirs "Qwen/Qwen3-0.6B" `
  --baseline_model_names "qwen3_base_direct" `
  --add_random_baseline `
  --add_majority_baseline `
  --temperature 0.0 `
  --max_new_tokens 1024 `
  --max_tool_calls 2 `
  --out_dir agents_as_trainable_tools\eval_pubmedqa_compare
```

### 9.2 Evaluation outputs

Under `--out_dir`:

- `leaderboard.csv`: sorted system ranking
- `summary.json`: full metrics + system config
- `predictions.jsonl`: per-example predictions

---

## 10) Metrics explanation

Key fields in `leaderboard.csv`:

- `accuracy`: overall exact match rate
- `macro_f1`: average F1 across labels (treats labels equally)
- `weighted_f1`: F1 weighted by class support
- `invalid_rate`: fraction of outputs not parsed as valid `ANSWER_*`
- `avg_tool_calls`: mean number of tool calls per sample
- `elapsed_sec`: runtime for that system

Recommended comparison order:

1. `pipeline_tools` vs `pipeline_no_tools` on `accuracy` and `macro_f1`
2. Check `invalid_rate` (format robustness)
3. Check `avg_tool_calls` (cost/efficiency)

---

## 11) Running with MedQA

Use `--task_name medqa` and `--data_path medqa`.

Example split creation:

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py `
  --stage make_splits `
  --task_name medqa `
  --data_path medqa `
  --max_samples 500 `
  --test_size 100 `
  --dev_size 100 `
  --split_path agents_as_trainable_tools\splits_medqa_500.json `
  --seed 42
```

Notes:

- Default labels are `A,B,C,D,E`
- Loader may print dropped invalid label rows
- This is expected for some MedQA source shards

---

## 12) Common errors and fixes

### `ModuleNotFoundError: datasets`

```powershell
pip install datasets
```

### `peft not available but adapter_path is set`

```powershell
pip install peft
```

### GPT synthesis error about teacher config

Set at least:

- `TEACHER_BASE_URL`
- `TEACHER_MODEL`

and usually:

- `TEACHER_API_KEY`

### `No rows matched split_key=...`

- Ensure `--split_path` and `--data_path` come from the same dataset build
- Ensure `--split_key` is valid (`test_ids` by default)

### High `invalid_rate` in evaluation

- Use `--temperature 0.0`
- Ensure model follows final-line format `ANSWER_*`
- Prefer model/checkpoint aligned with this training prompt format

---

## 13) Reproducibility checklist

For fair comparisons, keep these fixed:

- `--seed`
- `--sample_seed`
- `--max_samples`
- base model version/revision
- evaluation `--temperature` (recommended `0.0`)

---

## 14) Minimal reproducible experiment (recommended first run)

1. Build split (`make_splits`, PubMedQA 500)
2. Build tool SFT data (`build_tool_sft`, weak mode)
3. Train reasoning/context tools
4. Train manager GRPO
5. Run `evaluate_pipeline_vs_baselines.py` with:
   - `pipeline_tools`
   - `pipeline_no_tools`
   - one direct baseline
   - random baseline
   - majority baseline

This gives you immediate evidence whether tool-augmented pipeline beats baselines.

---

## 15) Quick command references

Training script help:

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\agents_as_tools.py --help
```

Evaluation script help:

```powershell
.venv\Scripts\python.exe agents_as_trainable_tools\evaluate_pipeline_vs_baselines.py --help
```

