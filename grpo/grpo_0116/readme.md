# GRPO Manager–Reasoner Pipeline

## Step 1: Install uv

```bash
pip install uv
```

---
## Create a uv venv with Python 3.10


```bash
# check what python you currently have
python --version

# find a python3.10 executable (examples)
which python3.10

# create venv using that interpreter
uv venv --python python3.10
```

## Step 2: Activate Environment

### macOS / Linux
```bash
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies

```bash
uv pip install -r requirements.txt
```
---

# TRL Multi-Agent Training Run Guide (Current `agents_as_tools.py`)

This README matches the **current** `agents_as_tools.py` stages and arguments.

Working directory:

```powershell
cd c:\Users\yyn07\Desktop\multi_agent_test\New_GRPO_code\trl_vllm
```

## 0) Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

If missing packages appear:

```powershell
pip install datasets peft
```

## 1) Stage List (Current)

`agents_as_tools.py --stage` supports:

- `make_splits`
- `build_tool_sft`
- `train_tool_reasoning`
- `train_tool_context`
- `train_manager_grpo`
- `evolve_build_manager_sft`
- `train_manager_sft`
- `evolve_round`

## 2) Build Data Split

```powershell
python agents_as_tools.py --stage make_splits `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
  --test_size 200 `
  --dev_size 160 `
  --seed 42
```

Output:

- `splits_pubmedqa_1000.json`

## 3) Build Tool SFT Data

```powershell
python agents_as_tools.py --stage build_tool_sft `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
  --tool_sft_out_dir sft_data_mvp_split `
  --top_k 20 `
  --tool_variants_train 4 `
  --tool_variants_dev 2 `
  --tool_synth_mode weak `
  --seed 42
```

Outputs:

- `sft_data_mvp_split/tool_reasoning_train.jsonl`
- `sft_data_mvp_split/tool_reasoning_dev.jsonl`
- `sft_data_mvp_split/tool_context_train.jsonl`
- `sft_data_mvp_split/tool_context_dev.jsonl`

If you want teacher synthesis, set environment vars and use `--tool_synth_mode gpt`.

Required:

- `TEACHER_BASE_URL`
- `TEACHER_MODEL`

Optional:

- `TEACHER_API_KEY` (some local OpenAI-compatible servers may not need it)
- `TEACHER_TIMEOUT` (default: `60`)

Example A (OpenAI API):

```powershell
$env:TEACHER_BASE_URL="https://api.openai.com"
$env:TEACHER_MODEL="gpt-4o-mini"
$env:TEACHER_API_KEY="YOUR_API_KEY"
```

Example B (local OpenAI-compatible server):

```powershell
$env:TEACHER_BASE_URL="http://127.0.0.1:8000"
$env:TEACHER_MODEL="your-local-model-name"
# Optional:
# $env:TEACHER_API_KEY=""
```

## 4) Train Reasoning Tool (SFT)

```powershell
python agents_as_tools.py --stage train_tool_reasoning `
  --base_model Qwen/Qwen3-0.6B `
  --tool_sft_out_dir sft_data_mvp_split `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --tool_max_seq_len 4096 `
  --tool_lr 2e-4 `
  --tool_epochs 2 `
  --tool_bs 1 `
  --tool_grad_accum 8 `
  --tool_use_lora `
  --seed 42
```

## 5) Train Context Tool (SFT)

```powershell
python agents_as_tools.py --stage train_tool_context `
  --base_model Qwen/Qwen3-0.6B `
  --tool_sft_out_dir sft_data_mvp_split `
  --context_tool_out context_lora_mvp_split `
  --tool_max_seq_len 4096 `
  --tool_lr 2e-4 `
  --tool_epochs 2 `
  --tool_bs 1 `
  --tool_grad_accum 8 `
  --tool_use_lora `
  --seed 42
```

## 6) Train Manager (GRPO + Tools)

```powershell
python agents_as_tools.py --stage train_manager_grpo `
  --base_model Qwen/Qwen3-0.6B `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
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

Important outputs:

- `manager_grpo_mvp_split/fail_buffer.jsonl`
- `manager_grpo_mvp_split/train_raw_trace.jsonl`
- manager checkpoint under `manager_grpo_mvp_split/`

Append behavior:

- By default, both files are overwritten each run.
- Set `FAIL_BUFFER_APPEND=1` to append fail buffer.
- Set `RAW_TRACE_APPEND=1` to append raw trace.

## 7) Evaluate on Test Split

```powershell
python evaluation_agents_tools.py `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
  --manager_dir manager_grpo_mvp_split `
  --base_model_for_tools Qwen/Qwen3-0.6B `
  --reasoning_adapter reasoning_lora_mvp_split `
  --context_adapter context_lora_mvp_split `
  --use_tools `
  --max_tool_calls 2 `
  --temperature 0.0 `
  --max_new_tokens 1024 `
  --out_jsonl test_predictions_debug.jsonl
```

Alternative aliases in evaluation still work:

- `--evidence_adapter` = `--reasoning_adapter`
- `--judge_adapter` = `--context_adapter`

## 8) Evolve Manager from Failures

Two-step mode:

```powershell
python agents_as_tools.py --stage evolve_build_manager_sft `
  --base_model Qwen/Qwen3-0.6B `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --context_tool_out context_lora_mvp_split `
  --manager_out manager_grpo_mvp_split `
  --evolve_out_dir evolve_manager_sft `
  --max_fail_samples 2000 `
  --planning_mode realistic `
  --seed 42

python agents_as_tools.py --stage train_manager_sft `
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

One-command round (GRPO -> build evolve SFT -> manager SFT):

```powershell
python agents_as_tools.py --stage evolve_round `
  --base_model Qwen/Qwen3-0.6B `
  --data_path pqal_question_context_groundtruth.json `
  --split_path splits_pubmedqa_1000.json `
  --reasoning_tool_out reasoning_lora_mvp_split `
  --context_tool_out context_lora_mvp_split `
  --manager_out manager_grpo_mvp_split `
  --evolve_out_dir evolve_manager_sft `
  --manager_sft_out manager_sft_evolved `
  --planning_mode realistic `
  --fail_buffer_jsonl manager_grpo_mvp_split\fail_buffer.jsonl `
  --raw_trace_jsonl manager_grpo_mvp_split\train_raw_trace.jsonl `
  --seed 42
```

`planning_mode`:

- `realistic`: teacher plans with only question/context.
- `oracle`: teacher can also see tool outputs (optimistic debugging mode).

## 9) Quick Checks

Check whether tools were called in evaluation:

```powershell
rg -n "\"tool_calls_used\": [1-9]" test_predictions_debug.jsonl
```

Check label distribution in evaluation:

```powershell
rg -n "\"pred\": \"yes\"|\"pred\": \"no\"|\"pred\": \"maybe\"" test_predictions_debug.jsonl
```

Check how many failures used tools during training:

```powershell
rg -n "\"tool_names\": \\[" manager_grpo_mvp_split\fail_buffer.jsonl
```

## 10) Common Warnings / Errors

- `DocstringParsingException ... has no docstring`  
  Tool functions must have docstrings. Current script already includes them.

- `ModuleNotFoundError: No module named 'datasets'`  
  Install with `pip install datasets` or `pip install -r requirements.txt`.

- HF unauthenticated warning  
  Optional: set `HF_TOKEN` to avoid rate limits while downloading models.

- Generation warnings about deprecated flags  
  Usually non-fatal; training can continue.

## 11) Notes

- Manager final line must be exactly one of:
  - `ANSWER_YES`
  - `ANSWER_NO`
  - `ANSWER_MAYBE`

- Tool-call format expected by prompt:

```text
<tool_call>
{"name": "reasoning_tool", "arguments": {"example_id": 123}}
</tool_call>
```

or

```text
<tool_call>
{"name": "context_tool", "arguments": {"example_id": 123}}
</tool_call>
```
