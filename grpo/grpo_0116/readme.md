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

## Step 4: Configure Paths and Models (`main.py`)

```python
MANAGER_MODEL = "Qwen/Qwen3-8B"
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
SAVE_PATH = "grpo_manager_qwen3_tools_optional_tool_0117_2353"

REASONER_MODEL = "Qwen/Qwen3-8B"
REASONER_MAX_NEW_TOKENS = 10000
```


## Step 4: Run Training

```bash
python main.py
```

---

## Step 5: Run Evaluation

```bash
python evaluate.py
```
---

根据赵老师的建议重新改了一版requirements.txt, 之前的版本为requirements2.txt，在remote电脑上可以成功运行。
