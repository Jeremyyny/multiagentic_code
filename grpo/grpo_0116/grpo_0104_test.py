import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from transformers.utils.chat_template_utils import get_json_schema

# -----------------------------
# Seed
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Config
# -----------------------------
MANAGER_MODEL = "Qwen/Qwen3-8B"
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
SAVE_PATH = "grpo_manager_qwen3_tools_optional_tool_v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FROZEN_REASONER = True
REASONER_MODEL = "Qwen/Qwen3-8B"
REASONER_DEVICE = "cpu"
REASONER_MAX_NEW_TOKENS = 1024

# Optional: encourage/discourage tool usage.
TOOL_PENALTY = 0.0  # penalty if tool used; set to 0.0 to disable
TOOL_CALL_BONUS = 0.05  # reward for using tool; set to 0.0 to disable
TOOL_BONUS_ONLY_IF_CORRECT = True  # only add bonus when final answer is correct

# -----------------------------
# Global caches
# -----------------------------
ID2EX: Dict[int, Dict[str, Any]] = {}
TOOL_CACHE: Dict[int, str] = {}

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "You are a manager agent solving PubMedQA-style clinical questions.\n"
    "Calling the tool is OPTIONAL.\n"
    "You have two phases: decide whether to call the tool, then answer.\n"
    "If you call the tool, do NOT answer the final label in the same message.\n"
    "Only return a tool call, then wait for the tool output.\n"
    "After tool output is provided, answer the question.\n"
    "If unsure, call the tool instead of guessing.\n\n"
    "Output rule for FINAL answer:\n"
    "The last line MUST be exactly one of:\n"
    "ANSWER_YES\n"
    "ANSWER_NO\n"
    "ANSWER_MAYBE\n"
    "Do not write anything after that last line.\n"
    "Do NOT output <think>.\n"
)

ANSWER_LASTLINE_RE = re.compile(r"(?:^|\n)\s*ANSWER_(YES|NO|MAYBE)\s*$", re.IGNORECASE)

def parse_answer_label_lastline(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANSWER_LASTLINE_RE.search(text.strip())
    if not m:
        return None
    last = m.group(1).upper()
    return {"YES": "yes", "NO": "no", "MAYBE": "maybe"}[last]

def build_messages(example_id: int, question: str, context: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Example ID: {example_id}\n\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "You may call the tool `reasoning_tool(example_id=...)` if needed.\n"
                "If you do NOT call the tool, answer directly.\n"
            ),
        },
    ]

# -----------------------------
# Data
# -----------------------------
def load_data(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for i, ex in enumerate(raw):
        eid = int(i)
        q = ex["question"]
        ctx = ex["context"]
        gt = ex["ground_truth"].strip().lower()
        ID2EX[eid] = {"question": q, "context": ctx, "ground_truth": gt}
        rows.append({"example_id": eid, "question": q, "context": ctx, "ground_truth": gt})
    return Dataset.from_list(rows)

# -----------------------------
# Frozen reasoner (optional)
# -----------------------------
@dataclass
class FrozenReasoner:
    model_name: str
    device: str = "cpu"
    max_new_tokens: int = 512

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        dtype = torch.float32 if self.device == "cpu" else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def infer(self, question: str, context: str) -> str:
        sys = (
            "You are a clinical reasoning assistant.\n"
            "Given a clinical question and corresponding context.\n"
            "Return step by step reasoning.\n"
            "Do NOT output <think>.\n"
        )
        user = f"Question:\n{question}\n\nContext:\n{context}\n"
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

_reasoner: Optional[FrozenReasoner] = None
if USE_FROZEN_REASONER:
    _reasoner = FrozenReasoner(REASONER_MODEL, REASONER_DEVICE, REASONER_MAX_NEW_TOKENS)

# -----------------------------
# Tool function (used by TRL tools)
# -----------------------------

PRINT_TOOL_OUTPUT = True
TOOL_OUTPUT_MAX_CHARS = 2500  # None 表示不截断

def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def reasoning_tool(example_id: int) -> str:
    """Return an expert report for a PubMedQA example.

    Args:
        example_id: Integer example id (index into ID2EX).

    Returns:
        A short report string.
    """
    eid = int(example_id)
    if eid in TOOL_CACHE:
        return TOOL_CACHE[eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = "TOOL_ERROR: example_id not found."
        TOOL_CACHE[eid] = out
        return out

    q = ex["question"]
    ctx = ex["context"]

    tail = ctx.strip()

    reasoner_out = ""
    if _reasoner is not None:
        try:
            reasoner_out = _reasoner.infer(q, ctx)
        except Exception as e:
            reasoner_out = f"(reasoner_error: {e})"

    out = (
        "EXPERT_TOOL_REPORT\n"
        f"TAIL_HINT: {tail}\n"
        f"REASONER:\n{reasoner_out}\n"
    )
    if PRINT_TOOL_OUTPUT:
        print(f"\n[TOOL] example_id={eid}\n{_truncate(out, TOOL_OUTPUT_MAX_CHARS)}\n")
        
    TOOL_CACHE[eid] = out
    return out



# -----------------------------
# Manager tokenizer
# -----------------------------
manager_tok = AutoTokenizer.from_pretrained(MANAGER_MODEL, trust_remote_code=True)
manager_tok.padding_side = "left"
if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
    manager_tok.pad_token_id = manager_tok.eos_token_id

def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
    eid = int(example["example_id"])
    msgs = build_messages(eid, example["question"], example["context"])
    return {
        "prompt": msgs,                      
        "ground_truth": example["ground_truth"],
        "example_id": eid,
    }

# -----------------------------
# Reward
# -----------------------------
def _ensure_len(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if len(x) == 0:
            return [None] * n
        if n % len(x) == 0:
            k = n // len(x)
            out = []
            for item in x:
                out.extend([item] * k)
            return out
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n

def _last_assistant_text(completion: Any) -> str:
    # conversational completion: list[dict]
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                return "" if content is None else str(content)
        return ""
    # sometimes dict
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return "" if content is None else str(content)
    # fallback
    return "" if completion is None else str(completion)

def _used_tool(completion: Any) -> bool:
    if isinstance(completion, list):
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool":
                return True
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return True
    return False

def accuracy_reward(prompts=None, completions=None, ground_truth=None, example_id=None, **kwargs):
    n = len(completions)
    gts = _ensure_len(ground_truth, n)

    rewards = []
    for c, gt in zip(completions, gts):
        gt = (gt or "").strip().lower()
        text = _last_assistant_text(c)
        pred = parse_answer_label_lastline(text)
        is_correct = pred is not None and pred == gt
        r = 1.0 if is_correct else 0.0

        used_tool = _used_tool(c)
        if TOOL_PENALTY > 0.0 and used_tool:
            r -= TOOL_PENALTY

        if TOOL_CALL_BONUS > 0.0 and used_tool and (is_correct or not TOOL_BONUS_ONLY_IF_CORRECT):
            r += TOOL_CALL_BONUS

        rewards.append(r)
    return rewards

# -----------------------------
# Main
# -----------------------------

from transformers.utils.chat_template_utils import get_json_schema

def main():
    dataset = load_data(DATA_PATH)
    splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)

    train_dataset = splits["train"].map(preprocess, remove_columns=splits["train"].column_names)
    print(get_json_schema(reasoning_tool))
    grpo_args = GRPOConfig(
        output_dir=SAVE_PATH,
        remove_unused_columns=False,
        max_completion_length=1024,
        temperature=0.7,
        num_generations=4,
        bf16=(DEVICE == "cuda"),
        beta=0.0,
        scale_rewards="group",
        report_to=[],
        use_vllm=False,  # ✅ 不需要 vLLM
        per_device_train_batch_size=4,
        max_tool_calling_iterations=1,  # ✅ 你这个任务基本只需要一次 tool call
        chat_template_kwargs={"enable_thinking": False},  # ✅ Qwen3 避免<think>
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=None,  # None = 全部打印当前 batch
        log_unique_prompts=False,
    )

    trainer = GRPOTrainer(
        model=MANAGER_MODEL,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward],
        processing_class=manager_tok,
        tools=[reasoning_tool],   # ✅ 关键：启用 TRL 内置 tools 流程
        rollout_func=None,        # ✅ 不再需要 rollout_func
    )

    trainer.train()
    trainer.model.save_pretrained(SAVE_PATH)
    manager_tok.save_pretrained(SAVE_PATH)
    print(f"Saved model to {SAVE_PATH}")

if __name__ == "__main__":
    main()
