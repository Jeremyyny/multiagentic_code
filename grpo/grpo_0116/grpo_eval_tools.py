import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response
from transformers.utils.chat_template_utils import get_json_schema


# -----------------------------
# Seed (match training)
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
MANAGER_MODEL_PATH = "grpo_manager_qwen3_tools_optional_tool_v1"
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
EVAL_SAVE_PATH = "grpo_eval_with_tools_seed42.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FROZEN_REASONER = True
REASONER_MODEL = "Qwen/Qwen3-0.6B"
REASONER_DEVICE = "cpu"
REASONER_MAX_NEW_TOKENS = 512

MAX_TOOL_CALLING_ITERATIONS = 1
MAX_NEW_TOKENS_FIRST = 128
MAX_NEW_TOKENS_FINAL = 128
TOOL_OUTPUT_MAX_CHARS = 1200  # truncate tool output in logs (None for no limit)

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# -----------------------------
# Global caches
# -----------------------------
ID2EX: Dict[int, Dict[str, Any]] = {}
TOOL_CACHE: Dict[int, str] = {}

# -----------------------------
# Prompt (match training)
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
# Frozen reasoner (optional tool)
# -----------------------------
@dataclass
class FrozenReasoner:
    model_name: str
    device: str = "cpu"
    max_new_tokens: int = 96

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
            "Given a PubMed abstract and a question, decide YES/NO/MAYBE.\n"
            "Return two lines:\n"
            "1) PRED: YES/NO/MAYBE\n"
            "2) One short justification sentence.\n"
            "Do NOT output <think>.\n"
        )
        user = f"Question:\n{question}\n\nAbstract:\n{context}\n"
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

PRINT_TOOL_OUTPUT = True
TOOL_OUTPUT_MAX_CHARS = 1200  # None 表示不截断

def _truncate(text: str, limit: Optional[int]) -> str:
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

    sents = re.split(r"(?<=[.!?])\s+", ctx.strip())
    tail = " ".join([s for s in sents[-2:] if s]) if sents else ""

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


def _generate(messages: List[Dict[str, Any]], tokenizer, model, max_new_tokens: int):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        tools=[reasoning_tool],
        **CHAT_TEMPLATE_KWARGS,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = outputs[0][prompt_len:].tolist()
    parsed = parse_response(tokenizer, completion_ids)
    return parsed, completion_ids


def _print_messages(messages: List[Dict[str, Any]]) -> None:
    print("PROMPT:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"[{role}] {content}")


def run_example(example: Dict[str, Any], tokenizer, model):
    messages = build_messages(example["example_id"], example["question"], example["context"])
    _print_messages(messages)

    parsed, _ = _generate(messages, tokenizer, model, MAX_NEW_TOKENS_FIRST)
    print("ASSISTANT (first):")
    print(parsed)

    tool_calls = parsed.get("tool_calls") or []
    final_parsed = parsed

    if tool_calls and MAX_TOOL_CALLING_ITERATIONS > 0:
        # Append assistant tool call message
        messages.append(parsed)
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                tool_msg = {"role": "tool", "name": "unknown", "content": "TOOL_ERROR: unsupported tool call type."}
                print("TOOL CALL ERROR:", tool_call)
            else:
                fn = tool_call["function"]
                name = fn["name"]
                args = fn.get("arguments", {})
                print("TOOL CALL:", {"name": name, "arguments": args})
                try:
                    result = reasoning_tool(**args)
                except Exception as e:
                    result = f"TOOL_ERROR: {e}"
                print("TOOL OUTPUT:")
                print(_truncate(str(result), TOOL_OUTPUT_MAX_CHARS))
                tool_msg = {"role": "tool", "name": name, "content": str(result)}
            messages.append(tool_msg)

        final_parsed, _ = _generate(messages, tokenizer, model, MAX_NEW_TOKENS_FINAL)
        print("ASSISTANT (final):")
        print(final_parsed)

    final_text = final_parsed.get("content", "")
    parsed_label = parse_answer_label_lastline(final_text)
    print("PARSED ANSWER:", parsed_label)
    print("GROUND TRUTH:", example["ground_truth"])

    return {
        "example_id": example["example_id"],
        "question": example["question"],
        "context": example["context"],
        "ground_truth": example["ground_truth"],
        "assistant_first": parsed,
        "assistant_final": final_parsed,
        "parsed_answer": parsed_label,
        "used_tool": bool(tool_calls),
    }


def main():
    dataset = load_data(DATA_PATH)
    splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    test_dataset = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(MANAGER_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        tokenizer = add_response_schema(tokenizer)
    except ValueError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        MANAGER_MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    total = 0
    correct = 0
    tool_calls = 0
    records = []

    for ex in test_dataset:
        print("=" * 88)
        record = run_example(ex, tokenizer, model)
        records.append(record)

        total += 1
        if record["parsed_answer"] == ex["ground_truth"]:
            correct += 1
        if record["used_tool"]:
            tool_calls += 1

    acc = correct / total if total > 0 else 0.0
    tool_ratio = tool_calls / total if total > 0 else 0.0
    print("=" * 88)
    print(f"TOTAL: {total}")
    print(f"ACCURACY: {acc:.4f}")
    print(f"TOOL CALL RATIO: {tool_ratio:.4f}")

    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)
    print(f"Saved eval details to {EVAL_SAVE_PATH}")


if __name__ == "__main__":
    main()
