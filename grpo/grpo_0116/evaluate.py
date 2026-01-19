import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response

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
# Config (match training)
# -----------------------------
MODEL_PATH = "grpo_manager_qwen3_tools_optional_tool_v2"   # trained output dir
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
EVAL_SAVE_PATH = "grpo_eval_with_tools_seed42_v2.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FROZEN_REASONER = True
REASONER_MODEL = "Qwen/Qwen3-8B"
REASONER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REASONER_MAX_NEW_TOKENS = 2048

MAX_TOOL_CALLING_ITERATIONS = 1
MAX_COMPLETION_LENGTH = 2048

# Generation settings
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = None
REPETITION_PENALTY = 1.0

PRINT_TOOL_OUTPUT = True
TOOL_OUTPUT_MAX_CHARS = 3000  # None = no truncation

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

# Extra user instruction only after tool output (helps parse rate a lot)
POST_TOOL_USER_REMINDER = (
    "You have received the tool output.\n"
    "Now provide the FINAL answer.\n"
    "The last line MUST be exactly one of:\n"
    "ANSWER_YES\n"
    "ANSWER_NO\n"
    "ANSWER_MAYBE\n"
    "Do not write anything after that last line.\n"
)

ANSWER_LASTLINE_RE = re.compile(r"(?:^|\n)\s*ANSWER_(YES|NO|MAYBE)\s*$", re.IGNORECASE)

# 1) strict <tool_call> ... </tool_call>
TOOL_CALL_RE_STRICT = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# 2) loose: <tool_call> { ... }  (no closing tag)
TOOL_CALL_RE_LOOSE = re.compile(r"<tool_call>\s*(\{.*\})\s*$", re.DOTALL)
# 3) last resort: find a json object containing reasoning_tool
REASONING_TOOL_JSON_RE = re.compile(r"(\{.*\"name\"\s*:\s*\"reasoning_tool\".*\})", re.DOTALL)


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
    max_new_tokens: int = 512

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        dtype = torch.float32 if self.device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
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
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


_reasoner: Optional[FrozenReasoner] = None
if USE_FROZEN_REASONER:
    _reasoner = FrozenReasoner(REASONER_MODEL, REASONER_DEVICE, REASONER_MAX_NEW_TOKENS)


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
# Generation helpers
# -----------------------------
def _prepare_inputs(
    messages: List[Dict[str, Any]],
    tokenizer,
    model,
    tools: Optional[List[Any]] = None,
):
    kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        **CHAT_TEMPLATE_KWARGS,
    )
    if tools is not None:
        kwargs["tools"] = tools

    inputs = tokenizer.apply_chat_template(messages, **kwargs)

    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


def _generate(
    messages: List[Dict[str, Any]],
    tokenizer,
    model,
    max_new_tokens: int,
    tools: Optional[List[Any]] = None,
):
    inputs = _prepare_inputs(messages, tokenizer, model, tools=tools)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if TOP_K is not None:
        gen_kwargs["top_k"] = TOP_K

    outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = outputs[0][prompt_len:].tolist()

    parsed = _parse_completion(tokenizer, completion_ids)
    return parsed, completion_ids


def _extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Robustly extract tool calls even if </tool_call> is missing.
    """
    tool_calls = []

    candidates = []
    m = TOOL_CALL_RE_STRICT.search(text)
    if m:
        candidates.append(m.group(1))

    m2 = TOOL_CALL_RE_LOOSE.search(text)
    if m2:
        candidates.append(m2.group(1))

    m3 = REASONING_TOOL_JSON_RE.search(text)
    if m3:
        candidates.append(m3.group(1))

    for payload in candidates:
        payload = payload.strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if name:
            tool_calls.append({"type": "function", "function": {"name": name, "arguments": args}})

    return tool_calls


def _strip_tool_call_markup(text: str) -> str:
    # remove both strict and loose tool_call blocks if present
    text = TOOL_CALL_RE_STRICT.sub("", text)
    text = re.sub(r"<tool_call>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_completion(tokenizer, completion_ids: List[int]) -> Dict[str, Any]:
    # Prefer schema-based parsing if available
    if hasattr(tokenizer, "parse_response"):
        return parse_response(tokenizer, completion_ids)

    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    tool_calls = _extract_tool_calls_from_text(text)
    content = _strip_tool_call_markup(text)

    parsed = {"role": "assistant", "content": content}
    if tool_calls:
        parsed["tool_calls"] = tool_calls
    return parsed


def _print_messages(messages: List[Dict[str, Any]]) -> None:
    print("PROMPT:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        name = msg.get("name")
        if name is not None and role == "tool":
            print(f"[{role}:{name}] {content}")
        else:
            print(f"[{role}] {content}")


def _normalize_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def run_example(example: Dict[str, Any], tokenizer, model):
    messages = build_messages(example["example_id"], example["question"], example["context"])
    _print_messages(messages)

    # -----------------------------
    # First pass: tools enabled
    # -----------------------------
    first_parsed, _ = _generate(
        messages, tokenizer, model, MAX_COMPLETION_LENGTH, tools=[reasoning_tool]
    )
    print("ASSISTANT (first):")
    print(first_parsed)

    tool_calls = first_parsed.get("tool_calls") or []
    executed_tool = False

    final_parsed = first_parsed

    # -----------------------------
    # Tool loop: execute tool if requested
    # -----------------------------
    if tool_calls and MAX_TOOL_CALLING_ITERATIONS > 0:
        messages.append(first_parsed)

        for tool_call in tool_calls[:MAX_TOOL_CALLING_ITERATIONS]:
            if tool_call.get("type") != "function":
                tool_msg = {
                    "role": "tool",
                    "name": "unknown",
                    "content": "TOOL_ERROR: unsupported tool call type.",
                }
                print("TOOL CALL ERROR:", tool_call)
                messages.append(tool_msg)
                continue

            fn = tool_call["function"]
            name = fn["name"]
            args = _normalize_args(fn.get("arguments", {}))

            print("TOOL CALL:", {"name": name, "arguments": args})
            try:
                result = reasoning_tool(**args)
                executed_tool = True
            except Exception as e:
                result = f"TOOL_ERROR: {e}"

            print("TOOL OUTPUT:")
            print(_truncate(str(result), TOOL_OUTPUT_MAX_CHARS))

            tool_msg = {"role": "tool", "name": name, "content": str(result)}
            messages.append(tool_msg)

        # Add a user reminder to enforce final format
        messages.append({"role": "user", "content": POST_TOOL_USER_REMINDER})

        # -----------------------------
        # Second pass: tools disabled (important)
        # -----------------------------
        final_parsed, _ = _generate(
            messages, tokenizer, model, MAX_COMPLETION_LENGTH, tools=[]
        )
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
        "assistant_first": first_parsed,
        "assistant_final": final_parsed,
        "parsed_answer": parsed_label,
        # record actual execution, not just "tool_calls existed"
        "used_tool": bool(executed_tool),
        "requested_tool": bool(tool_calls),
    }


def main():
    dataset = load_data(DATA_PATH)
    splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    test_dataset = splits["test"]

    # IMPORTANT: load tokenizer from MODEL_PATH so schema/template matches training output
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        tokenizer = add_response_schema(tokenizer)
    except ValueError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    total = 0
    correct = 0
    tool_exec = 0
    tool_req = 0
    parse_fail = 0
    records = []

    for ex in test_dataset:
        print("=" * 88)
        record = run_example(ex, tokenizer, model)
        records.append(record)

        total += 1
        if record["parsed_answer"] is None:
            parse_fail += 1
        if record["parsed_answer"] == ex["ground_truth"]:
            correct += 1
        if record["used_tool"]:
            tool_exec += 1
        if record["requested_tool"]:
            tool_req += 1

    acc = correct / total if total > 0 else 0.0
    tool_exec_ratio = tool_exec / total if total > 0 else 0.0
    tool_req_ratio = tool_req / total if total > 0 else 0.0
    parse_fail_ratio = parse_fail / total if total > 0 else 0.0

    print("=" * 88)
    print(f"TOTAL: {total}")
    print(f"ACCURACY: {acc:.4f}")
    print(f"PARSE_FAIL: {parse_fail} ({parse_fail_ratio:.4f})")
    print(f"TOOL_REQUEST_RATIO: {tool_req_ratio:.4f}")
    print(f"TOOL_EXEC_RATIO: {tool_exec_ratio:.4f}")

    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)
    print(f"Saved eval details to {EVAL_SAVE_PATH}")


if __name__ == "__main__":
    main()