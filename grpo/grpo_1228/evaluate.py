# =========================
# test_grpo_pubmedqa_explicit.py
# =========================

import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 全局随机种子（保证和训练 split 一致）
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================
# 配置（注意 MODEL_PATH 用训练后的）
# =========================

RAW_DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
MODEL_PATH = "grpo_manager_pubmedqa_explicit_12271438"   # 对齐训练脚本里的 SAVE_PATH
EVAL_SAVE_PATH = "pubmedqa_explicit_eval_seed42_reasoning.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 数据加载（和训练脚本保持一致）
# =========================

def load_data(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for ex in raw:
        rows.append({
            "question": ex["question"],
            "context": ex["context"],
            "ground_truth": ex["ground_truth"].lower(),  # yes / no / maybe
        })

    return Dataset.from_list(rows)

# =========================
# Manager Prompt（显式 action space）
# =========================

def build_prompt(question: str, context: str) -> str:
    return f"""
You are a manager agent.

You must choose exactly ONE action.

Available actions:
- CALL_REASONING
- ANSWER_YES
- ANSWER_NO
- ANSWER_MAYBE

Question:
{question}

Context:
{context}

Rules:
1. If the context is sufficient, answer directly.
2. If the situation is unclear or you need deeper analysis, you may call the reasoning agent.
3. Output ONLY the action name. Do not output anything else.
"""

# =========================
# 加载微调后的 tokenizer / model
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# 和训练时一样的 schema（CALL_REASONING）
tokenizer.response_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "action": {
            "type": "string",
            "enum": [
                "CALL_REASONING",
                "ANSWER_YES",
                "ANSWER_NO",
                "ANSWER_MAYBE"
            ]
        }
    },
    "required": ["action"]
}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# =========================
# 工具函数
# =========================

def run_chat(messages, max_new_tokens=128):
    """
    用于 reasoning_agent 的对话调用，保持 greedy，避免评测时引入额外随机性。
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

def reasoning_agent(question: str, context: str) -> str:
    """
    和训练脚本保持一致的 step-by-step reasoning agent。
    """
    messages = [
        {"role": "system", "content": "You are a clinical reasoning agent."},
        {"role": "user", "content": f"""
You will reason about a medical question step by step.

Question:
{question}

Abstract:
{context}

Instructions:
1. Think step by step about the clinical question.
2. Identify evidence that supports 'yes', evidence that supports 'no', and any conflicting or missing information.
3. Explain the uncertainty if the evidence is not clear.
4. At the end, give a short conclusion line that starts with:
   REASONING_SUMMARY: ...

Do NOT output 'yes', 'no', or 'maybe' as a single word answer. Focus on detailed reasoning.
"""}
    ]
    return run_chat(messages, max_new_tokens=512)

# =========================
# 显式 rollout（评测时用 greedy）
# =========================

def explicit_rollout(model, tokenizer, prompt, question, context):
    """
    返回：
    - completion: "yes || ACTIONS=CALL_REASONING|ANSWER_YES"
    - actions_taken: list[str]
    """
    messages = [{"role": "user", "content": prompt}]
    actions_taken = []

    for _ in range(2):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        # 评测阶段用 greedy，保证结果稳定可复现
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        parsed = tokenizer.parse_response(raw)
        action = parsed["action"]
        actions_taken.append(action)

        if action.startswith("ANSWER"):
            answer = action.replace("ANSWER_", "").lower()
            trace = "|".join(actions_taken)
            return f"{answer} || ACTIONS={trace}", actions_taken

        if action == "CALL_REASONING":
            reasoning = reasoning_agent(question, context)
            messages.append({
                "role": "system",
                "content": f"Reasoning:\n{reasoning}"
            })

    # 如果两步以内都没有给出 ANSWER_*，默认 maybe
    return "maybe || ACTIONS=NONE", actions_taken

# =========================
# 评测主函数
# =========================

def main():
    # 1. 重新加载原始数据，并用同一个 seed 做 train/test split
    dataset = load_data(RAW_DATA_PATH)
    splits = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        shuffle=True
    )
    test_dataset = splits["test"]

    total = 0
    correct = 0
    call_reasoning_count = 0

    records = []

    for ex in test_dataset:
        q = ex["question"]
        ctx = ex["context"]
        gt = ex["ground_truth"]   # yes/no/maybe

        prompt = build_prompt(q, ctx)

        with torch.no_grad():
            completion, actions = explicit_rollout(model, tokenizer, prompt, q, ctx)

        pred = completion.split("||")[0].strip().lower()
        is_correct = int(pred == gt)

        total += 1
        correct += is_correct
        if "CALL_REASONING" in completion:
            call_reasoning_count += 1

        records.append({
            "question": q,
            "context": ctx,
            "ground_truth": gt,
            "completion": completion,
            "pred": pred,
            "correct": bool(is_correct),
            "actions": actions,
        })

        if total % 20 == 0:
            print(f"Evaluated {total} examples...")

    acc = correct / total if total > 0 else 0.0
    call_ratio = call_reasoning_count / total if total > 0 else 0.0

    print("===================================")
    print(f"Total examples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print(f"CALL_REASONING ratio: {call_ratio:.4f}")
    print("===================================")

    # 保存详细结果，方便事后分析
    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Eval details saved to {EVAL_SAVE_PATH}")

if __name__ == "__main__":
    main()
