# =========================
# train_grpo_pubmedqa_explicit_split.py
# =========================

import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# =========================
# 全局随机种子
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================
# 配置
# =========================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
SAVE_PATH = "grpo_manager_pubmedqa_explicit_12271438"
TEST_SAVE_PATH = "pubmedqa_test_seed42_12271438.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 数据加载
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

def preprocess(example):
    return {
        "prompt": build_prompt(example["question"], example["context"]),
        "ground_truth": example["ground_truth"],
    }

# =========================
# Subagent: reasoning agent
# =========================

def reasoning_agent(question: str, context: str) -> str:
    """
    子 agent 做 step-by-step 推理，不直接给最终 yes/no/maybe，
    而是输出分析过程和一个简短总结。
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
    # 这里仍然用 greedy，保证 reasoning 文本稳定
    return run_chat(messages, max_new_tokens=512)

# =========================
# Tokenizer / Model
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

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
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

def run_chat(messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # reasoning agent 用 greedy
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

# =========================
# 显式 rollout（开启真实探索）
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

        # ⭐ 关键修改：manager 决策用 sampling，产生不同 rollout
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=True,           # 打开采样
            temperature=0.7,
            top_p=0.9,
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
# Reward functions（只看对错，不奖励也不惩罚工具）
# =========================

def correctness_reward(completions, ground_truth, **kwargs):
    """
    基础逻辑：
    - 答对：1.0
    - 答错：0.0
    """
    rewards = []
    for c, gt in zip(completions, ground_truth):
        gt = gt.strip().lower()
        pred = c.split("||")[0].strip().lower()

        base = 1.0 if pred == gt else 0.0
        rewards.append(base)
    return rewards

def agent_cost_reward(completions, **kwargs):
    """
    不惩罚用工具，全部返回 0.0。
    """
    return [0.0 for _ in completions]

# =========================
# Main
# =========================

def main():
    # ---- load ----
    dataset = load_data(DATA_PATH)

    # ---- split (80 / 20, seed=42) ----
    splits = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        shuffle=True
    )

    train_dataset = splits["train"]
    test_dataset = splits["test"]

    # ---- preprocess ----
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names
    )

    test_dataset = test_dataset.map(
        preprocess,
        remove_columns=test_dataset.column_names
    )

    # ---- GRPO config ----
    grpo_args = GRPOConfig(
        max_completion_length=16,
        temperature=0.7,
        num_generations=4,
        fp16=(DEVICE == "cuda"),
        bf16=False,
        beta=0.01,
        scale_rewards="group",
    )

    # ---- trainer (train only) ----
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[
            correctness_reward,
            agent_cost_reward,
        ],
        rollout_func=explicit_rollout,
    )

    trainer.train()

    # ---- save model ----
    trainer.model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # ---- save clean test set ----
    with open(TEST_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            test_dataset.to_list(),
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Training done. Test set saved to {TEST_SAVE_PATH}")

if __name__ == "__main__":
    main()
