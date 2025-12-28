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
import wandb  # 用于 Weights & Biases 追踪

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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"
SAVE_PATH = "grpo_manager_pubmedqa_explicit_12281438"
TEST_SAVE_PATH = "pubmedqa_test_seed42_12281438.json"

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
    # 推理 agent 用 greedy，保证稳定
    return run_chat(messages, max_new_tokens=512)

# =========================
# Tokenizer / Model（A100 上用 bf16）
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

# A100 上用 bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
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
# 显式 rollout（两步：第一步可 CALL_REASONING，第二步只允许 ANSWER_*）
# =========================

def explicit_rollout(model, tokenizer, prompt, question, context):
    """
    返回：
    - completion: "yes || ACTIONS=CALL_REASONING|ANSWER_YES"
    - actions_taken: list[str]

    逻辑：
    - 第一步：可以 CALL_REASONING 或 ANSWER_*
    - 第二步：只允许 ANSWER_YES / ANSWER_NO / ANSWER_MAYBE
    """
    messages = [{"role": "user", "content": prompt}]
    actions_taken = []
    used_reasoning = False

    for step in range(2):
        # 第二步前加一句 system 约束：只能选 ANSWER_*
        if step == 1:
            messages.append({
                "role": "system",
                "content": (
                    "Now you must choose a FINAL answer action.\n"
                    "Available actions:\n"
                    "- ANSWER_YES\n"
                    "- ANSWER_NO\n"
                    "- ANSWER_MAYBE\n\n"
                    "You are NOT allowed to choose CALL_REASONING in this step.\n"
                    "Output ONLY the action name."
                )
            })

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        # 训练阶段 manager 决策用采样做探索
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=True,
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

        # 安全兜底：第二步如果还是给 CALL_REASONING，强制改成 ANSWER_MAYBE
        if step == 1 and action == "CALL_REASONING":
            action = "ANSWER_MAYBE"

        # 第一轮里，如果已经用过 reasoning，就不再允许再次 CALL_REASONING
        if step == 0 and action == "CALL_REASONING" and used_reasoning:
            action = "ANSWER_MAYBE"

        actions_taken.append(action)

        if action.startswith("ANSWER"):
            answer = action.replace("ANSWER_", "").lower()
            trace = "|".join(actions_taken)
            return f"{answer} || ACTIONS={trace}", actions_taken

        if action == "CALL_REASONING":
            used_reasoning = True
            reasoning = reasoning_agent(question, context)
            messages.append({
                "role": "system",
                "content": f"Reasoning:\n{reasoning}"
            })

    # 理论上不会走到这里，因为第二步必然产生 ANSWER_*
    # 但为了安全，再兜底一次
    if not actions_taken or not actions_taken[-1].startswith("ANSWER"):
        actions_taken.append("ANSWER_MAYBE")
    answer = "maybe"
    trace = "|".join(actions_taken)
    return f"{answer} || ACTIONS={trace}", actions_taken

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
    # 显式登录 wandb（方案三）
    wandb.login(key="422b2738db53806489567fb2db277d201112125e")

    # Weights & Biases 初始化
    wandb.init(
        project="pubmedqa-grpo",
        name="manager_reasoning_bf16_0.5B",
    )

    # load
    dataset = load_data(DATA_PATH)

    # split (80 / 20, seed=42)
    splits = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        shuffle=True
    )

    train_dataset = splits["train"]
    test_dataset = splits["test"]

    # preprocess
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names
    )

    test_dataset = test_dataset.map(
        preprocess,
        remove_columns=test_dataset.column_names
    )

    # GRPO config（bf16 + wandb）
    grpo_args = GRPOConfig(
        max_completion_length=16,
        temperature=0.7,
        num_generations=4,
        fp16=False,               
        bf16=(DEVICE == "cuda"),  
        beta=0.01,
        scale_rewards="group",
        log_with="wandb",
        project_kwargs={
            "project": "pubmedqa-grpo",
            "name": "manager_reasoning_bf16",
        },
    )

    # trainer
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

    # save model
    trainer.model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # save clean test set
    with open(TEST_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            test_dataset.to_list(),
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Training done. Test set saved to {TEST_SAVE_PATH}")
    wandb.finish()

if __name__ == "__main__":
    main()
