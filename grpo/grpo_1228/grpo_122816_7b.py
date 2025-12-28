# =========================
# train_grpo_pubmedqa_explicit_split_clean_and_frozen_reasoner.py
# =========================

import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import wandb

try:
    import trl.extras.profiling as trl_profiling
    trl_profiling.wandb = wandb
except Exception:
    # 如果未来版本已经修了，这里就安静跳过
    pass

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
MANAGER_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
REASONING_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # ✅ 改：reasoning 用冻结 7B

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
        rows.append(
            {
                "question": ex["question"],
                "context": ex["context"],
                "ground_truth": ex["ground_truth"].lower(),  # yes / no / maybe
            }
        )
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
""".strip()

def preprocess(example):
    # ✅ 保留 question/context，rollout 需要用
    return {
        "prompt": build_prompt(example["question"], example["context"]),
        "ground_truth": example["ground_truth"],
        "question": example["question"],
        "context": example["context"],
    }

# =========================
# Manager tokenizer / model
# =========================
manager_tokenizer = AutoTokenizer.from_pretrained(
    MANAGER_MODEL_NAME,
    trust_remote_code=True
)

# schema：四个动作
MANAGER_SCHEMA_FULL = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "action": {
            "type": "string",
            "enum": ["CALL_REASONING", "ANSWER_YES", "ANSWER_NO", "ANSWER_MAYBE"],
        },
    },
    "required": ["action"],
}

# ✅ 改：第二步只允许 ANSWER_*
MANAGER_SCHEMA_FINAL_ONLY = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "action": {
            "type": "string",
            "enum": ["ANSWER_YES", "ANSWER_NO", "ANSWER_MAYBE"],
        },
    },
    "required": ["action"],
}

manager_tokenizer.response_schema = MANAGER_SCHEMA_FULL

manager_model = AutoModelForCausalLM.from_pretrained(
    MANAGER_MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

def run_chat_manager(messages, max_new_tokens=128, do_sample=False, temperature=0.0, top_p=1.0):
    inputs = manager_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(manager_model.device)

    outputs = manager_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=manager_tokenizer.eos_token_id,
    )

    return manager_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

# =========================
# ✅ 冻结的 reasoning tokenizer / model (Qwen2.5-7B-Instruct)
# =========================
reasoning_tokenizer = AutoTokenizer.from_pretrained(
    REASONING_MODEL_NAME,
    trust_remote_code=True
)

reasoning_model = AutoModelForCausalLM.from_pretrained(
    REASONING_MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# ✅ 改：冻结
reasoning_model.eval()
for p in reasoning_model.parameters():
    p.requires_grad_(False)

def run_chat_reasoning(messages, max_new_tokens=512):
    inputs = reasoning_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(reasoning_model.device)

    with torch.no_grad():  # ✅ 改：no_grad
        outputs = reasoning_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy
            temperature=0.0,
            pad_token_id=reasoning_tokenizer.eos_token_id,
        )

    return reasoning_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

def reasoning_agent(question: str, context: str) -> str:
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
""".strip()},
    ]
    return run_chat_reasoning(messages, max_new_tokens=512)

# =========================
# 显式 rollout：单样本（内部用）
# =========================
def explicit_rollout_one(prompt: str, question: str, context: str):
    """
    两步：
    - step0: schema=FULL (允许 CALL_REASONING/ANSWER_*)
    - step1: schema=FINAL_ONLY (只允许 ANSWER_*)
    返回：
      completion: "<yes/no/maybe> || ACTIONS=CALL_REASONING|ANSWER_YES"
      actions_taken: list[str]
    """
    messages = [{"role": "user", "content": prompt}]
    actions_taken = []
    used_reasoning = False

    for step in range(2):
        # ✅ 改：更干净的做法——第二步直接改 schema，让 parse_response 只能得到 ANSWER_*
        if step == 0:
            manager_tokenizer.response_schema = MANAGER_SCHEMA_FULL
        else:
            manager_tokenizer.response_schema = MANAGER_SCHEMA_FINAL_ONLY

        raw = run_chat_manager(
            messages,
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        parsed = manager_tokenizer.parse_response(raw)
        action = parsed["action"]

        # step0：如果已经用过 reasoning，就不让再 CALL_REASONING（这里主要是安全兜底）
        if step == 0 and action == "CALL_REASONING" and used_reasoning:
            action = "ANSWER_MAYBE"

        actions_taken.append(action)

        if action.startswith("ANSWER"):
            answer = action.replace("ANSWER_", "").lower()
            trace = "|".join(actions_taken)
            # 恢复 schema（避免影响别处）
            manager_tokenizer.response_schema = MANAGER_SCHEMA_FULL
            return f"{answer} || ACTIONS={trace}", actions_taken

        if action == "CALL_REASONING":
            used_reasoning = True
            reasoning = reasoning_agent(question, context)
            messages.append({"role": "system", "content": f"Reasoning:\n{reasoning}"})

    # 理论兜底（基本不会走到）
    manager_tokenizer.response_schema = MANAGER_SCHEMA_FULL
    trace = "|".join(actions_taken + ["ANSWER_MAYBE"])
    return f"maybe || ACTIONS={trace}", actions_taken + ["ANSWER_MAYBE"]

# =========================
# rollout_func：batch 版本（给 GRPOTrainer）
# =========================
def explicit_rollout_batch(model, tokenizer, prompts, **kwargs):
    questions = kwargs.get("question", None)
    contexts = kwargs.get("context", None)

    completions = []
    for i, prompt in enumerate(prompts):
        q = questions[i] if questions is not None else ""
        c = contexts[i] if contexts is not None else ""
        completion, _trace = explicit_rollout_one(prompt, q, c)
        completions.append(completion)

    return completions

# =========================
# Reward functions
# =========================
def correctness_reward(completions, **kwargs):
    ground_truth = kwargs["ground_truth"]
    rewards = []
    for c, gt in zip(completions, ground_truth):
        gt = str(gt).strip().lower()
        pred = c.split("||")[0].strip().lower()
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards

def agent_cost_reward(completions, **kwargs):
    return [0.0 for _ in completions]

# =========================
# Main
# =========================
def main():
    wandb.login(key="422b2738db53806489567fb2db277d201112125e")

    wandb.init(
        project="pubmedqa-grpo_1",
        name="manager_reasoning_bf16_0.5B_1",
        config={
            "manager_model": MANAGER_MODEL_NAME,
            "reasoning_model": REASONING_MODEL_NAME,
            "max_completion_length": 16,
            "temperature": 0.7,
            "num_generations": 4,
            "beta": 0.01,
            "scale_rewards": "group",
            "bf16": (DEVICE == "cuda"),
            "seed": SEED,
        },
    )

    dataset = load_data(DATA_PATH)
    splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    train_dataset = splits["train"].map(preprocess)
    test_dataset = splits["test"].map(preprocess)

    grpo_args = GRPOConfig(
        max_completion_length=16,
        temperature=0.7,
        num_generations=4,
        fp16=False,
        bf16=(DEVICE == "cuda"),
        beta=0.01,
        scale_rewards="group",
        report_to=["wandb"],
    )

    trainer = GRPOTrainer(
        model=manager_model,                 # ✅ manager 0.5B 训练
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[correctness_reward, agent_cost_reward],
        rollout_func=explicit_rollout_batch, # ✅ rollout 内部会调用冻结 7B reasoning
    )

    trainer.train()

    trainer.model.save_pretrained(SAVE_PATH)
    manager_tokenizer.save_pretrained(SAVE_PATH)

    with open(TEST_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_dataset.to_list(), f, indent=2, ensure_ascii=False)

    print(f"Training done. Test set saved to {TEST_SAVE_PATH}")
    wandb.finish()

if __name__ == "__main__":
    main()
