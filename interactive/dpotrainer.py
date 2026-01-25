import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW
from interactive.utils import LoggerFactory, PromptCompletionPair
import os
import csv
import time
import statistics
import shutil
import yaml
import json
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import HfApi
import wandb

logger = LoggerFactory.get_logger("dpo_trainer")

def fmt_tokens(n: int) -> str:
    if n >= 1_000: return f"{n // 1_000}K"
    return str(n)

def schedule_next_ckpt(words_so_far: int) -> int:
    if words_so_far < 1_000_000: step = 100_000
    else: step = 100_000_000
    return ((words_so_far // step) + 1) * step

class CustomDPOTrainer:
    def __init__(
        self,
        beta,
        config,
        model,
        ref_model,
        tokenizer: AutoTokenizer,
        dataset,
        reward_fn,
        *,
        hf_base_repo: str | None = None,
        hf_org: str | None = None,
        save_meta_dir: str | None = None,  
        word_budget: int = 100_000_000,
        gen_word_budget: int = 100_000_000,
        save_base_dir: str | None = None,
        wandb_project: str | None = None,
        wandb_tags: list[str] | None = None,
        watch_model: bool = False,
        push_to_hub: bool = False,
        **kwargs
    ) -> None:
        
        self.beta = beta
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        
        # We always generate 2 to form a pair
        self.num_generations = 2 
        
        if torch.cuda.is_available():
            self.model.cuda()
            self.ref_model.cuda()

        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        def collate_fn(batch):
            return {
                "input_ids": [torch.tensor(b["input_ids"]) for b in batch],
                "query": [b["query"] for b in batch],
            }
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.word_budget = word_budget
        self.gen_word_budget = gen_word_budget
        self._prompt_charged = False 
        self.push_to_hub = push_to_hub
        self.batch_logs = []
        self.generated_log = []

        self.api = HfApi()
        base_name = config.model_name.split("/")[-1]
        name_with_budget = f"{base_name}_dpo-{fmt_tokens(word_budget)}"
        self.repo_id = f"{hf_org}/{name_with_budget}" if hf_org else name_with_budget

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        model_dir_name = f"{name_with_budget}__{timestamp}"
        
        if save_base_dir is None: save_base_dir = "saved_models"
        self.base_out_dir = os.path.join(save_base_dir, model_dir_name)
        self.meta_dir = os.path.join(self.base_out_dir, "meta_data") if save_meta_dir is None else save_meta_dir
        os.makedirs(self.meta_dir, exist_ok=True)

        self.wandb_enabled = wandb_project is not None
        if self.wandb_enabled:
            wandb.init(project=wandb_project, name=name_with_budget, config=config.__dict__)

        self.gen_kwargs = {
            "max_new_tokens": 64,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

    def generate(self, queries, **kwargs):
        if isinstance(queries, list):
            padded = self.tokenizer.pad({"input_ids": queries}, padding=True, return_tensors="pt").to(self.model.device)
            input_ids = padded["input_ids"]
            attention_mask = padded["attention_mask"]
        else:
            input_ids = queries.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs

    def _push_to_hub(self, branch: str, msg: str):
        if not self.push_to_hub:
            return

        self.api.create_repo(repo_id=self.repo_id, exist_ok=True, repo_type="model")
        if branch != "main":
            self.api.create_branch(repo_id=self.repo_id, branch=branch, exist_ok=True)

        ckpt_dir = os.path.join(self.meta_dir, "_ckpt_upload")
        if os.path.exists(ckpt_dir): shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=ckpt_dir,
            revision=branch,
            commit_message=msg,
        )
        shutil.rmtree(ckpt_dir)
        logger.info("Pushed → %s:%s", self.repo_id, branch)

    def _push_final(self):
        local_save_path = os.path.join(self.base_out_dir, "final_model")
        logger.info(f"Saving final model locally to: {local_save_path}")
        os.makedirs(local_save_path, exist_ok=True)
        self.model.save_pretrained(local_save_path)
        self.tokenizer.save_pretrained(local_save_path)
        logger.info(f"✓ Model saved.")

        if not self.push_to_hub:
            logger.info("push_to_hub=False → skipping upload to Hugging Face Hub.")
            return
    
        logger.info("push_to_hub=True → uploading to Hugging Face Hub...")
        self._push_to_hub("main", "Final model push")

    def _log_batch(self, rewards, stats, teacher_rewards, length_bonuses, prompt_words, gen_words, global_step):
        teacher_rw = [float(r) for r in teacher_rewards]
        rw = [float(r) for r in rewards]
        record = {
            "avg_teacher_reward": statistics.mean(teacher_rw),
            "std_teacher_reward": statistics.stdev(teacher_rw) if len(teacher_rw) > 1 else 0.0,
            "loss": stats.get("loss", 0.0),
            "mean_advantage": stats.get("mean_advantage", 0.0),
            "prompt_words":   prompt_words,
            "gen_words":      gen_words,
            "global_step":    global_step
        }
        self.batch_logs.append(record)
        if self.wandb_enabled:
            wandb.log(record, step=global_step)


    def get_batch_logprobs(self, model, input_ids, attention_mask, response_start_indices):
        """
        Compute the log probability of the RESPONSE part only.
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logprobs = logits.log_softmax(dim=-1)

        # Shift logits/labels
        # Logits at t-1 predict token at t
        logits_shifted = logprobs[:, :-1, :]
        labels_shifted = input_ids[:, 1:]

        # Gather probability of the actual token
        token_logprobs = torch.gather(logits_shifted, 2, labels_shifted.unsqueeze(-1)).squeeze(-1)

        # Mask out query part (we only care about response)
        # We create a mask where 1 = response, 0 = query/padding
        mask = torch.zeros_like(token_logprobs)
        for i, start_idx in enumerate(response_start_indices):
            # start_idx is where response begins.
            # Since we shifted, the logprob for token at `start_idx` is at index `start_idx-1`
            # valid range in shifted tensor: [start_idx-1 : end]
            mask[i, start_idx-1:] = 1.0
            
            # Mask out padding if present (pad token = 0 usually, but check tokenizer)
            # Assuming attention_mask handles padding logic mostly, but let's be safe:
            # (Simple version: just mask query. Padding logprobs don't matter much if consistent)

        # Sum logprobs over the response
        sum_logprobs = (token_logprobs * mask).sum(dim=1)
        return sum_logprobs

    def dpo_step(self, queries, responses, rewards):
        """
        1. Form pairs (Winner vs Loser) based on rewards.
        2. Compute DPO Loss.
        """
        device = self.model.device
        
        # --- 1. Identify Pairs ---
        # We have N*2 inputs. We process them in chunks of 2.
        # Queries: [Q1, Q1, Q2, Q2...]
        # Responses: [R1a, R1b, R2a, R2b...]
        # Rewards: [0.8, 0.2, 0.5, 0.6...]
        
        chosen_input_ids = []
        rejected_input_ids = []
        chosen_rewards = []
        rejected_rewards = []
        response_start_indices = [] # Track where response starts for masking

        valid_pairs = 0

        # Loop through prompts
        batch_size = len(queries) // 2
        for i in range(batch_size):
            idx_a = 2 * i
            idx_b = 2 * i + 1
            
            r_a = rewards[idx_a]
            r_b = rewards[idx_b]
            
            # Tie-breaking: skip if rewards are equal (no clear signal)
            if r_a == r_b:
                continue
                
            if r_a > r_b:
                win_idx, lose_idx = idx_a, idx_b
                chosen_rewards.append(r_a)
                rejected_rewards.append(r_b)
            else:
                win_idx, lose_idx = idx_b, idx_a
                chosen_rewards.append(r_b)
                rejected_rewards.append(r_a)

            # Construct full tensors
            q = queries[win_idx].to(device)
            r_win = responses[win_idx].to(device)
            r_lose = responses[lose_idx].to(device)
            
            chosen_seq = torch.cat([q, r_win])
            rejected_seq = torch.cat([q, r_lose])
            
            chosen_input_ids.append(chosen_seq)
            rejected_input_ids.append(rejected_seq)
            
            # Record start index of response (length of query)
            response_start_indices.append(len(q))
            valid_pairs += 1

        if valid_pairs == 0:
            return {"loss": 0.0, "rewards/chosen": 0.0, "rewards/rejected": 0.0, "rewards/accuracies": 0.0}

        # --- 2. Prepare Batch ---
        # Pad chosen and rejected separately
        pad_token_id = self.tokenizer.pad_token_id
        
        def pad_batch(tensors):
            return self.tokenizer.pad({"input_ids": tensors}, padding=True, return_tensors="pt").to(device)

        chosen_batch = pad_batch(chosen_input_ids)
        rejected_batch = pad_batch(rejected_input_ids)

        # --- 3. Compute Logprobs ---
        # Policy Logprobs
        policy_chosen_logps = self.get_batch_logprobs(
            self.model, chosen_batch["input_ids"], chosen_batch["attention_mask"], response_start_indices
        )
        policy_rejected_logps = self.get_batch_logprobs(
            self.model, rejected_batch["input_ids"], rejected_batch["attention_mask"], response_start_indices
        )

        # Reference Logprobs
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logprobs(
                self.ref_model, chosen_batch["input_ids"], chosen_batch["attention_mask"], response_start_indices
            )
            ref_rejected_logps = self.get_batch_logprobs(
                self.ref_model, rejected_batch["input_ids"], rejected_batch["attention_mask"], response_start_indices
            )

        # --- 4. DPO Loss Calculation ---
        # pi_logratios = policy_chosen - policy_rejected
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = policy_logratios - ref_logratios
        
        # Loss = -log(sigmoid(beta * logits))
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()

        # Stats
        reward_accuracies = (policy_logratios > 0).float().mean() # Does policy prefer chosen?

        # --- 5. Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "rewards/chosen": statistics.mean([float(x) for x in chosen_rewards]),
            "rewards/rejected": statistics.mean([float(x) for x in rejected_rewards]),
            "rewards/accuracies": reward_accuracies.item(),
            "rewards/margins": (chosen_rewards[0] - rejected_rewards[0]) if len(chosen_rewards) > 0 else 0
        }

    def run_training_loop(self, num_epochs=1):
        logger.info("Start DPO training...")
        
        global_step = 0
        next_ckpt = schedule_next_ckpt(0)
        prompt_used = 0
        gen_used = 0
        jsonl_path = os.path.join(self.base_out_dir, "detailed_logs.jsonl")

        for epoch in range(num_epochs):
            for batch in tqdm(self.dataloader, desc="DPO Epoch"):
                try:
                    if prompt_used >= self.word_budget: break

                    # 1. Expand Queries (2x)
                    queries_raw = batch["input_ids"]
                    expanded_queries = []
                    for q in queries_raw:
                        expanded_queries.append(q) # Gen 1
                        expanded_queries.append(q) # Gen 2

                    # 2. Generate
                    gens = self.generate(expanded_queries, **self.gen_kwargs)
                    
                    resp_only = []
                    for g, q in zip(gens, expanded_queries):
                        q_gpu = q.to(g.device)
                        resp_only.append(g[len(q_gpu):])

                    dec_resp = [self.tokenizer.decode(r) for r in resp_only]
                    expanded_prompts = [self.tokenizer.decode(q) for q in expanded_queries]
                    pairs = [PromptCompletionPair(q, q + r) for q, r in zip(expanded_prompts, dec_resp)]

                    # 3. Reward
                    rewards_dict = self.reward_fn(pairs)
                    # Add length bonus logic here if needed (same as GRPO)
                    raw_rewards = [float(r) for r in rewards_dict["rewards"]]
                    
                    # 4. DPO Step
                    stats = self.dpo_step(expanded_queries, resp_only, raw_rewards)
                    
                    # 5. Logging
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        log_entry = {
                            "step": global_step,
                            "prompt": expanded_prompts[0],
                            "response_1": dec_resp[0],
                            "reward_1": raw_rewards[0],
                            "response_2": dec_resp[1],
                            "reward_2": raw_rewards[1],
                            "loss": stats.get("loss", 0.0)
                        }
                        f.write(json.dumps(log_entry) + "\n")
                    
                    if self.wandb_enabled:
                        wandb.log(stats, step=global_step)

                    global_step += 1
                    
                    # Budget tracking (simplified)
                    prompt_used += sum(len(x) for x in expanded_queries) # approximate
                    
                except Exception as e:
                    logger.exception(f"Error: {e}")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue

        # Save final
        final_path = os.path.join(self.base_out_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        self._push_final()