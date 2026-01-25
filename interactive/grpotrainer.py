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

logger = LoggerFactory.get_logger("grpo_trainer")

def fmt_tokens(n: int) -> str:
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)

def schedule_next_ckpt(words_so_far: int) -> int:
    if words_so_far < 1_000_000: step = 100_000
    elif words_so_far < 2_000_000: step = 200_000
    elif words_so_far < 10_000_000: step = 1_000_000
    elif words_so_far < 100_000_000: step = 10_000_000
    else: step = 100_000_000
    return ((words_so_far // step) + 1) * step

class CustomGRPOTrainer:
    """
    Standalone GRPO Trainer.
    """
    def __init__(
        self,
        num_generations, 
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
        
        self.num_generations = num_generations
        self.beta = beta
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        
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
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.word_budget = word_budget
        self.gen_word_budget = gen_word_budget
        self._prompt_charged = False 
        self.push_to_hub = push_to_hub
        self.batch_logs = []
        self.generated_log = []

        teacher_config_path = "config/teacher.yaml"
        try:
            with open(teacher_config_path, "r") as f:
                teacher_cfg = yaml.safe_load(f)
            teacher_seed = teacher_cfg.get("seed", None)
        except Exception:
            teacher_seed = None

        self.api = HfApi()

        base_name = config.model_name.split("/")[-1]
        if hasattr(config, "revision_name") and config.revision_name:
            base_name = f"{base_name}_{config.revision_name}"
            
        name_with_budget = f"{base_name}_grpo-{fmt_tokens(word_budget)}"
        if teacher_seed is not None:
            name_with_budget = f"{name_with_budget}-seed{teacher_seed}"
        self.repo_id = f"{hf_org}/{name_with_budget}" if hf_org else name_with_budget

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        model_dir_name = f"{name_with_budget}__{timestamp}"
        
        if save_base_dir is None: save_base_dir = "saved_models"
        self.base_out_dir = os.path.join(save_base_dir, model_dir_name)
        self.meta_dir = os.path.join(self.base_out_dir, "meta_data") if save_meta_dir is None else save_meta_dir

        logger.info(f"Saving to: {self.meta_dir}")
        os.makedirs(self.meta_dir, exist_ok=True)

        self.wandb_enabled = wandb_project is not None
        if self.wandb_enabled:
            run_name = (hf_base_repo or config.model_name).replace("/", "-")
            wandb.init(
                project=wandb_project,
                name=run_name,
                tags=wandb_tags,
                config={"word_budget": word_budget, "gen_word_budget": gen_word_budget, **config.__dict__},
            )
            if watch_model:
                wandb.watch(self.model, log="all", log_freq=1_000)

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
            padded = self.tokenizer.pad(
                {"input_ids": queries}, 
                padding=True, 
                return_tensors="pt"
            ).to(self.model.device)
            input_ids = padded["input_ids"]
            attention_mask = padded["attention_mask"]
        else:
            input_ids = queries.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
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

    # THE CORE GRPO LOGIC

    def grpo_step(self, queries, responses, rewards):
        """
        Perform a single GRPO update step.
        """
        # 1. Prepare Inputs (Concatenate Q + R)
        device = self.model.device
        full_sequences = []
        
        for q, r in zip(queries, responses):
            # Move individual tensors to GPU
            q_device = q.to(device)
            r_device = r.to(device)
            full_sequences.append(torch.cat([q_device, r_device]))
        
        # Pad inputs
        padded_inputs = self.tokenizer.pad(
            {"input_ids": full_sequences}, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]

        # 2. Compute Advantages
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        rewards_tensor = rewards_tensor.view(-1, self.num_generations)
        
        mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)
        std_rewards = rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_tensor - mean_rewards) / std_rewards
        advantages = advantages.view(-1) 

        # 3. Forward Pass (Student)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 

        # 4. Forward Pass (Reference)
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        # 5. Compute Loss
        logprobs = logits.log_softmax(dim=-1)
        ref_logprobs = ref_logits.log_softmax(dim=-1)
        
        loss_list = []
        
        for i in range(len(queries)):
            query_len = len(queries[i])
            
            valid_labels = input_ids[i, query_len:]
            
            valid_logprobs = logprobs[i, query_len-1 : -1]
            valid_ref_logprobs = ref_logprobs[i, query_len-1 : -1]

            # Slice to match label length
            seq_len = min(len(valid_logprobs), len(valid_labels))
            valid_logprobs = valid_logprobs[:seq_len]
            valid_ref_logprobs = valid_ref_logprobs[:seq_len]
            valid_labels = valid_labels[:seq_len]

            token_logprobs = torch.gather(valid_logprobs, 1, valid_labels.unsqueeze(-1)).squeeze(-1)
            ref_token_logprobs = torch.gather(valid_ref_logprobs, 1, valid_labels.unsqueeze(-1)).squeeze(-1)

            # GRPO Loss
            per_token_kl = torch.exp(ref_token_logprobs - token_logprobs) - (ref_token_logprobs - token_logprobs) - 1
            policy_loss = -1 * advantages[i] * token_logprobs.sum()
            kl_loss = self.beta * per_token_kl.sum()
            
            loss_list.append(policy_loss + kl_loss)

        total_loss = torch.stack(loss_list).mean()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(), 
            "mean_advantage": advantages.mean().item(),
        }

    def run_training_loop(self, num_epochs=1):
        logger.info("Start GRPO training...")
        
        global_step = 0
        total_prompt_words = 0
        next_ckpt = schedule_next_ckpt(0)
        jsonl_path = os.path.join(self.base_out_dir, "detailed_logs.jsonl")

        for epoch in range(num_epochs):
            prompt_used = 0
            gen_used = 0
            logger.info("Epoch %d/%d …", epoch + 1, num_epochs)

            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"epoch {epoch+1}")):
                try:
                    start = time.time()
                    if prompt_used >= self.word_budget or gen_used >= self.gen_word_budget: break

                    queries_raw = batch["input_ids"]
                    expanded_queries = []
                    for q in queries_raw:
                        for _ in range(self.num_generations):
                            expanded_queries.append(q)

                    queries_ready = time.time()
                    
                    gens = self.generate(expanded_queries, **self.gen_kwargs)
                    gens_ready = time.time()
                    
                    resp_only = []
                    for g, q in zip(gens, expanded_queries):
                        q_gpu = q.to(g.device) 
                        resp_only.append(g[len(q_gpu):])

                    dec_resp = [self.tokenizer.decode(r) for r in resp_only]
                    
                    expanded_prompts = [self.tokenizer.decode(q) for q in expanded_queries]
                    pairs = [PromptCompletionPair(q, q + r) for q, r in zip(expanded_prompts, dec_resp)]

                    rewards_dict = self.reward_fn(pairs)
                    teacher_rewards = rewards_dict["rewards"]
                    
                    # GRPO Step
                    stats = self.grpo_step(expanded_queries, resp_only, teacher_rewards)
                    
                    step_ready = time.time()

                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        for k in range(min(self.num_generations, len(dec_resp))):
                            log_entry = {
                                "step": global_step,
                                "prompt": expanded_prompts[k],
                                "response": dec_resp[k],
                                "reward": float(teacher_rewards[k]),
                            }
                            f.write(json.dumps(log_entry) + "\n")
                    
                    length_bonuses = [0.0] * len(dec_resp) 
                    self._log_batch(teacher_rewards, stats, teacher_rewards, length_bonuses, 0, 0, global_step)
                    
                    global_step += 1
                    
                except Exception as e:
                    logger.exception(f"Error: {e}")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue

        self._push_final()