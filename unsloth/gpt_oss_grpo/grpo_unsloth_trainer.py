#!/usr/bin/env python3
"""
GRPO Training with Unsloth for NVIDIA 4080 Super (16GB VRAM)
Optimized for 20B parameter models using vLLM and Unsloth's memory optimizations
Based on the Colab notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
"""

# Import unsloth first as recommended
try:
    import unsloth
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from unsloth.model_max_length import get_model_max_length
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Warning: Unsloth not available. Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    UNSLOTH_AVAILABLE = False
    is_bfloat16_supported = lambda: False

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Any, Dict
import wandb
import os
import json
import numpy as np
from tqdm import tqdm
import gc
from copy import deepcopy
import warnings
import sys
warnings.filterwarnings("ignore")

# Import our custom monitoring
from wandb_monitor import WandbMonitor

# Import vLLM for efficient inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")

@dataclass
class UnslothGRPOConfig:
    """Configuration for Unsloth GRPO training optimized for 16GB VRAM"""
    
    # Model configuration
    model_name: str = "unsloth/gpt-2"  # Start with smaller model, can be changed to 20B
    max_seq_length: int = 2048  # Reduced for memory efficiency
    dtype: str = "bfloat16" if is_bfloat16_supported() else "float16"
    load_in_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    load_in_8bit: bool = False
    
    # GRPO specific parameters
    num_generations: int = 4  # Number of responses per prompt
    max_prompt_length: int = 512
    max_generate_length: int = 1024
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 1  # Small batch size for memory efficiency
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    save_steps: int = 100
    logging_steps: int = 10
    
    # GRPO algorithm parameters
    beta: float = 0.1  # KL divergence coefficient
    clip_eps: float = 0.2  # PPO clipping parameter
    reward_weights: List[float] = None
    
    # Memory optimization
    use_vllm: bool = True  # Use vLLM for generation if available
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Output
    output_dir: str = "./output_unsloth_grpo"
    logging_dir: str = "./logs_unsloth_grpo"


class UnslothGRPODataset(Dataset):
    """Dataset class for GRPO training with Unsloth"""
    
    def __init__(self, data_path: str, tokenizer, max_samples: int = 1000):
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path, max_samples)
    
    def _load_data(self, data_path: str, max_samples: int):
        """Load and preprocess data"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Assume it's a directory with JSON files
            data = []
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                        data.extend(json.load(f))
        
        return data[:max_samples]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item.get('prompt', ''),
            'answer': item.get('answer', ''),
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
        }


class UnslothGRPOTrainer:
    """GRPO Trainer optimized for Unsloth and 16GB VRAM"""
    
    def __init__(self, config: UnslothGRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not UNSLOTH_AVAILABLE:
            print("Warning: Unsloth not available. Using fallback implementation for testing.")
            self._setup_model_fallback()
        else:
            # Initialize model and tokenizer with Unsloth optimizations
            self._setup_model()
        
        self._setup_optimizer()
        
        # Initialize vLLM for efficient generation if available
        self.vllm_model = None
        if self.config.use_vllm and VLLM_AVAILABLE:
            self._setup_vllm()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize Wandb monitoring
        self.monitor = WandbMonitor(
            project_name="grpo-unsloth-20b",
            config=config.__dict__
        )
        
    def _setup_model(self):
        """Setup model with Unsloth optimizations"""
        print("Setting up model with Unsloth optimizations...")
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
        )
        
        # Apply Unsloth optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Set up chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml",  # or "zephyr", "llama-2", etc.
        )
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        print(f"Model loaded with max sequence length: {get_model_max_length(self.model)}")
    
    def _setup_model_fallback(self):
        """Fallback model setup without Unsloth for testing"""
        print("Setting up model with fallback implementation...")
        
        # Use a smaller model for testing
        model_name = "gpt2"  # Use GPT-2 as fallback
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        print(f"Fallback model loaded: {model_name}")
        
    def _setup_vllm(self):
        """Setup vLLM for efficient generation"""
        if not VLLM_AVAILABLE:
            print("vLLM not available, using standard generation")
            return
            
        print("Setting up vLLM for efficient generation...")
        
        # vLLM configuration for 16GB VRAM
        vllm_config = {
            "model": self.config.model_name,
            "tensor_parallel_size": 1,  # Single GPU
            "gpu_memory_utilization": 0.85,  # Use 85% of VRAM
            "max_model_len": self.config.max_seq_length,
            "dtype": self.config.dtype,
            "quantization": "awq" if self.config.load_in_4bit else None,
            "trust_remote_code": True,
        }
        
        try:
            self.vllm_model = LLM(**vllm_config)
            print("vLLM model loaded successfully")
        except Exception as e:
            print(f"Failed to load vLLM model: {e}")
            self.vllm_model = None
    
    def _setup_optimizer(self):
        """Setup optimizer with memory-efficient settings"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
    def generate_responses(self, prompts: List[str], use_vllm: bool = True) -> List[str]:
        """Generate multiple responses for each prompt"""
        if use_vllm and self.vllm_model is not None:
            return self._generate_with_vllm(prompts)
        else:
            return self._generate_with_model(prompts)
    
    def _generate_with_vllm(self, prompts: List[str]) -> List[str]:
        """Generate responses using vLLM for efficiency"""
        # Prepare prompts for vLLM
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Think step by step and provide clear answers."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_generate_length,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        outputs = self.vllm_model.generate(formatted_prompts, sampling_params)
        
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text)
        
        return responses
    
    def _generate_with_model(self, prompts: List[str]) -> List[str]:
        """Generate responses using the main model"""
        self.model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Simple prompt formatting for fallback
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    try:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant. Think step by step and provide clear answers."},
                            {"role": "user", "content": prompt}
                        ]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except:
                        # Fallback to simple prompt
                        formatted_prompt = f"Question: {prompt}\nAnswer:"
                else:
                    # Simple prompt for models without chat template
                    formatted_prompt = f"Question: {prompt}\nAnswer:"
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    max_length=self.config.max_prompt_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_generate_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    generated_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
        
        return responses
    
    def compute_rewards(self, prompts: List[str], responses: List[str], answers: List[str]) -> torch.Tensor:
        """Compute rewards for responses using multiple reward functions"""
        # Simple reward functions for demonstration
        # In practice, you would use more sophisticated reward models
        
        rewards = []
        for prompt, response, answer in zip(prompts, responses, answers):
            reward = 0.0
            
            # Length reward (prefer longer responses)
            reward += min(len(response.split()) / 100, 1.0) * 0.1
            
            # Answer presence reward
            if answer and answer.lower() in response.lower():
                reward += 0.5
            
            # Format reward (prefer structured responses)
            if "<think>" in response and "<answer>" in response:
                reward += 0.3
            
            # Coherence reward (simple keyword matching)
            if any(word in response.lower() for word in ["because", "therefore", "thus", "so"]):
                reward += 0.1
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using group-based normalization"""
        # Group-based advantage computation (GRPO)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        if std_reward == 0:
            std_reward = torch.tensor(1.0, device=rewards.device)
        
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        return advantages
    
    def compute_grpo_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                         advantages: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute GRPO loss with PPO clipping"""
        
        # Compute importance ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
        
        # Compute losses
        loss1 = ratio * advantages.unsqueeze(-1)
        loss2 = clipped_ratio * advantages.unsqueeze(-1)
        
        # Take minimum and apply attention mask
        policy_loss = -torch.min(loss1, loss2) * attention_mask
        
        # Average over valid tokens
        valid_tokens = attention_mask.sum(dim=-1, keepdim=True)
        valid_tokens = torch.clamp(valid_tokens, min=1)
        
        policy_loss = policy_loss.sum(dim=-1) / valid_tokens.squeeze(-1)
        
        return policy_loss.mean()
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        prompts = batch['prompt']
        answers = batch['answer']
        
        # Generate multiple responses for each prompt
        all_responses = []
        for prompt in prompts:
            responses = self.generate_responses([prompt] * self.config.num_generations)
            all_responses.extend(responses)
        
        # Compute rewards
        extended_prompts = [p for p in prompts for _ in range(self.config.num_generations)]
        extended_answers = [a for a in answers for _ in range(self.config.num_generations)]
        rewards = self.compute_rewards(extended_prompts, all_responses, extended_answers)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Prepare inputs for loss computation
        # This is a simplified version - in practice, you'd need to tokenize and compute log probs
        # For now, we'll use dummy values
        batch_size = len(extended_prompts)
        seq_len = self.config.max_generate_length
        
        # Dummy log probabilities (in practice, compute from model)
        # Make sure they require gradients for training
        log_probs = torch.randn(batch_size, seq_len, device=self.device, requires_grad=True) * 0.1
        old_log_probs = log_probs.detach()
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        # Compute GRPO loss
        loss = self.compute_grpo_loss(log_probs, old_log_probs, advantages, attention_mask)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Prepare metrics for logging
        metrics = {
            "loss": loss.item(),
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "rewards_min": rewards.min().item(),
            "rewards_max": rewards.max().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        # Log to wandb
        self.monitor.log_training_metrics(metrics, self.global_step)
        self.monitor.log_memory_usage(self.global_step)
        self.monitor.log_gpu_utilization(self.global_step)
        
        # Log sample generations every 10 steps
        if self.global_step % 10 == 0:
            sample_prompts = prompts[:2]  # Log first 2 prompts
            sample_responses = all_responses[:2 * self.config.num_generations]
            sample_rewards = rewards[:2 * self.config.num_generations]
            self.monitor.log_generation_samples(
                sample_prompts, sample_responses, sample_rewards.tolist(), self.global_step
            )
        
        return metrics
    
    def train(self, train_dataset: UnslothGRPODataset):
        """Main training loop"""
        print("Starting GRPO training with Unsloth...")
        
        # Log system information
        self.monitor.log_system_info()
        self.monitor.log_configuration_summary(self.config.__dict__)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: {key: [item[key] for item in x] for key in x[0].keys()}
        )
        
        total_steps = len(dataloader) * self.config.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.monitor.update_epoch(epoch)
            
            for step, batch in enumerate(dataloader):
                # Training step
                metrics = self.train_step(batch)
                
                # Log metrics to console
                if step % self.config.logging_steps == 0:
                    print(f"Epoch {epoch}, Step {step}: Loss={metrics['loss']:.4f}, "
                          f"Rewards={metrics['rewards_mean']:.4f}, "
                          f"Advantages={metrics['advantages_mean']:.4f}")
                
                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    self.monitor.update_step(self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                    # Log model artifacts
                    checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                    self.monitor.log_model_artifacts(checkpoint_dir, self.global_step)
                
                progress_bar.update(1)
                
                # Memory cleanup
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        progress_bar.close()
        print("Training completed!")
        
        # Save final model
        self.save_model()
        
        # Log final model artifacts
        self.monitor.log_model_artifacts(self.config.output_dir, self.global_step)
        
        # Finish monitoring
        self.monitor.finish()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_model(self):
        """Save final model"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Final model saved to {self.config.output_dir}")


def main():
    """Main training function"""
    # Configuration optimized for 16GB VRAM
    config = UnslothGRPOConfig(
        model_name="unsloth/gpt-2",  # Start with smaller model
        max_seq_length=2048,
        batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        learning_rate=1e-5,
        output_dir="./output_unsloth_grpo",
    )
    
    # Create trainer
    trainer = UnslothGRPOTrainer(config)
    
    # Create dataset (you'll need to provide your data)
    # For now, create a dummy dataset
    dummy_data = [
        {"prompt": "What is 2+2?", "answer": "4", "instruction": "Solve the math problem"},
        {"prompt": "Explain photosynthesis", "answer": "Process by which plants make food", "instruction": "Explain the concept"},
    ] * 50  # Repeat to create more data
    
    # Save dummy data
    os.makedirs("./data", exist_ok=True)
    with open("./data/dummy_data.json", "w") as f:
        json.dump(dummy_data, f, indent=2)
    
    # Create dataset
    dataset = UnslothGRPODataset("./data/dummy_data.json", trainer.tokenizer)
    
    # Start training
    trainer.train(dataset)


if __name__ == "__main__":
    main()
