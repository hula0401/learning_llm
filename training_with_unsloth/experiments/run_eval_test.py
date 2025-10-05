#!/usr/bin/env python3
"""
Test script: Train on N samples, evaluate on M test samples every epoch.
Logs correctness@5 and correctness@10 to WandB.
Uses YAML config file for all parameters.

Usage:
    python run_eval_test.py <config.yaml>
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from pathlib import Path

sys.path.append("/home/hula0401/learning_llm")

from training_with_unsloth.framework.config import ExperimentConfig
from training_with_unsloth.framework.grpo_trainer import GRPOTrainer, GRPOArguments
from training_with_unsloth.framework.dataset import create_gsm8k_dataset_split
from training_with_unsloth.rewards.reward_functions import (
    correctness_reward,
    digit_reward,
    hard_format_reward,
    mark_reward,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    args_ns = parser.parse_args()
    
    # Load config
    print(f"Loading config: {args_ns.config}")
    cfg = ExperimentConfig.from_yaml(args_ns.config)
    cfg.materialize_paths()
    
    # Use train_samples and test_samples if provided, otherwise fallback
    train_samples = cfg.train_samples if cfg.train_samples else 20
    test_samples = cfg.test_samples if cfg.test_samples else 10
    
    print(f"Config: train={train_samples}, test={test_samples}, epochs={cfg.num_epochs}")
    
    # Set seed
    set_seed(cfg.random_seed)
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Initialize WandB with full config
    if cfg.use_wandb:
        lr_str = f"{float(cfg.learning_rate):.0e}".replace("e-0", "e-")
        run_name = f"{cfg.model_type}_grpo_train{train_samples}_test{test_samples}_lr{lr_str}_ep{cfg.num_epochs}_{cfg.remark}"
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=cfg.to_dict(),
            reinit=True
        )
    
    # Load model and tokenizer
    print(f"Loading model: {cfg.model_name}")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_prompt_length + cfg.max_generate_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=cfg.compute_dtype,
            device_map=cfg.device_map,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except Exception as e:
        print(f"Unsloth load failed: {e}, using transformers")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            device_map=cfg.device_map,
            torch_dtype=getattr(torch, cfg.compute_dtype) if hasattr(torch, cfg.compute_dtype) else torch.bfloat16,
        )
    
    # Enable gradient checkpointing
    if cfg.use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        except Exception:
            pass
    
    # Create train/test split
    print(f"Creating dataset: {train_samples} train, {test_samples} test")
    train_dataset, test_dataset = create_gsm8k_dataset_split(
        train_samples=train_samples,
        test_samples=test_samples,
        seed=cfg.dataloader_seed
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Setup trainer arguments (coerce types to handle YAML string values)
    def _to_int(v, default=0):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return default
    
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default
    
    args = GRPOArguments()
    args.lr = _to_float(cfg.learning_rate, 1e-5)
    args.batch_size = _to_int(cfg.batch_size, 1)
    args.epoch = _to_int(cfg.num_epochs, 3)
    args.num_generations = _to_int(cfg.num_generations, 4)
    args.max_prompt_length = _to_int(cfg.max_prompt_length, 256)
    args.max_generate_length = _to_int(cfg.max_generate_length, 256)
    args.clip_eps = _to_float(cfg.clip_eps, 0.2)
    args.beta = _to_float(cfg.beta, 0.0)
    args.eval_temperature = _to_float(cfg.eval_temperature, 0.75)
    args.gradient_accumulation_steps = _to_int(cfg.gradient_accumulation_steps, 4)
    args.output_dir = cfg.output_dir
    args.save_steps = 1000  # Don't save intermediate checkpoints
    
    # Wrap model to no-op .to() for quantized models
    import torch.nn as nn
    class _NoOpToModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def to(self, *args, **kwargs):
            return self
        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)
    
    trainer_model = _NoOpToModel(model)
    
    # Set system prompt
    import training_with_unsloth.framework.grpo_trainer as trainer_module
    if cfg.system_prompt:
        trainer_module.SYSTEM_PROMPT = cfg.system_prompt
    else:
        trainer_module.SYSTEM_PROMPT = (
            "Please answer in English with the following format, keep your answer short:\n"
            "<think> your step-by-step reasoning </think>\n"
            "<answer> the final short answer only </answer>\n"
        )
    
    # Create trainer with eval dataset
    reward_funcs = [
        correctness_reward,
        digit_reward,
        hard_format_reward,
        mark_reward,
    ]
    
    trainer = GRPOTrainer(
        model=trainer_model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Add eval dataset
        tokenizer=tokenizer,
    )
    
    # Train (will evaluate every epoch)
    print("\nStarting training with eval every epoch...")
    trainer.train()
    
    # Final save
    trainer.save_model()
    print(f"\nTraining complete! Model saved to {cfg.output_dir}")
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
