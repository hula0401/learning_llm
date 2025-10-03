#!/usr/bin/env python3
"""
Run a single experiment from YAML/JSON config using uv.
This wraps the existing grpo/ trainer without changing it.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import random
import numpy as np
import torch

sys.path.append("/home/hula0401/learning_llm")

from training_with_unsloth.framework.config import ExperimentConfig
from training_with_unsloth.framework.grpo_trainer import GRPOTrainer, GRPOArguments
from training_with_unsloth.rewards.reward_functions import (
    correctness_reward,
    digit_reward,
    hard_format_reward,
    mark_reward,
)
import training_with_unsloth.framework.grpo_trainer as new_trainer_module
import wandb
from training_with_unsloth.framework.dataset import create_gsm8k_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub.errors import RepositoryNotFoundError


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> ExperimentConfig:
    if path.endswith(".yaml") or path.endswith(".yml"):
        return ExperimentConfig.from_yaml(path)
    if path.endswith(".json"):
        return ExperimentConfig.from_json(path)
    raise ValueError(f"Unsupported config format: {path}")


def validate_model_selection(cfg: ExperimentConfig):
    """Enforce canonical model names for given model_type."""
    enforced_map = {
        "qwen3": ["Qwen/Qwen3-4B-Instruct-2507"],
    }
    allowed = enforced_map.get(getattr(cfg, "model_type", ""), None)
    if allowed is None:
        return
    model_name = getattr(cfg, "model_name", None)
    if not model_name:
        raise ValueError("model_name must be provided in config for model_type='qwen3'.")
    if model_name not in allowed:
        raise ValueError(
            f"Invalid model_name for model_type='{cfg.model_type}'. Expected one of {allowed}, got '{model_name}'."
        )


def build_legacy_args(cfg: ExperimentConfig) -> GRPOArguments:
    args = GRPOArguments()
    # Coerce potential string values from YAML to numeric types
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

    args.lr = _to_float(getattr(cfg, "learning_rate", 1e-6), 1e-6)
    args.batch_size = _to_int(getattr(cfg, "batch_size", 1), 1)
    args.epoch = _to_int(getattr(cfg, "num_epochs", 1), 1)
    args.num_generations = _to_int(getattr(cfg, "num_generations", 1), 1)
    args.max_prompt_length = _to_int(getattr(cfg, "max_prompt_length", 128), 128)
    args.max_generate_length = _to_int(getattr(cfg, "max_generate_length", 128), 128)
    args.clip_eps = _to_float(getattr(cfg, "clip_eps", 0.2), 0.2)
    # Disable reference model for quantized runs to avoid deepcopy/device issues
    args.beta = 0.0
    args.gradient_accumulation_steps = _to_int(getattr(cfg, "gradient_accumulation_steps", 1), 1)
    args.output_dir = getattr(cfg, "output_dir", "./output")
    return args


def create_dataset(cfg: ExperimentConfig, tokenizer):
    # Use our wrapper to provide prompt/answer items expected by legacy trainer methods
    return create_gsm8k_dataset(max_samples=cfg.max_samples, seed=cfg.dataloader_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to experiment config")
    args_ns = parser.parse_args()

    print(f"Config path: {args_ns.config}")
    cfg = load_config(args_ns.config)

    # Enforce expected model for given model_type
    try:
        validate_model_selection(cfg)
    except Exception as e:
        print(f"❌ Model selection error: {e}")
        sys.exit(2)

    cfg.materialize_paths()

    # Reproducibility
    set_seed(cfg.random_seed)

    # IO
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logs_dir).mkdir(parents=True, exist_ok=True)

    # Save resolved config into output
    cfg_out = Path(cfg.output_dir) / "experiment_config.yaml"
    cfg.to_yaml(str(cfg_out))

    # Initialize WandB with informative run name
    if cfg.use_wandb:
        run_name = cfg.run_name()
        wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.to_dict(), reinit=True)

    # Build trainer pieces
    print(f"Model name: {cfg.model_name}")
    if not cfg.model_name:
        print("❌ No model_name provided in config. Please set ExperimentConfig.model_name to a valid HF repo.")
        if cfg.use_wandb:
            wandb.finish()
        sys.exit(2)
    # Reduce fragmentation risk
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    chosen_model_name = cfg.model_name
    token_kw = {}
    hf_token = getattr(cfg, "hf_token", None) or os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try to read from env file if present
        env_file = Path("/home/hula0401/learning_llm/huggingface_token.env")
        if env_file.exists():
            try:
                for line in env_file.read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip().strip('"\'')
                        v = v.strip().strip('"\'')
                        if k in ("HF_TOKEN", "huggingface_token") and v:
                            hf_token = v
                            os.environ["HF_TOKEN"] = hf_token
                            break
            except Exception:
                pass
    if hf_token:
        token_kw["token"] = hf_token
    try:
        tokenizer = AutoTokenizer.from_pretrained(chosen_model_name, trust_remote_code=True, **token_kw)
    except Exception as e:
        print(f"Tokenizer load failed for {chosen_model_name}: {e}")
        print("❌ Aborting: cannot load tokenizer for specified model.")
        if cfg.use_wandb:
            wandb.finish()
        sys.exit(3)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load optional system prompt (from config or file)
    system_prompt = getattr(cfg, "system_prompt", None)
    if not system_prompt and getattr(cfg, "system_prompt_path", None):
        try:
            system_prompt = Path(cfg.system_prompt_path).read_text().strip()
        except Exception:
            system_prompt = None

    # Fallback simple chat template if model/tokenizer doesn't define one
    def _simple_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        user_msgs = [m for m in messages if m.get("role") == "user"]
        content = user_msgs[-1]["content"] if user_msgs else ""
        return f"Question: {content}\nAnswer:"
    try:
        _ = tokenizer.get_chat_template()
    except Exception:
        # Older versions: if chat template not present, override apply_chat_template
        setattr(tokenizer, "apply_chat_template", _simple_chat_template)

    # Prefer Unsloth loader when available
    use_unsloth = True
    model = None
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            max_seq_length = int(getattr(cfg, "max_prompt_length", 128) + getattr(cfg, "max_generate_length", 128))
            # Unsloth 4-bit load
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=chosen_model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=getattr(cfg, "load_in_4bit", True),
                dtype="bfloat16",
                device_map=getattr(cfg, "device_map", "auto"),
            )
            # Ensure left padding after Unsloth may return a new tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        except Exception as e:
            print(f"Unsloth load failed, falling back to transformers: {e}")
            use_unsloth = False
    if not use_unsloth:
        quant_cfg = None
        if getattr(cfg, "load_in_4bit", True):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model_kwargs = dict(
            trust_remote_code=True,
            device_map=getattr(cfg, "device_map", "auto"),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
        # Use eager attention to avoid SDPA/Triton issues
        model_kwargs["attn_implementation"] = "eager"
        try:
            model = AutoModelForCausalLM.from_pretrained(chosen_model_name, **model_kwargs)
        except RepositoryNotFoundError as e:
            print(f"Model load failed for {chosen_model_name}: {e}")
            print("❌ Aborting: cannot load specified model.")
            if cfg.use_wandb:
                wandb.finish()
            sys.exit(4)
        # Re-assert left padding with transformers path
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if getattr(cfg, "use_gradient_checkpointing", True):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        # Recommended with gradient checkpointing
        try:
            model.config.use_cache = False
        except Exception:
            pass
    dataset = create_dataset(cfg, tokenizer)
    legacy_args = build_legacy_args(cfg)

    # Wrap model to no-op .to() for quantized models; delegate calls
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

    # Ensure SYSTEM_PROMPT exists for trainer (default or config override)
    if not hasattr(new_trainer_module, "SYSTEM_PROMPT"):
        default_prompt = (
            "Please answer in English with the following format, keep your answer short:\n"
            "<think> your step-by-step reasoning </think>\n"
            "<answer> the final short answer only </answer>\n"
        )
        new_trainer_module.SYSTEM_PROMPT = cfg.system_prompt or default_prompt

    reward_funcs = [
        correctness_reward,
        digit_reward,
        hard_format_reward,
        mark_reward,
    ]
    trainer = GRPOTrainer(
        model=trainer_model,
        reward_funcs=reward_funcs,
        args=legacy_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Run
    trainer.train()

    # Print VRAM usage
    try:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"VRAM: used={used:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB")
    except Exception:
        pass
    trainer.save_model()

    # Write minimal history file placeholder
    with open(Path(cfg.output_dir) / "training_history.json", "w") as f:
        json.dump({"status": "completed"}, f)

    print(f"Done: {cfg.run_name()}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


