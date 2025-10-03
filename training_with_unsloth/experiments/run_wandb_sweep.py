#!/usr/bin/env python3
"""
Launch a Weights & Biases sweep for GRPO on Qwen3-4B.
It creates a sweep, then runs agents that call the existing run_experiment.py.
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile
import yaml
import wandb


BASE_DIR = Path("/home/hula0401/learning_llm/training_with_unsloth")
EXPS_DIR = BASE_DIR / "experiment_configs"
RUN_SCRIPT = BASE_DIR / "experiments" / "run_experiment.py"
PROJECT = "grpo-qwen3-experiments"


def build_temp_config(base_cfg_path: Path, overrides: dict) -> Path:
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides from sweep params
    for k, v in overrides.items():
        # Map wandb param names to our config keys
        if k == "learning_rate":
            cfg["learning_rate"] = v
        elif k == "gradient_accumulation_steps":
            cfg["gradient_accumulation_steps"] = int(v)
        elif k == "max_prompt_length":
            cfg["max_prompt_length"] = int(v)
        elif k == "max_generate_length":
            cfg["max_generate_length"] = int(v)

    # Ensure fixed settings per user request
    cfg["num_generations"] = 2
    cfg["batch_size"] = 1

    # Write a temp config file
    tmp = tempfile.NamedTemporaryFile(prefix="sweep_cfg_", suffix=".yaml", delete=False, dir=str(EXPS_DIR))
    with open(tmp.name, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return Path(tmp.name)


def train_one():
    # Start a run to access wandb.config
    run = wandb.init(project=PROJECT)
    try:
        # Base config to start from
        base_cfg = EXPS_DIR / "qwen3_grpo_bs1_lr1e-5_ng2_ga4_seed42_rmk.yaml"
        params = dict(wandb.config)
        tmp_cfg = build_temp_config(base_cfg, params)

        # Respect HF token if present in env file
        env_file = Path("/home/hula0401/learning_llm/huggingface_token.env")
        if env_file.exists():
            try:
                env_vars = {}
                for line in env_file.read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip().strip('"\'')
                        v = v.strip().strip('"\'')
                        if k in ("HF_TOKEN", "huggingface_token") and v:
                            env_vars["HF_TOKEN"] = v
                os.environ.update(env_vars)
            except Exception:
                pass

        cmd = [
            "uv", "run", "python",
            str(RUN_SCRIPT),
            str(tmp_cfg),
        ]
        subprocess.run(cmd, check=False)
    finally:
        run.finish()


def main():
    # Sweep config
    sweep_config = {
        "name": "qwen3_4b_grpo_sweep",
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]},
            "gradient_accumulation_steps": {"values": [4, 8, 16]},
            "max_prompt_length": {"values": [256]},
            "max_generate_length": {"values": [256]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3},
    }

    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    # Launch a few runs; increase count as needed
    wandb.agent(sweep_id, function=train_one, count=5)


if __name__ == "__main__":
    main()


