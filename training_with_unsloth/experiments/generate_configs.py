#!/usr/bin/env python3
"""
Generate YAML configs for multiple learning rates with bs=4, epochs=50.
Name format: oss20b_grpo_bs4_lrx_seed42_ep50_rmk<remark>
"""

import os
from pathlib import Path
from typing import List

from training_with_unsloth.framework.config import ExperimentConfig


def main():
    output_dir = Path("/home/hula0401/learning_llm/training_with_unsloth/experiment_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_rates: List[float] = [1e-5]

    created = []
    for lr in learning_rates:
        cfg = ExperimentConfig(
            remark="rmk",  # use literal 'rmk' as suffix
            model_type="qwen3",
            # Target Qwen3 4B Instruct; Unsloth loader will handle 4-bit
            model_name="Qwen/Qwen3-4B-Instruct",
            dataset_name="gsm8k",
            max_samples=50,
            dataloader_seed=42,
            learning_rate=lr,
            batch_size=1,
            num_epochs=50,
            gradient_accumulation_steps=4,
            num_generations=3,
            temperature=0.7,
            # Keep total <= 512 to avoid truncation warnings
            max_prompt_length=256,
            max_generate_length=256,
            use_wandb=True,
            wandb_project="grpo-qwen3-experiments",
            random_seed=42,
        )
        cfg.materialize_paths()
        file_path = output_dir / f"{cfg.run_name()}.yaml"
        cfg.to_yaml(str(file_path))
        created.append(file_path.name)

    index_path = output_dir / "experiments_index.txt"
    with open(index_path, "w") as f:
        f.write("Experiment Configurations Index\n")
        f.write("=" * 80 + "\n\n")
        for name in created:
            f.write(f"- {name}\n")

    print("Generated configs:")
    for name in created:
        print(f"  - {name}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()


