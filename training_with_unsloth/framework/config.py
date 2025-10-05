#!/usr/bin/env python3
"""
Experiment configuration dataclass and YAML/JSON IO utilities.
Naming: oss20b_grpo_bs{batch}_lr{lr}_seed{seed}_ep{epochs}_rmk{remark}
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import yaml


@dataclass
class ExperimentConfig:
    # Identification
    remark: str = ""
    model_type: str = "qwen3"
    model_name: Optional[str] = None

    # Data
    dataset_name: str = "gsm8k"
    dataset_path: Optional[str] = None
    max_samples: int = 50
    train_samples: Optional[int] = None  # For train/test split
    test_samples: Optional[int] = None   # For train/test split
    dataloader_seed: int = 42

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 50
    gradient_accumulation_steps: int = 4

    # Generation/GRPO
    num_generations: int = 3
    max_prompt_length: int = 512
    max_generate_length: int = 512
    temperature: float = 0.7
    eval_temperature: float = 0.75  # Temperature for evaluation sampling (Pass@5/10)
    beta: float = 0.1
    clip_eps: float = 0.2

    # Performance / memory
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    device_map: str = "auto"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "grpo-qwen3-experiments"
    wandb_tags: Optional[List[str]] = None
    hf_token: Optional[str] = None

    # Evaluation
    eval_every_epochs: int = 50

    # Prompting
    system_prompt: Optional[str] = None

    # Prompting
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None

    # IO
    output_dir: Optional[str] = None
    logs_dir: Optional[str] = None

    # Reproducibility
    random_seed: int = 42

    def run_name(self) -> str:
        # Coerce learning_rate to float if it was loaded as a string
        try:
            lr_val = float(self.learning_rate)
            lr_str = f"{lr_val:.0e}".replace("e-0", "e-")
        except Exception:
            lr_str = str(self.learning_rate)
        parts = [
            self.model_type,
            "grpo",
            f"bs{self.batch_size}",
            f"lr{lr_str}",
            f"ng{self.num_generations}",
            f"ga{self.gradient_accumulation_steps}",
            f"seed{self.random_seed}",
        ]
        if self.remark:
            parts.append(self.remark)
        return "_".join(parts)

    def materialize_paths(self):
        name = self.run_name()
        base_dir = "/home/hula0401/learning_llm/training_with_unsloth"
        if not self.output_dir:
            self.output_dir = f"{base_dir}/output/{name}"
        if not self.logs_dir:
            self.logs_dir = f"{base_dir}/logs/{name}"
        if not self.wandb_tags:
            try:
                lr_val = float(self.learning_rate)
                lr_str = f"{lr_val:.0e}".replace("e-0", "e-")
            except Exception:
                lr_str = str(self.learning_rate)
            self.wandb_tags = [
                f"bs{self.batch_size}",
                f"lr{lr_str}",
                f"ng{self.num_generations}",
                f"ga{self.gradient_accumulation_steps}",
                f"seed{self.random_seed}",
                self.model_type,
                self.dataset_name,
            ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        obj = cls(**d)
        obj.materialize_paths()
        return obj

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


