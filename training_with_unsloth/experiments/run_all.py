#!/usr/bin/env python3
"""
Run all experiment configs under training_with_unsloth/experiment_configs
"""

from pathlib import Path
import subprocess
import sys


def main():
    base = Path("/home/hula0401/learning_llm/training_with_unsloth/experiment_configs")
    configs = sorted(base.glob("*.yaml"))
    if not configs:
        print("No configs found. Generate with: uv run python training_with_unsloth/experiments/generate_configs.py")
        sys.exit(1)

    for cfg in configs:
        print("=" * 80)
        print(f"Running {cfg.name}")
        print("=" * 80)
        cmd = [
            "uv", "run", "python",
            "/home/hula0401/learning_llm/training_with_unsloth/experiments/run_experiment.py",
            str(cfg),
        ]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()



