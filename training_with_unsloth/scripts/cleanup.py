#!/usr/bin/env python3
"""
Cleanup outputs/logs under training_with_unsloth only.
"""

from pathlib import Path
import shutil


def rm(path: Path):
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted: {path}")


def main():
    base = Path("/home/hula0401/learning_llm/training_with_unsloth")
    rm(base / "output")
    rm(base / "logs")
    (base / "output").mkdir(exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()



