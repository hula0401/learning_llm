#!/usr/bin/env python3
"""
Dataset utilities for experiments.
Provides a GSM8K dataset wrapper returning dicts with 'prompt' and 'answer'.
"""

from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
import random


class SimpleQADataset(Dataset):
    """A simple QA dataset returning prompt/answer pairs."""

    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]


def create_gsm8k_dataset_split(train_samples: int = 20, test_samples: int = 10, seed: int = 42):
    """Create train/test split of GSM8K dataset.
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    from datasets import load_dataset
    import re

    rng = random.Random(seed)

    def extract_final_answer(answer_text: str) -> str:
        """Extract the final numerical answer from GSM8K format."""
        match = re.search(r"####\s*([^\n]+)", answer_text)
        if match:
            return match.group(1).strip()
        return answer_text.strip()

    try:
        ds = load_dataset("gsm8k", "main", split="train")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        
        # Split into test and train
        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:test_samples + train_samples]

        train_items: List[Dict[str, Any]] = []
        test_items: List[Dict[str, Any]] = []
        
        for i in train_indices:
            ex = ds[int(i)]
            prompt = ex.get("question", "")
            raw_answer = ex.get("answer", "")
            answer = extract_final_answer(raw_answer)
            train_items.append({"prompt": prompt, "answer": answer})
        
        for i in test_indices:
            ex = ds[int(i)]
            prompt = ex.get("question", "")
            raw_answer = ex.get("answer", "")
            answer = extract_final_answer(raw_answer)
            test_items.append({"prompt": prompt, "answer": answer})

        return SimpleQADataset(train_items), SimpleQADataset(test_items)
    except Exception as e:
        # Fallback to synthetic data
        base = [
            {"prompt": "小明有15个苹果，他吃了3个，又买了8个。现在他有多少个苹果？", "answer": "20"},
            {"prompt": "一个班级有30个学生，其中60%是女生。女生有多少人？", "answer": "18"},
            {"prompt": "一个圆的半径是5厘米，它的直径是多少厘米？", "answer": "10"},
            {"prompt": "一辆汽车每小时行驶60公里，行驶3小时能走多少公里？", "answer": "180"},
            {"prompt": "一个正方形的边长是8厘米，它的周长是多少厘米？", "answer": "32"},
        ]
        train_items = base[:3]
        test_items = base[3:5]
        return SimpleQADataset(train_items), SimpleQADataset(test_items)


def create_gsm8k_dataset(max_samples: int = 50, seed: int = 42) -> SimpleQADataset:
    """Create a GSM8K dataset subset from HF datasets with deterministic sampling.

    Returns items with keys: 'prompt', 'answer'.
    Answer is extracted from the '#### XXX' format in GSM8K.
    """
    from datasets import load_dataset
    import re

    rng = random.Random(seed)

    def extract_final_answer(answer_text: str) -> str:
        """Extract the final numerical answer from GSM8K format."""
        # GSM8K answers are like: "Step 1...\nStep 2...\n#### 200"
        match = re.search(r"####\s*([^\n]+)", answer_text)
        if match:
            return match.group(1).strip()
        return answer_text.strip()

    try:
        ds = load_dataset("gsm8k", "main", split="train")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:max_samples]

        items: List[Dict[str, Any]] = []
        for i in selected:
            ex = ds[int(i)]
            prompt = ex.get("question", "")
            raw_answer = ex.get("answer", "")
            # Extract just the final numerical answer
            answer = extract_final_answer(raw_answer)
            items.append({"prompt": prompt, "answer": answer})

        return SimpleQADataset(items)
    except Exception as e:
        # Fallback to a small synthetic/sample dataset
        base = [
            {"prompt": "小明有15个苹果，他吃了3个，又买了8个。现在他有多少个苹果？", "answer": "20"},
            {"prompt": "一个班级有30个学生，其中60%是女生。女生有多少人？", "answer": "18"},
            {"prompt": "一个圆的半径是5厘米，它的直径是多少厘米？", "answer": "10"},
            {"prompt": "一辆汽车每小时行驶60公里，行驶3小时能走多少公里？", "answer": "180"},
            {"prompt": "一个正方形的边长是8厘米，它的周长是多少厘米？", "answer": "32"},
        ]
        items = []
        while len(items) < max_samples:
            for b in base:
                if len(items) >= max_samples:
                    break
                items.append(dict(b))
        return SimpleQADataset(items)



