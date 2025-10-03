#!/usr/bin/env python3
"""
Quick test: load model/tokenizer with our pipeline, ask one question with system prompt,
and verify the output contains <think>...</think> and <answer>...</answer>.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

sys.path.append("/home/hula0401/learning_llm")

from training_with_unsloth.framework.config import ExperimentConfig


def load_tokenizer_model(model_name: str, max_prompt_len: int, max_gen_len: int, load_in_4bit: bool, device_map: str):
    # Try Unsloth first
    try:
        import unsloth  # ensure patches
        from unsloth import FastLanguageModel
        max_seq_length = int(max_prompt_len + max_gen_len)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype="bfloat16",
            device_map=device_map,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer, model
    except Exception as e:
        print(f"Unsloth load failed: {e}. Falling back to Transformers.")

    # Transformers fallback
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    kwargs = dict(trust_remote_code=True, device_map=device_map, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception:
        pass
    return tokenizer, model


def simple_chat_apply(tokenizer, system_prompt: str, user_q: str) -> str:
    # Prefer chat template; fallback to plain format
    try:
        _ = tokenizer.get_chat_template()
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_q}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{system_prompt}\nQuestion: {user_q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment config YAML")
    ap.add_argument("--question", required=False, default="If Lily bought 20 ducks and 10 geese, and Rayden bought 3x ducks and 4x geese, how many more animals does Rayden have than Lily?", help="Test question")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    # HF token from env file if present
    env_file = Path("/home/hula0401/learning_llm/huggingface_token.env")
    if env_file.exists() and not os.environ.get("HF_TOKEN"):
        try:
            for line in env_file.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip().strip('"\'')
                    v = v.strip().strip('"\'')
                    if k in ("HF_TOKEN", "huggingface_token") and v:
                        os.environ["HF_TOKEN"] = v
                        break
        except Exception:
            pass

    tokenizer, model = load_tokenizer_model(
        model_name=cfg.model_name,
        max_prompt_len=cfg.max_prompt_length,
        max_gen_len=cfg.max_generate_length,
        load_in_4bit=cfg.load_in_4bit,
        device_map=cfg.device_map,
    )

    system_prompt = cfg.system_prompt or (
        "Please answer in English with the following format:\n"
        "<think> your step-by-step reasoning </think>\n"
        "<answer> the final short answer only </answer>\n"
    )

    text = simple_chat_apply(tokenizer, system_prompt, args.question)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_generate_length,
            temperature=cfg.temperature if hasattr(cfg, "temperature") else 0.7,
            do_sample=True,
            top_p=0.9,
        )

    gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n==== RAW OUTPUT ====\n")
    print(gen)

    # Check format
    has_think = "<think>" in gen and "</think>" in gen
    has_answer = "<answer>" in gen and "</answer>" in gen
    print("\n==== FORMAT CHECK ====\n")
    print(f"Has <think>...</think>: {has_think}")
    print(f"Has <answer>...</answer>: {has_answer}")


if __name__ == "__main__":
    main()


