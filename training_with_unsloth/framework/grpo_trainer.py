from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
import os
import platform
import wandb
import re
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, split: str = 'train', max_samples: int = 500):
        self.tokenizer = tokenizer
        data = load_dataset("gsm8k", "main")
        subset = data[split]
        self.data = subset.select(range(min(max_samples, len(subset))))

    def __len__(self):
        return len(self.data)
    
    def _extract_final_answer(self, solution_text: str) -> str:
        if solution_text is None:
            return ""
        match = re.search(r"####\s*([^\n]+)", solution_text)
        if match:
            return match.group(1).strip()
        return solution_text.strip()

    def __getitem__(self, index):
        sample = self.data[index]
        prompt = sample.get('question', '')
        raw_answer = sample.get('answer', '')
        answer = self._extract_final_answer(raw_answer)
        return {'prompt': prompt, 'answer': answer}


# Will be overridden by the experiment runner if needed
SYSTEM_PROMPT = (
    """Please answer in English with the following format, keep your response concise:
    <think>
    your step-by-step reasoning
    </think>
    <answer>
    the final short answer only
    </answer>
    """
)


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 2 # 组内样本数
    max_prompt_length = 128 # 最大输入长度
    max_generate_length = 128 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1
    eval_temperature = 0.75 # Temperature for evaluation Pass@5/10 sampling

class GRPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args
        # load the model
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # whether to use the reference model
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            # if the reward function is a string, it means using the reward model, then load the model
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
            
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
            
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # cache a batch of data that has already been generated,可供模型多次训练迭代，无需重新生成s
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # the number of model updates
        self.update_steps = 0 
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        # Ensure tokenizer has proper padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    # generate samples, by group
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # Apply chat template with system and user roles
            try:
                input_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                # Fallback to simple format if chat template not available
                print("Warning: Chat template not available, using simple format")
                input_text = f"{SYSTEM_PROMPT}\nQuestion: {prompt}\nAnswer:"
            
            # generate a group of input data
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = inputs['input_ids']
            
            # Get stop token ID for </answer>
            stop_token_ids = []
            try:
                answer_close_id = self.tokenizer.encode("</answer>", add_special_tokens=False)
                if answer_close_id:
                    stop_token_ids = answer_close_id
            except Exception:
                pass
            
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": self.args.max_generate_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                # Add stop strings if tokenizer supports it
                if hasattr(self.tokenizer, 'eos_token') and stop_token_ids:
                    try:
                        # Try using stop_strings parameter (newer transformers)
                        gen_kwargs["stop_strings"] = ["</answer>"]
                        gen_kwargs["tokenizer"] = self.tokenizer
                    except Exception:
                        pass
                
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), **gen_kwargs)
                
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                pad_length = max_length - prompt_response_ids.size(1)
                padding_tensor = torch.full(
                    (prompt_response_ids.size(0), pad_length),
                    fill_value=self.tokenizer.pad_token_id,
                    device=prompt_response_ids.device
                )
                prompt_response_ids = torch.cat([prompt_response_ids, padding_tensor], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # store a group of data
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # generate experiences(advantage, the probability distribution of tokens)
    def generate_experiences(self, inputs):
        
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # calculate the probability of the policy model outputting tokens
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # whether to use the reference model
                if self.ref_model:
                    # calculate the probability of the reference model outputting tokens
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # store the rewards of each response in a group for each reward function
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # convert the output to text
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]

                # Upload a sample prompt/response/answer to Weights & Biases
                try:
                    wandb.log({
                        "sample/prompt": prompt_texts[0],
                        "sample/response": response_texts[0],
                        "sample/answer": answers[0],
                        "timestamp": time.time()
                    })
                except Exception:
                    pass
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    # Get reward function name
                    if isinstance(reward_func, PreTrainedModel):
                        func_name = f"model_{i}"
                    else:
                        func_name = getattr(reward_func, '__name__', f'func_{i}')
                    
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                        
                        # Debug: Print individual reward function outputs with name
                        print(f"Reward '{func_name}': {rewards_per_func[i]}")
                
                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # multiply the weights of each reward function
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # Debug: Print weighted rewards
                print(f"Weighted rewards per function: {rewards}")
                
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                
                # Handle NaN values in rewards
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
                
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # Ensure std is not zero to prevent division by zero
                if std_group_rewards == 0:
                    std_group_rewards = torch.tensor(1.0, device=rewards.device)
                
                # GRPO's advantage is at the sentence level, not at the token level
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
                
                # Log rewards for this group
                try:
                    wandb.log({
                        "rewards/mean": mean_group_rewards.item(),
                        "rewards/std": std_group_rewards.item(),
                        "rewards/max": rewards.max().item(),
                        "rewards/min": rewards.min().item(),
                        "timestamp": time.time()
                    })
                except Exception:
                    pass
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        """
        Compute GRPO (Group Relative Policy Optimization) loss.
        
        GRPO Loss Explanation:
        ----------------------
        GRPO loss trains the model to generate responses that maximize rewards by:
        
        1. **Advantage Calculation** (done in generate_experiences):
           - For each group of N generated responses, compute rewards
           - Normalize: advantage = (reward - mean_reward) / (std_reward + eps)
           - Responses with higher rewards get positive advantages
           - Responses with lower rewards get negative advantages
        
        2. **Policy Ratio with Clipping** (PPO-style):
           - ratio = exp(log_prob_new - log_prob_old)
           - This measures how much the policy changed for this action
           - Clip the ratio to [1-ε, 1+ε] to prevent too large updates
        
        3. **Loss Function**:
           - loss = -min(ratio * advantage, clipped_ratio * advantage)
           - The negative sign converts it to a minimization problem
           - When advantage > 0 (good response): loss encourages higher probability
           - When advantage < 0 (bad response): loss encourages lower probability
           - Clipping prevents the model from changing too drastically
        
        4. **Gradient Flow**:
           - loss.backward() computes gradients w.r.t. model parameters
           - Gradients flow: loss → ratio → log_probs → model_logits → model_params
           - Higher reward responses → positive advantage → model learns to increase their probability
           - Lower reward responses → negative advantage → model learns to decrease their probability
        
        This way, rewards directly influence parameter updates through the advantage signal.
        """
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']  # This contains the reward signal!
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        
        # Add safety checks for numerical stability
        log_ratio = action_log_probs - old_action_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20, max=20)  # Prevent extreme values
        
        coef_1 = torch.exp(log_ratio) # importance sampling shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # the advantage of each token in a sequence is the same
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # Add safety check for division
        action_mask_sum = action_mask.sum(dim=1)
        action_mask_sum = torch.clamp(action_mask_sum, min=1)  # Prevent division by zero
        
        loss = per_token_loss.sum(dim=1) / action_mask_sum # shape: [batch_size * num_generations]
        loss = loss.mean()
        
        # Check for NaN and replace with 0
        if torch.isnan(loss):
            print("Warning: NaN loss detected, replacing with 0")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        
        # calculate the probability of the policy model outputting tokens
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        
        # Add safety checks for numerical stability
        logits = torch.clamp(logits, min=-100, max=100)  # Prevent extreme values
        
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        
        # Additional safety check
        action_log_probs = torch.clamp(action_log_probs, min=-100, max=100)
        
        return action_log_probs

    
    
    def evaluate(self, eval_dataset, num_samples_per_prompt=10, eval_temperature=0.75, verbose=True):
        """
        Evaluate the model on a test set with Pass@K metrics.
        
        Pass@K Definition:
        - Pass@1: Greedy decoding (temperature=0, deterministic best answer)
                  Accuracy = # correct answers / # total questions
        - Pass@5: Sample 5 responses per prompt, success if ANY are correct
                  Pass@5 = # prompts with ≥1 correct in 5 samples / # total prompts
        - Pass@10: Sample 10 responses per prompt, success if ANY are correct
                   Pass@10 = # prompts with ≥1 correct in 10 samples / # total prompts
        
        Args:
            eval_dataset: Dataset with 'prompt' and 'answer' fields
            num_samples_per_prompt: Number of responses to generate per prompt (for Pass@5/10)
            eval_temperature: Temperature for sampling (used for Pass@5/10, not Pass@1)
            verbose: If True, print detailed results for each question
        
        Returns:
            Dictionary with pass@1, pass@5, pass@10 metrics
        """
        from training_with_unsloth.rewards.reward_functions import extract_answer, normalize_number
        
        self.model.eval()
        correct_pass_at_1 = 0  # Greedy decoding
        correct_pass_at_5 = 0  # At least 1 correct in 5 samples
        correct_pass_at_10 = 0  # At least 1 correct in 10 samples
        total_prompts = 0
        
        # Store results for detailed logging
        eval_results = []
        
        with torch.no_grad():
            for q_idx, item in enumerate(eval_dataset):
                prompt = item['prompt']
                answer = item['answer']
                total_prompts += 1
                norm_gt = normalize_number(str(answer))
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"Question {q_idx + 1}/{len(eval_dataset)}")
                    print(f"{'='*80}")
                    print(f"Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")
                    print(f"Ground Truth Answer: {answer}")
                    print(f"-" * 80)
                
                # 1. Pass@1: Generate with greedy decoding (temperature=0)
                input_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
                
                # Greedy generation for Pass@1
                gen_kwargs = {
                    "max_new_tokens": self.args.max_generate_length,
                    "temperature": 0.0,  # Greedy
                    "do_sample": False,  # Deterministic
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                greedy_output = self.model.generate(**inputs.to(self.args.device), **gen_kwargs)
                greedy_response = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
                greedy_answer = extract_answer(greedy_response)
                norm_greedy = normalize_number(greedy_answer)
                
                greedy_correct = norm_greedy == norm_gt
                if greedy_correct:
                    correct_pass_at_1 += 1
                
                if verbose:
                    print(f"Pass@1 (Greedy, temp=0.0):")
                    print(f"  Raw: {greedy_response[:150]}..." if len(greedy_response) > 150 else f"  Raw: {greedy_response}")
                    print(f"  Extracted: '{greedy_answer}' → Normalized: '{norm_greedy}'")
                    print(f"  ✓ CORRECT" if greedy_correct else f"  ✗ WRONG (expected '{norm_gt}')")
                    print(f"-" * 80)
                
                # 2. Pass@5 and Pass@10: Generate with sampling
                gen_kwargs_sample = {
                    "max_new_tokens": self.args.max_generate_length,
                    "temperature": eval_temperature,
                    "top_p": 0.9,
                    "do_sample": True,
                    "num_return_sequences": num_samples_per_prompt,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                sampled_outputs = self.model.generate(**inputs.to(self.args.device), **gen_kwargs_sample)
                sampled_responses = self.tokenizer.batch_decode(sampled_outputs, skip_special_tokens=True)
                
                # Extract and normalize all sampled answers
                sampled_answers = [extract_answer(r) for r in sampled_responses]
                norm_sampled = [normalize_number(a) for a in sampled_answers]
                
                # Check Pass@5: at least 1 correct in first 5 samples
                pass5_correct = any(ans == norm_gt for ans in norm_sampled[:5])
                if pass5_correct:
                    correct_pass_at_5 += 1
                
                # Check Pass@10: at least 1 correct in all 10 samples
                pass10_correct = any(ans == norm_gt for ans in norm_sampled[:10])
                if pass10_correct:
                    correct_pass_at_10 += 1
                
                if verbose:
                    print(f"Pass@5/10 Samples (temp={eval_temperature}):")
                    for i, (ans_raw, ans_norm) in enumerate(zip(sampled_answers, norm_sampled), 1):
                        is_correct = ans_norm == norm_gt
                        marker = "✓" if is_correct else "✗"
                        print(f"  Sample {i:2d}: '{ans_norm}' {marker}")
                    
                    correct_in_5 = sum(1 for ans in norm_sampled[:5] if ans == norm_gt)
                    correct_in_10 = sum(1 for ans in norm_sampled[:10] if ans == norm_gt)
                    print(f"-" * 80)
                    print(f"Summary for Q{q_idx + 1}:")
                    print(f"  Pass@1:  {marker if greedy_correct else '✗'} (greedy)")
                    print(f"  Pass@5:  {'✓' if pass5_correct else '✗'} ({correct_in_5}/5 correct)")
                    print(f"  Pass@10: {'✓' if pass10_correct else '✗'} ({correct_in_10}/10 correct)")
                
                # Store result
                eval_results.append({
                    'question': prompt,
                    'ground_truth': norm_gt,
                    'greedy_answer': norm_greedy,
                    'greedy_correct': greedy_correct,
                    'sampled_answers': norm_sampled,
                    'pass5_correct': pass5_correct,
                    'pass10_correct': pass10_correct
                })
        
        metrics = {
            'pass@1': correct_pass_at_1 / total_prompts if total_prompts > 0 else 0.0,
            'pass@5': correct_pass_at_5 / total_prompts if total_prompts > 0 else 0.0,
            'pass@10': correct_pass_at_10 / total_prompts if total_prompts > 0 else 0.0,
            'eval_samples': total_prompts,
            'detailed_results': eval_results
        }
        
        return metrics
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            try:
                wandb.log({
                    "grpo_loss": loss.item(), 
                    "step": self.update_steps,
                    "timestamp": time.time()
                })
            except Exception:
                pass
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        
        # Baseline evaluation before training
        if self.eval_dataset is not None:
            print(f"\n{'#'*80}")
            print(f"# BASELINE EVALUATION (Before Training - Epoch 0)")
            print(f"{'#'*80}")
            eval_temperature = getattr(self.args, 'eval_temperature', 0.75)
            baseline_metrics = self.evaluate(self.eval_dataset, num_samples_per_prompt=10, eval_temperature=eval_temperature, verbose=True)
            print(f"\n{'='*80}")
            print(f"BASELINE RESULTS:")
            print(f"  Pass@1:  {baseline_metrics['pass@1']:.4f}")
            print(f"  Pass@5:  {baseline_metrics['pass@5']:.4f}")
            print(f"  Pass@10: {baseline_metrics['pass@10']:.4f}")
            print(f"{'='*80}\n")
        
        for epoch in range(self.args.epoch):
            # Track accumulated rewards for this epoch
            epoch_total_rewards = []
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                inputs = self.generate_experiences(batch)
                # Track rewards from this batch
                if 'rewards' in inputs:
                    epoch_total_rewards.extend(inputs['rewards'].cpu().tolist())
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
            
            # Compute accumulated reward for this epoch
            if epoch_total_rewards:
                epoch_mean_reward = sum(epoch_total_rewards) / len(epoch_total_rewards)
                epoch_accumulated_reward = sum(epoch_total_rewards)
                print(f"\n=== Epoch {epoch + 1}/{self.args.epoch} Summary ===")
                print(f"Accumulated Reward: {epoch_accumulated_reward:.4f}")
                print(f"Mean Reward: {epoch_mean_reward:.4f}")
                print(f"Total Samples: {len(epoch_total_rewards)}")
            else:
                epoch_mean_reward = 0.0
                epoch_accumulated_reward = 0.0
            
            # Evaluate at the end of each epoch
            if self.eval_dataset is not None:
                print(f"\n{'#'*80}")
                print(f"# EVALUATION - End of Epoch {epoch + 1}/{self.args.epoch}")
                print(f"{'#'*80}")
                eval_temperature = getattr(self.args, 'eval_temperature', 0.75)
                # Verbose for first 3 epochs and every 10 epochs, otherwise summary only
                verbose_eval = (epoch < 3) or ((epoch + 1) % 10 == 0)
                eval_metrics = self.evaluate(self.eval_dataset, num_samples_per_prompt=10, eval_temperature=eval_temperature, verbose=verbose_eval)
                
                print(f"\n{'='*80}")
                print(f"EPOCH {epoch + 1} RESULTS:")
                print(f"  Pass@1:  {eval_metrics['pass@1']:.4f}")
                print(f"  Pass@5:  {eval_metrics['pass@5']:.4f}")
                print(f"  Pass@10: {eval_metrics['pass@10']:.4f}")
                print(f"  Accumulated Reward: {epoch_accumulated_reward:.4f}")
                print(f"  Mean Reward: {epoch_mean_reward:.4f}")
                print(f"{'='*80}\n")
                
                try:
                    wandb.log({
                        "eval/pass@1": eval_metrics['pass@1'],
                        "eval/pass@5": eval_metrics['pass@5'],
                        "eval/pass@10": eval_metrics['pass@10'],
                        "epoch/accumulated_reward": epoch_accumulated_reward,
                        "epoch/mean_reward": epoch_mean_reward,
                        "epoch": epoch + 1,
                        "timestamp": time.time()
                    })
                except Exception:
                    pass
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)