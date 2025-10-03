import re


def extract_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text: str) -> float:
    reward = 0.0
    if text.count("<think>\n") == 1:
        reward += 0.125
    if text.count("</think>\n") == 1:
        reward += 0.125
    if text.count("<answer>\n") == 1:
        reward += 0.125
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward


def normalize_number(text: str) -> str:
    """Normalize numbers by removing commas and extra whitespace."""
    # Remove commas from numbers (e.g., "360,000" -> "360000")
    normalized = re.sub(r'(\d),(\d)', r'\1\2', text)
    return normalized.strip()


def correctness_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    # Print a small sample for visibility
    try:
        k = min(4, len(responses))
        for i in range(k):
            print("\n[GRPO Sample]")
            print(f"Question: {prompts[i] if i < len(prompts) else ''}")
            print(f"GT Answer: {answers[i] if i < len(answers) else ''}")
            print(f"Full Response: {repr(responses[i][:200])}")  # Show raw response
            print(f"LLM Answer: {extracted_responses[i]}")
    except Exception:
        pass
    
    # Normalize both response and answer to handle comma formatting
    rewards = []
    for i, (response, ans) in enumerate(zip(extracted_responses, answers)):
        norm_response = normalize_number(response)
        norm_answer = normalize_number(str(ans))
        is_correct = norm_response == norm_answer
        rewards.append(2.0 if is_correct else 0.0)
        
        # Debug: Print first mismatch to understand why correctness is 0
        if not is_correct and i == 0:
            print(f"[Correctness Debug] Mismatch:")
            print(f"  Extracted: '{response}' → Normalized: '{norm_response}'")
            print(f"  GT Answer: '{ans}' → Normalized: '{norm_answer}'")
    return rewards


def digit_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


def hard_format_reward(prompts, responses, answers):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]


