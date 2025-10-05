import re


def extract_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text: str) -> float:
    """
    More lenient mark reward that accepts tags with or without newlines.
    Gives 0.125 for each properly placed tag (4 tags = 0.5 total).
    """
    reward = 0.0
    # Accept both <think>\n and <think> (without newline requirement)
    if "<think>" in text:
        reward += 0.125
    if "</think>" in text:
        reward += 0.125
    if "<answer>" in text:
        reward += 0.125
    if "</answer>" in text:
        reward += 0.125
    return reward


def normalize_number(text: str) -> str:
    """
    Normalize numbers by removing:
    - Dollar signs ($)
    - Commas (360,000 -> 360000)
    - Units (minutes, dollars, etc.)
    - Extra whitespace
    
    Examples:
        '$108' -> '108'
        '$360,000' -> '360000'
        '1350 minutes' -> '1350'
        '40 minutes' -> '40'
    """
    # Remove dollar signs
    normalized = text.replace('$', '')
    
    # Remove commas from numbers (e.g., "360,000" -> "360000")
    normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
    
    # Remove common units (minutes, hours, dollars, etc.) - keep only the number
    # Match: number at start, optionally followed by space and words
    match = re.match(r'^(\d+(?:\.\d+)?)', normalized.strip())
    if match:
        normalized = match.group(1)
    
    return normalized.strip()


def correctness_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    # Print a small sample for visibility
    try:
        k = min(4, len(responses))
        for i in range(k):
            has_answer_tag = "<answer>" in responses[i]
            has_think_tag = "<think>" in responses[i]
            print("\n[GRPO Sample]")
            print(f"Question: {prompts[i] if i < len(prompts) else ''}")
            print(f"GT Answer: {answers[i] if i < len(answers) else ''}")
            print(f"Full Response: {repr(responses[i][:200])}... (tags: <think>={has_think_tag}, <answer>={has_answer_tag})")
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
    """
    Flexible format reward that gives partial credit:
    - 0.5 if has both <think> and <answer> tags (any format)
    - 0.25 if has <answer> tags only
    - 0.0 if no proper tags
    """
    rewards = []
    for response in responses:
        has_think = "<think>" in response and "</think>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        
        if has_think and has_answer:
            rewards.append(0.5)
        elif has_answer:
            rewards.append(0.25)  # Partial credit for answer tag only
        else:
            rewards.append(0.0)
    
    return rewards


def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]


