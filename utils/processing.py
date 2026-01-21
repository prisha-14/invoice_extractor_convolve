import re
import torch
import numpy as np


def clean_cost(raw_cost):
    """Normalizes asset cost to integer."""
    if isinstance(raw_cost, (int, float)):
        return int(raw_cost)
    if isinstance(raw_cost, str):
        clean = re.sub(r'[^\d]', '', raw_cost)
        return int(clean) if clean else 0
    return 0


def sanity_check_hp(val):
    """Validates Horse Power (15-100)."""
    try:
        if not val:
            return None
        ival = int(val)
        return ival if 15 <= ival <= 100 else None
    except:
        return None


def smart_extract_hp(data):
    """Extracts HP from field or regex search in model description."""
    raw_hp = data.get("horse_power")
    if isinstance(raw_hp, str):
        match = re.search(r'(\d+)', raw_hp)
        if match:
            raw_hp = int(match.group(1))

    if sanity_check_hp(raw_hp):
        return int(raw_hp)

    model_str = data.get("model_name", "")
    if model_str and isinstance(model_str, str):
        matches = re.findall(
            r'(\d{2})\s*(?:HP|H\.P\.|hp|Hp)', model_str, re.IGNORECASE)
        for m in matches:
            if sanity_check_hp(m):
                return int(m)
    return None


def calculate_confidence(scores):
    """
    Computes confidence using Float32 precision to avoid '1.0' rounding errors.
    Focuses on the lowest probabilities (the hardest tokens) to give a realistic score.
    """
    if not scores:
        return 0.0

    probs = []
    for step_scores in scores:
        # CRITICAL FIX: Convert to float32 BEFORE softmax to prevent rounding to 1.0
        step_scores_f32 = step_scores.to(torch.float32)
        step_probs = torch.nn.functional.softmax(step_scores_f32, dim=-1)
        max_prob, _ = torch.max(step_probs, dim=-1)
        probs.append(max_prob.item())

    if not probs:
        return 0.0

    # Sort probabilities (Hardest tokens first)
    probs.sort()

    # Take the average of the BOTTOM 25% of tokens (The ones the model struggled with)
    # This ignores the 100% confident JSON brackets and quotes.
    cutoff = max(1, int(len(probs) * 0.25))
    hardest_tokens = probs[:cutoff]

    # Calculate average of the hardest parts
    avg_score = np.mean(hardest_tokens)

    return round(avg_score, 4)


def validate_bbox(field_data):
    """Ensures bbox fields are correctly formatted objects."""
    default = {"present": False, "bbox": [0, 0, 0, 0]}
    if not field_data:
        return default
    if isinstance(field_data, list):
        return {"present": True, "bbox": field_data if len(field_data) == 4 else [0, 0, 0, 0]}
    if isinstance(field_data, dict):
        if "bbox" not in field_data:
            field_data["bbox"] = [0, 0, 0, 0]
        if "present" not in field_data:
            field_data["present"] = (field_data["bbox"] != [0, 0, 0, 0])
        return field_data
    return default
