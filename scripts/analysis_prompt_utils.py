from __future__ import annotations

from typing import Dict, List, Tuple


SUPPORTED_CONDITIONS = {"none", "conflict", "conflict_compare", "stale", "opinion", "unverif", "ambig"}


def condition_requires_web_search(condition: str) -> bool:
    return condition.strip().lower() == "stale"


def labels_for_condition(condition: str) -> List[str]:
    condition = condition.strip().lower()
    if condition == "stale":
        return ["changed", "unchanged", "unclear"]
    if condition == "opinion":
        return ["objective", "opinion", "unclear"]
    if condition == "unverif":
        return ["has_justifiable_answer", "no_justifiable_answer", "unclear"]
    if condition == "ambig":
        return ["multiple_justifiable_answers", "single_clear", "unclear_or_none"]
    return []


def judge_prompts(condition: str, claim: str, claim_date: str | None = None) -> Tuple[str, str]:
    condition = condition.strip().lower()
    clean_claim = claim.strip()

    if condition == "stale":
        system = (
            "You are a careful claim-analysis judge. "
            "Reason conservatively. "
            "Output exactly one label from {changed, unchanged, unclear}."
        )
        if claim_date:
            time_ref = f"since this claim was made ({claim_date})"
        else:
            time_ref = "within the past 12 months"
        user = (
            "Claim:\n"
            f"{clean_claim}\n\n"
            "Question:\n"
            f"This claim may be outdated. Based on recent evidence you can find, do you believe the verifiability "
            f"of this claim (whether it has an established answer) has changed {time_ref}?\n\n"
            "Return only one label:\n"
            "- changed\n"
            "- unchanged\n"
            "- unclear"
        )
        return system, user

    if condition == "opinion":
        system = (
            "You are a careful claim-analysis judge. "
            "Output exactly one label from {objective, opinion, unclear}."
        )
        user = (
            "Claim:\n"
            f"{clean_claim}\n\n"
            "Question:\n"
            "Does this claim pertain to an objective fact, or a subjective opinion?\n\n"
            "Return only one label:\n"
            "- objective\n"
            "- opinion\n"
            "- unclear"
        )
        return system, user

    if condition == "unverif":
        system = (
            "You are a careful claim-analysis judge. "
            "Output exactly one label from {has_justifiable_answer, no_justifiable_answer, unclear}."
        )
        user = (
            "Claim:\n"
            f"{clean_claim}\n\n"
            "Question:\n"
            "Does this claim have at least one justifiable answer?\n\n"
            "Return only one label:\n"
            "- has_justifiable_answer\n"
            "- no_justifiable_answer\n"
            "- unclear"
        )
        return system, user

    if condition == "ambig":
        system = (
            "You are a careful claim-analysis judge. "
            "Output exactly one label from {multiple_justifiable_answers, single_clear, unclear_or_none}."
        )
        user = (
            "Claim:\n"
            f"{clean_claim}\n\n"
            "Question:\n"
            "Under a normal, non-contrived reading, does this claim have more than one substantively different, "
            "defensible answer? Use 'multiple_justifiable_answers' only when multiple answers could reasonably be correct, "
            "not merely because wording is broad or some details are omitted. If one answer is clearly best, use 'single_clear'.\n\n"
            "Return only one label:\n"
            "- multiple_justifiable_answers\n"
            "- single_clear\n"
            "- unclear_or_none"
        )
        return system, user

    raise ValueError(f"Unsupported condition for prompting: {condition}")


def normalize_judge_label(condition: str, raw_text: str) -> str:
    text = (raw_text or "").strip().lower()
    condition = condition.strip().lower()

    # Match exact labels first.
    for label in labels_for_condition(condition):
        if text == label:
            return label

    # Fallbacks for slight variations in model output.
    if condition == "stale":
        if "changed" in text and "unchanged" not in text:
            return "changed"
        if "unchanged" in text or "not changed" in text:
            return "unchanged"
        return "unclear"

    if condition == "opinion":
        if "objective" in text or "fact" in text:
            return "objective"
        if "opinion" in text or "subjective" in text:
            return "opinion"
        return "unclear"

    if condition == "unverif":
        if "has_justifiable_answer" in text:
            return "has_justifiable_answer"
        if "no_justifiable_answer" in text:
            return "no_justifiable_answer"
        if "yes" in text or "has at least one" in text:
            return "has_justifiable_answer"
        if "no" in text or "cannot be justified" in text:
            return "no_justifiable_answer"
        return "unclear"

    if condition == "ambig":
        if "multiple_justifiable_answers" in text:
            return "multiple_justifiable_answers"
        if "single_clear" in text:
            return "single_clear"
        if "unclear_or_none" in text:
            return "unclear_or_none"
        if "multiple" in text:
            return "multiple_justifiable_answers"
        if "single" in text or "none" in text:
            return "single_clear"
        return "unclear_or_none"

    return "unclear_or_none"


def summary_template(dataset: str, condition: str, total: int) -> Dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "total": int(total),
    }
