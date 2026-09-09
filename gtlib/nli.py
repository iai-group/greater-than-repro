"""Synthetic comparison-NLI probe generator (RQ4).

Mirrors the dataset-construction style of gtlib.datasets (rejection-sampling
single-token numbers) and the metric structure of gtlib.runner.year_prob_diff
(get_logit_positions + loss/mean flags), but for a two-way Yes/No NLI probe:

    "Premise: {name} has {x} {noun}. Hypothesis: {name} has {comparative} than
    {y} {noun}. Question: Is the hypothesis true? Answer:"

Truth: comparative="more" -> Yes iff x > y; "fewer" -> Yes iff x < y (ties are
excluded entirely so the probe is unambiguous). The corrupted item replaces x
with an x' sampled from the opposite side of y (flipping the truth value)
while keeping the name, y, noun, and comparative identical.
"""
import random

import pandas as pd
import torch

from gtlib.runner import get_logit_positions

# ~20 common English first names (single-token-ness is not required: names are
# never the measured token).
NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Linda", "Michael", "Barbara",
    "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", "Joseph",
    "Sarah", "Thomas", "Karen", "Charles", "Nancy",
]

# Plural count nouns so the grammar is always correct.
NOUNS = ["apples", "books", "coins", "cards", "marbles", "stamps", "pencils",
         "buttons", "shells", "tickets"]

# The scaffolding is factored out so the format-control probes in
# gtlib.format_control provably share it rather than copying it. A control
# whose prompt shape drifts from the probe it controls is not a control.
# .replace rather than .format: the substituted text itself contains braces.
PROMPT_SCAFFOLD = (
    "Premise: {premise}. "
    "Hypothesis: {hypothesis}. "
    "Question: Is the hypothesis true? Answer:"
)

PROMPT_TEMPLATE = (
    PROMPT_SCAFFOLD
    .replace("{premise}", "{name} has {x} {noun}")
    .replace("{hypothesis}", "{name} has {comparative} than {y} {noun}")
)


# add_special_tokens=False: TransformerLens tokenizers prepend BOS for some
# families (Pythia/Qwen), which would inflate every length check to >=2 and
# wrongly reject all pairs. No-op for GPT-2.
def _is_single_token(tokenizer, number: int) -> bool:
    return len(tokenizer(str(number), add_special_tokens=False).input_ids) == 1


def make_nli_dataset(n, seed, tokenizer, comparative="more", x_range=(2, 98), y_range=(2, 98),
                      require_single_token_numbers=True):
    """Build a synthetic comparison-NLI probe DataFrame.

    Columns: clean, corrupted, label (token id of " Yes"/" No"), label_str
    ("Yes"/"No"), x, y.

    require_single_token_numbers: if True (default, preserves prior behavior),
    reject (x, y, x') candidates unless each number is single-token under
    `tokenizer`. Set False for tokenizers that split multi-digit numbers into
    per-digit tokens (e.g. Qwen) when the numbers only appear in the prompt and
    are not the measured token (the measured tokens are " Yes"/" No").
    """
    if comparative not in ("more", "fewer"):
        raise ValueError(f"comparative must be 'more' or 'fewer', got {comparative!r}")

    # add_special_tokens=False: with a BOS-prepending tokenizer (TL Pythia/Qwen),
    # " Yes"/" No" would each pick up a leading BOS token, wrongly failing the
    # single-token check and corrupting yes_id/no_id here.
    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    assert len(yes_ids) == 1, f"' Yes' is not single-token under this tokenizer: {yes_ids}"
    assert len(no_ids) == 1, f"' No' is not single-token under this tokenizer: {no_ids}"
    yes_id, no_id = yes_ids[0], no_ids[0]

    rng = random.Random(seed)
    max_attempts = 200 * max(n, 1)

    def truth(x, y):
        return (x > y) if comparative == "more" else (x < y)

    rows = []
    attempts = 0
    for i in range(n):
        target_yes = (i % 2 == 0)

        x = y = None
        while True:
            if attempts >= max_attempts:
                raise ValueError(
                    f"make_nli_dataset: exceeded {max_attempts} attempts sampling a "
                    f"single-token (x, y) pair for comparative={comparative!r}; check "
                    "x_range/y_range against the tokenizer."
                )
            attempts += 1
            cx = rng.randint(*x_range)
            cy = rng.randint(*y_range)
            if cx == cy:
                continue
            if truth(cx, cy) != target_yes:
                continue
            # A corrupted x' on the opposite numeric side of cy, with the SAME
            # DIGIT COUNT as cx, must exist within x_range — otherwise
            # corruption would be impossible (e.g. cy sits at an extreme edge
            # of x_range, or the only opposite-side candidates have a
            # different digit count than cx). Same digit count matters
            # because per-digit tokenizers (e.g. Qwen) produce a different
            # sequence length for a different digit count, which breaks
            # positional clean/corrupted pairing in EAP evaluation.
            cx_ndigits = len(str(cx))
            if cx > cy:
                candidate_exists = any(
                    len(str(c)) == cx_ndigits and c < cy
                    for c in range(x_range[0], x_range[1] + 1)
                )
            else:
                candidate_exists = any(
                    len(str(c)) == cx_ndigits and c > cy
                    for c in range(x_range[0], x_range[1] + 1)
                )
            if not candidate_exists:
                continue
            if require_single_token_numbers and not (
                _is_single_token(tokenizer, cx) and _is_single_token(tokenizer, cy)
            ):
                continue
            x, y = cx, cy
            break

        # Corruption: sample x' on the opposite numeric side of y from x,
        # with the SAME DIGIT COUNT as x (flips the truth value regardless of
        # which comparative word is used, and keeps sequence length constant
        # under per-digit tokenizers like Qwen's).
        x_gt_y = x > y
        x_ndigits = len(str(x))
        xp = None
        while True:
            if attempts >= max_attempts:
                raise ValueError(
                    f"make_nli_dataset: exceeded {max_attempts} attempts sampling a "
                    "single-token, same-digit-count corrupted x'; check x_range "
                    "against the tokenizer."
                )
            attempts += 1
            cxp = rng.randint(*x_range)
            if cxp == y:
                continue
            if (cxp > y) == x_gt_y:
                continue
            if len(str(cxp)) != x_ndigits:
                continue
            if require_single_token_numbers and not _is_single_token(tokenizer, cxp):
                continue
            xp = cxp
            break

        name = rng.choice(NAMES)
        noun = rng.choice(NOUNS)

        clean = PROMPT_TEMPLATE.format(name=name, x=x, y=y, noun=noun, comparative=comparative)
        corrupted = PROMPT_TEMPLATE.format(name=name, x=xp, y=y, noun=noun, comparative=comparative)

        label_str = "Yes" if target_yes else "No"
        label = yes_id if target_yes else no_id

        rows.append({
            "clean": clean,
            "corrupted": corrupted,
            "label": label,
            "label_str": label_str,
            "x": x,
            "y": y,
        })

    return pd.DataFrame(rows, columns=["clean", "corrupted", "label", "label_str", "x", "y"])


def nli_logit_diff(tokenizer):
    """Logit-difference metric with the same call signature as year_prob_diff.

    Per-item value: logit(" Yes") - logit(" No") at the final position, negated
    for items labeled "No" so that positive always means "model favors the
    correct answer". loss=True negates; mean=True averages.
    """
    # add_special_tokens=False: with a BOS-prepending tokenizer (TL Pythia/Qwen),
    # input_ids[0] would be BOS for both " Yes" and " No", silently corrupting
    # the metric to always return 0.
    yes_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer(" No", add_special_tokens=False).input_ids[0]
    assert yes_id != no_id, (
        "yes_id == no_id after tokenizing ' Yes'/' No' — this tokenizer looks "
        "BOS-suspect (e.g. TransformerLens Pythia/Qwen); nli_logit_diff would "
        "silently return 0 for every item. Check add_special_tokens handling."
    )

    def logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        diff = logits[:, yes_id] - logits[:, no_id]
        labels = torch.as_tensor(labels, device=diff.device)
        sign = torch.where(labels == yes_id, torch.ones_like(diff), -torch.ones_like(diff))
        results = diff * sign
        if loss:
            results = -results
        return results.mean() if mean else results

    return logit_diff
