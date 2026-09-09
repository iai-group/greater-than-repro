"""Non-numerical Yes/No probes controlling RQ4's format confound.

Reviewer CxwV objected that GPT-2 and Pythia sitting at chance on the
numerical entailment probe may reflect inability to use the Yes/No prompt
format at all, rather than an unused comparison circuit. These two probes
separate those explanations. Both build on gtlib.nli.PROMPT_SCAFFOLD, so the
scaffolding is shared by construction, and both are digit-free.

make_identity_dataset is the format floor: the hypothesis restates the
premise's attribute, so the task is lexical identity and is strictly easier
than numerical comparison. A model at chance here cannot use the format,
which is decisive rather than suggestive.

make_ordinal_dataset is the matched-difficulty control: the same
greater-than comparison structure over alphabet positions instead of
numerals, separating numeracy from comparison ability.
"""
import random
import string

import pandas as pd

from gtlib.nli import NAMES, NOUNS, PROMPT_SCAFFOLD

# Single-token color adjectives under GPT-2, Pythia, and Qwen tokenizers, so
# clean and corrupted prompts stay length-matched after substitution.
ATTRIBUTES = ["red", "blue", "green", "black", "white", "brown"]

# Uppercase letters used as ordinal positions, restricted to a contiguous
# mid-alphabet band so a same-length opposite-side counterpart always exists
# for the corruption.
LETTERS = list(string.ascii_uppercase[2:23])


def _fill(premise: str, hypothesis: str) -> str:
    """Returns the shared scaffold filled with a premise and hypothesis."""
    return (PROMPT_SCAFFOLD
            .replace("{premise}", premise)
            .replace("{hypothesis}", hypothesis))


def _yes_no_ids(tokenizer):
    """Returns the single token ids for " Yes" and " No".

    add_special_tokens=False: TransformerLens tokenizers prepend BOS for some
    families (Pythia, Qwen), which would make each of " Yes" and " No" look
    multi-token and silently corrupt the labels.
    """
    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    assert len(yes_ids) == 1, f"' Yes' is not single-token: {yes_ids}"
    assert len(no_ids) == 1, f"' No' is not single-token: {no_ids}"
    return yes_ids[0], no_ids[0]


def _rows_to_frame(rows):
    """Returns `rows` as the standard probe DataFrame."""
    return pd.DataFrame(rows, columns=["clean", "corrupted", "label",
                                       "label_str"])


def make_identity_dataset(n, seed, tokenizer):
    """Builds the format-floor probe: lexical identity, no numerals.

    The hypothesis asserts an attribute; the answer is Yes iff it is the same
    attribute the premise gave. Corruption swaps the hypothesis attribute for
    another of equal token length, flipping the label.

    Args:
        n: Number of items; even values give exact label balance.
        seed: Sampling seed.
        tokenizer: Tokenizer used for the " Yes"/" No" label ids and for
            enforcing equal clean/corrupted token length.

    Returns:
        A DataFrame with columns clean, corrupted, label, and label_str.
    """
    yes_id, no_id = _yes_no_ids(tokenizer)
    rng = random.Random(seed)

    def n_tokens(word):
        return len(tokenizer(f" {word}", add_special_tokens=False).input_ids)

    lengths = {a: n_tokens(a) for a in ATTRIBUTES}

    rows = []
    for i in range(n):
        target_yes = (i % 2 == 0)
        name = rng.choice(NAMES)
        noun = rng.choice(NOUNS)

        # The counterpart must differ in identity but match in token length,
        # or clean and corrupted prompts misalign positionally.
        while True:
            premise_attr = rng.choice(ATTRIBUTES)
            others = [a for a in ATTRIBUTES
                      if a != premise_attr
                      and lengths[a] == lengths[premise_attr]]
            if others:
                break
        other_attr = rng.choice(others)

        hypothesis_attr = premise_attr if target_yes else other_attr
        corrupt_attr = other_attr if target_yes else premise_attr

        rows.append({
            "clean": _fill(f"{name} has {premise_attr} {noun}",
                           f"{name} has {hypothesis_attr} {noun}"),
            "corrupted": _fill(f"{name} has {premise_attr} {noun}",
                               f"{name} has {corrupt_attr} {noun}"),
            "label": yes_id if target_yes else no_id,
            "label_str": "Yes" if target_yes else "No",
        })

    return _rows_to_frame(rows)


def make_ordinal_dataset(n, seed, tokenizer):
    """Builds the matched-difficulty control: greater-than over letters.

    Keeps the comparison structure of the numerical probe while removing every
    digit, so a model that succeeds here but fails on the numerical probe
    lacks numeracy rather than comparison ability.

    Args:
        n: Number of items; even values give exact label balance.
        seed: Sampling seed.
        tokenizer: Tokenizer used for the " Yes"/" No" label ids.

    Returns:
        A DataFrame with columns clean, corrupted, label, and label_str.
    """
    yes_id, no_id = _yes_no_ids(tokenizer)
    rng = random.Random(seed)

    rows = []
    for i in range(n):
        target_yes = (i % 2 == 0)
        name = rng.choice(NAMES)

        while True:
            x = rng.choice(LETTERS)
            y = rng.choice(LETTERS)
            if x == y:
                continue
            if (x > y) != target_yes:
                continue
            # An opposite-side counterpart must exist, or the item cannot be
            # corrupted. Single letters are always one token, so length is
            # preserved automatically.
            counterparts = [c for c in LETTERS
                            if c != y and (c > y) != (x > y)]
            if not counterparts:
                continue
            xp = rng.choice(counterparts)
            break

        rows.append({
            "clean": _fill(f"{name} is at letter {x}",
                           f"{name} is past letter {y}"),
            "corrupted": _fill(f"{name} is at letter {xp}",
                               f"{name} is past letter {y}"),
            "label": yes_id if target_yes else no_id,
            "label_str": "Yes" if target_yes else "No",
        })

    return _rows_to_frame(rows)
