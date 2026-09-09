"""Tests for the synthetic comparison-NLI probe generator (RQ4)."""
import re

import pandas as pd
import pytest
import torch
from transformers import GPT2TokenizerFast

from gtlib.nli import make_nli_dataset, nli_logit_diff


@pytest.fixture(scope="module")
def tok():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def test_label_logic_more(tok):
    df = make_nli_dataset(200, seed=0, tokenizer=tok, comparative="more")
    for _, r in df.iterrows():
        assert (r.x > r.y) == (r.label_str == "Yes")


def test_label_logic_fewer(tok):
    df = make_nli_dataset(200, seed=0, tokenizer=tok, comparative="fewer")
    for _, r in df.iterrows():
        assert (r.x < r.y) == (r.label_str == "Yes")


def test_no_ties(tok):
    df = make_nli_dataset(200, seed=3, tokenizer=tok)
    assert (df.x != df.y).all()


def test_label_token_ids(tok):
    yes_id = tok(" Yes").input_ids[0]
    no_id = tok(" No").input_ids[0]
    assert len(tok(" Yes").input_ids) == 1
    assert len(tok(" No").input_ids) == 1
    df = make_nli_dataset(100, seed=0, tokenizer=tok)
    for _, r in df.iterrows():
        expected = yes_id if r.label_str == "Yes" else no_id
        assert r.label == expected


def test_numbers_single_token(tok):
    df = make_nli_dataset(100, seed=0, tokenizer=tok)
    for _, r in df.iterrows():
        assert len(tok(str(r.x)).input_ids) == 1
        assert len(tok(str(r.y)).input_ids) == 1
        xp = int(re.search(r"Premise: \w+ has (\d+)", r.corrupted).group(1))
        assert len(tok(str(xp)).input_ids) == 1


def test_corruption_same_digit_count_and_token_length(tok):
    """Regression: Qwen tokenizes numbers per-digit, so a corruption that
    changes digit count (e.g. x=5 -> x'=12) changes total sequence length,
    which breaks positional clean/corrupted pairing in EAP evaluation
    ("Attention mask shape ... does not match tokens shape"). The corrupted
    x' must have the same digit count as x, and (checked here with the GPT-2
    tokenizer, where every in-range number is single-token) the full clean
    and corrupted prompts must tokenize to equal length."""
    df = make_nli_dataset(200, seed=0, tokenizer=tok, x_range=(2, 98), y_range=(2, 98))
    for _, r in df.iterrows():
        xp = int(re.search(r"Premise: \w+ has (\d+)", r.corrupted).group(1))
        assert len(str(xp)) == len(str(r.x)), (r.x, xp)
        assert len(tok(r.clean).input_ids) == len(tok(r.corrupted).input_ids)


def test_corruption_flips_label(tok):
    df = make_nli_dataset(100, seed=0, tokenizer=tok, comparative="more")
    for _, r in df.iterrows():
        xp = int(re.search(r"Premise: \w+ has (\d+)", r.corrupted).group(1))
        clean_truth = r.x > r.y
        corrupted_truth = xp > r.y
        assert clean_truth == (r.label_str == "Yes")
        assert corrupted_truth != clean_truth


def test_balance(tok):
    df = make_nli_dataset(400, seed=1, tokenizer=tok)
    frac = (df.label_str == "Yes").mean()
    assert 0.45 <= frac <= 0.55


def test_only_x_changes(tok):
    df = make_nli_dataset(50, seed=2, tokenizer=tok)
    for _, r in df.iterrows():
        clean_hyp = re.search(r"Hypothesis.*", r.clean).group(0)
        corrupted_hyp = re.search(r"Hypothesis.*", r.corrupted).group(0)
        # Hypothesis sentence (mentions y, not x) must be byte-identical.
        assert clean_hyp == corrupted_hyp
        # Everything outside the premise number must also match.
        clean_masked = re.sub(r"has \d+", "has N", r.clean)
        corrupted_masked = re.sub(r"has \d+", "has N", r.corrupted)
        assert clean_masked == corrupted_masked


def test_columns(tok):
    df = make_nli_dataset(20, seed=0, tokenizer=tok)
    assert list(df.columns) == ["clean", "corrupted", "label", "label_str", "x", "y"]


def test_determinism(tok):
    df1 = make_nli_dataset(50, seed=42, tokenizer=tok)
    df2 = make_nli_dataset(50, seed=42, tokenizer=tok)
    pd.testing.assert_frame_equal(df1, df2)


def test_prompt_format(tok):
    df = make_nli_dataset(20, seed=0, tokenizer=tok)
    for _, r in df.iterrows():
        assert re.match(
            r"^Premise: \w+ has \d+ \w+\. "
            r"Hypothesis: \w+ has more than \d+ \w+\. "
            r"Question: Is the hypothesis true\? Answer:$",
            r.clean,
        )


def test_nli_logit_diff_shape_and_sign(tok):
    metric = nli_logit_diff(tok)
    yes_id = tok(" Yes").input_ids[0]
    no_id = tok(" No").input_ids[0]
    vocab = tok.vocab_size
    # Two fake positions, batch of 2, seq_len 3, vocab-sized logits.
    logits = torch.zeros(2, 3, vocab)
    # item 0 favors Yes strongly and is labeled Yes -> positive (correct)
    logits[0, -1, yes_id] = 5.0
    logits[0, -1, no_id] = 1.0
    # item 1 favors Yes strongly but is labeled No -> negative (incorrect)
    logits[1, -1, yes_id] = 5.0
    logits[1, -1, no_id] = 1.0
    input_length = torch.tensor([3, 3])
    labels = torch.tensor([yes_id, no_id])

    per_item = metric(logits, None, input_length, labels, mean=False, loss=False)
    assert per_item.shape == (2,)
    assert per_item[0].item() > 0
    assert per_item[1].item() < 0
    assert per_item[0].item() == pytest.approx(-per_item[1].item())

    mean_val = metric(logits, None, input_length, labels, mean=True, loss=False)
    assert mean_val.item() == pytest.approx(per_item.mean().item())

    loss_val = metric(logits, None, input_length, labels, mean=True, loss=True)
    assert loss_val.item() == pytest.approx(-mean_val.item())


def test_nli_logit_diff_accepts_plain_list_labels(tok):
    """Regression: gtlib.runner.collate_EAP yields `labels` as a plain Python
    tuple/list of ints (not a Tensor). `labels == yes_id` on a list/tuple is a
    Python bool, and torch.where(bool, Tensor, Tensor) raises TypeError. The
    metric must tensorize labels internally before comparing."""
    metric = nli_logit_diff(tok)
    yes_id = tok(" Yes").input_ids[0]
    no_id = tok(" No").input_ids[0]
    vocab = tok.vocab_size
    logits = torch.zeros(2, 3, vocab)
    logits[0, -1, yes_id] = 5.0
    logits[0, -1, no_id] = 1.0
    logits[1, -1, yes_id] = 5.0
    logits[1, -1, no_id] = 1.0
    input_length = torch.tensor([3, 3])
    labels = [yes_id, no_id]  # plain list, mirroring collate_EAP's tuple output

    per_item = metric(logits, None, input_length, labels, mean=False, loss=False)
    assert per_item.shape == (2,)
    assert per_item[0].item() > 0
    assert per_item[1].item() < 0


class _BosTokenizer:
    """Prepends a BOS id unless add_special_tokens=False — mimics TransformerLens
    tokenizers for Pythia/Qwen. Regression guard: yes/no id lookups and the
    single-token filter must tokenize with add_special_tokens=False or " Yes"/
    " No" both collapse to the BOS id, silently corrupting the metric.

    Ids are derived from sum(ord(c) for c in text) + 100 (not hash(), which is
    salted per process) so distinct texts get distinct, deterministic ids.
    """
    def __call__(self, text, add_special_tokens=True, **kwargs):
        class _Out:
            pass
        out = _Out()
        out.input_ids = ([0] if add_special_tokens else []) + [sum(ord(c) for c in text) + 100]
        return out


def test_bos_prepending_tokenizer_builds_dataset():
    df = make_nli_dataset(6, seed=0, tokenizer=_BosTokenizer())
    assert len(df) == 6


class _PerDigitTokenizer:
    """Tokenizes multi-digit numbers one-digit-per-token (mimics Qwen's per-digit
    number tokenization) while keeping non-numeric strings (e.g. ' Yes'/' No')
    single-token. Extends the _BosTokenizer pattern with per-char ids for digits."""
    def __call__(self, text, add_special_tokens=True, **kwargs):
        class _Out:
            pass
        out = _Out()
        prefix = [0] if add_special_tokens else []
        stripped = text.strip()
        if stripped.isdigit():
            out.input_ids = prefix + [ord(c) for c in stripped]
        else:
            out.input_ids = prefix + [sum(ord(c) for c in text) + 100]
        return out


def test_require_single_token_numbers_default_raises_on_multitoken_numbers():
    with pytest.raises(ValueError):
        make_nli_dataset(
            6, seed=0, tokenizer=_PerDigitTokenizer(), x_range=(10, 98), y_range=(10, 98)
        )


def test_require_single_token_numbers_false_builds_with_multitoken_numbers():
    df = make_nli_dataset(
        6, seed=0, tokenizer=_PerDigitTokenizer(), x_range=(10, 98), y_range=(10, 98),
        require_single_token_numbers=False,
    )
    assert len(df) == 6


def test_bos_prepending_tokenizer_nli_logit_diff_ids_distinct():
    fake_tok = _BosTokenizer()
    assert (
        fake_tok(" Yes", add_special_tokens=False).input_ids[0]
        != fake_tok(" No", add_special_tokens=False).input_ids[0]
    )
    nli_logit_diff(fake_tok)
