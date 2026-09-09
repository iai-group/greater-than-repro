"""Tests for the RQ4 format-control probes (gtlib/format_control.py)."""
from transformers import GPT2TokenizerFast

from gtlib.format_control import make_identity_dataset, make_ordinal_dataset
from gtlib.nli import PROMPT_SCAFFOLD


def _tok():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def test_identity_dataset_is_label_balanced():
    df = make_identity_dataset(n=40, seed=0, tokenizer=_tok())
    assert len(df) == 40
    assert (df["label_str"] == "Yes").sum() == 20


def test_probes_reuse_the_nli_scaffolding():
    """Format must match the numerical probe, or it is not a control."""
    head = PROMPT_SCAFFOLD.split("{premise}")[0]
    tail = PROMPT_SCAFFOLD.split("{hypothesis}")[-1]
    for build in (make_identity_dataset, make_ordinal_dataset):
        df = build(n=4, seed=0, tokenizer=_tok())
        assert df["clean"].iloc[0].startswith(head)
        assert df["clean"].iloc[0].endswith(tail)


def test_probes_have_no_digits():
    """A numeral anywhere would reintroduce the confound being controlled."""
    for build in (make_identity_dataset, make_ordinal_dataset):
        df = build(n=40, seed=0, tokenizer=_tok())
        for text in list(df["clean"]) + list(df["corrupted"]):
            assert not any(ch.isdigit() for ch in text), text


def test_identity_answer_is_yes_exactly_when_attributes_match():
    df = make_identity_dataset(n=40, seed=3, tokenizer=_tok())
    for _, row in df.iterrows():
        premise_attr = row["clean"].split("has ")[1].split(" ")[0]
        hypothesis_attr = row["clean"].split("has ")[2].split(" ")[0]
        expected = "Yes" if premise_attr == hypothesis_attr else "No"
        assert row["label_str"] == expected, row["clean"]


def test_corruption_flips_the_label_and_preserves_token_length():
    tok = _tok()
    for build in (make_identity_dataset, make_ordinal_dataset):
        df = build(n=20, seed=1, tokenizer=tok)
        for clean, corrupted in zip(df["clean"], df["corrupted"]):
            assert clean != corrupted
            assert (len(tok(clean, add_special_tokens=False).input_ids)
                    == len(tok(corrupted, add_special_tokens=False).input_ids))


def test_ordinal_answer_matches_letter_order():
    df = make_ordinal_dataset(n=40, seed=0, tokenizer=_tok())
    for _, row in df.iterrows():
        x = row["clean"].split("at letter ")[1].split(".")[0]
        y = row["clean"].split("past letter ")[1].split(".")[0]
        assert row["label_str"] == ("Yes" if x > y else "No"), row["clean"]


def test_labels_are_yes_no_token_ids():
    tok = _tok()
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    df = make_identity_dataset(n=4, seed=0, tokenizer=tok)
    assert df.loc[df["label_str"] == "Yes", "label"].iloc[0] == yes_id
    assert df.loc[df["label_str"] == "No", "label"].iloc[0] == no_id
