import pytest
from transformers import AutoTokenizer, GPT2TokenizerFast
from gtlib.datasets import make_year_dataset, TEMPLATES

@pytest.fixture(scope="module")
def tok():
    return GPT2TokenizerFast.from_pretrained("gpt2")

def test_five_templates():
    assert len(TEMPLATES) >= 5

def test_shape_and_labels(tok):
    df = make_year_dataset("war", n=50, seed=0)
    assert list(df.columns) == ["clean", "corrupted", "label"]
    assert len(df) == 50 and df.label.between(2, 98).all()

def test_corruption_is_01(tok):
    df = make_year_dataset("war", n=20, seed=0)
    for c, k, yy in zip(df.clean, df.corrupted, df.label):
        assert k == c.replace(f"{yy:02d} ", "01 ", 1)
        assert k != c

def test_single_token_years(tok):
    df = make_year_dataset("war", n=100, seed=1)
    for yy in df.label:
        assert len(tok(f"{yy:02d}").input_ids) == 1

def test_determinism():
    a = make_year_dataset("war", n=30, seed=7); b = make_year_dataset("war", n=30, seed=7)
    assert a.equals(b)

def test_custom_tokenizer_single_token_years():
    pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    df = make_year_dataset("war", n=20, seed=0, tokenizer=pythia_tok)
    for yy in df.label:
        assert len(pythia_tok(f"{yy:02d}").input_ids) == 1

class _FakeTokenizer:
    """Never produces a single-token encoding, so no (century, yy) pair passes
    the filter — used to exercise the bounded-attempts guard."""
    def __call__(self, *args, **kwargs):
        class _Out:
            input_ids = [1, 2]
        return _Out()

def test_bounded_attempts_raises_value_error():
    with pytest.raises(ValueError, match="war"):
        make_year_dataset("war", n=5, seed=0, tokenizer=_FakeTokenizer())

class _BosTokenizer:
    """Prepends a BOS id unless add_special_tokens=False — mimics TransformerLens
    tokenizers for Pythia/Qwen. Regression guard: the single-token filter must
    tokenize with add_special_tokens=False or it wrongly rejects every pair."""
    def __call__(self, text, add_special_tokens=True, **kwargs):
        class _Out:
            pass
        out = _Out()
        out.input_ids = ([0] if add_special_tokens else []) + [1000 + len(text)]
        return out

def test_bos_prepending_tokenizer_still_builds():
    df = make_year_dataset("war", n=10, seed=0, tokenizer=_BosTokenizer())
    assert len(df) == 10


def test_crossmodel_covers_the_camera_ready_scale_range():
    """The camera-ready extends the parameter range to 12B (reviewer yDuj)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import rq3_crossmodel

    assert "pythia-6.9b" in rq3_crossmodel.DISCOVER_MODELS
    assert "pythia-12b" in rq3_crossmodel.DISCOVER_MODELS
    assert rq3_crossmodel.BATCH_SIZES["pythia-12b"] <= 4
