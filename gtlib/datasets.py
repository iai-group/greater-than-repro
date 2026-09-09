"""Year-span greater-than datasets: multiple templates, number ranges, inversion."""
import random
import pandas as pd
from transformers import GPT2TokenizerFast

# Nouns copied from the original release (gpt2-greater-than/cache/potential_nouns.txt style).
NOUNS = ["war", "expedition", "dynasty", "marriage", "reign", "voyage", "empire",
         "occupation", "treaty", "famine", "alliance", "project", "strike", "siege"]

TEMPLATES = {
    "war":       "The {noun} lasted from the year {century}{yy} to the year {century}",
    "lived":     "The couple lived there from the year {century}{yy} until the year {century}",
    "worked":    "She worked at the company from the year {century}{yy} to the year {century}",
    "abroad":    "He was abroad from the year {century}{yy} until the year {century}",
    "operated":  "The mine operated from the year {century}{yy} to the year {century}",
}

_tok = None
def _tokenizer():
    global _tok
    if _tok is None:
        _tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return _tok

def make_year_dataset(template_id, n, seed, century_range=(11, 17), yy_range=(2, 98),
                       invert=False, tokenizer=None):
    rng = random.Random(seed)
    tok = tokenizer if tokenizer is not None else _tokenizer()
    template = TEMPLATES[template_id]
    rows = []
    max_attempts = 100 * n
    attempts = 0
    while len(rows) < n:
        if attempts >= max_attempts:
            raise ValueError(
                f"make_year_dataset: exceeded {max_attempts} attempts finding single-token "
                f"(century, yy) pairs for template {template_id!r} under the given tokenizer; "
                "no century/yy pair is single-token, so this template cannot be built."
            )
        attempts += 1
        noun = rng.choice(NOUNS)
        century = rng.randint(*century_range)
        yy = rng.randint(*yy_range)
        # add_special_tokens=False: TransformerLens tokenizers prepend BOS for some
        # families (Pythia/Qwen), which would inflate every length check to >=2 and
        # wrongly reject all pairs. No-op for GPT-2.
        if (len(tok(f"{yy:02d}", add_special_tokens=False).input_ids) != 1
                or len(tok(f"{century}", add_special_tokens=False).input_ids) != 1):
            continue
        clean = template.format(noun=noun, century=century, yy=f"{yy:02d}")
        corrupted = template.format(noun=noun, century=century, yy="01")
        rows.append({"clean": clean, "corrupted": corrupted, "label": yy})
    df = pd.DataFrame(rows)
    df.attrs["invert"] = invert
    return df
