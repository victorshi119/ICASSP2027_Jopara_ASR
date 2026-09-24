#!/usr/bin/env python3
"""
Post-processing normalization for Jopara ASR evaluation.

Applied symmetrically to BOTH reference and hypothesis before scoring.
Complements (does not replace) jopara_normalize.py and the SPL vocab bank.

Phase 2 complete (2026-06-21): ChatGPT classified top-80 confusion pairs.
  NORMALIZE: 17 pairs → rules populated and enabled below.
  AMBIGUOUS: 9 pairs → left disabled with notes.
  ASR_ERROR: 54 pairs → not normalized.

Usage:
    python postprocess_norm.py --ref ref.txt --hyp hyp.txt \\
        --ref-out ref_norm.txt --hyp-out hyp_norm.txt [--show-changes]

Pipeline order:
  raw text → jopara_normalize.py → postprocess_norm.py → sclite scorer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Rule configuration
# ---------------------------------------------------------------------------

RULE_CONFIG: dict[str, bool] = {
    # Safe — annotation artifact: "porque / por qué" → "porque"
    "slash_alternative":        True,

    # Safe — Spanish accent variants that are phonetically identical
    # Canon forms from ChatGPT Phase 2 review
    "spanish_accent_pairs":     True,

    # Safe — Guaraní accent/diacritic pairs confirmed by ChatGPT
    "guarani_accent_pairs":     True,

    # Safe — SPL-adjacent Guaraní-influenced Spanish spellings in ASR output
    "spl_guarani":              True,

    # Safe — strip/normalize non-lexical hesitation sounds (eh, mm, ah) both sides
    "filler_sounds":            True,

    # Safe — normalize laughter token spelling variants to canonical `jaja`
    "laughter_tokens":          True,

    # Safe — specific digit ↔ written-number pairs observed in confusion pairs
    "number_forms":             True,

    # DISABLED — voseo (tenés/tienes): morphological; ASR_ERROR per ChatGPT
    "spanish_voseo":            False,

    # DISABLED — Guaraní glottal stop drops (ha'e→ha): ASR_ERROR per ChatGPT
    "guarani_glottal":          False,

    # DISABLED — discourse filler words (este, bueno, pues): too risky without position-aware logic
    "filler_words":             False,

    # DISABLED — truncated forms (ent→entonces): risky without context
    "truncated_forms":          False,
}


# ---------------------------------------------------------------------------
# Rule 1: Slash annotation artifact
# ---------------------------------------------------------------------------

def rule_slash_alternative(text: str) -> str:
    """
    Strip slash-separated annotation alternatives.
    Annotators wrote 'porque / por qué' when unsure which form the speaker used.
    The model never outputs '/', causing guaranteed deletion errors (5 observed).
    Keep the first alternative only.
    """
    text = re.sub(r"(\S+)\s*/\s*\S+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Rule 2: Spanish accent pairs
# ---------------------------------------------------------------------------
# Pairs where accent mark is purely orthographic (phonetically identical).
# Both forms map to the canonical form; applied to ref AND hyp.
# ChatGPT Phase 2 canonical forms used directly.
# NOTE: qué/que is included because in speech "qué" (what) and "que" (that)
# are phonetically identical — ASR cannot distinguish them.

_SPANISH_ACCENT_MAP: dict[str, str] = {
    # non-canonical → canonical
    "si":       "sí",       # "yes" without accent → canonical with accent
    "que":      "qué",      # conjunction/interrogative; phonetically identical
    "péro":     "pero",     # wrongly-accented Spanish "but"
    # NOTE: así/asi excluded — "así" is correct orthography; omitted accent is ASR issue
    # Additional from per-subset analysis
    "iñunido":  "iñunído",  # Guaraní-Spanish word, accent omitted in HYP
}


def rule_spanish_accent_pairs(text: str) -> str:
    tokens = text.split()
    return " ".join(_SPANISH_ACCENT_MAP.get(t, t) for t in tokens)


# ---------------------------------------------------------------------------
# Rule 3: Guaraní accent / diacritic pairs
# ---------------------------------------------------------------------------
# Specific Guaraní words where annotators and ASR differ on accent placement
# or nasal vowel notation. ChatGPT Phase 2 NORMALIZE classification.

_GUARANI_ACCENT_MAP: dict[str, str] = {
    # non-canonical → canonical (ChatGPT canon)
    "hina":        "hína",      # Guaraní aspect particle; accent on í
    "peicha":      "péicha",    # "like that/so"; accent on é
    "upea":        "upéa",      # demonstrative; accent on é  (not upéva — see below)
    "cheve":       "chéve",     # dative "to me"; accent on é
    "rera":        "réra",      # "name"; accent on é
    "oikua":       "oikuaa",    # Guaraní verb; final vowel doubled in canonical
    "kaarupe":     "ka'arúpe",  # "in the afternoon"; glottal + accent restored
    "oñorãiro":    "oñorairõ",  # nasal vowel position variant; canonical has õ
    "ko'anga":     "ko'ág̃a",   # "now"; diacritic variant
    "ko'ãnga":     "ko'ág̃a",   # alternate nasal spelling → canonical
    "lomita":      "lomitã",    # nasal tilde on final a
    "atyra":       "atyrápe",   # suffix omission; restore suffix
    "iñunido":     "iñunído",   # also in Spanish accent map; duplicated for safety
    # upéa → upéva: demonstrative variant (different surface form, same reference)
    "upéa":        "upéva",
}


def rule_guarani_accent_pairs(text: str) -> str:
    tokens = text.split()
    return " ".join(_GUARANI_ACCENT_MAP.get(t, t) for t in tokens)


# ---------------------------------------------------------------------------
# Rule 4: SPL-adjacent Guaraní-influenced Spanish spellings
# ---------------------------------------------------------------------------
# These are analogous to SPL vocab bank entries but appear in the ASR output
# rather than only in the reference. Applied to both sides.

_SPL_GUARANI_MAP: dict[str, str] = {
    # non-canonical → canonical (ChatGPT SPL candidates)
    "kósa":     "cosa",     # Guaraní k-spelling + accent on Spanish "cosa"
    "kachaka":  "cachaca",  # k-spelling of Spanish "cachaza" (a drink)
    "jetá":     "heta",     # j/h-spelling of Guaraní "heta" (many)
    "ivesíno":  "ivecino",  # Guaraní-prefix + SPL s/c spelling of "vecino"
    "do":       "dos",      # final-s deletion: "dos" (two) → "do"; safe (count=3)
    "hente":    "gente",    # h/g SPL for Spanish "gente" (people)
    # Note: "voi/voy" left out — voi is Guaraní particle, voy is Spanish verb;
    # phonetically similar but distinct lexical items; ChatGPT left unclassified.
}


def rule_spl_guarani(text: str) -> str:
    tokens = text.split()
    return " ".join(_SPL_GUARANI_MAP.get(t, t) for t in tokens)


# ---------------------------------------------------------------------------
# Rule 5: Filler sounds
# ---------------------------------------------------------------------------
# Non-lexical hesitation sounds with no fixed spelling.
# Strategy: normalize length/spelling variants to a canonical form.
# These are stripped from both ref and hyp (canonical = "").
# ChatGPT confirmed eh (×8), mm (×3), ah (×3) in deletions are non-lexical.

_FILLER_SOUND_NORM: dict[str, str] = {
    # Strip entirely (map to "") — both sides
    "eh": "", "eeh": "", "eeh": "",
    "mm": "", "mmm": "", "mmmm": "",
    "hmm": "", "hm": "",
    "ehm": "", "em": "", "eem": "",
    "mhmm": "", "mhm": "",
    "ah": "", "aah": "",
    "uh": "", "uhh": "",
    # "eh" also appears as the Guaraní interjection but in Jopara context
    # it's almost always a hesitation sound. Acceptable to strip.
}


def rule_filler_sounds(text: str) -> str:
    tokens = text.split()
    out = [t for t in tokens if _FILLER_SOUND_NORM.get(t.lower(), t) != ""]
    return " ".join(out)


# ---------------------------------------------------------------------------
# Rule 6: Laughter tokens
# ---------------------------------------------------------------------------
# Non-lexical vocalizations; annotator spelling is arbitrary.
# Both ref and hyp: normalize all laughter variants to canonical "jaja".

_LAUGHTER_FORMS: set[str] = {
    # Only forms ≥ 4 chars or unambiguously non-lexical in Jopara
    "jaja", "jajaja", "jajajaja", "jajajajaja", "jajajajajaja",
    "ajajaja", "ajajajaja", "ájájájá",
    "jeje", "jejeje",
    "jee",
    # NOTE: "ja" and "je" excluded — both are common Guaraní words/clitics
    # NOTE: multi-token forms ("ja ja") skipped — function is token-level
}

# Match laughter: starts with j then 3+ base chars, or starts with a/e then j then 2+ base chars.
# Accent-bearing chars excluded to avoid false positives on Guaraní words (e.g. ajéa = "I saw it").
_LAUGHTER_PATTERN = re.compile(r"^(?:j[aje]{3,}|[ae]j[aje]{2,})$", re.IGNORECASE)

_LAUGHTER_CANONICAL = "jaja"   # change to "" to strip entirely


def rule_laughter_tokens(text: str) -> str:
    tokens = text.split()
    out = []
    for t in tokens:
        if t.lower() in _LAUGHTER_FORMS or _LAUGHTER_PATTERN.match(t):
            if _LAUGHTER_CANONICAL:
                out.append(_LAUGHTER_CANONICAL)
            # else: strip
        else:
            out.append(t)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Rule 7: Number forms
# ---------------------------------------------------------------------------
# ASR outputs digit forms; references use written-out word forms.
# Map digits → word forms to match references.
# Only specific pairs observed in confusion pairs (veintiocho/28, cuarenta/40).

_NUMBER_DIGIT_TO_WORD: dict[str, str] = {
    # Observed in confusion pairs (count≥1 across subsets)
    "28":   "veintiocho",
    "40":   "cuarenta",
    # Add more after inspecting full prediction files:
    "1":    "uno",
    "2":    "dos",
    "3":    "tres",
    "4":    "cuatro",
    "5":    "cinco",
    "6":    "seis",
    "7":    "siete",
    "8":    "ocho",
    "9":    "nueve",
    "10":   "diez",
    "20":   "veinte",
    "30":   "treinta",
    "50":   "cincuenta",
    "60":   "sesenta",
    "70":   "setenta",
    "80":   "ochenta",
    "90":   "noventa",
    "100":  "cien",
    "1000": "mil",
}


def rule_number_forms(text: str) -> str:
    tokens = text.split()
    return " ".join(_NUMBER_DIGIT_TO_WORD.get(t, t) for t in tokens)


# ---------------------------------------------------------------------------
# Disabled rules (stubs)
# ---------------------------------------------------------------------------

def _identity(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

RULES: list[tuple[str, Callable[[str], str]]] = [
    ("slash_alternative",       rule_slash_alternative),
    ("spanish_accent_pairs",    rule_spanish_accent_pairs),
    ("guarani_accent_pairs",    rule_guarani_accent_pairs),
    ("spl_guarani",             rule_spl_guarani),
    ("filler_sounds",           rule_filler_sounds),
    ("laughter_tokens",         rule_laughter_tokens),
    ("number_forms",            rule_number_forms),
    ("spanish_voseo",           _identity),
    ("guarani_glottal",         _identity),
    ("filler_words",            _identity),
    ("truncated_forms",         _identity),
]


def postprocess(text: str) -> str:
    """Apply all enabled normalization rules in sequence."""
    for name, fn in RULES:
        if RULE_CONFIG.get(name, False):
            text = fn(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref",          required=True)
    p.add_argument("--hyp",          required=True)
    p.add_argument("--ref-out",      required=True)
    p.add_argument("--hyp-out",      required=True)
    p.add_argument("--show-changes", action="store_true")
    args = p.parse_args()

    refs = Path(args.ref).read_text(encoding="utf-8").splitlines()
    hyps = Path(args.hyp).read_text(encoding="utf-8").splitlines()
    assert len(refs) == len(hyps)

    enabled = [k for k, v in RULE_CONFIG.items() if v]
    print(f"Enabled rules ({len(enabled)}): {', '.join(enabled)}", file=sys.stderr)

    refs_out, hyps_out = [], []
    n_ref_changed = n_hyp_changed = 0
    for i, (r, h) in enumerate(zip(refs, hyps)):
        r2, h2 = postprocess(r), postprocess(h)
        if r2 != r:
            n_ref_changed += 1
            if args.show_changes:
                print(f"REF[{i}]  {r!r}\n      →  {r2!r}")
        if h2 != h:
            n_hyp_changed += 1
            if args.show_changes:
                print(f"HYP[{i}]  {h!r}\n      →  {h2!r}")
        refs_out.append(r2)
        hyps_out.append(h2)

    Path(args.ref_out).write_text("\n".join(refs_out) + "\n", encoding="utf-8")
    Path(args.hyp_out).write_text("\n".join(hyps_out) + "\n", encoding="utf-8")
    print(f"Refs changed: {n_ref_changed}/{len(refs)}  "
          f"Hyps changed: {n_hyp_changed}/{len(hyps)}", file=sys.stderr)


if __name__ == "__main__":
    main()
