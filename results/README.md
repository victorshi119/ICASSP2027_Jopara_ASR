# Results

`table1/` holds the output of `evaluation/scoring/make_table1.py`:

- `table1.md` and `table1.json` hold Table 1 as printed in the paper: CER / WER (%) for each test
  category and overall. Rows 8–9 are mean ± sample standard deviation over seeds 13, 42, and
  123. `table1.json` also has each seed's values.
- `per_utterance_scores.csv` gives, for every system row (rows 8–9 once per seed) and every test
  utterance, the sclite substitution, deletion, and insertion counts and the reference length,
  at character and word level. It contains **no text**.

Each cell of the table is the sum of errors over its utterances divided by the sum of reference
lengths, so every CER and WER can be recomputed from the per-utterance file, for example:

```python
import pandas as pd
d = pd.read_csv("results/table1/per_utterance_scores.csv")
g = d.groupby(["row", "subset"])[["char_sub", "char_del", "char_ins", "char_ref"]].sum()
print((100 * (g.char_sub + g.char_del + g.char_ins) / g.char_ref).round(1))
```

These counts also allow significance tests between any two rows without the transcripts.
