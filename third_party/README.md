# omnilingual-asr

We used [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
at commit `81f51e2` with local changes, saved in
`omnilingual-asr/omnilingual-asr-local-changes.patch` (9 files):

- **Training recipe.** Adds a `trainable_param_patterns` option that freezes everything except
  matching parameters. The LoRA training uses it to train only the decoder adapters.
- **Manifests.** The manifest reader accepts an optional third column with a language code for
  each utterance.
- **Inference pipeline.** The maximum audio length rises from 40 s to 90 s.
- **HDMoLE hooks.** An optional adapter-routing module (`hdmole.py`, new file), its config, and
  two `*_hdmole_debug` model cards. These come from an earlier routing design that the paper does
  not use. They are disabled unless one of those model cards is chosen, but the model code imports
  them, so the patch has to include them.

Applying this patch to a clean checkout of `81f51e2` gives files byte-identical to the ones our
experiments ran on.

```bash
git clone https://github.com/facebookresearch/omnilingual-asr.git
cd omnilingual-asr && git checkout 81f51e2
git apply /path/to/ICASSP2027_Jopara_ASR/third_party/omnilingual-asr/omnilingual-asr-local-changes.patch
pip install -e .
```

The dataset cards (`src/omnilingual_asr/cards/datasets/jopara_*.yaml`) are not included. The
training launchers write them for each run.
