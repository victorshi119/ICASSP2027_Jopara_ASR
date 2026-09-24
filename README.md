# Jopara ASR: benchmark and baselines for spontaneous Spanish–Guaraní code-switching

Code and benchmark metadata for the paper *Jopara ASR: A Community-Verified Benchmark and
Baselines for Spontaneous Spanish–Guaraní Code-Switching* (Wenchen Shi, Shuju Shi; submitted
to ICASSP 2027).

Jopara is the mix of Spanish and Guaraní that most Paraguayans speak every day. This repository
releases:

- **The benchmark definition.** Per-utterance labels and train/dev/test assignments for 2,340
  utterances (3.81 hours) of spontaneous Jopara from the CEGPA corpus, verified against the audio in
  a six-month, 100+ person-hour collaboration with three native Paraguayan speakers of Indigenous
  origin. The utterances cover monolingual speech, sentence-level switching, and
  *jehe'a*, where one word fuses a Spanish root with Guaraní morphology.
- **The curation pipeline.** Audio preparation, the spelling-normalization bank, and word-level
  language tagging.
- **Training and evaluation code** for every system in the paper's Table 1: OmniASR language
  modes, three LoRA adapters, and a mixture-of-experts router over them.
- **Table 1 scores**, with per-utterance error counts for every system.

## Headline results

Character error rate / word error rate (%) on the 150-utterance test set (full table in
[results/table1/table1.md](results/table1/table1.md)):

| System | CER / WER |
|---|---|
| OmniASR, auto-detect language (baseline) | 28.4 / 50.4 |
| OmniASR, told to expect Guaraní (`grn_Latn`) | 25.0 / 47.9 |
| Best LoRA adapter applied to everything | 29.1 / 53.8 |
| Mixture of experts, code-switching slot empty (3 seeds) | **24.3±0.1 / 46.1±0.2** |

Without a specified language, current multilingual ASR still makes 28.4% character errors.
Telling it which language to expect is the most effective fix (25.0% CER), and our best system, a
mixture-of-experts router over fine-tuned adapters, reaches 24.3% CER. Under this data scarcity,
however, a dedicated code-switching adapter shows no measurable benefit, and routing does not
significantly beat naming the language.

## What is and is not in this repository

We do not redistribute anyone else's data. **No audio and no transcript text** are included,
whether from CEGPA or Common Voice. That covers our corrected transcripts and the model outputs,
which are close to transcripts. You can get the raw data from its original sources (see
[data/README.md](data/README.md)). Corrected transcripts and model outputs are available from
the corresponding author on request (shi16@iu.edu).

| Directory | Contents |
|---|---|
| [data/](data/) | Benchmark metadata: utterance IDs, time stamps, quality codes, categories, splits |
| [curation/](curation/) | Audio preparation, spelling normalization, language tagging, split building |
| [training/](training/) | LoRA adapter training and mixture-of-experts router training/inference |
| [evaluation/](evaluation/) | Inference for all baselines, sclite scoring, bootstrap significance tests |
| [results/](results/) | Table 1 numbers and per-utterance error counts (no text) |
| [third_party/](third_party/) | Our local changes to `omnilingual-asr`, as a patch |

The LoRA adapter and router weights are on Hugging Face:
[victorshi119/ICASSP2027_Jopara_ASR](https://huggingface.co/victorshi119/ICASSP2027_Jopara_ASR).

## Reproducing Table 1

Scoring is light: it runs sclite and needs no GPU.

```bash
# needs the test references and system outputs (on request), laid out as in the original project
export JOPARA_ROOT=/path/to/project        # where the hypothesis files and manifests live
python evaluation/scoring/make_table1.py   # writes results/table1/
```

Rows 8 and 9 are averaged over router seeds 13, 42, and 123. The bootstrap confidence intervals
quoted in the paper come from `evaluation/scoring/row8_seeds.py`, `row9_seeds.py`, and
`ci_check.py`.

## Running the full pipeline

The scripts were run on Indiana University's BigRed200 cluster (SLURM). Python scripts read
data from `$JOPARA_ROOT`, which should follow the original project layout. The `.sbatch` files are
kept as records of the exact commands we ran and contain cluster paths you will need to change.
Each directory's README gives the order of steps.

1. `curation/`: build the 3,288 candidate utterances from CEGPA, normalize spelling, tag
   languages, and build the splits.
2. `training/`: train the three LoRA adapters, then the router (three seeds).
3. `evaluation/`: decode the test set with every system and score it.

## Citation

```bibtex
@misc{shi2026jopara,
  title  = {Jopara {ASR}: A Community-Verified Benchmark and Baselines for Spontaneous
            {Spanish--Guaran\'{\i}} Code-Switching},
  author = {Shi, Wenchen and Shi, Shuju},
  note   = {Submitted to ICASSP 2027},
  year   = {2026}
}
```

## License

Code and metadata: [MIT](LICENSE). CEGPA and Common Voice keep their own licenses and terms of use.
