# Data

This directory holds **metadata only**: no audio and no transcript text.

## Getting the source data

- **CEGPA**, the *Corpus del Español y Guaraní Paraguayos de Asunción* (Bittar et al., 2022),
  is held by the California Language Archive, UC Berkeley: doi:10.7297/X2CC0ZW0. We use the
  seven interviews with Jopara speech.
- **Common Voice Guaraní** (scripted speech, release 24.0), from https://commonvoice.mozilla.org.
  It supplies the Guaraní adapter's read-aloud training clips and the Guaraní-only
  development set.

Corrected transcripts are available from the corresponding author on request (shi16@iu.edu).

## Files

`metadata/cegpa_utterances.csv` has one row per candidate utterance (3,288 in total):

| Column | Meaning |
|---|---|
| `utt_id` | `<session>_<index>`, also the clip file name (`<utt_id>.wav`) |
| `session` | CEGPA recording (`tr-005` … `tr-011`) |
| `start_time`, `end_time`, `duration` | Position in the session recording, in seconds |
| `quality_codes` | Annotator verdict: `Good`, `CRT-D`/`CRT-A` (transcript corrected by deleting / adding words to match the audio), `MTS` (multiple speakers talking in the same utterance; `BCN` in our working files), `EXC` (excluded). Codes can co-occur. |
| `flags` | Extra annotation flags: `BLK` (a name was blanked out of the audio for privacy), `SPL` (a Spanish word spelled the Guaraní way), `BCO` (backchannel voices overlap and degrade the audio), `NCT` (needs cutting), `USR` (annotator unsure of the phrase), `TMC` (too much coarticulation), `NSY` (noisy) |
| `category` | `spanish_only`, `guarani_only`, `mixed` (inter-sentential), or `jehe'a`, from the word-level tags. Empty for excluded utterances. |
| `split` | `train`, `dev`, or `test`. Empty for utterances outside the 2,340 verified ones. |

Split sizes: train 1,973, dev 217, test 150 (25 Guaraní-only, 67 inter-sentential, 33 *jehe'a*,
25 Spanish-only). The split is made at the utterance level, so test speakers and sessions also
appear in training (see the paper, §2.3).

`metadata/common_voice_clips.csv` lists the Common Voice clip IDs we used, and which of our
split files include each one.

## Rebuilding the metadata

`build_benchmark_metadata.py` generates both files from the curation outputs:

```bash
JOPARA_ROOT=/path/to/project python data/build_benchmark_metadata.py
```
