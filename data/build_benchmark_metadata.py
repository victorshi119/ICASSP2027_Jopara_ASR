#!/usr/bin/env python3
"""Build the released benchmark metadata: labels and split assignments only, no transcript text.

Inputs ($JOPARA_ROOT, original project layout; not shipped):
  updated_csvs/June2026/Everything.csv          all 3,288 candidate utterances (curation output)
  updated_csvs/v2_test_subsets/test_*.csv        the 150-utterance test set
  updated_csvs/training_splits_v3/{train,dev}_*.csv   training/development pools
Outputs (data/metadata/):
  cegpa_utterances.csv    one row per candidate utterance: utt_id, session, start/end/duration (s),
                          quality codes, annotation flags, category, split
  common_voice_clips.csv  Common Voice Guaraní clip IDs used, and which split files include them

Quality code BCN in the working data is the "MTS" (multiple speakers) code in the paper.
"""
import csv, glob, os
from pathlib import Path

ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
OUT = Path(__file__).resolve().parent / "metadata"
CODE_NAMES = {"BCN": "MTS"}


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def codes(s):
    return ",".join(CODE_NAMES.get(t.strip(), t.strip()) for t in s.split(",") if t.strip())


split_of, cv_files = {}, {}
for f in sorted(glob.glob(str(ROOT / "updated_csvs/v2_test_subsets/test_*.csv"))) + \
         sorted(glob.glob(str(ROOT / "updated_csvs/training_splits_v3/*.csv"))):
    name = Path(f).stem
    split = name.split("_")[0]  # test / train / dev
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            u = stem(r["directory"])
            if u.startswith("tr-"):
                assert split_of.get(u, split) == split, f"{u} in more than one split"
                split_of[u] = split
            else:
                cv_files.setdefault(u, []).append(name)

OUT.mkdir(parents=True, exist_ok=True)
n = 0
with open(ROOT / "updated_csvs/June2026/Everything.csv", newline="", encoding="utf-8") as fh, \
     open(OUT / "cegpa_utterances.csv", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["utt_id", "session", "start_time", "end_time", "duration", "quality_codes", "flags",
                "category", "split"])
    for r in csv.DictReader(fh):
        u = stem(r["directory"])
        w.writerow([u, r["file_id"], r["start_time"], r["end_time"], r["duration"], codes(r["quality_comments"]),
                    codes(r["additional_comments"]), r["v2_category"], split_of.get(u, "")])
        n += 1
with open(OUT / "common_voice_clips.csv", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["clip_id", "split_files"])
    for u in sorted(cv_files):
        w.writerow([u, ";".join(sorted(set(cv_files[u])))])

counts = {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "dev", "test")}
print(f"{n} CEGPA utterances; in splits: {counts} (total {sum(counts.values())}); {len(cv_files)} Common Voice clips")
