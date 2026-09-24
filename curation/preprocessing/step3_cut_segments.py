"""Step 3 of CEGPA preprocessing: cut each of the 3,288 utterances out of its source recording.

Writes <file_id>_<row index:05d>.wav (the utt_id naming used everywhere downstream).
Saved output for this run: Total rows 3288, segments with valid path 3288.

Extracted verbatim from the Colab notebook Pre_Data_Processing_Meta_Omnilingual_ASR.ipynb, code cells 112-117
(only change: shell "!" lines commented out). Paths are the original Colab/Google Drive paths;
edit them to point at your own copy of the data.
"""

# ---- notebook cell 112 ----
# ============================================================================
# Cell 1: Imports & path setup
# ============================================================================

import os
from pathlib import Path
import pandas as pd

# Project root in your Colab / Google Drive
proj_root = Path("/content/drive/MyDrive/635 Speech Processing/Final Project 635/omnilingual-asr-colab-main/colab_test_package")

data_dir   = proj_root / "data"
audio_root = proj_root / "audio"   # directory with tr-005.wav, tr-006.wav, etc.
out_seg_dir = proj_root / "audio_segments_full_v3"  # new directory for trimmed full dataset

print("proj_root :", proj_root)
print("data_dir  :", data_dir)
print("audio_root:", audio_root)
print("out_seg_dir:", out_seg_dir)

# ---- notebook cell 113 ----
# ============================================================================
# Cell 2: Load FINAL_CLEANED full dataset
# ============================================================================

full_csv_path = data_dir / "omnilingual_FULL_dataset_FINAL_CLEANED_norm_allreps_inaud.csv"

full_df = pd.read_csv(full_csv_path)

print("Loaded full dataset from:", full_csv_path)
print("Rows:", len(full_df))
print("Columns:", full_df.columns.tolist())

full_df.head()

# ---- notebook cell 115 ----
# ============================================================================
# Cell 4: Remap audio paths to local audio_root
# ============================================================================

def remap_audio_path(path_str: str, audio_root: Path) -> str:
    """
    Take a path like '/N/slate/wencshi/JoparaASR/data/tr-005.wav'
    and map it to 'audio_root / tr-005.wav'.
    """
    if not isinstance(path_str, str):
        return None
    basename = os.path.basename(path_str)
    return str(audio_root / basename)

full_df["audio_mapped"] = full_df["audio"].apply(lambda p: remap_audio_path(p, audio_root))

print("Sample remapped paths:")
full_df[["audio", "audio_mapped"]].head()

# ---- notebook cell 116 ----
# ============================================================================
# Cell 5: Install pydub + ensure ffmpeg
# ============================================================================

# !pip -q install pydub

from pydub import AudioSegment
print("pydub ready (ffmpeg is usually preinstalled on Colab).")

# ---- notebook cell 117 ----
# ============================================================================
# Cell 6: Trim all full-dataset utterances into audio_segments_full_v3
# ============================================================================

out_seg_dir.mkdir(parents=True, exist_ok=True)

audio_cache = {}  # cache loaded source wavs
segment_paths = []

for idx, row in full_df.iterrows():
    src_path = row["audio_mapped"]
    start_t = row["start_time"]
    end_t   = row["end_time"]

    if not isinstance(src_path, str) or not os.path.exists(src_path):
        segment_paths.append(None)
        continue

    basename = os.path.basename(src_path)

    # Lazy-load each long wav once
    if basename not in audio_cache:
        try:
            audio_cache[basename] = AudioSegment.from_file(src_path)
        except Exception as e:
            print(f"[WARN] Could not load {src_path}: {e}")
            segment_paths.append(None)
            continue

    audio = audio_cache[basename]

    # Convert to ms
    start_ms = int(float(start_t) * 1000)
    end_ms   = int(float(end_t) * 1000)

    # Clamp bounds
    start_ms = max(0, start_ms)
    end_ms   = min(len(audio), max(start_ms + 1, end_ms))

    seg = audio[start_ms:end_ms]

    # name segments: fileid_rowindex
    file_id = row.get("file_id", basename.replace(".wav", ""))
    seg_name = f"{file_id}_{idx:05d}.wav"

    out_path = out_seg_dir / seg_name
    try:
        seg.export(out_path, format="wav")
        segment_paths.append(str(out_path))
    except Exception as e:
        print(f"[WARN] Failed to export segment {seg_name}: {e}")
        segment_paths.append(None)

full_df["segment_audio"] = segment_paths

print("\nDone trimming full dataset.")
print("Total rows:", len(full_df))
print("Segments with valid path:", full_df['segment_audio'].notna().sum())

full_df[["audio_mapped", "start_time", "end_time", "segment_audio"]].head()
