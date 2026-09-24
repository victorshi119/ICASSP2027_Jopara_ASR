"""Step 4 of CEGPA preprocessing: convert the 3,288 clips to 16 kHz mono 16-bit PCM.

Saved output for this run: Converted 3288 files, skipped 0.

Extracted verbatim from the Colab notebook L635_Final_Presentation_Demo.ipynb, code cells 32-34
(only change: shell "!" lines commented out). Paths are the original Colab/Google Drive paths;
edit them to point at your own copy of the data.
"""

# ---- notebook cell 32 ----
# Normalize the full dataset
import os
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
from tqdm import tqdm

# --- MODIFY THIS TO MATCH YOUR PATH ---
BASE = Path("/content/drive/MyDrive/635 Speech Processing/Final Project 635/omnilingual-asr-colab-main/colab_test_package")

SRC = BASE / "audio_segments_full_v3"       # <-- source directory
DST = BASE / "audio_segments_full_v3_16k"   # <-- output directory

DST.mkdir(parents=True, exist_ok=True)

print("Source:", SRC)
print("Destination:", DST)

# ---- notebook cell 33 ----
import soundfile as sf
from collections import Counter

print("🔍 Scanning WAV metadata...\n")

sample_rates = Counter()
subtypes = Counter()

wav_files = list(SRC.rglob("*.wav"))
print(f"Found {len(wav_files)} WAV files in {SRC}")

for f in wav_files:
    try:
        info = sf.info(str(f))
        sample_rates[info.samplerate] += 1
        subtypes[info.subtype] += 1
    except:
        print("⚠️ Error reading file:", f)

print("\n🎧 Sample rates found:")
for sr, count in sample_rates.items():
    print(f"  {sr} Hz : {count} files")

print("\n💾 Subtypes found:")
for st, count in subtypes.items():
    print(f"  {st} : {count} files")

# ---- notebook cell 34 ----
print("🎼 Converting all audio in full_v3 → 16kHz mono PCM_16...\n")

converted = 0
skipped = 0

for src_file in tqdm(wav_files):
    try:
        # Output path mirrors directory structure
        rel = src_file.relative_to(SRC)
        dst_file = DST / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Load using pydub
        audio = AudioSegment.from_file(src_file)

        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 2 bytes = 16-bit

        # Export
        audio.export(dst_file, format="wav")
        converted += 1

    except Exception as e:
        print(f"⚠️ Error converting {src_file}: {e}")
        skipped += 1

print("\n🎉 Conversion complete!")
print(f"Converted: {converted} files")
print(f"Skipped:   {skipped} files")
