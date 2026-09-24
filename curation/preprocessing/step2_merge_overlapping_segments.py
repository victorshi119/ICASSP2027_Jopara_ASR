"""Step 2 of CEGPA preprocessing: 6,635 segments -> 3,288 candidate utterances.

Within each recording, segments whose time spans overlap are merged into one (texts joined,
duplicates dropped). The notebook's saved output for this run reads: Original 6,635 / Cleaned 3,288.

Extracted verbatim from the Colab notebook Pre_Data_Processing_Meta_Omnilingual_ASR.ipynb, code cells 11-19
(only change: shell "!" lines commented out). Paths are the original Colab/Google Drive paths;
edit them to point at your own copy of the data.
"""

# ---- notebook cell 11 ----
# ==============================================================================
# CELL 1: Import Libraries and Setup
# ==============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully!")

# ---- notebook cell 12 ----
# ==============================================================================
# CELL 2: Load Original Dataset
# ==============================================================================
# Update this path to match your file location
INPUT_FILE = "/content/drive/MyDrive/635 Speech Processing/Final Project 635/omnilingual-asr-colab-main/colab_test_package/data/omnilingual_FULL_dataset_FIXED_with_norm.csv"
OUTPUT_FILE = INPUT_FILE.replace('.csv', '_CLEANED.csv')

print("="*80)
print("LOADING ORIGINAL DATASET")
print("="*80)

# Load the dataset
df_original = pd.read_csv(INPUT_FILE)

print(f"\n✓ Loaded: {INPUT_FILE}")
print(f"  Total rows: {len(df_original):,}")
print(f"  Total duration: {df_original['duration'].sum()/3600:.2f} hours")
print(f"  File IDs: {sorted(df_original['file_id'].unique())}")
print(f"  Columns: {list(df_original.columns)}")

# ---- notebook cell 13 ----
# ==============================================================================
# CELL 3: Analyze Original Dataset Statistics
# ==============================================================================
print("="*80)
print("ORIGINAL DATASET STATISTICS")
print("="*80)

def get_duration_stats(df):
    """Calculate duration statistics"""
    durations = df['duration']

    # Create duration bins
    bins = [0, 2, 5, 10, float('inf')]
    labels = ['< 2s', '2-5s', '5-10s', '> 10s']

    duration_ranges = pd.cut(durations, bins=bins, labels=labels, right=False)
    distribution = duration_ranges.value_counts().sort_index()
    percentages = (distribution / len(df) * 100).round(1)

    stats = {
        'count': len(df),
        'mean': durations.mean(),
        'median': durations.median(),
        'std': durations.std(),
        'min': durations.min(),
        'max': durations.max(),
        'total_hours': durations.sum() / 3600
    }

    return distribution, percentages, stats

# Get original statistics
dist_orig, pct_orig, stats_orig = get_duration_stats(df_original)

print(f"\n📊 Duration Distribution (Original):")
print(f"{'Range':<10} {'Count':>10} {'Percentage':>12}")
print("-" * 35)
for range_label in ['< 2s', '2-5s', '5-10s', '> 10s']:
    if range_label in dist_orig.index:
        print(f"{range_label:<10} {dist_orig[range_label]:>10,} {pct_orig[range_label]:>11.1f}%")

print(f"\n📈 Statistics (Original):")
print(f"  Total segments: {stats_orig['count']:,}")
print(f"  Total duration: {stats_orig['total_hours']:.2f} hours")
print(f"  Mean:           {stats_orig['mean']:.1f} seconds")
print(f"  Median:         {stats_orig['median']:.1f} seconds")
print(f"  Std Dev:        {stats_orig['std']:.1f} seconds")
print(f"  Min:            {stats_orig['min']:.1f} seconds")
print(f"  Max:            {stats_orig['max']:.1f} seconds")

# ---- notebook cell 14 ----
# ==============================================================================
# CELL 4: Define Overlap Removal Function
# ==============================================================================
def merge_overlapping_segments(df, file_id):
    """
    Merge overlapping segments for a specific file.
    Returns cleaned segments with overlaps merged.
    """
    # Get all segments for this file, sorted by start time
    file_df = df[df['file_id'] == file_id].sort_values('start_time').reset_index(drop=True)

    if len(file_df) == 0:
        return []

    cleaned_segments = []

    # Initialize with first segment
    current_segment = {
        'start_time': file_df.iloc[0]['start_time'],
        'end_time': file_df.iloc[0]['end_time'],
        'texts': [file_df.iloc[0]['text']],
        'original_texts': [file_df.iloc[0].get('original_text', file_df.iloc[0]['text'])],
        'languages': [file_df.iloc[0]['language']],
        'categories': [file_df.iloc[0]['category']],
        'merged_count': 1,
        'merged_indices': [0]
    }

    # Process remaining segments
    for i in range(1, len(file_df)):
        row = file_df.iloc[i]

        # Check if current segment overlaps with accumulated segment
        if row['start_time'] < current_segment['end_time']:
            # OVERLAP DETECTED - Merge this segment
            current_segment['end_time'] = max(current_segment['end_time'], row['end_time'])
            current_segment['texts'].append(row['text'])
            current_segment['original_texts'].append(row.get('original_text', row['text']))
            current_segment['languages'].append(row['language'])
            current_segment['categories'].append(row['category'])
            current_segment['merged_count'] += 1
            current_segment['merged_indices'].append(i)
        else:
            # NO OVERLAP - Save current segment and start new one
            cleaned_segments.append(current_segment)

            current_segment = {
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'texts': [row['text']],
                'original_texts': [row.get('original_text', row['text'])],
                'languages': [row['language']],
                'categories': [row['category']],
                'merged_count': 1,
                'merged_indices': [i]
            }

    # Don't forget the last segment
    cleaned_segments.append(current_segment)

    return cleaned_segments


def create_cleaned_row(segment_data, file_id, audio_template):
    """Create a cleaned row from merged segment data"""

    # Calculate new duration
    duration = segment_data['end_time'] - segment_data['start_time']

    # Merge texts (remove duplicates while preserving order)
    seen_texts = set()
    unique_texts = []
    for text in segment_data['texts']:
        if text not in seen_texts:
            seen_texts.add(text)
            unique_texts.append(text)
    combined_text = ' '.join(unique_texts)

    # Same for original texts
    seen_orig = set()
    unique_orig = []
    for text in segment_data['original_texts']:
        if text not in seen_orig:
            seen_orig.add(text)
            unique_orig.append(text)
    combined_original = ' '.join(unique_orig)

    # Determine primary language (most common)
    from collections import Counter
    language = Counter(segment_data['languages']).most_common(1)[0][0]

    # Determine primary category (most common)
    category = Counter(segment_data['categories']).most_common(1)[0][0]

    return {
        'audio': audio_template.format(file_id=file_id),
        'text': combined_text,
        'start_time': segment_data['start_time'],
        'end_time': segment_data['end_time'],
        'duration': duration,
        'language': language,
        'file_id': file_id,
        'category': category,
        'original_text': combined_original,
        'num_merged': segment_data['merged_count'],
        'is_merged': 'yes' if segment_data['merged_count'] > 1 else 'no'
    }

print("✓ Overlap removal functions defined!")

# ---- notebook cell 15 ----
# ==============================================================================
# CELL 5: Process Each File and Remove Overlaps
# ==============================================================================
print("="*80)
print("REMOVING OVERLAPS - PROCESSING FILES")
print("="*80)

# Get audio path template from original data
if 'audio' in df_original.columns and len(df_original) > 0:
    sample_audio = df_original.iloc[0]['audio']
    if '/' in sample_audio:
        audio_template = sample_audio.rsplit('/', 1)[0] + "/{file_id}.wav"
    else:
        audio_template = "{file_id}.wav"
else:
    audio_template = "/N/slate/wencshi/JoparaASR/data/{file_id}.wav"

print(f"\nAudio path template: {audio_template}\n")

# Process each file
all_cleaned_rows = []
processing_summary = []

for file_id in sorted(df_original['file_id'].unique()):
    # Count original segments
    original_count = len(df_original[df_original['file_id'] == file_id])

    # Merge overlapping segments
    merged_segments = merge_overlapping_segments(df_original, file_id)

    # Create cleaned rows
    for merged_seg in merged_segments:
        cleaned_row = create_cleaned_row(merged_seg, file_id, audio_template)
        all_cleaned_rows.append(cleaned_row)

    # Calculate statistics
    cleaned_count = len(merged_segments)
    reduction = original_count - cleaned_count
    reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0

    # Count how many were actually merged vs kept separate
    merged_segs = sum(1 for seg in merged_segments if seg['merged_count'] > 1)
    unmerged_segs = cleaned_count - merged_segs

    processing_summary.append({
        'file_id': file_id,
        'original': original_count,
        'cleaned': cleaned_count,
        'merged_segments': merged_segs,
        'unmerged_segments': unmerged_segs,
        'removed': reduction,
        'reduction_pct': reduction_pct
    })

    print(f"{file_id}:")
    print(f"  Original:  {original_count:>4} segments")
    print(f"  Cleaned:   {cleaned_count:>4} segments ({unmerged_segs} unmerged + {merged_segs} merged)")
    print(f"  Removed:   {reduction:>4} segments ({reduction_pct:.1f}% reduction)")

# Create cleaned dataframe
df_cleaned = pd.DataFrame(all_cleaned_rows)

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Original segments:  {len(df_original):>6,}")
print(f"  Cleaned segments:   {len(df_cleaned):>6,}")
print(f"  Removed segments:   {len(df_original) - len(df_cleaned):>6,} ({((len(df_original) - len(df_cleaned))/len(df_original)*100):.1f}%)")
print(f"{'='*80}")

# ---- notebook cell 16 ----
# ==============================================================================
# CELL 6: Calculate Cleaned Dataset Statistics
# ==============================================================================
print("="*80)
print("CLEANED DATASET STATISTICS")
print("="*80)

# Get cleaned statistics
dist_clean, pct_clean, stats_clean = get_duration_stats(df_cleaned)

print(f"\n📊 Duration Distribution After Merging:")
print(f"{'Range':<10} {'Count':>10} {'Percentage':>12}")
print("-" * 35)
for range_label in ['< 2s', '2-5s', '5-10s', '> 10s']:
    if range_label in dist_clean.index:
        print(f"{range_label:<10} {dist_clean[range_label]:>10,} {pct_clean[range_label]:>11.1f}%")

print(f"\n📈 Statistics:")
print(f"  Total segments: {stats_clean['count']:,}")
print(f"  Total duration: {stats_clean['total_hours']:.2f} hours")
print(f"  Mean:           {stats_clean['mean']:.1f} seconds")
print(f"  Median:         {stats_clean['median']:.1f} seconds")
print(f"  Std Dev:        {stats_clean['std']:.1f} seconds")
print(f"  Min:            {stats_clean['min']:.1f} seconds")
print(f"  Max:            {stats_clean['max']:.1f} seconds")

# Count merged vs unmerged
num_merged = len(df_cleaned[df_cleaned['is_merged'] == 'yes'])
num_unmerged = len(df_cleaned[df_cleaned['is_merged'] == 'no'])

print(f"\n📋 Segment Composition:")
print(f"  Merged segments (had overlaps):     {num_merged:>6,} ({num_merged/len(df_cleaned)*100:.1f}%)")
print(f"  Unmerged segments (no overlaps):    {num_unmerged:>6,} ({num_unmerged/len(df_cleaned)*100:.1f}%)")


# ---- notebook cell 18 ----
# ==============================================================================
# CELL 8: Verify No Overlaps Remain
# ==============================================================================
print("="*80)
print("VERIFICATION: Checking for Remaining Overlaps")
print("="*80)

total_overlaps = 0
overlaps_by_file = {}

for file_id in sorted(df_cleaned['file_id'].unique()):
    file_df = df_cleaned[df_cleaned['file_id'] == file_id].sort_values('start_time')

    overlaps = 0
    for i in range(len(file_df) - 1):
        row1 = file_df.iloc[i]
        row2 = file_df.iloc[i + 1]

        if row1['end_time'] > row2['start_time']:
            overlaps += 1

    overlaps_by_file[file_id] = overlaps
    total_overlaps += overlaps

if total_overlaps == 0:
    print("\n✅ SUCCESS! No overlaps detected in cleaned dataset.")
    print("   All segments are either sequential or have gaps between them.")
else:
    print(f"\n⚠️  WARNING: {total_overlaps} overlaps still present!")
    print("\nOverlaps by file:")
    for file_id, count in overlaps_by_file.items():
        if count > 0:
            print(f"  {file_id}: {count} overlaps")

# ---- notebook cell 19 ----
# ==============================================================================
# CELL 9: Save Cleaned Dataset
# ==============================================================================
print("="*80)
print("SAVING CLEANED DATASET")
print("="*80)

# Reorder columns to match original structure
desired_columns = [
    'audio', 'text', 'start_time', 'end_time', 'duration',
    'language', 'file_id', 'category', 'original_text',
    'num_merged', 'is_merged'
]

# Add any remaining columns from original that might be useful
for col in df_original.columns:
    if col not in desired_columns and col in df_cleaned.columns:
        desired_columns.append(col)

# Keep only columns that exist in cleaned data
final_columns = [col for col in desired_columns if col in df_cleaned.columns]
df_cleaned_final = df_cleaned[final_columns]

# Save to CSV
df_cleaned_final.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Cleaned dataset saved to:")
print(f"   {OUTPUT_FILE}")
print(f"\n📊 File details:")
print(f"   Rows:    {len(df_cleaned_final):,}")
print(f"   Columns: {len(df_cleaned_final.columns)}")
print(f"   Size:    {df_cleaned_final.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

print(f"\n📋 Columns in cleaned dataset:")
for i, col in enumerate(df_cleaned_final.columns, 1):
    print(f"   {i:2d}. {col}")
