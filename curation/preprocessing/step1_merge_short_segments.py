"""Step 1 of CEGPA preprocessing: 12,099 raw segments -> 6,635 merged segments.

Recovered verbatim from the author's project notes (the script was originally saved as
merge_short_utterances_final.py in a cluster directory that no longer exists). Only change:
blank lines dropped when copying the code out of the notes. Paths are the original cluster paths.

What it does: within each recording (file_id), segments are sorted by start time and a
segment keeps absorbing the next one while its span is shorter than 2 s; it is written out
once it reaches 2 s, or as soon as it exceeds 50 s. The speaker column is not used.

How it was run: the 12,099 segments had first been
split at random, row by row, into omnilingual_train_verified.csv (8,590 rows) and
omnilingual_val_verified.csv (3,509 rows), and each file was merged separately. Within each
file, "consecutive" segments can therefore skip over rows that went to the other file, so a
merged span can include silence and other speakers' turns. The resulting 6,635-row table is
available from the authors on request.
"""
import pandas as pd
import soundfile as sf
import numpy as np
import os
from pathlib import Path
print("="*60)
print("Merging Short Utterances (< 2s)")
print("="*60)
# Load verified data
train_df = pd.read_csv('/N/slate/wencshi/JoparaASR/data/omnilingual_train_verified.csv')
val_df = pd.read_csv('/N/slate/wencshi/JoparaASR/data/omnilingual_val_verified.csv')
print(f"\nBEFORE MERGING:")
print(f"  Train: {len(train_df)} samples, {(train_df['duration'] < 2).sum()} < 2s")
print(f"  Val: {len(val_df)} samples, {(val_df['duration'] < 2).sum()} < 2s")
def merge_short_segments(df, min_duration=2.0, max_duration=50.0):
    """
    Merge consecutive short utterances from the same audio file.
    Uses start_time and end_time to create merged segments.
    """
    merged_rows = []
    # Group by file_id to process each conversation separately
    for file_id, group_df in df.groupby('file_id'):
        # Sort by start_time
        group_df = group_df.sort_values('start_time').reset_index(drop=True)
        buffer = None
        for idx, row in group_df.iterrows():
            duration = row['duration']
            if buffer is None:
                # Start new buffer
                buffer = {
                    'audio': row['audio'],
                    'texts': [row['text']],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'total_duration': duration,
                    'language': row['language'],
                    'file_id': row['file_id'],
                    'num_merged': 1
                }
            else:
                # Check if we should merge
                if buffer['total_duration'] < min_duration and buffer['audio'] == row['audio']:
                    # Merge with buffer
                    buffer['texts'].append(row['text'])
                    buffer['end_time'] = row['end_time']  # Extend end time
                    buffer['total_duration'] = buffer['end_time'] - buffer['start_time']
                    buffer['num_merged'] += 1
                else:
                    # Flush buffer (it's long enough or different file)
                    merged_rows.append({
                        'audio': buffer['audio'],
                        'text': ' '.join(buffer['texts']),
                        'start_time': buffer['start_time'],
                        'end_time': buffer['end_time'],
                        'duration': buffer['total_duration'],
                        'language': buffer['language'],
                        'file_id': buffer['file_id'],
                        'num_merged': buffer['num_merged']
                    })
                    # Start new buffer with current row
                    buffer = {
                        'audio': row['audio'],
                        'texts': [row['text']],
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'total_duration': duration,
                        'language': row['language'],
                        'file_id': row['file_id'],
                        'num_merged': 1
                    }
                # Check if buffer exceeds max_duration
                if buffer['total_duration'] > max_duration:
                    # Flush immediately
                    merged_rows.append({
                        'audio': buffer['audio'],
                        'text': ' '.join(buffer['texts']),
                        'start_time': buffer['start_time'],
                        'end_time': buffer['end_time'],
                        'duration': buffer['total_duration'],
                        'language': buffer['language'],
                        'file_id': buffer['file_id'],
                        'num_merged': buffer['num_merged']
                    })
                    buffer = None
        # Don't forget last buffer
        if buffer is not None:
            merged_rows.append({
                'audio': buffer['audio'],
                'text': ' '.join(buffer['texts']),
                'start_time': buffer['start_time'],
                'end_time': buffer['end_time'],
                'duration': buffer['total_duration'],
                'language': buffer['language'],
                'file_id': buffer['file_id'],
                'num_merged': buffer['num_merged']
            })
    return pd.DataFrame(merged_rows)
# Merge train and val
print("\nMerging train set...")
train_merged = merge_short_segments(train_df, min_duration=2.0, max_duration=50.0)
print("Merging val set...")
val_merged = merge_short_segments(val_df, min_duration=2.0, max_duration=50.0)
# Statistics
print(f"\n{'='*60}")
print("AFTER MERGING:")
print(f"{'='*60}")
print(f"\nTrain:")
print(f"  Original samples: {len(train_df)}")
print(f"  Merged samples: {len(train_merged)}")
print(f"  Reduction: {len(train_df) - len(train_merged)} samples ({(len(train_df) - len(train_merged))/len(train_df)*100:.1f}%)")
print(f"  Total utterances merged: {train_merged['num_merged'].sum()}")
print(f"  Avg utterances per merged sample: {train_merged['num_merged'].mean():.2f}")
print(f"\n  Duration distribution:")
print(f"    < 1s:    {(train_merged['duration'] < 1).sum():4d} ({(train_merged['duration'] < 1).sum()/len(train_merged)*100:5.1f}%)")
print(f"    < 2s:    {(train_merged['duration'] < 2).sum():4d} ({(train_merged['duration'] < 2).sum()/len(train_merged)*100:5.1f}%)")
print(f"    2-5s:    {((train_merged['duration'] >= 2) & (train_merged['duration'] <= 5)).sum():4d} ({((train_merged['duration'] >= 2) & (train_merged['duration'] <= 5)).sum()/len(train_merged)*100:5.1f}%)")
print(f"    5-10s:   {((train_merged['duration'] > 5) & (train_merged['duration'] <= 10)).sum():4d} ({((train_merged['duration'] > 5) & (train_merged['duration'] <= 10)).sum()/len(train_merged)*100:5.1f}%)")
print(f"    > 10s:   {(train_merged['duration'] > 10).sum():4d} ({(train_merged['duration'] > 10).sum()/len(train_merged)*100:5.1f}%)")
print(f"\nVal:")
print(f"  Original samples: {len(val_df)}")
print(f"  Merged samples: {len(val_merged)}")
print(f"  Reduction: {len(val_df) - len(val_merged)} samples ({(len(val_df) - len(val_merged))/len(val_df)*100:.1f}%)")
print(f"  Total utterances merged: {val_merged['num_merged'].sum()}")
print(f"  Avg utterances per merged sample: {val_merged['num_merged'].mean():.2f}")
print(f"\n  Duration distribution:")
print(f"    < 1s:    {(val_merged['duration'] < 1).sum():4d} ({(val_merged['duration'] < 1).sum()/len(val_merged)*100:5.1f}%)")
print(f"    < 2s:    {(val_merged['duration'] < 2).sum():4d} ({(val_merged['duration'] < 2).sum()/len(val_merged)*100:5.1f}%)")
print(f"    2-5s:    {((val_merged['duration'] >= 2) & (val_merged['duration'] <= 5)).sum():4d} ({((val_merged['duration'] >= 2) & (val_merged['duration'] <= 5)).sum()/len(val_merged)*100:5.1f}%)")
print(f"    5-10s:   {((val_merged['duration'] > 5) & (val_merged['duration'] <= 10)).sum():4d} ({((val_merged['duration'] > 5) & (val_merged['duration'] <= 10)).sum()/len(val_merged)*100:5.1f}%)")
print(f"    > 10s:   {(val_merged['duration'] > 10).sum():4d} ({(val_merged['duration'] > 10).sum()/len(val_merged)*100:5.1f}%)")
# Show examples
print(f"\n{'='*60}")
print("Example merged utterances:")
print(f"{'='*60}")
print("\nMerged from multiple short utterances:")
multi_merged = train_merged[train_merged['num_merged'] > 2].head(5)
for _, row in multi_merged.iterrows():
    print(f"\n  {row['duration']:.2f}s (merged {row['num_merged']} utterances):")
    print(f"    {row['text'][:150]}...")
# Save
print(f"\n{'='*60}")
print("Saving merged data...")
print(f"{'='*60}")
train_merged.to_csv('/N/slate/wencshi/JoparaASR/data/omnilingual_train_merged.csv', index=False)
val_merged.to_csv('/N/slate/wencshi/JoparaASR/data/omnilingual_val_merged.csv', index=False)
print(f"\n✓ Saved:")
print(f"  - omnilingual_train_merged.csv ({len(train_merged)} samples)")
print(f"  - omnilingual_val_merged.csv ({len(val_merged)} samples)")
print(f"\n{'='*60}")
print("✓ MERGING COMPLETE!")
print(f"{'='*60}")
print("\nNote: Audio extraction will be done during training using start_time/end_time.")
print("No need to create separate audio files - the model can read segments directly.")
