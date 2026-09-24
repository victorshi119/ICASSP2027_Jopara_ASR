#!/usr/bin/env python3
"""
Add final_corrected_reference_norm column to Everything.csv.

Rule:
  - Use spl_corrected_text if non-empty
  - Else use corrected_reference_norm
  - EXC rows: both are empty → result is empty

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/add_final_corrected_column.py
"""

import csv
import shutil
from pathlib import Path

CSV_IN  = Path("updated_csvs/June2026/Everything.csv")
BAK_OUT = Path("updated_csvs/June2026/Everything.csv.bak_pre_final_col")
NEW_COL = "final_corrected_reference_norm"


def main() -> None:
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows      = list(reader)

    if NEW_COL in fieldnames:
        print(f"Column '{NEW_COL}' already exists — nothing to do.")
        return

    shutil.copy2(CSV_IN, BAK_OUT)
    print(f"Backup written: {BAK_OUT}")

    new_fieldnames = fieldnames + [NEW_COL]
    filled = 0

    with open(CSV_IN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            spl = (row.get("spl_corrected_text") or "").strip()
            crn = (row.get("corrected_reference_norm") or "").strip()
            final = spl if spl else crn
            row[NEW_COL] = final
            if final:
                filled += 1
            writer.writerow(row)

    total = len(rows)
    print(f"Written {CSV_IN}")
    print(f"  {filled}/{total} rows have non-empty {NEW_COL}")
    print(f"  (of which {sum(1 for r in rows if (r.get('spl_corrected_text') or '').strip())} "
          f"came from spl_corrected_text)")


if __name__ == "__main__":
    main()
