#!/usr/bin/env python3
"""Pre-download the fairseq2 tokenizer so Stage 4 (and other recipe runs) can use the cache.
Use the same HOME/FAIRSEQ2_CACHE_DIR as your batch job (e.g. slate) so the job finds the cache.
Tokenizer name must match configs (e.g. stage4_hdmole.yaml: tokenizer.name)."""
import argparse
import sys


def main():
    p = argparse.ArgumentParser(description="Pre-download tokenizer for HDMoLE recipe.")
    p.add_argument(
        "--name",
        default="omniASR_tokenizer_written_v2",
        help="Tokenizer card name (default: same as stage4_hdmole.yaml)",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable download progress")
    args = p.parse_args()

    try:
        from fairseq2.data.tokenizers.hub import load_tokenizer
    except ImportError as e:
        print("fatal: fairseq2 not available:", e, file=sys.stderr)
        print("Activate the omniasr_fairseq2 conda env and set FAIRSEQ2_CACHE_DIR.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.name} (cache: $FAIRSEQ2_CACHE_DIR)", flush=True)
    try:
        tok = load_tokenizer(args.name, progress=not args.no_progress)
        print("Tokenizer loaded successfully.", flush=True)
        return 0
    except Exception as e:
        print(f"fatal: tokenizer load failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main() or 0)
