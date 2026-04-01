"""
Validate canonical or legacy V-CoT JSONL datasets.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_schema import to_canonical_sample, validate_canonical_sample


def main():
    parser = argparse.ArgumentParser(description="Validate V-CoT JSONL data")
    parser.add_argument("--input", required=True, help="Path to a JSONL dataset file")
    parser.add_argument("--require-boxes", action="store_true", help="Fail if samples do not contain grounding boxes")
    parser.add_argument("--allow-missing-images", action="store_true", help="Do not fail when image files are missing")
    parser.add_argument("--max-errors", type=int, default=20, help="Maximum number of individual errors to print")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total = 0
    error_count = 0
    warning_count = 0
    printed_errors = 0
    printed_warnings = 0

    with open(args.input, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            total += 1
            try:
                record = json.loads(line)
                canonical = to_canonical_sample(record, sample_id=f"line_{index:06d}")
                errors, warnings = validate_canonical_sample(
                    canonical,
                    dataset_file=args.input,
                    repo_root=repo_root,
                    require_image=not args.allow_missing_images,
                    require_boxes=args.require_boxes,
                )
            except Exception as exc:
                errors = [f"line_{index:06d}: failed to parse sample: {exc}"]
                warnings = []

            error_count += len(errors)
            warning_count += len(warnings)

            for message in errors:
                if printed_errors < args.max_errors:
                    print("ERROR:", message)
                    printed_errors += 1

            for message in warnings:
                if printed_warnings < args.max_errors:
                    print("WARN:", message)
                    printed_warnings += 1

    print("\nValidation summary")
    print("==================")
    print(f"File: {args.input}")
    print(f"Samples checked: {total}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")

    if error_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
