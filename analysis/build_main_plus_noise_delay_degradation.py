from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    main_rows = _read_rows(args.main)
    extension_rows = [
        row for row in _read_rows(args.extension)
        if row.get("condition") == "noise-med-delay-3"
    ]
    rows = main_rows + extension_rows
    if not rows:
        raise ValueError("No rows found to merge.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(main_rows[0].keys())
    extra_fields = sorted({key for row in rows for key in row if key not in fieldnames})
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"({len(main_rows)} main + {len(extension_rows)} noise+delay extension)."
    )


if __name__ == "__main__":
    main()
