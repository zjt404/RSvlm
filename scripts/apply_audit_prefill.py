from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID_DECISIONS = {"accept", "ambiguous", "reject"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multimodal audit suggestions into a review CSV")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    suggestions = json.loads(args.decisions.read_text(encoding="utf-8"))
    if not isinstance(suggestions, dict):
        raise ValueError("decisions file must contain an object keyed by audit_id")

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    audit_ids = {row["audit_id"] for row in rows}
    if set(suggestions) != audit_ids:
        missing = sorted(audit_ids - set(suggestions))
        extra = sorted(set(suggestions) - audit_ids)
        raise ValueError(f"suggestion IDs do not match audit CSV; missing={missing}, extra={extra}")

    counts: Counter[str] = Counter()
    for row in rows:
        suggestion = suggestions[row["audit_id"]]
        decision = str(suggestion.get("decision", "")).strip().lower()
        reason = str(suggestion.get("reason", "")).strip()
        if decision not in VALID_DECISIONS or not reason:
            raise ValueError(f"invalid suggestion for {row['audit_id']}: {suggestion}")
        row["decision"] = decision
        row["notes"] = reason
        row["model_decision"] = decision
        row["model_reason"] = reason
        row["review_source"] = "codex_multimodal_visual_review_v1"
        row["human_review"] = "pending"
        counts[decision] += 1

    fields = original_fields + [
        field
        for field in ("model_decision", "model_reason", "review_source", "human_review")
        if field not in original_fields
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"records": len(rows), "decisions": dict(counts), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
