from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID_DECISIONS = {"accept", "ambiguous", "reject"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a delta-reviewed ReCon1M audit CSV without overwriting the source review."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pending-means-model-decision",
        action="store_true",
        help="Treat human_review=pending as agreement with model_decision.",
    )
    args = parser.parse_args()

    if not args.pending_means_model_decision:
        raise ValueError(
            "Refusing to resolve pending rows. Pass --pending-means-model-decision "
            "only for a completed delta review."
        )

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    required = {"audit_id", "model_decision", "human_review"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    audit_ids = [row["audit_id"] for row in rows]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("audit_id values must be unique")

    final_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    overrides = 0
    for row in rows:
        model = row["model_decision"].strip().lower()
        review = row["human_review"].strip().lower()
        if model not in VALID_DECISIONS:
            raise ValueError(f"invalid model_decision for {row['audit_id']}: {model!r}")
        if review not in {"pending", *VALID_DECISIONS}:
            raise ValueError(f"invalid human_review for {row['audit_id']}: {review!r}")

        if review == "pending":
            final, source = model, "model_decision_confirmed_by_delta_review"
        else:
            final, source = review, "human_override"
            overrides += 1
        row["final_decision"] = final
        row["final_decision_source"] = source
        final_counts[final] += 1
        source_counts[source] += 1

    output_fields = fields + [
        field for field in ("final_decision", "final_decision_source") if field not in fields
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "records": len(rows),
                "human_overrides": overrides,
                "final_decisions": dict(final_counts),
                "decision_sources": dict(source_counts),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
