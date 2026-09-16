from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rs_vlm.data.assemble import DEFAULT_EVAL_QUOTAS, DEFAULT_TRAIN_QUOTAS, assemble, load_examples, parse_quotas
from rs_vlm.data.dior import convert_dior
from rs_vlm.data.recon1m import PREDICATE_RULES
from rs_vlm.data.recon1m import build_dataset as build_recon1m_dataset
from rs_vlm.data.recon1m import select_pilot as select_recon1m_pilot
from rs_vlm.data.recon1m import scan as scan_recon1m
from rs_vlm.data.vrsbench import convert_vrsbench
from rs_vlm.schema import write_jsonl


def _dior(args: argparse.Namespace) -> None:
    examples = list(convert_dior(args.root, seed=args.seed, max_spatial_per_image=args.max_spatial_per_image))
    count = write_jsonl(args.output, examples)
    print(json.dumps({"written": count, "tasks": Counter(e.task_type for e in examples)}, default=dict, indent=2))


def _vrsbench(args: argparse.Namespace) -> None:
    examples = list(
        convert_vrsbench(
            args.annotations,
            args.images_root,
            source_split=args.source_split,
            task=args.task,
            seed=args.seed,
        )
    )
    count = write_jsonl(args.output, examples)
    print(json.dumps({"written": count, "tasks": Counter(e.task_type for e in examples)}, default=dict, indent=2))


def _assemble(args: argparse.Namespace) -> None:
    summary = assemble(
        args.inputs,
        args.output_dir,
        train_quotas=parse_quotas(args.train_quotas, DEFAULT_TRAIN_QUOTAS),
        eval_quotas=parse_quotas(args.eval_quotas, DEFAULT_EVAL_QUOTAS),
        grpo_size=args.grpo_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate(args: argparse.Namespace) -> None:
    examples = load_examples(args.inputs)
    missing = [image for example in examples for image in example.images if not Path(image).is_file()]
    report = {
        "records": len(examples),
        "tasks": dict(Counter(example.task_type for example in examples)),
        "splits": dict(Counter(str(example.meta.get("split")) for example in examples)),
        "missing_images": len(missing),
        "missing_examples": missing[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing and args.require_images:
        raise SystemExit(2)


def _recon_scan(args: argparse.Namespace) -> None:
    print(json.dumps(scan_recon1m(args.annotations, max_images=args.max_images), ensure_ascii=False, indent=2))


def _recon_build(args: argparse.Namespace) -> None:
    summary = build_recon1m_dataset(
        args.annotations,
        args.image_root,
        args.output_dir,
        seed=args.seed,
        max_images=args.max_images,
        max_queries=args.max_queries,
        audit_index=args.audit_index,
        audit_exclude_decisions=args.audit_exclude_decisions,
        exclude_query_ids=args.exclude_query_ids,
        predicates=args.predicates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _recon_select(args: argparse.Namespace) -> None:
    summary = select_recon1m_pilot(args.annotations, args.output, max_images=args.max_images, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare VRSBench and DIOR for Qwen3-VL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dior = subparsers.add_parser("dior", help="Convert Pascal VOC style DIOR annotations")
    dior.add_argument("--root", required=True)
    dior.add_argument("--output", required=True)
    dior.add_argument("--seed", type=int, default=42)
    dior.add_argument("--max-spatial-per-image", type=int, default=2)
    dior.set_defaults(func=_dior)

    vrs = subparsers.add_parser("vrsbench", help="Convert official VRSBench JSON")
    vrs.add_argument("--annotations", required=True)
    vrs.add_argument("--images-root", required=True)
    vrs.add_argument("--output", required=True)
    vrs.add_argument("--source-split", choices=("train", "validation", "eval", "test"), default="train")
    vrs.add_argument("--task", choices=("auto", "vqa", "grounding"), default="auto")
    vrs.add_argument("--seed", type=int, default=42)
    vrs.set_defaults(func=_vrsbench)

    merge = subparsers.add_parser("assemble", help="Sample task quotas and write SFT/GRPO splits")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--train-quotas", help="vqa=15000,count=10000,grounding=10000,spatial=5000")
    merge.add_argument("--eval-quotas", help="vqa=1500,count=1000,grounding=1000,spatial=500")
    merge.add_argument("--grpo-size", type=int, default=8000)
    merge.add_argument("--seed", type=int, default=42)
    merge.set_defaults(func=_assemble)

    validate = subparsers.add_parser("validate", help="Validate schema and image paths")
    validate.add_argument("--inputs", nargs="+", required=True)
    validate.add_argument("--require-images", action="store_true")
    validate.set_defaults(func=_validate)

    recon_scan = subparsers.add_parser("recon1m-scan", help="Scan ReCon1M relation Parquet files")
    recon_scan.add_argument("--annotations", nargs="+", required=True)
    recon_scan.add_argument("--max-images", type=int)
    recon_scan.set_defaults(func=_recon_scan)

    recon_select = subparsers.add_parser("recon1m-select", help="Select a balanced ReCon1M pilot subset")
    recon_select.add_argument("--annotations", nargs="+", required=True)
    recon_select.add_argument("--output", required=True)
    recon_select.add_argument("--max-images", type=int, default=500)
    recon_select.add_argument("--seed", type=int, default=42)
    recon_select.set_defaults(func=_recon_select)

    recon_build = subparsers.add_parser("recon1m-build", help="Build conditional set-grounding JSONL from ReCon1M")
    recon_build.add_argument("--annotations", nargs="+", required=True)
    recon_build.add_argument("--image-root", required=True)
    recon_build.add_argument("--output-dir", required=True)
    recon_build.add_argument("--seed", type=int, default=42)
    recon_build.add_argument("--max-images", type=int)
    recon_build.add_argument("--max-queries", type=int)
    recon_build.add_argument("--audit-index", help="Human audit CSV used to filter reviewed queries")
    recon_build.add_argument(
        "--audit-exclude-decisions",
        nargs="+",
        default=["reject"],
        choices=("accept", "ambiguous", "reject", "pending"),
    )
    recon_build.add_argument(
        "--exclude-query-ids",
        nargs="+",
        default=[],
        help="Explicit query IDs to exclude, e.g. 21129:parked-at 10430:parked-at",
    )
    recon_build.add_argument(
        "--predicates",
        nargs="+",
        choices=tuple(PREDICATE_RULES),
        help="Restrict generation to selected relation predicates.",
    )
    recon_build.set_defaults(func=_recon_build)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
