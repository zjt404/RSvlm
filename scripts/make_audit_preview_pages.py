from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local HTML wrappers for audit images")
    parser.add_argument("--audit-dir", required=True, type=Path)
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    pages_dir = audit_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((audit_dir / "audit_index.csv").open(encoding="utf-8-sig", newline="")))

    links: list[str] = []
    for row in rows:
        audit_id = row["audit_id"]
        image_name = Path(row["image"]).name
        page_name = f"{audit_id}.html"
        title = f"{audit_id} | {row['predicate']} | {row['difficulty']}"
        page = (
            "<!doctype html><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><style>html,body{margin:0;background:#111;color:#fff;"
            "font:16px sans-serif;text-align:center}img{display:block;margin:auto;"
            "max-width:100vw;max-height:calc(100vh - 50px);width:auto;height:auto;object-fit:contain}"
            "header{padding:8px}</style><header>"
            + html.escape(title)
            + "</header><img src='../images/"
            + html.escape(image_name, quote=True)
            + "'>"
        )
        (pages_dir / page_name).write_text(page, encoding="utf-8")
        links.append(f"<li><a href='pages/{html.escape(page_name, quote=True)}'>" + html.escape(title) + "</a></li>")

    index = "<!doctype html><meta charset='utf-8'><title>Audit preview</title><ol>" + "".join(links) + "</ol>"
    (audit_dir / "preview_index.html").write_text(index, encoding="utf-8")
    print({"pages": len(rows), "output": str(pages_dir)})


if __name__ == "__main__":
    main()
