#!/usr/bin/env python3
"""Validate and summarize the paper slice manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run strict reproduction checks against the manifest.",
    )
    args = parser.parse_args()

    manifest_path = CODE_ROOT / "slice_manifests" / "paper_slices.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    print(f"manifest={manifest_path.relative_to(ROOT)}")
    print(f"version={manifest['version']}")
    print(f"main_table_qwen_cells={len(manifest['main_table_qwen'])}")
    print(f"sender_delta_claims={len(manifest['sender_delta_claims'])}")
    for note in manifest.get("notes", []):
        print(f"note: {note}")

    if args.check:
        cmd = [
            sys.executable,
            str(CODE_ROOT / "scripts" / "reproduce_paper.py"),
            "main-table-qwen",
            "--strict",
        ]
        subprocess.run(cmd, check=True)
        cmd = [
            sys.executable,
            str(CODE_ROOT / "scripts" / "reproduce_paper.py"),
            "sender-deltas",
            "--strict",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
