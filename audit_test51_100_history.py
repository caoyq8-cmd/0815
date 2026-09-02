#!/usr/bin/env python3
"""Conservative, read-only history audit for candidate holdout IDs test51--100.

The script scans local text artifacts, result paths, and optionally Git history. It
never claims that a sample is unseen; it reports only evidence found in the supplied
roots. Generated files are written under --output_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TEXT_EXTENSIONS = {
    ".py", ".sh", ".bash", ".zsh", ".slurm", ".txt", ".log", ".out", ".err",
    ".json", ".jsonl", ".csv", ".tsv", ".md", ".yaml", ".yml", ".toml",
    ".ini", ".cfg",
}
DEFAULT_EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", "node_modules", "wandb", "tensorboard", "final_holdout_audit",
}
AUDIT_TOOL_NAME_MARKERS = {
    "audit_test51_100_history",
    "run_test51_100_audit",
    "test51_100_audit.log",
    "test51_100_audit_tools",
    "audit_download_readme",
}
DATA_ASSET_HINTS = {
    "condition_cache", "self_consistent_cbs", "local_alpha01", "datasets",
    "dataset", "data", "speed", "dobs", "target", "prepared",
}
HIGH_EVIDENCE_HINTS = {
    "results", "result", "runs", "run", "checkpoints", "predictions", "evaluation",
    "sweep", "validation", "dev", "experiment", "logs", "output", "summary",
}
START_KEYS = r"(?:eval|test|sample|base|id)(?:_|-)(?:start|begin|first|id_start|start_id)"
END_KEYS = r"(?:eval|test|sample|base|id)(?:_|-)(?:end|stop|last|id_end|end_id)"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if parts & HIGH_EVIDENCE_HINTS:
        return "HIGH"
    if path.suffix.lower() in {".log", ".out", ".err", ".csv", ".json", ".jsonl"}:
        return "HIGH"
    if path.suffix.lower() in {".sh", ".bash", ".zsh", ".slurm", ".yaml", ".yml", ".toml"}:
        return "MEDIUM"
    if parts & DATA_ASSET_HINTS:
        return "DATA_ONLY"
    return "LOW"


def iter_files(roots: list[Path], output_dir: Path, excluded: set[str], max_bytes: int):
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for directory, dirnames, filenames in os.walk(root):
            current = Path(directory)
            dirnames[:] = [
                name for name in dirnames
                if name not in excluded and not (current / name).resolve().is_relative_to(output_dir.resolve())
            ]
            for name in filenames:
                path = current / name
                lower_name = name.lower()
                # Do not let this audit package become evidence against every
                # candidate merely because its filenames mention test51--100.
                if any(marker in lower_name for marker in AUDIT_TOOL_NAME_MARKERS):
                    continue
                try:
                    resolved = path.resolve()
                    if resolved in seen or resolved.is_relative_to(output_dir.resolve()):
                        continue
                    seen.add(resolved)
                    if path.stat().st_size > max_bytes:
                        continue
                except (OSError, RuntimeError):
                    continue
                yield path


def excerpt(text: str, start: int, end: int, limit: int = 240) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    value = " ".join(text[line_start:line_end].strip().split())
    return value[:limit]


def add_evidence(store: list[dict], sample_id: int, severity: str, kind: str,
                 path: str, line: int | str, matched: str, context: str) -> None:
    store.append({
        "sample_id": sample_id,
        "severity": severity,
        "kind": kind,
        "path": path,
        "line": line,
        "matched": matched,
        "context": context,
    })


def scan_file(path: Path, candidates: range, evidence: list[dict], repo_root: Path) -> None:
    rel = str(path.resolve().relative_to(repo_root.resolve())) if path.resolve().is_relative_to(repo_root.resolve()) else str(path)
    severity = classify_path(path)
    name_text = str(path)
    for sample_id in candidates:
        pattern = re.compile(rf"(?<!\d)test[_-]?0*{sample_id}(?!\d)", re.IGNORECASE)
        if pattern.search(name_text):
            add_evidence(evidence, sample_id, severity, "path_reference", rel, "", pattern.search(name_text).group(0), "file/path name")

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return
    try:
        raw = path.read_bytes()
        if b"\x00" in raw[:4096]:
            return
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return

    line_starts = [0]
    for match in re.finditer("\n", text):
        line_starts.append(match.end())

    def line_number(position: int) -> int:
        import bisect
        return bisect.bisect_right(line_starts, position)

    for sample_id in candidates:
        pattern = re.compile(rf"(?<!\d)test[_-]?0*{sample_id}(?!\d)", re.IGNORECASE)
        for match in pattern.finditer(text):
            add_evidence(evidence, sample_id, severity, "direct_text_reference", rel,
                         line_number(match.start()), match.group(0), excerpt(text, match.start(), match.end()))

    # Detect common start/end pairs in a local window. This catches CLI, JSON and config ranges.
    pair_pattern = re.compile(
        rf"(?P<start_key>{START_KEYS})\s*(?:=|:|\s)\s*(?P<lo>\d{{1,4}})"
        rf"[\s\S]{{0,500}}?"
        rf"(?P<end_key>{END_KEYS})\s*(?:=|:|\s)\s*(?P<hi>\d{{1,4}})",
        re.IGNORECASE,
    )
    reverse_pattern = re.compile(
        rf"(?P<end_key>{END_KEYS})\s*(?:=|:|\s)\s*(?P<hi>\d{{1,4}})"
        rf"[\s\S]{{0,500}}?"
        rf"(?P<start_key>{START_KEYS})\s*(?:=|:|\s)\s*(?P<lo>\d{{1,4}})",
        re.IGNORECASE,
    )
    for pattern in (pair_pattern, reverse_pattern):
        for match in pattern.finditer(text):
            lo, hi = int(match.group("lo")), int(match.group("hi"))
            if lo > hi or hi - lo > 5000:
                continue
            covered = [sample_id for sample_id in candidates if lo <= sample_id <= hi]
            for sample_id in covered:
                add_evidence(evidence, sample_id, severity, "range_reference", rel,
                             line_number(match.start()), f"{lo}..{hi}", excerpt(text, match.start(), match.end()))


def scan_git(repo_root: Path, candidates: range, evidence: list[dict]) -> dict:
    if not (repo_root / ".git").exists():
        return {"available": False, "reason": "no .git directory"}
    checked = 0
    for sample_id in candidates:
        regex = rf"test[_-]?0*{sample_id}([^0-9]|$)"
        command = [
            "git", "-C", str(repo_root), "log", "--all", "--format=%H%x09%s",
            f"-G{regex}", "--", ":(exclude)final_holdout_audit",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "reason": str(exc), "checked_ids": checked}
        checked += 1
        for line in result.stdout.splitlines()[:50]:
            if not line.strip():
                continue
            sha, _, subject = line.partition("\t")
            add_evidence(evidence, sample_id, "HIGH", "git_history_reference",
                         ".git", "", sha, subject[:240])
    return {"available": True, "checked_ids": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--historical_root", type=Path, action="append", default=[])
    parser.add_argument("--output_dir", type=Path, default=Path("final_holdout_audit/test51_100"))
    parser.add_argument("--start", type=int, default=51)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max_file_mb", type=float, default=20.0)
    parser.add_argument("--exclude_dir", action="append", default=[])
    parser.add_argument("--scan_git_history", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [repo_root] + [path.resolve() for path in args.historical_root]
    candidates = range(args.start, args.end + 1)
    excluded = DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)
    evidence: list[dict] = []
    scanned_files = 0
    for path in iter_files(roots, output_dir, excluded, int(args.max_file_mb * 1024 * 1024)):
        scanned_files += 1
        scan_file(path, candidates, evidence, repo_root)

    git_status = {"available": False, "reason": "disabled"}
    if args.scan_git_history:
        git_status = scan_git(repo_root, candidates, evidence)

    # Deduplicate mechanically identical evidence.
    unique = {}
    for item in evidence:
        key = tuple(item[name] for name in ("sample_id", "severity", "kind", "path", "line", "matched", "context"))
        unique[key] = item
    evidence = sorted(unique.values(), key=lambda row: (row["sample_id"], row["severity"], row["path"], str(row["line"])))
    by_id = defaultdict(list)
    for item in evidence:
        by_id[item["sample_id"]].append(item)

    status_rows = []
    provisional_rows = []
    for sample_id in candidates:
        items = by_id[sample_id]
        counts = Counter(item["severity"] for item in items)
        if counts["HIGH"]:
            status = "USED_OR_EXPOSED_HIGH"
        elif counts["MEDIUM"]:
            status = "USED_OR_EXPOSED_MEDIUM"
        elif counts["LOW"]:
            status = "CODE_ONLY_REFERENCE"
        elif counts["DATA_ONLY"]:
            status = "DATA_ASSET_ONLY"
        else:
            status = "NO_EVIDENCE_FOUND"
        status_rows.append({
            "sample_id": sample_id,
            "sample_name": f"test_{sample_id}",
            "status": status,
            "high_evidence": counts["HIGH"],
            "medium_evidence": counts["MEDIUM"],
            "low_evidence": counts["LOW"],
            "data_asset_evidence": counts["DATA_ONLY"],
            "total_evidence": len(items),
        })
        if status == "NO_EVIDENCE_FOUND":
            provisional_rows.append({
                "sample_id": sample_id,
                "sample_name": f"test_{sample_id}",
                "freeze_status": "PROVISIONAL_DO_NOT_FREEZE",
                "reason": "No evidence found in scanned roots; manual completeness review required",
            })

    write_csv(output_dir / "candidate_status.csv", status_rows, list(status_rows[0]))
    write_csv(output_dir / "evidence.csv", evidence,
              ["sample_id", "severity", "kind", "path", "line", "matched", "context"])
    write_csv(output_dir / "provisional_no_evidence_manifest.csv", provisional_rows,
              ["sample_id", "sample_name", "freeze_status", "reason"])

    summary = {
        "status": "AUDIT_COMPLETE_NOT_A_FREEZE_DECISION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "historical_roots": [str(path) for path in roots],
        "candidate_range": [args.start, args.end],
        "scanned_files": scanned_files,
        "git_history": git_status,
        "status_counts": dict(Counter(row["status"] for row in status_rows)),
        "no_evidence_ids": [row["sample_id"] for row in status_rows if row["status"] == "NO_EVIDENCE_FOUND"],
        "warning": (
            "NO_EVIDENCE_FOUND means only that no evidence was found in supplied local roots and reachable Git history. "
            "It does not prove that the sample was never viewed or used. Confirm that all historical logs, result folders, "
            "external notebooks, deleted branches and checkpoint-selection records are present before freezing a holdout."
        ),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print("test51--100 historical-use audit")
    print("=" * 100)
    print("scanned files :", scanned_files)
    print("git history   :", git_status)
    print("status counts :", summary["status_counts"])
    print("no evidence   :", summary["no_evidence_ids"])
    print("saved to      :", output_dir)
    print("IMPORTANT: this is an evidence audit, not an automatic holdout freeze decision.")


if __name__ == "__main__":
    main()
