"""Standalone scanner — fails on hardcoded SSH credentials under scripts/.

R12 — referenced by spec §6 and Subtask #3 review checklist.
Scope: ``le-wm-JR/scripts/*.py`` only. Pre-existing ``_remote_*`` /
``_probe_*`` helpers are pre-flagged in ``CLAUDE.md`` §Security and
excluded from this scan — they are tracked as the documented
known-issue R11.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# Files pre-flagged in CLAUDE.md §Security (R11) — excluded from scan.
LEGACY_PREFIXES = ("_remote_", "_probe_")

IP_PATTERN = re.compile(r"\b192\.168\.1\.3\b")
PWD_PATTERN = re.compile(
    r"(?ix)\b(password|pwd|passwd|LEWM_SSH_PASSWORD)\b"
    r"\s*[:=]\s*['\"]?[^'\"\s,)]+"
)


def _scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not IP_PATTERN.search(text):
        return []
    pwd_hits = [m.group(0) for m in PWD_PATTERN.finditer(text)]
    if not pwd_hits:
        return []
    return pwd_hits


def main() -> int:
    failures: list[tuple[Path, list[str]]] = []
    if not SCRIPTS.exists():
        print(f"OK: scripts/ does not exist at {SCRIPTS}")
        return 0

    for path in sorted(SCRIPTS.rglob("*.py")):
        if path.name.startswith(LEGACY_PREFIXES):
            continue
        hits = _scan(path)
        if hits:
            failures.append((path, hits))

    if failures:
        print("FAIL: hardcoded credential heuristic tripped:")
        for path, hits in failures:
            rel = path.relative_to(ROOT)
            print(f"  - {rel}: {hits}")
        return 1

    print(f"OK: scanned {sum(1 for _ in SCRIPTS.rglob('*.py'))} script(s); "
          "no new credential leaks detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
