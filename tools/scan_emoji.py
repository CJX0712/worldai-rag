# -*- coding: utf-8 -*-
"""P0 gate: no emoji anywhere in the repo (they break on some
Windows consoles/encodings and look unprofessional in logs/docs).

Usage: python tools/scan_emoji.py   -> exit 1 if any found.
"""
import os
import sys

EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0xFE00, 0xFE0F),
    (0x1F000, 0x1F02F),
    (0x2B00, 0x2BFF),  # stars / squares used as decorative icons
]

SKIP_DIRS = {".git", "models", "data", "__pycache__", ".pytest-tmp", "node_modules"}
SKIP_FILES = {"scan_emoji.py"}
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg"}


def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in EMOJI_RANGES)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES or os.path.splitext(fn)[1] not in TEXT_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        for ch in line:
                            if is_emoji(ch):
                                findings.append(
                                    "{}:{}: U+{:04X}".format(
                                        os.path.relpath(path, root), ln, ord(ch)
                                    )
                                )
                                break
            except (OSError, UnicodeDecodeError):
                continue
    if findings:
        print("EMOJI FOUND (P0):")
        for f in findings:
            print("  " + f)
        return 1
    print("emoji scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
