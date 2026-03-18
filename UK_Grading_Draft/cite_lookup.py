#!/usr/bin/env python3
"""
Look up citation keys in reference.bib and reference_appendix.bib
Usage: python cite_lookup.py [KEY]  or  python cite_lookup.py KEY1 KEY2 ...
With no args: list all keys with titles
"""
import re
import sys
from pathlib import Path

BIB_DIR = Path(__file__).parent
BIB_FILES = [BIB_DIR / "reference.bib", BIB_DIR / "reference_appendix.bib"]


def parse_bib(filepath):
    """Parse .bib file and yield (key, entry_dict) for each entry."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    # Find entries by brace matching (handles nested {})
    pos = 0
    while True:
        m = re.search(r"@\w+\{([^,\s]+)\s*,\s*", text[pos:])
        if not m:
            break
        key = m.group(1).strip()
        # Find the opening brace of the entry (after @type)
        brace_start = pos + m.start() + m.group(0).index("{")
        body_start = pos + m.end()
        depth = 1
        i = brace_start + 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[body_start : i - 1]
        pos = i

        entry = {}
        for field in ["title", "author", "journal", "year", "institution"]:
            # Match field = {value}, field = "value", or field = value
            fm = re.search(
                rf'{field}\s*=\s*(?:"([^"]*)"|\{{([^{{]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}|(\d{{4}}))',
                body,
                re.DOTALL | re.IGNORECASE,
            )
            if fm:
                val = (fm.group(1) or fm.group(2) or fm.group(3) or "").strip()
                val = re.sub(r"\s+", " ", val)
                if field == "institution" and "title" not in entry:
                    entry["title"] = val  # fallback for techreports
                elif field != "institution":
                    entry[field] = val
        yield key, entry


def load_all():
    entries = {}
    for bib in BIB_FILES:
        if bib.exists():
            for key, entry in parse_bib(bib):
                if key not in entries:  # first occurrence wins
                    entries[key] = entry
    return entries


def main():
    entries = load_all()

    if len(sys.argv) == 1:
        print("=== Citation key → paper title index ===\n")
        for key in sorted(entries.keys()):
            title = entries[key].get("title", "(no title)")
            if len(title) > 70:
                title = title[:67] + "..."
            print(f"{key:35} → {title}")
        return

    for key in sys.argv[1:]:
        print(f"\\cite{{{key}}}")
        if key in entries:
            e = entries[key]
            for f in ["title", "author", "journal", "year"]:
                if f in e and e[f]:
                    print(f"  {f.capitalize():8}: {e[f]}")
        else:
            print("  NOT FOUND in .bib files")
        print()


if __name__ == "__main__":
    main()
