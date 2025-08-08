#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

LOG_PATTERN = re.compile(
    r"^(?P<path>/.+?):(?P<line>\d+):(?P<col>\d+):\s+(?:warning|error):\s+unused variable '(?P<var>[^']+)'"
)

def parse_log(log_path: str):
    issues = defaultdict(list)  # file_path -> list of (line_num, col_num, var_name)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.match(line.strip())
            if m:
                path = m.group("path")
                line_num = int(m.group("line"))
                col_num = int(m.group("col"))
                var_name = m.group("var")
                issues[path].append((line_num, col_num, var_name))
    return issues

def annotate_file(file_path: str, entries, dry_run: bool):
    if not os.path.isfile(file_path):
        print(f"[skip] File not found: {file_path}")
        return 0

    # Group by line number (keep columns/vars for context)
    by_line = {}
    for ln, col, var in entries:
        by_line.setdefault(ln, []).append((col, var))

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    changed = 0
    for ln in sorted(by_line.keys()):
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            print(f"[warn] Line out of range {ln} in {file_path}")
            continue

        src_line = lines[idx]

        # Skip if already annotated anywhere meaningful on this line
        if "[[maybe_unused]]" in src_line:
            continue

        # Convenience helpers
        def insert_at(pos: int, text: str, s: str) -> str:
            return s[:pos] + text + s[pos:]

        def first_non_ws_position(s: str) -> int:
            m = re.match(r"^(\s*)", s)
            return len(m.group(1)) if m else 0

        # Decide insertion position using heuristics and available context
        insertion_pos = None

        # Case 1: range-based for loop: place just after opening '('
        if 'for' in src_line and '(' in src_line and (':' in src_line or ';' in src_line):
            paren_idx = src_line.find('(')
            if paren_idx != -1:
                # Ensure we insert after any immediate whitespace after '('
                j = paren_idx + 1
                while j < len(src_line) and src_line[j] == ' ':
                    j += 1
                # Do not double-insert if already present between '(' and the colon/semicolon
                end_decl = src_line.find(':', j)
                if end_decl == -1:
                    end_decl = src_line.find(';', j)
                segment = src_line[j:end_decl] if end_decl != -1 else src_line[j:]
                if '[[maybe_unused]]' not in segment:
                    insertion_pos = j

        # Case 2: generic declaration on a single line – try to put before 'auto' if present
        if insertion_pos is None:
            auto_match = None
            for m in re.finditer(r"\bauto\b", src_line):
                auto_match = m
                break
            if auto_match is not None:
                insertion_pos = auto_match.start()

        # Case 3: fall back – before first non-whitespace token on the line
        if insertion_pos is None:
            insertion_pos = first_non_ws_position(src_line)

        # Sanity: avoid inserting in preprocessor or comment-only lines
        leading = src_line[:first_non_ws_position(src_line)]
        trimmed = src_line[first_non_ws_position(src_line):]
        if trimmed.startswith('#') or trimmed.startswith('//'):
            continue

        new_line = insert_at(insertion_pos, '[[maybe_unused]] ', src_line)
        if new_line != src_line:
            print(f"[change] {file_path}:{ln}")
            if not dry_run:
                lines[idx] = new_line
            changed += 1

    if changed and not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return changed

def main():
    ap = argparse.ArgumentParser(description="Annotate unused variables with [[maybe_unused]] based on build log.")
    ap.add_argument("log", nargs="?", default="debug.log", help="Path to build log (default: debug.log)")
    ap.add_argument("--dry-run", action="store_true", help="Show planned changes without editing files")
    args = ap.parse_args()

    issues = parse_log(args.log)
    if not issues:
        print("No unused variable entries found.")
        return

    total_changes = 0
    for file_path, entries in issues.items():
        total_changes += annotate_file(file_path, entries, args.dry_run)

    print(f"Done. Changes{' (planned)' if args.dry_run else ''}: {total_changes}")

if __name__ == "__main__":
    main()