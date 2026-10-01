# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare single-return function pairs in objdump -drz output."""

import argparse
import difflib
import re
import sys

FUNCTION_SECTION_PREFIX = "Disassembly of section .text."
# objdump annotations: "# comment" and "<symbol+offset>".
OBJDUMP_ANNOTATION_PATTERN = r"#.*|<[^>]+>"
# Instruction and relocation lines start with a hexadecimal offset, e.g. "  10: ".
INSTRUCTION_OR_RELOCATION_PATTERN = r"\s*[0-9a-fA-F]+:\s"


def read_codegen(path):
    with open(path, encoding="utf-8") as dump:
        for line in dump:
            if not line.startswith(FUNCTION_SECTION_PREFIX):
                continue
            name = line.removeprefix(FUNCTION_SECTION_PREFIX).strip().removesuffix(":")
            code_lines = []
            for line in dump:
                if not re.match(INSTRUCTION_OR_RELOCATION_PATTERN, line):
                    continue
                # Local branch destinations are already encoded in the instructions.
                if "R_RISCV_BRANCH" in line:
                    continue
                line = re.sub(OBJDUMP_ANNOTATION_PATTERN, "", line)
                normalized_line = " ".join(line.split())
                code_lines.append(normalized_line)
                # Test wrappers have one final ret; extend this if that convention changes.
                if normalized_line.endswith(" ret"):
                    yield name, code_lines
                    break
            else:
                raise ValueError(f"{path}: missing ret in {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="text output from objdump -drz")
    args = parser.parse_args()

    failed = False
    found_pair = False
    try:
        functions = read_codegen(args.dump)
        for name, actual in functions:
            reference_name, reference = next(functions, ("", []))
            if name.startswith("reference_"):
                name, reference_name = reference_name, name
                actual, reference = reference, actual
            if reference_name != f"reference_{name}":
                raise ValueError(
                    f"{args.dump}: unpaired functions {name!r}, {reference_name!r}"
                )
            found_pair = True
            for line in difflib.unified_diff(
                reference, actual, fromfile=reference_name, tofile=name, lineterm=""
            ):
                print(line, file=sys.stderr)
                failed = True
        if not found_pair:
            raise ValueError(f"{args.dump}: no function pairs found")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
