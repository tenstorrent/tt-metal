# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare each function with its reference_ counterpart, including relocations."""

import argparse
import difflib
import re
import subprocess
import sys


def read_labels(lines):
    # Compiler-generated label numbers differ between paired functions.
    # Resolve them to section + offset so branch destinations still participate
    # in the comparison, including branches whose encodings await relocation.
    labels = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4 or not re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
            continue
        if parts[-1].startswith(".L"):
            section = parts[-3].replace(".text.reference_", ".text.", 1)
            labels[parts[-1]] = f"{section}+0x{int(parts[0], 16):x}"
    return labels


def read_codegen(objdump, path):
    dump = subprocess.run(
        [objdump, "-t", "--special-syms", "-drz", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = dump.splitlines()
    labels = read_labels(lines)
    actual = {}
    reference = {}
    body = None
    for line in lines:
        section = re.fullmatch(r"Disassembly of section (.+):", line)
        if section:
            if not section[1].startswith(".text."):
                raise ValueError(
                    f"{path}: expected a named function section, got {section[1]}"
                )
            name = section[1].removeprefix(".text.")
            functions = actual
            if name.startswith("reference_"):
                functions = reference
                name = name.removeprefix("reference_")
            if name in functions:
                raise ValueError(f"{path}: duplicate section {section[1]}")
            body = functions[name] = []
            continue
        if body is None or not re.match(r"\s*[0-9a-fA-F]+:\s", line):
            continue

        # Object paths, symbol headings and explanatory objdump comments are
        # incidental. Keep instruction offsets, bytes, operands, and relocation
        # types/targets/addends. Both function sections start at zero.
        line = line.split("#", 1)[0]
        line = re.sub(r"\s*<[^>]+>", "", line)
        line = re.sub(r"(?<![\w.])\.L[\w.$]+", lambda m: labels.get(m[0], m[0]), line)
        body.append(" ".join(line.split()))

    bodies = list(actual.values()) + list(reference.values())
    if not bodies or not all(bodies):
        raise ValueError(f"{path}: expected nonempty disassembled function sections")
    return actual, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("object")
    args = parser.parse_args()

    try:
        actual, reference = read_codegen(args.objdump, args.object)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1

    failed = False
    for name in sorted(actual.keys() | reference.keys()):
        if name not in actual or name not in reference:
            missing = name if name not in actual else f"reference_{name}"
            print(f"{args.object}: missing function {missing}", file=sys.stderr)
            failed = True
        elif actual[name] != reference[name]:
            diff = difflib.unified_diff(
                reference[name],
                actual[name],
                fromfile=f"reference_{name}",
                tofile=name,
                lineterm="",
            )
            print(*diff, sep="\n", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
