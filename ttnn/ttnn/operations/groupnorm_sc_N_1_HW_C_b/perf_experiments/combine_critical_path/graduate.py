#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the bake-off switches of cand_src/compute.cpp for one variant into a switch-free kernel (graduate/) and
write its unified diff against the op's current compute kernel (graduate/compute.diff). Host-only.

usage: graduate.py [VPUB=1 FIN=1 MM1=0]
Only the CCP_* conditionals are resolved; every other preprocessor line passes through.
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "cand_src" / "compute.cpp"
OP = HERE.parents[1] / "kernels" / "groupnorm_sc_N_1_HW_C_b_compute.cpp"
OUT = HERE / "graduate" / "groupnorm_sc_N_1_HW_C_b_compute.cpp"
OLD_PUBLISH_COMMENT = (
    "// Per-channel moments [1 x W] -> the FPU-readable chan CBs (matmul in0 of the collapse). CopyTile from an\n"
    "// UnpackToDestFp32 CB is exact. One copy per block replaces a last-chunk variant of both reduces, which would\n"
    "// double the reduce instantiations (kernel text must fit the kernel-config ring).\n"
)


def resolve(text, sw):
    out, stack = [], []  # stack entries: (is_ccp, keep_this_branch, parent_active)
    active = True
    for line in text.splitlines(keepends=True):
        s = line.strip()
        m = re.match(r"#\s*(if|ifdef|ifndef)\s+(.*)", s)
        if m:
            kind, cond = m.groups()
            name = cond.split()[0] if cond else ""
            if name.startswith("CCP_"):
                val = sw.get(name, 0)
                keep = bool(val) if kind in ("if", "ifdef") else not val
                if kind == "ifdef":
                    keep = name in sw and bool(sw[name])
                stack.append((True, keep, active))
                active = active and keep
                continue
            stack.append((False, None, active))
            if active:
                out.append(line)
            continue
        if re.match(r"#\s*else\b", s):
            is_ccp, keep, parent = stack[-1]
            if is_ccp:
                stack[-1] = (True, not keep, parent)
                active = parent and not keep
                continue
            if stack[-1][2]:
                out.append(line)
            continue
        if re.match(r"#\s*endif\b", s):
            is_ccp, keep, parent = stack.pop()
            active = parent
            if not is_ccp and parent:
                out.append(line)
            continue
        if active:
            out.append(line)
    return "".join(out)


def main(argv):
    sw = {"CCP_VPUB": 1, "CCP_FIN": 1, "CCP_MM1": 0, "CCP_SUBZONES": 0}
    for a in argv:
        k, v = a.split("=")
        sw[f"CCP_{k}"] = int(v)
    text = SRC.read_text()
    # Drop the bake-off switch block (comments + defaults).
    head = text.index("// ---- combine_critical_path bake-off switches")
    tail = text.index("#include <cstdint>")
    text = text[:head].rstrip("\n") + "\n" + text[tail:]
    text = resolve(text, sw)
    # Scope braces that only existed to host the diagnosis sub-zones.
    tidy = [
        ("        {\n        publish_channel_moments();\n        }\n", "        publish_channel_moments();\n"),
        ("        {\n        matmul_with_map<false>(chan_scale_cb", "        matmul_with_map<false>(chan_scale_cb"),
        (
            "MatmulBlockShape::of(1, 1, 1, 1, W, 1));\n        }\n        }\n",
            "MatmulBlockShape::of(1, 1, 1, 1, W, 1));\n        }\n",
        ),
        (
            "        {\n            cb_wait_front(cb_group_shift, 1);\n            reconfig_data_format_srca(cb_group_shift);\n"
            "            copy_init(cb_group_shift);\n            copy_tile(cb_group_shift, 0, 2);\n        }\n",
            "        cb_wait_front(cb_group_shift, 1);\n        reconfig_data_format_srca(cb_group_shift);\n"
            "        copy_init(cb_group_shift);\n        copy_tile(cb_group_shift, 0, 2);\n",
        ),
        ("}\n\n\n#ifdef TRISC_MATH", "}\n\n#ifdef TRISC_MATH"),
    ]
    for a, b in tidy:
        if text.count(a) == 1:
            text = text.replace(a, b)
    if sw["CCP_VPUB"]:
        text = text.replace(OLD_PUBLISH_COMMENT, "")
    if sw["CCP_FIN"]:  # the cb_group_sums round trip and the chain's stats_out view are gone
        for dead in (
            "    CircularBuffer group_sums_cb(cb_group_sums);\n",
            "    DataflowBuffer group_sums_dfb(cb_group_sums);\n",
            "    DataflowBuffer stats_out_dfb(cb_stats_out);\n",
        ):
            assert text.count(dead) == 1, dead
            text = text.replace(dead, "")
    OUT.write_text(text)
    diff = difflib.unified_diff(
        OP.read_text().splitlines(keepends=True),
        text.splitlines(keepends=True),
        fromfile="a/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/kernels/groupnorm_sc_N_1_HW_C_b_compute.cpp",
        tofile="b/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/kernels/groupnorm_sc_N_1_HW_C_b_compute.cpp",
    )
    (HERE / "graduate" / "compute.diff").write_text("".join(diff))
    print(f"{OUT} ({sw})")


if __name__ == "__main__":
    main(sys.argv[1:])
