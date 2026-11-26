#!/usr/bin/env python3
"""Compare compute-kernel text+data per build (pairs baseline vs candidate builds by their generated CT-arg headers).

usage: text_size.py <jit cache kernels/groupnorm_sc_N_1_HW_C_b_compute dir> <variantA> <variantB>
"""
import hashlib, os, re, subprocess, sys
from collections import defaultdict

SIZE = "/localdev/mmoscicki/gn-regen/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-size"
root, va, vb = sys.argv[1:4]
builds = defaultdict(dict)
for h in os.listdir(root):
    d = os.path.join(root, h)
    try:
        src = open(os.path.join(d, "chlkc_math.cpp")).read()
    except OSError:
        continue
    m = re.search(r"pass1_moments_accumulate/(kernels_[a-z0-9_]+)", src)
    if not m or m.group(1) not in (va, vb):
        continue
    key = hashlib.md5(
        (
            open(os.path.join(d, "defines_generated.h")).read() + open(os.path.join(d, "chlkc_descriptors.h")).read()
        ).encode()
    ).hexdigest()[:8]
    tot, per = 0, []
    for t in range(3):
        out = (
            subprocess.run([SIZE, f"{d}/trisc{t}/trisc{t}.elf.xip.elf"], capture_output=True, text=True)
            .stdout.split("\n")[1]
            .split()
        )
        per.append(int(out[0]) + int(out[1]))
        tot += per[-1]
    builds[key][m.group(1)] = (tot, per)
print(f"{'ct-key':10} {va:>28} {vb:>28} {'delta':>7}")
for k, v in sorted(builds.items()):
    a, b = v.get(va), v.get(vb)
    print(f"{k:10} {str(a):>28} {str(b):>28} {(b[0] - a[0]) if a and b else '':>7}")
