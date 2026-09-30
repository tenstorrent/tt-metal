# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static check of a TRISC1 disassembly for SFPU Dst read-after-write hazards.

BH ISA (Dst.md, Instruction scheduling): after an instruction writes Dst, the aligned 8x16 block containing the write
cannot be read for the next four cycles; only FPU / PACR readers are auto-stalled, SFPLOAD is NOT. So an SFPSTORE to
dst address A followed within 4 issued instructions by an SFPLOAD / SFPLOADMACRO of the same address reads stale data.
(Straight-line windows only; macro-scheduled stores are not modelled -- the generated passes are checked by
gen_sinkhorn_lm.py.)  usage: python3 dst_raw_check.py <trisc1.elf.dis> [window=4]
"""
import re
import sys

WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4
lines = open(sys.argv[1]).read().splitlines()
ins = []
for ln in lines:
    m = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)", ln)
    if m:
        ins.append((m.group(1), m.group(2), m.group(3)))
    elif re.match(r"^[0-9a-f]+ <\.L[0-9]+>:", ln) or re.match(r"^[0-9a-f]+ <[A-Za-z_]", ln):
        ins.append((None, "LABEL", ln))  # branch targets / functions (not the .LBB / .LM debug labels)
sfpu = []
for a, op, args in ins:
    if op == "LABEL" or not op.startswith("sfp") and op not in ("ttreplay",):
        if op == "LABEL" or op.startswith(("b", "j")):
            sfpu.append((a, "BARRIER", args))  # control flow: reset the window conservatively
        continue
    sfpu.append((a, op, args))
hits = 0
for k, (a, op, args) in enumerate(sfpu):
    if op != "sfpstore":
        continue
    f = args.split(",")
    addr = f[1]
    for d in range(1, WIN + 1):
        if k + d >= len(sfpu):
            break
        a2, op2, args2 = sfpu[k + d]
        if op2 == "BARRIER":
            break
        if op2 in ("sfpload", "sfploadmacro"):
            if args2.split(",")[1] == addr:
                hits += 1
                print(f"RAW  store @{a} addr {addr} -> {op2} @{a2} after {d} SFPU instr")
print(f"{hits} store->load same-address pairs within {WIN} SFPU instructions")
