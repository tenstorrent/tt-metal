# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generates sinkhorn_lm_gen.hpp: the hand-scheduled SFPLOADMACRO passes of the E2 Sinkhorn (n = 4).

Pass = 16 elements in 4 groups of 4 (pass C: group = row i, element j, multiplier rc_j; pass R: group = column j,
element i, multiplier rr_i). Per element one SFPLOADMACRO (macro q = position in group): load m -> temp T, the macro's
MAD sub-unit does T = L(4+q) * T one cycle later (delay 0), its store sub-unit writes T back to m two issued
instructions later. Per group, issued: A1 acc = T0 + T1, A2 acc += T2, A3 acc += T3 (the plain formulation's sum
order), ST acc -> scratch. Registers: T = L0..L2 (SFPLOADMACRO VD must be L0..L3 for even DEST addresses), acc = L3,
multipliers L4..L7, L9 = 0, L10 = 1, L12 = 2.0 and L13 = eps (programmable constants).

Scheduling rules (BH SFPLOADMACRO spec; macro instructions have no interlocks, and a scheduled instruction wins over
an issued one on the same sub-unit, silently dropping the issued one):
  * LM at slot s: macro MAD executes with slot s+1 (so slot s+1 must not be a MAD-sub-unit instruction: ADD / MUL /
    MAD), its result is readable from slot s+3; the macro store executes with slot s+3 (so slot s+3 must not be an
    issued SFPSTORE) and reads T there.
  * T is reusable by a later LM from slot max(s+4, last ADD read + 1).
  * issued ADD chain: each ADD >= 2 slots after the previous one (no auto-stall), ST >= 2 slots after A3.
The schedule below was found by an exhaustive DFS (bench/../README.md); verify() re-checks every rule.
"""
from pathlib import Path

N, LOGIT0 = 4, 8
MIX = N * (N + 2)
SCR = MIX + 1

# best DFS schedule for one sums pass (42 slots): element indices e (group = e // 4, position = e % 4)
SUMS_TRACE = """LM 0;LM 1;LM 2;NOP;ADD 0 1;LM 3;LM 4;NOP;ADD 2;LM 5;NOP;ADD 3;LM 6;ST;ADD 4 5;LM 7;LM 8;NOP;ADD 6;LM 9;NOP;
ADD 7;LM 10;ST;ADD 8 9;LM 11;LM 12;NOP;ADD 10;LM 13;NOP;ADD 11;LM 14;ST;ADD 12 13;LM 15;NOP;ADD 14;NOP;ADD 15"""
# The last group's ST is dropped: its sum stays in L3 for the following recips (no Dst store -> load round trip;
# BH Dst.md: a Dst block written by an instruction cannot be read for the next four cycles, and SFPLOAD is not
# interlocked). The pass ends with the last ADD; the last macro store (LM 15 at slot 35) executes with slot 38.
NT = 3


def final_trace():
    """16 LMs, no sums: T reusable 4 slots after its LM."""
    out, free, s, e = [], [0] * NT, 0, 0
    while e < 16:
        t = e % NT
        if free[t] <= s:
            out.append(f"LM {e}")
            free[t] = s + 4
            e += 1
        else:
            out.append("NOP")
        s += 1
    out += ["NOP"] * 3  # let the last macro store execute (delays count issued instructions)
    return out


def parse(trace):
    return [x.strip() for x in trace.replace("\n", "").split(";") if x.strip()]


def verify(slots, sums):
    lm, adds, sts = {}, [], []
    for s, x in enumerate(slots):
        f = x.split()
        if f[0] == "LM":
            lm[int(f[1])] = s
        elif f[0] == "ADD":
            adds.append((s, [int(v) for v in f[1:]]))
        elif f[0] == "ST":
            sts.append(s)
    assert sorted(lm) == list(range(16))
    kind = {s: x.split()[0] for s, x in enumerate(slots)}
    for e, s in lm.items():
        assert kind.get(s + 1) not in ("ADD",), f"issued ADD at macro MAD slot {s + 1}"
        assert kind.get(s + 3) != "ST", f"issued ST at macro store slot {s + 3}"
        assert s + 3 < len(slots) + 3, "macro store too far past the end of the pass"
    # temp lifetimes
    reads = {e: [] for e in lm}
    for s, es in adds:
        for e in es:
            reads[e].append(s)
            assert s >= lm[e] + 3, f"ADD reads e{e} at {s} before its macro MAD result ({lm[e] + 3})"
    for e in lm:
        t = e % NT
        nxt = e + NT
        if nxt in lm:
            last = max([lm[e] + 3] + reads[e])
            assert lm[nxt] > last, f"e{nxt} overwrites T{t} at {lm[nxt]} before e{e}'s last use {last}"
    if sums:
        assert len(adds) == 12 and len(sts) == 3
        # RAW on the accumulator: A2 >= A1 + 2, A3 >= A2 + 2, ST >= A3 + 2 (A1 only writes it; ST -> A1 is WAR)
        for g in range(4):
            chain = [x[0] for x in adds if x[1][0] // 4 == g] + ([sts[g]] if g < 3 else [])
            assert all(b >= a + 2 for a, b in zip(chain, chain[1:])), f"accumulator chain hazard in group {g}"
        assert all(sts[g] < [x[0] for x in adds if x[1][0] // 4 == g + 1][0] for g in range(3))
        # the scratch sums are re-read by the next recips (>= 1 slot after the pass): keep >= 5 slots of distance
        assert all(len(slots) - st >= 5 for st in sts), "scratch store too close to the end of the pass"
        for g in range(4):
            a = [x for x in adds if x[1][0] // 4 == g]
            assert [x[1] for x in a] == [[4 * g, 4 * g + 1], [4 * g + 2], [4 * g + 3]], a
    return True


def emit(slots, name, col_major_groups):
    lines = [f"// {len(slots)} issue slots", f"sfpi_inline void {name}() {{"]
    for x in slots:
        f = x.split()
        if f[0] == "LM":
            e = int(f[1])
            g, p = divmod(e, 4)
            i, j = (p, g) if col_major_groups else (g, p)
            addr = 2 * (LOGIT0 + i * N + j)
            t = e % NT
            lines.append(f"    TTI_SFPLOADMACRO(({p} << 2) | {t}, 0, 7, {addr});  // m[{i}][{j}] *= L{4 + p}")
        elif f[0] == "ADD":
            es = [int(v) for v in f[1:]]
            if len(es) == 2:
                lines.append(f"    TTI_SFPADD(10, {es[0] % NT}, {es[1] % NT}, 3, 0);  // acc = e{es[0]} + e{es[1]}")
            else:
                lines.append(f"    TTI_SFPADD(10, 3, {es[0] % NT}, 3, 0);  // acc += e{es[0]}")
        elif f[0] == "ST":
            lines.append("    TTI_SFPSTORE(3, 0, 7, ADDR);".replace("ADDR", "SCR_ADDR"))
        else:
            lines.append("    TTI_SFPNOP;")
    # fix up ST addresses (group order)
    g = 0
    for k, ln in enumerate(lines):
        if "SCR_ADDR" in ln:
            lines[k] = ln.replace("SCR_ADDR", str(2 * (SCR + g))) + f"  // group sum {g} -> scratch"
            g += 1
    lines.append("}")
    return "\n".join(lines)


def main():
    sums = parse(SUMS_TRACE)
    fin = final_trace()
    verify(sums, True)
    verify(fin, False)
    hdr = [
        "// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.",
        "// SPDX-License-Identifier: Apache-2.0",
        "// GENERATED by bench/gen_sinkhorn_lm.py -- do not edit. Hand-scheduled SFPLOADMACRO Sinkhorn passes (n = 4).",
        "#pragma once",
        "",
        "// pass C: m[i][j] *= rc_j (L4 + j); row sums (j order) -> scratch slot SCR + i",
        emit(sums, "skm_pass_c_sums", False),
        "",
        "// pass R: m[i][j] *= rr_i (L4 + i); column sums (i order) -> scratch slot SCR + j",
        emit(sums, "skm_pass_r_sums", True),
        "",
        "// last pass C: m[i][j] *= rc_j, no sums",
        emit(fin, "skm_pass_c_final", False),
        "",
    ]
    out = Path(__file__).parent / "sinkhorn_lm_gen.hpp"
    out.write_text("\n".join(hdr))
    print("wrote", out, "sums slots", len(sums), "final slots", len(fin))


if __name__ == "__main__":
    main()
