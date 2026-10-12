#!/usr/bin/env python3
"""initwin.py <run.cyc> <thread 0|1|2> <elf> [<run2.cyc> <elf2>]

Finds the thread's last INIT window in a Versim run of the INIT measurement build (LLK_PERF_INIT_ONLY): from the commit
of the restart point after the INIT entry park to the commit of the end clock read in llk_perf::init_zone. Reports
cycles, instructions, instruction-cache line fetches from L1, mispredicts, blocked cycles and the thread's own L1 data
accesses. With a second run, aligns the two committed-instruction streams (addresses in the changed function shift)
and prints where the cycle difference grows.
"""
import re
import subprocess
import sys

OD = "/home/nstojic/tt-metal/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objdump"


def n(v):
    try:
        return int(v, 2)
    except ValueError:
        return -1


def window_pcs(elf):
    """restart pc after the INIT entry park, end clock read pc, and the INIT body range"""
    dis = subprocess.run([OD, "-d", "-C", elf], capture_output=True, text=True).stdout
    sym = None
    lines = []
    for l in dis.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", l)
        if m:
            sym = m.group(2)
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+([0-9a-f]+)\s+(.*)$", l)
        if m:
            lines.append((int(m.group(1), 16), sym, m.group(3)))
    tramp = [x for x in lines if x[1] and x[1].startswith("void llk_perf::init_zone")]
    # restart point: the auipc/addi before the park's ebreak names it
    restart = None
    for i, (a, s, t) in enumerate(tramp):
        if t.startswith("auipc") and i + 1 < len(tramp):
            m = re.search(r"# ([0-9a-f]+) <", tramp[i + 1][2])
            if m:
                restart = int(m.group(1), 16)
                break
    # the wall clock loads (RISCV_DEBUG_REG_WALL_CLOCK_L at 0xffb121f0): the first is the start read, the second the end read
    clk = [a for a, s, t in tramp if t.startswith("lw") and "ffb121f0" in t]
    end = clk[1] if len(clk) > 1 else None
    body = [a for a, s, t in lines if s and "{lambda()#1}::operator()" in s]
    return restart, end, (min(body), max(body)) if body else None, {a: (s, t) for a, s, t in lines}


def load(cyc, t, restart, end):
    f = open(cyc)
    keys = f.readline().split()
    ix = {k: i for i, k in enumerate(keys)}
    g = lambda v, k: v[ix[k]] if k in ix else "0"
    rows = []
    for line in f:
        v = line.split()
        rows.append((n(g(v, f"t{t}_cmt_pc")), g(v, f"t{t}_ic_rden") == "1", g(v, f"t{t}_mp") == "1",
                     g(v, f"t{t}_ibuf_wren") == "1" and g(v, f"t{t}_ibuf_ready") != "1",
                     g(v, f"t{t}_l1_rden") == "1" or g(v, f"t{t}_l1_wren") == "1", n(g(v, f"t{t}_ic_addr"))))
    starts = [i for i in range(1, len(rows)) if rows[i][0] == restart and rows[i - 1][0] != restart]
    if not starts:
        sys.exit(f"{cyc}: thread {t} never commits the restart pc {restart:#x}")
    s = starts[-1]
    e = next((i for i in range(s, len(rows)) if rows[i][0] == end), None)
    if e is None:
        sys.exit(f"{cyc}: no end read {end:#x} after the last restart")
    w = rows[s:e + 1]
    commits = []
    prev = None
    for i, r in enumerate(w):
        if r[0] != prev:
            commits.append((i, r[0]))
            prev = r[0]
    return w, commits, len(starts)


def summary(name, w, commits, nst):
    print(f"{name}: INIT window {len(w)} cycles ({nst} INIT launches seen, last used), {len(commits)} committed pcs, "
          f"code-line fetches from L1 {sum(r[1] for r in w)}, mispredicts {sum(r[2] for r in w)}, "
          f"cycles blocked on a full instruction queue {sum(r[3] for r in w)}, own L1 data accesses {sum(r[4] for r in w)}")


if __name__ == "__main__":
    t = int(sys.argv[2])
    r1, e1, b1, d1 = window_pcs(sys.argv[3])
    w1, c1, n1 = load(sys.argv[1], t, r1, e1)
    summary("A", w1, c1, n1)
    if len(sys.argv) > 5:
        r2, e2, b2, d2 = window_pcs(sys.argv[5])
        w2, c2, n2 = load(sys.argv[4], t, r2, e2)
        summary("B", w2, c2, n2)
        # align on instruction text (addresses differ when code moved): walk both commit streams
        ta = [d1.get(p, ("?", "?"))[1] for _, p in c1]
        tb = [d2.get(p, ("?", "?"))[1] for _, p in c2]
        i = j = 0
        last = 0
        print("  where the difference grows (cycle in A, cycle in B, delta, instruction in A / B):")
        while i < len(c1) and j < len(c2):
            if ta[i] != tb[j]:
                # skip inserted nops in either stream
                if tb[j] == "nop":
                    j += 1
                    continue
                if ta[i] == "nop":
                    i += 1
                    continue
            delta = c2[j][0] - c1[i][0]
            if delta != last:
                pa, pb = c1[i][1], c2[j][1]
                print(f"    {c1[i][0]:6d} {c2[j][0]:6d} {delta:+4d}  {pa:#x} {d1.get(pa, ('?', '?'))[1][:40]} / {pb:#x} "
                      f"{d2.get(pb, ('?', '?'))[1][:40]}  fetches A/B so far {sum(r[1] for r in w1[:c1[i][0]])}/{sum(r[1] for r in w2[:c2[j][0]])}")
                last = delta
            i += 1
            j += 1
