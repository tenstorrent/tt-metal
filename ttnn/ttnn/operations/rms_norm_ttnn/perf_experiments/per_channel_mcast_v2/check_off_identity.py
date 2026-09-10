"""`per_channel_mcast_v2` -- HALF ONE of the round-2 entry condition, proved on the SOURCE.

Round 1's candidate regressed NON-ENGAGED plans by 4-5% because its off path was not
byte-identical: it carried the multicast switch as an always-emitted compile-time ARG,
so every plan -- engaged or not -- compiled a different reader from a different argument
list.  Round 2 carries it as a DEFINE the host emits only on an engaged plan.

This script re-derives the off build's text: it deletes every `#ifdef RMS_PC_MCAST ...
#endif` block from k_mcast (keeping the `#else` arm where there is one, which is exactly
what the preprocessor does with the define absent) and diffs the result against k_base --
a byte-for-byte copy of the shipped kernels.  An empty diff means the OFF build compiles
the shipped translation unit.

The other half -- that the off build also emits the shipped CBs, semaphores, runtime args
and compile-time args -- is `check_off_descriptor.py`, which diffs the two descriptors on
device.

    python3 check_off_identity.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = "RMS_PC_MCAST"


def strip(text):
    out, depth, in_else = [], 0, []
    for line in text.splitlines(keepends=True):
        t = line.strip()
        if depth == 0:
            if t == f"#ifdef {GUARD}":
                depth, in_else = 1, [False]
                continue
            out.append(line)
            continue
        # inside a guarded block: track nested conditionals so an inner #endif
        # cannot be mistaken for ours
        if t.startswith("#if"):
            depth += 1
            in_else.append(False)
        elif t.startswith("#endif"):
            depth -= 1
            in_else.pop()
            if depth == 0:
                continue
        elif t.startswith("#else") and depth == 1:
            in_else[-1] = True
            continue
        if depth == 1 and in_else[0]:
            out.append(line)
    assert depth == 0, "unbalanced #ifdef"
    return "".join(out)


def main():
    bad = 0
    for name in sorted(p.name for p in (HERE / "k_base").iterdir()):
        base = (HERE / "k_base" / name).read_text()
        cand = (HERE / "k_mcast" / name).read_text()
        got = strip(cand)
        if got == base:
            n = cand.count(f"#ifdef {GUARD}")
            print(f"OFF-IDENTICAL  {name:32s} ({n} guarded block(s) removed, {len(base)} bytes match)")
        else:
            bad += 1
            print(f"OFF-DIVERGENT  {name}")
            import difflib

            for line in list(difflib.unified_diff(base.splitlines(), got.splitlines(), "k_base", "k_mcast(off)"))[:40]:
                print("   " + line)
    print("ENTRY-CONDITION(source): " + ("PASS" if bad == 0 else f"FAIL ({bad} file(s))"))
    return 1 if bad else 0


sys.exit(main())
