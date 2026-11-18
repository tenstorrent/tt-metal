# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone timelines from a device profiler log (no ttnn).

    python3 timeline.py <profile_log_device.csv> core <run_idx> <x,y>     one core's zones (us from the run's start)
    python3 timeline.py <profile_log_device.csv> zone <run_idx> <zone> [n] the n-th (0-based) execution of a zone on
                                                                          every core: start/end, sorted by end
    python3 timeline.py <profile_log_device.csv> runs                     run ids, core counts, span

run_idx is the index of the run among the multi-core runs, in host-id order (0 = first dispatch).
Times are relative to the earliest marker of the run over all cores.
"""

import collections
import csv
import gzip
import sys


def load(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        hdr = f.readline()
        freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rd = csv.reader(f)
        cols = [c.strip() for c in next(rd)]
        ix = {c: i for i, c in enumerate(cols)}
        runs = collections.defaultdict(list)
        for r in rd:
            if len(r) < len(cols):
                continue
            runs[int(r[ix["run host ID"]])].append(
                (
                    (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                    r[ix["RISC processor type"]].strip(),
                    int(r[ix["time[cycles since reset]"]]),
                    r[ix["zone name"]].strip(),
                    r[ix["type"]].strip(),
                )
            )
    big = sorted(k for k, v in runs.items() if len({e[0] for e in v}) >= 8)
    return freq, runs, big


def spans(events, freq):
    t0 = min(e[2] for e in events)
    st = collections.defaultdict(list)
    out = []
    for core, risc, t, z, k in sorted(events, key=lambda e: e[2]):
        key = (core, risc, z)
        if k == "ZONE_START":
            st[key].append(t)
        elif k == "ZONE_END" and st[key]:
            s = st[key].pop()
            out.append((core, risc, z, (s - t0) / freq, (t - t0) / freq))
    return out


def main():
    path, mode = sys.argv[1], sys.argv[2]
    freq, runs, big = load(path)
    if mode == "runs":
        for i, r in enumerate(big):
            ev = runs[r]
            print(i, r, len({e[0] for e in ev}), (max(e[2] for e in ev) - min(e[2] for e in ev)) / freq)
        return
    run = big[int(sys.argv[3])]
    sp = spans(runs[run], freq)
    if mode == "core":
        core = tuple(int(x) for x in sys.argv[4].split(","))
        for c, risc, z, s, e in sorted((x for x in sp if x[0] == core), key=lambda x: x[3]):
            if risc in ("TRISC_0", "TRISC_2") or "KERNEL" in z:
                continue
            print(f"{s:7.2f} {e:7.2f} {e - s:6.2f} {risc:8s} {z}")
    elif mode == "zone":
        zone = sys.argv[4]
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        per = collections.defaultdict(list)
        for c, risc, z, s, e in sp:
            if z == zone:
                per[(c, risc)].append((s, e))
        rows = []
        for (c, risc), lst in per.items():
            lst.sort()
            if n < len(lst):
                rows.append((lst[n][1], lst[n][0], c, risc))
        for e, s, c, risc in sorted(rows):
            print(f"{c} {risc:8s} {s:7.2f} -> {e:7.2f}")


if __name__ == "__main__":
    main()
