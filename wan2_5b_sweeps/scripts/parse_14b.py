#!/usr/bin/env python3
"""Extract warm-iteration E2E + section breakdown from Wan2.2-14B generate logs.

Sections are logged by log_event_section as `[>>] name` / `[<<] name` with loguru
timestamps; we pair them per iteration and report the last (warm) iteration.
"""
import glob
import re
import sys
from datetime import datetime

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
MARK = re.compile(r"\[(>>|<<)\]\s+(\w+)")
E2E = re.compile(r"E2E_14B \[([^\]]+)\] iter=(\d+): ([\d.]+)s")
CLIP = re.compile(r"CLIP_14B scores: mean=([\d.]+)")


def parse_ts(line):
    m = TS.search(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def main():
    for log in sorted(sys.argv[1:] or glob.glob("/home/ttuser/wan14b_*_s*.log")):
        starts, sections = {}, []  # name->ts ; list of (name, dur)
        e2es, clip = [], None
        for line in open(log, errors="ignore"):
            mk = MARK.search(line)
            if mk:
                ts = parse_ts(line)
                kind, name = mk.group(1), mk.group(2)
                if ts is None:
                    continue
                if kind == ">>":
                    starts[name] = ts
                elif name in starts:
                    sections.append((name, (ts - starts.pop(name)).total_seconds()))
            e = E2E.search(line)
            if e:
                e2es.append((e.group(1), int(e.group(2)), float(e.group(3))))
            c = CLIP.search(line)
            if c:
                clip = float(c.group(1))

        # collapse to per-section warm (last occurrence of each name)
        last = {}
        for name, dur in sections:
            last[name] = dur  # keep last => warm iteration
        print(f"\n===== {log} =====")
        for tag, it, dt in e2es:
            print(f"  E2E [{tag}] iter={it}: {dt:.2f}s")
        if clip is not None:
            print(f"  CLIP mean: {clip:.2f}")
        order = ["encoder", "prepare_latents", "denoising", "vae"]
        for name in order:
            if name in last:
                print(f"  {name:16s}: {last[name]:.2f}s (warm)")
        # derived denoise/step if E2E warm present
        steps = re.search(r"_s(\d+)\.log", log)
        if steps and "denoising" in last:
            n = int(steps.group(1))
            print(f"  denoise/step    : {last['denoising'] / n * 1000:.0f} ms  ({n} steps)")


if __name__ == "__main__":
    main()
