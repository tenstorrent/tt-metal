"""Per-column rotating-sender view: python3 rot_view.py <dev csv> <run> <vx>..."""
import sys
import timeline

freq, runs = timeline.load(sys.argv[1])
big = [r for r in runs if len({c for c, *_ in r}) >= 60]
ev = big[int(sys.argv[2])]
sp = timeline.spans(ev)
t0 = min(s for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for s, _ in v)
f = lambda v: f"{(v - t0) / freq:.1f}"
for x in [int(a) for a in sys.argv[3:]]:
    for core in [(x, y) for y in range(2, 12)]:
        iss = sp.get((core, "BRISC", "bwr_issue"), [])
        snd = sp.get((core, "BRISC", "bwr_send"), [])
        wf = sp.get((core, "BRISC", "writer_weight_fetch"), [])
        print(
            core,
            "issue",
            f(iss[0][0]) + ".." + f(iss[-1][1]) + f" ({len(iss)})" if iss else "-",
            "send",
            [f(a) + "-" + f(b) for a, b in snd],
            "land",
            [f(b) for a, b in wf],
        )
