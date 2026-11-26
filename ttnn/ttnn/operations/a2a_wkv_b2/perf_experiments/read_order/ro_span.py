"""read_order: summary of per-replay spans (trace_gaps.py output): per replay the max span over chips, mean of
that over replays 1.., and the mean over all chips/replays. Usage: ro_span.py csv..."""
import subprocess, sys, re, os

tg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../perf/trace_gaps.py")
for p in sys.argv[1:]:
    out = subprocess.run([sys.executable, tg, p], capture_output=True, text=True).stdout
    spans = []
    for line in out.splitlines():
        if line.startswith("chip"):
            vals = [float(v) for v in re.findall(r"(\d+\.\d)\(", line)]
            spans.append(vals[1:])  # skip the first (eager-after) replay
    per_rep = [max(c[i] for c in spans) for i in range(min(len(c) for c in spans))]
    allm = sum(sum(c) for c in spans) / sum(len(c) for c in spans)
    print(
        f"{os.path.basename(p):12s} max-over-chips per replay {['%.1f' % v for v in per_rep]} mean {sum(per_rep)/len(per_rep):.1f} | mean span {allm:.1f}"
    )
