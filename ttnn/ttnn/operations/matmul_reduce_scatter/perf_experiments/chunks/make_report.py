"""Build the MMRS output-chunks report (static HTML: tables + inline SVG, no JS) from the sweep JSON lines."""
import glob
import json
import math
import statistics
import sys
from html import escape

SWEEP_DIR, OUT = sys.argv[1], sys.argv[2]
R = []
for f in sorted(glob.glob(SWEEP_DIR + "/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        r["_file"] = f.rsplit("/", 1)[-1]
        R.append(r)


def pick(sweep, forced, file=None):
    out = [r for r in R if r["sweep"] == sweep and r["forced"] == forced and (file is None or r["_file"] == file)]
    return out[-1] if out else None


def us(r):
    return r.get("median_us") if r else None


def fmt_t(t):
    if t is None:
        return "–"
    return f"{t / 1000:.2f} ms" if t >= 1000 else f"{t:.1f} µs"


# ------------------------------------------------------------------ colours (both themes: light cell + dark ink)
def ramp(ratio):
    """ratio = t / best >= 1 -> background (green fast ... red slow), log scale up to 5x."""
    x = min(1.0, math.log(max(ratio, 1.0)) / math.log(5.0))
    hue = 150 - 150 * x
    light = 86 - 18 * x
    return f"hsl({hue:.0f} 62% {light:.0f}%)"


def heatmap(sweep, ms, ns, file=None, kb=None, note=""):
    cells = {}
    for m in ms:
        for n in ns:
            key = [m, n, 2] if kb is None else [m, n, 2, kb]
            r = pick(sweep, key, file)
            cells[(m, n)] = r
    times = [us(r) for r in cells.values() if us(r)]
    best = min(times)
    h = ['<div class="hm-wrap"><table class="hm"><thead><tr><th class="corner">n<sub>m</sub> ╲ n<sub>n</sub></th>']
    h += [f"<th>{n}</th>" for n in ns] + ["</tr></thead><tbody>"]
    for m in ms:
        h.append(f'<tr><th class="rowh">{m}</th>')
        for n in ns:
            r = cells[(m, n)]
            if r is None:
                h.append('<td class="na">not run</td>')
            elif r.get("error"):
                h.append('<td class="nofit">does not fit</td>')
            else:
                t = us(r)
                p = r["plan"]
                ch = p["chunks"]
                dup = "" if ch == [m, n] else f'<span class="dup">ran as {ch[0]}×{ch[1]}</span>'
                tag = "R" if p["xblock"] == "resident" else "S"
                star = " ★" if t == best else ""
                h.append(
                    f'<td style="background:{ramp(t / best)}" title="{escape(json.dumps(p))}">'
                    f'<b>{fmt_t(t)}{star}</b><span class="meta">{tag} · P{p["P"]} · kb{p["kb"]} · '
                    f'{p["grid"][0] * p["grid"][1]}c</span>{dup}</td>'
                )
        h.append("</tr>")
    h.append("</tbody></table></div>")
    if note:
        h.append(f'<p class="cap">{note}</p>')
    return "".join(h), best


# ------------------------------------------------------------------ SVG line chart (log-log)
def loglog_chart(series, xlab, ylab, xmin, xmax, ymin, ymax, w=760, h=380, fits=()):
    L, Rm, T, B = 64, 150, 18, 46
    pw, ph = w - L - Rm, h - T - B

    def X(v):
        return L + pw * (math.log(v) - math.log(xmin)) / (math.log(xmax) - math.log(xmin))

    def Y(v):
        return T + ph * (1 - (math.log(v) - math.log(ymin)) / (math.log(ymax) - math.log(ymin)))

    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{escape(ylab)} vs {escape(xlab)}">']
    xt = [v for v in (1, 2, 4, 8, 16, 32, 64, 128, 256) if xmin <= v <= xmax]
    yt = [v for v in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if ymin <= v <= ymax]
    for v in xt:
        s.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{T + ph}" class="grid"/>')
        s.append(f'<text x="{X(v):.1f}" y="{T + ph + 18}" class="tick" text-anchor="middle">{v}</text>')
    for v in yt:
        s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" class="grid"/>')
        lab = f"{v / 1000:g} ms" if v >= 1000 else f"{v} µs"
        s.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    s.append(f'<text x="{L + pw / 2}" y="{h - 8}" class="axl" text-anchor="middle">{escape(xlab)}</text>')
    for k, (name, pts, cls, dashed) in enumerate(series):
        pts = sorted(p for p in pts if p[1])
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        s.append(f'<path d="{d}" class="ln {cls}{" dash" if dashed else ""}"/>')
        for x, y in pts:
            s.append(
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" class="pt {cls}"><title>{escape(name)}: '
                f"{x} chunks, {fmt_t(y)}</title></circle>"
            )
        lx, ly = pts[-1]
        s.append(
            f'<text x="{min(X(lx) + 8, L + pw + 6):.1f}" y="{Y(ly) + 4:.1f}" class="lab {cls}">{escape(name)}</text>'
        )
    for a, b, x0, x1, cls, label in fits:
        xs = [x0 * (x1 / x0) ** (i / 40) for i in range(41)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(a + b * x):.1f}" for i, x in enumerate(xs))
        s.append(f'<path d="{d}" class="ln fit {cls}"/>')
        s.append(
            f'<text x="{X(x1) + 8:.1f}" y="{Y(a + b * x1) + 16:.1f}" class="lab fitlab {cls}">{escape(label)}</text>'
        )
    s.append("</svg>")
    return "".join(s)


def linfit(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


# ------------------------------------------------------------------ data
KIMI_FILE = "kimi_now.jsonl"  # current code (kb cap 32 for chunked plans)
kimi_now = [r for r in R if r["_file"] == KIMI_FILE]


def kseries(pred):
    return [(r["plan"]["chunks"][0] * r["plan"]["chunks"][1], us(r)) for r in kimi_now if us(r) and pred(r)]


n_split = kseries(lambda r: r["forced"][0] == 1 and r["forced"][2] == 2)
m_split = kseries(lambda r: r["forced"][1] == 1 and r["forced"][2] == 2)
two_d = kseries(lambda r: r["forced"][0] > 1 and r["forced"][1] > 1 and r["forced"][2] == 2)
z1 = kseries(lambda r: r["forced"][2] == 1)
k1 = pick("kimi", [1, 1, 2], KIMI_FILE)
abl = {tuple(r["forced"][:2]) + (r["ablate"],): us(r) for r in R if r["sweep"] == "kimi_ablate"}
transport_only = [(m * n, abl[(m, n, "dram,mcast,compute")]) for (m, n) in ((1, 1), (1, 2), (1, 4), (1, 8))]
fit_pts = [p for p in n_split if p[0] <= 8]
fa, fb = linfit(fit_pts)
ta, tb = linfit(transport_only)


def run_median(files):
    out = {}
    for r in R:
        if r["_file"] in files:
            out.setdefault(r["sweep"], []).append(us(r))
    return {k: statistics.median(v) for k, v in out.items()}


old = run_median(("loose_old.jsonl", "kimi_ab_old.jsonl"))
new = run_median(("loose_new.jsonl", "kimi_ab_new.jsonl"))

wan_auto = pick("auto", None)
synth_auto = pick("synth_auto", None)
wan_fewest = pick("wan", None, "kb_wan.jsonl")  # the old fewest-chunks policy's pick (3x1)
synth_fewest = pick("synth", [1, 4, 2])

MS = [1, 2, 4, 8, 16, 32]
NS = [1, 2, 4, 8, 16]
synth_hm, synth_best = heatmap(
    "synth32",
    MS,
    NS,
    kb=32,
    note="8192×4096×8192 per device (bf16 × bf16, HiFi2, fp32 DEST), 4 devices, scatter −1. "
    "Each cell: median traced latency, regime (R resident A / S stream), slot tiles per "
    "core P, K-block, matmul cores. ★ fastest. Colour: latency relative to the fastest "
    "cell, log scale to 5×.",
)
synth16_hm, _ = heatmap("synth", MS, NS, note="Same grid with the K-block capped at 16 (the unchunked op's cap).")
WM = [2, 4, 8, 16, 32]
WN = [1, 2, 4, 8]
wan_hm, wan_best = heatmap(
    "wan32",
    WM,
    WN,
    kb=32,
    note="Wan2.2 FF2 720p, 9472×3456×5120 per device (bf16 × bf16, HiFi2, fp32 DEST). Unchunked it "
    "misses L1. ★ fastest cell on this grid; the planner's own pick (9×1) is not on the grid "
    "and ran faster still. Not run: splits outside the sweep list.",
)

ablate_rows = []
for m, n in ((1, 1), (1, 2), (1, 4), (1, 8), (2, 1)):
    ablate_rows.append(
        (f"{m}×{n}", abl[(m, n, "")], abl[(m, n, "dram")], abl[(m, n, "dram,mcast")], abl[(m, n, "dram,mcast,compute")])
    )

kimi_tbl = []
for r in kimi_now:
    if not us(r):
        continue
    p = r["plan"]
    kimi_tbl.append(
        (
            p["chunks"][0],
            p["chunks"][1],
            r["forced"][2],
            p["xblock"],
            p["P"],
            p["grid"][0] * p["grid"][1],
            p["kb"],
            us(r),
            r["rel_rms"],
        )
    )

# cost model accuracy (from the fit script's output, pasted by the caller)
MODEL = json.load(open(SWEEP_DIR + "/model_fit.json")) if glob.glob(SWEEP_DIR + "/model_fit.json") else {}

golden = (
    open(SWEEP_DIR + "/golden_summary.txt").read().strip()
    if glob.glob(SWEEP_DIR + "/golden_summary.txt")
    else "pending"
)


def kimi_table():
    rows = [
        '<div class="tbl-wrap"><table class="data"><thead><tr><th>chunks</th><th>zone sets</th><th>regime</th>'
        "<th>P</th><th>cores</th><th>kb</th><th>latency</th><th>vs 1×1</th><th>rel-RMS</th></tr></thead><tbody>"
    ]
    base = us(k1)
    for nm, nn, z, xb, P, cores, kb, t, rms in sorted(kimi_tbl, key=lambda x: (x[2] != 2, x[0] * x[1], x[0])):
        rows.append(
            f"<tr><td>{nm}×{nn}</td><td>{z}</td><td>{xb}</td><td>{P}</td><td>{cores}</td><td>{kb}</td>"
            f"<td class='num'>{fmt_t(t)}</td><td class='num'>{(t / base - 1) * 100:+.0f}%</td>"
            f"<td class='num'>{rms:.5f}</td></tr>"
        )
    rows.append("</tbody></table></div>")
    return "".join(rows)


def ablate_table():
    rows = [
        '<div class="tbl-wrap"><table class="data"><thead><tr><th>chunks</th><th>full op</th><th>no DRAM reads</th>'
        "<th>+ no multicast</th><th>+ no matmul<br>(transport only)</th></tr></thead><tbody>"
    ]
    for name, *ts in ablate_rows:
        rows.append(f"<tr><td>{name}</td>" + "".join(f"<td class='num'>{fmt_t(t)}</td>" for t in ts) + "</tr>")
    rows.append("</tbody></table></div>")
    return "".join(rows)


chart = loglog_chart(
    [
        ("split n (A stays resident)", n_split, "c1", False),
        ("split m (W re-read per chunk)", m_split, "c2", False),
        ("split both", two_d, "c3", False),
        ("transport only, split n", transport_only, "c4", True),
    ],
    "number of chunks (log)",
    "latency",
    1,
    64,
    50,
    5000,
    fits=[(fa, fb, 1, 8, "c1", f"fit ≤ 8: {fa:.0f} + {fb:.1f}·C µs")],
)


# ------------------------------------------------------------------ page
def card(big, small, label):
    return f'<div class="card"><div class="lbl">{label}</div><div class="big">{big}</div><div class="sm">{small}</div></div>'


k_old, k_new = old.get("loose_kimi"), new.get("loose_kimi")
g_old, g_new = old.get("loose_glm"), new.get("loose_glm")
mi_old, mi_new = old.get("loose_mimo"), new.get("loose_mimo")


def delta(a, b):
    return "–" if not (a and b) else f"{(b / a - 1) * 100:+.1f}%"


loose_rows = "".join(
    f"<tr><td>{name}</td><td class='num'>{fmt_t(a)}</td><td class='num'>{fmt_t(b)}</td><td class='num'>{delta(a, b)}</td></tr>"
    for name, a, b in (
        ("Kimi K2.7 o_proj 640×2048×7168 (focus)", k_old, k_new),
        ("GLM 5.x o_proj 640×4096×6144", g_old, g_new),
        ("MiMo-V2 o_proj 2048×2048×4096, scatter −2", mi_old, mi_new),
    )
)

per_chunk = fb
kimi2 = next((t for c, t in n_split if c == 2), None)

model_rows = ""
for name, (pk, pt, ft, fbest) in MODEL.get("picks", {}).items():
    model_rows += (
        f"<tr><td>{escape(name)}</td><td>{escape(pk)}</td><td class='num'>{fmt_t(pt)}</td>"
        f"<td class='num'>{fmt_t(ft)}</td><td class='num'>{fmt_t(fbest)}</td></tr>"
    )

page = f"""<title>MMRS Output Chunks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --bg: #f6f4ef; --panel: #fffdf8; --ink: #1d2226; --muted: #5d666d; --rule: #dcd6ca; --accent: #0b6e6e;
  --c1: #0b6e6e; --c2: #b4472c; --c3: #6a4fa3; --c4: #7a7a7a; --hm-ink: #15191c; --na: #e7e3da; --chip: #ece7dc;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #14181b; --panel: #1b2024; --ink: #e6e3dc; --muted: #9aa3a9; --rule: #2e353a; --accent: #4fc1bd;
  --c1: #4fc1bd; --c2: #f08a66; --c3: #b39ae8; --c4: #9a9a9a; --hm-ink: #15191c; --na: #262c31; --chip: #232a2f; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #14181b; --panel: #1b2024; --ink: #e6e3dc; --muted: #9aa3a9; --rule: #2e353a; --accent: #4fc1bd;
  --c1: #4fc1bd; --c2: #f08a66; --c3: #b39ae8; --c4: #9a9a9a; --hm-ink: #15191c; --na: #262c31; --chip: #232a2f; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 "IBM Plex Sans", system-ui, sans-serif; }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 28px; }}
header {{ display: grid; gap: 6px; border-bottom: 2px solid var(--ink); padding-bottom: 14px; }}
.kicker {{ font: 500 12px/1.2 "IBM Plex Mono", monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font-size: 30px; line-height: 1.15; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 21px; margin: 0; text-wrap: balance; }}
h3 {{ font-size: 16px; margin: 0; }}
.sub {{ color: var(--muted); margin: 0; max-width: 70ch; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
p {{ margin: 0; max-width: 72ch; }}
ul {{ margin: 0; padding-left: 20px; max-width: 72ch; display: grid; gap: 4px; }}
code, .mono {{ font-family: "IBM Plex Mono", monospace; font-size: 13px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.card {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; display: grid; gap: 2px; min-width: 0; }}
.card .lbl {{ font: 500 11px/1.3 "IBM Plex Mono", monospace; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.card .big {{ font: 600 24px/1.2 "IBM Plex Mono", monospace; color: var(--accent); }}
.card .sm {{ color: var(--muted); font-size: 13px; }}
.panel {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 12px; min-width: 0; }}
.tbl-wrap, .hm-wrap, .chart-wrap {{ overflow-x: auto; min-width: 0; }}
table.data {{ border-collapse: collapse; font-size: 13.5px; }}
table.data th, table.data td {{ padding: 5px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; }}
table.data th {{ font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
td.num {{ font-family: "IBM Plex Mono", monospace; text-align: right; }}
table.hm {{ border-collapse: separate; border-spacing: 3px; }}
table.hm th {{ font: 500 12px "IBM Plex Mono", monospace; color: var(--muted); padding: 4px 6px; }}
table.hm th.corner {{ text-align: left; }}
table.hm td {{ min-width: 112px; padding: 7px 8px; border-radius: 4px; color: var(--hm-ink); vertical-align: top; }}
table.hm td b {{ display: block; font: 600 14px "IBM Plex Mono", monospace; }}
table.hm td .meta, table.hm td .dup {{ display: block; font: 11px "IBM Plex Mono", monospace; opacity: .78; }}
table.hm td.nofit, table.hm td.na {{ background: var(--na); color: var(--muted); font-size: 12px; text-align: center; vertical-align: middle; }}
.cap {{ color: var(--muted); font-size: 13px; }}
svg.chart {{ width: 100%; min-width: 620px; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .tick, svg .axl {{ fill: var(--muted); font: 11px "IBM Plex Mono", monospace; }}
svg .ln {{ fill: none; stroke-width: 2; }}
svg .dash {{ stroke-dasharray: 5 4; }}
svg .fit {{ stroke-dasharray: 2 3; stroke-width: 1.5; opacity: .9; }}
svg .lab {{ font: 500 11.5px "IBM Plex Sans", sans-serif; }}
svg .fitlab {{ font: 11px "IBM Plex Mono", monospace; }}
{''.join(f'svg .ln.{c} {{ stroke: var(--{c}); }} svg .pt.{c} {{ fill: var(--{c}); }} svg .lab.{c} {{ fill: var(--{c}); }}' for c in ('c1', 'c2', 'c3', 'c4'))}
.qa {{ display: grid; gap: 14px; }}
.q {{ border-left: 3px solid var(--accent); padding-left: 14px; display: grid; gap: 6px; }}
.q .ans {{ font-weight: 600; }}
.chip {{ display: inline-block; background: var(--chip); border-radius: 3px; padding: 0 6px; font: 12px "IBM Plex Mono", monospace; }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
.flow > div {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; font-size: 13.5px; min-width: 0; }}
.flow b {{ display: block; font: 600 12px "IBM Plex Mono", monospace; color: var(--accent); margin-bottom: 2px; }}
footer {{ color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 12px; display: grid; gap: 4px; }}
</style>
<div class="wrap">
<header>
  <div class="kicker">matmul_reduce_scatter · QuietBox 2 · 1×4 ring · 2 links</div>
  <h1>Output chunks for shapes that miss L1</h1>
  <p class="sub">The per-chip output block is cut into n<sub>m</sub>×n<sub>n</sub> chunks and the whole reduce-scatter walk runs once per chunk inside the one program. Correctness on every split tried, the perf cost of chunking on Kimi K2.7, Wan2.2 FF2 720p (which did not run before), and an 8192×4096×8192 heatmap.</p>
</header>

<div class="cards">
  {card(fmt_t(us(wan_auto)), f"Wan2.2 FF2 720p, planner's pick {wan_auto['plan']['chunks'][0]}×{wan_auto['plan']['chunks'][1]}; raised NotImplementedError before", "now runs")}
  {card(delta(k_old, k_new), f"Kimi focus unchunked: {fmt_t(k_old)} → {fmt_t(k_new)}", "fitting shapes")}
  {card(f"+{per_chunk:.0f} µs", f"per extra chunk (Kimi, split n, ≤ 8 chunks); linear, not sublinear", "chunk overhead")}
  {card(f"{us(synth_fewest) / us(synth_auto):.1f}×", f"8192×4096×8192: fewest chunks {fmt_t(us(synth_fewest))} vs cost model {fmt_t(us(synth_auto))}", "how you split matters")}
</div>

<section>
  <h2>What changed</h2>
  <div class="flow">
    <div><b>1 · chunk loop</b>All five kernels walk chunk × (G entries). Grid, slot size P and zones are sized for one chunk; the resident operand's axis is the outer loop, so resident A reloads only when the m chunk changes.</div>
    <div><b>2 · zone reuse</b>Arrival zones hold <span class="chip">ZONE_SETS = 2</span> chunks. A port sends chunk c only after the downstream chip consumed chunk c − 2.</div>
    <div><b>3 · credits</b>Each matmul reader reports consumed chunks (arrival CB acks) to its chip's port that sends back upstream; that port forwards one fabric credit per chunk, polling while it waits on its own credits.</div>
    <div><b>4 · planner</b>1 chunk first (plans unchanged for every shape that fit). Otherwise every candidate split ≤ 256 chunks is planned and a fitted cost model picks one. Chunked plans cap the K-block at 32. Staging is sized exactly per call.</div>
  </div>
</section>

<section>
  <h2>Correctness</h2>
  <ul>
    <li><b>17/17</b> chunk cases (<code>test_matmul_reduce_scatter_chunked.py</code>): Kimi o_proj split along m, n, both, ragged 3×3, one zone set, Linear; MiMo-V2 (scatter −2); Wan2.2 FF2 720p auto and forced 4×1, 2×4, 8×4, 16×2. Every device's rel-RMS against the fp32 torch reference matches the unchunked op: 0.0050 (Kimi, MiMo), 0.0050–0.0051 (Wan).</li>
    <li>Back-to-back chunked and unchunked calls are bit-identical per chunking (counters, credits and the ready fence re-arm).</li>
    <li>Every perf-sweep config below was also checked against the reference (rel-RMS ≤ 0.0051, gate 0.01 bf16 / 0.035 bf8 weights).</li>
    <li>Acceptance test <b>65/65</b>. Golden suite: {escape(golden)}</li>
  </ul>
</section>

<section>
  <h2>Shapes that fit: no regression</h2>
  <p>A shape that fits L1 unchunked keeps the exact plan it had. The kernels now carry the chunk loop, so the three perf cases were re-measured against the pre-chunk commit, same harness.</p>
  <div class="tbl-wrap"><table class="data"><thead><tr><th>LOOSE perf case</th><th>before (0389e677)</th><th>after</th><th>Δ</th></tr></thead><tbody>{loose_rows}</tbody></table></div>
</section>

<section>
  <h2>Kimi K2.7 o_proj, forced chunks</h2>
  <p>The focus case fits unchunked, so every forced split shows the pure cost of chunking. Splitting N in 2 (640×896 per chunk per device) costs {fmt_t(kimi2 - us(k1)) if kimi2 else '–'} at 2 chunks.</p>
  <div class="panel"><div class="chart-wrap">{chart}</div>
  <p class="cap">Traced latency vs number of chunks, log-log. Split n keeps A resident; split m re-reads all 15.6 MB of W per m chunk and shrinks the grid (4×1 runs on 50 cores). Dashed: the same split-n configs with DRAM reads, multicasts and the matmul stubbed (MRS_ABLATE, kb 16): transport alone. Dotted: linear fit over 1–8 chunks.</p></div>
  {kimi_table()}
</section>

<section>
  <h2>Wan2.2 FF2 720p</h2>
  <p>9472×3456×5120 per device, the shape the earlier analysis said could not fit. The planner now picks <b>{wan_auto['plan']['chunks'][0]}×{wan_auto['plan']['chunks'][1]}</b> ({wan_auto['plan']['xblock']}, P {wan_auto['plan']['P']}, kb {wan_auto['plan']['kb']}): <b>{fmt_t(us(wan_auto))}</b>, rel-RMS {wan_auto['rel_rms']:.4f}. The first-fit policy would have picked 3×1 ({fmt_t(us(wan_fewest))}). Matmul floor on 88 cores at HiFi2 peak: 1.38 ms.</p>
  <div class="panel">{wan_hm}</div>
</section>

<section>
  <h2>Heatmap: 8192×4096×8192</h2>
  <p>Synthetic, large enough that nothing below 4 chunks fits (bf16 operands: A and W 64 MB per device each). Rows split m, columns split n; the per-chip block is 256×64 tiles. Matmul floor on 88 cores: 2.26 ms. Fastest: <b>{fmt_t(synth_best)}</b>.</p>
  <div class="panel">{synth_hm}</div>
  <details><summary>Same grid at K-block cap 16</summary><div class="panel">{synth16_hm}</div></details>
</section>

<section class="qa">
  <h2>The three questions</h2>
  <div class="q"><h3>1 · Is splitting M the same as splitting N?</h3>
    <p class="ans">No, and which one is better flips with the shape.</p>
    <ul>
      <li>Kimi: 2 chunks along n {fmt_t(next((t for c, t in n_split if c == 2), None))}, along m {fmt_t(next((t for c, t in m_split if c == 2), None))}. W (15.6 MB) is the big operand and an m split re-reads it per chunk; an n split keeps A resident and re-reads nothing.</li>
      <li>8192×4096×8192: 4 chunks along m {fmt_t(us(pick('synth32', [4, 1, 2, 32])))}, along n {fmt_t(us(pick('synth32', [1, 4, 2, 32])))}. Here A does not fit resident, so the stream regime re-reads A for every entry and every n chunk (G·n<sub>n</sub>·|A|), and narrow chunks also shrink the grid (1×16 runs on 44 of 88 cores).</li>
      <li>So "perf depends only on the number of output tiles" does not hold: what matters is which operand gets re-read, whether the resident regime fits, and whether a chunk still fills the grid.</li>
    </ul></div>
  <div class="q"><h3>2 · Is there an optimal chunk size?</h3>
    <p class="ans">Yes: the fewest chunks that get the resident regime with a long K-block while each chunk still fills the grid. Not simply the fewest chunks.</p>
    <ul>
      <li>8192×4096×8192: best at 8×4 (32 chunks of 32×16 tiles, P 6 per core) — {fmt_t(synth_best)}, ~71% of the FPU floor. 4 chunks is the minimum that fits and runs 1.4–3.6× slower in the stream regime.</li>
      <li>Wan: best at 9×1 (33×40 tiles per chunk, P 15, resident); 8×4 (32 chunks) is within 8%. Long thin splits (23×1, 30×1) and everything past 32 chunks are slower (≥ 2.39 ms).</li>
      <li>The cost model (below) finds these from the plan alone.</li>
    </ul></div>
  <div class="q"><h3>3 · How does the chunk overhead scale?</h3>
    <p class="ans">Linearly: about {per_chunk:.0f} µs per extra chunk on Kimi (fit {fa:.0f} + {fb:.1f}·C µs, 1–8 chunks), and faster than linear once chunks get too small to fill the grid.</p>
    <ul>
      <li>Nothing is amortised across chunks. Every chunk re-walks Kt/kb K-steps per entry (~1.4 µs fixed each, fitted: the multicast handshakes and read barriers of a step) and every port re-does a mailbox / read / ack / packet handoff per core per entry (~0.29 µs each, whatever the size).</li>
      <li>The stage peel shows the handoffs taking over: transport alone grows {fmt_t(transport_only[0][1])} → {fmt_t(transport_only[-1][1])} from 1 to 8 chunks ({ta:.0f} + {tb:.1f}·C), and at 1×8 it is ~{transport_only[-1][1] / abl[(1, 8, '')] * 100:.0f}% of the full op.</li>
      <li>Levers that would bend it: fewer, larger handoffs (batch a core's entries per port turn), and K-steps whose fixed cost overlaps (deeper look-ahead); neither is built.</li>
    </ul></div>
  {ablate_table()}
  <p class="cap">Stage peel (MRS_ABLATE; outputs wrong by design), Kimi, K-block 16, traced. Each column removes one more stage.</p>
</section>

<section>
  <h2>The cost model behind the auto split</h2>
  <p class="mono">t = {MODEL.get('floor', 8)} µs + max( C·G·P·Kt·32 cyc / ({MODEL.get('eff', 0.93)}·1.35 GHz) + {MODEL.get('a', 1.4)} µs·steps (stream {MODEL.get('a_s', 2.1)}),  traffic / {MODEL.get('bw', 200)} GB/s,  {MODEL.get('b', 0.29)} µs·C·(G−1)·cores/L )</p>
  <ul>
    <li>Least-squares fit to {MODEL.get('n', 222)} traced configs of the three shapes: median error {MODEL.get('median_err', '~10%')}, mean |log error| {MODEL.get('mean_log_err', 0.13)}. Traffic: resident A once + n<sub>m</sub>·|W|, or stream G·n<sub>n</sub>·|A| + n<sub>m</sub>·|W| (scatter −1; mirrored for −2).</li>
    <li>Used only when 1 chunk misses L1; ~300 candidates are planned in ~0.05 s and the pick is cached per shape.</li>
    <li>It is fitted on one box and three shapes: a ranking tool, not a roofline.</li>
  </ul>
  <div class="tbl-wrap"><table class="data"><thead><tr><th>shape</th><th>model pick</th><th>measured</th><th>first-fit pick</th><th>best measured</th></tr></thead><tbody>{model_rows}</tbody></table></div>
</section>

<section>
  <h2>Caveats</h2>
  <ul>
    <li>Measured with one 4-chip group (QB2). A Galaxy runs 8 groups at once and has slower links; the transport term would grow.</li>
    <li>Latency = max over the 4 devices of DEVICE KERNEL DURATION per traced replay, median of 5, stage zones on (as in the earlier perf rounds). Re-running a config moves it by up to ~5% (Kimi 1×1: 116–122 µs across runs).</li>
    <li>The staging tensor is now sized exactly per call (the old grow-only cache made a large call leave a floor that cost later shapes their resident regime). A trace captured before a call with a different staging size would replay stale addresses; same caveat as before, now on shrink too.</li>
    <li>Wan's addcmul fusion (residual + ff2·gate) is not part of this op.</li>
  </ul>
</section>

<footer>
  <div>Branch <span class="mono">malimpic/2026_10_15_1521_run1_mmrs_chunked</span> (tenstorrent/tt-metal), on top of the run branch <span class="mono">0389e677</span>.</div>
  <div>Raw data: <span class="mono">ttnn/ttnn/operations/matmul_reduce_scatter/perf_experiments/chunks/</span> (JSON lines, one per traced config). Harness: <span class="mono">test_matmul_reduce_scatter_chunked_perf.py</span>.</div>
</footer>
</div>
"""
open(OUT, "w").write(page)
print("wrote", OUT, len(page))
