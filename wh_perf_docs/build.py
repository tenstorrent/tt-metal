import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"
os.makedirs(OUT, exist_ok=True)
FX = open(f"{os.path.dirname(os.path.abspath(__file__))}/fx.html").read() if os.path.exists(f"{os.path.dirname(os.path.abspath(__file__))}/fx.html") else ""

ROWS = {
    "WH-01": ("Packer rhythm", "both", "yes", "yes (resync proof)", "hardware; triggers fixed one by one"),
    "WH-02": ("L1 accesses while packing", "both", "yes (hand-placed read)", "yes", "fixed"),
    "WH-03": ("First tiles of the pack loop", "no-work change", "yes", "yes", "fixed for code outside the loop"),
    "WH-04": ("Idle threads in isolate run types", "no-work change", "yes (fix too)", "yes (full suite)", "fixed in #58068 (52412ea8e67), checked: WH-10"),
    "WH-05": ("Zone helper size", "no-work change", "yes", "yes", "isolate: no longer seen; L1_CONGESTION, L1_TO_L1: open (WH-10)"),
    "WH-06": ("Pack branch predictor aliasing", "no-work change", "yes", "yes", "fixed for code outside the loop"),
    "WH-07": ("Branch-type cache", "no-work change", "yes (values swapped)", "yes", "partly"),
    "WH-08": ("Instruction cache sets", "no-work change", "yes", "yes", "avoided by pads"),
    "WH-09": ("Previous kernel state", "re-measure", "no (runs stop)", "yes", "fixed"),
}


def box(x, y, w, h, id_, sub, core=False):
    u = URLS.get(id_)
    inner = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" class="{"core" if core else "box"}"/><text x="{x+w/2}" y="{y+20}" class="t">{id_}</text><text x="{x+w/2}" y="{y+36}" class="s">{sub}</text>'
    return f'<a href="{u}">{inner}</a>' if u else inner


def diagram():
    s = ['<svg viewBox="0 0 900 330" class="wave dg" role="img" xmlns="http://www.w3.org/2000/svg"><defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" class="ah"/></marker></defs>']
    s.append('<text x="10" y="18" class="s" style="text-anchor:start">M1: the loop\'s code address sets its speed (TRISC front end)</text>')
    s += [box(10, 28, 200, 48, "WH-06", "pack branch predictor"), box(230, 28, 200, 48, "WH-07", "branch-type cache"), box(450, 28, 200, 48, "WH-08", "instruction cache sets")]
    s.append('<text x="10" y="112" class="s" style="text-anchor:start">M2: what pushes the packers (triggers)</text>')
    s += [box(10, 122, 200, 48, "WH-02", "L1 access while packing"), box(230, 122, 200, 48, "WH-03", "first tiles of the loop"), box(450, 122, 200, 48, "WH-04", "idle threads (isolate)"), box(670, 122, 200, 48, "WH-05", "zone helper size")]
    s.append(box(330, 230, 240, 52, "WH-01", "packer rhythm (DEST crossbar)", core=True))
    for x in (110, 330, 550, 770):
        s.append(f'<path d="M{x},170 C{x},205 450,205 450,228" class="ar"/>')
    s.append('<path d="M550,146 L432,146" class="ar"/><path d="M770,160 C770,190 340,190 330,172" class="ar"/>')
    s.append(box(690, 28, 200, 48, "WH-09", "previous kernel state"))
    s.append('<text x="450" y="312" class="s">arrows: "pushes" / "is a case of". WH-04 and WH-05 are cases of WH-02 and WH-03.</text></svg>')
    return "".join(s)


def index():
    rows = "".join(f'<tr><td>{chip(k)}</td><td>{v[0]}</td><td>{v[1]}</td><td>{v[2]}</td><td>{v[3]}</td><td>{v[4]}</td></tr>' for k, v in ROWS.items())
    flow = "".join(f"<li>{t}</li>" for _, t in SECTIONS)
    WH10 = URLS.get("WH-10", "#")
    return f"""<title>WH-00 Perf Mechanisms Index</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{A['css']}{EXTRA_CSS}</style>
<div class="wrap">
<header><div class="docid">WH-00 · Wormhole LLK perf</div><h1>Wormhole perf mechanisms</h1>
<p class="lede">One document per mechanism. Each one has the same ten sections, so you can read one at a time. Start with WH-01: the packer rhythm is what WH-02 to WH-05 push.</p></header>
<section><h2>How they connect</h2><div class="figwrap">{diagram()}</div></section>
<section><h2>The documents</h2><div class="tw"><table><tr><th>Doc</th><th>Mechanism</th><th>Problem it causes</th><th>Versim</th><th>Card</th><th>Status at #58068 head</th></tr>{rows}</table></div>
<p class="small">Problems: "re-measure" = the same build gives a different value when measured again; "no-work change" = a change that does no work moves values.</p></section>
<section><h2>New fixes in #58068 (9 October)</h2><p>Five new commits, each checked on the card and in Versim: <a href="{WH10}">WH-10 Check of the new #58068 fixes</a>. All five hold up. Open: L1_CONGESTION and L1_TO_L1 still jump between packer-rhythm states with any change of a few cycles.</p></section>
<section><h2>Full-suite check of the fixes (8 October, old head)</h2><p>CI, Wormhole, 842,256 points per arm, two runs each. "Final" = #58068 head + the WH-04 settle + no fixed kernel address + no warm-up on Wormhole (branch <code>nstojictt/p58-final</code>). Values that move by more than 2% (TILE_LOOP) when one never-executed nop is put in front of every function:</p>
<div class="tw"><table><tr><th>Run type</th><th class="n">#58068 head</th><th class="n">Final</th></tr>
<tr><td>PACK_ISOLATE</td><td class="n">13,537 (max 28.4%)</td><td class="n">0 (max 0.8%)</td></tr>
<tr><td>UNPACK_ISOLATE</td><td class="n">0</td><td class="n">0</td></tr>
<tr><td>MATH_ISOLATE</td><td class="n">632 (max 19.9%)</td><td class="n">61 (max 7.2%; matmul, WH-07)</td></tr>
<tr><td>L1_TO_L1</td><td class="n">96 (max 11.3%)</td><td class="n">100 (max 10.6%; WH-01, all threads run)</td></tr>
<tr><td>L1_CONGESTION (pack / unpack)</td><td class="n">1,292 / 136</td><td class="n">1,322 / 136 (by design)</td></tr></table></div>
<p>Run against rerun: every TILE_LOOP value identical in all four arms; only 40–44 KERNEL values differ.</p>
<p><b>The largest remaining movers do not reproduce on the lab card.</b> For the top MATH_ISOLATE mover (perf_matmul, CI 15,243 → 15,991) and the top L1_TO_L1 mover (perf_matmul, CI 8,240 → 8,762), the bgd-lab-08 card and Versim give the same values as each other, and they differ from CI: MATH_ISOLATE 21,591 / 21,591 (no move), L1_TO_L1 8,442 / 8,374 (−0.8%). So these CI residuals depend on the CI runner, not on our lab card; we did not explain them.</p></section>
<section><h2>Reading order</h2><ol><li>WH-01 (the base mechanism), then WH-02 and WH-03 (the two kinds of push).</li><li>WH-04 and WH-05 (two cases that are still open at #58068 head).</li><li>WH-06, WH-07, WH-08 (the front-end mechanisms, independent of the packers).</li><li>WH-09 (fixed; explains why the warm-up pass can go).</li></ol></section>
<section><h2>Every document has the same sections</h2><ol>{flow}</ol><p>Each also has a "To sketch" box after section 3 with one drawing that captures the mechanism.</p></section>
<section><h2>Before these become issues</h2><ul><li>The documents quote RTL file names and lines (internal). An issue in the public tt-metal repository must not.</li><li>Issue drafts without RTL references: branch <code>nstojictt/wh-perf-issue-drafts</code>, folder <code>wh_perf_issue_drafts/</code>.</li><li>The data and tools: branch <code>nstojictt/p58-versim</code> (switches and <code>repro_phase_tools/</code>); waveforms in <code>/proj_sw/user_dev/nstojic/versim-waveforms/</code>.</li></ul></section>
<p class="small">Status 10 October 2026. Documents WH-01 to WH-09: #58068 head 3ff45cbd1c7; WH-10: head 24ffcccb24c.</p>
</div>
"""


for i in range(1, 10):
    m = importlib.import_module(f"d{i:02d}")
    importlib.reload(m)
    d = m.D
    for k in SECTIONS:
        d[k[0]] = d[k[0]].replace("{{FX}}", FX)
    open(f"{OUT}/{d['id']}.html", "w").write(page(d))
open(f"{OUT}/WH-00.html", "w").write(index())
print(sorted(os.listdir(OUT)))
