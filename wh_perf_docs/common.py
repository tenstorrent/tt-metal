import json, os, re

D = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(f"{D}/assets.json"))
F = A["figs"]
import tiles as _tiles
F = list(F)
for _i in (8, 9):  # WH-01 waveforms: tile bars above each packer
    F[_i] = _tiles.add_tiles(F[_i])
URLS = json.load(open(f"{D}/urls.json")) if os.path.exists(f"{D}/urls.json") else {}

PR = "3ff45cbd1c75094e1846fcd61f87252140a6f40c"
RTL_ROOT = "/proj_sw/user_dev/lpremovic/tensix/src/hardware"


def code(path, line, label=None):
    """link to tt-metal source at #58068 head"""
    url = f"https://github.com/tenstorrent/tt-metal/blob/{PR}/{path}#L{line}"
    return f'<a href="{url}"><code>{label or path.split("/")[-1]}:{line}</code></a>'


def rtl(path, line, label=None):
    return f'<code title="{RTL_ROOT}/{path}">{label or path.split("/")[-1]}:{line}</code>'


def isa(path, label):
    return f'<a href="https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/{path}">{label}</a>'


def doc(id_):
    u = URLS.get(id_)
    t = TITLES.get(id_, id_)
    return f'<a href="{u}">{id_} {t}</a>' if u else f"<b>{id_}</b> {t}"


def chip(id_):
    u = URLS.get(id_)
    return f'<a class="chip" href="{u}">{id_}</a>' if u else f'<span class="chip">{id_}</span>'


def fig(i, cap=None):
    c = f'<p class="small">{cap}</p>' if cap else ""
    return f'{c}<div class="figwrap">{F[i]}</div>'


TITLES = {
    "WH-00": "Index",
    "WH-01": "Packer rhythm on the DEST crossbar",
    "WH-02": "L1 accesses while the packers run",
    "WH-03": "The first tiles of the pack loop",
    "WH-04": "Idle threads in isolate run types",
    "WH-05": "Size of the profiler zone helpers",
    "WH-06": "Pack branch predictor aliasing",
    "WH-07": "Branch-type cache with random replacement",
    "WH-08": "Instruction cache set conflicts",
    "WH-09": "State left by the previous kernel",
    "WH-10": "Check of the new #58068 fixes",
}

EXTRA_CSS = """
.docid{font:600 13px/1 var(--mono);color:var(--accent);letter-spacing:.06em}
.chip{display:inline-block;font:600 12px/1 var(--mono);padding:5px 8px;border-radius:3px;background:var(--accent-soft);color:var(--accent);text-decoration:none;margin-right:6px}
.status{display:inline-block;font:600 12px/1 var(--mono);padding:5px 8px;border-radius:3px}
.st-ok{background:var(--good-soft);color:var(--good)} .st-open{background:var(--bad-soft);color:var(--bad)} .st-part{background:var(--bk2);color:var(--fg)}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:15px;color:var(--muted)}
.flow{display:flex;flex-wrap:wrap;gap:6px 14px;font:13px var(--mono)}
.flow a{color:var(--muted)}
h2 .n{font:600 14px var(--mono);color:var(--accent);margin-right:10px}
.sketch{border:1px dashed var(--accent);border-radius:6px;padding:14px 16px;background:var(--panel)}
.sketch b{color:var(--accent)}
.refs li{font-size:15px}
.wave .grant{fill:var(--good)} .wave .wait{fill:var(--refc)} .wave .l1{fill:var(--accent)} .wave .markbg{fill:var(--accent-soft)} .wave .base{stroke:var(--rule);stroke-width:1} .wave .lwrite{fill:var(--good);opacity:.6} .wave .lwait{fill:var(--refc)}
.wave .mp{fill:var(--refc)} .wave .emp{fill:var(--bk2)} .wave .acc{fill:var(--good)} .wave .blk{fill:var(--muted);opacity:.45}
.wave .okc{fill:var(--good)} .wave .refc{fill:var(--refc)}
.wave .ln0{fill:none;stroke:var(--good);stroke-width:2.2} .wave .ln1{fill:none;stroke:var(--refc);stroke-width:2.2}
.wave .ln0d{fill:var(--good)} .wave .ln1d{fill:var(--refc)}
.xstrip{width:100%;height:auto;display:block}
.xstrip .rd{fill:var(--good)} .xstrip .rf{fill:var(--refc)} .xstrip .id{fill:var(--code)}
.xstrip .lab,.xstrip .tick{font-family:var(--mono);font-size:10px;fill:var(--muted)} .xstrip .xx{font-size:9px;fill:var(--bg);font-family:var(--mono)}
.dg .box{fill:var(--panel);stroke:var(--fg);stroke-width:1.2} .dg .core{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5}
.dg .t{font:600 13px var(--display);fill:var(--fg);text-anchor:middle} .dg .s{font:11px var(--mono);fill:var(--muted);text-anchor:middle}
.dg .ar{stroke:var(--muted);stroke-width:1.4;fill:none;marker-end:url(#ah)} .dg .ah{fill:var(--muted)}
.dg a .box:hover{stroke:var(--accent)}
"""

SECTIONS = [
    ("what", "What you see"),
    ("hw", "The hardware"),
    ("how", "How it happens"),
    ("versim", "Proof in Versim"),
    ("card", "Proof on the card"),
    ("fix", "The fix and how it works"),
    ("ba", "Before and after"),
    ("open", "What is not proven yet"),
    ("repro", "How to reproduce"),
    ("refs", "References"),
]


def page(d):
    deps = "".join(chip(x) for x in d.get("depends", [])) or "none"
    used = "".join(chip(x) for x in d.get("used_by", [])) or "none"
    nav = "".join(f'<a href="#{k}">{i+1}. {t}</a>' for i, (k, t) in enumerate(SECTIONS))
    body = []
    for i, (k, t) in enumerate(SECTIONS):
        body.append(f'<section id="{k}"><h2><span class="n">{i+1}</span>{t}</h2>{d[k]}</section>')
        if k == "how" and d.get("sketch"):
            body.append('<section class="sketch"><p><b>To sketch.</b> ' + d["sketch"] + "</p></section>")
    idx = URLS.get("WH-00")
    back = f'<a href="{idx}">WH-00 Index</a>' if idx else "WH-00 Index"
    return f"""<title>{d['id']} {d['short']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{A['css']}{EXTRA_CSS}</style>
<div class="wrap">
<header>
  <div class="docid">{d['id']} · Wormhole LLK perf · {back}</div>
  <h1>{TITLES[d['id']]}</h1>
  <p class="lede">{d['summary']}</p>
  <div class="meta"><span class="status {d['status_cls']}">{d['status']}</span><span>Depends on: {deps}</span><span>Used by: {used}</span><span>Problem: {d['problem']}</span></div>
  <nav class="flow">{nav}</nav>
</header>
{''.join(body)}
<p class="small">Status 8 October 2026. #58068 head {PR[:11]}. Code links point to that commit. RTL paths are relative to <code>{RTL_ROOT}</code> (Wormhole, internal). Every Versim run uses the same ELF as the card.</p>
</div>
"""


def l1_ports_svg():
    """Which Wormhole L1 port each packer, unpacker and RISC-V core uses (tt_tensix.sv, tt_tdma.sv)."""
    W, H = 900, 625
    s = [f'<svg viewBox="0 0 {W} {H}" class="wave dg lp" role="img" xmlns="http://www.w3.org/2000/svg">'
         '<defs><marker id="lpa" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" class="ah"/></marker></defs>']

    def box(x, y, w, h, t, sub="", cls="box"):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" class="{cls}"/>')
        ty = y + h / 2 + (-3 if sub else 5)
        s.append(f'<text x="{x+w/2}" y="{ty}" class="t">{t}</text>')
        if sub:
            s.append(f'<text x="{x+w/2}" y="{ty+15}" class="s">{sub}</text>')

    def wire(x1, y1, x2, y2, hot=False):
        mx = (x1 + x2) / 2
        s.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2-2},{y2}" class="{"arh" if hot else "ar"}" marker-end="url(#lpa)"/>')

    CX, CW, BH = 10, 210, 34            # clients
    AX, AW = 300, 170                    # TDMA arbiters
    PX, PW = 545, 140                    # ports
    LX = 760                             # L1
    s.append(f'<text x="{CX}" y="16" class="s" style="text-anchor:start">who asks</text>')
    s.append(f'<text x="{AX}" y="16" class="s" style="text-anchor:start">TDMA round-robin</text>')
    s.append(f'<text x="{PX}" y="16" class="s" style="text-anchor:start">L1 port (arbitrated)</text>')

    def client(y, t, sub="", kind="box"):
        box(CX, y, CW, BH, t, sub, kind)
        return (CX + CW, y + BH / 2)

    def port(y, h, t, sub):
        box(PX, y, PW, h, t, sub, "portb")
        return (PX, y + h / 2)

    # port 8: packer 0 alone
    p8 = port(28, 40, "Port 8", "packer 0 only")
    a = client(31, "Packer 0", "write", "core")
    wire(*a, *p8, True)
    # port 1: packer 1, unpacker 1, scrubber
    p1 = port(100, 100, "Port 1", "3 clients")
    for y, t, sub, k in [(88, "Packer 1", "read / write", "core"), (133, "Unpacker 1", "read", "box"), (178, "Scrubber", "write", "box")]:
        c = client(y, t, sub, k)
        wire(*c, *p1, k == "core")
    # port 2: TDMA arbiter 0 + BRISC, NCRISC, TRISC0
    a0 = (AX, 262)
    box(AX, 237, AW, 50, "TDMA arbiter 0", "tt_tdma.sv:3848", "arb")
    p2 = port(268, 150, "Port 2", "4 clients")
    c = client(232, "Packer 2", "write", "core"); wire(*c, *a0, True)
    c = client(272, "Mover rd, THCON,", "unpacker exp. rd", "box"); wire(*c, *a0)
    wire(AX + AW, 262, PX, 300, True)
    for y, t, sub in [(320, "TRISC0 (unpack)", "code + data", ), (360, "BRISC", "code + data"), (400, "NCRISC", "code + data")]:
        c = client(y, t, sub, "risc")
        wire(*c, PX, min(max(y + BH / 2, 300), 410))
    # port 3: TDMA arbiter 1 + TRISC1, TRISC2
    a1 = (AX, 480)
    box(AX, 455, AW, 50, "TDMA arbiter 1", "tt_tdma.sv:3855", "arb")
    p3 = port(466, 146, "Port 3", "3 clients")
    c = client(450, "Packer 3", "write", "core"); wire(*c, *a1, True)
    c = client(490, "Mover wr,", "RISC via TDMA", "box"); wire(*c, *a1)
    wire(AX + AW, 480, PX, 490, True)
    for y, t, sub in [(532, "TRISC1 (math)", "code + data"), (574, "TRISC2 (pack)", "code + data")]:
        c = client(y, t, sub, "risc")
        wire(*c, PX, min(y + BH / 2, 600))
    # L1 and port wires
    s.append(f'<rect x="{LX}" y="28" width="130" height="584" rx="6" class="l1b"/><text x="{LX+65}" y="320" class="t">L1</text>')
    for py in (48, 150, 343, 539):
        wire(PX + PW, py, LX, py)
    s.append("</svg>")
    legend = ('<div class="legend2"><span><span class="sw" style="background:var(--accent)"></span>packer</span>'
              '<span><span class="sw" style="background:var(--refc)"></span>TRISC / RISC-V core: code-line fetches and data</span>'
              '<span><span class="sw" style="background:var(--muted)"></span>other client</span></div>')
    return legend + '<div class="figwrap">' + "".join(s) + "</div>"


LP_CSS = """
.lp .portb{fill:var(--accent-soft);stroke:var(--fg);stroke-width:1.4}
.lp .arb{fill:var(--bk2);stroke:var(--fg);stroke-width:1.2}
.lp .risc{fill:var(--panel);stroke:var(--refc);stroke-width:1.6;stroke-dasharray:5 3}
.lp .l1b{fill:var(--bk1);stroke:var(--fg);stroke-width:1.4}
.lp .arh{stroke:var(--accent);stroke-width:2.2;fill:none}
"""
EXTRA_CSS += LP_CSS + _tiles.TILE_CSS + ".steps li{margin-bottom:8px}"

# Layout: a centred page no wider than 1200 px; text in a ~80-character reading column, figures and tables use the full page width.
EXTRA_CSS += """
.wrap{max-width:min(100%,1200px)}
header>*,section>p,section>ul,section>ol,section>h2,section>h3,section>details,.sketch p{max-width:80ch}
.figwrap,.tw{max-width:100%;overflow-x:auto}
"""
