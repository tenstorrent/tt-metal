# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate noc_placement_matrix.html from code + tt-npe (NPE), no hardcoded numbers.

Pipeline (all driven from here):
  1. device-ns pass  -> run test_noc_placement.py::test_noc_placement_device_perf via
     run_safe_pytest (writes {cell: ns/op} to a temp JSON).
  2. trace pass      -> profile_this.py --collect-noc-traces on ::test_noc_placement_matrix
     (one op launch per cell) -> <OUT>/.logs/noc_trace_dev0_*ID*.json.
  3. simulate        -> tt-npe analyze_noc_traces_in_dir(...) in-process -> <OUT>/npe_viz/*.npeviz
     + manifest.json, plus a Stats object (per-op DRAM-BW util).
  4. aggregate+emit  -> join measured ns + tt-npe per-link demand + DRAM-BW into
     noc_placement_matrix.html and refresh the matrix block in report.md.

tt-npe is located via $TT_NPE_HOME, then <tt-metal>/../tt-npe, then a fallback; its
install/lib + install/bin are put on sys.path (see skills/perf-roofline-dm/tt_npe.sh).

    python -m ttnn.operations.examples.noc_placement --report      # (thin wrapper over this)
    python -m ttnn.operations.examples.noc_placement.noc_report     # direct

Wormhole B0 geometry is baked into the grid drawing (DRAM in NoC cols 0 & 5); single-arch,
matching report.md. Re-run on a different arch and append, do not overwrite.
"""

import collections
import datetime
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # ttnn/ttnn/operations/examples/noc_placement -> tt-metal
HTML_OUT = HERE / "noc_placement_matrix.html"
REPORT_MD = HERE / "report.md"
TEST = "tests/ttnn/unit_tests/operations/examples/test_noc_placement.py"

# Must match test_noc_placement.py BENCH_MATRIX order (trace op-id sequence -> cell).
NOCS = ("noc0", "noc1")
OPS = ("read", "write")
PLACEMENTS = ("column", "row", "diagonal")
BENCH_MATRIX = [(op, noc, pl) for noc in NOCS for op in OPS for pl in PLACEMENTS]


# ---------------------------------------------------------------- tt-npe location
def locate_ttnpe():
    candidates = []
    if os.environ.get("TT_NPE_HOME"):
        candidates.append(Path(os.environ["TT_NPE_HOME"]))
    candidates += [REPO.parent / "tt-npe", Path("/localdev/dnijemcevic/tt-npe")]
    for root in candidates:
        lib, binp = root / "install" / "lib", root / "install" / "bin"
        if lib.is_dir() and binp.is_dir():
            for p in (str(lib), str(binp)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return root
    raise RuntimeError(
        "tt-npe install not found. Build it (cd tt-npe && ./build-npe.sh) and set $TT_NPE_HOME, "
        f"or place it at {REPO.parent / 'tt-npe'}. Looked in: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------- device passes
def run_ns_pass(tmp):
    ns_json = tmp / "matrix_ns.json"
    env = dict(os.environ, NP_NS_OUT=str(ns_json))
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", f"{TEST}::test_noc_placement_device_perf"]
    print(f"[noc_report] device-ns pass: {' '.join(cmd)}")
    if subprocess.call(cmd, cwd=str(REPO), env=env) != 0 or not ns_json.exists():
        raise RuntimeError("device-ns pass failed (no matrix_ns.json produced)")
    return json.loads(ns_json.read_text())


def run_trace_pass(tmp):
    out = tmp / "traces"
    # Do NOT put tt-npe on PYTHONPATH here: let profile_this only capture traces; we simulate
    # in-process afterward for control over compression/output.
    env = dict(os.environ, TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="10000")
    cmd = [
        sys.executable,
        "tools/tracy/profile_this.py",
        "--collect-noc-traces",
        "-c",
        f"pytest {TEST}::test_noc_placement_matrix",
        "-o",
        str(out),
    ]
    print(f"[noc_report] trace pass: {' '.join(cmd)}")
    if subprocess.call(cmd, cwd=str(REPO), env=env) != 0:
        raise RuntimeError("trace pass failed")
    logs = out / ".logs"
    traces = sorted(logs.glob("noc_trace_dev0_*ID*.json"))
    if len(traces) < len(BENCH_MATRIX):
        raise RuntimeError(f"expected {len(BENCH_MATRIX)} noc traces, found {len(traces)} in {logs}")
    return logs


def run_ttnpe(logs_dir):
    from npe_analyze_noc_trace_dir import analyze_noc_traces_in_dir

    print("[noc_report] tt-npe: analyzing noc traces ...")
    stats = analyze_noc_traces_in_dir(
        noc_trace_dir=str(logs_dir),
        emit_viz_timeline_files=True,
        compress_timeline_files=False,
        quiet=True,
    )
    npe_viz = logs_dir.parent / "npe_viz"
    return stats, npe_viz


# ---------------------------------------------------------------- aggregation
def _dram_bw_by_op_id(stats):
    """map ttnn_op_id -> DRAM-BW util%, keeping the valid (non-nan) datapoint per op."""
    out = {}
    for _key, dp in stats:
        val = getattr(dp.result, "dram_bw_util", float("nan"))
        oid = dp.op_uid.ttnn_op_id
        if val == val and (oid not in out or out[oid] == 0):  # non-nan; prefer a real value
            out[oid] = val
    return out


def aggregate(npe_viz, stats, ns):
    manifest = json.loads((npe_viz / "manifest.json").read_text())
    manifest.sort(key=lambda e: e["global_call_count"])  # dispatch order == BENCH_MATRIX order
    if len(manifest) != len(BENCH_MATRIX):
        raise RuntimeError(f"manifest has {len(manifest)} ops, expected {len(BENCH_MATRIX)}")
    dram_bw = _dram_bw_by_op_id(stats)

    agg = {}
    for (op, noc, pl), entry in zip(BENCH_MATRIX, manifest):
        d = json.loads((npe_viz / entry["file"]).read_text())
        nts = len(d["timestep_data"])
        ls = collections.defaultdict(float)
        for ts in d["timestep_data"]:
            for _dev, r, c, lt, dem in ts["link_demand"]:
                ls[(r, c, lt)] += dem
        link_avg = {f"{r},{c},{lt}": ls[(r, c, lt)] / nts for (r, c, lt) in ls}
        involved = set()
        for t in d["noc_transfers"]:
            involved.add((t["src"][1], t["src"][2]))
            for dd in t["dst"]:
                involved.add((dd[1], dd[2]))
        oid = entry["global_call_count"] >> 10
        agg[f"{noc}_{op}_{pl}"] = {
            "noc": noc,
            "op": op,
            "placement": pl,
            "link_avg": link_avg,
            "max_link_avg": max(link_avg.values()) if link_avg else 0.0,
            "involved_cores": sorted([list(c) for c in involved]),
            "device_ns": ns.get(f"{op}_{noc}_{pl}"),
            "dram_bw": dram_bw.get(oid),
        }
    return agg


# ---------------------------------------------------------------- HTML (Wormhole B0 grid)
NROWS, NCOLS = 12, 10
DRAM = {(r, 0) for r in (0, 1, 5, 6, 7, 11)} | {(r, 5) for r in range(12)}
ETH = {(r, c) for r in (0, 6) for c in (1, 2, 3, 4, 6, 7, 8, 9)}
UNDEF = {(r, 0) for r in (2, 3, 4, 8, 9, 10)}
CELL, PAD = 20, 16
W, H = NCOLS * CELL + 2 * PAD, NROWS * CELL + 2 * PAD


def _ctype(r, c):
    if (r, c) in DRAM:
        return "dram"
    if (r, c) in ETH:
        return "eth"
    if (r, c) in UNDEF:
        return "undef"
    return "worker"


def _cx(c):
    return PAD + c * CELL + CELL / 2


def _cy(r):
    return PAD + r * CELL + CELL / 2


def _color(d, gmax):
    t = max(0.0, min(1.0, (d / gmax) ** 0.5)) if gmax else 0.0
    if t < 0.5:
        u = t / 0.5
        r, g, b = 250 + (240 - 250) * u, 240 + (140 - 240) * u, 150 + (40 - 150) * u
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 240 + (150 - 240) * u, 140 + (10 - 140) * u, 40 + (10 - 40) * u
    return f"rgb({int(r)},{int(g)},{int(b)})"


def _svg_cell(agg_cell, gmax):
    workers = {(r, c) for (r, c) in map(tuple, agg_cell["involved_cores"]) if _ctype(r, c) == "worker"}
    la = agg_cell["link_avg"]
    p = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="grid">',
        '<defs><marker id="ar" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" '
        'refX="5" refY="2.8" orient="auto"><path d="M0,0.4 L5.5,2.8 L0,5.2 Z" fill="context-stroke"/></marker></defs>',
    ]
    for r in range(NROWS):
        for c in range(NCOLS):
            cls = "cell " + _ctype(r, c) + (" reader" if (r, c) in workers else "")
            p.append(
                f'<rect x="{PAD+c*CELL+1}" y="{PAD+r*CELL+1}" width="{CELL-2}" height="{CELL-2}" rx="2" class="{cls}"/>'
            )
    off = 3
    for k, dem in sorted(la.items(), key=lambda kv: kv[1]):
        if dem <= 0.5:
            continue
        r, c, lt = k.split(",")
        r, c = int(r), int(c)
        col = _color(dem, gmax)
        lw = 1.0 + 3.6 * min(1.0, dem / gmax)
        if lt == "NOC0_SOUTH":
            x = _cx(c) + off
            p.append(
                f'<line x1="{x}" y1="{_cy(r)}" x2="{x}" y2="{_cy(r)+CELL if r<NROWS-1 else _cy(r)+CELL/2}" stroke="{col}" stroke-width="{lw}" marker-end="url(#ar)"/>'
            )
        elif lt == "NOC1_NORTH":
            x = _cx(c) - off
            p.append(
                f'<line x1="{x}" y1="{_cy(r)}" x2="{x}" y2="{_cy(r)-CELL if r>0 else _cy(r)-CELL/2}" stroke="{col}" stroke-width="{lw}" marker-end="url(#ar)"/>'
            )
        elif lt == "NOC0_EAST":
            y = _cy(r) - off
            p.append(
                f'<line x1="{_cx(c)}" y1="{y}" x2="{_cx(c)+CELL if c<NCOLS-1 else _cx(c)+CELL/2}" y2="{y}" stroke="{col}" stroke-width="{lw}" marker-end="url(#ar)"/>'
            )
        elif lt == "NOC1_WEST":
            y = _cy(r) + off
            p.append(
                f'<line x1="{_cx(c)}" y1="{y}" x2="{_cx(c)-CELL if c>0 else _cx(c)-CELL/2}" y2="{y}" stroke="{col}" stroke-width="{lw}" marker-end="url(#ar)"/>'
            )
        elif lt.endswith("_IN") or lt.endswith("_OUT"):
            p.append(f'<circle cx="{_cx(c)}" cy="{_cy(r)}" r="{1.5+2*min(1.0,dem/gmax)}" fill="{col}" opacity="0.5"/>')
    p.append("</svg>")
    return "\n".join(p)


def _schematic():
    """Illustrative routing schematic (real NPE dimension-order rule + torus wrap): one DRAM
    bank -> 3 readers on distinct columns. NoC0=east->south (spreads), NoC1=north->west (stacks)."""
    SC, SP, C, R = 40, 26, 9, 7
    w, h = C * SC + 2 * SP, R * SC + 2 * SP
    sx = lambda c: SP + c * SC + SC / 2
    sy = lambda r: SP + r * SC + SC / 2
    DCOL, bank, readers = 1, (1, 1), [(3, 3), (4, 5), (6, 7)]
    hues = ["#1f9d57", "#2e86c1", "#8e44ad"]

    def route(src, dst, noc):
        sr, sc = src
        dr, dc = dst
        cells = [(sr, sc)]
        if noc == "NOC0":
            c = sc
            while c != dc:
                c = (c + 1) % C
                cells.append((sr, c))
            r = sr
            while r != dr:
                r = (r + 1) % R
                cells.append((r, dc))
        else:
            r = sr
            while r != dr:
                r = (r - 1) % R
                cells.append((r, sc))
            c = sc
            while c != dc:
                c = (c - 1) % C
                cells.append((dr, c))
        return cells

    def frame():
        p = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="grid">',
            '<defs><marker id="ah" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="6.5" refY="3.5" orient="auto">'
            '<path d="M0,0.5 L7,3.5 L0,6.5 Z" fill="context-stroke"/></marker></defs>',
        ]
        p.append(f'<rect x="{SP+DCOL*SC}" y="{SP}" width="{SC}" height="{R*SC}" fill="var(--dramcol)" opacity="0.55"/>')
        p.append(f'<text x="{sx(DCOL)}" y="{SP-8}" class="axcap">DRAM column</text>')
        for c in range(C):
            for r in range(R):
                p.append(f'<rect x="{SP+c*SC+2}" y="{SP+r*SC+2}" width="{SC-4}" height="{SC-4}" rx="3" class="scell"/>')
        p.append(
            f'<rect x="{SP+bank[1]*SC+2}" y="{SP+bank[0]*SC+2}" width="{SC-4}" height="{SC-4}" rx="3" class="sbank"/>'
        )
        p.append(f'<text x="{sx(bank[1])}" y="{sy(bank[0])+3}" class="slbl blk">DR</text>')
        for i, (rr, rc) in enumerate(readers):
            p.append(
                f'<rect x="{SP+rc*SC+2}" y="{SP+rr*SC+2}" width="{SC-4}" height="{SC-4}" rx="3" class="sreader" style="stroke:{hues[i]}"/>'
            )
            p.append(f'<text x="{sx(rc)}" y="{sy(rr)+3}" class="slbl rd" style="fill:{hues[i]}">R{i+1}</text>')
        return p

    def draw(cells, color, offx):
        s = []
        for a, b in zip(cells, cells[1:]):
            ar, ac = a
            br, bc = b
            ax, ay, bx, by = sx(ac) + offx, sy(ar), sx(bc) + offx, sy(br)
            wc, wr = abs(ac - bc) > 1, abs(ar - br) > 1
            if not wc and not wr:
                s.append(
                    f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{color}" class="rt" marker-end="url(#ah)"/>'
                )
            elif wc:
                if ac == 0 and bc == C - 1:
                    s.append(
                        f'<line x1="{ax}" y1="{ay}" x2="{SP-6}" y2="{ay}" stroke="{color}" class="rt" marker-end="url(#ah)"/>'
                    )
                    s.append(f'<line x1="{SP+C*SC+6}" y1="{by}" x2="{bx}" y2="{by}" stroke="{color}" class="rt"/>')
                else:
                    s.append(
                        f'<line x1="{ax}" y1="{ay}" x2="{SP+C*SC+6}" y2="{ay}" stroke="{color}" class="rt" marker-end="url(#ah)"/>'
                    )
                    s.append(f'<line x1="{SP-6}" y1="{by}" x2="{bx}" y2="{by}" stroke="{color}" class="rt"/>')
            else:
                if ar == 0 and br == R - 1:
                    s.append(
                        f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{SP-6}" stroke="{color}" class="rt" marker-end="url(#ah)"/>'
                    )
                    s.append(f'<line x1="{bx}" y1="{SP+R*SC+6}" x2="{bx}" y2="{by}" stroke="{color}" class="rt"/>')
                else:
                    s.append(
                        f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{SP+R*SC+6}" stroke="{color}" class="rt" marker-end="url(#ah)"/>'
                    )
                    s.append(f'<line x1="{bx}" y1="{SP-6}" x2="{bx}" y2="{by}" stroke="{color}" class="rt"/>')
        return "\n".join(s)

    out = {}
    for noc, pn in (("NOC0", "p0"), ("NOC1", "p1")):
        p = frame()
        for i, rd in enumerate(readers):
            p.append(draw(route(bank, rd, noc), hues[i], (i - 1) * 3.0))
        p.append("</svg>")
        out[pn] = "\n".join(p)
    return out["p0"], out["p1"]


def _fmt_ns(v):
    return "n/a" if v is None else f"{v/1000:.0f}k"


def render_html(agg):
    gmax = max((c["max_link_avg"] for c in agg.values()), default=1.0) or 1.0
    sch0, sch1 = _schematic()
    legend = "".join(f'<stop offset="{i}%" stop-color="{_color(gmax*(i/100)**2, gmax)}"/>' for i in range(0, 101, 5))

    def cell_ns(op, noc, pl):
        return agg[f"{noc}_{op}_{pl}"]["device_ns"]

    def panel(noc, op, pl):
        d = agg[f"{noc}_{op}_{pl}"]
        other = agg[f"{'noc1' if noc=='noc0' else 'noc0'}_{op}_{pl}"]
        win = d["device_ns"] is not None and other["device_ns"] is not None and d["device_ns"] < other["device_ns"]
        tag = "<span class='w'>fastest</span>" if win else "<span class='l'>slower</span>"
        bw = "" if d["dram_bw"] is None else f" &middot; DRAM {d['dram_bw']:.0f}%"
        return (
            f'<div class="cell-card {"win" if win else "lose"}"><div class="ct"><b>{pl}</b> {tag}</div>'
            f'<div class="cm">{_fmt_ns(d["device_ns"])} ns{bw} &middot; peak {d["max_link_avg"]:.0f}</div>'
            f"{_svg_cell(d, gmax)}</div>"
        )

    def section(noc, op, arrows):
        cards = "".join(panel(noc, op, pl) for pl in PLACEMENTS)
        return (
            f'<h3 class="sech {noc}">{noc.upper()} &mdash; {op} <span class="dir">({arrows})</span></h3>'
            f'<div class="row3">{cards}</div>'
        )

    trows = ""
    for op in OPS:
        for pl in PLACEMENTS:
            n0, n1 = cell_ns(op, "noc0", pl), cell_ns(op, "noc1", pl)
            c0 = "w" if (n0 is not None and n1 is not None and n0 < n1) else "l"
            c1 = "w" if (n0 is not None and n1 is not None and n1 < n0) else "l"
            faster = "&ndash;"
            if n0 is not None and n1 is not None:
                faster = f"{'NoC0' if n0<n1 else 'NoC1'} ({max(n0,n1)/min(n0,n1):.1f}&times;)"
            trows += (
                f"<tr><td class='pl'>{op}</td><td class='pl'>{pl}</td>"
                f"<td class='{c0}'>{_fmt_ns(n0)}</td><td class='{c1}'>{_fmt_ns(n1)}</td><td>{faster}</td></tr>"
            )

    html = _TEMPLATE.format(
        gmax=f"{gmax:.0f}",
        legend=legend,
        sch0=sch0,
        sch1=sch1,
        trows=trows,
        s_n0r=section("noc0", "read", "east&rarr;south"),
        s_n0w=section("noc0", "write", "east&rarr;south"),
        s_n1r=section("noc1", "read", "north&rarr;west, wraps"),
        s_n1w=section("noc1", "write", "north&rarr;west, wraps"),
    )
    HTML_OUT.write_text(html)
    print(f"[noc_report] wrote {HTML_OUT} ({len(html)} bytes; gmax {gmax:.0f})")


def _stamp():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        commit = "unknown"
    shape = tuple(int(x) for x in os.environ.get("NP_SHAPE", "1024,2048").split(","))
    tiles = (shape[0] // 32) * (shape[1] // 32)
    return {
        "box": socket.gethostname(),
        "arch": os.environ.get("ARCH_NAME", "wormhole_b0"),
        "commit": commit,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "shape": f"{tuple(shape)} = {tiles} tiles",
        "cores": os.environ.get("NP_CORES", "8"),
        "params": f"block={os.environ.get('NP_BLOCK','16')}, kernel_iters={os.environ.get('NP_ITERS','8')}, "
        f"{os.environ.get('NP_PROFILE_ITERS','20')} profiled launches averaged",
    }


def write_report_md(agg, ns):
    """Regenerate the WHOLE report.md: fresh stamp + placement(copy) block + NoC×op matrix.
    noc_report.py owns this file, so no stale content survives a re-run."""
    s = _stamp()
    out = ["# noc_placement — measured report", "", "| stamp | value |", "|---|---|"]
    for k in ("box", "arch", "commit", "date", "shape", "cores", "params"):
        out.append(f"| {k} | `{s[k]}` |")
    out += [
        "| metric | `DEVICE KERNEL DURATION [ns]`, in-process profiler |",
        "| generated by | `python -m ttnn.operations.examples.noc_placement --report` |",
        "",
        "> Numbers are illustrative of the effect, single-box single-arch. Regenerate with `--report`;",
        "> append a different arch as its own run rather than overwriting.",
        "",
        "## Placement — identity copy (canonical: reads NoC0 / writes NoC1)",
        "",
        "```",
        f"    {'placement':<10} {'ns/op':>12} {'vs column':>10}",
    ]
    base = ns.get("copy_noc0_column")
    for pl in PLACEMENTS:
        v = ns.get(f"copy_noc0_{pl}")
        if v is None:
            out.append(f"    {pl:<10} {'n/a':>12}")
            continue
        tag = "(baseline)" if pl == "column" else (f"-> {base/v:.2f}x" if base else "")
        out.append(f"    {pl:<10} {v:>12.1f} {tag:>10}")
    out += [
        "```",
        "",
        "## NoC × operation × placement (isolated read/write benches)",
        "",
        "```",
        f"    {'op':<6} {'placement':<10} {'noc0 ns':>10} {'noc1 ns':>10}  faster",
    ]
    for op in OPS:
        for pl in PLACEMENTS:
            n0 = agg[f"noc0_{op}_{pl}"]["device_ns"]
            n1 = agg[f"noc1_{op}_{pl}"]["device_ns"]
            f = "-" if (n0 is None or n1 is None) else f"{'NoC0' if n0<n1 else 'NoC1'} {max(n0,n1)/min(n0,n1):.1f}x"
            s0 = "n/a" if n0 is None else f"{n0:.0f}"
            s1 = "n/a" if n1 is None else f"{n1:.0f}"
            out.append(f"    {op:<6} {pl:<10} {s0:>10} {s1:>10}  {f}")
    out += [
        "```",
        "",
        "Mirror symmetry: read·NoC0 ≈ write·NoC1, read·NoC1 ≈ write·NoC0 (same links, reversed).",
        "See `noc_placement_matrix.html` for the per-cell NoC link-demand heatmaps.",
        "",
    ]
    REPORT_MD.write_text("\n".join(out))
    print(f"[noc_report] wrote {REPORT_MD}")


def main(argv=None):
    locate_ttnpe()
    with tempfile.TemporaryDirectory(prefix="noc_placement_report_") as td:
        tmp = Path(td)
        ns = run_ns_pass(tmp)
        logs = run_trace_pass(tmp)
        stats, npe_viz = run_ttnpe(logs)
        agg = aggregate(npe_viz, stats, ns)
    render_html(agg)
    write_report_md(agg, ns)
    return 0


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>noc_placement — NoC × operation × placement</title>
<style>
  :root {{ --bg:#f6f8fa; --panel:#fff; --ink:#1c2430; --muted:#5a6675; --line:#dde3ea;
    --accent:#0969da; --warn:#c0392b; --good:#1f9d57; --dramcol:#c2ccd8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 22px 80px; }}
  h1 {{ font-size:25px; margin:0 0 4px; letter-spacing:-0.01em; }}
  h2 {{ font-size:19px; margin:34px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  h3.sech {{ font-size:15.5px; margin:20px 0 6px; padding-left:8px; border-left:4px solid var(--line); }}
  h3.sech.noc0 {{ border-left-color:var(--good); }} h3.sech.noc1 {{ border-left-color:var(--warn); }}
  h3.sech .dir {{ color:var(--muted); font-weight:500; font-size:13px; }}
  .sub {{ color:var(--muted); margin:0 0 8px; }}
  p {{ max-width:82ch; }} code {{ font-family:ui-monospace,Menlo,monospace; background:var(--panel);
    padding:1px 5px; border-radius:4px; font-size:13px; border:1px solid var(--line); }}
  .row3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
  .cell-card {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:8px 8px 4px; }}
  .cell-card.win {{ border-color:var(--good); box-shadow:0 0 0 1px var(--good) inset; }}
  .cell-card .ct {{ font-size:13px; }} .cell-card .cm {{ color:var(--muted); font-size:11.5px; font-family:ui-monospace,monospace; margin:1px 0 4px; }}
  .w {{ color:var(--good); font-weight:700; font-size:11px; }} .l {{ color:var(--muted); font-size:11px; }}
  svg.grid {{ width:100%; height:auto; display:block; }}
  .cell {{ fill:var(--panel); stroke:var(--line); stroke-width:0.8; }}
  .cell.dram {{ fill:#dfe7ef; stroke:#c2ccd8; }} .cell.eth {{ fill:#eef2f6; stroke:#dde3ea; }}
  .cell.undef {{ fill:#f0f2f5; stroke:#e2e6eb; }} .cell.reader {{ fill:#cfe4fb; stroke:var(--accent); stroke-width:1.6; }}
  .scell {{ fill:var(--panel); stroke:var(--line); stroke-width:1; }}
  .sbank {{ fill:#dfe7ef; stroke:#8fa3b8; stroke-width:1.5; }}
  .sreader {{ fill:#cfe4fb; stroke:var(--accent); stroke-width:2; }}
  .slbl {{ text-anchor:middle; font-size:9px; font-weight:700; font-family:ui-monospace,monospace; }}
  .slbl.blk {{ fill:#1c2430; }} .slbl.rd {{ fill:var(--accent); }}
  .axcap {{ fill:var(--muted); font-size:10px; text-anchor:middle; font-family:ui-monospace,monospace; }}
  .rt {{ fill:none; stroke-width:2.4; stroke-linecap:round; }}
  table {{ border-collapse:collapse; width:100%; margin:8px 0; font-size:14px; }}
  th,td {{ padding:7px 10px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td.pl {{ text-align:left; }} td.pl {{ font-weight:600; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
  td.w {{ color:var(--good); font-weight:700; }} td.l {{ color:var(--muted); }}
  .legend {{ display:flex; align-items:center; gap:10px; margin:12px 0; color:var(--muted); font-size:12.5px; flex-wrap:wrap; }}
  .legend .bar {{ height:12px; width:200px; border-radius:6px; border:1px solid var(--line); }}
  .callout {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--good);
    border-radius:8px; padding:12px 16px; margin:14px 0; }}
  .schem {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:10px 0; }}
  .schem .cell-card {{ padding:10px 12px; }} .schem h4 {{ margin:2px 0 4px; font-size:14.5px; }}
  .schem .noc0 {{ border-left:3px solid var(--good); }} .schem .noc1 {{ border-left:3px solid var(--warn); }}
  .schem .m {{ color:var(--muted); font-size:12px; margin-bottom:4px; }}
  ul {{ max-width:82ch; }} .foot {{ color:var(--muted); font-size:12.5px; margin-top:30px;
    border-top:1px solid var(--line); padding-top:12px; }}
</style></head>
<body><div class="wrap">
  <h1>noc_placement &mdash; NoC &times; operation &times; placement</h1>
  <p class="sub">Every DRAM access can run on either NoC. This maps all four combinations &mdash;
  <b>read</b>/<b>write</b> &times; <b>NoC0</b>/<b>NoC1</b> &mdash; across the three core placements, each isolated
  (read-only or write-only bench). Regenerated from code by <code>noc_report.py</code> (device profiler + tt-npe).</p>

  <div class="callout"><b>Bottom line.</b> DRAM is column-localized (NoC cols 0 &amp; 5).
  <b>NoC0 (east&rarr;south)</b> disperses traffic across columns; <b>NoC1 (north&rarr;west)</b> concentrates it on the DRAM
  columns and along the consumers&rsquo; row. Reads and writes are <b>mirror images</b>: <code>read&middot;NoC0 &asymp; write&middot;NoC1</code>
  and <code>read&middot;NoC1 &asymp; write&middot;NoC0</code>. So the tt-metal default &mdash; <b>reads on NoC0, writes on NoC1</b>
  &mdash; is the fast pairing for the spread placements (row/diagonal).</div>

  <h2>How each NoC routes (real NPE rule, torus wrap)</h2>
  <p>One DRAM bank feeding 3 readers spread across distinct columns. The NoC is a <b>torus</b>: a route running off an edge
  (arrowhead) wraps to the opposite edge (same color).</p>
  <div class="schem">
    <div class="cell-card noc0"><h4>NoC0 &mdash; EAST then SOUTH</h4>
      <div class="m">turns down at each reader&rsquo;s own column &rarr; vertical legs on <b>distinct</b> columns (spread)</div>{sch0}</div>
    <div class="cell-card noc1"><h4>NoC1 &mdash; NORTH then WEST (wraps)</h4>
      <div class="m">climbs the DRAM column first &rarr; vertical legs <b>stack</b> on it; horizontal leg wraps west</div>{sch1}</div>
  </div>

  <h2>The matrix &mdash; per-link demand (measured device ns, tt-npe link demand)</h2>
  <p>Each square is a physical NoC core (row=y, col=x). <b>DR</b>=DRAM (cols 0 &amp; 5), <b>R</b>=a core doing the op.
  Arrows show flow direction; color/thickness = time-averaged link demand on a <b>shared scale across all 12 panels</b>.
  Green border = the faster NoC (measured) for that op+placement.</p>
  <div class="legend"><span>link demand</span>
    <svg width="200" height="14"><defs><linearGradient id="lg">{legend}</linearGradient></defs>
    <rect class="bar" width="198" height="12" x="1" y="1" fill="url(#lg)"/></svg>
    <span>low</span><span>high ({gmax})</span><span>&nbsp;&nbsp;<b>&rarr;</b> = flow direction</span></div>

  <h2 style="border:none;margin-bottom:0">NoC0</h2>
  {s_n0r}
  {s_n0w}
  <h2 style="border:none;margin-bottom:0">NoC1</h2>
  {s_n1r}
  {s_n1w}

  <h2>Device timing &mdash; all 12 cells</h2>
  <table><thead><tr><th>op</th><th>placement</th><th>NoC0 ns</th><th>NoC1 ns</th><th>faster</th></tr></thead>
    <tbody>{trows}</tbody></table>
  <p class="sub" style="font-size:12.5px">In-process profiler, ns/op. Green = faster NoC for that row. Single box, single arch (WH B0).</p>

  <h2>What it shows</h2>
  <ul>
    <li><b>Mirror symmetry:</b> read&middot;NoC0 &asymp; write&middot;NoC1 and read&middot;NoC1 &asymp; write&middot;NoC0 &mdash; same links, reversed.</li>
    <li><b>NoC0</b> wants work spread across columns (row/diagonal): low peak link. Stacked in one column it chokes.</li>
    <li><b>NoC1</b> pays a DRAM-column toll (north legs stack on cols 0/5) and collapses when consumers share a row.</li>
    <li><b>Default pairing wins:</b> reads&rarr;NoC0 + writes&rarr;NoC1 gives the two fast cells for row/diagonal.</li>
  </ul>
  <div class="foot">Regenerate: <code>python -m ttnn.operations.examples.noc_placement --report</code>
  (needs tt-npe built + reachable via $TT_NPE_HOME). Numbers are measured on one box/arch.</div>
</div></body></html>
"""


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
