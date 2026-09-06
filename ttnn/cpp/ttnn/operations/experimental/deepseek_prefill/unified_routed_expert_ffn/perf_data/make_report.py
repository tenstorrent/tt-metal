# Generate the markdown report from the JSONL results in results/.
import json, os, sys, collections, statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def fpcc(v):
    return "" if v is None else f"{v:.3f}"


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


env = json.load(open(os.path.join(R, "env.json"))) if os.path.exists(os.path.join(R, "env.json")) else {}
nb = env.get("dram_channels", 7)
peak = 64 * nb
out = []
w = out.append
w("# Grouped `unified_routed_expert_ffn` — design notes and P100 measurements\n")
narr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_narrative.md")
DESC = {
    "kimi_u": "Kimi-K2.7 dims, 12 experts x 107 tok (production average, EP32)",
    "kimi_e4": "Kimi-K2.7 dims, 4 experts x 107 tok",
    "kimi_e24": "Kimi-K2.7 dims, 24 experts x 107 tok (intragalaxy 2-stage)",
    "kimi_zipf": "Kimi-K2.7 dims, 12 experts, Zipf counts 640..32 with 2 empty",
    "kimi_zeros": "Kimi-K2.7 dims, 12 experts, 7 empty",
    "kimi_giant": "Kimi-K2.7 dims, one 2048-token expert + 11 x 64",
    "m3_u4": "MiniMax-M3 dims, 4 experts x 160 tok (EP32)",
    "m3_u8": "MiniMax-M3 dims, 8 experts x 160 tok (EP16 / 2-stage PP)",
    "m3_u16": "MiniMax-M3 dims, 16 experts x 160 tok (EP8 / 4-stage PP)",
    "m3_skew8": "MiniMax-M3 dims, 8 experts skewed 800..0",
    "m3_giant4": "MiniMax-M3 dims, one 2048-token expert + 3 small",
}


def headline():
    ab = load("ab.jsonl")
    rows = [r for r in ab if "error" not in r]
    if not rows:
        return "(A/B sweep results not available)"
    lines = [
        "| case | dtype | legacy us | best grouped | us | GB/s | speedup | PCC min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    seen = collections.OrderedDict()
    for r in rows:
        seen.setdefault((r["dist"], r["dtype"]), []).append(r)
    for (dist, dtype), rs in seen.items():
        base = next((x for x in rs if x["cfg"] == "legacy"), None)
        cand = [x for x in rs if x["cfg"] != "legacy" and x.get("pcc_ok") is not False]
        if not base or not cand:
            continue
        best = min(cand, key=lambda x: x["ns"])
        lines.append(
            f"| {DESC.get(dist, dist)} | {dtype} | {base['ns']/1e3:.0f} | {best['cfg']} | {best['ns']/1e3:.0f} | {best['GBps']:.0f} | "
            f"**{base['ns']/best['ns']:.2f}x** | {fpcc(best.get('pcc_min'))} |"
        )
    return "\n".join(lines)


if os.path.exists(narr):
    w(open(narr).read().replace("{{HEADLINE}}", headline()))
w("\n# Measurements\n")
w(
    f"Board: Blackhole p100a, grid {env.get('grid_x')}x{env.get('grid_y')}, **{nb} DRAM channels (peak {peak} GB/s)**, "
    f"realtime profiler {'active' if env.get('rt_profiler') else 'INACTIVE'}, AICLK 1.35 GHz. bf4_b tile 576 B, bf8_b 1088 B. "
    "All device times are realtime-profiler program durations (median of iterations, warm program cache).\n"
)

# ---- DRAM ceiling
dc = load("dram_ceiling.jsonl")
if dc:
    w("\n## 1. DRAM read-bandwidth ceilings (generic_op microbenchmark)\n")
    w("| mode | dtype | xfer B | placement | NoCs | outstanding/trid group | readers | GB/s | % peak |")
    w("|---|---|---|---|---|---|---|---|---|")
    keep = [
        r
        for r in dc
        if (r["mode"] == 0 and r["placement"] == "row0" and r["group"] in (4, 16) and r["k"] in (8, 22, 44, 110))
        or (r["mode"] == 1 and r["xfer"] > 4000 and r["k"] == nb and r["placement"] == "bank" and r["group"] == 4)
        or (
            r["mode"] == 1
            and r["placement"] == "col"
            and r["xfer"] > 4000
            and r["group"] == 4
            and r["k"] in (nb, 5 * nb, 10 * nb)
        )
    ]
    for r in sorted(
        keep, key=lambda r: (r["mode"], r["dtype"], r["xfer"], r["placement"], r["nocs"], r["group"], r["k"])
    ):
        w(
            f"| {'interleaved pages' if r['mode']==0 else 'bank-direct bursts'} | {r['dtype']} | {r['xfer']} | {r['placement']} | {r['nocs']} | {r['group']} | {r['k']} | {r['GBps']:.0f} | {100*r['GBps']/peak:.0f}% |"
        )
    w(
        "\nTakeaways: single-tile interleaved reads on ONE NoC cap at ~245 GB/s (bf4) / ~340 (bf8) regardless of reader count; "
        "the same reads from all cores on BOTH NoCs reach 90-95% of peak; 16 KB bursts from 7 bank-adjacent cores reach 437 GB/s but "
        "the same bursts from a column-per-bank layout only ~300-365 GB/s (placement), so the op keeps interleaved weights and "
        "makes every core read on both NoCs."
    )

# ---- legacy baseline
lg = load("legacy.jsonl")
if lg:
    w("\n## 2. Legacy op baseline (num_row_groups=0)\n")
    w("| case | model | dtype | x layout | E | tokens | us | GB/s | TFLOP/s | PCC min |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in lg:
        w(
            f"| {r['tag']} | {r['model']} | {r['dtype']} | {r['layout']} | {r['E']} | {r['tokens']} | {r['ns']/1e3:.1f} | {r['GBps']:.0f} | {r['TFLOPs']:.0f} | {fpcc(r['pcc_min'])} |"
        )

# ---- A/B
ab = load("ab.jsonl")
if ab:
    w("\n## 3. A/B: legacy vs grouped configurations\n")
    w(
        "Config key: G = row groups, r = rows used, (m = per-core M cap, d = weight CB depth, s = band mode, ds = down split). "
        "GB/s counts weight bytes actually streamed (one full read per M-chunk).\n"
    )
    groups = collections.OrderedDict()
    for r in ab:
        if "error" in r:
            continue
        groups.setdefault((r["dist"], r["dtype"], r.get("layout", "x_rm")), []).append(r)
    for (dist, dtype, layout), rows in groups.items():
        base = next((x for x in rows if x["cfg"] == "legacy"), None)
        w(
            f"\n### {dist} — {dtype} — {layout} (E={rows[0]['E'] if 'E' in rows[0] else len(rows[0]['counts'])}, counts={rows[0]['counts']})\n"
        )
        w("| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |")
        w("|---|---|---|---|---|---|---|")
        best = min(rows, key=lambda x: x["ns"])
        for x in rows:
            sp = f"{base['ns']/x['ns']:.2f}x" if base else "-"
            mark = "**" if x is best and x["cfg"] != "legacy" else ""
            w(
                f"| {mark}{x['cfg']}{mark} | {x['ns']/1e3:.1f} | {x['GBps']:.0f} | {100*x['GBps']/peak:.0f}% | {x['TFLOPs']:.0f} | {sp} | {fpcc(x.get('pcc_min'))} |"
            )
    errs = [r for r in ab if "error" in r]
    if errs:
        w("\nErrors:\n")
        for r in errs:
            w(f"- {r['dist']} {r['dtype']} {r['cfg']}: {r['error'][:200]}")

# ---- grouped correctness/perf log
gr = load("grouped.jsonl")
if gr:
    w("\n## 4. Grouped-path development log (test_grouped.py runs, chronological; pre-fix rows included)\n")
    w("| dist | dtype | layout | G | rows | cols | mmax | depth | strided | PCC ok | PCC min | us |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in gr[-80:]:
        p = r.get("perf") or {}
        w(
            f"| {r['dist']} | {r['dtype']} | {r['layout']} | {r['G']} | {r['rows']} | {r['cols']} | {r['mmax']} | {r['depth']} | {r['strided']} | {r['ok']} | {fpcc(r['pcc_min'])} | {p.get('ns', 0)/1e3:.1f} |"
        )

# ---- stress + pytest summary (from the overnight chain log)
chain = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "chain.log")
if os.path.exists(chain):
    lines = [l.strip() for l in open(chain) if l.strip()]
    st = [l for l in lines if l.startswith("stress ")]
    if st:
        w(
            "\n## 5. Repeated-dispatch stress (no watcher; 7 configs x 2 rounds x 4 dispatches per pair, PCC checked after each config)\n"
        )
        w("| distribution | dtype | exit | configs OK | PCC failures |")
        w("|---|---|---|---|---|")
        for l in st:
            t = l.split()
            w(f"| {t[1]} | {t[2]} | {t[3].split('=')[1]} | {t[4].split('=')[1]}/14 | {t[5]} |")
        w(
            "\nPlus 56/56 dispatches of the previously hanging kimi_u bf8 sequence under the light watcher after the fix."
        )
    pt = [l for l in lines if l.startswith("pytest ")]
    if pt:
        w("\n## 6. Correctness suites\n")
        for l in pt:
            w(f"- {l[7:]}")
        w(
            "\n`test_grouped_routed_expert.py` covers the distributions above x geometries (G10r10, G5r10, G8r8, G4r8, G2r8) x bf4/bf8 x x_rm/x_tile, "
            "plus cache-hit with changing counts, all-empty, count clamp and the legacy path."
        )

print("\n".join(out))
