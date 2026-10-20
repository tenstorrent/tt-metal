import json

BEST = ' class="best"'
R = json.load(open("gb.json"))
MODELS = ["Kimi K3", "GLM-5.3", "GLM-5.3-Flash", "DSv4.1-Flash", "Kimi K2.7"]
LAY = [("2D-tp", "2D token-split"), ("own2D", "Gather → 2D owners"), ("1D", "1D column")]


def row(name, cells, best_lo=True, fmt=lambda v: f"{v:,.0f}"):
    vals = [c[0] for c in cells]
    b = min(vals)
    tds = []
    for v, extra in cells:
        cls = ' class="best"' if v <= b * 1.005 else (' class="bad"' if v >= b * 1.15 else "")
        tds.append(f"<td{cls}>{fmt(v)}{extra}</td>")
    return f"<tr><td>{name}</td>{''.join(tds)}</tr>"


def layout_table(key, lev, label_extra=lambda r: ""):
    h = "<tr><th>Model</th>" + "".join(f"<th>{l}</th>" for _, l in LAY) + "</tr>"
    rows = [h]
    for st in ("4x4", "8x4"):
        for T in (5120, 16384):
            rows.append(f'<tr class="grp"><td colspan="4">{st[0]} × {st[2]} · chunk {T:,}</td></tr>')
            for m in MODELS:
                r = R[f"{m}|{st}|{T}|{lev}"]
                rows.append(row(m, [(r[k][key], label_extra(r[k])) for k, _ in LAY]))
    return "\n".join(rows)


bound = lambda v: f' <span class="muted">{"D" if v["full_bound"] == "DRAM" else "L"}</span>'
t_ov = layout_table("ov", "tie")
t_full = layout_table("full", "tie", bound)
# serial (today's ops one after another) vs overlapped SE, 2D-rep
lv = [
    "<tr><th>Model</th><th>Stage</th><th>Today, all serial</th><th>SE overlapped</th><th>Full overlap, today's combine</th><th>+ ties split</th><th>+ fp8 rows</th></tr>"
]
for T in (5120, 16384):
    lv.append(f'<tr class="grp"><td colspan="7">chunk {T:,} · 2D token-split, TP-split SE</td></tr>')
    for m in MODELS:
        for st in ("4x4", "8x4"):
            a = R[f"{m}|{st}|{T}|today"]["2D-tp"]
            t = R[f"{m}|{st}|{T}|tie"]["2D-tp"]
            f8 = R[f"{m}|{st}|{T}|tie+fp8"]["2D-tp"]
            vals = [a["serial"], a["ov"], a["full"], t["full"], f8["full"]]
            b = min(vals)
            tds = "".join(
                f'<td{BEST if v <= b * 1.005 else ""}>{v:,.0f}{bound(x) if i >= 2 else ""}</td>'
                for i, (v, x) in enumerate(zip(vals, [a, a, a, t, f8]))
            )
            lv.append(f'<tr><td>{m}</td><td class="l">{st[0]} × {st[2]}</td>{tds}</tr>')
t_lev = "\n".join(lv)
sec = f"""<section id="measured">
  <h2>With measured efficiencies: 2D vs 1D, and where the shared expert goes</h2>
  <p class="muted">Added 2026-10-20. Same layouts, re-costed with what the LoudBox ops actually reach instead of floors, and with link loads from a uniform-routing simulation of each stage (4 × 4: TP axis wraps, SP a line; 8 × 4: torus). µs per chip per MoE layer. Green is the best layout in its row, red is 15% or more behind it. D / L: the overlapped middle is DRAM- or link-bound.</p>
  <ul class="notes">
    <li><b>The shared expert is TP-split in every layout here</b> (gate / up split on I_s, down reduce-scattered). Replicating it is ruled out.</li>
    <li><b>1D never wins.</b> It is 3–39% behind the best 2D layout (usually 15–30%) on every model, stage and chunk. Its combine is no cheaper: once ring ties are split, the 8 × 4 combine hot link is the same in 1D and 2D, and on 4 × 4 both hot links sit in the middle of the SP line. Its dispatch carries 1.4–2× the link load, because every TP column sends its own copy of a token along its SP ring, where 2D sends it once and branches. The x all-gather and the final reduce-scatter also land on the critical path. Sharing that reduce-scatter with the shared expert does not make up for it.</li>
    <li><b>The TP-split shared expert has to be overlapped.</b> Run serially, its all-gather, matmuls and reduce-scatter cost 17–41% (K3 on 4 × 4: 3,852 vs 3,163 µs, +22%), matching the LoudBox measurement (traced 4,402 µs TP vs 3,700 replicated, serial). Overlapped, the all-gather and the matmuls fit under router + routing gather + planner + dispatch, and the reduce-scatter fits on the TP links next to the combine.</li>
    <li><b>Gather → 2D owners costs the same as token-split</b> once overlapped (within 0–4%; the x all-gather hides under router + routing gather + planner). The residual layout can follow what attention prefers (MLA: H-split).</li>
    <li><b>At larger chunks the combine is the bottleneck</b>: at 16K nearly every case is link-bound. Splitting ties is worth 11–16% on the 8 × 4 torus wherever it is link-bound, and nothing on 4 × 4, whose hot link is in the middle of the SP line. An fp8 / bfp8 wire is worth 34–43% at 16K and little at 5,120. Source pre-reduction (K3: 2.3× fewer rows) is the other lever.</li>
  </ul>
  <h3>Dispatch before the experts, shared expert overlapped with routing and dispatch, ties split</h3>
  <div class="tw"><table>{t_ov}</table></div>
  <h3>Full overlap: dispatch, combine and shared expert under the weight stream, ties split</h3>
  <div class="tw"><table>{t_full}</table></div>
  <h3>Steps for 2D token-split</h3>
  <div class="tw"><table>{t_lev}</table></div>
  <ul class="notes">
    <li><b>Measured inputs</b> (LoudBox 2 × 4, 200G): fused flat expert + combine at the single-chip-calibrated flat model, plus 5% of the smaller of the two (GLM-5.3-Flash S640: 1,305 / 1,113 → 1,362 µs). Combine at 81% (K3), 75% (GLM), 70% (V4.1-Flash) of the busiest-link rate with the full-L1 arena. Sparse 2D dispatch at 82% of its floor. TP all-gather 87% and reduce-scatter 74% of the link floor (ttnn CCLs, K3 shapes). lin2d shared expert at 338 GB/s. Router 86, routing gather 36 → ~54 on 32 chips, planner 57 µs per 41k routing entries (linear), prep and zeroing 19 µs.</li>
    <li><b>Check</b>: the model gives ~3.5 ms for the K3 block under LoudBox conditions, against 3.32 ms measured traced (within 5%).</li>
    <li><b>1D</b> here is the best 1D: drop-off along each column's SP ring, combine back along it into fp32 partials of the row's tokens, one reduce-scatter over TP for routed + shared (K3: the routed partials reduce-scattered in latent space, then latent up on own tokens).</li>
    <li><b>Full overlap</b> is a target structure: weights stream from the start, so dispatch, the combine and the shared expert all hide under the weight read, unless the links or the extra DRAM bytes (the shared expert's quarter, K3's latent weights) take longer. Dispatch and combine rows on the same directed link are added together.</li>
    <li>Uniform routing only. Model source: <code>global_block.py</code> in the session scratchpad (2026-10-20).</li>
  </ul>
</section>

"""
p = "/tmp/claude-1211409210/-localdev-mstaletovic-tt-metal/12382886-eb89-49cd-bf2a-5a73e27dba59/scratchpad/artifact-files/57b97233-83c7-447a-b660-7526ac571a1f/index.html"
s = open(p).read()
anchor = "<section>\n  <h2>What these numbers assume</h2>"
a = s.index('<section id="measured">')
b = s.index(anchor)
s = s[:a] + sec + s[b:]
if "h3{" not in s:
    s = s.replace(
        "h2{font-size:21px;margin:0;font-weight:600;text-wrap:balance}",
        "h2{font-size:21px;margin:0;font-weight:600;text-wrap:balance}\nh3{font-size:16px;margin:8px 0 0;font-weight:600;text-wrap:balance}",
    )
open(p, "w").write(s)
print("ok", len(s))
