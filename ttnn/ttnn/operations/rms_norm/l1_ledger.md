# L1 Ledger: rms_norm

Symbols: `Rb` ≤ 16 (G_CAP); `Wb ≤ Wt`; `Sb=min(Wb,16)`; `G=Rb*Sb ≤ 16`; `Tx`∈{2048,4096} input tile bytes; `T32`=4096; `es`∈{2,4}; `Wt` bounded by predicate `Rb*Wb*x_depth*Tx ≤ X_BUDGET=512KB` (R1) else `Wb` = largest multiple of Sb with `2*Wb*Tx ≤ X_BUDGET` (R2). `Wt_g` bounded by `gamma_resident ⇒ Wt*Tg ≤ 128KB`. `x_depth∈{1,2}`, `rm_depth=2`, `out_depth=2`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_x_rm (RM only) | `rm_depth*32` pages of `Sb*32*es` B | 32 (one tile-row segment) | rows: streams; Wt: streams (segments); Rb: streams (tile-row at a time) | input dtype sticks | reader | compute | segment | Not aliased with cb_x: both live while tilize runs. Capacity>live = double buffer |
| cb_x | `x_depth*Rb*Wb` | `Rb*Wb` (R1, held through pass 2) / `Rb*Sb` (R2) | Rb: spans; Wb: spans (R1) / streams (R2) | input dtype tile | reader (TILE) / compute (RM) | compute | row block (R1) / chunk (R2) | Largest CB; cannot alias cb_sq/out (all live in pass 2). x_depth=2 gap = prefetch next block |
| cb_scaler | 1 (constant) | 1 | none | bf16 | reader | compute | program | constant, trivially tiny |
| cb_gamma | `gamma_resident ? Wt : 2*Sb` | same | Wt: spans (resident) / streams; Rb: not scaled (shared across rows) | gamma dtype tile | reader | compute | program / segment | no sharing candidate (reuse-shared operand) |
| cb_sq | `G` | `G` | Rb: spans; Sb: spans; Wt: streams | fp32 | compute | compute | segment | Could alias cb_out_tiles (disjoint lifetimes: pass 1 vs pass 2) — rejected: costs ≤ 64 KB and DFB aliasing lockstep risk; revisit under L1 pressure |
| cb_accum | **removed** | – | – | – | – | – | – | Implementation: `Accumulate::at(cb_stat, seg)` uses cb_stat as both accumulator and reduce output (the reduce helper reloads/pops then re-pushes), so no separate CB |
| cb_stat | `Rb` | `Rb` | Rb: spans; Wt: not scaled | fp32 | compute | compute | row block (through pass 2) | Σx² → inv_rms **in place** (Rule 3.2): no cb_inv_rms |
| cb_psend (W-split only) | `Rb` | `Rb` | Rb: spans; n_w: not scaled (this core's own partial); Wt: reduced | im dtype | compute | writer | row block | Hand-off copy of the finished partial: `Accumulate` keeps cb_stat in flux until the last segment, so the writer cannot consume cb_stat directly |
| cb_part (W-split only) | `2*Rb*n_w` | `Rb*n_w` | spans `n_w` (one tile per slice per row); streams over row blocks (2 parity halves) | im dtype | writer (peers' NoC unicast writes + local push) | compute | row block | Double-buffered by block parity so a faster peer's next-block partial never clobbers an unconsumed block; bounded by PART_BUDGET (256 KB) in `pick_w_split` |
| cb_out_tiles | `out_depth*G` | `G` | Rb: spans; Sb: spans; Wt: streams | output dtype | compute | writer (TILE) / compute untilize (RM) | segment | Scale result packs straight here (pack-into-destination); no cb_xn (fused DestReuse chain) |
| cb_xn (gamma only) | `G` | `G` | Rb/Sb: span | fp32 | compute | compute | segment | Added: DestReuseBinary cannot express the Row broadcast, so x*inv_rms goes to cb_xn then *gamma -> cb_out_tiles (the design's sanctioned fallback) |
| cb_gamma_rm (gamma only) | 1 page of `Sb*32*es_g` | 1 | – | gamma dtype | reader | reader | per gamma segment | reader-private scratch for the stick segment before the row-0 scatter |
| cb_out_rm (RM only) | `2*Sb` tile pages (output tile size; untilize helper is symmetric) | `Sb` | rows streams; Wt streams | output dtype tile-sized pages | compute | writer | segment | untilize output; cannot be in place (different page format) |

Footprint (bytes): `x_depth*Rb*Wb*Tx + [RM] rm_depth*32*Sb*32*es + [gamma] (Wt or 2*Sb)*Tg + G*T32 + 2*Rb*T32 + out_depth*G*Tx + [RM] 64*Sb*32*es + 2*2048(scaler page)`.
Scaling: cb_x with Rb·Wb (bounded by X_BUDGET); cb_gamma with Wt (bounded by GAMMA_BUDGET else Sb); all others with `G ≤ 16`, `Sb ≤ 16` constants.
Worst case (fp32 RM, gamma resident 128KB): 512K (x) + 128K (rm in) + 128K (gamma) + 64K (sq) + 32K (stat/accum, Rb≤16·4K·2 worst) + 128K (out tiles) + 128K (rm out) ≈ 1.12 MB < usable L1 (~1.3 MB). Inventory was minimized first: no cb_inv_rms (in place), no cb_xn (fused chain), no gamma tilize CB, no solve beyond two byte budgets.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (R1) / 2 (R2) | R1: row resident across both passes. R2: row too wide to stay resident; second pass re-reads | none |
| out | 1 | written once per valid element | none |
| gamma | 1 per core (resident) / 1 per row block (streamed, ≈ W·es bytes vs 32·W·es per tile-row of x) | reuse-shared operand read by each core | none (mcast deferred R6) |

Totals R1: `2·(N_valid·es)` DRAM bytes + gamma·n_cores; R2: `3·(N_valid·es)`.

> Cheapest-traffic split considered: W-split across cores + combine (R3) — saves the R2 second x read (`N·es` DRAM bytes) and fills the grid when rows < cores, adds ≈ `n_w·4 KB` NoC bytes per row block. Implemented: row-parallel R1 with R2 fallback. Deferred because Phase 0's regimes cover every shape correctly and R3 is a scheme-change requiring a combine (mcast/semaphore); the `cb_stat` slot and the `Accumulate` partial structure keep it reachable. For R1-fitting shapes with rows ≥ cores, the implemented split IS the cheapest (1× x, no cross-core traffic).

> **Refinement 2 (W-split, implemented)**: when `pick_w_split` finds `n_w > 1` (divisor of Wt) that cuts the estimated per-core work below 0.75x, a row group of `n_w` cores each holds one W slice (x read once, slice resident / streamed per the slice width), computes its partial Sigma x^2 tile per row, and **all-gathers** the partials inside the group (unicast NoC writes into every peer's `cb_part`, parity semaphore, then local L1-accumulating-pack sum into `cb_stat`). Extra cross-core traffic: `n_w * n_w * Rb * tile` bytes per group per row block. x DRAM crossings stay 1 (slice resident) — the R2 second read disappears for wide rows whose slice fits. Row-split path (n_w == 1) is unchanged.
