# Operation Requirements: all_gather_rms_norm

## Definition
- **Formula**: every device of a `cluster_axis` group of G devices holds `x_p` of shape `(…, M, W_local)`;
  `x = concat_p x_p` along the last dim (`W = G·W_local`, group-position order). Every device returns the full
  `(…, M, W)` row:
  - `rms_norm`: `out = x · rsqrt(mean_W(x²) + eps) · gamma`
  - `layer_norm`: `out = (x − μ) · rsqrt(σ² + eps) · gamma + beta`, `μ`, `σ²` (biased) over the full gathered W.
- **PyTorch Reference**: `eval/golden_tests/all_gather_rms_norm/helpers.py::pytorch_all_gather_rms_norm`
  (concat the group's slices, `torch.nn.functional.rms_norm` / `layer_norm` over W in fp32, hand the row to every
  member).
- **Import Path**: `from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm`
- **Function Signature**:
  `all_gather_rms_norm(input_tensor, *, cluster_axis: int, gamma=None, beta=None, norm_type="rms_norm",
  epsilon=1e-6, topology=Topology.Linear, num_links=None, compute_kernel_config=None, memory_config=None,
  output_tensor=None) -> ttnn.Tensor` — one `ttnn.generic_op` over a `MeshProgramDescriptor` per call.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Measuring (perf refinements)**: whole-op device time only through the eval protocol (`eval/ccl_perf.run`: spin +
> 5 traced calls, fastest chip is the target, keep/revert by `python -m eval.ccl_perf compare`). The stock
> `eval/golden_tests/all_gather_rms_norm/perf_measure.py` currently fails with `OSError: File name too long` (it names
> its JSON after the full case id); `tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_agrms.py` runs the same
> protocol with short names (`case0` = PERF FOCUS rms 640×384, `case1` = PERF FOCUS layer_norm 640×32):
> `CCL_PERF_EXPERIMENTS=5 CCL_PERF_CASES=all CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_agrms.py -s`.
> Verifier baseline (same session, 5 experiments, fastest chip): case0 **21.29 µs** (roofline 7.6, target 10.1),
> case1 **16.33 µs** (roofline 0.6, target 0.8); neither dispatch-bound.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]; **fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode both, gamma_dtype {fp32, bf16, none}, gamma_layout {TILE, ROW_MAJOR,
  none}, norm_type {rms_norm, layer_norm}, beta_mode both, memory_layout {INTERLEAVED, HEIGHT_SHARDED,
  WIDTH_SHARDED}, cluster_axis {0, 1}, topology {Linear, Ring}, num_links {1, 2} — SUPPORTED == TARGET on every axis.
- **EXCLUSIONS**: `{float32, fp32_dest_acc_en=False}` (precision convention), `{bfloat8_b, w_non_aligned}` (block-float
  tail; measured necessary). The base op's `{bfloat8_b, h_non_aligned}` was dropped by the verifier (measured
  unnecessary).
- **Cores**: R1 mirror_resident_gather — ports (one per direction × link) next to their Ethernet cores, row-groups ×
  k members over `worker grid − ports_union` (focus: 20 × 4 = 80 compute cores), one block per member per call.
- **Compute config**: `compute_kernel_config` exposed (fidelity / approx / fp32_dest_acc_en honoured).
- **Golden baseline**: hash-sampled golden matrix + every non-matrix test (ring mock, regression, loose):
  2357 supported_pass, 174 supported_fail (143 numerical-precision → Refinement 2, 31 OOM → Refinement 1),
  644 xfail_expected (the two EXCLUSIONS), 0 xpass_drift, 0 xfail_wrong_mode.

### [ ] Refinement 1 — L1 fit for wide rows and L1-sharded I/O: port the base op's owner combine (k > 8) and compact gamma pages

**Goal**: move every cell that fails with `OOM: no member split fits L1` to passing. Phase 0 caps the member count at
`K_SYM_MAX = 8` (the base op's owner combine was deferred), so wide decode rows and every cell whose L1-sharded input +
gathered-output shards leave little L1 (budget measured 19–855 KB) run out of candidates. Port from the base op:

1. **Owner combine for k > 8** (base op R3 / its Refinement 5, `base_ops/rms_norm/op_design.md` "R3 wide_combine",
   `K_CAP = min(K_CAP_MAX, grid)`, `OWNER_MEMBER_COST` in `_occ_cost`): row r of a block → owner member r % k gathers
   the k partials, combines + finalizes, returns the row statistic(s) to the k members (faces 0/2 payload; for LN the
   owner runs the Chan combine on `(mean_m, M2_m)` pairs and returns `(K_g, inv_std, shift)`). Extend
   `_k_candidates` past 8 with the slice-aligned rule (`k = G·k_s`) and let `select_split` price k > 8 with the base
   op's owner cost. A larger k shrinks `cb_x` on the compute cores AND on the port cores (their `cb_x` mirrors the
   member block for address identity).
2. **Compact gamma / beta pages** (base op Perf 2 "Gamma residency": 192 B stride, row 0 of faces 0/1): `cb_gamma` /
   `cb_beta` are FULL-tile pages today (24 KB of the focus case's 98 KB; l1_ledger.md "Verifier deltas").
3. If a named cell still does not fit after 1–2: the deferred `R-zero-copy own shard` / output-shard regimes
   (op_design.md Regimes) — the shards already hold the bytes, so backing the member block on them removes a
   duplicate copy. Only for cells 1–2 leave failing.

Cells currently failing (sample; same pattern on the unsampled cells of these shapes):
HEIGHT_SHARDED `(1,32,4096)`, `(1,1,32,4096)`, `(1,1,32,8192)`, `(1,1,128,4096)`, `(128,8192)`, `(32,2048)`,
`(1,1,32,2048)`, `(1,1,640,1792)`, `(1,640,1792)`, `(1,1,2048,1280)` (fp32); WIDTH_SHARDED `(1,1,32,8192)`,
`(128,8192)`; INTERLEAVED `(128,8192)` layer_norm (k = 8 needs 858 KB vs 855 KB).

**Implementation skill**: /memory-budget-metal

**Verifier notes**: first because it is the from-base port Phase 0 did not land (a knob-turn on k; the owner topology
already exists in the base op, so it is ported, not re-derived). The two perf-flagged cases do not change regime
(focus k = 4, indexer k = 1): the owner path must not be selected for them — keep them in the guard set. The mirror
layout constrains only that every member of a group sits at the same core on every chip (`compute_cores` row-major,
groups of k consecutive cores — already true for any k). Ledger fold-ins: `cb_moment_a`/`cb_moment_b` aliasing on the
LN SFPU path, compact gamma (item 2).

**Done when**: every sampled OOM cell above passes (re-run their full golden families), no new failure elsewhere,
both PERF FOCUS cases within noise of the baseline above.

### [ ] Refinement 2 — layer_norm precision with 16-bit DEST (fp32_dest_acc_en=False)

**Goal**: move the bfloat16 `layer_norm` cells with `fp32_dest_acc_en=False` out of `numerical-precision` (143 sampled
cells, about half of that family: rel RMS 0.015–0.025 vs 0.015, PCC ≥ 0.9997). Measured decomposition (verifier
probes, op_design.md "Verification notes"): (a) a per-row **offset** of rms 0.012–0.020 that vanishes without row
offsets — μ is rounded to bf16 before `x − μ` (`K_g = bf16(μ)` and `shift` are formed from a μ that itself was
accumulated in 16-bit DEST); (b) a per-row **scale** of +0.7…1.3 % — the variance is under-estimated by bf16
accumulation of `Σc'²` across the member's tiles and of the Chan between-member term; residual after (a)+(b): 0.003
(output rounding). Levers: carry the mean as two words end to end — member: `mean_m = K_m + D1/n_m` kept as
`(K_m, D1/n_m)`; combine: `K_g` = a bf16 pivot, residual `r = Σ n_m (K_m − K_g)/W + Σ n_m d_m/W` (each term exact or
small), `shift = −r·inv_std` — and accumulate `Σc'²` per tile first (reduce each tile's 32 columns, then add the few
per-tile row sums) instead of DEST-accumulating full tiles across the member width. Not a scale bug: fp32 DEST is
unbiased after the Phase 0 verifier fix (rel RMS 0.0019) and the 16-bit residual after (a)+(b) is rounding-level.

**Verifier notes**: verifier-authored (the lever is an algorithmic accumulation order, not a CB format); method:
`.claude/references/numerical_stability_analysis_reference.md` (compensated / pivoted moments). Rerouting 16-bit LN
through the SFPU path was measured WORSE (90/90 sampled cells failed) — do not take it. The fp32-DEST path and the
indexer perf case must not regress (precision baseline test + case1 timing). Lands after Refinement 1 because the owner
combine of Refinement 1 also carries the LN Chan payload; build the two-word mean into that payload once.

**Done when**: the sampled `numerical-precision` cells (bf16, fp32_dest_acc_en=False, layer_norm) pass; the precision
baseline's `bf16_ln_acc16` rows meet the (bf16, False) tolerance and the test drops its known-gap carve-out.

### [ ] Refinement 3 — TRANSPORT: raise the focus case's link bandwidth

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` — PERF FOCUS, MLA q_a latent: `(1,1,640,384)` bf16, gamma ROW_MAJOR bf16,
rms_norm, Ring, 2 links, 1×4 ring, fp32_dest_acc_en=True — roofline 7.6 µs (links), target 10.1 µs, measured
21.29 µs. Per-chip breakdown (implementer zones, traced): ≈ 5 µs to the end of the ready fence (launch 2.8 + port
connection open + fence), fabric send ≈ 12.3 µs per port = ≈ 30 GB/s ≈ 62 % of a link — the largest single
component and the only one with a roofline to compare against (7.6 µs at full link rate for 1.5 slices over 2
links). TRANSPORT levers from the fabric examples (`ttnn/ttnn/operations/examples/fabric_all_gather/README.md` +
`report.md`, `examples/master.md` fabric entries, `.claude/references/blackhole-fabric.md`): packet fill (a 12-tile
member run is cut 7 + 5 at `chunk_pages = 7`; a slice-major or row-pair run gives full 14 KB packets), the
`port_batch_chunks` / `INC_EVERY` co-tune (example: 4 per batch, increment every 8 chunks — measured for DRAM-sourced
shards, not for 20-core L1 gathers), and the port reader's L1 reads from up to 20 compute cores per entry (one barrier
per batch). Measure transport in isolation by bytes / bandwidth as the examples do; the headline number comes from the
harness. No SUPPORTED change.

**Done when**: measured fastest-chip device time on case0 improves beyond noise with the slowest chip not regressing
(`eval.ccl_perf compare`), golden green, and no regression across the config-spanning guard set (case1; one golden
representative per kernel path × layout × placement: rms / LN FPU / LN SFPU / dense w_non_aligned / HEIGHT / WIDTH
sharded, Linear 2×2 and Ring 1×4).

### [ ] Refinement 4 — OVERLAP: shorten the focus case's exposed tail and start

**Type**: perf

**Goal**: same PERF FOCUS case (`LOOSE_CASES[0]`). The chain after the last landing is exposed ≈ 5.8 µs (moments
≈ 1, partial exchange + combine ≈ 1.1, normalize ≈ 3 of a 1 × 12-tile member block, last store ≈ 0.5), and the
transport starts only after ≈ 5 µs of launch + connection + fence. OVERLAP levers (op_design.md "Perf lamps",
measured only in the op): wave size (`INC_EVERY`-derived `wave_row_groups`: half / double / one row-group per wave),
`relay_lag_waves` 0 / 2, a narrower block for the last wave's row-groups only (k_s = 2 there) so its normalize + store
tail halves, landing notification straight to the member (fused inc on the entry's last packet to that member) instead
of the port reader's event loop, per-wave own-loaded counters. Catalog: `examples/master.md` (`double_buffer`,
`compute_block_size`). No SUPPORTED change.

**Verifier notes**: after Refinement 3 — the exposed tail is defined relative to the last landing, which moves with
the transport rate; re-measure the breakdown first.

**Done when**: measured fastest-chip win on case0 beyond noise, slowest chip not higher, golden green, no regression
on the guard set of Refinement 3.

### [ ] Refinement 5 — Indexer layer_norm case: local LN tail (column-0 SFPU scope) + own-slice moments before the landing

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[1]` — PERF FOCUS (LayerNorm), MLA indexer k: `(1,1,640,32)` bf16, gamma + beta
ROW_MAJOR bf16, layer_norm, Ring, 2 links — roofline 0.6 µs, target 0.8 µs, measured 16.33 µs. Zone breakdown
(verifier, op_design.md "Verification notes"): the local LN chain after the last landing is ≈ 8.5 µs of the call on
a 1 × 4-tile member block (k = 1: one member holds all 4 one-tile slices): moments 1.5–3 µs, the k = 1 Chan combine
≈ 3.4 µs (≈ 15 full-tile SFPU init+op pairs on column-0 data), normalize ≈ 1.7 µs. This local part dominates with
headroom, and it is NOT base-op code (the base op has no LN), so its perf rounds never worked it. Levers: the
catalog's `sfpu_tile_scope` (⭐⭐/⭐⭐⭐: column-0 / per-face SFPU scope; the base op's `finalize_col0_body` raw
pattern) for the member finalize and the Chan combine; fewer init switches per tile; OVERLAP — start the own slice's
member moments while the three remote one-tile slices are in flight (per-slice moments Chan-combined), so only the
last slice's moments sit on the tail. No SUPPORTED change.

**Verifier notes**: placed after the focus-case cross-device refinements per the queue rule; it precedes this case's
TRANSPORT refinement because the measured breakdown shows the local part dominating. Keep the Phase 0 verifier's
srcA-format fix in the normalize chain (it costs +0.17 µs here; it removed a −0.6 % per-row scale bias).

**Done when**: measured fastest-chip win on case1 beyond noise, slowest chip not higher, the precision baseline still
green (no bias regression), golden green, no regression on the guard set.

### [ ] Refinement 6 — TRANSPORT: indexer layer_norm case packetization and fixed exchange cost

**Type**: perf

**Goal**: same PERF FOCUS (LayerNorm) case (`LOOSE_CASES[1]`). Each slice is one tile wide, so the k = 1 member's
runs are one tile per (row, slice): the fabric carries 1-tile (2 KB) packets — packet-rate-bound (op_design.md
TRANSPORT lamp "Narrow-slice packetization"; `.claude/references/blackhole-fabric.md` rule 1) — and the exchange
(connection open, ready fence, ≈ 4.5–6.5 µs of sends for 61 KB per busiest link direction, relay hop) is far above
the 0.6 µs roofline. Levers: a slice-major `cb_x` layout (`[q][t][j]`, per-slice chain calls) so a slice's 20
tile-rows form one contiguous run (3 full packets instead of 20 one-tile packets), or k = G members (one slice each);
`num_links` 1 vs 2 at this payload (fabric_ring_exchange: a second link does not help ≤ 8 KiB). Take the TRANSPORT
numbers from the examples (`fabric_all_gather`, `fabric_ring_exchange` README / report) and cite them. No SUPPORTED
change.

**Done when**: measured fastest-chip win on case1 beyond noise, slowest chip not higher, golden green, no regression
on the guard set (incl. case0).
