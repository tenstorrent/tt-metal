# Operation Requirements: all_gather_rms_norm

## Definition
- **Formula**: per `cluster_axis` group of G devices (group position p), `full = concat_p x[p]` along the last dim
  (`W = G·W_local`), `out[q] = full · rsqrt(mean_w(full²) + eps) · gamma` on every member q (the same `(..., M, W)`
  on every device of the group).
- **PyTorch Reference**:
  ```python
  def reference(x_per_device, gamma, eps):  # x_per_device: [x[0], ..., x[G-1]], each (..., M, W_local)
      full = torch.cat(x_per_device, dim=-1).float()
      return torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=gamma, eps=eps)  # on every member
  ```
- **Import Path**: `from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm`
- **Function Signature**:
  ```python
  all_gather_rms_norm(input_tensor: ttnn.Tensor, *, cluster_axis: int, gamma: ttnn.Tensor | None = None,
                      epsilon: float = 1e-6, topology: ttnn.Topology = ttnn.Topology.Linear,
                      num_links: int | None = None, compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
                      memory_config: ttnn.MemoryConfig | None = None,
                      output_tensor: ttnn.Tensor | None = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Fused CCL labels**: every cross-device perf item is labelled **TRANSPORT** (how the bytes cross the fabric —
> measured in isolation, judged by bandwidth against the bytes sent, taken from the `fabric_*` examples where they
> cover it) or **OVERLAP** (hiding the transport behind the op's own work — measured only in the op, as the
> transport time the op leaves exposed per call). Headline numbers come from
> `eval/golden_tests/all_gather_rms_norm/perf_measure.py` (fastest chip; keep/revert by `python -m eval.ccl_perf compare`).
> Breakdowns: the op's ablation hook `AGRMS_ABLATE=EMPTY|LOCAL|TRANSPORT` via
> `tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_ablation_all_gather_rms_norm.py`, per-stage zones via
> `AGRMS_PROFILE_ZONES=1` (off by default: they overflow the harness's profiler buffers).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]
- **SUPPORTED fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **SUPPORTED gamma axes**: gamma_mode ∈ {gamma, no_gamma}, gamma_dtype ∈ {float32, bfloat16, none},
  gamma_layout ∈ {TILE, ROW_MAJOR (sticks), none}
- **SUPPORTED memory_layout**: {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED} — sharded I/O through sharded
  `TensorAccessor`s (the work split ignores the shard grid; see Refinement 1)
- **SUPPORTED cross-device axes**: cluster_axis ∈ {0, 1}, topology ∈ {Linear, Ring}, num_links ∈ {1, 2}
- **EXCLUSIONS**: `{float32, fp32_dest_acc_en=False}` (precision convention, permanent);
  `{bfloat8_b, w_non_aligned}` (Refinement 2)
- **Cores**: multi-core: 2·L port cores per chip (one per direction × link, placed next to their Ethernet core) +
  96–104 member cores on QB2 in round-robin row groups of k slice-nested members (focus: k = 8, 12 groups)
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, approx off; any
  fidelity / approx honored
- **Golden baseline**: 10,503 supported cells pass, 0 supported_fail / xpass_drift / xfail_wrong_mode;
  3,000 xfail_expected (the two EXCLUSIONS), 180 infeasible_skipped (harness shard geometry), 66 ring-mock /
  regression tests pass (`verifier_report.json`)
- **Perf baseline (fastest chip, eval harness, QB2 1×4 ring, 2 links)**: 640×384 [FOCUS] **31.36 µs**
  (roofline 7.6, target 10.1); 640×512 **41.51 µs** (roofline 10.1, target 13.5); 640×1792 **99.89 µs**
  (roofline 35.5, target 47.3). Not dispatch-bound.
- **Focus breakdown**: full 31.36 · TRANSPORT-only 26.22 · LOCAL-only 14.73 · empty program 0.75 µs →
  the transport binds; 5.1 µs is exposed beyond it; the local work is the next binder below ~15 µs.

### [x] Refinement 1 — Native sharded placement: consume the input shard in place, pack into the output shard

**Goal**: complete `HEIGHT_SHARDED` / `WIDTH_SHARDED` (already in `SUPPORTED["memory_layout"]`) **natively**. Today
every member reads its rows through a sharded `TensorAccessor` with a work split that ignores the shard grid, and
writes the output the same way. That is correct, but it is not the base op's shard-resident mechanism. Port that
mechanism (base op Refinements 2 and 4: the split follows the shard grid, one group per shard row (HEIGHT: k = 1, full
rows) or a single group (WIDTH: k = shard columns); `cb_x` / `cb_out` backed on the input / output shards via
`ttnn.cb_descriptor_from_sharded_tensor`; the block is the resident shard) and adapt it to the fused landing:

- HEIGHT: a shard core holds rows [a, b) × W_local in, and rows [a, b) × W out (the golden output shard is built for
  the gathered shape with the same strategy). Its own-slice columns come from the local input shard with no NoC read.
  The G−1 remote slices land straight into this core's **output shard** at their final column offset, so the output
  shard *is* the landing buffer. The core then normalizes in place and packs into the output shard. This keeps the
  invariant that the landing address is computable by the sender (the output shard sits at one address on every chip
  for the same per-device spec).
- WIDTH: the shard columns are the members (k = shard grid size, symmetric / owner combine as built). The landing
  for the remote slices follows the same in-output-shard rule.

No `SUPPORTED` change: this replaces a tolerated path with the implemented one.

**Verifier notes**:
- Scheme-change (complicated sharding, difficulty tier 1): the work split moves from round-robin row groups to the
  shard grid, so it stands alone and goes first.
- The design deferred this (`sharded_zero_copy`) because port cores can sit on shard cores. That constraint is
  real for the golden harness: `eval.sharding.auto_shard_config` shards over the full compute grid, and on QB2 the
  port cores occupy logical row 0. Shards resident on a port core are the only ones that may be consumed through a
  cross-core read by a worker; every other shard must be consumed in place. Keep the `TensorAccessor` path as the
  fallback for shard grids the native path cannot place, and state the predicate.
- `cb_x_landing` must stay at an identical address on every chip (senders compute peer addresses from it). If the
  output shard becomes the landing buffer, prove that address invariant for the mesh's per-device buffers, or keep
  `cb_x_landing` and copy into the output shard (a local L1 copy, not a DRAM round trip).
- No skill covers placement / sharding (`/memory-layouts` is RM↔TILE, not placement): do not attach it.

**Done when**: every golden HEIGHT_SHARDED / WIDTH_SHARDED cell still passes. On a non-port shard core, the input
shard is read with no `TensorAccessor` / NoC read of the core's own shard, and the output is packed directly into
the output shard. Device-ns on a representative sharded shape (e.g. HEIGHT `(1,1,640,384)` ring L2 and WIDTH
`(1,1,64,512)`) does not regress against Phase 0. Interleaved cells and the perf cases do not regress.
**Outcome**: native path built for both layouts, behind a stated predicate (`native_shard_split` + `native_keeps_up`).
HEIGHT: one group per shard, k = 1. The own slice is read in place from `cb_x_in` (compute, compose, ports), and the
output is packed into `cb_out_tiles` backed on the output shard. WIDTH: k = output shard columns, own tiles gathered
from the input shards, output packed into the shard. The ports move off the shard cores. Remote slices still land in
`cb_x_landing`, not the output shard: the ports relay from the landing, so normalizing it in place would forward
normalized data downstream.
Fastest chip, QB2 1×4 ring L2, fallback → native: HEIGHT 640×384 **41.4 → 34.9 µs**, HEIGHT 2048×384 93.4 → 68.2 µs.
These shapes lose natively: HEIGHT 128×512 (20.2 → 27.6), WIDTH 64×512 (14.3 → 32.6: 64 one-tile members, so
one-tile packets and a 64-way combine; TRANSPORT-only 9.5 → 21.2, LOCAL-only 6.7 → 14.6), WIDTH 32×4096 (24.9 → 32.7).
The keep-up gate (per-core work ≤ T_xfer, the split's own cost model) keeps those on the fallback, so they measure as
in Phase 0 (20.25, 14.28 µs). Perf cases unchanged: 31.46 / 41.70 / 100.02 µs (Phase 0 31.36 / 41.51 / 99.89).
Next: make narrow-shard WIDTH keep up. Merge adjacent one-tile members' runs into one packet (scatter headers,
Refinement 3) and give the k-way combine a hierarchy. Not done here: that is transport / combine work, not placement.

### [ ] Refinement 2 — bfloat8_b with W_local non-tile-aligned (regime N compose on a bf16 staging row)

**Goal**: move the 720 `{dtype: bfloat8_b, alignment: w_non_aligned}` cells out of `EXCLUSIONS` to passing. Regime N
composes each dense output tile by shifting datums across tile columns (slices land at non-tile offsets). That
cannot be done on packed bfp8 tiles, where 16 datums share an exponent. The native path: the transport keeps moving
whole bfp8 tiles unchanged. Before the compose, compute unpacks the row's landing tiles and packs them into a
**bf16 staging window**. The reader composes dense tiles from that window (the existing `compose_tile` on 2-byte
datums), and the normalize packs the result as bfp8 output. Zero-filling the slice tails stays as built: zeroing
bfp8 mantissa bytes yields exact zeros. Then drop the EXCLUSION.

**Verifier notes**:
- Easiest remaining generality (dtype × alignment on an existing path), so it goes after Refinement 1.
- Size the staging window as a stream (`compose_depth` units of `out_unit`, or one landed row of `land_w_tiles`),
  not as a resident copy of the landing. Add a row for it to `l1_ledger.md` (a compute→reader→compute round trip,
  `Float16_b` pages, lifetime one row).
- The base op excludes the same cell, so there is no mechanism to port. The `/numeric-formats-metal` skill sends
  this cell class to EXCLUSIONS by default, so no skill pointer: this is kernel work in regime N only.
- `{float32, fp32_dest_acc_en=False}` stays excluded. It is the prompt-mandated precision-convention EXCLUSION, not a
  refinement candidate.

**Done when**: all 720 golden `bfloat8_b × w_non_aligned` cells pass (PCC ≥ 0.99 against the device-quantized input,
the golden gate). The EXCLUSION is removed from the op file. No regression on the bf16 / fp32 regime N cells or on
the perf cases.

### [ ] Refinement 3 — TRANSPORT: fill the packets (perf, focus case)

**Type**: perf

**Goal**: speed up the transport on the PERF FOCUS case (`feature_spec.LOOSE_CASES[0]`: `(1,1,640,384)` per chip,
bf16, fp32_dest_acc_en=True, gamma bf16 ROW_MAJOR sticks, INTERLEAVED, cluster_axis 1, Ring, 2 links, 1×4 ring
under FABRIC_2D_TORUS_XY; perf goal: target_us 10.1, roofline 7.6). Measured: TRANSPORT-only ablation **26.2 µs** of
the 31.4 µs call. The `fabric_all_gather` example moves the same ~240–256 tiles/chip in about 18 µs on 2 links on
this ring (fit: 8.4 µs + bytes / 77.2 GB/s per link, balanced ring), so about 8 µs is transport overhead the example
does not pay. The implementer's zones show the ports are packet-rate bound: ~0.45 µs per packet whatever its size.
Own-slice packets carry 6 tiles (one k = 8 member's run) and far-half packets 3 tiles (link part `Wt_lp/(2L)`), not
the payload's 7 (14 KiB + 64 B). Levers, from the design's TRANSPORT lamps and `ttnn/ttnn/operations/examples/master.md`
(`fabric_all_gather`, `fabric_ring_exchange`):
- split the far half across links **by rows**, not by columns (6-tile packets instead of 3);
- `NocUnicastScatterCommandHeader` packets (≤ 4 chunks) to merge two members' runs into one full packet;
- re-tune `item_min_chunks` (the fused-increment frequency: 4 now; the example found every 4–8 chunks ≈ free) and
  `port_cb_bytes` / `group` for 4–7-chunk items;
- `relay_lag` (2, measured: 1 → 2 was 35.3 → 31.4 µs) re-checked after the packet change.
No SUPPORTED change.

**Done when**: the TRANSPORT-only ablation and the full-op fastest-chip mean on the focus case both drop, with
`eval.ccl_perf compare` calling the full-op change a win (fastest chip lower beyond noise, slowest chip not higher).
Report the transport time against bytes received per chip, next to the example's fit. The golden suite stays green.
No regression across the guard set: one representative per transport path × regime: Linear and Ring, L = 1 and 2,
regime A k = 1 / symmetric / owner, regime N, one sharded cell, and 640×512 / 640×1792.

### [ ] Refinement 4 — OVERLAP: shorten the exposed tail and co-tune the split around the ports (perf, flagged cases)

**Type**: perf

**Goal**: reduce what stays exposed beyond the transport on the flagged cases. Focus: full 31.4 µs vs TRANSPORT-only
26.2 µs, so **5.1 µs exposed**. The LOCAL-only ablation is 14.7 µs and becomes the binder once Refinement 3 brings
the transport under ~15 µs, so hiding it is the point of this phase. Second target: the third flagged case
`(1,1,640,1792)` (99.9 µs vs roofline 35.5 / target 47.3). There the split rule picks k = 12 (owner, 3 blocks per
core), but the implementer measured 87.4 µs at k = 8 symmetric vs 99.8 µs at k = 12. That is the design's lamp
"KEEP_UP_FRACTION / cost constants". OVERLAP levers (design lamps, `ttnn/ttnn/operations/examples/master.md`
`double_buffer`, `compute_block_size`):
- co-tune `block_row_tiles` (1 now; the lamp asks for 2 on row-rich shapes) with k and the keep-up rule, against the
  measured tail;
- landing notification: fuse the increment into the member's own last chunk instead of the port reader's NoC
  increment after `sem_arrival` (the "port-mediated notifications" lamp);
- per-row-group readiness instead of "all local members loaded" before the first own packet;
- the compute lookahead on the combine wait (non-blocking landed check of block b+1);
- Linear only (not on the flagged cases, fix if touched): a receive-only line-end port notifies its members only
  after **all** its items arrived (`has_out == false` skips the per-step notify loop).
No SUPPORTED change.

**Done when**: measured on the focus case, the exposed time (full − TRANSPORT-only, same build) drops and the full-op
fastest chip improves (`eval.ccl_perf compare` win). The 640×1792 case improves toward its target. The golden suite
stays green. No regression across the same guard set as Refinement 3.

### [ ] Refinement 5 — Local work: cut the members' compute + combine time (perf, focus case)

**Type**: perf

**Goal**: the LOCAL-only ablation on the focus case is **14.7 µs** for 0.49 MB in + 1.97 MB out per chip (DRAM
floor ≈ 4.8 µs at 512 GB/s nominal, ~5.5 µs at the 447 GB/s eltwise-achieved rate). Once Refinements 3–4 pull the
transport and the tail down, this sets the call time, still above target_us 10.1. The fused split is not the base
op's: k = 8 members of 6 tiles each, 2 tile-rows per core, a 64-message symmetric combine per row and group,
HiFi4 + fp32 DEST. So the base op's perf rounds do not already cover it. Levers, from `ttnn/ttnn/operations/examples/master.md`
(`tensix_all_reduce`, `compute_fusion`, `compute_block_size`) and the base op's changelog (Perf 1/2): fewer combine
messages (smaller k at equal tail, or the owner topology's 2k messages), batching the member writes per barrier,
the x·γ pre-scale lamp during the combine wait (gate on the precision baseline). No SUPPORTED change.

**Done when**: the LOCAL-only ablation and the full-op fastest chip on the focus case both improve
(`eval.ccl_perf compare` win), precision stays within the baseline in `verification_report.md`, the golden suite
stays green, and the guard set does not regress.
