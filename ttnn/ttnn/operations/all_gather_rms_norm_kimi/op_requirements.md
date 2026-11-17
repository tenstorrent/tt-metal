# Operation Requirements: all_gather_rms_norm_kimi

## Definition
- **Formula**: `full = concat_p x[p]` along the last dim, over the G devices of a `cluster_axis=1` ring group. Then `out[q] = full / sqrt(mean_w(full²) + ε) · γ` on every member q. The per-row sum of squares is computed and summed in fp32, in the fixed order p = 0..G−1.
- **PyTorch Reference**:
  ```python
  def ref(x_stacked, gamma, eps=1e-6):          # x_stacked (R, C, 1, 1, M, W_local), gamma (W,)
      out = []
      for r in range(x_stacked.shape[0]):
          full = torch.cat([x_stacked[r, c].float() for c in range(x_stacked.shape[1])], dim=-1)
          out.append(torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=gamma.float(), eps=eps))
      return out                                 # every device of row r returns out[r]
  ```
- **Import Path**: `from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi`
- **Function Signature**:
  ```python
  all_gather_rms_norm_kimi(input_tensor: ttnn.Tensor, *, gamma: ttnn.Tensor, cluster_axis: int,
                           epsilon: float = 1e-6, topology: ttnn.Topology = ttnn.Topology.Ring,
                           num_links: int = 2,
                           compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Queue shape.** `feature_spec.py` is a single-cell suite. Every TARGET axis has one value and Phase 0 already supports all of them, so TARGET − SUPPORTED is empty and there are **no generality refinements**. Every entry below is a measured perf refinement on the one PERF FOCUS `LOOSE_CASES` cell, in the order the fused-CCL rule asks for: cross-device first (labelled TRANSPORT / OVERLAP), local compute after. The order is ranked by the measured per-chip breakdown in `verification_report.md` § Performance.

**The perf target (all refinements).** `LOOSE_CASES[0]`, marked `PERF FOCUS — the mandatory target`:
- **Shape**: per device `(1,1,640,384)` in, `(1,1,640,1536)` out.
- **Topology**: G = 4 Ring along `cluster_axis=1`, `num_links=2`, 1×4 mesh, `FABRIC_2D_TORUS_XY`.
- **Data**: bf16 TILE DRAM-interleaved input; γ bf16 ROW_MAJOR sticks `(1,1,48,32)`.
- **Compute config**: `fp32_dest_acc_en=True`, the default HiFi4.
- **Bounds**: `roofline_us = 7.6` (links), `target_us = 10.1`.

Every axis value and knob of this contract is in SUPPORTED today. Optimize this exact config, never a proxy.

**How to measure (all refinements).**
- **Headline number**: `CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile eval/golden_tests/all_gather_rms_norm_kimi/perf_measure.py -s`.
- **Keep or revert**: `python -m eval.ccl_perf compare <base.json> <cand.json>`. A win needs the fastest chip lower beyond noise **and** the slowest chip not higher beyond noise.
- **Baseline after verification**: `perf/verifier_final.json`, fastest chip 29.82 µs, slowest 32.29 µs.
- **Per-chip breakdown**: apply `perf/ablation_hooks.patch` (env `AGRN_ABLATE` ∈ `nobulk`, `nocompute`, `nostore`, comma-combined), run `perf/perf_ablate_nocheck.py.txt` copied into `eval/golden_tests/all_gather_rms_norm_kimi/`, then revert. Never commit the hooks.

**Phase 0 breakdown (fastest chip, µs; ablations measured on the code as handed over):**

| Variant | µs |
|---|---|
| skeleton (fence + stats round + loads) | 8.84 |
| + store only | 11.74 |
| + compute only | 17.81 |
| local only (`nobulk`) | 23.98 |
| transport only (`nocompute,nostore`) | 26.32 |
| full op | 29.92 (29.82 after the verifier fixes) |

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = [tile_aligned], rank = [4]
- **SUPPORTED op-specific axes**: gamma_dtype = [bfloat16], gamma_layout = [ROW_MAJOR], memory_layout = [INTERLEAVED], cluster_axis = [1], topology = [Ring], num_links = [2]
- **Cores**: 80 compute cores (G × 20 row blocks, two merged rectangles) + 4 ports + 4 receivers + 1 hub on the 110-core grid; 21 idle
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32 DEST acc, approx off; fidelity / approx honoured, `fp32_dest_acc_en` gated to True
- **Golden baseline**: 7 / 7 (1 `supported_pass` + 6 regression checks; 0 loud categories)
- **Perf**: fastest chip 29.82 µs, slowest 32.29 µs (roofline 7.6, target 10.1); not dispatch-bound

### [x] Refinement 1 — TRANSPORT: bring the bulk ring exchange to the example's measured transport

**Type**: perf

**Goal (TRANSPORT)**: on the PERF FOCUS cell, the L1→L1 bulk gather alone takes 26.3 µs. That is the `nocompute,nostore` ablation: the bulk adds 17.5 µs over the 8.8 µs skeleton. `examples/fabric_all_gather` measured its balanced ring at 256 tiles per chip on 2 links at about 18 µs per call on this same ring, DRAM→DRAM, including its ≈ 8.4 µs fixed cost. This op sends 240 tiles from L1, so about 8 µs of its transport is excess over the example. Close that gap by taking the example's measured transport choices (cite its `report.md`; do not re-derive them).

The levers, roughly in order of expected payoff:
- **Port placement.** Phase 0 centres the 4 ports on logical row 0. The example places each port at the worker nearest its link's Ethernet core by NoC1 hops (`_worker_nearest_noc1`). The example finds those Ethernet cores with a probe program, which this op may not run (one dispatch per invocation). Derive each link's Ethernet core on the host without a device dispatch instead (e.g. from the forwarding link index → fabric channel → Ethernet core mapping in the control plane / SoC descriptor), then place the ports, and move the receivers / hub out of their way.
- **Packet size.** The chunk is 6 tiles = 12 KiB against a 14 KiB payload. A 7-tile chunk needs a block tile count that is a multiple of 7, e.g. through `block_h` / `block_w`.
- **Knob co-tune.** `PORT_BATCH_CHUNKS` (2, 3, 4) and `INC_EVERY` (2, 4, 8). Finer increments shorten the relay's wait for `relay_ready`; L1 landing acks faster than the DRAM landing the default was measured on.
- **Stats placement.** The link-0 stats entries interleave with the bulk. Check whether they delay the link-0 bulk.

No SUPPORTED change.

**Verifier notes**: this is first because transport-only (26.3 µs) is the largest term in the breakdown, ahead of local-only (24.0 µs). Measure it as TRANSPORT: report the transport-only ablation time, and its bandwidth against the bytes sent (busiest link direction 1.5 slices / 2 links = 360 KiB per link), next to the headline. Re-run the full breakdown at the end. It decides whether Refinement 2 or Refinement 3 is the bigger remaining term, and the next implementer should know. Never call `ttnn.set_fabric_config`, and never add a second dispatch, even a one-time probe.

**Done when**:
- The transport-only ablation and the harness's fastest chip both improve beyond noise on the PERF FOCUS cell, with the slowest chip not higher.
- The golden suite (7/7), the acceptance tests (including the tall `block_h > 1` regime) and the bit-identical-repeat regression test stay green.
- No regression on the tall regime shape `(1,1,2048,384)`, which is the other distinct kernel path (`block_h = 3`, ragged last block).

**Outcome**: harness fastest chip 29.83 → 24.67 µs, slowest 32.32 → 27.77 µs (`perf/r1_final.json`). Transport-only (`nocompute,nostore`) 26.32 → 19.57 µs: the bulk adds 11.4 µs over the 8.19 µs skeleton (was 17.5), i.e. 360 KiB per link direction at 32.4 GB/s = 67% of the 48.5 GB/s link (was 21 GB/s, 43%). The example's 256-tile 2-link ring moves its bytes at about 39 GB/s per link. Tall `(1,1,2048,384)`: 80.4 → 58.0 µs. Levers: (1) **port placement**: each port at the row-0 worker nearest by NoC1 hops to its link's Ethernet core, the core found on the host from the fabric connection's router channel (no probe dispatch). This gave −4.05 µs. (2) **`PORT_BATCH_CHUNKS` 4 → 2**: −1.12 µs (1 and 3 are worse). `INC_EVERY` 2 and 4 regress at batch 4 and at batch 2; 16 only skews the chips; 8 stays. Not done: **7-tile packets** (a 12-tile slice block splits 6+6 or 7+5, the same 2 packets per block; only a block of 7k tiles gains, which means restructuring `block_h`/`block_w`, so it belongs to Refinement 3's block knobs) and **stats on link 1** (the overlap item already filed in Refinement 2). **Breakdown now** (fastest chip, µs): skeleton 8.19 / +store 11.79 / +compute 17.92 / local-only (`nobulk`) **23.71** / transport-only 19.57 / nocompute 20.78 / nostore 20.40 / full 24.70. **Local-only is now the binding term** (it is 1.0 µs under the full op; transport-only is 4.1 µs under local-only), so per the verifier note Refinement 3 (local compute) is the bigger remaining term than Refinement 2's transport tail.

### [ ] Refinement 2 — OVERLAP: shrink the exposed tail and the stats-critical chain

**Type**: perf

**Goal (OVERLAP)**: two exposed costs on the PERF FOCUS cell.
- **Tail.** The full op takes 3.6 µs longer than transport alone (29.9 vs 26.3). That is the chain from the last landing to the output write: the last chunk's normalize, its DRAM store, and per-chunk `landed` notifications that arrive late because `inc_every` batches the arrival counter, so a chunk's notification can lag by up to 7 chunks mid-stream.
- **Stats chain.** The 8.8 µs skeleton is the chain the S precompute waits on: ready fence → own partial (x² + reduce) → export → hub → 2-hop fp32 relay → hub multicast. The own-slice cores have no landing to wait for, so this chain plus their compute sets their finish time.

Shorten both with overlap structure:
- notify per landed entry/chunk so the consumer starts earlier (the design's "finer last-entry chunks vs whole-block wait" lamp);
- let the store of chunk k overlap the normalize of chunk k+1 (`out_depth_chunks`);
- start the own partial earlier: as soon as the own x lands, before γ prep;
- send P_{p−1} on link 1's forward port so the relayed stats do not queue behind link-0 bulk (the design's "stats on link 0 only" lamp);
- tolerate launch skew between chips (the spread is 2.5 µs).

Measure only in the op; never on a dummy exchange. Relevant catalog entries: `fabric_ring_exchange` (fence ≈ 5 µs is the floor of the stats round), `double_buffer` (writer depth). No SUPPORTED change.

**Verifier notes**: after Refinement 1, because transport sets when the last chunk lands, and the tail is measured from there. If Refinement 1's re-measured breakdown shows `nobulk` (local-only) above transport-only, swap the order with Refinement 3. The checkboxes and the IDs stay as filed.

**Done when**:
- `(full − transport-only)` and the harness's fastest chip both improve beyond noise on the PERF FOCUS cell, with the slowest chip not higher.
- The golden suite, the acceptance tests and the determinism test stay green.
- No regression on the tall regime shape `(1,1,2048,384)`.

### [ ] Refinement 3 — Local compute chain: cut the per-core compute + store time

**Type**: perf

**Goal**: on the PERF FOCUS cell the op takes 24.0 µs even with no bulk at all (`nobulk`). Per-core compute adds 9.0 µs over the skeleton (`nobulk,nostore` 17.8 vs 8.8) for only about 36 tile operations per core (12 × x², 1 reduce, the combine/rsqrt, 1 bcast, 12 × scale, 12 × normalize). The DRAM store is serialized behind it (+6.2 µs). This is the floor that remains once transport is fixed, and the own-slice cores already finish at about this time in the full op.

Speed up the per-core chain using the relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`:
- `compute_block_size`: fewer per-phase reconfig/init passes; drop the format reconfig where the format is constant across a boundary.
- `sfpu_tile_scope`: rsqrt / combine only on column 0 of the partial tiles; the rest are garbage lanes that are never read.
- `compute_fusion`, `row_reduce_accumulate`: the x² + row-sum path.
- The **block knobs** (knob-turns the planner exposed): use the 21 idle cores, e.g. `block_w` < `Wl_t` with same-slice partial cores, or test `block_h = 2` (the design's grid-sync lamp), to trade per-core tiles against notification / multicast fan-out.
- Overlapping the store with compute (writer depth).

Profile per TRISC (`DeviceZoneScopedN`) first to see which phase dominates. Keep HiFi4 and `fp32_dest_acc_en=True`: they are the target's config. No SUPPORTED change.

**Verifier notes**: filed after the cross-device refinements per the fused-CCL ordering rule, but it is nearly co-dominant: local-only 24.0 µs vs transport-only 26.3 µs. Fixing transport alone cannot take the op below about 24 µs, so this refinement must run even if Refinements 1–2 bring large wins. Any `block_w` split must keep the fp32 partial path bit-exact and the fixed q = 0..G−1 sum order; the determinism test checks bit-identical repeats. The L1 headroom is large (132 KiB of about 1.5 MB per compute core).

**Done when**:
- The `nobulk` ablation and the harness's fastest chip both improve beyond noise on the PERF FOCUS cell, with the slowest chip not higher.
- The golden tolerance `(PCC ≥ 0.999, rel-RMS ≤ 0.01)` still holds (Phase 0 measures 0.999998 / 0.0024).
- The golden suite, the acceptance tests and the determinism test stay green.
- No regression on the tall regime shape `(1,1,2048,384)`.
