# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-07
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One
  fused `ttnn.generic_op` (one MeshProgramDescriptor) per call: ports (one per direction × link) relay raw input
  tiles over Fabric 2D straight into the L1 landing slots of the member cores (balanced ring / line, ready fence,
  one fused write + increment per item). Members (round-robin row groups of k slice-nested cores) compute Σx²,
  combine partials in-chip (symmetric k ≤ 8 / owner k > 8, ported from the base op `rms_norm`) and normalize as rows
  land. Regime N composes dense output tiles for W_local % 32 ≠ 0.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False], layout=[TILE],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma],
  gamma_dtype=[float32, bfloat16, none], gamma_layout=[TILE, ROW_MAJOR, none],
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2]. EXCLUSIONS: {float32, fp32_dest_acc_en=False}, {bfloat8_b, w_non_aligned}.
- **Accuracy achieved** (4 shapes × 4 precision corners, `test_all_gather_rms_norm_precision_baseline.py`):
  bf16 + fp32 DEST PCC ≥ 0.999997, max_abs ≤ 0.086, rel_rms ≤ 0.0038; bf16 + fp16 DEST PCC ≥ 0.999992,
  rel_rms ≤ 0.0062; fp32 PCC = 1.000000, max_abs ≤ 0.025, rel_rms ≤ 0.0012; bf8b PCC ≥ 0.99995, rel_rms ≤ 0.0099.
- **Golden suite at Phase 0**: 10,503 / 10,503 supported cells passing; 3,000 xfail_expected (the two EXCLUSIONS),
  180 infeasible_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`, run with
  `EVAL_CAPTURE_L1=0`, see verification_report.md).
- **Perf at Phase 0** (eval harness, fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 31.36 µs (target 10.1);
  640×512 41.51 µs (target 13.5); 640×1792 99.89 µs (target 47.3). Focus breakdown: TRANSPORT-only 26.22,
  LOCAL-only 14.73, empty 0.75 µs.
- **Issues encountered** (verifier fixes):
  - Correctness: race in the in-chip partial combine. A single cumulative partials semaphore let a fast peer's
    block-(b+1) increment complete a slow peer's block-b count, so a stale partial produced a wrong per-row
    inv_rms (intermittent `inf` / precision misses on the 2048-row golden cells after a program switch). Fixed with
    one semaphore per block parity (symmetric and owner combine). Reproducer: 3/20 bad → 0/40.
  - Per-stage device zones gated behind `AGRMS_PROFILE_ZONES=1`: they overflowed the perf harness's profiler buffers
    and inflated the headline number.
  - `GAMMA_READ_ALIGN` single-sourced (writer CT arg). `dst_full_sync_en` added to the program hash key. Ledger and
    design docs updated.
- **Tests added**: `test_all_gather_rms_norm_precision_baseline.py` (acceptance `test_all_gather_rms_norm.py` and
  `test_all_gather_rms_norm_debug.py` came from the planner / implementer).

## Refinement 1 — Native sharded placement: consume the input shard in place, pack into the output shard
- Date: 2026-11-07
- What was done: native shard-resident path for HEIGHT_SHARDED / WIDTH_SHARDED (scheme change: the shard grid fixes
  the split), ported from the base op's Refinements 2 / 4 and adapted to the fused landing.
  - HEIGHT (input and output share the row partition): one group per shard, k = 1, contiguous rows
    (`rows_per_group` in the port walk). `cb_x_in` (new CB 11) is backed on the input shard. Compute reads the
    own-slice columns from it for Σx² and the normalize segments, the regime-N compose reads them from it, and the
    ports send the own slice straight from it. No own-slice load and no NoC read of the core's own shard. Regime N
    zeroes the own tail tile's W padding in place: that is tile padding, not logical data.
  - WIDTH (full-height output shards, regime A): one group, k = output shard columns, member m = output shard core m.
    Own tiles are gathered from the input shards (cross-core reads: the input partition is not the output's). A
    ragged last member pushes pad pages so each row keeps the shard pitch.
  - Both: `cb_out_tiles` is backed on the output shard. Compute packs every row at its final place; the writer only
    waits / pops. Ports are placed off the member cores (`plan(avoid=...)`): the shard grid starts at core (0,0),
    where Phase 0 placed ports.
  - Remote slices still land in `cb_x_landing`, not in the output shard. The ports relay from the landing, so
    normalizing it in place would forward normalized data (the verifier note's "keep cb_x_landing" branch, with no copy
    at all: compute packs the landing straight into the shard).
  - Predicate (`native_shard_split` + `native_keeps_up`): TILE L1 shards, ROW_MAJOR orientation, the partitions above,
    the CBs fit L1, >= 2L free cores off the shard grid, and the shard-fixed split keeps up with the transport
    (per-core work ≤ `NATIVE_KEEP_UP_FRACTION` × T_xfer, the split's own cost model). Otherwise the sharded
    `TensorAccessor` fallback (Phase 0 path) runs. `AGRMS_NATIVE_SHARD=0` forces the fallback for A/B.
  - Reused: the landing / port item walk / combine / gamma paths, unchanged except a row-placement hook and an
    own-slice source switch. Added: `native_shard_split`, `native_keeps_up` / `transfer_time` / `combine_cost`
    (the cost model factored out of `select_split`), `cb_x_in`, the shard-backed `cb_out_tiles`, compute's
    `normalize_segment` / `pad_out_row`, and the writer's zero-copy store.
- Perf (fastest chip, QB2 1×4 ring L2, fallback → native): HEIGHT 640×384 41.41 → 34.93 µs; HEIGHT 2048×384
  93.41 → 68.21 µs. Kept on the fallback by the gate (natively slower): HEIGHT 128×512 (20.25 vs 27.60), WIDTH 64×512
  (14.28 vs 32.63; TRANSPORT-only 9.45 vs 21.15, LOCAL-only 6.66 vs 14.56: 64 one-tile members), WIDTH 32×4096 (24.92
  vs 32.71, k = 103). Perf cases: 31.46 / 41.70 / 100.02 µs (Phase 0 31.36 / 41.51 / 99.89, within noise).
- Accuracy achieved: unchanged numerics (same compute chain). Acceptance PCC gates (bf16 0.995, fp32 0.999, bf8b 0.99)
  met on every case.
- Golden test progress: targeted slices of sharded cells: 76/76 (one per shape × layout, final code) + 114/114 on a
  second random sample, plus 6 infeasible-L1 skips, all in Phase 0's infeasible HEIGHT shapes. The full suite is the
  harness's run.
- Issues encountered: native WIDTH loses on every golden geometry wider than a few tiles. Its shard grid gives
  one-tile members, which fragments the fabric packets and widens the combine, so the keep-up gate holds it to small
  shapes. Few-shard HEIGHT loses for the same reason (whole gathered rows on 1–4 cores).
- Tests added: `test_all_gather_rms_norm_native_shard.py` (6 cases: HEIGHT regime A / N, WIDTH symmetric / owner /
  ragged, each asserting the native placement was taken, plus a WIDTH case asserting the gate's fallback).
  `perf_ablation_all_gather_rms_norm.py` gains `AGRMS_PERF_SHAPE` / `AGRMS_PERF_MEMORY_LAYOUT` overrides for sharded A/B.
