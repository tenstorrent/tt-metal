# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-23
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). SF-GN scheme (stats-first, gather-normalized), one `ttnn.generic_op` / `MeshProgramDescriptor` per call; details below ("Phase 0 (ttnn-implementer)").
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, ROW_MAJOR, "none"], norm_type=[rms_norm, layer_norm], beta_mode=[beta, no_beta], memory_layout=[INTERLEAVED], cluster_axis=[0, 1], topology=[Linear, Ring], num_links=[1, 2]; EXCLUSIONS={float32 + fp32_dest_acc_en=False; bfloat8_b + w_non_aligned}
- **Accuracy achieved** (verifier, `test_all_gather_rms_norm_precision_baseline.py`, 10 cases / 7 shapes, 1×4 ring, 2 links): bf16 RMS PCC ≥ 0.999997, max_abs ≤ 0.088, rel_rms ≤ 0.0040; fp32 RMS PCC 1.000000, rel_rms 0.0017; bfloat8_b RMS PCC 0.999953, rel_rms 0.0097; bf16 LN PCC ≥ 0.999998, rel_rms ≤ 0.0019; fp32 LN rel_rms 0.0014.
- **Golden suite at Phase 0**: gate tier (1720 cells): 1080 supported_pass, 480 xfail_expected, 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode after the verifier fixes (before: 404 supported_fail, 3 xfail_wrong_mode). `agent_logs/verifier_report_summary.json`.
- **Perf at Phase 0** (eval harness, 5 experiments, fastest / slowest chip): focus 640×384 RMS 35.56 / 35.94 µs (roofline 7.6, target 10.1); LN 640×32 + γ + β 29.60 / 29.88 µs (roofline 0.6, target 0.8). Not dispatch-bound.
- **Issues encountered**: see "Verifier (Phase 0 hardening)" below — persistent L1 stats-table leak (404 golden failures), the stats-table cap ignored by the split, non-fp32 RMS partial sums (prompt MUST), the base op's bf8b × h_non_aligned exclusion lifted.
- **Tests added**: test_all_gather_rms_norm.py (planner), test_all_gather_rms_norm_debug.py (implementer), test_all_gather_rms_norm_precision_baseline.py (verifier)

## Verifier (Phase 0 hardening)
- **Persistent L1 stats table leaked, then collided with the CB region** (fixed). `stats_table()` cached a mesh L1 tensor *per table size* and never freed it. L1 allocates top-down, so every new shape pushed a new table lower (probe: 1570624 → 1529664 → …) until programs failed with "Statically allocated circular buffers ... clash with L1 buffers" (49 cells). The golden conftest deallocates the device tensors a failed call's frames hold, which freed the cached table under the cache and failed every later call with "Tensor is not allocated" (321 cells). The 27 remaining cells are the next item. **Fix:** the table is now `cb_stats_table` (CB 25), a program-scoped CB on the full core range. Identical CB lists give it the same L1 address on every core of every chip, and peers write into it only after the ready fence, i.e. while the receiving program is live. No persistent L1 is left besides the global semaphores. The cached DRAM staging buffer is re-created if it is no longer allocated. Gate tier: 404 supported_fail → 0.
- **Stats-table cap not honoured by `select_split`** (fixed): LN at R = 64 picked k = 3, a 288 KiB table over `STATS_TABLE_CAP` (27 cells: 2048×1280, 2048×384, 4×1×512×512, LN). `stats_k_cap(R, G, nstats)` now clamps k; the RuntimeError remains only if k = 1 does not fit.
- **RMS partial sums were truncated to tf32 when summed** (fixed; prompt rule "per-row partial sums MUST stay fp32 on the wire and when summed"): the local collapse (`cb_sumsq`) and the cross-chip combine (`cb_gathered`) went through the FPU / AccumulateViaAdd path, which reads Float32 CBs through srcA as tf32. Under fp32 DEST both CBs are now UnpackToDestFp32-tagged and reduced with `ReduceFp32Mode::Accurate` (SFPU), the same pattern LN's S1 uses. Perf null within noise (35.542 → 35.557 µs, noise 0.050).
- **EXCLUSIONS re-measured**: `{bfloat8_b, h_non_aligned}` lifted (19/19 gate cells pass: H padding is never element-masked); `{bfloat8_b, w_non_aligned}` kept (19/19 severity=bug with it lifted).
- Docs: stale reader header comment (word-copy mean broadcast / column scatter, both gone) rewritten; `l1_ledger.md` and `op_design.md` updated for `cb_stats_table`.

## Phase 0 (ttnn-implementer)

**Built:** the SF-GN scheme of `op_design.md` end to end, one `ttnn.generic_op` (one `MeshProgramDescriptor`) per
call.

- Workers = grid − 2L port cores, row-groups × k W members (`select_split`, fused cost model), **common runtime args
  only** (identity from the logical position + the port list). Local stats → 128 B fp32 records into both link-0 hub
  tables → stats round over the fabric (relay, one fused increment per slot) → hub multicast "stats ready" →
  per-worker combine of the P = G·k partitions → normalize into the resident `cb_out_block` → own write.
- Ports pull the own normalized slice from worker L1 and send it into the peers' **final output pages** (bank-
  contiguous runs, up to 4 runs per scatter packet, relay with arrival counters, balanced ring far slot), ready fence
  per call, counters re-armed every call (trace-safe, determinism test passes).
- R_U (W_local % 32 ≠ 0): slot-contiguous DRAM staging + element-shifted assembly by the workers.
- LayerNorm: one stats round with shifted sums (m, T = Σ(x−K)², K = tf32(m)); exact combine in fp32 SFPU;
  centering as (x − K) − (mean − K).
- Gamma / beta: TILE `(1,1,1,W)` or ROW_MAJOR sticks `(1,1,ceil(W/32),32)`, this chip's global columns.

**Acceptance:** `tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm.py` 15/15 (production
mode); debug matrix `test_all_gather_rms_norm_debug.py` 8/8. `--dev` (watcher): the cases that got a device passed;
on this box the device bring-up under `--dev` repeatedly stalls on "waiting for active ethernet core" (fixture
errors, environment, not the op).

**SUPPORTED (Phase 0):** dtype {fp32, bf16, bfp8_b}; fp32_dest_acc_en {True}; TILE; alignment all three; rank 2–4;
gamma / beta all formats; norm_type both; memory_layout {INTERLEAVED} (DRAM input, DRAM-interleaved output);
cluster_axis {0, 1}; topology {Linear, Ring}; num_links {1, 2}.

**Not landed from the from-base list (one generality refinement, base op mechanisms):**
- `memory_layout` HEIGHT_SHARDED / WIDTH_SHARDED: zero-copy `cb_x` on the shard (base op
  `cb_descriptor_from_sharded_tensor`, its `RMS_SHARDED` reader path), port cores chosen off the shard grid, k = shard
  count for WIDTH, W-tail masked (not zeroed) in the caller's shard; sharded output landing walks shard cores
  instead of DRAM banks (the walk's `nb` / page→core mapping).
- `fp32_dest_acc_en=False` (bf16 / bfp8): the LN SFPU paths use UnpackToDestFp32 (needs fp32 DEST); the design's
  16-bit-DEST variant (FPU centering, row 12' of Block Operation Realization) is not built. RMS would only need the
  tag list emptied (the gather transpose would then truncate the partial sums to tf32 — base op Refinement 1
  tolerances).

**Perf (eval `perf_measure.py`, 5 experiments, fastest chip, QB2 1×4 ring, 2 links):**

| Step | focus 640×384 RMS | LN 640×32 + γ + β |
|------|-------------------|-------------------|
| first correct version | 52.63 µs | 37.86 µs |
| scatter packets (4 runs) + x before gamma + gather zeroing off the critical path | 38.41 | 33.19 |
| row-form gather + exact DEST transpose + stats-ready multicast | 35.54 | 32.22 |
| LN centering by the local shift K (no fp32 mean broadcast) | 35.54 | 29.58 |
| roofline / target | 7.6 / 10.1 | 0.6 / 0.8 |

Not dispatch-bound (max gap 0.6 µs). Measured nulls / rejects (knobs kept at the design values): scatter runs 1 / 2
(49.2 / 44.0 µs), batch 2 / 6 / 8 (36.0 / 37.0 / 39.0), inc_every 4 / 16 (38.7 / 38.4, pre-gather-change base
38.4); own entry from DRAM instead of the L1 pull 34.85 (−0.7 µs, the pull is not the bottleneck); aligned slice
forced through the staging regime 40.3 µs (final-address landing stays).

**Where the time goes now (focus, chip without launch-skew wait, single profiled call):** local stats done ~2.3 µs;
stats round to ~6 µs; gather + combine + normalize + own write to ~11 µs; **bulk own entry ~15 µs** (240 KB per
port at ~16 GB/s: interleaved final pages give only 1–2-page bank runs per row, so a 4-run scatter packet is ~6
pages / 4 DRAM writes), relay ~4 µs, landing tail ~3 µs. LN 640×32: local stats ~3 µs (x → partial), records
1.4 µs, combine + normalize ~7 µs of compute for one tile-row, bulk ~11 µs for a 20-tile slot.

**Refinement candidates (ranked by the breakdown):**
1. TRANSPORT — bulk packet efficiency on interleaved final pages (the dominant term): a destination-run layout with
   longer runs, or fewer / larger DRAM writes per packet; measure against `fabric_all_gather`'s long-run rate.
2. OVERLAP — bulk start granularity (lamp): the own entry waits for all 80 workers' own-ready; per-row-group
   readiness would start the first packets earlier.
3. local (LN) — the LN combine / normalize chain (~7 µs per tile-row on the critical path: many small chains and
   reduces with per-call init / reconfig).
