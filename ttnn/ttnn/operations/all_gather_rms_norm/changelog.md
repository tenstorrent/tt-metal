# all_gather_rms_norm — changelog

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
