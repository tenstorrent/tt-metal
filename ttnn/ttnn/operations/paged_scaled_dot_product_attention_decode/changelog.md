# Changelog: paged_scaled_dot_product_attention_decode

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier): a flash-decode over a virtual key sequence (key/value heads laid end to end, row-masked), split over `G_u` clusters × 4 NEOs by a √N worker rule, partials in an L1-interleaved scratch, a semaphore-gated combine on the group's rank-0 cluster; one `ttnn.generic_op` (Metal 2.0 `ProgramSpec`), thread plan (4, 4, 2).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], memory_config=[L1], sharding=[height], output_memory=[DRAM], kv_heads_mode=[grouped], batch=[single], alignment=[tile_aligned]
- **Accuracy achieved**: fp32 DEST: PCC ≥ 0.999995, max_abs_err ≤ 0.0031, rel-RMS ≤ 0.0032; 16-bit DEST: PCC ≥ 0.999915, max_abs_err ≤ 0.0086, rel-RMS ≤ 0.0132 (measured on 4 shapes × 2 accumulators via test_paged_scaled_dot_product_attention_decode_precision_baseline.py; got/true ratio median 0.996–1.000)
- **Golden suite at Phase 0**: 23 / 23 supported cells passing (14 PHASE0 cells + 9 loose cases), 178 xfail_expected, 64 invalid_skipped, 10 / 10 rejections; 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: verifier fixes — (1) the writer stored partial tiles one per barrier; now one barrier per served NEO's whole partial (barriers per writer hart per user on the partials ring 2·(Dt+2) → 2; gain in time unconfirmed on ttsim); (2) `_resolve_compute_config` restated the default's values as `getattr` fallbacks (rule R8); now read from `default_compute_kernel_config()`. Implementer deviations from the design, each forced by Quasar/ttsim, are listed below.
- **Tests added**: test_paged_scaled_dot_product_attention_decode.py (harness acceptance), test_paged_scaled_dot_product_attention_decode_debug.py, test_paged_scaled_dot_product_attention_decode_threads.py, test_paged_scaled_dot_product_attention_decode_precision_baseline.py, test_paged_scaled_dot_product_attention_decode_callers.py (rules R10 / R11)

### Implementer notes (Phase 0)


Built from op_design.md: the virtual-key-sequence flash-decode, split over `G_u · C` workers (one NEO each) of a
cluster group, partials written to an L1-interleaved scratch tensor, a semaphore-gated combine on the group's rank-0
cluster (each of its NEOs merging a share of the head dimension), one `ttnn.generic_op` (Metal 2.0 ProgramSpec),
thread plan (R, C, W) = (4, 4, 2) from the one host constant `THREAD_PLAN`. SUPPORTED = PHASE0.

Verified (ttsim, Quasar): acceptance test 7/7; golden suite 33 passed, 0 failed, 64 skipped (INVALID), the rest
refused (xfail); debug matrix (`test_..._debug.py`) 16/16 over plans (1,1,1), (2,2,1), (2,2,2), (4,4,2), one and
several clusters, workers with no unit; thread test (`test_..._threads.py`): every worker slot written with that
worker's own row max at (4,4,2) on one cluster, with empty workers, at (1,1,1), and on 8 clusters × 4 NEOs = 32
workers (LLAMA_CAPTURED at position 4095); the op after another program and before `ttnn.to_layout` in one session.

### Deviations from the design, each forced by Quasar (file:line evidence in the kernel headers)

1. **Reader rings are all explicit-sync** (design: implicit for q/K/V/partials). Implicit sync's DM0 ISR credits a
   ring in batches that take entries from every producer hart (tt_metal/impl/dataflow_buffer/DataflowBuffer.md,
   "How the ISR threshold is programmed"); with R = C each hart feeds one worker's range, the streams differ in
   length, and a hart that finished its range held back the other NEOs' last tiles until kernel exit while it
   waited on the combine (hang whenever per-NEO unit counts differ). Reads now go in blocks with one barrier each.
   The design's mechanism-cap row "R = C ... unequal ranges need no padding on reader rings" is wrong for implicit
   sync (op_design.md, Mechanism caps, "Reader streams differ per NEO").
2. **8 compute-owned rings, not 10**: a ring the compute kernel both produces and consumes costs two of a NEO's 16
   Tensix-only tile counters (dataflow_buffer_config.h:42-47; host TT_FATAL dataflow_buffer.cpp:1212). `max_step` +
   `alpha` became `col_step` and `sum_step` + `out_step` became `step`, shared by role (l1_ledger.md).
3. **Neutral state instead of a first-chunk special case**: every worker starts from m = -1e30, l = 0, O = 0 and
   merges each chunk (the neutral partial of an empty worker is that state, emitted as is). Same algorithm and step
   order per chunk; one code path (the compute binary has to fit 24 KiB; it is also built at -Os).
4. **Both DEST banks primed with a datacopy at kernel start**: the first op on a bank (the fills) is SFPU-only
   (DEST format is taken from the last FPU write).
5. **ALU-format latch dropped before the Float32 eltwise multiplies**.
6. **Semaphore: cumulative `wait_min` + `set(0)` at the end, one semaphore per combiner reader hart** (design: one
   semaphore, `down(W_active)` per hart). A single
   shared semaphore with `down(W_active)` per hart lets a hart pass after a quarter of the partials.
7. **Partial piece order [m, O share, l]** on the combiner's reader ring (design: [m, l, O share]) so the β product
   is one helper call over `d_share_tiles + 1` tiles.
8. **`MulUnary` chain element declared in the kernel** (`ScaleByScalar`): kernel_lib's scalar.hpp is not used.
9. **`pv_subblock_tiles`** is the largest divisor of `tensor_d_tiles` within DEST (a host CT arg; the compute kernel
   static_asserts it against `DEST_AUTO_LIMIT`), so V blocks never wrap their ring.

### Perf lamps left as built (structural, ttsim has no timing): writer barriers once per tile (partials: fixed by the verifier, one barrier per NEO partial); reader once per block;
the q read is per worker (R-q-reuse deferred); k_chunk_tiles = whole per-worker assignment at N_max.
