# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - It is a fused CCL op with one `ttnn.generic_op` dispatch per call.
  - Local half: ported from the base op `rms_norm`, with regimes R1 / R3-i / R1-HS / R3-s re-run on the grid left after
    the 2L port cores.
  - Cross-device half: a relay all-gather of one fp32 Σx² record (128 B) per tile-row, in waves, driven by port cores
    adapted from `examples/fabric_ring_exchange`, and overlapped with the local work by a non-blocking root event loop.
- **SUPPORTED at Phase 0** (equal to TARGET on every axis):
  - dtype = [float32, bfloat16, bfloat8_b]; fp32_dest_acc_en = [True, False]; layout = [TILE]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]; rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [float32, bfloat16, none]; gamma_layout = [TILE, ROW_MAJOR, none]
  - memory_layout = [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED]
  - cluster_axis = [0, 1]; topology = [Linear, Ring]; num_links = [1, 2, None]
  - EXCLUSIONS = [{float32, fp32_dest_acc_en=False}]
- **Accuracy achieved** (worst device, G = 4 Ring on the 1×4 QuietBox 2 ring, bf16 TILE γ,
  `test_distributed_rms_norm_precision_baseline.py`, 4 shapes × 3 dtypes × 2 DEST modes):
  - bf16, fp32 DEST: PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019, ≤ 1.05 ULP.
  - bf16, 16-bit DEST: PCC ≥ 0.999993, rel_rms ≤ 0.0081, ≤ 5.3 ULP, got/true ratio +0.3 to +0.7 %.
  - fp32: PCC 1.000000, rel_rms ≤ 0.0012.
  - bf8b: PCC ≥ 0.999949, rel_rms ≤ 0.014.
- **Golden suite at Phase 0** (after the verifier fixes; `verifier_report.json`):
  - supported_pass 10,908; supported_fail 133; xpass_drift 0; xfail_wrong_mode 1 (harness interaction);
    xfail_expected 2,160; invalid_skipped 33,696; infeasible_skipped 40.
  - The 133 failing cells are tracked by Refinements 1 (120), 2 (12) and 4 (1).
  - First run on the code as handed over: supported_pass 6,197; supported_fail 3,704; xpass_drift 35;
    xfail_wrong_mode 226.
- **Perf at Phase 0** (`perf_measure.py`, 5 experiments, fastest / slowest chip, µs):
  - 640×1792 (PERF FOCUS) 23.93 / 25.77 (target 11.9);
  - 32×1792 12.03 / 13.61 (target 0.6);
  - 2048×1280 44.10 / 47.41 (target 27.3);
  - exposed transport (op − `DRMS_ABLATE_EXCHANGE`) 2.94 / 1.60 / 10.00.
  - The verifier's fixes are perf-neutral (`ccl_perf compare`: null on all three cases).
- **Issues encountered (verifier fixes)**:
  1. **Kernel bug: ragged block × multiple W chunks × γ.**
     - Cause: `cb_x_gamma` (an `R·C` ring) advanced only `R_b·C` per chunk on cores with fewer rows than `R`. The
       next chunk's indexed reads then ran past the ring end.
     - Effect: an LLK "Indexed tile read exceeds CB boundary" assert and hang under `--dev`, garbage output (PCC ≈ 0)
       otherwise. Exposed by the translated DeepSeek prefill 8192.
     - Fix: nominal push count (filler pages) per chunk on ragged blocks.
     - Test: `test_distributed_rms_norm_debug.py::test_ragged_block_chunked_gamma`, verified failing without the fix.
  2. **Persistent exchange resources pinned L1.**
     - Cause: one landing + semaphore set was cached per port placement (sharded inputs move the ports), each created
       mid-call below the caller's L1 tensors and never freed. Over a long session the CB space ratcheted down.
     - Effect: ~340 golden sharded cells were refused late in the run while passing in a fresh process.
     - Fix: the landing is now allocated per call (an op intermediate) and freed after the dispatch; the global
       semaphores are one grow-only set per mesh over the whole grid. Device time unchanged.
  3. **Long sequences crashed.**
     - Cause / effect: `CBs collide with the landing scratch` assert, or an allocator OOM.
     - Fix: the plan is re-solved against the landing's real bytes, and the design's missing port-landing cap
       (`PORT_LANDING_MAX_BYTES`) is now a clean `NotImplementedError` before allocation. Prefill 32768 now passes;
       131072 is refused (Refinement 4).
  4. **EXCLUSION `{bfloat8_b, w_non_aligned}` removed.** It was carried from the base op. The compute-side W-pad
     mask (partial reduce scaler) now also runs for interleaved Bfp8_b, and 720 golden cells pass.
  5. **`num_links=None` declared in SUPPORTED.** It was always accepted; the registry cell of a call records the
     literal `None`.
  6. **DRY.** `cb_inventory()` is the one CB list behind both the L1 solve and the CB descriptors.
  7. **Docs.** L1 ledger made current (stale `cb_dev_partial` / `Rw` / total rows); `op_design.md` implementation
     notes extended.
- **Tests added**:
  - `test_distributed_rms_norm_precision_baseline.py`;
  - `test_distributed_rms_norm_debug.py::test_ragged_block_chunked_gamma`;
  - existing: `test_distributed_rms_norm.py` (acceptance), `test_distributed_rms_norm_knobs.py`,
    `test_distributed_rms_norm_debug.py::test_repeat_rows`, `perf_ablate.py`.
