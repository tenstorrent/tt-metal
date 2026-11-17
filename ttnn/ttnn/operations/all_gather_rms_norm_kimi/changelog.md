# Changelog: all_gather_rms_norm_kimi

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R1: a Fabric ring all-gather that lands remote slices directly in compute-core L1, with fp32 per-row partials sent ahead of the bulk.
  - `S = rstd ⊗ γ` is precomputed before the data lands, then each landed chunk is normalized.
  - Everything runs in one `ttnn.generic_op` (one program per chip).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned], rank=[4], gamma_dtype=[bfloat16], gamma_layout=[ROW_MAJOR], memory_layout=[INTERLEAVED], cluster_axis=[1], topology=[Ring], num_links=[2]. This equals TARGET (single-cell suite).
- **Accuracy achieved**: worst device and shape over 4 shapes (`test_all_gather_rms_norm_kimi_precision_baseline.py`):
  - PCC = 0.999998
  - max_abs_err = 0.029
  - mean_abs_err = 0.0013
  - rel_rms_err = 0.0025
  - max ULP = 1.19 (bf16), mean ULP = 0.30
  - median got/true ratio = 0.9989 (tf32 truncation of S; not a scale bug)
- **Golden suite at Phase 0**: 7 / 7 (1 `supported_pass` + 6 regression checks; 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`, per `verifier_report.json`)
- **Perf at Phase 0 (harness, PERF FOCUS cell)**: roofline 7.6 µs, target 10.1 µs.
  - Fastest chip: 29.92 µs as handed over, 29.82 µs after the verifier fixes.
  - Slowest chip: 32.71 µs as handed over, 32.29 µs after the verifier fixes.
  - Not dispatch-bound.
  - Per-chip ablation breakdown: skeleton 8.8 / local-only 24.0 / transport-only 26.3 µs (`verification_report.md`).
- **Issues encountered / verifier fixes**:
  - Built the three CB aliases the L1 ledger planned (`cb_gamma`/`cb_out`, `cb_xsq_sum`/`cb_rstd`, `cb_partial`/`cb_rstd_full`); compute-core L1 dropped from 164 to 132 KiB.
  - The writer now flushes each chunk instead of a full barrier per chunk, with one barrier at the end. Measured win: −0.10 µs fastest chip, −0.42 µs slowest.
  - DRY:
    - the compute kernel's CB indices now come from the host;
    - `FP32_TILE_BYTES` / `STATS_ROW_BYTES` are defined once (kernels and host);
    - the γ stick read length comes from the `gamma_page` CT arg.
  - Ledger implementation notes rewritten to match the code.
  - Known infra artifact: running two test modules that each open the fabric mesh in one pytest session can fail the second one at mesh open. Run each file on its own.
- **Tests added**: `test_all_gather_rms_norm_kimi.py` (planner; acceptance), `test_all_gather_rms_norm_kimi_precision_baseline.py` (verifier).
- **Perf tooling added**: `perf/ablation_hooks.patch`, `perf/perf_ablate_nocheck.py.txt`, `perf/ablate_*.json`, `perf/verifier_baseline.json`, `perf/verifier_final.json`.

## Refinement 1 — TRANSPORT: bring the bulk ring exchange to the example's measured transport
- Date: 2026-11-17
- What was done:
  - **Port placement (TRANSPORT)**, taken from `examples/fabric_all_gather` (`_worker_nearest_noc1`, report.md):
    - each port sits at the logical row-0 worker with the fewest NoC1 hops to its link's Ethernet core;
    - the Ethernet core is resolved on the host with no device dispatch. The first fabric connection RT arg is the router's Ethernet channel (a throwaway `ProgramDescriptor` absorbs its semaphores), mapped through the example's `_chip_maps` (cluster + SoC descriptor);
    - the receivers and the hub take the row-0 cores left;
    - the placement is now per chip (`Row0Placement`), so a sender addresses the peer chip's ready port, receiver and hub through the peer's placement;
    - the stats scratch and the semaphores cover all of logical row 0;
    - `PORT_PLACEMENT` knob: `"centred"` reproduces Phase 0 exactly, and it is the automatic fallback when the host lookup fails.
  - **`PORT_BATCH_CHUNKS` 4 → 2** after a co-tune sweep: batch 1/2/3/4 × `INC_EVERY` 2/4/8/16. `INC_EVERY` stays 8.
  - `perf/ablation_hooks.patch` regenerated against the new descriptor. Added `perf/perf_tall.py.txt` (ccl_perf driver for the tall regime shape) and `perf/r1_*.json` (final + every ablation).
- Perf (harness, PERF FOCUS cell, 5 experiments):
  - fastest chip 29.83 → **24.67 µs** (placement −4.05, batch −1.12);
  - slowest chip 32.32 → 27.77 µs;
  - not dispatch-bound.
- Tall `(1,1,2048,384)`: fastest chip 80.4 → 58.0 µs.
- Transport-only (`nocompute,nostore`): 26.32 → 19.57 µs.
  - The bulk now adds 11.4 µs over the skeleton (was 17.5).
  - That is 360 KiB per link direction at 32.4 GB/s, 67% of the 48.5 GB/s link (was 43%).
- Breakdown now (fastest chip, µs):

  | Variant | µs |
  |---|---|
  | skeleton | 8.19 |
  | +store | 11.79 |
  | +compute | 17.92 |
  | local-only (`nobulk`) | 23.71 |
  | transport-only | 19.57 |
  | full | 24.70 |

  **Local-only now binds.** Refinement 3 is the larger remaining term, ahead of Refinement 2's tail.
- Accuracy achieved: PCC = 0.999998, relative RMS ≤ 0.00246 on shapes [(1,1,32,32), (1,1,128,256), (1,1,640,384), (1,1,2048,384)] (unchanged).
- Golden test progress: 7/7 (test_golden 1 + test_regression 6); acceptance 8/8; precision baseline 4/4.
- Issues encountered:
  - The ablation patch no longer applied after the placement change; it is regenerated.
  - Not done, recorded as findings:
    - 7-tile packets: a 12-tile slice block still needs 2 packets, so this needs a `block_h`/`block_w` restructure (Refinement 3's knobs).
    - stats on link 1: Refinement 2's overlap item.
- Tests added: none (the acceptance + determinism tests cover the changed path). Probe `probes/probe_001.py` printed the per-chip Ethernet channels and the nearest workers.
