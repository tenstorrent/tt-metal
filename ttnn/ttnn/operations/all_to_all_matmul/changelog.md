# Changelog: all_to_all_matmul

## Phase 0 — Core Implementation
- **Date**: 2026-11-24
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1: heads are split across the compute grid (grid minus every chip's 2·num_links port cores). Remote row quarters go over Fabric 2D straight into the consuming core's L1 (zero-copy landing scratch, one arrival counter per head slot, per-port ready fence). Weights are read from DRAM while the rows are in flight. `matmul_block` per head with A retained across n-blocks. Transport T1: the far quarter goes direct on the routing table's 2-hop route. One `generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat16], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned], output_buffer=[DRAM], memory_layout=[INTERLEAVED], cluster_axis=[1], topology=[Ring], num_links=[2]
- **Accuracy achieved**: PCC=0.9999986, max_abs_err=0.0163, mean_abs_err=0.00117, rel_rms_err=0.00169, ULP p99 ≤ 4 (measured on 4 shapes via test_all_to_all_matmul_precision_baseline.py). The exchange is bit-exact.
- **Golden suite at Phase 0**: 16 / 16 supported cells passing; 132 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`). Regression suite 7 / 8 pass. `test_small_packet_size` cannot open a 4160 B fabric on this board (ERISC kernel config buffer overflow in `open_mesh_device`); this is an environment issue.
- **Perf (Phase 0 loose case, Kimi, 7232 B, harness)**: fastest chip 94.4 us, slowest 104.2 us, not dispatch-bound (roofline 63.0, target 84.0). Exchange + fence alone 49.3 us.
- **Issues encountered (verifier fixes)**:
  - `cb_weight` was allocated `WEIGHT_DEPTH` blocks on cores that process one block per call. It is now `min(WEIGHT_DEPTH, heads_per_core_max * num_n_blocks)`. Perf is neutral (within noise); L1 peak fell 368,672 → 270,368 B (Kimi) and 458,752 → 262,144 B (GLM).
  - `validate()` raised two caller errors (weight layout, input buffer) after the support refusal; they now run before it.
  - The acceptance-test modules each imported the golden mesh fixtures, which created a second session cache when the directory ran. The fixtures now come from the directory `conftest.py`.
  - Stale NoC note in the program-descriptor docstring corrected.
- **Tests added**: test_all_to_all_matmul.py (planner), test_all_to_all_matmul_precision_baseline.py (verifier)

## Refinement 1 — Model configuration: 16-bit DEST, bfloat8_b weight, L1 output
- Date: 2026-11-24
- What was done: `SUPPORTED["fp32_dest_acc_en"]` += False, `SUPPORTED["weight_dtype"]` += bfloat8_b, `SUPPORTED["output_buffer"]` += L1. No kernel change was needed: `cb_weight`'s format follows `weight.dtype` (Bfp8_b pages, 1088 B), `block_n` derives from the byte knob `WEIGHT_BLOCK_BYTES`, the host subblock rule picks the 8-tile limit under 16-bit DEST, `matmul_block` reconfigures the unpacker between the bf16 A and bfp8 weight, and the writer's `TensorAccessor` follows the L1-interleaved output. Added a live knob `COMPUTE_WRITER_NOC_L1_OUT` (writer NoC for an L1 output), parked at NoC0 = the prior choice: NoC1 measured fastest 81.01 -> 78.53 us but slowest 84.92 -> 86.15 us on the Kimi model case, rejected by `eval.ccl_perf compare` (slowest regressed beyond noise).
- Accuracy achieved (precision baseline, 4 shapes small / medium / Kimi / deep K): fp32 DEST HiFi4 (bf16 or bf8b weight, reference on the quantized weight): PCC 0.9999986, rel RMS 0.00169. 16-bit DEST HiFi2 L1 out: PCC 0.99998, rel RMS 0.0051-0.0062 (K <= 128); deep K 4096: PCC 0.9990, rel RMS 0.070 (gate 0.13) with a median got/true ratio of 1.058 — `ttnn.matmul` with 16-bit DEST on the same data gives 1.0584 / 0.0701 (probes/probe_013.py), so this is the hardware's 16-bit accumulation over 128 K-tiles, not an op bug.
- Golden test progress: tile_aligned 120/120 pass (every DEST x weight dtype x output buffer cell), LOOSE_CASES 4/4 pass; the 24 m_non_aligned cells remain xfail (Refinement 2). 124 supported pass / 24 xfail_expected of 148.
- Perf (harness, 5 experiments, CCL_PERF_CASES=all, none DISPATCH-BOUND): Kimi model (L1 out, HiFi2, 16-bit DEST) fastest 81.01 / slowest 84.92 us; Kimi Phase 0 94.43 / 104.30 us (unchanged); DeepSeek model 99.12 / 104.94 us; GLM model 79.10 / 83.86 us.
- Issues encountered: the deep-K 16-bit-DEST scale drift above (test's ratio check scoped to fp32 DEST or K <= 512, with the probe as evidence). None otherwise.
- Tests added: `test_all_to_all_matmul_precision_baseline.py` gains a CONFIGS axis (phase0, model 16-bit DEST HiFi2 L1, 16-bit DEST HiFi4 L1, bf8b weight fp32 DEST, bf8b weight 16-bit DEST HiFi2 L1) — 20 cases; `probes/probe_013.py` (ttnn.matmul 16-bit DEST deep-K reference).
