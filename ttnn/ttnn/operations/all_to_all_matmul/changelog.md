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
