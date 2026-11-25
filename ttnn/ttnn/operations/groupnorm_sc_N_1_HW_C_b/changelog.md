# Changelog: groupnorm_sc_N_1_HW_C_b

## Phase 0 — Core Implementation
- **Date**: 2026-11-25
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier).
  - The op is one `ttnn.generic_op` dispatch with an `n × xs × ht` multi-core split.
  - Statistics are one pass of Σx and Σx² per channel, using fp32 `UnpackToDestFp32` accumulators.
  - These are collapsed to group lanes by a one-hot map matmul, gathered to a per-unit root, finalized, and returned to the unit with `ttnn.Mcast`.
  - The normalize step is folded into `y = x·a − b'`.
  - Input chunks are kept resident in L1 as far as the budget allows (`dram_resident` / `dram_partial_resident`).
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16]
  - layout = [TILE, ROW_MAJOR] (RM is tilized inside the kernel; the output is TILE)
  - alignment = [tile_aligned]
  - affine = [gamma_beta, gamma_only, no_affine]
  - affine_dtype = [bfloat16, "none"]
  - affine_layout = [ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - shard_fit = ["none"]
  - in_place = [False]
  - Partial channels per group (`(C/G) % 32 ≠ 0`) are supported, with no gate.
  - Compute config is fixed: HiFi3 + fp32 DEST.
- **Accuracy achieved** (`test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py`, 5 zero-mean shapes from (1,1,32,32) to (1,1,65536,128)):
  - PCC = 0.999996
  - max_abs_err ≤ 0.094
  - relative RMS ≈ 0.0030–0.0032
  - got/true ratio median 1.0000 (p5/p95 ≈ 0.993/1.006)
  - Offset-heavy data (per-group mean 10, std 1) gives rel-RMS 0.027. That is one-pass variance cancellation; it is a strict xfail and assigned to Refinement 1.
- **Golden suite at Phase 0**: 399 `supported_pass`; 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`. That is 434 of 47 152 collected cells passed; the rest are 4 365 `xfail_expected` and 42 345 `invalid_skipped` (`verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - `_shard_placement` called the non-existent `Tensor.shard_spec()`. Every BLOCK_SHARDED cell crashed in `validate()` with `AttributeError`, giving 300 `xfail_wrong_mode`. It now reads `memory_config().shard_spec`.
  - The L1 budget was taken from `ttnn.get_max_worker_l1_unreserved_size()`. That value includes the 70 688 B kernel-config ring, so 6 DRAM perf cases ran out of L1 (9–54 KB over). It is replaced by `l1_cb_budget(device)`: allocator-base-relative, 1 460 288 B on BH, and capped at the lowest live L1 buffer.
  - The alias group B allocation expression was duplicated in three places. It is now the single source `group_b_bytes(T)`.
  - The ledger and design docs were updated to match.
- **Tests added**:
  - `test_groupnorm_sc_N_1_HW_C_b.py` (planner / implementer)
  - `test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py` (verifier)
  - `_op_bench.py`: device-time bench, run with `scripts/run_safe_pytest.sh --profile --no-py-zones`; not collected by default.

### Device kernel duration, PERF FOCUS cases (Blackhole)

Measurements are DEVICE KERNEL DURATION [ns], median of the trial loop, bf16 with γ+β.
- **"Before"** is the implementer's build with the planner's L1 budget. It ran out of L1 at program creation on 6 of 9 cases, so it has no time for them.
- **"Phase 0"** is after the verifier's budget fix.
- **Native** is `MEASURED_NS_BLACKHOLE`, the existing `ttnn.group_norm` with the model's args.

| Case | Before (ns) | Phase 0 (ns) | Native (ns) | Phase 0 vs native |
|---|---|---|---|---|
| vae_dram_1048576x128 | OOM | 1 972 728 | 9 425 472 | 4.78× faster |
| vae_dram_65536x512 | = Phase 0 † | 436 297 | 1 007 233 | 2.31× faster |
| vae_dram_262144x256 | OOM | 914 726 | 2 860 516 | 3.13× faster |
| vae_dram_262144x512 | OOM | 2 006 876 | 3 924 191 | 1.96× faster |
| vae_dram_1048576x256 | OOM | 4 213 458 | 11 404 370 | 2.71× faster |
| refiner_dram_16384x1152 | = Phase 0 † | 242 230 | 2 261 146 | 9.33× faster |
| flux1_dram_16384x256 | = Phase 0 † | 61 871 | 330 455 | 5.34× faster |
| flux1_dram_1048576x64 | OOM | 863 864 | 21 712 690 | 25.1× faster |
| mochi_dram_8x407040x128 | OOM | 6 382 888 | 117 736 535 | 18.4× faster |

† These three cases are `dram_resident` (`K = B`), so they already fit before the fix. With the old and new budgets, their plans differ only in the unused `K_max` headroom (14/15, 18/19 and 17/18 vs `B_max` = 10, 7 and 2), as compared by `probes/probe_008.py`. The CBs, RT args and program are identical, so the "before" time is the Phase 0 time; it was not separately re-timed. No perf win is claimed for them.

The 12 block-sharded PERF FOCUS cases (`unet_rm_ip_*`, `refiner_rm_ip_*`, `unet_tile_*`, `vae_tile_16384x512`) are outside the Phase 0 SUPPORTED set. They xfail until Refinements 1–2.
