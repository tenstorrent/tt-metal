# Changelog: wq_b_linear

## Phase 0 — Core Implementation
- **Date**: 2026-11-16
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1 `n_split_x_bcast_resident`: n split over the full grid (110 cores on p100a), x read once and multicast to every core (8 rotating injectors, Flag + handshake), W streamed per core whole-K resident, one `matmul_block` per core with packer L1 accumulation, y written to L1 interleaved. One `ttnn.generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[False] (== TARGET; one cell)
- **Accuracy achieved**: PCC=0.999968, max_abs_err=0.045, mean_abs_err=0.0051, rel_rms_err=0.0083, ULP p99≈85 bf16 ULP (standard distribution; 8 distribution/seed cases via test_wq_b_linear_precision_baseline.py, all PCC ≥ 0.999968)
- **Performance**: 26.6–27.0 µs device kernel time vs 16.0 µs SOL (1.66–1.69×). Ablation: the x broadcast binds (25.4 µs alone; one-page sync floor 14.7 µs); the W stream alone takes 20.8 µs.
- **Golden suite at Phase 0**: 2 / 2 registry cells passing + 1/1 regression (per `verifier_report.json`); 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Issues encountered**: verifier fixes: removed the unused `x_cb_k_blocks` / `w_cb_k_blocks` host aliases (kernels assume whole-K residency); recorded the as-built knob values in `op_design.md`; updated `l1_ledger.md` currency and the R5 deferral note (measurement falsifies "cross-core tier does not bind").
- **Tests added**: test_wq_b_linear.py (planner), test_wq_b_linear_precision_baseline.py, _op_bench.py (perf ablation bench, not collected by default)

## Refinement 1 — Take the x broadcast off the critical path
- **Date**: 2026-11-16
- **What was done**: x broadcast restructured with the multicast helper, writer kernel + descriptor only. Each round has its own fixed-sender Flag channel (`x_mcast_<r>`, one semaphore each, fired once per launch), so no ready handshake. The old rotating Flag+handshake channel remains selectable via `X_MCAST_CHANNELS="rotating"`. New knobs, single-source in the descriptor: `X_ROUND_K_BLOCKS` sets round size in compute K-blocks, decoupled from `K_BLOCK_TILES`; the writer still pushes per K-block. `X_ROUNDS_CONCURRENT` is parked at `False`. `X_INJECTOR_PLACEMENT` places injectors on a diagonal over the last rows. `MAX_X_MCAST_CHANNELS = 8` matches the channels declared in the writer. `K_BLOCK_TILES` re-tuned 6 → 3, giving 4 rounds of 4 K-blocks.
- **Performance**: 26.53 → 23.85 µs device kernel time (median of 9; SOL 16.0 µs, 1.66× → 1.49×).

  | Variant | Phase 0 (µs) | Refinement 1 (µs) |
  |---|---|---|
  | full | 26.5 | 23.85 |
  | x only (`no_compute_no_w_read`) | 25.4 | 16.0 |
  | `w_only` | 20.8 | 21.0 |
  | `w_compute` (new) | — | 22.95 |
  | `mcast_only` (new) | — | 13.6 |
  | `sync_only` | 14.7 | 8.4 |

  The binding term is now the W stream plus the compute tail on the 18 two-column cores: NCRISC ends at 21.4 µs, TRISC2 at 22.7 µs.
- **Accuracy achieved**: PCC=0.999972, relative RMS=0.0077 on [1,1,160,1536] x [1536,4096] (8 precision-baseline cases, all PCC ≥ 0.999972; Phase 0: 0.999968 / 0.0083)
- **Golden test progress**: 3/3 passing (2 registry cells + regression bit-identity)
- **Issues encountered**: Concurrent injectors are correct but slow and erratic (39–88 µs), with both Counter/unlinked and Flag/linked multicast. They are parked behind `X_ROUNDS_CONCURRENT=False`. The NoC0 prefetch lever was not built: in dedicated-NoC mode BRISC shares read cmd buffer 1 with NCRISC on NoC0, so it needs a reader-side prefetch plus a handoff. Injector placement already removed most of the prefetch contention.
- **Tests added**: `test_wq_b_linear_x_mcast_knobs.py` (7 knob settings: correctness + bit-identity). `_op_bench.py` gains `WQB_KNOBS` overrides and the `mcast_only` / `w_compute` ablations.
