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

## Refinement 2 — W DRAM stream co-tune toward the DRAM SOL
- Date: 2026-11-16
- What was done: Host n split rewritten around three new single-source knobs in the program descriptor. `CORE_N_TILES = 2` gives an even split: 65 active cores instead of 18 × 2 + 92 × 1. `W_BANK_COALESCE = True`: each run of 7 (DRAM bank count) two-column cores owns columns `(n, n + 7)`, so a core's W tiles of one k row sit adjacent in one bank and arrive in one 2176 B read. The 2 unpairable columns go to single-column cores. `CORE_PLACEMENT` (row_major / spread) is added. The reader becomes one kernel per core group so read depth can differ per group (`W_READS_IN_FLIGHT_G1`). A NoC1 share of the W stream is built and parked at 0 (`W_NOC1_K_BLOCKS`, `W_NOC1_AFTER_X`): the writer reads the last K-blocks on NoC1 into their slots under its own trid and raises a local handoff semaphore, and the reader pushes them, so it stays the CB's only producer. The W block-read loop moved to `kernels/wq_b_linear_w_read.hpp`, shared by reader and writer. The writer writes y at `n_start + j * n_stride`. The handoff semaphore takes id 0; the x channels start at id 1.
- Performance (p100a, device kernel ns, `_op_bench.py --profile`):

  | Variant | Refinement 1 (µs) | Refinement 2 (µs) |
  |---|---|---|
  | full (median of 9) | 23.85 | 22.39 |
  | `w_only` | 21.0 | 19.8 |
  | `w_compute` | 22.95 | 21.64 |
  | x only (`no_compute_no_w_read`) | 16.0 | 16.15 |
  | `no_y_write` (new) | — | 21.69 |

  W-reader (NCRISC end) max/p50: 1.045 in the full op, 1.028 in `w_only`. Re-measured at the Refinement 1 defaults: 1.087 (`w_only`); the verifier's 1.25 was taken at kb=6.
- Findings: the W stream is not imbalance-bound. Every split, depth, placement and transfer size measured lands at 19–21 µs alone, from 32 to 110 cores, at 1–4 KB reads and depths 1–4. That is ≈343 GB/s from reader start, 77% of the 448 GB/s peak. The cap is NoC0 link load next to the DRAM endpoints, not the DRAM banks: the same stream on NoC1 takes 36 µs, and moving 6 of 16 K-blocks onto NoC1 cuts W-only to 17.9 µs. In the full op that NoC1 share loses (24–31 µs issued at start, 22.7–25.2 µs after the x phase), because the x broadcast holds BRISC and NoC1 until about 13–15 µs. Remaining tail after the last W byte: last-K-block compute + pack ≈ 1.3 µs and the y store ≈ 0.8 µs.
- Accuracy achieved: PCC=0.999972, relative RMS=0.0077 on [1,1,160,1536] x [1536,4096] (per-core accumulation order unchanged; 8 precision-baseline cases green)
- Golden test progress: 3/3 passing (2 registry cells + regression bit-identity)
- Issues encountered: the W reader on NoC1 for the whole stream (both data-movement RISCs on NoC1) hung once at 110 cores. Only the experiment knob was affected; it was removed, and the shipped NoC1 share runs on BRISC, which already owns NoC1.
- Tests added: `test_wq_b_linear_w_stream_knobs.py` (12 knob settings: split, placement, coalescing, per-group depth, NoC1 share with both x channel modes; correctness + bit-identity). `_op_bench.py`: `no_y_write` ablation; W gating now patches the shared read header, so the W ablations also cover the writer's NoC1 share.

## Refinement 3 — Balanced W split: n + K-remainder with deterministic combine (design R4)
- Date: 2026-11-16
- What was done: Built regime R4 `n_split_k_remainder_combine` in the shared kernels and descriptor, behind `K_REMAINDER_SPLIT` (parked at `False`, so the Refinement 2 split stays the default). With it on, 110 cores each own one full column and the 18 leftover columns are cut into 6 K-slices of 2–3 K-blocks over 108 cores. The reader reads a core's slice into `cb_w_slice` first. x reaches the slice through `cb_x_slice`, an alias index on `cb_x_resident`'s allocation that the writer pushes alongside it. The slice is a second `matmul_block` call into `cb_partial`, nested in the main call's `PostKBlockFn` at a host-chosen K-block (`SLICE_EARLY_K_BLOCKS`). Slice cores unicast their partial into the owner's grid-wide `cb_landing` and bump `LANDING_SEMAPHORE_ID`. The owner (slice 0) combines in the same functor at `COMBINE_EARLY_K_BLOCKS`: an `eltwise_chain` copy of its own partial, then L1-accumulating copies (`L1Accumulation::AddToExisting`) of the landed partials in slice order. Reused: x broadcast, W read helper, main matmul call, y store. Added: slice/combine stages, 3 CBs + 1 alias index, 1 semaphore (x channels now start at semaphore id 2).
- Performance (p100a, `_op_bench.py --profile`, median of 5 calls per run):

  | Variant | µs |
  |---|---|
  | Refinement 2 split (default), 3 runs | 22.40 / 22.60 / 22.32 |
  | R4, slice 9 / combine 6, 3 runs | 22.70 / 22.78 / 22.34 |
  | R4, slice at its K end, combine after the main loop | 24.0–24.2 |
  | R4 `w_only` / `w_compute` / x only (first placement) | 23.25 / 23.58 / 17.61 |

  R4 does what it was meant to: busiest-core W 104,448 → 62,016 B, compute 480 → 285 tile-matmuls, and the gap from W end to compute end goes ≈ 1.35 → ≈ 0.6 µs. But the W stream itself is slower: 110 cores of single-tile 1088 B reads end at NCRISC p50 20.2–20.6 µs, against 19.3 µs for 65 cores of bank-coalesced 2176 B reads. Placement matters: with the slice computed at its K end and the combine after the main loop, the owners trail by 2 µs (waiting for late partials, then 30 single-tile combine passes).
- Bottleneck: unchanged from Refinement 2. W transfer efficiency at the NoC0 links next to DRAM binds; per-core imbalance does not (Refinement 2's W-reader max/p50 is already 1.045). Next: coalesced R4 (core pairs share a bank-strided column pair and split its K, so reads stay 2176 B while compute halves), or the y store overlap (≈ 0.9 µs after compute on every core in both regimes).
- Accuracy achieved: default path unchanged, PCC=0.999972, relative RMS=0.0077. R4: PCC=0.999973, relative RMS=0.0076, leftover-column max abs error 0.029 (main columns 0.047), on [1,1,160,1536] x [1536,4096] seed 0.
- Golden test progress: 3/3 passing (2 registry cells + regression bit-identity)
- Issues encountered: `matmul_block` returns with packer L1 accumulation enabled. The first combine pass therefore accumulated onto stale L1 (Inf / p0+p1 on the leftover columns); `pack_reconfig_l1_acc(0)` before the combine chain fixes it. Found with DEVICE_PRINT: the sender partial and the landed slot matched the reference slice exactly.
- Tests added: `test_wq_b_linear_w_stream_knobs.py` gains 4 R4 cases (default positions, slice + combine last, slices first, `K_BLOCK_TILES=4`), each checking correctness and bit-identity across calls. Probes 004–014 under `probes/`.
