# Changelog: wq_b_linear

## Phase 0 — Core Implementation
- **Date**: 2026-11-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One `generic_op` program:
  - x is read once from DRAM by a sender core (NCRISC/NoC0) and multicast to 64 workers (BRISC/NoC1, Counter data-ready, no handshake, x resident).
  - Each worker streams its 2 bank-spread W columns, runs one `matmul_block` over K with packer L1 accumulation, and stores 5×2 output tiles to L1-interleaved y.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[False]. This equals TARGET (one cell).
- **Accuracy achieved** (`test_wq_b_linear_precision_baseline.py`, 4 input profiles on the single shape, golden distribution):
  - PCC=0.999931, max_abs_err=0.090, mean_abs_err=0.0083, rel_rms=0.0139, ULP p99≈105.
  - Global scale 1.0069, from 16-bit DEST accumulation over `block_k_tiles = 8`.
- **Performance**: device span 26.6 µs (p150, 8 banks), 1.9× the 14.0 µs SOL. The op is x-delivery-bound: the single sender reads x in 14.6 µs and finishes multicasting at 22.4 µs.
- **Golden suite at Phase 0**: 2 / 2 registry cells passing, plus regression (bit-identical repeats). supported_fail / xpass_drift / xfail_wrong_mode = 0.
- **Issues encountered** (verifier fixes):
  - The DEST subblock limit and the compute `fp32_dest_acc_en` now derive from the compute config instead of constants.
  - Documented the `block_k_tiles ≤ 16` precision ceiling: PCC falls below 0.9998 at kb ≥ 24, from 16-bit DEST drift.
  - Corrected stale kb-4 figures in `l1_ledger.md` and appended the verifier audit.
- **Tests added**: `test_wq_b_linear_precision_baseline.py`, `_op_bench.py` (device-time bench for perf phases, not collected by default).

## Refinement 1 — Speed up x delivery on the perf-flagged profile
- Date: 2026-11-23
- **Calculator predictions (written before measuring)**:
  - `op_model.py matmul 5 48 128 --designs --c-mem l1 ...`: current (kb 8, 64 workers + sender) 18.75 µs; rank 1 16.53 µs (73 cores, kb 2, linked x mcast, c_noc split); same-grid kb 2 linked 16.60 µs. Model's sender: x read done 6.9 µs, last multicast sent 7.6 µs.
  - `estimate.py` (`dm_estimates.json`, phase "Refinement 1"): x read alone, 1 core, 2 KB pages — barrier per 40-page round (current) **8.37 µs** vs all rounds in flight (trid window) **6.63 µs**. x multicast, 30 × 16 KB unlinked bursts to 64 receivers — write barrier every 5 (current) **15.3 µs** vs every 30 **13.8 µs**; the tool binds on per-burst path reservation (352 cycles) and has no linked option, so the linked candidate is priced only by the op model.
- **What was done** (all levers are live knobs in `wq_b_linear_program_descriptor.py`; the Phase 0 schedule is one knob setting):
  - x-reader: whole-x CB reserved once, `X_READ_TRIDS` rounds of DRAM reads in flight (trid window via `set_read_trid` / `async_read_barrier_with_trid`), retired and pushed in order. Shipped at 2.
  - x multicast payload: each round is one linked burst sequence (`X_MCAST_LINKED`), one path reservation per round instead of per 16 KB burst.
  - x publish: `X_ORDERED_SIGNAL` writes the channel's round count into every receiver's Counter cell with a 4-byte multicast right behind the payload (same path + VC, so it lands after the data). This replaces the helper's write barrier + atomic increment + atomic barrier per round (~1.1 µs each, measured). Workers still use `ReceiverPipe::receive(r)` unchanged. The helper path (`send_signal`, optionally deferred one round by `X_SIGNAL_LAG` with a partial ack wait) stays as the alternative.
  - Multi-injector support: `NUM_X_SENDERS` (1–2) sender cores, each on its own `ttnn.Mcast` channel at `next_semaphore_id()`, owning every other round. The worker writer receives round r on channel r % S. `X_SENDER_NOCS` picks each tree's NoC. Shipped at 1.
  - `W_WAIT_X0`: the worker W stream can wait for x round 0. Shipped off.
  - `BLOCK_K_TILES` 8 → 4.
  - Helper deviation (sender payload + publish) is documented at the top of `wq_b_linear_sender.cpp`.
- **Measurements** (device span from `profile_log_device.csv`, `verify/span.py`, median of 5 dispatches; p150):

  | Variant | Span |
  |---|---|
  | Phase 0 (kb 8), this session | 26.3 µs |
  | + trid window 4, linked, lagged atomic signal (kb 8) | 25.2 µs |
  | + 2 senders, NoC1 + NoC0 (kb 8) | 25.8 µs |
  | 2 senders, both NoC1 | 27.4 µs (trees collide) |
  | 2 senders, kb 4, window 2 | 23.0 µs |
  | 2 senders, kb 2, window 2 | 26.2 µs (counter atomic ~1.8 µs/round binds) |
  | + ordered signal, 2 senders, kb 2 / kb 4 | 23.4 / 23.3 µs |
  | ordered signal, 1 sender, kb 4, window 2 / 4 | 22.6 / 22.05 µs |
  | ordered signal, 1 sender, kb 2, window 4 | 24.0 µs |
  | ordered signal, 1 sender, kb 8, window 2 | 25.2 µs |
  | `W_WAIT_X0` (kb 8, window 4 / 1) | 31.5 / 29.8 µs |
  | **Shipped: 1 sender, kb 4, window 2, linked, ordered signal** | **22.55 µs** (3 runs: 22.55 / 22.79 / 22.42) |

  Shipped is −15% vs 26.6 µs, 1.61× the 14.0 µs SOL. The model's predicted 16.6 µs region was not reached; see Bottleneck now.
- **Parts vs calculator** (ablations; recorded with `estimate.py measure` where an estimate exists):
  - x read alone (W stubbed), Phase 0: 9.6 µs vs estimate 8.37 µs.
  - x multicast alone (all DRAM stubbed), 491 KB to 64 receivers: unlinked 17.7 µs vs estimate 15.3 µs; linked 12.9 µs. `estimate.py` has no linked option.
  - One 64-receiver tree delivers ~33–40 GB/s. That matches the estimator's path-reservation/tree-delivery bound, so one sender cannot deliver x faster than ~12 µs. The op model assumes ~7.6 µs.
  - Two senders reach 6.5 µs per tree with no DRAM traffic. Under load, the NoC0 tree slows the W reads that share NoC0, and two NoC1 trees collide, so one NoC1 sender wins once the per-round cost is gone.
- **Bottleneck now** (zones on worker x-arrival / W-block barriers, since removed; ablation with W reads stubbed = 21.9 µs, the same as with W):
  - The critical path is x round 0 arriving at ~2.5 µs, then the worker matmul at ~15 µs for kb 4 (standalone compute ablation at kb 8: 14.0 µs vs 11.4 µs ideal), then the output store tail of ~2 µs.
  - Later x rounds arrive every ~0.85 µs, ahead of compute (~1.2 µs/block), so x no longer binds after round 0.
  - The span is set by the slowest worker: W bank imbalance (W end median 14.7 µs, max 17.2 µs) leaves it ~3 µs behind the median worker.
- **Accuracy achieved**: precision baseline `randn_s0` PCC=0.999957, relative RMS=0.00986 (was 0.999931 / 0.0139 at kb 8). `wide_range` 0.999957 / 0.00985. `pos_mean` 0.996608 / 0.0829. Shape [1,1,160,1536] × [1536,4096].
- **Golden test progress**: 3/3 (2 registry cells + regression bit-identity). Unit tests 14/14 (plain), 10/10 under `--dev`.
- **Issues encountered**:
  - `Mcast.attach` binds a channel once, so channel s is attached to its sender kernel and the writer in one call (prefix `x_mcast` / `x_mcast_b`), with unused prefixes attached absent.
  - Two NoC1 trees and a NoC0 tree beside the W reads both measured worse than one NoC1 tree.
- **Tests added**: none (`verify/span.py`, `verify/zones.py`: profile readers for the device span and custom zones).

## Refinement 2 — Trim the tail and the W side on the perf-flagged profile
- Date: 2026-11-23
- **Calculator predictions (written before measuring)**, `estimate.py` W stream alone, 64 cores, phase "Refinement 2" in `dm_estimates.json`:
  - Current, bank_spread, 2 × 1,088 B per k-row: 16.9 µs. Coalesced 2,176 B: 16.4 µs. Contiguous map: 15.7 µs. Coalesced + NoC split: 15.2 µs, with the tool's caveat that split reads on part of the grid measured 17–26% slower than estimated. All are within 10%, so they tie. The binding resource is the NoC link out of the DRAM column.
  - y store as a DRAM-write proxy (y is L1, which the tool refuses): NoC1 3.43 µs vs split 3.41 µs, a tie. The op model rates `c_noc` split vs NoC1 at 1% (Refinement 2 entry).
  - So the calculator gave no ranking. Every knob was measured on device.
- **What was done** (all live knobs in `wq_b_linear_program_descriptor.py`; the Refinement 1 schedule is `COLUMN_MAP="bank_spread"`, `W_COALESCE=False`, `OUT_SUBBLOCK_MAJOR=False`, `Y_STORE_NOC_SPLIT=False`, `W_DEPTH_BLOCKS=None`):
  - `W_COALESCE` (reader): one `noc_async_read` of `block_n_tiles · w_tile_bytes` per k-row. The host enables it only when the worker's columns are `c, c + num_banks, …` and `tensor_n_tiles % num_banks == 0`, which places them in consecutive slots of one bank.
  - `W_DEPTH_BLOCKS` 4 (was whole slice resident).
  - `OUT_SUBBLOCK_MAJOR`: compute's `matmul_block` uses `OutputCBLayout::SubblockMajor`, which pushes per 5×1 subblock on the last K-step. The writer waits cumulatively per subblock, stores it, and pops the block after the barrier.
  - `Y_STORE_NOC_SPLIT`: the writer (BRISC/NoC1) issues every other y tile on NoC0, which is idle by then because the reader only does reads on NoC0. It barriers both NoCs.
  - `COLUMN_MAP` knob: `bank_spread` (shipped) or `contiguous`.
  - `_op_bench.py` takes `WQB_KNOBS="NAME=value,..."` overrides and prints PCC.
- **Measurements** (device span, `verify/span.py`, median of 5 dispatches, p150; "slow" = slowest worker's TRISC_2 end):

  | Variant | Span | Note |
  |---|---|---|
  | R1 schedule (2 runs) | 22.36 / 22.70 µs | slow worker compute ends 20.5–20.9, median 18.0–18.5 |
  | + `W_COALESCE` | 21.09 µs | W end max 16.7 → 15.5 µs |
  | + `OUT_SUBBLOCK_MAJOR` alone | 22.11 µs | median store end −0.9 µs; span held by slow worker |
  | + `Y_STORE_NOC_SPLIT` alone | 22.65 µs | median store end −0.7 µs; span held by slow worker |
  | `COLUMN_MAP="contiguous"` | 23.62 µs | W end max 18.6 µs: worse |
  | coalesce + subblock store | 20.64 µs | |
  | coalesce + subblock store + split | 20.69 µs | slow worker still binds |
  | coalesce + subblock + depth 3 / 4 / 6 | 20.27 / 20.24 / 20.50 µs | depth 4: slow worker TRISC_2 18.8 vs median 18.6 |
  | coalesce + depth 4 (no subblock store) | 21.12 µs | |
  | coalesce + subblock + depth 2 + split | 19.51 µs | |
  | **Shipped: coalesce + subblock + depth 4 + split** | **19.48 / 19.42 µs** | store end 0.7 µs after compute |

  - Result: −14% vs R1's 22.55 µs, 1.39× the 14.0 µs SOL.
  - The store levers only show once depth 4 removes the slow-worker outlier. Before that, the wall was the slowest worker's compute, which started late because its early W blocks were served last.
- **Parts vs calculator** (`estimate.py measure`): W stream, slowest worker's last W read in the op, not an isolated ablation. Current 16.67 µs vs 16.9 estimated (+1.5%). Coalesced 15.43 µs vs 16.4 estimated (+6.5%). The order matches the estimate.
- **Bottleneck now**: worker compute, ~16 µs from x round 0 to the last pack, vs 11.4 µs ideal. W reads end at ~14.4 µs median, and the store tail is ~0.7 µs.
- **Accuracy achieved**: output bit-identical to Refinement 1 (bench PCC 0.9999583 for every variant). Precision baseline passes: `randn_s0` PCC=0.999957, relative RMS=0.00986. Shape [1,1,160,1536] × [1536,4096].
- **Golden test progress**: 3/3 (2 registry cells + regression). Unit tests 14/14 plain, 10/10 `test_wq_b_linear.py` under `--dev`.
- **Issues encountered**: the bench's first PCC print shows nan during the precompile pass; the real dispatch prints the valid value.
- **Tests added**: none. `_op_bench.py` gained knob overrides and a PCC print.

## Perf 1
- Date: 2026-11-23. Board: Blackhole p150b, 11×10 workers, 8 DRAM banks (512 GB/s), 1.35 GHz. Whole-op SOL on this board 14.0 µs (the feature spec's `sol_ns_blackhole` 16,018 is for the 7-bank p100a). Focus case = the op's only cell: x [1,1,160,1536] bf16, W [1536,4096] bf8_b, y L1, HiFi2, fp32_dest_acc_en=False. SUPPORTED unchanged.
- **Instrumentation (permanent)**: `MaybeDeviceZoneScope` zones in every kernel — x-reader `x_read_issue/x_read_drain`; sender `x_mcast_wait/x_mcast_issue/x_mcast_barrier`; worker reader `w_reserve/w_issue/w_barrier` per K-step; writer `x_wait_r0/x_wait_rest/y_wait/y_issue/y_barrier`; compute `compute_wait_x0` (hoisted wait for x round 0 + W block 0) and `compute_matmul`. Max 146 markers per RISC (cap 250). The kernel_lib header `perf_instrumentation.hpp` did not exist in this tree; added a minimal one (`DeviceZoneScopedN` alias). Ablation knob `ABLATE` in the descriptor (`"w_read+compute"` form, maps to `ABLATE_*` defines and the helper's `SKIP_COMPUTE`); `verify/stages.py` summarises zones.

### Measured breakdown (before; device span, median of 5 dispatches)
| Run (cumulative peel) | Span ns | Note |
|---|---|---|
| full | 19,727 | x round 0 lands 2.77 µs; compute 2.8 → 18.3 µs; y done 19.4 µs |
| compute stubbed | 17,114 | DM alone: W end max 15.8 µs, x delivered 13.1 µs |
| + W stubbed | 14,316 | |
| + x multicast stubbed | 10,827 | |
| + x read, y store stubbed (floor; unpack/pack/L1-acc still run) | 6,047 | |
| DM stubbed (compute alone) | 16,587 | matmul zone 15.3 µs |

| Part | Alone ns | Part SOL ns | Measured / SOL |
|---|---|---|---|
| compute (480 tile-matmuls/core, 64 cores, 32 cyc HiFi2) | 15,339 (zone) | 11,378 | 1.35 |
| W read (6.68 MB DRAM) | 14,251 (last W read) | 13,060 | 1.09 |
| x read + multicast (one 64-receiver tree) | 12,599 (delivered) | 5,690 (sender egress) | 2.2 — at the schedule estimate (13,835) |
| x read (1 core) | 8,638 | 5,690 (one core's NoC ingress) | 1.5 |

- Overlap loss = 19,727 − 17,114 = 2.6 µs (fill: x round 0 at 2.77 µs; tail ~1.1 µs).
- **Ranked bottleneck**: compute (3.9 µs above its SOL, balanced with DM), then fill/tail, then x delivery pacing; W near SOL.
- Paper search (`/perf-calculator`): x multicast sweep `be65a47c` (phase "Perf 1"): the current schedule is rank 1 (13.84 µs); nothing beats it by more than the 10% band. Two disjoint half-rect trees (`b23920c8`) 12.21 µs: a tie, not fanned out. Op model `--designs`: current 65 cores kb 4 17.39 µs vs best 73 cores 17.22 µs (tie); ksplit 2 on 110 cores 19.04 µs. No split beats the current one by 15%, so no split idea was fanned out. The op model's compute bound (12.1 µs) is below the measured 15.3 µs.
- `estimate.py report`, Perf 1: sweep be65a47c, 25 evaluated, 4 shortlisted, 1 built and measured. The predicted best (`current_x_mcast`) measured rank 1, 0% selection slowdown. The measurement is 12,599 ns against 13,835 estimated, and it is marked post-hoc because the sweep ran after the ablation.

### Portfolio and verdicts (whole-op span ns; PCC gate 0.9998)
| Idea | Verdict | Before → after | PCC | Domain |
|---|---|---|---|---|
| `compute_dest_window`: DEST window 8–16 k-tiles, decoupled from the 4-k-tile x round | **NULL / REGRESSION** for the idea: windows > 4 k-tiles cost +0.5 to +7.5 µs whole-op. Found the real cause instead (next row) | 19,703 → 19,576…27,247 | 0.99992–0.99996 | measured-regression for windows > 1 K-step |
| `compute_llk_inner`: inner-loop efficiency | **WIN** — the helper's `LastBlockTarget::Out` + L1-acc path drained `cb_interm_acc` on the unpack thread after every K-step (12 exposed pack latencies) and reloaded on the last. Packing every K-step with L1-acc straight into `cb_out_block` gives compute alone 16,581 → 13,464 (matmul 15.3 → 12.2 µs, 1.07× SOL, ~34 cyc/tile ≈ LLK issue floor) | helper `caller_owns` 19,667 → 18,970; raw 18,846 / 18,804 | 0.999972 | everywhere (kb 2/4/8 and block_n 4 also faster) |
| `x_round0_fill`: sender BRISC reads round 0 on NoC1 | WIN at kb 4 (x0 2.78 → 2.14 µs; 19,690 → 19,293, −2%), **flat at the graduated kb 2** (18.70–18.74 without vs 18.73 with) → not graduated | — | 0.99996 | — |
| `tail_store`: per-tile last K-step + per-tile push (raw) | WIN on the old compute (19,690 → 19,141). Half of it, removing the last-K-step reload, is already in the graduated compute. The per-tile overlap was not re-measured on the new compute → not graduated; carried to Perf 2 | — | 0.999972 | inexpressible under TileRowMajor (needs a per-tile output order) |

### Graduated
- **Compute** (`kernels/wq_b_linear_compute.cpp`): still one kernel_lib `matmul_block`, now `LastBlockTarget::Interm` + `TileRowMajor` + `caller_owns_pack_target`, with the pack target = `cb_out_block`, reserved once and pushed once. It is the only compute path. Deleted: `cb_interm_acc`, the `OUT_SUBBLOCK_MAJOR` knob and the writer's per-subblock store path (the writer stores the whole TileRowMajor block). **No raw LLK.** The raw variant (keeps the per-column store overlap) measured 18,348 / 18,397 vs the helper's 18,700 / 18,742 at kb 2 (−1.9%, within noise), so the helper path ships.
- **`BLOCK_K_TILES` 4 → 2**: kb 2 had lost only because of the per-K-step drain. With the new compute, interleaved runs gave kb 2 18,613 / 18,659 vs kb 4 19,053 / 19,333 (every kb 2 dispatch below every kb 4 dispatch). x round 0 lands 1.69 µs instead of 2.2 µs and all workers finish together. On its own this is at the edge of noise (−2.9%). It ships as part of the combination, which is −5.5% end to end.
- No carve-outs (one cell; nothing incorrect or slower).
- **Whole op: 19,727 → 18,613 / 18,659 / 18,793 ns (3 runs, −5.5%)**, 1.33× the 14.0 µs SOL. PCC 0.9999734 (was 0.9999583). Compute alone 16,587 → 13,688 ns. DM alone (compute stubbed) 17,619 ns.
- Bottleneck now: x delivery. 24 rounds of 20 KB on one 64-receiver tree end at 16.5 µs, and compute ends right behind the last round (16.8 µs). After that comes the whole-block y store tail (~1.5 µs: pack 0.36, write issue 0.77, barrier 0.34). Candidates for Perf 2: a faster x delivery (more trees or both NoCs, now on the critical path), per-tile output overlap (`perf_experiments/tail_store`), and the raw compute's per-column store (−1.9%).
- Guard set = the single supported cell (one kernel path × TILE × DRAM-in/L1-out): golden 3/3 passed (registry cells + regression bit-identity), unit tests 14/14, `test_wq_b_linear.py` 10/10 under `--dev`. No regression.
- Options not taken (none trade precision): raw compute (−1.9%, noise), x round-0 BRISC read (flat at kb 2), W depth 3/6/whole at kb 4 (18.7–19.2, noise), kb 6/8 on the old compute (20,844 / 21,786, slower).
- Experiment dirs (uncommitted by the subagents, committed here): `perf_experiments/{compute_dest_window,compute_llk_inner,x_round0_fill,tail_store}/`.

### Helper bypasses — none
The graduated compute stays on `compute_kernel_lib::matmul_block`. Feedback for the helper library from this round, not a bypass: with `LastBlockTarget::Out` + `packer_l1_acc`, the helper drains `cb_interm_acc` on the unpack thread after every K-step and reloads the partials into DEST on the last K-step. This cost 3.1 µs of 15.3 µs here. The `caller_owns_pack_target` + `Interm` configuration avoids both, but the header does not say that the default L1-acc path serializes unpack behind pack.
