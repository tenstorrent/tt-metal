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
