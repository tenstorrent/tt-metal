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

## Perf 1
- Date: 2026-11-16
- Board: Blackhole p100a, 11x10 = 110 worker cores, AICLK 1350 MHz. Metric: `DEVICE KERNEL DURATION [ns]` (column 20 of this build's ops_perf_results CSV; column 19 is `DEVICE FW DURATION`, about 2.3 µs longer). Focus case = the only cell (LOOSE_CASES perf flag), SOL 16,018 ns.
- Instrumentation (permanent): added `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`). Zones: reader `reader_w_stream` / `reader_w_first_issue` / `reader_w_issue` / `reader_w_barrier` / `reader_y_wait` / `reader_y_store`; writer `writer_x_phase` / `writer_x_prefetch_issue` / `writer_x_prefetch_barrier` / `writer_x_send` / `writer_x_receive` / `writer_y_wait` / `writer_y_store`; compute `compute_first_kblock_wait` (unpack-side wait for K-block 0, hoisted out of the helper) / `compute_matmul`. No measurable overhead (24.86 µs FW duration with and without). Summarizer: `perf_experiments/breakdown/zones.py`.

### Measured breakdown (before)

| Variant (`_op_bench.py`) | Device ns | Part | Part SOL ns | Gap |
|---|---|---|---|---|
| full | 22,609 | whole op | 16,018 | 1.41× |
| w_only | 19,819 | W DRAM stream (NCRISC span p50 19,085; slowest core ends 20,300) | 14,920 (6,684,672 B / 448 GB/s; busiest core 104,448 B / 86.4 GB/s = 1,209) | 1.33× raw (floor 9,874 is hidden under W, so not subtracted) |
| x + y only | 16,736 | x broadcast (phase ends 14,350; round 0 lands 6,100) | 5,690 (491,520 B per core / 86.4 GB/s) | 2.52× |
| mcast_only | 14,465 | x multicast without prefetch | 5,690 | 2.54× |
| compute_only | 16,783 | matmul (math span 12,300) | 11,380 (480 tile-matmuls × 32 cycles) | 1.08× |
| no_y_write | 21,840 | y store exposed cost 769; per core p50 870 / max 1,540 | 240 (20,480 B / 86.4 GB/s) | 3.6–6.4× |
| no_compute | 21,033 | x + W + y | — | — |
| w_compute | 21,778 | W + compute | — | — |
| sync_only | 9,874 | floor (pack of 160 L1-acc tiles under SKIP_COMPUTE: 7.25 µs; 4 one-page rounds; y store) | 0 | — |

Overlap loss = full − w_only = 2,790 ns. Slowest core: W stream ends 20.30 µs → last K-block math 21.27 → pack 21.66 → y store 22.70.
Ranked: (1) W stream, 1.33× on the critical path; the parked NoC1 W share is blocked because the x phase holds BRISC/NoC1 until 14.4 µs. (2) The post-W tail, 2.4 µs (math 1.0, pack 0.4, y store 1.0). (3) The x broadcast, 2.5×, off the critical path except as the NoC1 blocker.

### Portfolio (cap: 2 experiments)
Floated: `x_bcast_active_rect` (multicast x only to the active-core rectangle from idle injectors outside it, to free NoC1 for the W share); `y_store_dual_noc` (split the y store across NCRISC/NoC0 and BRISC/NoC1); per-subblock early y release on the last K-block (needs a helper change); position-aware W column assignment (W end spreads 17.4–20.3 µs by core position); idle-core W forwarding on NoC1. Tested: the first two.

### Verdicts
- `x_bcast_active_rect`: NULL on the focus cell. Whole op 22,617–22,847 vs base median ≈22,670; graduate copy on/off 22,633 / 22,642. The x part alone does improve: x-only 14,666 → 12,253 with active rectangle + outside injectors + a 1:3:4:4:4 round split; round 0 lands 6.1 → 2.6 µs; x phase ends 14.6 → 12.7 µs. The premise failed: the NoC1 W share still loses at every setting (23.2–34.7 µs). NoC1 DRAM reads run at ≈185 GB/s, and the x phase would need to end by ≈8–9 µs, below the chained-multicast floor (≈40–48 B/cycle; shrinking 110 → 66 receivers barely changes the rate). NCRISC/NoC0 injector prefetch is fastest for x alone (11,098) but regresses the whole op +3.5% (contends with W on NoC0). Concurrent rounds: +35%. Domain: −8% on the non-shipped `CORE_N_TILES=3/4` splits (compute-bound there, mostly from the small first round); inexpressible where fewer idle cores than rounds sit outside the active rectangle. Not graduated (flat on the only supported cell; it would add a chain semaphore and a fallback path). Artifacts: `perf_experiments/x_bcast_active_rect/` (incl. `graduate/graduate.patch`).
- `y_store_dual_noc`: WIN, graduated. Isolated 65-core store (all cores at once): 2,310 → 1,544 ns. Whole op, subagent (3 alternated sessions): 22,672 / 22,733 / 22,745 → 22,315 / 22,314 / 22,317. Coordinator A/B (3 alternated sessions × 3 calls, `Y_READER_SHARE_TENTHS` 0 vs 4): median 22,681 → 22,441 (−240 ns, −1.1%; share 4 lower in 66 of 81 pairs). This sits inside the nominal 2–3% band, so the claim rests on the repeated direction and on the zone-level tail: slowest core, last pack → y end 1,040 → 780 ns; exposed y cost (full − no_y_write) 769 → 516 ns. Options: share 3/10 ≈ 4/10; 2/10 22,438; 5/10 and above flat to worse; 10/10 (all on NoC0) 23,705, a regression, because NoC0 still carries other cores' W responses. Split order: column-major ≥ row-major ≈ alternate. Only column-major was kept. Domain: everywhere tested, including the parked R4 path (flat, correct). No exceptions. PCC 0.999972, bit-identical repeats.

### Graduated
- The y store is one path: the reader stores the last `round(0.4 · 5·core_n)` y slots in column-major order on NoC0 after its W stream, peeking `cb_y_tiles`. The writer stores the rest on NoC1, then waits on `Y_DONE_SEMAPHORE_ID` and pops, so it stays the only consumer. The writer's old store loop was deleted; the shared loop lives in `kernels/wq_b_linear_y_store.hpp`. The subagent's row-major / alternate split orders were dropped, leaving no dual path. `Y_READER_SHARE_TENTHS` is a count, not a second implementation. No carve-outs.
- Whole op on the focus cell: 22,681 → 22,441 ns (median of 9), 1.40× SOL.
- `_op_bench.py` `no_y_write` now gates the writes inside `wq_b_linear_y_store.hpp` for both reader and writer.

### Guard set / correctness
One cell, one kernel path × layout × placement. Golden `eval/golden_tests/wq_b_linear/` 3/3; `tests/ttnn/unit_tests/operations/wq_b_linear/` 49/49 (acceptance 18, x-mcast knobs 7, W-stream knobs incl. 4 R4 cases 16, precision baseline 8). No regression.

### Helper bypasses — none
New code uses `Noc` / `TensorAccessor` / `CircularBuffer` and plain local semaphores; no raw LLK.

## Perf 2
- Date: 2026-11-16
- Board: Blackhole p100a, 110 workers, 7 DRAM banks, AICLK 1350 MHz. Metric: `DEVICE KERNEL DURATION [ns]` from `_op_bench.py --profile`. Focus case = the only cell (LOOSE_CASES perf flag), SOL 16,018 ns, pcc_threshold 0.9998.
- Instrumentation: zones from Perf 1 kept; new `compute_last_kblock` (TRISC 0/1/2) around the raw last K-block. `reader_y_wait` / `writer_y_wait` now cover the wait for column 0 only; later column waits fall inside `*_y_store`.
- `_op_bench.py`: `no_y_write` needle was stale after Perf 1 (assert failed); fixed, and both W and y gating needles moved to the new read/store lines.

### Measured breakdown (before)

| Variant | Device ns | Part | Part SOL ns | Gap |
|---|---|---|---|---|
| full | 22,313 / 22,410 / 22,333 | whole op | 16,018 | 1.39× |
| no_y_write | 21,736 | y store exposed ≈600 | 237 | — |
| w_compute | 21,543 | W + compute | — | — |
| no_compute | 21,476 | W + x + y | — | — |
| w_only | 20,696 | W stream (`reader_w_stream` end p50 18,840 / max 19,512) | 14,921 (6,684,672 B / 448 GB/s) | 1.31× (stream end) |
| x + y only | 16,081 | x broadcast (x phase end 14,740) | 5,690 | 2.6×, off the critical path |
| mcast_only | 13,643 | x multicast without prefetch | 5,690 | 2.4× |
| compute_only | 16,130 | matmul (math span 13,870) | 11,378 (480 tile-matmuls × 32 cycles) | 1.22× |
| sync_only | 9,201 | floor | 0 | — |

Cumulative peel: full 22,333 → −x 21,543 → −compute 20,696 → −W 9,201. Overlap loss = full − w_only = 1,637 ns (W ends 0.7 µs later in the full op than alone, plus the tail). Slowest core: W end 20.18 → math end 21.07 → pack end 21.46 → y store end 22.29 µs.
Ranked: (1) W stream, 1.31× its SOL, on the critical path; (2) post-W tail 2.1 µs (last K-block math 0.9, pack 0.4, y store 0.8); (3) x broadcast, off the critical path.

### Portfolio (cap: 2 experiments)
Floated: `w_dram_port_spread` (read each W bank through more than its one preferred NoC0 DRAM endpoint); `y_last_kblock_early_release` (pack and release y per output column on the last K-block so the store overlaps the remaining math + pack); route-aware dual-NoC W reads from NCRISC (needs dynamic NoC mode); a smaller last K-block (inexpressible with the fixed-k helper). Tested: the first two.

### Verdicts
- `w_dram_port_spread`: WIN, graduated. Mechanism: the cap is per DRAM port (≈50 GB/s per bank on one NoC2AXI port), not the row egress links (forcing distinct egress rows was inconclusive: 17.7 / 19.0 µs). W stream alone: TensorAccessor 19,123–19,550 → `rot2` (alternate the NoC0 endpoint and the bank's free port by k row) 16,219–16,336 (1.09× its SOL); `rot` (all 3 ports) 15,944–16,186. Whole op (subagent): `rot2` 20,930 / 20,988; `rot` 23,330, a regression: W on the NoC1 endpoint port queues the x prefetch behind it (round-0 prefetch barrier 0.6 → 3.1 µs). Graduated `rot2`. Domain (W stream alone, `acc` → `rot2`): N=7168 −13%, bf16 W −17%, K=768 −12%. Measured regression at N=2048 (36 cores, below the port cap): 12,739 → 13,879 (+9%); not a supported cell, so no carve-out. PCC 0.999972, bit-identical.
- `y_last_kblock_early_release`: WIN, graduated. Whole op (subagent): 22,371 (median of 12) → 22,040 (median of 9), candidate faster in 8/8 same-session pairs; slowest-core tail after W 2,146 → 1,742. Options: column release share 4 (graduated) ≈ row release share 4 (22,019); row share 0 21,873 (n=5, inside noise). Domain: CORE_N_TILES=4 −4.5%, K=96 −9%, CORE_N_TILES=0 / M=32 / M=64 / R4 flat; no exceptions. PCC 0.999972, bit-identical.

### Graduated
- W read (`kernels/wq_b_linear_w_read.hpp`): one path. Odd k rows go to the bank's free NoC2AXI port, even rows to TensorAccessor's preferred endpoint; port coordinates derived on device from `dram_bank_to_noc_xy`, no host change. Carve-outs (all correctness/expressibility, written as `if (cannot) preferred endpoint`): non-Blackhole builds; Blackhole with programmable DRAM cores enabled (non-endpoint ports are in stream mode and return wrong bytes; checked at runtime via `tensor_prefetcher_signal_noc_xy`); DRAM coordinates not matching the Blackhole port layout; W not in DRAM. The prefetcher fallback was not run on device.
- Last K-block (`kernels/wq_b_linear_compute.cpp`): the helper runs K-blocks 0..N-2, a raw-LLK loop runs the last block with column groups outermost and pushes each group as it is packed. y store (`wq_b_linear_y_store.hpp`, reader + writer): column by column after `wait_front(n + 1)`; in each column the reader (NoC0) stores the last `round(0.4 · block_m)` rows and the writer (NoC1) the rest. The old slot-order split (`reader_owns_y_slot`) was deleted. Carve-out: the parked R4 path (`K_REMAINDER_SPLIT`) keeps the single helper call, because its leftover-column combine packs at write-pointer offsets that assume nothing was pushed yet; its DM side uses the column-wise store (correct, flat).
- Whole op on the focus cell, coordinator A/B (3 alternated sessions × 3 calls, HEAD kernels vs new): median 22,346 → 20,509 ns (−8.2%), 1.40× → 1.28× SOL. Port spread alone (2 sessions × 3 calls): median ≈21,037; adding the column release: 20,476.
- After: W stream ends max 17.71 µs, math ends max 19.08, y store ends 20.56. Compute now trails W by ≈1.4 µs: its first K-block waits for x round 0 until 6.6 µs and then needs 12.2 µs of math. Next round's bottleneck is x round-0 arrival + math (Perf 1's NULL `x_bcast_active_rect`, which brought round 0 to 2.6 µs, now addresses the critical path).

### Guard set / correctness
One cell, one kernel path × layout × placement. Golden `eval/golden_tests/wq_b_linear/` 3/3; `tests/ttnn/unit_tests/operations/wq_b_linear/` 49/49 (incl. R4, x-mcast and W-stream knob cases). No regression.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `TensorAccessor` page read (`noc.async_read(accessor, ...)`) | capability | always addresses a DRAM bank's one preferred endpoint per NoC; no way to target the bank's other NoC2AXI ports, and one port caps a bank at ≈50 GB/s | 19,123–19,550 (W stream alone); 22,346 whole op | 16,219–16,336; ≈21,037 whole op | `kernels/wq_b_linear_w_read.hpp:78` |
| `compute_kernel_lib::matmul_block` | capability | (1) no per-subblock / per-column-group push on the last K-block under `caller_owns_pack_target`; (2) the L1-accumulation switch is keyed on the block index inside one call, so a separate one-block call for the last K-block overwrites the earlier partials instead of accumulating; (3) no column-outer subblock order on the last block | ≈21,037 (whole op, port spread only) | 20,476 | `kernels/wq_b_linear_compute.cpp:30` |
