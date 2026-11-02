# Changelog: qkv_proj_heads_rope

## Phase 0 - Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1: 2D token_rows x head_cols output-stationary grid (108/110 cores on p100a), unicast reads, matmul with the bias folded as the 25th K tile into Float32 L1-accumulated partials, rotary as two DEST-accumulated FPU multiplies per output tile, one `generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED], buffer_type=[L1], out_buffer_type=[L1], table_buffer_type=[L1]
- **Accuracy achieved**: PCC >= 0.9999984, max_abs_err <= 0.0212, rel_rms <= 1.91e-3 (gate 4e-3), ULP p99 <= 4.9 (measured on 4 shapes via test_qkv_proj_heads_rope_precision_baseline.py); q/k got/true median 0.99925 (truncation bias of the rotary FPU multiply, inside tolerance), v 1.0000
- **Golden suite at Phase 0**: 25 supported_pass / 175 xfail_expected / 0 loud of 210 (per `verifier_results/verifier_report.json`, run with a local, reverted patch for two golden-harness defects; see verification_report.md)
- **Perf at Phase 0**: 8x512 279-281 us (stock 226.3, target 90); every flagged row slower than stock
- **Issues encountered**: verifier DRY fix: the reader derived the table row size and page index from literals; `TILES_PER_HEAD` is now a reader CT arg. Golden harness defects (`packer_l1_acc` kwarg, tuple tolerance) reported, not edited.
- **Tests added**: test_qkv_proj_heads_rope.py, test_qkv_proj_heads_rope_perf.py, test_qkv_proj_heads_rope_precision_baseline.py

## Refinement 1 - x multicast within each M-group (regime R3) + `buffer_type = DRAM`
- **Date**: 2026-11-02
- **What was done**: Built regime R3. Logical cores are reordered to `l = m_group*grid_n + n_group`, so each M-group is a consecutive run that `ttnn.Mcast` can partition. One `Mcast` channel on the reader (NOC_0, handshake on, fixed sender with `Staggered` placement) delivers each x K-block: every core reserves its `cb_activation_block` slot, the group's sender reads the x tiles plus the ones column into it and multicasts the leading `rows*k_block` tiles to the same slot on the other `grid_n - 1` cores. Removed the per-core x read stagger (one reader per K-block now). Compute kernel, writer, CB contracts and selection function are unchanged. Added `ttnn.BufferType.DRAM` to `SUPPORTED["buffer_type"]`.
- **Deviation**: the verifier asked for a rotating sender. `McastRotatingSenderConfig` requires each receiver group to be a rectangle (`mcast.cpp:89`), and `grid_n` consecutive cores on the 11-wide grid usually are not, so the sender is fixed per group and staggered across groups. Because the kernel serializes rounds (read, multicast, next round), rotation would not overlap reads anyway.
- **Perf (device us, x in L1)**: 8x512 279 -> 265, 16x256 278 -> 267, 32x128 275 -> 270, 8x384 252 -> 252, 4x512 215 -> 145, 8x256 213 -> 147, 2x512 206 -> 93, 4x256 205 -> 97. x in DRAM: 8x512 272 (+2.5% vs L1, was 317), 4x512 147, 2x512 92 (was 221).
- **Bottleneck now**: at 8x512, stubbing W reads gives 134 us and stubbing the x path gives 277 us. The W unicast re-read (12 DRAM crossings) binds, which is Refinement 3 (R2). The M-group-consecutive order means R2 needs per-N-group `Mcast` channels or a rectangular-grid constraint (see `op_design.md`).
- **Accuracy achieved**: unchanged data path; unit suite gates PCC >= 0.9999, rel RMS <= 4e-3 per output on 1x32, 1x128, 2x64, 3x96, 4x256, 8x256, 32x128, 8x512 (x in L1) and 1x32, 2x512, 4x256, 8x384, 8x512 (x in DRAM).
- **Golden test progress**: 42/42 tile_aligned cells with out/table in L1 pass (17 of them with x in DRAM, `test_op` + `test_op_loose`). `test_regression.py`: 6 passed, 4 failed (all `h_non_aligned`, Refinement 2 scope, failing before this change).
- **Issues encountered**: the `-k` filter in `run_safe_pytest.sh` cannot take parentheses or `=`, so golden slices were run by explicit node id.
- **Tests added**: `test_qkv_proj_heads_rope.py::test_x_in_dram` (5 shapes, one per grid class), `test_qkv_proj_heads_rope_perf.py::test_qkv_proj_heads_rope_perf_x_dram` (8x512, 4x512, 2x512).

## Refinement 2 - Non-aligned S (`h_non_aligned`) + DRAM outputs and tables
- Date: 2026-11-02
- What was done: SUPPORTED widened: alignment += h_non_aligned, out_buffer_type += DRAM, table_buffer_type += DRAM. Writer gets CT arg `last_valid_rows = S % 32`; on the last tile-row of each image it zeroes rows `[S % 32, 32)` of every output tile in cb_out_tiles via `noc.async_write_zeros` (partial rows of faces 2*(r/16)+{0,1}, whole lower faces when r < 16) before the NoC write. `if constexpr` removes it for aligned S. DRAM placements needed no kernel change (TensorAccessor). Reused: existing per-row output path, CB, zeroing primitive used for the bias tile. Added: one CT arg and the pad-zero block.
- Accuracy achieved: unit + golden tolerance (PCC >= 0.9999, rel RMS <= 4e-3) on [2x37, 16x8, 13x300, 7x511, 4096x1 (multi-block), 2x512, 8x512, 3x100]; pad rows exactly 0 with inf in x and table padding.
- Golden test progress: 210/210 passing (test_golden.py 200 + test_regression.py 10).
- Issues encountered: `async_write_zeros` on a CircularBuffer addresses from the write pointer; the pad-zero offsets are taken relative to it from the read pointer. 4096x1 cannot be L1-resident (tile padding makes x ~200 MB), so the unit case places x and outputs in DRAM.
- Perf: 8x512 aligned all-L1 268 us (post-R1 265 us; noise band).
- Tests added: `test_non_aligned_padding_zero` (5 shapes, inf padding, pad rows == 0), `test_dram_outputs_and_tables` (3 cases) in test_qkv_proj_heads_rope.py.

## Refinement 3 - W multicast within each N-group (regime R2) on the flagged 8x512 profile
- Date: 2026-11-02
- What was done: Built R2. The writer gets a "wmc" `McastArgs` channel: when present, the N-group's sender reads each W K-block (bias tile row + its padding zeroing included) into its `cb_weight_block` slot and multicasts it to the same slot on the other `grid_m - 1` cores; receivers `receive(round)`. When absent (R1), the old per-core read runs unchanged. Host: `ttnn.Mcast` only partitions consecutive runs, so both channels are now built as one `Mcast` + one kernel descriptor per group (`_group_kernels`, shared semaphore IDs across disjoint groups). `_choose_layout` places the logical grid so every group is <= 3 rectangles: row-major (R3 placement) if its N-groups qualify, else a folded layout (12x9 on 11x10: M-groups one line, N-groups a column + a 2-core row). R2 is gated on a qualifying layout, `grid_m >= W_MCAST_MIN_GROUP` (4) and equal block counts across M-groups. Reused: R3's per-round reserve/send/receive/push pattern, existing W read loop (now a lambda), CB contracts, compute kernel, selection function. Added: layout + rectangle cover + per-group attach helper on the host, the W channel in the writer.
- Perf (device us, x in L1): 8x512 268.7 -> 123.7, 16x256 269.8 -> 123.9, 32x128 269.9 -> 124.1, 8x384 253.0 -> 92.4, 4x512 146.1 -> 144.7 (R1, unchanged path), 8x256 143.8 -> 145.3 (same), 2x512 92.2 -> 95.1, 4x256 95.6 -> 97.2 (R1 + R3, noise band). x in DRAM: 8x512 125.3, 4x512 145.3, 2x512 91.5.
- Accuracy achieved: unchanged arithmetic (W bytes identical); unit suite gate PCC >= 0.9999, rel RMS <= 4e-3 per output on every unit shape (1x32 ... 8x512, 2x37, 13x300, 7x511, 4096x1 multi-block).
- Golden test progress: `test_golden.py -k "loose or h_non_aligned"` 64/64, `test_regression.py` 10/10.
- Issues encountered: `if constexpr (channel.active)` in non-template `kernel_main` still instantiated the absent channel's `sender()` static_assert; moved into a generic lambda. W multicast on the 3x36 grid (3 one-core rectangles per N-group) measured 105-129 us vs 94 us unicast for 2x512, so it is gated off there. 6x18 has no <= 3-rectangle layout under the row-major/folded candidates and stays R1.
- Bottleneck now: 8x512 with the matmul stubbed runs 66.8 us (vs 123.7), so the matmul is the larger share; Refinement 4.
- Tests added: none (existing unit shapes cover R2 on 12x9, non-aligned S, multi-block 4096x1, and the R1 fallbacks on 6x18 / 3x36).

## Refinement 4 — Block / compute knob co-tune on the flagged 4096-token rows
- Date: 2026-11-02
- What was done: Ablation at 8x512 (123.5 us): matmul alone 87 us (FPU-bound at HiFi3), DM alone 64 us, x or W stubbed alone < 1 us, table + output stores stubbed together -24 us. The binding stage was the post-K-loop tail, not the K-loop. Built:
  - Table prefetch in the K-loop slack: `cb_rotary_table` holds one slot per distinct sequence position of a block (`min(block_m, St)` rows); the reader reads a chunk after each x K-block push (one barrier per chunk), compute waits/pops once per block and indexes slot `row % St` (`TileAddressing::Offset` on the table operand).
  - Output stores on both NoCs: new `cb_out_tiles_noc0`; compute packs odd rows there and the reader stores them on NoC0, the writer stores even rows on NoC1. The store path (pad-row zeroing + head-layout scatter) moved into `kernels/qkv_proj_heads_rope_store.hpp`, shared by both kernels. Knob `OUT_WRITE_NOCS` (1 = writer only).
  - Epilogue chain setup (reconfig + `mul_init` / `copy_init`) hoisted to once per block via `InitReconfigOwner::Caller` (was per row x head).
  - `block_m` knob (`_block_m`): equal block counts across M-groups first (R2's W rounds must line up), then fewest blocks, then fewest nominal rows. `block_m_cap` is footprint-derived with the table term capped at St.
  Reused: reader/writer K-loop, Mcast channels, matmul path, pad-zero code (moved, not copied). Added: one CB, one shared header, the table chunking, the `_block_m` rule.
- Perf (device us, x in L1): 8x512 123.5 -> 106.2, 16x256 123.7 -> 106.6, 32x128 124.0 -> 105.8, 8x384 92.4 -> 78.2, 4x512 145.2 -> 122.6, 8x256 147.0 -> 118.8, 2x512 94.0 -> 82.9, 4x256 95.3 -> 73.9. x in DRAM: 8x512 125.3 -> 107.8, 4x512 145.3 -> 122.1, 2x512 91.5 -> 83.1. Guards vs the Refinement 3 kernels: 8x512 DRAM out 165.9 -> 147.2, 7x511 213.5 -> 185.2, 2x512 DRAM out 94.6 -> 82.0, 4096x1 (multi-block) 4955 -> 4494, 38x106 200 -> 181, 77x53 200 -> 180, 18x225 384 -> 169 (R2 now on), 8x500 165 -> 148.
- Measured, not kept: subblock shape (1x4 / 2x2 / 4x1 within 0.5 us, matmul FPU-bound), 2 blocks per core (128.4 vs 117.5 us: nominal-block rows + second W pass), per-third row rotation of the stores (no change), dropping the per-row write flush (no change).
- Accuracy achieved: arithmetic unchanged (same matmul, same rotary products); unit suite gate PCC >= 0.9999, rel RMS <= 4e-3 per output on every unit shape (1x32 ... 8x512, 2x37, 13x300, 7x511, 4096x1 multi-block).
- Golden test progress: `test_golden.py -k "h_non_aligned or 8x512 or 2x512 or 128x32 or 64x64"` 90/90, `test_regression.py` 10/10.
- Issues encountered: a whole-block table CB (`block_m` rows) lowered `block_m` 13 -> 11 at 4096x1, which made M-group block counts unequal and silently turned R2 off (4960 -> 8286 us); fixed by sizing the table by distinct positions and by the `_block_m` rule. A host sweep over 4815 (B, S) shapes: no shape loses R2 vs Refinement 3, 77 gain it.
- Bottleneck now: at 8x512 the K-loop ends at ~88 us on every core; the output stores then take ~12-18 us (both NoCs, y-gradient across the grid). Next lever: start the stores inside the last K-block (split it per row chunk with the epilogue interleaved), which needs `matmul_block` to accumulate onto an existing L1-acc target from its first K-block (today it hard-codes L1-acc off on block 0).
- Tests added: none (unit shapes cover single/multi-block, non-aligned S, R1/R2 layouts; the perf file covers the eight rows).

## Refinement 5 - Small-token flagged rows (2048 / 1024 tokens)
- Date: 2026-11-02
- What was done: Forced-h_c sweep (60 runs over 12 shapes) showed h_c = 4 with R2 (W multicast) is fastest on every shape with grid_m >= 4; the Phase 0 rule (occupancy first, then ingress) assumed both multicasts and picked h_c = 2 / 1 for Mt = 64 / 32, which run W unicast. Built:
  - Selection by modelled device time (`_stream_ns`): max of matmul (`tiles*25*phases*13 ns`), per-core W stream (110 ns/tile), grid-wide W DRAM crossings (10.8 ns/tile, x grid_m under R1) and x multicast (7.1 ns per tile per receiver), then the old tie-breaks. It ranks the measured best first on all 12 shapes; 347 of 876 (B, S) shapes move to h_c = 4, every one checked is faster. `OCCUPANCY_SLACK` removed; `_w_multicast` / `_row_ranges` factored out of `_core_assignment` so the selector sees R2.
  - R2 sender read-ahead: `weight_depth` 3 (`WEIGHT_DEPTH_PREFETCH`) when W is multicast, the modelled W stream >= 0.75x the matmul (`PREFETCH_MIN_W_SHARE`) and the footprint fits. The writer splits the W fill into issue / finish; the next round's sender issues its reads into slot r+1 before round r's transfer (after it on round 0, the fill), when that slot is already free.
  - `W_SENDERS` knob (second "wmc1" Mcast channel, senders alternate rounds); measured slower, parked at 1 (channel attached absent, single-sender path).
  Reused: select_blocking candidate loop, `_choose_layout`, `_block_m`, writer W read loop (split, not copied), Mcast channel attach (`_group_kernels` now takes a prefix list). Added: cost model, read-ahead branch, opt-in `test_qkv_proj_heads_rope_perf_sweep.py` (QKV_SWEEP=1; forces h_c / depth / senders).
- Perf (device us, all-L1): 4x512 123.0 -> 59.2, 8x256 119.3 -> 59.3, 2x512 82.7 -> 39.2, 4x256 74.1 -> 38.7; 8x512 105.8, 16x256 105.8, 32x128 105.9, 8x384 78.5 (unchanged). Guards: 7x511 189.7 -> 95.8, 38x106 134.2, 77x53 134.7, 18x225 118.0, 8x500 106.3 (flat vs the depth-2 / old-h_c runs); 1x512 107.5 -> 37.1, 3x100 107.7 -> 35.4, 1x225 54.7 -> 27.8, 2x37 54.1 -> 30.4; 1x32 23.4 (unchanged, h_c = 1).
- Measured, not kept: W_SENDERS = 2 (2x512 41-49, 8x256 69-83 us, high variance); read-ahead after a CallerManaged send (2x512 40.9); read-ahead before round 0's send (2x512 37.8 but 4x512 +1.8, 8x384 +1.2, 38x106 +2.5; hence the round-0 rule and the W-share gate).
- Accuracy achieved: arithmetic unchanged (same matmul, same rotary products, same W bytes); unit gate PCC >= 0.9999, rel RMS <= 4e-3 per output on every unit shape (1x32 ... 8x512, 2x37, 13x300, 7x511, 4096x1 multi-block) and the sweep shapes.
- Golden test progress: `test_golden.py` + `test_regression.py -k "2x512 or 4x256 or 4x512 or 8x256 or 3x100 or 2x128 or 2x37 or 7x511 or 16x8 or regression or large_mean or padded"` 95/95.
- Issues encountered: ablated kernels that fail the correctness check leak L1 outputs and break later tests' CB allocation, so ablations ran through the sweep test with QKV_SWEEP_NOCHECK=1. One profiling run came back ~1.7x slow on every shape (outlier, not reproduced).
- Bottleneck now: 4x512 is matmul-bound at HiFi3 (stubbing the matmul gives 42.2 us). 2x512 is bound by the single W sender per N-group (W path 30.8 us, stubbing W gives 30.0). Next: spread W reads across the N-group, or an R2 layout for 6x18.
- Tests added: `test_qkv_proj_heads_rope_perf_sweep.py` (opt-in, skipped by default).

## Perf 1
- Date: 2026-11-02. Device: Blackhole p100a (108 of 110 cores used at the focus shape), AICLK 1350 MHz. Metric: DEVICE KERNEL DURATION [ns], one fresh-cache run per point.
- Focus: flagged LOOSE_CASE 8x512, exact config (x / cos / sin / q / k / v in L1, W / bias in DRAM, bf16 TILE interleaved, default compute config HiFi3 + fp32_dest_acc_en + approx off). All flagged rows share that config and are in SUPPORTED.
- Instrumentation (permanent `MaybeDeviceZoneScope`): reader `reader_x_reserve / _read_issue / _read_barrier / _mcast_send / _receive`, `reader_table`, `reader_bias`, `reader_store_wait_first`, `reader_store`, `reader_write_barrier`; writer `writer_w_reserve / _read / _mcast_send / _receive`, `writer_store_wait_first`, `writer_store`, `writer_write_barrier`; compute `compute_bias_wait`, `compute_partials_reserve`, `compute_bias_seed`, `compute_matmul`, `compute_last_k_wait`, `compute_last_k_epilogue`. Tools: `perf_experiments/_breakdown/` (zone summarizer, ablation runner, ablation-hook patch).

### Measured breakdown (8x512, before)
| point | us |
|---|---|
| full | 106.4 |
| input DM stubbed (x / W / table reads + both multicasts) | 100.7 |
| + output stores stubbed | 92.5 |
| + epilogue math stubbed | 88.3 |
| + matmul stubbed (unpack / pack / sync floor) | 24.5 |
| compute + stores stubbed (input DM alone) | 69.2 |
| stores stubbed alone | 99.1 |

Zones (median core): matmul 0.3-88.2 us, FPU-bound (~54 cycles per tile-product at HiFi3, 2200 products per core); W rounds done 63 us; epilogue 88.9-99.2 (max 105.4); stores end 100.0 (max 106.0). Ranked: (1) serial post-K-loop tail, ~8 us epilogue + ~7 us store burst (bisection-bound, 19 MB from 108 cores at once); (2) pipeline fill ~6.5 us (W round-0 DRAM read ~4 us + multicast); (3) bias K tile = 1/25 of the FPU-bound matmul (~3.5 us). The matmul payload itself is at the HiFi3 FPU roofline and was not targeted; the precision contract was not touched.

### Portfolio and verdicts
| idea | verdict | 8x512 before -> after (isolated vs the then-current op) | domain |
|---|---|---|---|
| epilogue_overlap: last K-block per row, epilogue + store follow each row (lead 0 / 1 / 2: 102.1 / 100.5 / 100.3 us) | WIN | 106,263 -> 100,530 | everywhere; 4x512 -3.1%, 7x511 -5.0%, 256x16 (2 blocks) -4.0%, 2x512 flat |
| epilogue_fast: rotary as whole-tile phase-major ELWMUL (menu: per-row DEST windows 7787, face-major whole tile 7936, phase-major 4319 vs 8422 ns per 11x8 block, isolated) | WIN, bit-identical | 106,553 -> 104,641 (store-bound then; full ~3.7 us shows with stores stubbed: 99.0 -> 95.3) | everywhere; HiFi3 / HiFi4 / LoFi, 1-6 heads, multi-block; 4x256 h3 and multi-block flat |
| fill_shrink: ragged first K-blocks [2, 4] + table chunks off round 0 | WIN (isolated), not graduated | 106,600 -> 103,900 | everywhere (x_local0 variant: measured regression, 107-118 us, rejected) |
| bias_seed: K' = 24, bias row-broadcast seeded into the fp32 partials (menu: DEST seed 106.7-112.0, late full-tile bias read 112.2, seed + 64 B bias spans at k_block 3 / 4 / 5 / 6: 103.5 / 103.4 / 104.6 / 105.0) | WIN | 105,822 -> 103,388 (k_block 4) | everywhere; R1 6x18 grids -6 to -8%, 4096x1 -1.5% |
| (coordinator) bias read on the writer / NoC1 instead of the reader | REGRESSION | 95.0 -> 96.8 (all rows +1-3 us: bias barrier 3.1 us on NoC1, W round-0 multicast 3.4 -> 5.2 us) | not graduated; patch in `perf_experiments/_breakdown/` |

fill_shrink was not graduated: after bias_seed the fill is bound by the bias DRAM read (~3.2 us at t = 0) plus the seed's 88 fp32 packs (~3.2 us), which must land before K-block 0 packs, so the fill floor is ~6.4 us against 7.0 us measured; a smaller round 0 can recover at most ~0.6 us, and porting the ragged schedule needs a ragged per-row last K-block. Round 2 target.

### Graduated (each the one unqualified path, code it replaced deleted, no carve-outs)
| step | perf rows 8x512 / 16x256 / 32x128 / 8x384 / 4x512 / 8x256 / 2x512 / 4x256 (ns) | golden |
|---|---|---|
| before | 106,373 / 106,501 / 106,002 / 78,665 / 59,424 / 59,108 / 39,489 / 38,635 | 210/210 |
| + epilogue_overlap | 100,440 / 100,440 / 100,403 / 75,320 / 57,971 / 58,194 / 38,939 / 38,870 | 210/210 |
| + epilogue_fast (rotary eltwise_chain deleted) | 96,730 / 96,514 / 96,828 / 73,060 / 55,816 / 56,062 / 39,097 / 38,725 | 210/210 |
| + bias_seed (ones column, bias K row, bias pad zeroing deleted) | 95,023 / 95,097 / 94,967 / 72,029 / 55,128 / 54,973 / 37,003 / 36,644 | 210/210 |

Whole op, focus 8x512: 106.4 -> 95.0 us (-10.7%; stock 226.3, target 90). Every flagged row faster; still above target_ns on every row.
Helper change: `matmul_block` gains `seeded_pack_target` (template bool after `caller_owns_pack_target`, default false = unchanged behavior; no existing caller passes that many template args): K-block 0 packs with L1 accumulation on, onto a caller-written target.

### Guard set (`test_qkv_proj_heads_rope_perf_guard.py`, before -> after ns, all correct)
| row | before | after |
|---|---|---|
| 8x512 all L1 (R2 + R3) | 106,404 | 94,981 |
| 2x512 (W-multicast bound, depth 3) | 39,452 | 36,859 |
| 7x511 h_non_aligned | 96,280 | 87,510 |
| 13x300 h_non_aligned | 106,619 | 95,445 |
| 8x512 x in DRAM | 108,234 | 98,312 |
| 8x512 tables + outputs in DRAM | 147,141 | 137,280 |
| 1x32 (R1 W unicast) | 23,451 | 22,143 |
| 1x128 | 30,239 | 28,056 |
| 2x37 | 30,869 | 28,407 |
| 16x8 | 37,981 | 34,705 |
| 4096x1 multi-block, all DRAM | 4,506,236 | 4,319,166 |
No regression on any cell. SUPPORTED unchanged.

### Bottleneck now (8x512, 95.0 us)
Bias read + seed ~6.4-7 us before K-block 0, matmul K-blocks 0..4 ~70 us, last K-block + epilogue ~19 us (stores hidden). Next: shrink the bias/seed fill (bias via the W multicast, or seed after K-block 0), then the ragged first K-block.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| matmul_block | capability | No per-row (per M-subblock-row) hook after a row's last-K pack, so an epilogue cannot follow each row; a separate per-row call for only the last K-block was also impossible because K-block 0's L1 accumulation was hard-coded off (now `seeded_pack_target`; the per-row hook is still missing). | 106,263 (whole K-loop in helper, serial epilogue) | 100,530 | qkv_proj_heads_rope_compute.cpp:177 (`last_k_row`) |
| eltwise_chain BinaryFpu (mul_tiles) | capability | Only the face-major ELWMUL sequence: each ELWMUL accumulates into DEST rows written two instructions earlier and the fp32 DEST read-modify-write stalls (~2.5 cycles per ELWMUL); no whole-tile src unpack with fidelity-phase-outermost MOP. Also `DestAccumulation::PerRow` gives one output tile per DEST window, so a chain cannot batch the 4 products of a head into one window. | 8,422 per 11x8 block (isolated); 100,440 e2e | 4,319; 96,730 e2e | tile_src_mul.hpp:84, qkv_proj_heads_rope_compute.cpp:134 |
| eltwise_chain CopyTile / no helper for a broadcast seed | capability | No element packs one DEST tile to several CB tiles (one-to-many pack), so a row-broadcast seed of a block_m x N_c region through a chain costs block_m unpacks per bias tile instead of one; seed written with the compute API (`unary_bcast<ROW>` + `pack_tile<true>`). Chain variant not built, so no helper-side number. | not measured | seed 0.4 us unpack / 3.1 us math / 6.0 us pack zone at 8x512, inside the fill | qkv_proj_heads_rope_compute.cpp:154 (`seed_partials`) |

## Perf 2
- Date: 2026-11-02. Device: Blackhole p100a (108 of 110 cores at the focus shape), AICLK 1350 MHz. Metric: DEVICE KERNEL DURATION [ns], one fresh-cache run per point.
- Focus: flagged LOOSE_CASE 8x512, exact config (x / cos / sin / q / k / v in L1, W / bias in DRAM, bf16 TILE interleaved, default compute config HiFi3 + fp32_dest_acc_en + half-sync DEST + approx off). In SUPPORTED. Precision knobs untouched.
- Instrumentation: Perf 1 zones reused; new `reader_bias_barrier`. Ablation hooks for this round: `perf_experiments/_breakdown/ablation_hooks_perf2.patch` (`ABL_NO_{STORE,XREAD,WREAD,TABLE,MCAST,BIAS,EPI,MATMUL}`); flagged / guard row runner `perf_experiments/_breakdown/perfrows.sh`.

### Measured breakdown (8x512, before)
| point (cumulative peel) | ns |
|---|---|
| full | 95,024 |
| stores stubbed | 93,602 |
| + input DM stubbed (x / W / table / bias reads, both multicasts) | 88,186 |
| + epilogue math stubbed | 86,302 |
| + matmul math stubbed (seed / pack / sync floor) | 10,162 |
| compute stubbed (matmul + epilogue), DM alone | 74,153 |
| input DM stubbed alone | 91,127 |

Compute alone is 88.2 us; DM alone 74.2 us, so DM does not bind. With no DM the K-loop ran ~52.4 cycles per tile product against 48 at HiFi3 (16 per fidelity phase): ~4.4 cycles per product of fixed issue overhead, ~8 us. Ranked: (1) matmul issue overhead ~8 us; (2) DM fill + store tail ~6.8 us (x round 0 waited behind the 3 us bias barrier, bias + seed pack before K-block 0); (3) epilogue math 1.9 us exposed; (4) M imbalance (11 vs 10.67 rows per core, ~2.3 us; structural, not floated). The matmul payload at 48 cycles per product is the HiFi3 roofline and was not targeted.

### Portfolio and verdicts (8x512 e2e ns in each subagent's op fork, baseline ~94.9k)
| idea | verdict | before -> after | domain |
|---|---|---|---|
| matmul_kloop: one math MOP per `matmul_block` call (lean: CLR_AB on the call's last MVMUL, DEST offset written only on change). Menu: raw K-loop with stock LLK 95,271 (flat); fused + end op 93,793; fused, no end op 92,560; lean 92,045; kb6 97,060 and lean+kb6 93,769 (bigger fill and last K-block outweigh 52.4 -> 52.0 cycles); kb8 inexpressible at 8x512 (CB region collides with L1-resident x) | WIN, bit-identical | 94,919 -> 92,045 | everywhere: 4x512 -2.8%, 7x511 -3.3%, 32x128 -3.2%, 4096x1 -2.0%, HiFi4 / HiFi2 / LoFi / full sync bit-identical; 2x512, 4x256 h3 (2x2 subblock) and 1x32 flat |
| fill_v2: shorten the fill before K-block 0. Menu: XFIRST (x round 0 issued behind the bias spans, one barrier) 93,866; BMC+WSHIFT (bias read by 9 W senders, multicast as a leading W round, W sender moved off the x sender) 93,531; BMC+BWRITER+KS (ragged [2,4,4,4,5,5], k_block 5 slots) 92,996; RAMP [2,2] 95,855; BTRID, BCOMPACT, BPOLL flat | WIN (XFIRST graduated) | 94,853 -> 93,866 | XFIRST everywhere (2x512 and 4096x1 flat). BMC+WSHIFT: 2x512 +1.7% over 3 runs. BMC+BWRITER+KS: 2x512 +12.6% (measured regression) |
| tail_v2: (a) fuse last K-block + epilogue in DEST; (b) cheaper epilogue / schedule. Menu: preuse (rotary unpacks each fp32 partial once for 2 products) 94,109; vblk (v convert one DEST window per dest_tiles) 94,470; preuse+vblk 93,547-93,684; halfsplit 93,758 (flat); row groups 2 / 4 / 10 95,650 / 99,157 / 101,565; LAST_K_LEAD 2 95,690 | WIN (preuse+vblk), bit-identical. Fusion (a): NULL, not built; removing the whole rotary epilogue saves only 1.1 us, and fusion would remove just the partial unpacks out of that | 94,987 -> 93,590 | everywhere: 4x512 / 2x512 / 7x511 -1.3%; 4096x1 and 1x32 flat; HiFi4 / LoFi / full sync / 3 heads per core bit-identical |

BMC+WSHIFT was not graduated: over XFIRST it saves 0.3-0.5 us at 8x512, inside the run-to-run band, and it costs 0.6 us on flagged 2x512. To get that, it needs a host-side W-sender move plus an extra multicast round.

### Graduated (each the one unqualified path, code it replaced deleted, no carve-outs)
| step | 8x512 / 16x256 / 32x128 / 8x384 / 4x512 / 8x256 / 2x512 / 4x256 (ns) | golden |
|---|---|---|
| before (Perf 1 kernels, same session) | 94,929 / 95,119 / 94,613 / 71,973 / 55,133 / 54,953 / 37,032 / 36,654 | 210/210 |
| + fused-MOP matmul (helper K-loop replaced by raw `block_k_loop`, `last_k_row` fused) | 91,825 / 92,125 / 91,719 / 69,776 / 53,401 / 53,090 / 37,217 / 36,467 | 210/210 |
| + rotary srcA reuse + v convert DEST windows (4-call `tile_src_mul` deleted) | 90,713 / 90,966 / 90,503 / 68,961 / 52,761 / 52,372 / 36,664 / 36,262 | 210/210 |
| + x round 0 behind the bias spans | 89,963 / 89,609 / 89,574 / 67,659 / 52,267 / 51,707 / 36,707 / 36,119 | 210/210 |

Whole op, focus 8x512: 94.9 -> 90.0 us (-5.2%; target 90.0). 16x256 and 32x128 under their 90 us target; 8x384 (68 us target) at 67.7; 4x512 / 8x256 (46) and 2x512 / 4x256 (28) still above. Unit suite 25/25. x in DRAM after: 8x512 94,276, 4x512 53,668, 2x512 37,063.

### Guard set (`test_qkv_proj_heads_rope_perf_guard.py`, same session, before -> after ns, all correct)
| row | before | after |
|---|---|---|
| 8x512 all L1 | 94,812 | 89,522 |
| 2x512 | 36,971 | 36,648 |
| 7x511 h_non_aligned | 87,379 | 82,238 |
| 13x300 h_non_aligned | 95,550 | 90,912 |
| 8x512 x in DRAM | 97,867 | 94,075 |
| 8x512 tables + outputs in DRAM | 137,725 | 132,853 |
| 1x32 (R1) | 22,339 | 22,581 |
| 1x128 | 28,064 | 27,907 |
| 2x37 | 28,139 | 28,358 |
| 16x8 | 34,576 | 34,519 |
| 4096x1 multi-block, all DRAM | 4,323,452 | 4,244,288 |
1x32 (+1.1%) and 2x37 (+0.8%) are inside the ~2% noise band that the subagents measured on those shapes (1x32 moved +1.9% / -1.8% depending on run order). No material regression. SUPPORTED unchanged.

### Bottleneck now (8x512, 90.1 us, zones)
- The K-loop is at ~49.6 cycles per product (math zone 4.0 -> 68.9 us), close to the 48-cycle roofline.
- The fill is bias barrier (3.0 us at 108 readers) + seed (math until 4.0, pack about 3 us after the bias).
- Last K-block + epilogue 17.8 us.
- Stores end 88-90 us.
- Next candidates: a seed that does not occupy the pack thread before K-block 0 (with bias multicast); the M-row imbalance (11 vs 10.67 rows per core); upstreaming the fused matmul MOP into LLK so `matmul_block` can be used again.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| matmul_block (via ckernel::matmul_block / llk_math_matmul) | capability | The math half issues a DEST target offset write plus one MOP run per tile product (~4.4 cycles per product regardless of fidelity). No helper or API mode runs a call's ct_dim x rt_dim products as one MOP with the DEST stride in an address mod. A raw K-loop with the stock LLK is flat vs the helper (95,271 vs 94,919), so the helper itself costs nothing. The fix belongs in the LLK matmul MOP, and the helper would inherit it. | 94,919 | 92,045 | kernels/mm_fused_mop.hpp:99, qkv_proj_heads_rope_compute.cpp (`block_k_loop`, `last_k_row`) |
| eltwise_chain BinaryFpu / mul_tiles (extends the Perf 1 `tile_src_mul` bypass) | capability | Every product unpacks both operands, so there is no way to keep srcA resident across two products with different srcB. The rotary epilogue is unpack-bound (one fidelity phase measured the same), so each fp32 partial tile was unpacked twice. | 94,880 (Perf 1 raw tile_src_mul; mul_tiles not re-measured) | 94,109 | kernels/tile_src_mul.hpp (`tile_src_mul_pair`), qkv_proj_heads_rope_compute.cpp (`epilogue_row`) |

## Perf 3
- Date: 2026-11-02. Device: Blackhole p100a (108 of 110 cores on the 12x9 rows), AICLK 1350 MHz. Metric: DEVICE KERNEL DURATION [ns], one fresh-cache run per point. Third perf round on this op (tournament round 1 of this run; `## Perf 1` / `## Perf 2` are earlier rounds).
- Focus: the flagged LOOSE_CASES 4x512, 8x256 (target 46 us), 2x512, 4x256 (target 28 us), exact config (x / cos / sin / q / k / v in L1, W / bias in DRAM, bf16 TILE interleaved, tables cut to S, default compute config HiFi3 + fp32_dest_acc_en + half-sync DEST + approx off). All in SUPPORTED. Guard rows 8x512 / 16x256 / 32x128 / 8x384 must not regress. Precision knobs untouched.
- Instrumentation: the Perf 1/2 zones cover every stage and were reused, with none added (the graduated tail keeps `*_store_wait_first` / `*_store` on the unit loops). Tools added in `perf_experiments/_breakdown/`: `ablation_hooks_perf3.patch` (`ABL_NO_{STORE,XREAD,WREAD,TABLE,BIAS,MCAST,EPI,SEED,MATMUL}` via `QKV_ABLATE`), `ablate_rows.sh`, `percore.py` (per-core zone sums / ends), `test_placement_rows.py` (`QKV_ROWS="BxS:x:tables:outputs,..."`).

### Measured breakdown (before)
| point (cumulative peel) | 4x512 | 8x256 | 2x512 | 4x256 |
|---|---|---|---|---|
| full | 51,661 | 51,594 | 36,441 | 36,165 |
| stores stubbed | 50,519 | 50,407 | 35,384 | 34,991 |
| + input DM stubbed (x / W / table / bias reads, both multicasts) | 46,439 | 46,449 | 23,793 | 23,743 |
| + epilogue math stubbed | 45,645 | 45,619 | 23,489 | 23,452 |
| + bias seed stubbed | 45,447 | 45,444 | 23,375 | 23,307 |
| + matmul math stubbed (pack / sync floor) | 14,501 | 14,502 | 7,898 | 8,050 |
| compute stubbed (DM alone) | 44,502 | 43,418 | 34,365 | 34,005 |
| multicasts stubbed alone | 51,387 | 51,486 | 28,258 | 28,313 |
| W reads stubbed alone | 51,695 | 51,107 | 28,818 | 28,513 |
Table, bias, x reads, epilogue and seed each stubbed alone: flat (within 0.5 us) on all four rows.

Ranked:
1. 2x512 / 4x256: the R2 W stream binds (DM alone 34.4 vs compute alone 23.4 us). N-group 8's sender took 3.5 us per round against 2.3 elsewhere, and its 12 cores ended last (36.7 vs 33-35 us).
2. 4x512 / 8x256: the K-loop runs at the HiFi3 roofline on the 6-row cores, and Mt=64 over 12 M-groups puts 6 rows on them against a mean of 5.33. Structural, ~4.7 us.
3. The fill before K-block 0's first pack, 3.5-4.9 us, bound by bias, W round 0 and x round 0 together.
4. The tail: epilogue ~0.8 us exposed and the store ~1.1 us.

The matmul payload was not targeted: it is at the HiFi3 FPU roofline (~49.6 cycles per product).

### Portfolio and verdicts (each subagent's fork vs same-session baseline)
| idea | verdict | before -> after | domain |
|---|---|---|---|
| w_stream: R2 W stream. Menu: layout rule "fewest N-group rectangles as Mcast lowers them" 2x512 36.7k -> 34.1-34.4k, 4x256 36.0-36.2k -> 33.8-34.0k; + W split over 2 senders (median 32.7k, noisy, flat elsewhere); + sender's reader reads half of each round on NoC0 (33.1k, but 8x512 / 4x512 / 8x384 / 7x511 +3-4% and inexpressible at 1x32); 4 senders 45.8k; leftover core as sender 49.8k; relay 62.3k; read-ahead during multicast 37.3k | WIN (layout rule), bit-identical | 2x512 36,705 -> 34,436; 4x256 36,217 -> 33,824 | everywhere, host-only. On 11x10 it changes only the 12x9 grid |
| fill_v3: shorter fill. Menu: LATESEED (K0 L1-acc off, bias after K0) 53.1k at 4x512 / 36.7k at 2x512; ragged 2:2:4.. 52.8k / 39.8k; BSEND bias channel 52.8-59.8k (stall in one run); BSEND + ragged 51.8k / 41.9k; DEST seed 53.0-53.7k; weight depth 3 + ramps 52.4-57.2k | NULL | 4x512 52,093 -> best 51,840 (inside noise), 2x512 +8-14% for the ragged variants | not graduated: the fill is bound by the W stream (3.1 us DRAM read + 2.2 us multicast per round). Separate bias delivery is worth ~0.9 us only once W is faster |
| m_balance: mixed-h_c core assignment at Mt=64 (best busiest core 44 tiles vs 48). Plans A_mc 74.7k, A2 99.7k, B2 78.1k, B2 + per-rectangle x senders 70.2k (4x512) / 59.9k (8x256) | REGRESSION | 4x512 52,433 -> 70,198 at best | not graduated. Compute alone does drop 46.5 -> 43.2 us, but DM binds: the reader's sin sign flip (`negate_bf16_tile`) costs ~3.2 us per table row on NCRISC (39.8 us on 11-row h_c=2 cores), and the h_c=2 x (88 KB rounds) and 4-core W streams are slow. Mt=32 / 96 / 128 have no better grid |
| tail_v3: the block's last row as two head-aligned units for the last-K matmul, every unit stored half on the writer (NoC1) and half on the reader (NoC0). Menu: halves only 51.3-51.8k; units only flat; + lastlead0 flat; LAST_K_LEAD 2 52.6k; lean inits 51.1-51.6k (flat, raw LLK, not taken); batched epilogues 52.4-54.2k; one NoC 57.7k | WIN, bit-identical | 4x512 52,113 -> 51,406 (medians of 3); 8x384 67,681 -> 66,744 | everywhere it ran correct (13 cases incl. HiFi4 / LoFi / full sync / 1 and 3 heads per core). Coordinator found two exceptions at graduation (below) |

### Graduated
| step | 8x512 / 16x256 / 32x128 / 8x384 / 4x512 / 8x256 / 2x512 / 4x256 (ns) | golden |
|---|---|---|
| before | 89,864 / 90,232 / 89,959 / 67,626 / 52,364 / 51,905 / 37,008 / 35,833 | 210/210 |
| + layout rule | 89,934 / 90,216 / 90,111 / 68,474 / 52,006 / 51,854 / 34,361 / 33,766 | - |
| + tail units (final, with both carve-outs) | 90,007 / 90,263 / 89,552 / 67,319 / 51,398 / 51,307 / 33,653 / 33,570 | 210/210 |

- Layout rule: `_lowered_rectangles` counts a group's rectangles the way `Mcast` lowers it. `_choose_layout` now takes the qualifying layout whose largest N-group lowers to the fewest rectangles, instead of the first that qualifies, so 12x9 on 11x10 now uses the transposed fold. The old first-qualifies rule is deleted, with no carve-out.
- Tail units (`kernels/tail_units.hpp`; `store_row` became `store_tiles`; `last_k_tiles` / `epilogue_tiles` / `epilogue_unit_routed` in compute): the whole-row-alternation code is deleted, and `split_units` = false selects whole rows alternating inside the same unit loops. Two carve-outs:
  - **Outputs in DRAM** (`split_units` = false, host): a measured regression with the split. Before -> layout -> layout + split, two runs each:
    - 8x512: 133.0 -> 134.5 -> 136.3 us
    - 4x512: 71.5 -> 71.4 -> 74.0 us
    - 2x512: 42.6 -> 41.5 -> 44.9 us
    - 4096x1 all DRAM: 4.24 -> 4.18 -> 4.37 ms

    Tables in DRAM are not affected (8x512 / 2x512 tables DRAM: flat / -1.6 us), so the gate is on the output buffer only.
  - **3 or 6 heads per core** (no last-row split): incorrect. A CB ring wraps only on an exact `fifo_limit`, and odd half-head counts mix push sizes so a push straddles the end. Example: 6 heads, 4 rows, 6 + 6 + 6 + 4 + 4 on a 24-tile ring. The forced runs passed only because the overrun landed on consumed or free L1. 1, 2, 4 and 12 heads keep the split. A forced h_c 1/2/3/6/12 sweep on 9x256 / 4x256 / 2x37 passes.

Whole op, focus rows: 4x512 52.4 -> 51.4 us (-1.8%), 8x256 51.9 -> 51.3 (-1.2%), 2x512 37.0 -> 33.7 (-9.1%), 4x256 35.8 -> 33.6 (-6.3%). All four are still above target_ns (46 / 46 / 28 / 28). 8x512 / 16x256 / 32x128 / 8x384 are flat. Unit suite 25/25, golden 210/210.

### Guard set (`test_qkv_proj_heads_rope_perf_guard.py`, same session, before -> after ns, all correct)
| row | before | after |
|---|---|---|
| 8x512 all L1 | 89,727 | 89,864 |
| 2x512 | 36,743 | 33,608 |
| 7x511 h_non_aligned | 82,216 | 81,890 |
| 13x300 h_non_aligned | 90,296 | 90,160 |
| 8x512 x in DRAM | 94,465 | 93,866 |
| 8x512 tables + outputs in DRAM | 132,630 (132,199 / 132,496 in reruns) | 134,108 |
| 1x32 (R1) | 23,328 | 22,470 |
| 1x128 | 28,003 | 27,644 |
| 2x37 | 28,224 | 27,850 |
| 16x8 | 34,410 | 31,929 |
| 4096x1 multi-block, all DRAM | 4,237,167 | 4,203,573 |

Placement rows (`test_placement_rows.py`, two runs each, before -> after): 8x512 out DRAM 132,821 / 133,303 -> 134,078 / 134,353; 8x512 tables DRAM 89,999 / 89,961 -> 89,996 / 89,795; 4x512 out DRAM 71,288 / 71,671 -> 70,609 / 71,833; 2x512 out DRAM 42,781 / 42,475 -> 41,657 / 41,439; 2x512 tables DRAM 36,127 / 36,207 -> 34,320 / 34,808.

The layout rule's cost is +1.1% (~1.4 us, consistent across 4 runs) on 8x512 with outputs in DRAM, against -7% on the production 2x512. It is under the ~2% band treated as noise in earlier rounds, so the rule stays one path. That cell is the place to look if it grows. SUPPORTED unchanged.

### Bottleneck now
- 2x512 / 4x256: the W stream. The sender's 2-rectangle send (~2.3 us) plus its NoC1 read issue (~1.3 us) is ~3.6 us per round, against ~2.5-3.5 us of compute per K-block. A w_stream micro-benchmark puts NoC1 W reads from 9 senders at ~3.1 us per round, NoC0 at 2.3, half on each at 1.75. Moving W reads onto NoC0 collides with the x / bias reads (measured +3-4% on the larger rows).
- 4x512 / 8x256: M imbalance (6 vs 5.33 rows on the FPU roofline). Any rebalancing needs cheaper DM per row first. The reader's per-row sin sign flip (~3.2 us of NCRISC per row) is the first blocker; candidates are table multicast per M-group or a sign-free rotary product.

### Helper bypasses — none
No raw-LLK path was added. The layout rule is host-only. The tail units reuse the Perf 1/2 raw paths (`mm_fused_mop.hpp` last-K, `tile_src_mul_pair`), whose rows above still apply. The lean-init variant that bypassed `mul_init` / `matmul_block_init` measured flat and was not taken.
