# Changelog: nomic_moe_route_dispatch

## Phase 0 - Core Implementation
- **Date**: 2026-11-04
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1 `nd_push_l1`: one `generic_op` program on the 11x10 grid (router matmul + softmax + top-2 + count exchange + row pushes + in-place tilize).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], in_placement=[nd_norm], out_buffer_type=[L1]
- **Accuracy achieved**: combine_weights PCC > 0.9999, max_abs_err 0.80e-3..1.85e-3, rel RMS 1.14e-3..1.38e-3, rerouted 0.07%..0.54% (5 shape/profile cases via test_nomic_moe_route_dispatch_precision_baseline.py); dispatched / combine_index / expert_counts exact
- **Golden suite at Phase 0**: 21 supported_pass, 82 xfail_expected, 0 loud (per `verifier_report.json`); 41 regression tests pass standalone
- **Perf at Phase 0 (device us, warm, model L1 occupancy)**: 8x512 58.7, 6x512 31.4, 8x384 31.9, 4x512 29.5, 8x256 32.3, 2x512 24.7, 4x256 22.2 (slower than stock on 8x512, 8x256, 2x512)
- **Issues encountered**: verifier fixes: slot-page header and handshake page sizes routed to the kernels as CT args with static_asserts (DRY); dead `MAX_PAD_ROWS` removed; L1 ledger total / symbol table rewritten to the built inventory. Harness: 20 regression tests fail under `eval_test_runner.sh` L1 capture (no device fixture / no allocation), pass standalone.
- **Tests added**: test_nomic_moe_route_dispatch.py (planner), test_nomic_moe_route_dispatch_precision_baseline.py, test_nomic_moe_route_dispatch_perf.py

## Refinement 1 - DRAM-interleaved `dispatched` (output-side scheme change)
- Date: 2026-11-05
- What was done: `out_buffer_type = DRAM` added to SUPPORTED (regime R4). Reused: R1 pushes, owner map (round robin `shard % 110`, as balanced as a row-wise `split_work_to_cores`, so push addressing stays shared), `cb_gathered_rows` staging, the compute `tilize` call. Added: `Plan.out_dram`; `CB_DISPATCHED` becomes a 2-block streaming CB (`DRAM_OUT_BLOCKS_IN_FLIGHT * w` tiles) instead of the sharded output CB; the writer RISC drains each owned block to DRAM through a TensorAccessor (w tile writes, one flush a block) on every core; `_l1_room` reserves no output bytes for DRAM; `circular_buffer_bytes(..., buffer_type=)` and `cb_bytes` include the streaming CB.
- Accuracy achieved: dispatched / combine_index / expert_counts exact; combine_weights max|w - p64| 1.6e-3..1.9e-3 (<= 3e-3) on 1x1024, 2x512 and the remaining tile-aligned nd_norm shapes
- Golden test progress: 14/14 tile-aligned nd_norm DRAM cells pass (ragged and interleaved-input DRAM cells wait on Refinement 2); L1 spot checks (8x512, 2x512, 1x3744_spare1) and test_regression.py 41/41 pass
- Perf (device us, DRAM out): 8x512 101.7, 6x512 66.1, 8x384 69.9, 4x512 55.5, 8x256 52.8, 2x512 35.2, 4x256 34.5; DRAM-write bandwidth bound after the landing barrier. L1 rows before/after within noise (8x512 57.9 -> 58.7, 2x512 23.9 -> 20.7).
- Issues encountered: None
- Tests added: test_nomic_moe_route_dispatch_perf.py::test_perf_row_dram_out

## Refinement 2 - Interleaved input placements + ragged token counts
- Date: 2026-11-05
- What was done: `l1_interleaved`, `dram_interleaved` (regime R2) and `ragged` (R3) added to SUPPORTED. Reused: the whole R1 pipeline with G as a parameter (G = 1: one-member groups, the self-only partial exchange, W multicast of the single 24-tile slice, k push writes per row), count table, pushes, R4 output side. Added: `Plan.x_interleaved` / `Plan.group_ranges` (`_interleaved_ranges`: row-wise `split_work_to_cores` of the T_pad/32 tile rows; nd_norm ranges clamped to the real tokens); a real `cb_x_tiles` of `rps*24` pages filled by one TensorAccessor burst (barrier shared with the W slice read on the sender); a G = 1 logits chain in compute; the reader/writer token split is now a per-core RT arg (`reader_tokens_of`, 16-token granule) so single-tile-row cores still push on both RISCs; tile-padding slot words are zeroed and written to combine_index by the RISC holding the core's last tokens; combine_weights tiles use ceil(core_tokens/32).
- Accuracy achieved: dispatched / combine_index / expert_counts exact; combine_weights within 3e-3 of fp64 on 8x512, 2x512, 1x1000, 1x3000 (interleaved and nd_norm)
- Golden test progress: 68/68 interleaved/ragged cells + 3 prior-passing nd_norm representatives pass (71/71 in the slice)
- Perf (device us, interleaved x, L1 out, model occupancy): 8x512 55.9 (dram) / 50.0 (l1), 2x512 29.0 / 25.8; production nd_norm rows unchanged within the +-3 us run-to-run noise (A/B vs the pre-change commit).
- Issues encountered: None
- Tests added: test_nomic_moe_route_dispatch.py::test_interleaved_and_ragged (5 cases), test_nomic_moe_route_dispatch_perf.py::test_perf_row_interleaved_in

## Refinement 3 - Speed up the row-push phase on the primary perf shape
- Date: 2026-11-05
- What was done: Reused: RowPusher / push_rows_block, the reader/writer token split, the untilize -> cb_x_rows staging, the landing barrier. Added: (1) pad rows spread over every pushing RISC (pad p -> unit p mod 2*num_groups, the unit's j-th pad carries its j mod risc_tokens-th token; was token r mod T, which loaded the first ~2 groups with all pads); (2) whole-piece push (`PUSH_WHOLE_PIECES`, host `_push_layout`): where a slice straddles pieces (G=3,k=2; G=2,k=3) each piece is pushed by the member holding most of it, which NoC-reads the piece's other tiles from its group peers' resident x shards into a new `cb_stage_tiles` and untilizes that range (`stage_tiles` wide); ties split rows by parity; per-RISC piece tables are RT args; 8x512 goes from 4 to 2 writes a row and member 1 pushes nothing; (3) leaner issue path: local issued/ack accounting folded into the barrier counters once (`finish()`), piece loop unrolled over k, pusher local to a noinline push_rows_block, no escaping pad lambda; (4) `PUSH_BYTES_IN_FLIGHT` counted against each RISC's average write. Knobs kept at no-op defaults after measured nulls: `PUSH_STAGGER_TOKENS = 0` (L7, flat), `PUSH_CMD_BUFS = 1` (2 slower), `PUSH_READER_SHARE = 0.5` (0.375 balanced the old layout; under whole pieces 0.5 is best at the 16-token granule). The `tokens` CT arg is gone (pads no longer map by r mod T).
- Accuracy achieved: dispatched / combine_index / expert_counts exact; combine_weights unchanged (selection path untouched), precision baseline 5/5 pass on [2x512-layer0, 4x512-layer0, 8x384-layer3, 8x512-layer0, 8x512-layer5]
- Golden test progress: 21/21 targeted slice (7 loose perf cells + nd_norm 8x512 / 2x512 / ragged / spare, L1 and DRAM out); test_regression.py 41/41; unit suite 16/16
- Perf (device us, model occupancy; before = this run's pre-change measurement): L1 out 8x512 57.7 -> 45.9 (median of 4), 6x512 31.8 -> 31.1/32.5, 8x384 30.2 -> 30.3/30.1, 4x512 29.0 -> 28.2/29.0, 8x256 32.6 -> 28.8/27.3, 2x512 22.8 -> 20.7/18.7, 4x256 22.5 -> 21.4/19.6; DRAM out 8x512 104.1 -> 89.8/92.3, 2x512 34.5 -> 31.7/32.2; interleaved in: A/B against the pre-change commit back to back, equal or faster (8x512 dram 56.3 -> 55.4, l1 53.3 -> 51.2). Slowest pushing RISC on 8x512: ~30k -> ~16.6k cycles.
- Findings: the push is RISC-issue-bound, not credit-bound (bytes in flight 2-16 KB flat): ~35 cycles of software + ~55 of NoC register path a write. Member-0/2 reader (BRISC/NOC0) pushes on the top grid rows run up to ~1.4x the writer's for the same work; swapping the RISCs' NoCs helps L1-out 8x512 (~4 us) but costs DRAM out ~30 us, so not adopted. 8x512 now sits at stock parity; the binding stage is the pre-push path (push starts at ~37-39k of ~60k cycles, +-4 us run to run): Refinement 4. Next push lever: full-row staging on every member with row-round-robin pieces (member 1 idle during the push today), ~5k cycles, blocked on its 96-tile untilize overrunning the W-fetch shadow.
- Issues encountered: the ablation build hung once (ablated writes still counted as issued); fixed before measuring. A first `git add -A` swept harness pipeline_logs into the pre-commit hook; reverted the hook's newline edits, nothing committed.
- Tests added: None (existing unit / golden coverage spans both new push layouts: 8x512, 2x512, spare rows, ragged)

## Refinement 4 - Shorten the pre-push critical path on every perf-flagged row
- Date: 2026-11-05
- What was done: Reused: the W multicast channels, the stage gather and untilize, the softmax chain, select_tokens. Added: (1) W senders read + multicast before anything else; slice c goes out on RISC c % `W_SLICE_NOCS` (2: odd slices on the writer / NOC1, both NoCs carry a slice at once), and the member's slice is received by the RISC whose NoC carries it (one cb_router_weight producer a core); shared `deliver_weight_slice` / `receive_weight_slice` in the push header; (2) whole-piece stage gather moved from the reader (NOC0) to the writer (NOC1); (3) compute order: matmul first, untilize after the softmax (select shadow); with whole-piece push x is published once, otherwise round 1 = matmul, round 2 = untilize; (4) `ExpExpertFaces` chain element (exp_tile in VectorMode::C) + sum scaler over the 8 expert columns + finite -1e30 mask fill instead of -inf; (5) select_tokens with two packed u32 rank counters and a 16-row face stride walk. Kernel zone markers used for the timeline were removed; the compute kernel now also takes the ablation defines.
- Accuracy achieved: dispatched / combine_index / expert_counts exact; combine_weights precision baseline 5/5 pass (expert columns of the exp and sum are unchanged: the dropped columns were exactly 0 before)
- Golden test progress: 49/49 targeted slice (perf cells, 8x512, 2x512, 6x512, 4x256, spare, ragged; every placement, L1 and DRAM out); test_regression.py + unit + precision 62/62
- Perf (device us, model L1 occupancy, median of 3; before -> after (stock)): 8x512 46.0 -> 39.6 (45.6), 6x512 31.0 -> 27.2 (32.6), 8x384 30.8 -> 26.9 (32.6), 4x512 30.5 -> 24.9 (30.3), 8x256 26.9 -> 24.3 (30.1), 2x512 18.9 -> 16.4 (22.6), 4x256 21.1 -> 17.2 (23.5). DRAM out 8x512 85.3, 2x512 29.0 (R3: ~90 / ~32). Interleaved x 8x512 A/B vs the R3 commit: dram 54.2 -> 52.3, l1 51.3 -> 52.4 (noise).
- Findings: full-grid linked multicasts on one NoC serialize (~1.3-2.5k cycles per 32 KB slice); unlinked 1 KB multicasts cost ~550 cycles each, so a faces-only W delivery is slower (reverted). On 8x512 the pre-push path is now W ~6k, partial exchange ~9k, logits chain ~4.7k, max ~0.6k, exp ~2.3k, sum ~1.4k, select ~7.2k, table + scan ~3k; push starts at ~30.4k of ~54k cycles. 2x512 (rps = 1): W ~3.5k, compute chain ~3.1k of fixed per-phase cost, select ~2.3k, table ~2k, scan ~1k, push ~5k. Next lever: member-split selection with a slot/count exchange.
- Issues encountered: the Mcast host adapter requires every kernel a channel is attached to to use the channel's NoC on the sender core, so the NOC1 channels are sent and received by the writer. Restoring files after an A/B run also reverted the harness-written edit to pipeline_logs/refinements/refinement_3_output.json.
- Tests added: None (existing unit / golden coverage spans G = 1, 2, 3 and both push layouts)

## Refinement 5 - Overlap tilize with the pushes (per-owner counted arrival)
- Date: 2026-11-05
- What was done: Reused: RowPusher's piece table (now `PushPiece` + `load_push_pieces`, shared), `push_pad_rows` (as the arrival counter's visitor), the coordinator landing barrier and go channel, the combine_weights write. Added: (1) `ARRIVAL_PER_BLOCK` knob (op file, default False; kernel define): per-owner counted arrival, one semaphore per owned block counting 64 B row granules (complete at 32*w); `ArrivalCounter` / `signal_arrivals` tally each RISC's slot rows (per-expert contiguous ranges) and pads per output tile row and issue one atomic per (tile row, piece) run after the RISC's write barrier; owners publish each block to the tilize as it completes and issue their combine_weights tiles in between. The go channel is attached only for the global barrier (16-semaphore budget). (2) `WEIGHTS_SPREAD` (default True): member c of a group writes combine_weights tiles c, c+G, ... instead of the leader writing all 2*rps; every member zeroes and fills its weight tiles in select_tokens.
- Accuracy achieved: dispatched / combine_index / expert_counts exact; combine_weights unchanged (same values, different writer); precision baseline 5/5 pass
- Golden test progress: 45/45 targeted slice (8x512, 2x512, 4x512, ragged; every placement, L1 and DRAM out) in both arrival modes; test_regression.py 41/41; unit dir 49/49
- Perf (device us, model L1 occupancy, 3 runs, before -> after): 8x512 39.7/39.9/40.6 -> 38.9/38.8/39.2; other rows unchanged within noise; DRAM out 8x512 84.5-87.6 both ways; interleaved x 8x512 dram 52.3 -> 52.4, l1 48.2-50.2 -> 50.0, 2x512 dram 26.5 -> 26.0. `ARRIVAL_PER_BLOCK = True` (with spread): 8x512 39.2-39.5, 6x512 28.0-28.4, 8x384 28.6-28.8, 4x512 25.0-25.7, 8x256 24.8-25.0, 2x512 18.1-18.3, 4x256 18.4-18.5.
- Findings: the 8x512 baseline tail was last push end ~48.3k cycles -> coordinator go ~51.8k -> tilize end ~55.4k. Per-block arrival removes that hop, but Blackhole NoC atomics are non-posted INCR_GET (forced) at ~160-200 cycles each; a RISC issues 13-21 of them (counted with DEVICE_PRINT), 2.8-3.4k cycles after its push, so it only reaches parity at 8x512 and loses 1-2 us on the rps <= 2 rows. Unlinked NoC writes and a later atomic to the same core have no ordering guarantee (noc.h: only linked commands are ordered), so the atomics cannot be issued inside the push. Full-grid multicast atomics (one per RISC instead of per block) measured 160+ us. Alternating the atomics over the write_reg command buffer hung. The DRAM combine_weights writes (1 MB; 8 x 4 KB on each of 32 leaders at 8x512) take 5-10k cycles to inject under push traffic; issuing them from the non-pushing member 1 during the push slowed the pushes and blocked that core's owner publish behind its barrier (measured, dropped); spreading them over the G members is what won.
- Issues encountered: a non-template `if constexpr` discarded branch still evaluates its named CT args, so the go channel's args forced the channel to be attached in per-block mode and overran 16 semaphores at 8x512; the landing mode is now a kernel define.
- Tests added: None (existing unit / golden coverage spans both modes; the parked knob was run on the golden slice)

## Perf 1 - Fan-out perf tournament (round 1 of 2)
- Date: 2026-11-05
- Focus: LOOSE_CASES `8x512_perf`: nd_norm x, `dispatched` in L1, default compute config (HiFi4, fp32 DEST, approx off), model L1 occupancy. Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before)
Whole op 39.2-40.1 us = ~54k cycles. Every kernel now carries permanent `MaybeDeviceZoneScope` stage zones with waits split from work (new `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`); parser and zone dumps are in `perf_experiments/` (`zone_report.py`, `zones_8x512_*.txt`). Critical path in cycles: W lands ~4.7k (slice 2 ~5.8k) -> K=8 matmul ~1.5k -> partial exchange ~2.8k -> logits chain 4.4k -> softmax 4.9k (scores ~19.9k) -> select 7.8k on each DM RISC (64 tokens) -> count table ~2.3k -> assign ~1k -> push 13.6k p50 / 16.5k max -> go barrier + tilize ~5k.
Cumulative ablation: full 40.1 us; push NoC writes stubbed 36.8 us, with the push loop still at 7.2k cycles of RISC software (~55 cycles a write); matmul and eltwise payloads also stubbed 28.6 us (~11k cycles of compute on the path).
Ranked by headroom: push (RISC-issue bound; ~100 KB a RISC against a ~1.6k-cycle link floor), logits + softmax (~9.3k), select (7.8k), W + exchange (~10k to the logits). The `noc_estimate` CLI is not built in this tree, so the DM gate used the 64 B/cycle NoC peak.
G=1 reference (`l1_interleaved` 8x512, 53.5 us): K=24 matmul over M<=2 takes 2.7k, softmax 2.9k, select 2.1k. That regime is slow only because its single W sender needs 26k cycles under the x read traffic.

### Portfolio and verdicts (perf_experiments/<idea>/, isolated benches on the real grid)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `softmax_lean`: logits + softmax fused in one DEST window, raw SFPI, 8 expert columns packed into one vector per 4-row band, rotate-butterfly max/sum | WIN, graduated | 9475 -> 5545 cycles (4 tiles, G=3); 6x512 4501 -> 2484; 2x512 2612 -> 1275; G=1 2282 -> 1087 | every G and rps; max err 2.27e-3 -> 1.8e-7 (exact fp32, no tf32 FPU sources); NaN/Inf in untrusted columns cannot reach experts |
| `select_fast`: select_tokens rewrite | WIN, graduated | 7772 -> 6081 cycles (64 tokens); 32: 4042 -> 3130; 16: 2063 -> 1633 | everywhere, bit-identical. Deferred-store variant (4471 + 3334 flush) needs a new handshake: not taken |
| `w_delivery`: W as two 12-tile halves, one full-grid multicast per NoC | WIN in isolation, measured whole-op REGRESSION, not graduated | isolated: slowest member slice 6611 -> 4846, all landed 6417 -> 4897. Whole op after the two graduations: 34.1/34.2/34.3 -> 35.1/35.3 us (both-halves wait and member-only wait) | members 0/1 lose their early slice (3.3-3.6k -> 4.8k) and every group's partial exchange starts at once (issue 1.4k -> 2.1-2.6k p50). Patch: `perf_experiments/w_delivery/op_integration_half2.patch`. rd_consts reorder: NULL |
| `push_issue`: cheaper per-write issue (dest table, lean register writes, credits, merging, posted) | marginal, not graduated | makespan 13640 -> 12836 (-0.6 us); 2x512 6495 -> 6761 (measured regression) | under full-grid contention the slowest RISC is NoC/ack bound. Merging runs and posted writes lose, and posted writes are not landing-safe. 192 pushing RISCs are 9% slower than 128 |
| `member_units`: each nd_norm member a whole-row token unit (gather peers' slices on NOC1, K=24 local matmul, no partial exchange, selects/pushes only its tokens) | WIN pre-push in isolation, deferred to round 2 | select-done 28.7k -> 20.8k cycles (granule split), 20.5k (tile-row split); 6x512 -24%, 4x512 -30%, 2x512 flat | measured against the pre-`softmax_lean` path. softmax_lean already removes most of the logits/softmax cost this also removes, the count scan grows +1.1k, and the push half is unmeasured (push_issue measured 192 RISCs slower). Needs a whole-op prototype |

### Graduated (commit 371c9b88894)
- Fused SFPU logits + softmax (`kernels/nomic_moe_route_dispatch_softmax8.hpp`), the op's only path for every G / rps / placement. Deleted: the eltwise_chain / reduce / mul sequence, `ExpExpertFaces`, the keep-mask tile and both reduce scalers (reader `rd_consts` 3.8k -> 1.5k cycles), and four intermediate CBs (72 KB a core at rps=4). No carve-outs. `math_approx_mode=True` now gets the accurate fp32 exp (higher precision, never lower).
- `select_tokens` drop-in, everywhere, no carve-outs.
- Ablation hook: `NOMIC_MOE_ROUTE_DISPATCH_ABLATE` also passes the helpers' own `SKIP_COMPUTE` / `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`.
- Whole op 8x512: 40.1 us -> 34.1-34.5 us (scores at ~14.5k cycles instead of ~19.9k, select done at ~20.6k instead of ~28k).

### Guard set (all 28 perf rows, one profiled run each, pre-tournament 3fa479c208f vs HEAD; device us)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 39.0 | 34.5 | -11.6% |
| test_perf_row[6x512_perf] | 27.9 | 26.1 | -6.7% |
| test_perf_row[8x384_perf] | 27.1 | 25.1 | -7.4% |
| test_perf_row[4x512_perf] | 23.8 | 23.1 | -3.1% |
| test_perf_row[8x256_perf] | 24.0 | 21.8 | -9.3% |
| test_perf_row[2x512_perf] | 16.4 | 15.9 | -2.8% |
| test_perf_row[4x256_perf] | 16.5 | 15.9 | -3.3% |
| test_perf_row_dram_out[8x512_perf] | 84.9 | 80.4 | -5.3% |
| test_perf_row_dram_out[6x512_perf] | 63.1 | 61.6 | -2.4% |
| test_perf_row_dram_out[8x384_perf] | 62.7 | 60.7 | -3.1% |
| test_perf_row_dram_out[4x512_perf] | 48.0 | 45.8 | -4.5% |
| test_perf_row_dram_out[8x256_perf] | 47.4 | 46.2 | -2.6% |
| test_perf_row_dram_out[2x512_perf] | 29.2 | 28.6 | -2.1% |
| test_perf_row_dram_out[4x256_perf] | 29.2 | 28.7 | -1.8% |
| test_perf_row_interleaved_in[8x512_perf-in_placement=dram_interleaved] | 54.1 | 50.7 | -6.2% |
| test_perf_row_interleaved_in[8x512_perf-in_placement=l1_interleaved] | 52.6 | 47.5 | -9.7% |
| test_perf_row_interleaved_in[6x512_perf-in_placement=dram_interleaved] | 40.6 | 38.5 | -5.4% |
| test_perf_row_interleaved_in[6x512_perf-in_placement=l1_interleaved] | 39.5 | 34.3 | -13.1% |
| test_perf_row_interleaved_in[8x384_perf-in_placement=dram_interleaved] | 39.6 | 37.7 | -4.9% |
| test_perf_row_interleaved_in[8x384_perf-in_placement=l1_interleaved] | 34.4 | 36.2 | +5.4% |
| test_perf_row_interleaved_in[4x512_perf-in_placement=dram_interleaved] | 34.3 | 31.9 | -7.0% |
| test_perf_row_interleaved_in[4x512_perf-in_placement=l1_interleaved] | 32.4 | 30.6 | -5.6% |
| test_perf_row_interleaved_in[8x256_perf-in_placement=dram_interleaved] | 31.4 | 29.6 | -5.7% |
| test_perf_row_interleaved_in[8x256_perf-in_placement=l1_interleaved] | 28.5 | 28.1 | -1.1% |
| test_perf_row_interleaved_in[2x512_perf-in_placement=dram_interleaved] | 26.4 | 24.3 | -7.9% |
| test_perf_row_interleaved_in[2x512_perf-in_placement=l1_interleaved] | 23.4 | 22.3 | -4.5% |
| test_perf_row_interleaved_in[4x256_perf-in_placement=dram_interleaved] | 25.0 | 22.6 | -9.7% |
| test_perf_row_interleaved_in[4x256_perf-in_placement=l1_interleaved] | 21.5 | 20.7 | -3.4% |

The only positive delta, 8x384 l1_interleaved, re-ran 3x each: HEAD 36.8 / 35.9 / 34.1 vs base 37.4 / 37.3 / 40.3 (median 35.9 vs 37.4, no regression). Golden `eval/golden_tests/nomic_moe_route_dispatch/` 144/144; unit + precision baseline 21/21.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `eltwise_chain` + `reduce<MAX, REDUCE_ROW>` + `reduce<SUM, REDUCE_ROW>` (recip post-op) + `mul` bcast-col | capability | no SFPU row reduction over a column sub-range (8 of 32) that broadcasts the result back across the row, and no way to keep intermediates in DEST across a reduce. The helper path round-trips 4 intermediate CBs and runs max / sum / mul on the FPU with tf32 sources | 7019 (9475 cycles) | 4107 (5545 cycles) | `kernels/nomic_moe_route_dispatch_compute.cpp:131` -> `kernels/nomic_moe_route_dispatch_softmax8.hpp` |

- Next round: member_units as a whole-op prototype on the new path, re-measured breakdown (push is now ~30% of the op, 13.5k of ~46k cycles).

## Perf 2 - Fan-out perf tournament (round 2 of 2)
- Date: 2026-11-05
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off, model L1 occupancy). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD 3f3352e4f3e)
Whole op 34.2-35.4 us (4 runs) = ~46.3k cycles. Zone dumps: `perf_experiments/zones_8x512_perf2_before.txt`, `zones_8x512_perf2_final.txt`. Critical path (cycles from kernel start): W slices land 3.5-5.7k; matmul ends 6.4k p50 / 8.2k max; partials ready 9.7k (exchange 1.4k p50 / 2.7k max); fused softmax 9.7k -> 15.4k; select (both DM RISCs, 64 tokens each, ~94 cycles a token) 15.5k -> 21.5k; count table -> 23.5k; assign ~1k; push rd 13.6k p50 / 17.0k max, wr 12.3k p50 / 13.6k max (reader ends 39-41k on the slowest cores, writer ~36.8k); reader push barrier up to 2.5k on member-0 cores; go ~43.7k; tilize 1.75k -> 46.3k.
Ablation (push NoC writes stubbed, the only payload hook on the push): 46.3k -> 40.3k cycles; the push loop alone is 7.2k cycles of RISC software on both RISCs, and the reader's push barrier then reached 4.4k cycles: `write_combine_index` sat inside the landing barrier, queued behind the other cores' 1 MB of NOC0 `combine_weights` DRAM writes. The pre-push chain is serial behind two global barriers (table, go), so its zones are payload.
Ranked: push (13.6-17k; ~7k software + NoC), redundant softmax + select on all G members (11.6k), tail (reader/writer push imbalance + DRAM writes inside the landing barrier, ~2.5-4k), matmul end -> partials ready (3.3-4.4k), W landing (~4k), tilize (1.75k, ~29 cycles a tile, no headroom). `noc_estimate` is not built in this tree; the DM gate used the 64 B/cycle link peak (push ~4.4k-cycle torus link floor for 12.9 MB vs 13.6k measured).

### Portfolio and verdicts (perf_experiments/r2_<idea>/)
| idea | verdict | baseline -> candidate | domain / notes |
|---|---|---|---|
| `r2_tail_land`: combine_index after the landing increment; combine_weights tiles alternated writer (NOC1, first) / reader (NOC0) after each RISC lands | WIN, graduated | 8x512 34462 -> 32630 ns (integration copy, median of 3); push start -> op end 22.2k -> 19.3k cycles | all 28 perf rows, no exceptions. Sub-options measured: combine_index move alone flat (34356); weights after go 36761 and before push 36459 (regressions); all weights on writer 33687; coordinator at core 104 / 49 / 60 flat; reader push share 0.375 regression (34156) |
| `r2_front_lean`: W faces 0/2 only, every slice of a NoC in ONE multicast (`wmerge`); partial exchange split east on NOC0 (reader) / west on NOC1 (writer), compute packs straight into its own `cb_partials` region (`xsplit`) | WIN (combined only), graduated | 8x512 34801 -> 33681 ns (median of 3); partials ready max 9825 -> 8297 cycles | everywhere measured (G = 1/2/3, rps 1/2/4, interleaved x), bit-identical outputs. Nulls: `wfaces` alone flat/slower (35.2-35.9k; NOC0 multicasts serialize per command), `xsplit` alone flat (~35.0k), `wmerge` alone 35647 (exchange congests at 2.8k), `xtiles` 37210, `xpull` 35006, `xmcast` 44953 and `xfmcast` 43583 (some groups stall 10-16k) |
| `r2_sfpu_top2`: top-2 on the SFPU inside softmax8 (one code tile a score tile: code, w1, w2), RISC decode by ctz, code tiles published one at a time | WIN, graduated (`codeA` + per-tile publish) | bench softmax -> select end, rps 4 / G 3: 12455 -> 9836 cycles; rps 2 / G 2: 6551 -> 5796; G 1: 3483 -> 3257; ragged 100 tokens: 12252 -> 8755 | everywhere, bit-identical slots / weights / counts incl. exact first/second/three-way ties, underflow ties, NaN/Inf in columns 8..31. Menu, not graduated: `codeA_perm` 9228 (needs W columns permuted on the host), `codeB` 9669 (packer writes combine_weights; +256 cycles a tile), `codeB_nz` 9490 (padding columns not zeroed: not bit-identical), `codeA_cnt` 9136 (counts-only select, changes the slot-page contract, +0.4k on the push walk) |
| `r2_member_units`: each nd_norm member a whole-row token unit (peer slice gather on NOC1, K = 24 matmul, softmax + select on its own tokens only, count row per unit, dynamic two-ended push split) | WIN rps >= 2, measured REGRESSION rps = 1; not graduated | whole op vs the op of its session: 8x512 34.4 -> 30.7, 6x512 26.1 -> 24.1, 8x384 25.5 -> 23.9, 4x512 22.0 -> 20.8, 8x256 22.00 -> 21.31; 2x512 15.82 -> 16.44 (+3.9%), 4x256 15.98 -> 16.40 (+2.6%); G = 1 dram_interleaved 8x512 51.4 -> 54.1 (x read moved to NOC1) | Conflicts with `r2_front_lean` (it deletes the partial exchange) and would need an rps = 1 carve-out plus a second path. The no-carve-out set (tail + front + top-2) measured 29.2-29.6 us on 8x512, faster than units alone (30.4-31.0). Units alone gave only -4%; the rest came from its push-split changes. Prototype in `r2_member_units/mu_op/` |
| `r2_path_aware_push`: per-write shortest NoC (dynamic NoC mode, per-RISC transaction-id credits); RISC re-split by owner | REGRESSION | push makespan 8x512 13329 -> 17367 cycles, 2x512 6233 -> 8107; 6x512 12972 -> 10860 (win), 4x512 flat | dynamic-mode mechanics cost +2.9k before any path choice (dynown 16236), and the shorter path itself is flat: a write plus its ack loop the whole one-way ring, so credit turnaround does not depend on the destination. Static owner split unbalanced (227 vs 35 writes), runtime split doubles the walk (software floor 6.8k -> 18.0k) |

### Graduated
- `r2_tail_land` (commit 2d03527f230): the op's only path; no carve-outs. The writer now ends with a write barrier (it issues DRAM writes after its landing increment). Counted-arrival mode (`ARRIVAL_PER_BLOCK`, parked) keeps its own order.
- `r2_front_lean` (commit 6f17778c094): the op's only path; deleted the full-tile W delivery, the per-slice multicast channel at G = 3 (one semaphore freed) and the `cb_own_partial` tile staging (`rps` x 4 KB -> 16 B handshakes). W CB 96 KB -> 48 KB. No carve-outs.
- `r2_sfpu_top2` codeA (commit cead3d9e458): the op's only path; deleted the RISC `top2_scan` and the all-tiles scores wait. softmax8 writes the code tile; `select_tokens` decodes it and waits per tile (the writer peeks at `cb_scores` without popping; the reader stays its consumer). The bench's two-tokens-an-iteration decode was kept (one token an iteration measured 63 vs 46 cycles a token in the op). No carve-outs.
- Whole op 8x512: 34.2-35.4 us -> 29.2-29.6 us (-15%). Path at the end (cycles): partials ready 7.6k; softmax + top-2 -> code tiles 9.5k-15.7k; select ends 13.2k (reader) / 17.2k (writer); table 19.2k; push 20.3k -> 32.9k p50 / 35.1k max; go ~37.2k; tilize ends ~39.6k.
- Golden `eval/golden_tests/nomic_moe_route_dispatch/` 144/144 after every graduation; unit + precision baseline 21/21; unit suite under `--dev` 16/16.

### Guard set (all 28 perf rows, one profiled run each, pre-tournament 3f3352e4f3e vs HEAD cead3d9e458; device us)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 34.96 | 29.16 | -16.6% |
| test_perf_row[6x512_perf] | 26.39 | 24.22 | -8.2% |
| test_perf_row[8x384_perf] | 25.41 | 23.58 | -7.2% |
| test_perf_row[4x512_perf] | 23.62 | 20.37 | -13.7% |
| test_perf_row[8x256_perf] | 21.84 | 20.89 | -4.3% |
| test_perf_row[2x512_perf] | 15.85 | 13.96 | -11.9% |
| test_perf_row[4x256_perf] | 15.94 | 14.30 | -10.3% |
| test_perf_row_dram_out[8x512_perf] | 82.00 | 75.06 | -8.5% |
| test_perf_row_dram_out[6x512_perf] | 61.72 | 58.91 | -4.6% |
| test_perf_row_dram_out[8x384_perf] | 61.04 | 58.74 | -3.8% |
| test_perf_row_dram_out[4x512_perf] | 45.64 | 44.23 | -3.1% |
| test_perf_row_dram_out[8x256_perf] | 45.58 | 44.53 | -2.3% |
| test_perf_row_dram_out[2x512_perf] | 28.69 | 26.43 | -7.9% |
| test_perf_row_dram_out[4x256_perf] | 28.93 | 26.80 | -7.4% |
| test_perf_row_interleaved_in[8x512_perf-in_placement=dram_interleaved] | 50.69 | 48.25 | -4.8% |
| test_perf_row_interleaved_in[8x512_perf-in_placement=l1_interleaved] | 52.01 | 47.54 | -8.6% |
| test_perf_row_interleaved_in[6x512_perf-in_placement=dram_interleaved] | 38.43 | 36.13 | -6.0% |
| test_perf_row_interleaved_in[6x512_perf-in_placement=l1_interleaved] | 39.05 | 36.18 | -7.3% |
| test_perf_row_interleaved_in[8x384_perf-in_placement=dram_interleaved] | 38.14 | 36.24 | -5.0% |
| test_perf_row_interleaved_in[8x384_perf-in_placement=l1_interleaved] | 36.79 | 33.69 | -8.4% |
| test_perf_row_interleaved_in[4x512_perf-in_placement=dram_interleaved] | 31.40 | 29.63 | -5.7% |
| test_perf_row_interleaved_in[4x512_perf-in_placement=l1_interleaved] | 30.83 | 25.37 | -17.7% |
| test_perf_row_interleaved_in[8x256_perf-in_placement=dram_interleaved] | 29.47 | 27.22 | -7.6% |
| test_perf_row_interleaved_in[8x256_perf-in_placement=l1_interleaved] | 26.84 | 25.42 | -5.3% |
| test_perf_row_interleaved_in[2x512_perf-in_placement=dram_interleaved] | 24.12 | 22.18 | -8.0% |
| test_perf_row_interleaved_in[2x512_perf-in_placement=l1_interleaved] | 22.31 | 19.93 | -10.7% |
| test_perf_row_interleaved_in[4x256_perf-in_placement=dram_interleaved] | 22.88 | 20.88 | -8.7% |
| test_perf_row_interleaved_in[4x256_perf-in_placement=l1_interleaved] | 21.20 | 18.74 | -11.6% |

No row regressed. Every production row (`test_perf_row`) is below its `stock_ns` (8x512: 29.2 vs 45.6 us) and still above its `target_ns` (5.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `matmul_block` (`ckl::matmul_block`) | capability | in1 tiles cannot be addressed through a caller index map (the faces-only W layout puts tile t of a slice at a non-linear 1 KB-page index, `w_face_tile`), and the output cannot be packed into a region of a CB another thread produces (`cb_partials` is the reader's CB; the helper reserves / pushes its own output CB) | 941 (1271 cycles, TRISC_0 `cmp_matmul`, full-tile W) | 942 (1272 cycles) | `kernels/nomic_moe_route_dispatch_compute.cpp:79` |
| no helper covers this (SFPU row reduction family: top-k / argmax over a lane range) | capability | no per-row top-k over a column sub-range that returns indices, and no way to keep softmax intermediates in DEST across it; the only alternative is RISC scalar selection over packed scores | 9226 (previous raw softmax8 + RISC select, bench softmax -> select end 12455 cycles) | 7286 (9836 cycles) | `kernels/nomic_moe_route_dispatch_softmax8.hpp:82` (top-2 after the softmax in `band()`) |

- Not graduated, recorded as options: `codeA_perm` (-0.6k more, needs a host-side W column permutation), `codeA_cnt` (changes the slot-page contract).
- Leads for a next round: (1) the push is now ~35% of the op (12.5k p50 / 14.6k max a RISC; ~7k of it software); (2) the reader/writer select split is token-contiguous, so the writer waits for code tiles 2-3 and the reader then idles 4.1k cycles in `rd_wait_writer_counts` (a per-tile interleaved split would end select ~1.5k earlier); (3) `member_units` on top of the graduated top-2 / tail for rps >= 2, if a single path that does not regress rps = 1 can be found.

## Perf R3.1 - Fan-out perf tournament (continuation round 1 of 3)
- Date: 2026-11-05
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off, model L1 occupancy). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD 60a745c9f3d)
Whole op 29.5 / 29.7 us. Zone dumps: `perf_experiments/zones_8x512_r31_before.txt`, `zones_8x512_r31_ablate_push_writes.txt`, `zones_8x512_r31_arrival_per_block.txt`, `zones_8x512_r31_final.txt`. Critical path (cycles from kernel start): W lands 3.8k; partials ready 7.6k; fused softmax + top-2 runs on every G member, 7.6k -> 15.5k (~2k cycles a tile, 4 tiles); select ends 13.2k (reader) / 17.2k (writer), and the reader idles 4.1k in `rd_wait_writer_counts`; table 19.2k; assign ends 20.2k; push 20.3k -> 33.0k p50 / 35.1k max; go 37.6k; tilize ends 39.6k.
Ablation: push NoC writes stubbed 24.1 / 23.9 us. The push loop is still 7.1-7.3k cycles of RISC software (~131 writes a RISC, ~55 cycles a write). Turning on `ARRIVAL_PER_BLOCK` gave 29.9 / 29.3 us (flat).
Ranked: push (12.6-14.6k), softmax + top-2 (7.9k, redundant on G members), select tail + table + assign (17.2k -> 20.3k, plus the writer trailing by 4k), tail (push end -> tilize end 4.4k), W (3.8k). The DM gate used the 64 B/cycle link peak because `noc_estimate` is not built in this tree; it puts the push floor near 4.4k cycles.

### Portfolio and verdicts (perf_experiments/r3_<idea>/; whole-op ns on integration copies)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r3_softmax_split`: the partials are reduce-scattered per 16-token half; each member runs softmax8 + top-2 on ~1/G of the halves, one face per pass, ranked in the order the two select RISCs consume them; each code face is streamed to the group with a landed-pass flag write (same NoC / cmd buf / VC, no barrier) | WIN, graduated | 29.0-29.4 -> 26.45 us; softmax end 15.5k -> 11.1k, push start 20.3k -> 16.9k | every G > 1 row -2..-9%; G = 1 has no group and keeps the per-tile path (flat). Bulk-exchange variant `rs` 27.8-28.3 us |
| `r3_select_split`: `ilv` (select split per 16-token granule), `lean` (select stores no weights; weights filled from the code rows in the table wait), `early` (count row sent from packed totals), `fastscan` (branch-free table scan) | WIN, graduated as `lean + early + fastscan` (ported onto the split softmax in `r3_select_split_on_rs2`) | on the pre-split op 29518 -> 28211; on top of the split 27130 -> 25815 (medians of 3) | everywhere, bit-identical; 4x512 flat. `ilv` dropped: under the split softmax the RISCs already get alternating halves, nothing left. Nulls: `tbl1` flat, table scan split over both RISCs slower (30131), all-gather of count rows by 32 full-grid multicasts broken (103 us), `nobar` 27673 but relies on undocumented write/atomic ordering |
| `r3_softmax_diet`: band instruction diet (one SFPU start/done per tile, constants in LREGs, fewer shuffles), same exact fp32 | WIN in isolation, NULL on top of the split: not graduated | isolated 184 -> 161 cycles a band, 29566 -> 28654 ns on the pre-split op; on top of the split (`r3_softmax_diet_on_rs2`, plus one-face copy / pack) softmax window 3763 -> 3238 cycles, whole op 26785 -> 27100 (median of 3, noise) | after the split the softmax is off the critical path (select pace sets it). Bit-identical, golden 144/144; ready in `r3_softmax_diet_on_rs2/op/` if the softmax comes back onto the path |
| `r3_run_push`: stage source rows grouped by expert (local self-copies) and push each expert run with one write per (tile row, piece, chunk) | REGRESSION | best 32489 / 33718 ns | the grouping costs 4.4k cycles a RISC. Run writes make the slowest RISC worse under full-grid contention (16-19k vs 13.7-14.4k), even at 768 B chunks in expert order. Counted arrival does not rescue it. Measured slower on every regime (+2..+11%) |
| `r3_push_lean`: per-expert destination cursors (no divide, no common-arg load per write), lighter register writes, work stealing between RISCs, bytes-in-flight sweep | NULL (8x512), REGRESSION 2x512 | cursor 29266 -> 29158; stealing 29.7-31.7 us; 2-16 KB in flight 30.5-36.5 us | 2x512 cursor +4.9%. Issue software drops ~15 cycles a write, but the slowest RISC is NoC/ack bound |

### Graduated
- `r3_softmax_split` (commit 3881681b310) is the op's only path for G > 1. It deleted the redundant per-member softmax over all tiles and the all-to-all partial-face exchange (replaced by the reduce-scatter). It adds `cb_code_stage` (ceil(2 rps / G) fp32 tiles, 12 KB at 8x512) and the 176 B `cb_code_flags`. G = 1 keeps the per-tile publish because it has no group to split over, not because of a carve-out.
- `r3_select_split` `lean + early + fastscan` (commit e1c4a5cdadc) is the only path everywhere. `select_tokens` no longer loads or stores weights, and the select handshake lost its weights word. New zones: `rd_weights_fill` / `wr_weights_fill`.
- Whole op 8x512: 29.5 / 29.7 us -> 25.0-26.1 us (four runs: 25.8, 25.0, 26.1, 25.0). Path at the end, in cycles: softmax ends 11.1k; select ends 12.4k (reader) / 12.7k (writer); table 14.3k; assign ends 15.2k; push 15.3k -> 28.0k p50 / 29.7k max; go 31.3k; tilize ends 33.2k. The push is now ~40% of the op.
- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144 after each graduation. Unit + precision baseline: 21/21.

### Guard set (all 28 perf rows, one profiled run each, pre-tournament 60a745c9f3d vs HEAD; device ns)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 29177 | 25758 | -11.7% |
| test_perf_row[6x512_perf] | 24183 | 22316 | -7.7% |
| test_perf_row[8x384_perf] | 23444 | 21080 | -10.1% |
| test_perf_row[4x512_perf] | 20648 | 19313 | -6.5% |
| test_perf_row[8x256_perf] | 20389 | 18980 | -6.9% |
| test_perf_row[2x512_perf] | 14229 | 13279 | -6.7% |
| test_perf_row[4x256_perf] | 13991 | 13127 | -6.2% |
| test_perf_row_dram_out[8x512_perf] | 75646 | 71724 | -5.2% |
| test_perf_row_dram_out[6x512_perf] | 58550 | 57004 | -2.6% |
| test_perf_row_dram_out[8x384_perf] | 58036 | 55604 | -4.2% |
| test_perf_row_dram_out[4x512_perf] | 43910 | 42064 | -4.2% |
| test_perf_row_dram_out[8x256_perf] | 43638 | 42053 | -3.6% |
| test_perf_row_dram_out[2x512_perf] | 26952 | 25799 | -4.3% |
| test_perf_row_dram_out[4x256_perf] | 26810 | 25881 | -3.5% |
| interleaved_in[8x512 dram_interleaved] | 49121 | 48228 | -1.8% |
| interleaved_in[8x512 l1_interleaved] | 47278 | 44019 | -6.9% |
| interleaved_in[6x512 dram_interleaved] | 36256 | 35857 | -1.1% |
| interleaved_in[6x512 l1_interleaved] | 34722 | 34965 | +0.7% |
| interleaved_in[8x384 dram_interleaved] | 36518 | 35638 | -2.4% |
| interleaved_in[8x384 l1_interleaved] | 35714 | 31630 | -11.4% |
| interleaved_in[4x512 dram_interleaved] | 29468 | 28841 | -2.1% |
| interleaved_in[4x512 l1_interleaved] | 28139 | 26220 | -6.8% |
| interleaved_in[8x256 dram_interleaved] | 27153 | 26511 | -2.4% |
| interleaved_in[8x256 l1_interleaved] | 22794 | 23859 | +4.7% |
| interleaved_in[2x512 dram_interleaved] | 22558 | 22135 | -1.9% |
| interleaved_in[2x512 l1_interleaved] | 20087 | 19603 | -2.4% |
| interleaved_in[4x256 dram_interleaved] | 20884 | 20183 | -3.4% |
| interleaved_in[4x256 l1_interleaved] | 18492 | 17753 | -4.0% |

I re-ran the two positive rows. 8x256 l1_interleaved, 3 runs each: HEAD 26509 / 27134 / 26191 vs base 28428 / 27268 / 28945, so HEAD is faster. 6x512 l1_interleaved, 9 samples each (the first 4 in separate sessions, then 5 alternating head / base): HEAD median 34093 (31291-35645) vs base median 34722 (31207-36938). Neither row regressed. Every production row is below its `stock_ns` (8x512: 25.8 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `_llk_math_eltwise_unary_sfpu_params_` (VectorMode) | capability | no VectorMode runs face 0 alone or face 2 alone, so a pass that owns one 16-token half would run the SFPU over both faces | 29.5-29.7 us (whole op, VectorMode::C on every member's 4 tiles; isolated per-face cost not separately measured) | 26.75-27.08 us (whole op with the split) | `kernels/nomic_moe_route_dispatch_softmax8.hpp:130` (`share_faces`) |
| no helper covers this (multicast / collective family: streamed intra-group all-gather) | capability | the mcast helper is one-to-many and barriers each send; nothing sends a producer's pages to a peer set as they are pushed, with a per-sender landed counter that the consumer can poll per page | not available | 26.75-27.08 us (whole op) | `kernels/nomic_moe_route_dispatch_push.hpp:210` (`CodePump`), `:282` (`wait_code_half`) |

- Not graduated, recorded as options: the softmax diet + one-face copy/pack (`r3_softmax_diet_on_rs2/op/`, bit-identical, -525 cycles of softmax window, whole op flat); `nobar` (-0.5 us, depends on write/atomic delivery order).
- Leads for the next round: (1) the push, 12.6-14.2k cycles of a 33.3k-cycle kernel. Its slowest RISC is NoC/ack bound under full-grid contention, and fewer larger writes, faster issue and work stealing all measured flat or worse, so the lever now is less traffic per link or a better schedule of destinations, not issue cost. (2) The select, now the head of the chain once the first half lands (~37 cycles a token). (3) The tail: go 31.3k -> tilize end 33.2k.

## Perf R3.2 - Fan-out perf tournament (continuation round 2 of 3)
- Date: 2026-11-05
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off, model L1 occupancy). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD 2697cf6e4de)
Whole op 25.3 / 25.4 / 25.8 us (subagent same-session median 25.57 us over 16 calls). New permanent zones `rd_land_wait` / `rd_go_send` on the coordinator. Zone dumps: `perf_experiments/zones_8x512_r32_before.txt`, `zones_8x512_r32_ablate_PUSH_WRITES.txt`, `zones_8x512_r32_ablate_PUSH.txt`, `zones_8x512_r32_final.txt`. Path in cycles: W lands 3.8k; matmul math ends 5.4k, pack 6.3k; partials ready 7.4k; softmax 7.4k -> 10.2k; select ends 12.3k (reader) / 12.6k (writer); table 13.7k; assign ends 15.1k; push 15.2k -> 28.0k p50 / 29.8k max; last landing atomic issued 30.3k, coordinator sees the count 31.2k, go lands 31.85k; tilize 32.0k -> 33.7k; kernel end 34.2k.
Ablation (cumulative): push writes stubbed 20.5 / 20.3 us (push loop 7.1-7.3k cycles of software); whole push stubbed 15.0 / 15.0 us. With the push stubbed, the last landing atomic left at 15.5k but the coordinator saw the full count at 16.9k and go landed at 17.6k: ~1.5-1.9k cycles of barrier overhead independent of the push.
Ranked: push 13-14.5k (NoC bound, see `r32_push_floor`), W + matmul + exchange 7.4k, softmax + select 5.2k, table + assign 2.6k, landing barrier + go 1.5-1.9k, tilize 1.75k (fast tilize already). `noc_estimate` is not built in this tree; the push gate is the measured replay floor below.

### Portfolio and verdicts (perf_experiments/r32_<idea>/; whole-op ns on integration copies, same session as the op)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r32_owner_land`: posted pushes; each owner counts its landed push writes on both NIUs' `NIU_SLV_POSTED_WR_REQ_RECEIVED` (32 per owned block) and tilizes when its own rows are in; no landing atomics, no go | WIN, graduated | median 25566 -> 24546 (16 calls each, 4 sessions) | everywhere: 2x512 -5.8%, 4x512 -4.6%, 6x512 -6.4%, DRAM out -2.4..-2.5%, dram_interleaved -1.5%. Options: posted without the in-flight cap ~25.4k (pushers end 1-2k later); cap + old go 24.7-25.2k (posted writes alone cost nothing). Counter safety: race probe (`race_probe/`) 0 stale words in 160k samples on REQ_RECEIVED vs up to 100% stale on REQ_STARTED; a `--dev` exact-count assert held on 29 cases; 300 calls x 8 configs bit-exact |
| `r32_w_early`: both RISCs of a W sender core issue its DRAM reads (half the K tile pairs each, own NoC), local semaphore handoff, one multicast as before | WIN, graduated | median 25490 -> ~24880; `cmp_wait_w` end 3.92k -> 3.32k | everywhere; flat at G = 1 (W off the path). Nulls in the same dir: K halves per NoC with early matmul (flat), 4 senders / two concurrent multicasts a NoC (up to 34 us, unsafe), chunked read -> multicast pipelining (26010, slower: shared sender NIU), deferring the stage gather (25324, collides with the exchange), byte balance across NoCs alone (flat) |
| `r32_count_chain`: coordinator-side prefix table, flag-written count rows (no barrier / atomic), writer-sent row, coordinator on core 0, byte-packed rows | NULL / REGRESSION | copy 25305 -> flag 25741, flag + prefix 25722, coord 0 25358, writer-sent flat-to-slower, bytes slower | flag polling costs ~20 cycles a row on the coordinator; its serial prefix pass ends ~1k after the last row (every row toucher pays ~1k); 2x512 +3.5%, G = 1 +1.2% |
| `r32_half_mm`: 32x16 in1 / output router matmul (`is_in1_32x16` LLK path, half the MVMULs, half the pack bytes), bit-identical | NULL | 25478 -> ~25170 (inside noise) | zones unmoved (math end 5.45k, pack end 6.3k in all variants): the stage is not FPU or pack bound (likely the bf16 srcB unpack, not confirmed). Would also drop firmware dynamic throttling and halve `cb_partials` (24 KB); not graduated |
| `r32_push_floor`: replay of the push's real write list with a near-zero-software kernel, then a schedule search | NULL (gate) | replay floor 13.2-13.5k makespan vs ~15.0k for the op's push; issue stubbed 3.8-4.0k; perfectly balanced synthetic 12.2k | the push is ~90% of its NoC floor. Reorders flat (shuffle -6% in one replay). Only per-write NoC rebalancing within a core moves the replay (-16..-21%), but the run-time handoff in the op costs +35..70 cycles a write: whole op 25345 -> 30284 (+19.5%), 2x512 +23%, 6x512 -3.5% |
| `r32_unit_select`: selection split over the 2G DM RISCs of a group with slot / count all-gather (v1..v4), lean count row, softmax diet | REGRESSION (split), NULL (diet) | v1 +0.9%, v2 +2.7%, v3 +1.2..+5.2%, v4 +14.7%, lean row +7.9%; diet alone 25430 -> 25152 pooled (-1.1%, noise) | every split adds more DM-RISC work (rank rebase, extra writes, polling) than it removes; diet 2x512 +2.8% (3 of 3 runs). Found the `wr_weights_fill` race below |

### Graduated
- `r32_owner_land` (commit 042e85559bb): the op's only landing path. Pushes are posted writes capped on `NIU_MST_POSTED_WR_REQ_SENT` (`RowPusher`); the reader snapshots both NIUs' posted-write slave counters at kernel start and publishes its blocks at `+32 * num_blocks` (`rd_wait_land`); each DM RISC ends with a posted-write flush. Deleted: the push write barriers, the landing atomics, the coordinator go multicast and its channel, `sem_arrived`, and the parked `ARRIVAL_PER_BLOCK` mode (`ArrivalCounter`, `signal_arrivals`, its per-block semaphores). The `ABLATE_PUSH*` hooks skip the land wait. No carve-outs (DRAM `dispatched` lands in L1 before its tilize the same way).
- `r32_w_early` split reads (commit d20bf60e695): the only W delivery path; the split is half the channel's K tile pairs, computed in the kernel (no host knob). Adds `SEM_W_LOCAL` (one semaphore). Stays on `McastArgs` (no bypass). No carve-outs.
- Correctness fix in the same commit as owner_land: the writer's combine_weights fill (`wr_weights_fill`, G > 1) read the code faces of the reader's halves without waiting for them (introduced in R3.1). HEAD failed 5/16 unit tests under `--dev`; the writer now waits on every half's flag first (`wr_wait_all_code`, ~100 cycles, off the path). Second hazard reported by `r32_unit_select` (`RowPusher::start` snapshotting the non-posted ack counter with writes in flight) is gone: the pusher now counts posted requests sent, and no other posted write precedes it.
- Whole op 8x512: 25.3-25.8 us -> 23.6-24.5 us (last calls 23637, 24487, 24486, 23636, 23785). Path at the end, in cycles: W lands 3.37k; matmul pack ends 5.8k; select ends 11.8k / 12.1k; table 13.2k; assign ends 14.6k; push 14.7k -> 26.6k p50 / 29.4k max; land wait ends 27.8k p50 / 29.8k max; tilize ends 29.5k p50 / 31.6k max; kernel end ~32.1k.
- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144 after each graduation, also under `--dev`. Unit + precision baseline 21/21; unit suite under `--dev` 16/16.

### Guard set (all 28 perf rows, one profiled run each, pre-round 2697cf6e4de vs HEAD d20bf60e695; device ns)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 25056 | 23594 | -5.8% |
| test_perf_row[6x512_perf] | 22121 | 20732 | -6.3% |
| test_perf_row[8x384_perf] | 21291 | 20575 | -3.4% |
| test_perf_row[4x512_perf] | 19586 | 17693 | -9.7% |
| test_perf_row[8x256_perf] | 19021 | 17110 | -10.0% |
| test_perf_row[2x512_perf] | 13262 | 12297 | -7.3% |
| test_perf_row[4x256_perf] | 13400 | 12377 | -7.6% |
| test_perf_row_dram_out[8x512_perf] | 71121 | 69549 | -2.2% |
| test_perf_row_dram_out[6x512_perf] | 57369 | 53781 | -6.3% |
| test_perf_row_dram_out[8x384_perf] | 56916 | 52064 | -8.5% |
| test_perf_row_dram_out[4x512_perf] | 42179 | 39533 | -6.3% |
| test_perf_row_dram_out[8x256_perf] | 43045 | 39438 | -8.4% |
| test_perf_row_dram_out[2x512_perf] | 26179 | 24909 | -4.9% |
| test_perf_row_dram_out[4x256_perf] | 26285 | 24833 | -5.5% |
| interleaved_in[8x512 dram_interleaved] | 47658 | 46867 | -1.7% |
| interleaved_in[8x512 l1_interleaved] | 48399 | 45775 | -5.4% |
| interleaved_in[6x512 dram_interleaved] | 36156 | 35089 | -3.0% |
| interleaved_in[6x512 l1_interleaved] | 33913 | 33903 | 0.0% |
| interleaved_in[8x384 dram_interleaved] | 35372 | 34626 | -2.1% |
| interleaved_in[8x384 l1_interleaved] | 35213 | 31593 | -10.3% |
| interleaved_in[4x512 dram_interleaved] | 28777 | 28301 | -1.7% |
| interleaved_in[4x512 l1_interleaved] | 27378 | 26366 | -3.7% |
| interleaved_in[8x256 dram_interleaved] | 26682 | 25869 | -3.0% |
| interleaved_in[8x256 l1_interleaved] | 24975 | 22425 | -10.2% |
| interleaved_in[2x512 dram_interleaved] | 21422 | 21731 | +1.4% |
| interleaved_in[2x512 l1_interleaved] | 19521 | 19597 | +0.4% |
| interleaved_in[4x256 dram_interleaved] | 20427 | 19635 | -3.9% |
| interleaved_in[4x256 l1_interleaved] | 17844 | 16706 | -6.4% |

I re-ran the two positive rows, 3 alternating runs each. 2x512 dram_interleaved: base 22141 / 21490 / 22118, HEAD 21915 / 21796 / 22009 (median -0.9%). 2x512 l1_interleaved: base 19632 / 18810 / 19149, HEAD 19347 / 19210 / 19447 (median +1.0%, ranges overlap). Both flat, so neither earns a carve-out. Every production row is below its `stock_ns` (8x512: 23.6 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (NoC sync family: owner-side arrival detection; replaces the `ttnn.Mcast` / `McastArgs` go signal + `Semaphore::up` landing atomics) | capability | nothing in kernel_lib reads the NIU slave counters (`NIU_SLV_POSTED_WR_REQ_RECEIVED`), so a receiver cannot count landed writes without the senders signalling it; the only helper-expressible landing proof is sender-side (write barrier + atomic + a coordinator multicast), which costs ~1.5-1.9k cycles after the last landing on 8x512 | 25566 (whole op, median of 16) | 24546 (whole op, median of 16) | `kernels/nomic_moe_route_dispatch_push.hpp:39` (`owner_land_count`), `kernels/nomic_moe_route_dispatch_reader.cpp:511` (`rd_wait_land`) |

- Recorded as options, not graduated: `r32_half_mm` 32x16 matmul (bit-identical, flat, saves 24 KB of `cb_partials` at 8x512, drops dynamic throttling for that matmul); the softmax diet (flat; 2x512 +2.8%).
- Leads for the next round: (1) the push is ~90% of its replay floor; the only lever left there is moving writes to the NoC with fewer hops per write, and it pays only if the RISC assignment is free (e.g. decided by the select) since a per-write run-time decision costs more than it saves. (2) The tail is now the slowest pusher plus one owner's full tilize (~2.3k); the NIU counter is per core, so per-block publishing would need another signal. (3) W land -> partials ready (3.4k -> 7.4k) is not FPU or pack bound (`r32_half_mm`); the srcB unpack of the 32 bf16 x tiles and the partial exchange are the unmeasured candidates. (4) G = 1 count-row straggler: one core sends its row at 39.1k vs the median 25.5k on 8x512 dram_interleaved.

## Perf R3.3 - Fan-out perf tournament (continuation round 3 of 3)
- Date: 2026-11-05
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off, model L1 occupancy). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD c278c3d0de8)
Whole op 24.1 us (warm call); 9-call medians 23.6 / 24.0 us. Zone dumps: `perf_experiments/zones_8x512_r33_before.txt`, `zones_8x512_r33_ablate_push_writes.txt`, `zones_8x512_r33_ablate_push.txt`, `zones_8x512_r33_final.txt`; scripts in `perf_experiments/r33_breakdown/` (`prof.sh` one profiled row + zones, `many.sh` 10-call medians, `guard.sh` all 28 rows).
Path in cycles (p50): W lands 3.3k; matmul math ends 4.94k, pack 5.79k; partials ready 6.83k; softmax ends 10.5k; select ends 11.75k / 12.07k; count rows sent 12.62k; table lands 13.64k; assign 13.67k -> 14.50k (821 cycles); push 14.63k -> 26.4k p50 / 29.5k max; tilize ends 32.1k max.
Ablation (cumulative): push writes stubbed 19.5 us; whole push stubbed 14.7 us. `SKIP_COMPUTE` does not stub this op's raw matmul / softmax, so the compute half was not ablated whole-op (the subagents ablated it locally, below).
Per core: on 8x512 (G = 3, k = 2) the 32 member-1 cores push only pads (push ends ~17.0k); the slowest pushers are the reader RISCs of (2,3) and (2,6) (29.3-29.5k vs <= 28.6k elsewhere).
Ranked: push 12-15k (~90% of its R3.2 replay floor, NoC bound), select end -> push start 2.56k (serial, not NoC bound), math end -> partials ready 1.9k (serial), softmax + select 5.2k (tried twice before), tail ~2.5-3k. `noc_estimate` is not built in this tree; the push gate is the R3.2 replay floor.

### Portfolio and verdicts (perf_experiments/r33_<idea>/; whole-op ns against the production op in the same session)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r33_short_noc_push`: each pushing RISC issues each write on the NoC with fewer torus hops (cross-NoC writes on the other NIU's cmd buf 2, per-NIU in-flight accounting shared by both RISCs), re-floated because pushes are posted since R3.2 | REGRESSION | own-NoC control 24090 -> short 24653 / 24530 (`cap=0`); `cap=1` 25570; `short8` 24907 | mean hops 13.1 -> 8.6 and 49% of writes move NoC, but push p50 +0.4..0.7k cycles; the shared-NIU bookkeeping alone costs 0.7-1.1k. 2x512 +6.1%. Also incorrect under the shared cmd-buf contract on interleaved-input rows (the W-sender / coordinator reader uses all 4 cmd bufs) |
| `r33_member_share`: give the idle member 1 of G = 3, k = 2 a token share of the push (deferred stage gather) | REGRESSION | control 24121 / 23758 -> token share 25217 / 24956, piece share 26015 / 26069 | issue ends 2-4k earlier but owners land 1.5-2k later: the push is NoC-delivery bound with 64 pushing cores. Only 8x512 has an idle member (2x512 / 4x256 are G = 2, k = 3). DRAM out flat |
| `r33_table_assign`: `posted` count rows counted on the coordinator's NIU slave counter; `wself` writer computes its own bases from the landed table; `split` table scan over both RISCs; `filllate` combine_weights fill after the push. Nulls: `scan4` (16 loads in flight) flat, byte-lane rows slower | WIN, graduated as `posted + wself + split + filllate` | 6-call medians 23760 -> 22587; table lands 13.64k -> 13.35k, push start 14.63k -> 14.12k | everywhere: 6x512 -2.7%, 2x512 / DRAM out / G = 1 inputs flat (G = 1 assign 1.8k -> 1.2k cycles). Scan is L1 load-latency bound (~4.5 cycles a load). The ~480-cycle table multicast latency was not attacked |
| `r33_partials_window`: matmul one partial tile at a time in the softmax's order; writer sends each tile to every peer with a landed-tile count write (no barrier, no atomic); reader publishes a tile when every peer's count reached it | WIN, graduated (`WRITER_SENDS + FLAG_WRITE + PUMP_ALL`) | median ~24.2k -> ~23.3k; partials ready 6.74k -> 6.32k, select end 12.08k -> 10.99k, push start 14.64k -> 13.63k | everywhere; rps = 1 and G = 1 rows flat (nothing to pipeline). Ablations: the 850-cycle "pack tail" is the FPU draining HiFi4 MVMULs (pack itself ~150 cycles); the exchange costs ~600 cycles on the path. Nulls: reader-sent per-tile with barrier + atomic (no gain), exchange pumped inside the select waits (slower) |

### Graduated
- `r33_partials_window` (commit 9f5ed0bf4f5) is the only exchange path: `TileOrder` (w_layout.hpp), per-tile `matmul_block` (rt_dim = 1, same K accumulation order, bit-identical), `send_tile_partials` from the writer to every peer, per-contributor landed-tile semaphores `sem_partials0..2` polled by the reader (`rd_wait_peers`). Deleted: `send_share_partials`, the reader's east-side send, both write barriers and the `Semaphore::up` per peer, and the dead `send_partial_faces`. `cb_own_partial` / `cb_own_partial_west` grow from 1 to `rps` 16 B pages. G = 1 has no peer to send to; the writer only drains its handshake pages (not a carve-out). 8x512: 9-call medians 23989 / 23576 (base) -> 22693 / 22981.
- `r33_table_assign` posted + wself + split + filllate (commit fa474e615b1) is the only table path. Leaders post their count row (no barrier, no atomic) and the coordinator waits on `owner_land_count()` for them, advancing `land_base` so the push landing count stays exact; a leader flushes its posted writes after the table, before `RowPusher` snapshots. The reader publishes the slot page before the table wait; a selecting writer scans the upper half of the rows (`scan_rows_packed` / `combine_scans`). Deleted: `SEM_COUNTS`, the per-row semaphore, the reader-computed writer bases (`kHeaderWriterBase`), the old full-table lambda. A reader whose writer holds no tokens scans the whole table itself (no partner RISC; not a carve-out). 8x512: 9-call medians 22.7-23.0 -> 21563 / 21569.
- Whole op 8x512: 23.6-24.0 us -> 21.6 us (9-call medians). Path at the end (single profiled call, `zones_8x512_r33_final.txt`): pack ends 5.3k; select ends 10.5k / 10.8k; table lands 12.2k; assign ends 12.9k; push 12.9k -> 24.7k p50 / 28.3k max; tilize ends 31.0k max. The slowest pushers still set the end.
- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144 after each graduation, also under `--dev`. Unit + precision baseline 21/21 (also under `--dev` after the second graduation).
- Instrumentation: `rd_wait_peers` now covers the per-tile peer waits, `wr_wait_own_partial` / `wr_xchg_send` run per tile, `cmp_wait_partials` covers the first tile only (later tiles are waited inside `cmp_softmax`), new `rd_table_rows` (coordinator row wait), `wr_wait_table`, `wr_assign`; `rd_weights_fill` / `wr_weights_fill` moved after the push.

### Guard set (all 28 perf rows, one profiled run each, pre-round c278c3d0de8 vs HEAD fa474e615b1; device ns, last call)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 23623 | 21788 | -7.8% |
| test_perf_row[6x512_perf] | 20837 | 19681 | -5.5% |
| test_perf_row[8x384_perf] | 20683 | 19787 | -4.3% |
| test_perf_row[4x512_perf] | 17207 | 15420 | -10.4% |
| test_perf_row[8x256_perf] | 16733 | 15739 | -5.9% |
| test_perf_row[2x512_perf] | 12237 | 11748 | -4.0% |
| test_perf_row[4x256_perf] | 12513 | 11796 | -5.7% |
| test_perf_row_dram_out[8x512_perf] | 69292 | 68232 | -1.5% |
| test_perf_row_dram_out[6x512_perf] | 53771 | 52563 | -2.2% |
| test_perf_row_dram_out[8x384_perf] | 52446 | 51554 | -1.7% |
| test_perf_row_dram_out[4x512_perf] | 39233 | 38616 | -1.6% |
| test_perf_row_dram_out[8x256_perf] | 39673 | 38889 | -2.0% |
| test_perf_row_dram_out[2x512_perf] | 24921 | 24018 | -3.6% |
| test_perf_row_dram_out[4x256_perf] | 24841 | 23970 | -3.5% |
| interleaved_in[8x512 dram_interleaved] | 46865 | 46203 | -1.4% |
| interleaved_in[8x512 l1_interleaved] | 43934 | 44743 | +1.8% |
| interleaved_in[6x512 dram_interleaved] | 35359 | 33895 | -4.1% |
| interleaved_in[6x512 l1_interleaved] | 28644 | 32210 | +12.4% |
| interleaved_in[8x384 dram_interleaved] | 34870 | 33860 | -2.9% |
| interleaved_in[8x384 l1_interleaved] | 33913 | 33264 | -1.9% |
| interleaved_in[4x512 dram_interleaved] | 27572 | 27464 | -0.4% |
| interleaved_in[4x512 l1_interleaved] | 25617 | 26111 | +1.9% |
| interleaved_in[8x256 dram_interleaved] | 25816 | 25177 | -2.5% |
| interleaved_in[8x256 l1_interleaved] | 20527 | 22461 | +9.4% |
| interleaved_in[2x512 dram_interleaved] | 22067 | 21936 | -0.6% |
| interleaved_in[2x512 l1_interleaved] | 19326 | 19187 | -0.7% |
| interleaved_in[4x256 dram_interleaved] | 19816 | 19483 | -1.7% |
| interleaved_in[4x256 l1_interleaved] | 16880 | 16556 | -1.9% |

The four l1_interleaved rows that read positive were re-measured with 9-call medians in alternating base / HEAD sessions (this placement's W delivery swings by several us call to call): 6x512 base 33769 / 34189 vs HEAD 32593 / 33122; 8x256 base 23070 / 23849 vs HEAD 22962 / 23308; 4x512 base 25681 vs HEAD 25531; 8x512 base 43030 / 44260 / 44221 vs HEAD 43943 / 44025 / 43930. All flat or faster, so no carve-out was earned. Every production row stays below its `stock_ns` (8x512: 21.6 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `Semaphore::up` / `Semaphore::wait` (count-row arrival on the coordinator) | capability | no kernel_lib helper counts posted writes landed on a receiver; the semaphore path needs a write barrier plus a non-posted atomic per row on every leader, and a separate semaphore | 23760 (whole op, 6-call median, unchanged copy 24016) | 22587 (whole op with posted + wself + split + filllate; `posted` alone 23402) | `kernels/nomic_moe_route_dispatch_reader.cpp:210` (`rd_table_rows`), count-row send in `rd_count_send` |
| no helper covers this (multicast / collective family: streamed intra-group reduce-scatter with a per-sender landed counter) | capability | `ttnn.Mcast` / `McastArgs` is one-to-many and barriers each send; nothing sends a producer's tiles to a peer set as compute packs them, with an in-order count write per peer that the consumer polls per tile | not available (previous raw path with barrier + `Semaphore::up`: ~24.2k median) | ~23.3k median (`r33_partials_window`) | `kernels/nomic_moe_route_dispatch_push.hpp:261` (`send_tile_partials`), reader `rd_wait_peers` |

- The per-tile router matmul still calls `ckernel::matmul_block` directly for the same reason as before (compact W tile indexing, output packed into a CB this thread does not produce); not a new bypass.
- Not graduated, recorded as options: none with a precision trade (all variants were bit-exact at the fixed HiFi4 / fp32 DEST contract).
- Observation from `r33_partials_window`: one `--dev` session of the pre-round production op hung once on 1x1000 with the subagent's harness seed (compute waiting for partials, both DM RISCs waiting for the own-partial signal; `perf_experiments/r33_partials_window/log_dev4.txt.gz`); two re-runs passed. That exchange code is now deleted, and the new path passed golden 144/144 twice (once under `--dev`), but the cause was not found.
- Leads: (1) the push is still ~12-15k cycles and NoC-delivery bound; shortest-NoC routing and more pushing cores both measured slower, so what is left is fewer bytes on the links (none found) or a destination schedule that the R3.2 reorders did not cover. (2) The slowest pushers (readers at (2,3), (2,6)) end 1-2k after the rest; nobody has explained why those two. (3) The ~480-cycle table multicast latency and the ~1.6k softmax -> select chain remain on the pre-push path.

All ideas measured: 2 graduated, 2 regressions (not graduated); 8x512 23.6-24.0 -> 21.6 us; no measured regression on the 28-row guard set.
