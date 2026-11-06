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

## Perf R4.1 - Fan-out perf tournament (continuation round 1 of 3)
- Date: 2026-11-06
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off, model L1 occupancy). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD e2633f6861a)
Whole op 9-call medians 21557 / 21651 ns. Zone dumps: `perf_experiments/zones_8x512_r41_before.txt`, `zones_8x512_r41_ablate_push_writes.txt`, `zones_8x512_r41_ablate_push_pieces.txt`, `zones_8x512_r41_ablate_push.txt`, `zones_8x512_r41_final.txt`. Scripts and per-core CSVs are in `perf_experiments/r41_breakdown/` (`prof.sh`, `many.sh`, `guard.sh`, `percore.py`).
Path in cycles (p50): select ends 10.5k / 10.85k; table lands 12.30k / 12.37k; assign ends 13.0k / 13.1k; push 13.05k -> 25.0k p50 / 28.3k max; tilize ends 31.0k max.
Cumulative push ablation, whole op / pushing-RISC push zone:

| stubbed | whole op | push zone |
|---|---|---|
| nothing | 23.0 us | 12.0k p50 / 14.4k max |
| issue only (`ABLATE_PUSH_WRITES`) | 17.7 us | 7.2k, every pushing RISC within 7.1-7.4k |
| issue and per-piece address math, slot walk kept (`ABLATE_PUSH_PIECES`, new) | 15.9 us | walk 2.43k (3.17k on leaders) |
| whole push (`ABLATE_PUSH`) | 13.8 us | |

On 8x512 every pushing RISC issues 131-132 writes of 768 B.

**The brief's contradiction, resolved.** The stall probe (`ABLATE_MEASURE_STALLS`, new: `DeviceTimestampedData` from `RowPusher`) shows zero credit-cap stalls on every pushing RISC and 0.2-1.5k cycles of command-buffer-ready stalls, so the push does not wait on its own in-flight cap. It costs about 7.2k cycles of software (walk ~2.4k plus address math and credits ~4.7k, ~36 cycles a write) and ~4.8k of NoC register issue.

Even so, the push is not issue bound in the sense that cutting issue shortens it. Three subagents made issue 1.7-4.7k faster per RISC (below), and each time the slowest pushers and the owners' last landing moved *later*. The software rate is pacing injection below the point where delivery becomes unfair. The landing floor (~13.2k after push start, the R3.2 replay floor) is where the push sits today.

**The stragglers.** The per-RISC push spread (11.7-14.4k) correlates 0.97-0.98 with command-buffer-ready stall time, and it comes in one burst between writes 48 and 64 (`r41_push_balance`). That burst lines up with the idle member's combine_weights DRAM writes (G = 3: member 1 pushes nothing and writes its 3 x 4 KB tiles a core mid-push). Those writes shared the static unicast VC with the pushes, so a 4 KB write held at a DRAM endpoint blocked the pushes queued behind it. Which RISCs straggle depends on the session; (2,3) / (2,6) were not special.

Ranked: push 12-15k (paced at its delivery floor; stragglers set by head-of-line blocking), select end -> push start 2.5k (table multicast ~650, assign ~680), tail 2.7k. W arrival (3.3k) and softmax were not re-floated (tried in R3.1 / R3.2).

### Portfolio and verdicts (`perf_experiments/r41_<idea>/`; whole-op ns against the unchanged op in the same session)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r41_push_walk`: per-stream destination cache (no division or common-arg L1 load per write), pusher state in registers, piece loop specialized on the host's max pieces per member | REGRESSION | 21613 / 21726 -> 23877 / 23791; on top of the two graduations below: 20868 / 20857 -> 23127 / 22915 | ~131 -> 64 instructions a token. rd_push p50 11.95k -> 7.3k but max 12.9k -> 14.3k, wr max 13.8k -> 16.8k. 4x512 +4.4%, DRAM out -2.6%, small shapes flat. In-flight caps of 768-3072 B and stagger 16 / 40 did not recover the baseline |
| `r41_push_issue`: length set once, coordinate write skipped when unchanged, host pre-encoded coordinate register values, credit cap removed | NULL | best (`coord_pre+len_once`) 21698 -> 21521, inside the noise band; `no_cap` 23151 / 23379; `coord_pre+no_cap` 24011 | a skipped register store costs more (compare + branch) than the store; only 14 of 131 writes repeat a coordinate. The cap never stalls, yet removing it moves the slowest pusher +2.4-3.5k: it acts as a pacer |
| `r41_push_fused`: run-ordered push (each expert's rows as one contiguous run), coordinate per tile row, 3-4 register stores a write; `cursor` (token order, per-expert cursors) | REGRESSION | 21599-21813 -> fused 26578, cursor 24246-24470; cursor with min-issue-interval pacing 23192-26399 | owners' last landing 27.9k -> 30.8k (cursor) / 32.9k (fused). 2x512 +5.2%, 4x256 +9.2% (cursor) |
| `r41_push_balance`: explain the stragglers; combine_weights DRAM writes on static VC 2 | WIN, graduated (with a DRAM-out carve-out) | 21.65-21.79 us -> 21.17-21.19 us; push max 13.5k -> 12.6k | 6x512 -2.5%, 6x512 DRAM out -2.0%, 2x512 / 4x256 / dram_interleaved flat. Nulls / regressions: pushes on VC 2 instead (same gain), idle member skipping its walk 22.2-22.6 us, idle member's weights before the table 23.0 us, after its land wait 24.1 us, all weights on NOC0 26.0 us, idle member writing the group's weights 22.3-27.4 us |
| `r41_table_mcast`: count table as one posted multicast without path reservation, detected on each receiver's NIU0 posted-write slave counter | WIN, graduated | 21661 / 21572 -> 21348 / 21346; table lands 12.14k -> 11.75k, push start 12.88k -> 12.44k | everywhere: 2x512 -3.2%, 4x256 -2.8%, 6x512 -1.4%, DRAM out -0.6%, dram_interleaved -0.7%, l1_interleaved flat (noisy row). Path reservation alone was the cost: posted with reservation is flat (21658). Nulls: assign without 64-bit shifts / rebalanced scan (flat), two rectangles (slower) |

### Graduated (commit e1dd8e95f6a)
- **Count table** (`r41_table_mcast`) is now the only table path. `send_table` in push.hpp sends one posted `ncrisc_noc_fast_write` multicast to the grid rectangle with `multicast_path_reserve = false`; `table_mcast_dst` builds the rectangle before the row wait. Every non-coordinator core waits on its NIU0 `NIU_SLV_POSTED_WR_REQ_RECEIVED` (`wait_table`), token core or not, and advances `land_base` by 1. A posted write that lands after its target core exits would land after the next program's counter snapshot, so every receiver must wait for it.
  - The writer polls NIU0 itself; a coordinator token core keeps `kHeaderTableFlag`.
  - Deleted: the table `McastArgs` channel (host channel, its semaphore and runtime args) and the reader's helper send / receive.
  - Deadlock argument (BH NoC doc: unreserved broadcasts make deadlock freedom software's job): broadcasts use their own VC class. A reserved broadcast acquires its whole tree or backs off and retries, so it never holds a partial tree. This is the only unreserved broadcast in tt-metal, and the op's W multicasts landed thousands of cycles earlier. The `--dev` NoC sanitizer passed.
- **combine_weights DRAM writes on VC 2** (`r41_push_balance`): `write_weight_tiles` uses `NocOptions::CUSTOM_VC` with `kWeightsVc<out_dram>`. `out_dram` moved into the common named compile-time args, so the reader sees it too.
  - **Carve-out (earned by measurement):** with DRAM `dispatched` the weights stay on `NOC_UNICAST_WRITE_VC` (the same path as before). On DRAM out, VC 2 alone measured 2x512 +3.3% / +6.8% and 4x256 +1.7% / +2.2% against HEAD (two sessions, 9-call medians). Table-only against both changes: 4x512 -3.0% / -0.9%, 8x512 -1.1% / -0.5%, 6x512 +1.0% / +0.4%, 8x384 / 8x256 flat. On that placement the tail is the DRAM block writes, not the push.
- Instrumentation: zones unchanged. New `ABLATE_PUSH_PIECES` (push() returns at once; the land wait is skipped like the other push ablations) and `ABLATE_MEASURE_STALLS` (per-RISC `push_writes`, `push_stall_credit`, `push_stalls_credit`, `push_stall_cmdbuf` as `DeviceTimestampedData`). Both compile out unless their define is set.
- Combination check: the push-walk diet stays a regression on top of both graduations (8x512 20.86 -> 22.9-23.1 us). The VC fix removes the head-of-line stalls but not the delivery bound that faster issue runs into.
- Whole op 8x512: 9-call medians 21557 / 21651 (HEAD) -> 20884 / 20930 / 20936 (-3.2..-3.4%). Profiled call (`zones_8x512_r41_final.txt`): table lands 11.6k, push 12.3k -> 24.0k p50 / 27.3k max, tilize ends 30.0k max (was 12.3k / 13.05k -> 25.0k / 28.3k / 31.0k).
- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144, also under `--dev`. Unit + precision baseline 21/21 under `--dev`.

### Guard set (all 28 perf rows, one profiled run each, HEAD e2633f6861a vs e1dd8e95f6a; device ns, last call)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 21682 | 20936 | -3.4% |
| test_perf_row[6x512_perf] | 19896 | 19238 | -3.3% |
| test_perf_row[8x384_perf] | 19593 | 18587 | -5.1% |
| test_perf_row[4x512_perf] | 16605 | 15054 | -9.3% |
| test_perf_row[8x256_perf] | 15664 | 14986 | -4.3% |
| test_perf_row[2x512_perf] | 11908 | 11505 | -3.4% |
| test_perf_row[4x256_perf] | 11873 | 11417 | -3.8% |
| test_perf_row_dram_out[8x512_perf] | 67659 | 68105 | +0.7% |
| test_perf_row_dram_out[6x512_perf] | 52750 | 52144 | -1.1% |
| test_perf_row_dram_out[8x384_perf] | 50593 | 50161 | -0.9% |
| test_perf_row_dram_out[4x512_perf] | 37989 | 38616 | +1.7% |
| test_perf_row_dram_out[8x256_perf] | 38776 | 38297 | -1.2% |
| test_perf_row_dram_out[2x512_perf] | 24439 | 23771 | -2.7% |
| test_perf_row_dram_out[4x256_perf] | 23943 | 23593 | -1.5% |
| interleaved_in[8x512 dram_interleaved] | 46616 | 45466 | -2.5% |
| interleaved_in[8x512 l1_interleaved] | 43226 | 43256 | +0.1% |
| interleaved_in[6x512 dram_interleaved] | 34191 | 33404 | -2.3% |
| interleaved_in[6x512 l1_interleaved] | 33646 | 31783 | -5.5% |
| interleaved_in[8x384 dram_interleaved] | 33884 | 33206 | -2.0% |
| interleaved_in[8x384 l1_interleaved] | 33770 | 30674 | -9.2% |
| interleaved_in[4x512 dram_interleaved] | 27976 | 26281 | -6.1% |
| interleaved_in[4x512 l1_interleaved] | 26479 | 21479 | -18.9% |
| interleaved_in[8x256 dram_interleaved] | 25587 | 25001 | -2.3% |
| interleaved_in[8x256 l1_interleaved] | 20331 | 21412 | +5.3% |
| interleaved_in[2x512 dram_interleaved] | 22144 | 20931 | -5.5% |
| interleaved_in[2x512 l1_interleaved] | 19356 | 17668 | -8.7% |
| interleaved_in[4x256 dram_interleaved] | 19440 | 18978 | -2.4% |
| interleaved_in[4x256 l1_interleaved] | 16455 | 16094 | -2.2% |

The positive rows were re-measured with 9-call medians in two alternating HEAD / final sessions:

| row | HEAD | final |
|---|---|---|
| DRAM out 8x512 | 68126 / 68091 | 68122 / 67710 |
| DRAM out 4x512 | 38397 / 38381 | 38130 / 38319 |
| l1_interleaved 8x256 | 23390 / 23507 | 22337 / 22648 |

All three are flat or faster; the l1_interleaved row swings about +/-10% call to call. No carve-out beyond the DRAM-out VC choice was earned. Every production row stays below its `stock_ns` (8x512: 20.9 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `McastArgs` / `ttnn.Mcast` (count table) | capability | the helper always sets `NOC_CMD_PATH_RESERVE` on its multicast and signals with a separate flag multicast or semaphore. It cannot send a posted, unreserved multicast whose landing receivers detect on their posted-write slave counter, and path reservation costs ~400 cycles on the 11x10 grid before any data leaves | 21661 / 21572 (whole op 8x512, 9-call medians; posted but still reserved: 21658, flat) | 21348 / 21346 | `kernels/nomic_moe_route_dispatch_push.hpp` (`send_table`, `wait_table`), `kernels/nomic_moe_route_dispatch_reader.cpp` (`receive_or_send_table`) |

- The VC 2 change is not a bypass: it uses `Noc::async_write<NocOptions::CUSTOM_VC>`.
- Recorded as options, not graduated (all bit-exact, DM only, no precision trade):
  - `r41_push_issue` `coord_pre+len_once` (flat).
  - `r41_push_walk` cache + register state (-2.6% on DRAM out 8x512 in one run, +10% on the focus row).
  - Pacing by minimum issue interval (`r41_push_fused`, all slower).
- Leads for the next round:
  1. Make delivery fair at a higher issue rate. Faster issue is worth up to ~4-5k cycles of push on the median RISC, but every attempt so far lost it to the slowest owners' landing. A destination schedule that spreads the hot owners in time, or a per-owner rate limit, is what would turn the walk / issue diets into wins.
  2. The pushers' own post-push weights writes (~2/3 MB) now set the NCRISC tail together with the tilize (`wr_final_barrier`).
  3. Assign (~600-700 cycles) is not load-count bound; the remaining pre-push chain is softmax -> select -> table (~1.1k after the select now).

All ideas measured: 2 graduated (one with a measured DRAM-out carve-out), 1 null, 2 regressions. 8x512: 21.6 -> 20.9 us. No measured regression on the 28-row guard set.

## Perf R4.2 - Fan-out perf tournament (continuation round 2 of 3)
- Date: 2026-11-06
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD aac14b6cb1b)
- Whole op, 9-call medians: 20830 / 20959 ns.
- Zone dumps: `perf_experiments/zones_8x512_r42_before.txt`, `perf_experiments/r42_breakdown/zones_head_sm.txt`. The second carries the new `cmp_wait_pass` zone.
- Scripts are in `perf_experiments/r42_breakdown/`: `prof.sh`, `many.sh`, `guard.sh`, `percore.py`.

Path in cycles (p50):

| stage | cycles |
|---|---|
| W lands (`cmp_wait_w` end, TRISC_0) | 3.35k |
| matmul ends | 5.2k |
| softmax ends (TRISC_1) | 9.24k |
| select ends | 10.4k / 10.8k |
| table lands | 11.8k |
| assign ends | 12.45k / 12.57k |
| push | 12.5k -> 24.2k p50 / 26.1k max |
| tilize ends | 28.8k max |

The brief's push contradiction (stubbed writes still 7.2k) was already resolved in R4.1. The push is software-paced at its delivery floor: faster issue makes delivery unfair.

**Softmax was occupancy-ambiguous, so it was split.** The new `cmp_wait_pass` zone sits around each pass's partials wait. Measured:
- Peer waits are 20-140 cycles a pass.
- Each pass is ~1.22k cycles of real work on TRISC_1, plus ~470 cycles of init before the first pass.
- Each member runs 3 passes (2 on member 2).

**Critical-path probe.** `ABLATE_DELAY_SOFTMAX` (new) adds 500 cycles a pass with `riscv_wait` on MATH. It moved the whole op by +1045 ns, which is ~94% of the 1.5k cycles injected. The softmax is on the critical path end to end.

Ranked:

| stage | cycles | note |
|---|---|---|
| push | 11.7k p50 / 13.7k max | near the 12.2k balanced replay floor; stragglers |
| softmax | 4.1k | critical, SFPU work |
| W arrival | 3.35k | two sender RISC zones of ~2.4k |
| last pass -> push start | 3.1k | serial |
| tail after the last landing | 2.3k | 1.76k tilize at ~29 cycles a tile; not floated |

`noc_estimate` is not built in this tree. The push gate is still the R3.2 replay floor.

### Portfolio and verdicts (`perf_experiments/r42_<idea>/`)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r42_softmax_pass`: per-pass overhead and band diet. Options: FACEIO (copy / pack only the pass's 1 KB face through 16x16-tile views), FUSEDCOPY (G contributor faces under one unpack <-> math handshake), DIET (r3 band diet), HOIST (SFPU init before the matmul) | WIN, graduated (all four) | pass 1204 -> 915 cycles; `cmp_softmax` window 4229 -> 3272; whole op 22911 -> 21719 (subagent session, median of 3). Menu, all bit-identical: FACEIO 1091; DIET 1146; FACEIO+DIET 21975 ns; +HOIST 1027 cycles / 22093 ns; FACEIO+FUSEDCOPY 998 cycles / 21896 ns | Everywhere. G = 3 / 2 / 1 and rps 1-4 rows faster or flat: 4x512 -5.2%, 2x512 -6.4%, G = 1 dram_interleaved 8x512 -1.7%, ragged flat. G = 1 uses DIET + HOIST only. Precision: no option trades precision |
| `r42_push_vc`: the push's posted writes spread over static unicast VCs by destination (on the current pusher and on the r41 walk diet) | NULL / REGRESSION, not graduated | Current pusher: every spread is +1.6..+9% (VC changes cost a NOC_CTRL store a write; fair delivery but every pusher ~0.9k slower). Diet + owner&3 -> VC 0-3: 20837-20929 -> ~19.9k (-4.5%) | No static mapping holds across shapes. Diet + owner&3: 6x512 +4.7%, 8x256 +13.9%, dram_interleaved 8x512 +3.2..+4.3%. Diet + owner%3 -> VC 0/1/3: 6x512 +11..+14%, 4x512 +7..+13%. Recorded as an option. Untested lead: a key relative to the source, `(owner - self) & 3` |
| `r42_w_arrival`: attribute the W sender's 2.4k, then unreserved raw multicast with the flag behind the data, and the other RISC's half sent as soon as it lands | WIN, graduated in a changed form (below) | `cmp_wait_w` end 3340 -> 2958; whole op 21053 -> 20801 (19-call pooled) | Attribution: read issue ~40 cycles a KB (not CPU-bound); read latency 450-650; a reserved multicast at 27-30 B/cycle. NOC1 is shared with `wr_stage_gather`. Nulls: W first in the kernel 21019, slice 1 partly on NOC0 21084, 8 KB / 4 KB commands slower, per-block arrival (trids) flat / slower, holding the gather 25214 |
| `r42_select_chain`: the 9.4k -> 12.5k chain; `ld8` (8 code-word loads in flight in `select_tokens`) + `xl` (staging waited after the assign, writer gated by a header flag) | WIN, graduated (`ld8 + xl`) | Isolated: 20966-21044 -> 20650-20699 (-1.5%). On top of the softmax / W graduations: A/B 20343 / 20371 / 20231 -> 20250 / 20199 / 20265 (-0.4%) | Everywhere measured: 2x512 / 4x256 / DRAM out / dram_interleaved flat. Attribution: member 0's untilize (x_rows at 11.84k) gated its table wait; the select is L1-load-latency bound (~28-36 cycles a token, same for a count-only loop). Nulls / regressions: `as` (scan sums in registers), `pump` (-200 cycles of chain, flat whole op), count-only select (+1.6%), deferred rank (+2.3..+4%) |

### Graduated (commits ad7784c23ed, e1e5e875b94)
- **Softmax pass** (`r42_softmax_pass`), the only path:
  - G > 1: copy_init / pack_init on the new 16x16-tile views `CB_PARTIAL_FACES` (26, UnpackToDestFp32) and `CB_CODE_FACES` (27). `route_dispatch_fused_copy::unpack_contributors` / `contributors_ready` put the half's face of each contributor into face 0 of DEST tiles 0..G-1. `face0_only<G>` runs the band. `pack_tile<true>` packs only the code face into its place in the pass's `cb_code_stage` page.
  - Every G: `band` is the r3 diet band, and `softmax8::init()` runs before the matmul.
  - Deleted: the old band, `share_faces`, the full-tile copies and pack, and the reciprocal init.
  - **m1 rule change:** the m1 lanes are now those with x == 1 rather than y == rcp. A lane whose x is one fp32 ulp below 1 can round to y == rcp. The old rule read that as a tie at m1; the new one makes it the m2 lane. The two experts are the same; only their order can differ, and only when two logits are within ~1e-7. Outputs were bit-identical on every perf row, golden and unit case.
- **W delivery** (`r42_w_arrival`, changed form), the only path. `deliver_weight_slice` replaces the `ttnn.Mcast` W channels with a raw multicast, which is:
  - unreserved;
  - sent as ONE multicast once both read halves landed;
  - split into burst chunks that are all `linked`, with the flag (semaphores `SEM_W_FLAGS = (4, 5)`) closing the linked sequence on the same command buffer and VC.

  Receivers wait on the flag. The reader-side sender leaves its acks to `rd_final_barrier`.
  - Why linked: the subagent's unlinked form regressed G = 1, which multicasts 48 KB in 16 KB bursts. Each unlinked chunk re-arbitrates the tree, and W landed ~950 cycles later on l1_interleaved 2x512 (`cmp_wait_w` 5.11k -> 6.06k; linked 5.22k). Measured 20-call medians: l1_interleaved 2x512 +4.6% and 4x256 +5.3%.
  - A reserved-on-G = 1 carve-out was measured and is not needed. Unreserved + linked everywhere (V2) beat reserved + linked on G = 1 (V1) on every row (`perf_experiments/r42_breakdown/w_linked_ab.txt`).
  - The early-half split was dropped: it was flat on 8x512 (20264 / 20167 vs 20232 / 20143) and slower on l1_interleaved 6x512.
  - Deleted: the host `ttnn.Mcast` channels for W, `send_weight_channel`, `receive_weight_channel`, and the `mcast_args.hpp` includes.
- **Select chain** (`r42_select_chain`, `ld8 + xl`), the only path:
  - `select_tokens` loads 8 code words before 8 emits.
  - The reader publishes `x_rows.get_read_ptr()` before the table wait. It waits for the untilize after its assign and sets `kHeaderStagingReady` (`REGION_WORDS` 40 -> 41).
  - The writer waits on that flag in the new `wr_wait_staging` zone.
- Instrumentation:
  - New zones: `cmp_wait_pass` (each softmax pass's partials wait) and `wr_wait_staging`. `rd_wait_xrows` moved with its wait.
  - New probe: `ABLATE_DELAY_SOFTMAX` (+500 cycles a pass on MATH; compiles out unless set).
- Whole op 8x512, 9-call medians, alternating sessions: HEAD 20850 / 20900 -> final 20281 / 20219 (-2.9..-3.3%). Start of round: 20830 / 20959.
- Profiled call (`zones_8x512_r42_final.txt`):

  | stage | before | after |
  |---|---|---|
  | W lands | 3.35k | 3.04k |
  | softmax ends | 9.24k | 8.27k |
  | select ends | 10.4k / 10.8k | 9.65k / 9.76k |
  | table lands | 11.8k | 11.0k |
  | push starts | 12.5k | 11.76k |
  | push max ends | 26.1k | 25.5k |
  | tilize ends | 28.8k | 28.2k |

- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144, also under `--dev` (NoC sanitizer on the unreserved multicast). Unit + precision baseline: 21/21 under `--dev`.

### Guard set (all 28 perf rows, one profiled run each, aac14b6cb1b vs final; device ns, last call)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 20974 | 20417 | -2.7% |
| test_perf_row[6x512_perf] | 19031 | 18655 | -2.0% |
| test_perf_row[8x384_perf] | 18689 | 18367 | -1.7% |
| test_perf_row[4x512_perf] | 15095 | 14239 | -5.7% |
| test_perf_row[8x256_perf] | 15270 | 14227 | -6.8% |
| test_perf_row[2x512_perf] | 11584 | 11353 | -2.0% |
| test_perf_row[4x256_perf] | 11419 | 11139 | -2.5% |
| test_perf_row_dram_out[8x512_perf] | 67921 | 67848 | -0.1% |
| test_perf_row_dram_out[6x512_perf] | 52610 | 52022 | -1.1% |
| test_perf_row_dram_out[8x384_perf] | 50939 | 50124 | -1.6% |
| test_perf_row_dram_out[4x512_perf] | 38684 | 38775 | +0.2% |
| test_perf_row_dram_out[8x256_perf] | 38705 | 38119 | -1.5% |
| test_perf_row_dram_out[2x512_perf] | 23639 | 23686 | +0.2% |
| test_perf_row_dram_out[4x256_perf] | 23448 | 23061 | -1.7% |
| interleaved_in[8x512 dram_interleaved] | 46035 | 45469 | -1.2% |
| interleaved_in[8x512 l1_interleaved] | 41161 | 44689 | +8.6% |
| interleaved_in[6x512 dram_interleaved] | 33356 | 33328 | -0.1% |
| interleaved_in[6x512 l1_interleaved] | 31559 | 32224 | +2.1% |
| interleaved_in[8x384 dram_interleaved] | 33044 | 32759 | -0.9% |
| interleaved_in[8x384 l1_interleaved] | 30203 | 28576 | -5.4% |
| interleaved_in[4x512 dram_interleaved] | 26224 | 26064 | -0.6% |
| interleaved_in[4x512 l1_interleaved] | 23582 | 20393 | -13.5% |
| interleaved_in[8x256 dram_interleaved] | 24541 | 25207 | +2.7% |
| interleaved_in[8x256 l1_interleaved] | 24066 | 22856 | -5.0% |
| interleaved_in[2x512 dram_interleaved] | 20748 | 20681 | -0.3% |
| interleaved_in[2x512 l1_interleaved] | 17778 | 17798 | +0.1% |
| interleaved_in[4x256 dram_interleaved] | 18919 | 18913 | -0.0% |
| interleaved_in[4x256 l1_interleaved] | 16016 | 15808 | -1.3% |

Single profiled calls on l1_interleaved swing +/-10%. Every l1_interleaved row was re-measured with 20-call medians in two alternating sessions (`r42_breakdown/l1_interleaved_ab_final.txt`, final / base):

| row | final | base |
|---|---|---|
| 8x512 | 44477 / 42817 | 42959 / 43466 |
| 6x512 | 31747 / 32016 | 31867 / 31795 |
| 8x384 | 31962 / 31762 | 31547 / 31176 |
| 4x512 | 24283 / 23801 | 23919 / 23383 |
| 2x512 | 17912 / 17770 | 17896 / 17977 |
| 4x256 | 15887 / 15852 | 15970 / 16003 |
| 8x256 | 22901 / 23136 | 22396 / 22274 |

- **Unresolved, recorded rather than carved out:** l1_interleaved 8x256 reads +2.4% over six final sessions (22.2-23.1k) against a steady base (22.25-22.40k). 8x384 / 4x512 read ~+1.6%.
  - Removing the softmax change or the select change does not restore base beyond the session spread (`-softmax` 22739 / 22004, `-select` 22896 / 22754, final 23078 / 22165).
  - The profiled call shows no slower stage. On this placement the W sender's DRAM reads swing 7.5k-11.2k cycles under x-read contention, and the final call was faster there (24304 vs 25351 ns).
  - No change is identified to carve out.
- The dram_interleaved 8x256 single-call +2.7% was flat in 9-call medians earlier in the round (24810 / 24703 vs 24528 / 24802).
- Every production row stays below its `stock_ns` (8x512: 20.3 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `McastArgs` / `ttnn.Mcast` (W slices) | capability | The helper always sets `NOC_CMD_PATH_RESERVE` and signals with a separate barrier + semaphore multicast. It cannot send an unreserved multicast, and it cannot make a flag write the terminator of a linked data sequence on the same command buffer / VC with no barrier. Path reservation costs ~400 cycles on the 11x10 grid, and per-chunk reservation of a multi-burst payload costs ~950 cycles at 48 KB | 21053 (whole op 8x512, 9-call medians, helper path) | 20801 (subagent's unreserved form); final linked form flat against that on 8x512 (20232 / 20143 vs 20264 / 20167) and -0.6..-0.8% on l1_interleaved 2x512 / 4x256 vs the helper | `kernels/nomic_moe_route_dispatch_push.hpp` (`w_mcast`, `deliver_weight_slice`, `receive_weight_slice`) |
| `copy_tile` (contributor faces of a softmax pass) | capability | Each `copy_tile` is a full unpack <-> math round trip: mailbox dest address, MATH_DONE / UNPACK_TO_DEST semaphores, config restore. That is ~90 cycles of math per 1 KB face. There is no call that unpacks several tiles (or faces) to DEST under one handshake. The raw sequence must end by re-enabling the SrcA address swizzle (`SETC16(SRCA_SET_Base_ADDR32, 0x4)`), which only the stock `unpack_to_dest_tile_done` shows; without it a later untilize silently corrupts every row | pass 1027 cycles / whole op 22093 ns (FACEIO + DIET + HOIST with `copy_tile`) | pass 915 cycles / 21719 ns | `kernels/nomic_moe_route_dispatch_fused_copy.hpp`, `kernels/nomic_moe_route_dispatch_compute.cpp` (softmax pass loop) |
| `_llk_math_eltwise_unary_sfpu_params_` (VectorMode) | capability | No VectorMode runs one face. Since R4.2 a pass's input is a single face in face 0 of G DEST tiles; the R3.1 bypass (`share_faces`) is now `face0_only` | not measured separately (R3.1 pair: 29.5-29.7 us -> 26.75-27.08 us) | see the softmax row above | `kernels/nomic_moe_route_dispatch_softmax8.hpp` (`face0_only`) |

- The softmax band itself stays raw SFPI (Perf 1 bypass row, unchanged gap).
- The select / staging change uses no helper.
- Options recorded, not graduated (no precision trade in any of them): push VC spreading (above), `pump` (-200 cycles of chain, flat whole op), FACEIO without FUSEDCOPY.
- Leads for the next round:
  1. The push is now the whole middle: 11.76k -> 23.5k p50 / 25.5k max. The r42_push_vc data shows the walk diet + VC spreading reaching ~19.9k on 8x512. A mapping that holds on 6x512 / 4x512 / 8x256 (`(owner - self) & 3` untested) would graduate it.
  2. W arrival is now capped by NOC1 contention from `wr_stage_gather` (ch1 lands ~300 cycles after ch0).
  3. Softmax 3 x 915 cycles: overlapping pass i+1's unpack with pass i's SFPU (~200 cycles a pass) needs a raw DEST-half handoff.
  4. The l1_interleaved 8x256 +2.4% above.

All ideas measured: 3 graduated (W in a changed form, with no carve-out), 1 null / regression (push VC). 8x512: 20.85-20.90 -> 20.22-20.28 us. No regression on the 28-row guard set beyond the unresolved l1_interleaved 8x256 reading above.

## Perf R4.3 - Fan-out perf tournament (continuation round 3 of 3)
- Date: 2026-11-06
- Focus: LOOSE_CASES `8x512_perf` (nd_norm x, `dispatched` in L1, default compute config HiFi4 / fp32 DEST / approx off). Blackhole p100a, AICLK 1350 MHz. SUPPORTED unchanged.

### Measured breakdown (8x512, before; HEAD dd8177be0f5)
- Whole op, 9-call median: 20292 ns. Profiled call 21946 ns.
- Zone dumps: `perf_experiments/zones_8x512_r43_before.txt`, `zones_8x512_r43_ablate_push_writes.txt`, `zones_8x512_r43_ablate_push_pieces.txt`, `zones_8x512_r43_ablate_push.txt`. Scripts in `perf_experiments/r43_breakdown/` (`prof.sh`, `many.sh`, `guard.sh`, `percore.py`).
- The steering brief's push contradiction was already resolved in R4.1. The ablation on HEAD reproduces it:

  | stubbed (cumulative) | whole op (profiled) | push zone p50 |
  |---|---|---|
  | nothing | 21946 ns | 11.8k |
  | NoC issue (`ABLATE_PUSH_WRITES`) | 16727 ns | 7.2k, every RISC 7.0-7.3k |
  | + per-piece address math (`ABLATE_PUSH_PIECES`) | 14540 ns | 2.45k (slot walk) |
  | whole push (`ABLATE_PUSH`) | 12466 ns | |

  So ~7.2k cycles of the push are software on the DM RISCs, and the spread between RISCs is entirely NoC-side.

Path in cycles (p50):

| stage | cycles |
|---|---|
| W lands (TRISC_0) | 3.06k |
| matmul ends | 4.95k |
| softmax ends | 8.27k |
| select ends | 9.67k |
| table lands | 10.88k |
| assign ends | 11.53k |
| push | 11.6k -> 23.4k p50 / 26.4k max |
| last owner landing | 27.3k |
| tilize ends | 29.6k max |

- Straggler in this dump: the reader of (2,6) (push 14.8k vs 11.8k median). Its own land wait ended 90 cycles after its push, so its tilize was gated by its own push. Across 40 later calls (`r43_land_decouple`) the critical owner always waited on the chip's last push instead, so this was not the common case.
- Ranked: push 12-15k (software paced; faster issue delivers unfairly, R4.1), softmax 3.3k (critical, R4.2 probe), matmul 1.85k, select -> push 1.9k, tail 2.3k (fast tilize already in use).

### Portfolio and verdicts (`perf_experiments/r43_<idea>/`; whole-op ns against the real op in the same session)
| idea | verdict | baseline -> candidate (8x512) | domain / notes |
|---|---|---|---|
| `r43_push_vc_rel`: r41 walk diet + per-write static VC keyed on the destination relative to the source | WIN, graduated (`(owner - self) & 1` -> VC 1 / VC 3, one command buffer per VC) | 20109-20240 -> 19160-19357 (-3.7..-5.3%, 3 sessions) | Faster on all 9 shapes tested: 4x512 -3.5..-4.5%, 8x256 -1.3..-2.8%, dram_interleaved 8x512 -4.1..-4.3%, DRAM out 8x512 -3.7%. Options: `(owner - self) % 3` over VCs 0/1/3 (6x512 -8%, 4x512 flat, dram_interleaved outlier calls); `(owner - self) & 3` flat; global keys (`owner & 1` +3.3%, `owner & 3` regresses 6x512 / 8x256); route keys (column / row / hop class) -3.2..+10%; per-write NOC_CTRL store instead of per-VC buffers -3.6%; in-flight cap 2-8 KB no effect. Stall counters: 0 credit stalls, cmd-buf stall max 218 -> 31 cycles a RISC |
| `r43_softmax_overlap`: next pass's contributor unpack into the other DEST half while the current band runs | WIN, graduated (OVL) | `cmp_softmax` 3290 -> 2990 cycles; whole op (subagent) 21618 -> 21356 profiled median | Bit-identical. Attribution of one pass: band 735-750 cycles (32 `SFPSHFT2` rotates cost 2 cycles each, ~256 a pass), unpack + handshake 177, transitions ~80; the code-face pack was already overlapped. OVL0 (first pass unpacked under the matmul tail) is wrong on hardware (4004 / 8192 slots differ: unpack to DEST while the FPU runs). OVL0B (same after the drain) exact but slower. G = 1 unchanged |
| `r43_land_decouple`: compute polls the NIU posted-write counters itself, tilize no longer waits on the owner's own push | NULL | 20160 / 20309 -> 20243 / 20239; 20333 / 20334 -> 20185 / 20164 | Correct (golden 144/144 under `--dev`); TRISC can read the NIU status registers. In 30 profiled calls the critical owner's land wait ends 150-300 cycles after the last push anywhere, never on its own push |
| `r43_half_mm`: 32x16 in1 / half-output router matmul on the per-tile path | NULL | 20262 -> 20218 (`w32x16`), `half` 20225 / 20313, `rt2` 20348 | Bit-identical, flat on 9 shapes. Probe: 2000 cycles injected on MATH -> +394 ns (~58% transmission). Halving the MVMULs does not move the unpack zone (1805 -> 1805): the stage is per-pair unpack / handshake bound (~56 cycles a pair, ~40 with math stubbed) |
| `r43_push_steal`: the core's DM RISC that finishes its push first takes the partner's remaining tokens / pads from the back | REGRESSION | 20202-20319 -> 20455-20596 (claim chunk 16, best) | 6x512 -7.4..-8.1% (a core's two RISCs are 3-7k apart there from cmd-buf stalls), 8x512 dram_interleaved -4.7%, but 2x512 +4.6..+6.0%, 4x256 +4.3..+5.2%, 4x512 +1.9..+3.9%: ~520 cycles a RISC of fixed protocol cost, little stolen on 8x512 (0-17 items a call). Grant / claim protocols, chunk 4-16, all bit-exact |

### Graduated
- **Push walk diet + relative VC key** (`r43_push_vc_rel`, commit 4d09f98a6b2), the only push path:
  - `RowPusher` caches each walk stream's destination (owner coordinate register value, block address, VC command buffer) per output tile row. One stream per expert plus one for the pads. The piece loop is unrolled over the host's max pieces a member pushes (`push_max_pieces`), and the row-parity filter compiles out unless a piece is split by parity (`push_row_parity`).
  - The VC is `(owner - self) & 1` -> VC 1 on `write_cmd_buf`, VC 3 on `write_reg_cmd_buf`. Both buffers are programmed once in `start()`, so a write adds no register store.
  - Deleted: the round-robin `cmd_bufs` path and the `push_cmd_bufs` host knob (`PUSH_CMD_BUFS`), `PushPiece` / `load_push_pieces`, and the per-write division and common-arg load.
  - Why it holds now when the diet alone regressed (R4.1, R4.2): the relative key spreads one source's writes over two VCs by destination. Global owner keys regressed other shapes.
  - 8x512 9-call medians: 20292 -> 19167.
- **Softmax pass overlap** (`r43_softmax_overlap` OVL, commit 969ca02bf3c), the only G > 1 pass path:
  - `contributors_ready` is split into `contributors_send` (first pass: drain, current half to the unpacker) and `contributors_wait` (token + zero-flag clear).
  - `pass_marker` posts MATH_DONE behind the previous commit. `face0_only<G, 2>(hook)` runs `send_next_half` after two of the four row bands: it waits for the marker and for the packer to release, then mails the other half's address to the unpacker.
  - Deleted: `contributors_ready` and the hook-less `face0_only`.
  - Alternating sessions on top of the VC change, 9-call medians: 19199 / 19209 -> 19007 / 19052 (-0.8..-1.0%). 4x512 13728 -> 13292, 8x256 14037 -> 13725.
- Instrumentation: zones unchanged; both new paths run inside the existing `rd_push` / `wr_push` and `cmp_softmax` / `cmp_wait_pass` zones. `ABLATE_PUSH_PIECES` / `ABLATE_PUSH_WRITES` / `ABLATE_MEASURE_STALLS` keep working on the new pusher. Re-run on the final op (`zones_8x512_r43_final_ablate_push_*.txt`): issue stubbed 15006 ns, push zone p50 5.0k (was 7.2k); per-piece math stubbed 13348 ns, walk 1.87k (was 2.45k). The push zone is 8.4k p50 with the writes, so ~3.4k is NoC issue.
- Golden `eval/golden_tests/nomic_moe_route_dispatch/`: 144/144 after each graduation, and under `--dev` (NoC sanitizer) at the end. Unit + precision baseline 21/21 under `--dev`.
- Whole op 8x512, 9-call medians: 20292 (start of round) -> 19007 / 19052 (-6.1..-6.3%).
- Profiled call (`zones_8x512_r43_final.txt`):

  | stage | before | after |
  |---|---|---|
  | softmax ends | 8.27k | 7.98k |
  | push | 11.6k -> 23.4k p50 / 26.4k max | 11.3k -> 19.8k p50 / 23.1k max |
  | last owner landing | 27.3k | 24.0k |
  | tilize ends (max) | 29.6k | 26.3k |

### Guard set (all 28 perf rows, one profiled run each, dd8177be0f5 vs final; device ns, last call)
| row | before | after | delta |
|---|---|---|---|
| test_perf_row[8x512_perf] | 20280 | 19066 | -6.0% |
| test_perf_row[6x512_perf] | 18752 | 18493 | -1.4% |
| test_perf_row[8x384_perf] | 18042 | 18090 | +0.3% |
| test_perf_row[4x512_perf] | 14430 | 13659 | -5.3% |
| test_perf_row[8x256_perf] | 14461 | 13533 | -6.4% |
| test_perf_row[2x512_perf] | 11363 | 11276 | -0.8% |
| test_perf_row[4x256_perf] | 11179 | 11338 | +1.4% |
| test_perf_row_dram_out[8x512_perf] | 68187 | 64173 | -5.9% |
| test_perf_row_dram_out[6x512_perf] | 51445 | 51700 | +0.5% |
| test_perf_row_dram_out[8x384_perf] | 50536 | 49524 | -2.0% |
| test_perf_row_dram_out[4x512_perf] | 37116 | 37085 | -0.1% |
| test_perf_row_dram_out[8x256_perf] | 38235 | 37919 | -0.8% |
| test_perf_row_dram_out[2x512_perf] | 23391 | 23560 | +0.7% |
| test_perf_row_dram_out[4x256_perf] | 23553 | 23138 | -1.8% |
| interleaved_in[8x512 dram_interleaved] | 45548 | 43367 | -4.8% |
| interleaved_in[8x512 l1_interleaved] | 44113 | 41615 | -5.7% |
| interleaved_in[6x512 dram_interleaved] | 33199 | 31147 | -6.2% |
| interleaved_in[6x512 l1_interleaved] | 33210 | 30449 | -8.3% |
| interleaved_in[8x384 dram_interleaved] | 33219 | 31043 | -6.6% |
| interleaved_in[8x384 l1_interleaved] | 31553 | 30304 | -4.0% |
| interleaved_in[4x512 dram_interleaved] | 26166 | 24808 | -5.2% |
| interleaved_in[4x512 l1_interleaved] | 24031 | 23647 | -1.6% |
| interleaved_in[8x256 dram_interleaved] | 24716 | 24606 | -0.4% |
| interleaved_in[8x256 l1_interleaved] | 21379 | 22983 | +7.5% |
| interleaved_in[2x512 dram_interleaved] | 20701 | 19979 | -3.5% |
| interleaved_in[2x512 l1_interleaved] | 18110 | 17010 | -6.1% |
| interleaved_in[4x256 dram_interleaved] | 18921 | 18974 | +0.3% |
| interleaved_in[4x256 l1_interleaved] | 16008 | 15952 | -0.3% |

Positive and neighbouring rows were re-measured with 9-call medians in two alternating base / final sessions (`r43_breakdown/ab_*.txt`):

| row | base | final |
|---|---|---|
| L1 4x256 | 11164 / 11152 | 11059 / 11090 |
| L1 8x256 | 14176 / 14330 | 13604 / 13709 |
| L1 2x512 | 11219 / 11285 | 11205 / 11176 |
| L1 6x512 | 18650 / 18647 | 18438 / 18484 |
| l1_interleaved 4x256 | 15821 / 15845 | 16047 / 15970 |
| l1_interleaved 8x256 | 23035 / 23238 | 22573 / 22340 |
| l1_interleaved 2x512 | 17761 / 17832 | 16864 / 16890 |
| l1_interleaved 6x512 | 32844 / 30628 | 30157 / 28967 |
| DRAM out 4x256 | 23328 / 23616 | 23213 / 23259 |
| DRAM out 8x256 | 37788 / 37953 | 37185 / 37444 |
| DRAM out 2x512 | 23336 / 23488 | 23425 / 23119 |
| DRAM out 6x512 | 52112 / 52031 | 51684 / 51604 |

- **Recorded, not carved out:** l1_interleaved 4x256 reads +0.7% in four alternating sessions (base 15821-15947, final 15904-16047; `ab_l1i_4x256.txt`). That is ~110 ns, below the ~1.5% band used for carve-outs in earlier rounds. Every other row on that placement is 1-8% faster. It is a G = 1 row, so the push change is the one that applies there. A guard on it would fence a single shape.
- The single-call +7.5% on l1_interleaved 8x256 is that row's known +/-10% swing; 9-call medians read -2..-4%.
- Every production row stays below its `stock_ns` (8x512: 19.0 vs 45.6 us).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `copy_tile` / `math_unpack_to_dest_math_ready` + `set_dst_write_addr` (contributor unpack to DEST, softmax pass) | capability | The stock unpack-to-DEST handshake drains MATH / SFPU and can target only the DEST half math currently holds. Nothing lets math hand the unpacker the *other* half's address while its SFPU work on the current half is still queued, so pass i+1's unpack cannot overlap pass i's band. The raw form needs a MATH_DONE marker behind the previous commit and a MATH_PACK == 0 wait before mailing the address | stock path not re-measured this round (it cannot overlap; R4.2: `copy_tile` 22093 vs fused 21719). Non-overlapped raw handshake (R4.2 `contributors_ready`): 19199 / 19209 (whole op 8x512, 9-call medians) | 19007 / 19052 | `kernels/nomic_moe_route_dispatch_fused_copy.hpp` (`contributors_send`, `contributors_wait`, `pass_marker`, `send_next_half`), `kernels/nomic_moe_route_dispatch_softmax8.hpp` (`face0_only` hook), `kernels/nomic_moe_route_dispatch_compute.cpp` (softmax pass loop) |

- The push VC change is not a new bypass: `RowPusher` was already raw NoC (stateful `ncrisc_noc_write_set_state` / `ncrisc_noc_write_with_state`), and the change sets a different static VC per command buffer through the same calls. For the record, no kernel_lib NoC helper keeps per-VC command-buffer state for a stream of one-packet writes.
- `r43_land_decouple` read the NIU counters from TRISC in its experiment copy; nothing from it graduated.
- Options recorded, not graduated (none trade precision; every variant was bit-exact at HiFi4 / fp32 DEST / approx off):
  - push VC `(owner - self) % 3` over VCs 0/1/3 (6x512 -8%, 4x512 flat);
  - push steal (6x512 -7.4..-8.1%, small shapes +4..6%);
  - `w32x16` / `half` / `rt2` matmul (flat).
- Leads for the next round:
  1. Push steal on 6x512: a core's two RISCs end 3-7k apart there from cmd-buf stalls on one NoC. A protocol without the ~520 cycles a RISC of fixed cost (claim set-up plus the failed end steal) would keep the 6x512 / interleaved-input wins without the small-shape regressions.
  2. The softmax band's 32 `SFPSHFT2` rotates (~256 cycles a pass) come from experts across SFPU lanes; experts on DEST rows would need a different matmul output layout.
  3. The router matmul stage is per-pair unpack / handshake bound (~56 cycles a pair), not FPU bound.
  4. The tail (last landing -> tilize end, ~2.3k) is the chip's last push landing plus the owner's fast tilize; the critical owner does not wait on its own push.

All ideas measured: 2 graduated (no carve-out), 2 null, 1 regression. 8x512: 20.29 -> 19.01-19.05 us. No material regression on the 28-row guard set; l1_interleaved 4x256 +0.7% recorded above.

## R4 review fixes
- Count table carries a per-call valid word; receivers wait for the NIU0 counter and the word, so a peer's early push cannot pass for the table.
- W channel 0 goes out from the coordinator's reader: both unreserved NOC0 broadcasts leave one command buffer and VC in order.
- Host guard against a writer-sent W slice on a token core restored.
- Measured flat against R4: 28-row guard set, positive single-call rows re-measured with 9-call medians (-0.6..+0.7%). Golden 144/144 (also --dev), unit 49/49.
