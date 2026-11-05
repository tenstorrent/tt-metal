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
