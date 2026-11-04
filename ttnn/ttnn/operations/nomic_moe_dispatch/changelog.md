# Changelog: nomic_moe_dispatch

## Phase 0 — Core Implementation
- **Date**: 2026-11-04
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Regime R1 `local_tilize_two_noc`:
  - the output is ND-sharded in L1 (32 rows × w tiles per shard, round robin over 110 cores);
  - two reader RISCs (rows 0-15 on NOC0, 16-31 on NOC1) do stateful one-packet row-piece gathers;
  - the fast tilize writes straight into each core's own shard, and there is no writer;
  - gather_index is read on device, one block ahead (`ids_lookahead = 1`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], table_buffer=[l1], output_buffer=[l1], window=[aligned, ragged]
- **Accuracy achieved**: bit-exact. PCC = 1.0, max_abs_err = 0, rms_err = 0, max ULP = 0, measured on 4 shapes (R = 2240 / 3296 / 4320 / 8416) plus 2 non-default compute configs, via `test_nomic_moe_dispatch_precision_baseline.py`
- **Golden suite at Phase 0**: 29 / 62 cells passing (`supported_pass` 29, `xfail_expected` 33, `supported_fail` / `xpass_drift` / `xfail_wrong_mode` 0), per `verifier_results/verifier_report.json`
- **Device perf (kernel ns per call, perf cases' setup)**: R = 8416: 9.62 us (accept 15.0, stretch 11.0). R = 4320: 6.99 us (accept 9.0). R = 2272: 4.58 us (accept 6.5).
- **Issues encountered / verifier fixes**:
  - **Perf lamp L4:** k = 1 was dropped from the column-split candidates. k = 2 fits wherever k = 1 does. Measured: R = 3296 went 6.59 → 4.86 us, and R = 6368 went 14.97 → 7.47 us.
  - **L1 ledger:** the ids scratch was resized from all blocks to a ring of `ids_lookahead + 1` pages, matching its live set.
  - **DRY:** `table_page_bytes`, the kernel's `TILE_BYTES` (slot size) and `IDS_ALIGN` now each come from one host source.
  - **Ledger and design:** `l1_ledger.md` was updated, and `op_design.md` gained an amendment under the selection rule.
  - **Implementer, earlier:** zero-block tilize InitOnly / UninitOnly trips a helper assert, so init and uninit were folded into the first and last blocks.
- **Tests added**:
  - acceptance: `test_nomic_moe_dispatch.py` (planner);
  - precision baseline: `test_nomic_moe_dispatch_precision_baseline.py` (verifier);
  - perf harnesses: `perf/test_nomic_moe_dispatch_perf_sweep.py` and `perf/test_nomic_moe_dispatch_knob_sweep.py` (verifier), run via `eval/eval_test_runner.sh`.

## Refinement 3 — Speed up the perf-flagged profile: gather_index fetch topology + issue stagger
- **Date**: 2026-11-04
- **What was done**: scheme change of the gather_index fetch (new regime row R10 `ids_row_mcast`, op_design.md) plus an explicit issue stagger and a k = 2 in-flight cap.
  - **Topology:** a grid row's 11 cores hold 11 consecutive shards in every round, so the row's round-j ids are one contiguous span (≤ 896 B). The row's x = 0 core (its NOC0 reader) reads that span from the one gather_index DRAM bank and multicasts it along the row (disjoint row paths). It then multicasts the round count to the row's `ids_ready` semaphore from a never-rewritten L1 cell, after the data on the same NoC/VC. Rounds 0-1 are published first thing; round j+2 is published at the sender's block j start, fetched `IDS_LOOKAHEAD` blocks earlier. Every reader reads its ids from a local full-size image (`cb_gather_ids`, one CB for both readers, ≤ 34 KB). DRAM ids requests drop from 220 a round (64 B each, R·4·k B in total) to 10 a round (≈ R·4 B).
  - **Issue stagger** (the verifier's required pair): core c walks its half's rows from row (s·c mod 16), in steps of 2·(c mod d)+1, with (s, d) a per-k plan value: `ROW_STAGGER_BY_K = {2: (3, 4)}`, `ROW_STAGGER_DEFAULT = (5, 2)`. Each step is host-checked coprime with the half. At k = 2, (3, 4) keeps R = 6368 at Phase 0 parity (7.48-7.58 vs 7.50); (5, 2) is +1.5% there and ~0.07 us better at R = 8416. At k ≥ 3, (5, 2) is best or within noise.
  - **In-flight cap:** `READ_BYTES_IN_FLIGHT = 6·768` caps reads in flight by bytes (6 at k = 2, still 8 at k ≥ 3).
  - Why both are needed: with every reader's ids local, the grid starts within ~0.3 us and k = 2 shapes went bimodal. About half the calls hit a NoC congestion collapse in the last blocks (R = 8416: 14-17 us instead of 8.6; zones or dummy DRAM reads perturb it away). In-order walks gave 6/16 outliers, stagger (3, 4) 1/16, stagger + 6 in flight 0/48 (R = 8416 / 6368 / 3296).
  - Parked knobs (byte-identical when off, measured, kept as live tunables): `IDS_SENDER_NOC` (NOC1: no better), `START_SPREAD_CYCLES` (a per-core start delay: no cure for the collapse, slower), `IDS_LOOKAHEAD` (2 / 4: flat).
- **Alternatives built and measured first** (device us, mean of 5 calls, R = 2272 / 4320 / 8416; Phase 0 4.61 / 7.02 / 9.81). The grid-multicast form is in WIP commit c69c206:
  - Whole index multicast to the grid (round-sized chunks, one sender): 4.16 / 6.97 / 14.44. A 7 KB grid multicast takes 0.56 us idle and ~1-1.25 us behind the read traffic, so the sender was the 11.5 us tail; non-sender cores ended at 8.4 us.
  - Five senders: 15.0 at R = 8416 (grid multicasts serialize). Finer chunks: worse (R = 8416, 16 tile rows: 16.6; 4: 46.9).
  - First chunk multicast plus later blocks fetched per reader: 4.33 / 6.77 / 9.84, and R = 3296 / 6368 slower.
  - Inside R10: fetching round 0 alone and publishing later rounds at the end of a block's issue was +5-17% (round 1 late). Publishing them opportunistically between the sender's row reads was +2-13% (a multicast under load stalls the sender). Both reverted.
- **Device perf** (kernel us a call, eval real-time profiler, mean of 5 warm calls, two runs; Phase 0 from the same harness and board):

  | R | k | Phase 0 | Refinement 3 | Δ |
  |---|---|---|---|---|
  | 2240 | 3 | 4.56 | 3.84-3.85 | −16% |
  | 2272 (guard) | 3 | 4.61 | 3.73-3.75 | −19% |
  | 2528 | 4 | 5.82 | 5.59-5.61 | −4% |
  | 2656 (k = 6 rep.) | 6 | 8.89 | 8.56 | −4% |
  | 3296 | 2 | 4.85-4.88 | 4.64-4.67 | −4% |
  | 4320 (guard) | 3 | 7.02 | 6.94-7.03 | ≈ 0 |
  | 5344 | 3 | 9.02 | 8.67-8.68 | −4% |
  | 6368 | 2 | 7.50-7.52 | 7.52-7.59 | ≈ 0 (+0.3-0.8%, ≤ 0.06 us) |
  | 7392 | 4 | 14.81-14.86 | 13.98-14.00 | −6% |
  | 8064 | 3 | 12.36 | 11.34-11.36 | −8% |
  | 8224 | 2 | 9.60 | 9.32-9.37 | −3% |
  | **8416 (PERF FOCUS)** | 2 | 9.81 (9.62 at verification) | **9.13-9.22** (loose case 9.12) | **−6.7%** |

- **Bottleneck now** (zone profile, R = 8416 / 4320 / 6368):
  - Start: first block at ~1.3 us, which is one DRAM round trip on the gather_index bank plus a row multicast.
  - Per-block gather: ~1.06-1.13 us a block on the NOC0 reader. At k = 3 (R = 4320) the NOC1 reader takes only ~0.75 us, so the NOC0 / NOC1 row split is now unbalanced. That is Refinement 4's lamp L1 and why R = 4320 did not improve.
  - Tail: the last 12-tile tilize (~0.4 us, lamp L7).
- **Accuracy achieved**: bit-exact (PCC = 1.0, atol = rtol = 0, max ULP 0) on R = 2240 / 3296 / 4320 / 8416 and both non-default compute configs (precision baseline 8/8). All index patterns (random, reversed, hot, uniform, extreme) and spare tile rows are exact.
- **Golden test progress**: 29/29 supported cells pass (11 `test_op` L1/L1, 3 loose perf cases, 15 regression). The 33 DRAM cells stay xfail (deferred regimes). SUPPORTED is unchanged.
- **Issues encountered**:
  - The k = 2 congestion collapse described above, found by per-call profiling: the 5-call means hid it as noise.
  - The mcast helper (`McastArgs`) could not express a counter both reader RISCs of the sender core can poll. Its Counter atomics skip the source core, and attach refuses a second kernel with another NoC on the sender core. So the multicast is raw, documented at the top of the gather kernel.
  - R = 6368 (k = 2, 4 blocks) was 1.5-4% slower than Phase 0 at stagger (5, 2). Raising reads in flight only brings the collapse back (8: 8.64-8.67; 7: 7.93; 4: 8.00-8.10). The per-k stagger table (k = 2: (3, 4)) returns it to parity.
- **Tests added**:
  - `perf/test_nomic_moe_dispatch_zone_probe.py`: a per-call profiler probe with `NMD_PROBE_ROWS` / `NMD_PROBE_CALLS` / `NMD_PROBE_KNOBS`.
  - `perf/test_nomic_moe_dispatch_knob_sweep.py`: cases for the stagger, bytes in flight, lookahead, sender NoC and start spread.

## Refinement 4 — Speed up the perf-flagged profile: NoC / bank contention levers
- **Date**: 2026-11-04
- **What was done**: Classified the bottleneck first, then applied the issue-path fix and the contention levers it made necessary (knob-turns on the R1 block surface; no SUPPORTED or CB change).
  - **Ablation** (R = 8416 / 7392, mean of 5, eval harness): every row read stubbed with the address math kept gave 8.62 / 13.39 us, against 9.17 / 13.99 real. With the tilize stubbed too, 8.11 / 13.11. In zones, a block body with no NoC reads still took ~0.85 us (~72 cycles a row). Removing the two NIU register reads per row (the in-flight counter and the cmd-buf poll) dropped it to ~0.55. **The readers are RISC-issue-bound, not NoC / bank bound**, which is why every extra per-row cycle showed 1:1.
  - **Issue path (gather.hpp `issue_row`):** `noc_id = noc_index` (compile time, so the NIU register addresses fold). Credit-batched in-flight accounting: the response counter is read only when the credits from its last reading are spent. The ids are read non-volatile after the ready wait (compiler barrier).
  - **Lamp L2, re-measured:** the faster issue moved the congestion-collapse cliff (Refinement 3) down and onto k = 3 too. At k = 3 with 8 in flight and the old (5, 2) stagger, 40-50% of calls at R = 4320 / 8064 took 10.6-13.5 us instead of 5.6 / 8.9. The new rule is `READ_BYTES_IN_FLIGHT = 4096`, `READS_IN_FLIGHT = 12`, giving 5 / 8 / 10 / 12 reads at k = 2 / 3 / 4 / 6, each the last stable step below that k's cliff.
  - **Lamp L6 (issue order):** `ROW_STAGGER_BY_K = {2: (5, 1)}`, `ROW_STAGGER_DEFAULT = (7, 1)`. At k = 3 / 8 in flight, (7, 1), (3, 4) and (1, 8) never collapsed while (5, 2) did. At k = 2, (7, 1) and (1, 8) collapse; (5, 1) is stable in 28 calls, and (11, 1) / (13, 1) collapse in 1-2 of 14.
  - **Lamp L1, sender-column rebalance (built, default on):** the column-0 cores (the ids row senders) finished ~0.45 us after the rest. Their sender reader now sheds 2 rows a block to the other reader (14 / 18). The host gives column 0 its own reader kernel pair, so the default positional split stays elsewhere. `SENDER_ROW_SHED_BY_K` / `SENDER_ROW_SHED_DEFAULT = 2`; 0 is byte-identical with one kernel pair. Per call, R = 4320 5.60 → 5.31, 8064 8.84 → 8.21, 7392 10.31 → 9.59, 2656 6.09 → 5.76, 8416 (with (5, 1)) 8.24 → 7.97. 1 and 3 fall in between, and 4 gives no gain.
  - **Lamp L1, path-aware split (built, parked `ROW_SPLIT_PATH = False`, `ROW_SPLIT_SLACK`):** each row goes to the NoC whose response path from the row's bank is shorter. The score separates into per-column and per-row torus terms, so the table is 29 entries. Both readers derive the same owner mask per block from the 32 ids, with a balance cap. It is bit-exact, but its per-block owner pass sits on the issue path. Ablation modes, since removed: the machinery with a positional owner +1.4-1.8 us, with the score computed +2.6-3.4 us; the assignment itself was −0.6 / +0.14 us at equal RISC cost, and the inverted score was just as slow. A static NOC0 share (`ROWS_NOC0` 13 / 19) was mixed (4320 −9%, 8416 +3%), so it stays at 16.
  - **Lamps L3 / L7, not built:** `row_slots` already holds the whole per-core assignment. The tail tilize is ~0.31 us at R = 8416, but a column-split last block would double that block's reads on an issue-bound RISC (~45 cycles a row), costing more than the ≤ 0.15 us it could hide.
- **Device perf**, eval harness (kernel us a call, mean of 5; same board, same session; before = Refinement 3 as committed):

  | R | k | before | after (two runs) | Δ |
  |---|---|---|---|---|
  | 2240 | 3 | 3.79 | 3.77 / 3.77 | ≈ 0 |
  | 2272 (guard) | 3 | 3.74 | 3.78 / 3.74 | ≈ 0 (+0-1%) |
  | 2528 | 4 | 5.59 | 4.59 / 4.56 | −18% |
  | 3296 | 2 | 4.59 | 4.64 / 4.67 | +1-1.7% (per call: 4.14 → 4.15, parity) |
  | 4320 (guard) | 3 | 7.02 | 7.06* / 5.85 (and 5.80-5.84 ×3) | −17% |
  | 5344 | 3 | 8.67 | 6.78 / 6.79 | −22% |
  | 6368 | 2 | 7.54 | 7.38 / 7.31 | −2-3% |
  | 7392 (k = 4 rep.) | 4 | 13.99 | 10.11 / 10.13 | −28% |
  | 8064 | 3 | 11.28 | 8.71 / 8.76 | −23% |
  | 8224 | 2 | 9.32 | 8.68 / 8.67 | −7% |
  | **8416 (PERF FOCUS)** | 2 | 9.17 | **8.46 / 8.49** (loose case 8.46) | **−7.6%** |

  \* One of the five calls collapsed; that is the one collapsed call in ~140 k = 3 calls measured after the change.
- **Device perf**, per call (median of 12-14, Tracy, `perf/test_nomic_moe_dispatch_percall.py`): 8416 8.65 → 7.97, 8224 8.74 → 8.12, 6368 7.07 → 6.73, 3296 4.14 → 4.15, 4320 6.45 → 5.30, 5344 8.17 → 6.26, 8064 10.76 → 8.21, 2240 3.40 → 3.32, 2272 3.29 → 3.31, 7392 13.40 → 9.56, 2528 5.08 → 4.11, 2656 8.02 → 5.72.
- **Bottleneck now**: the per-row issue (~45 cycles of address math and register writes a row), the start (~1.3 us: the ids DRAM round trip plus the row multicast) and the 0.31 us tail tilize. The reads-in-flight caps sit just under the collapse cliff, whose mechanism is still only understood empirically.
- **Accuracy achieved**: bit-exact (PCC = 1.0, atol = rtol = 0). Acceptance and precision baseline 36/36. Every perf / knob harness case passes its bit-exact check (87/87, including the parked path-split and shed variants).
- **Golden test progress**: 29/29 supported cells pass (11 `test_op` L1/L1, 3 loose perf cases, 15 regression). The 33 DRAM cells stay xfail (deferred regimes). SUPPORTED is unchanged.
- **Issues encountered**:
  - The collapse moved with the issue rate. Every cap and stagger choice above was judged per call (12-40 calls a point), because 5-call means hide it.
  - The legacy Refinement 3 stagger cases in the knob sweep needed the shed off: their (3, 4) / (5, 2) steps are not coprime with an 18-row half. They now carry `SENDER_ROW_SHED_BY_K=NO_SHED`, and their test ids are unchanged.
- **Tests added**:
  - `perf/test_nomic_moe_dispatch_percall.py` plus `perf/percall_report.py`: per-call device time per (R, knobs) case from the Tracy ops CSV.
  - Knob-sweep cases for `ROWS_NOC0`, `ROW_SPLIT_PATH` / `ROW_SPLIT_SLACK` and `SENDER_ROW_SHED_DEFAULT`.
  - `NMD_SKIP_CHECK` in the knob sweep and zone probe, for stubbed-kernel ablations.
