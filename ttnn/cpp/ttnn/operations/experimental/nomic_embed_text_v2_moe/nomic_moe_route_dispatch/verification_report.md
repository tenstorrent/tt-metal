# Verification Report: nomic_moe_route_dispatch

## Code Review

Fixed in place:
- **Slot-page header size duplicated between host and kernels.** The host sized `cb_slot_rows` with `REGION_WORDS = 32` while the kernels laid out the header with their own offsets (`kHeaderWriterBase + 8 = 26` words), with nothing tying the two. Now `region_words` and `handshake_page_bytes` are named CT args. `push.hpp` defines `kHeaderWords` / `kHandshakeWords`, and the reader and writer `static_assert` that they fit, so a header change that outgrows the host size fails to compile instead of corrupting L1.
- **Dead constant** `MAX_PAD_ROWS` removed from the op file. The pad list is no longer stored; pad rows are derived from the region header.
- **L1 ledger out of date** (see below): the closed-form total and the symbol table described the planned aliased inventory (304,184 B at 8x512) while `circular_buffer_bytes()` returns the built one (476,960 B). Both were rewritten to match the code.

Reviewed, no change:
- Helper usage: untilize, `matmul_block` (with an `In1BaseOffsetFn` for the W slice), `eltwise_chain` (logits sum + NaN-safe `Where` mask), `reduce` MAX/SUM with a precise-recip post-op, `mul` bcast-col, `tilize`, `prepare_reduce_scaler`, and the `ttnn.Mcast` / `McastArgs` helper for the W slices, the count table and the go signal. The raw-NoC parts are the intra-group partial exchange, the count gather, the per-row scatter pushes and the landing increments. They are justified in op_design.md (the helpers are one-to-many; these are all-to-all, many-to-one or data-dependent scatter).
- CB sync: every CB's push count equals its wait/pop count. `cb_x_rows` and `cb_mask_keep` are never popped. Each is used once per program and the pointers reset at launch, so this is harmless.
- Kernel syntax (`void kernel_main()`), include paths (`api/dataflow/...`) and TensorAccessor usage are correct. There is no `InterleavedAddrGen`.
- One dispatch per call: a single `ttnn.generic_op`. No tensor transforms, no readback.
- Prompt Rules: the MUST rules hold (one program, no host sync or readback, default HiFi4 + fp32 acc, caller config honored, `fp32_dest_acc_en=False` refused, no writes outside the outputs).
- Blocking-model fidelity: the block knobs are parameters with one source each. `rps`, `ht` and `G` come from x's shard spec through `plan()`; `w`, `k` and `max_blocks_per_core` come from `nomic_moe_dispatch.plan(R)`. `PUSH_READER_SHARE` resolves to `reader_tokens`, and `PUSH_BYTES_IN_FLIGHT` to `writes_in_flight`. Every token-axis CB scales with `rps` (the shard), and no CB scales with T. Each block operation runs once per core over the whole shard (untilize, matmul, softmax and tilize are one call each). The per-row push loop is inherent: each row is one scatter write to a data-dependent owner. Neither reader nor writer handshakes per unit.

Deviations from op_design.md (accepted, recorded in the ledger):
- Aliases A-D were not built. Headroom at model occupancy is > 660 KB at 8x512.
- One `cb_x_rows` replaces the reader/writer pair. The push split is by token (`PUSH_READER_SHARE`), not by tile row, so rps = 1 shards also use both RISCs.
- Selection is split over both DM RISCs (two 16 B handshake CBs).
- `combine_weights` is written by the leader (member 0) after its landing increment, not by member G-1.

Design-conformance performance finding (filed as perf refinements, not fixable in this pass): the op misses the prompt's "must beat stock on every row" contract on 3 of 7 rows (table below). The grid is fully used (96 token cores at 8x512, tilize on all 110), but the push phase is imbalanced 2.1x across cores. All pad rows land on the first groups' cores (`r mod T`), and with G=3, k=2 the middle member writes twice. See Refinement 3.

## Registry Conformance
- `INPUT_TAGGERS` (`alignment`, signature `(inputs, axes)`), `SUPPORTED` (all six TARGET axes), `EXCLUSIONS = []` and `validate()` are present. `validate()` checks malformed calls (ValueError) first, then SUPPORTED per axis, then EXCLUSIONS, then the contract refusals. The entry point calls `validate()` first. The op file does not declare `INVALID`.
- No SUPPORTED auto-fixes were needed: no XPASS drift.
- INVALID audit (`feature_spec.py`): one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`, the canonical structural entry. It is single-tensor and inert under the current TARGET (bf16 TILE only). There are no cross-tensor entries. Not a norm op, so no weight-canonicalization cells apply.

## L1 Ledger Audit
- **Currency**: every one of the 19 declared CBs has a row in the "Implemented" table, and the size expressions match `_cb_specs()`. The closed-form total and the symbol table were stale (they described the planned aliased inventory); both were rewritten to the implemented form. The planned table is kept and marked superseded, because its alias proofs are the first lever if a later regime runs short of L1.
- **Capacity vs live set**:
  - `cb_router_weight` is 24 tiles while each member uses ht. This is justified: every slice multicast lands at its own offset.
  - `weight_tiles` is full fp32 tiles with column 0 used. This is justified: the output ABI is routing_metadata's tile contents (lamp L6).
  - No collapsed extents.
- **Page format vs DEST**: fp32 pages with `fp32_dest_acc_en` on, for the logits and softmax chain. The bf16 pages are pure data (x rows, gathered rows) or scalers. No finding.
- **Disjoint lifetimes**: aliases A-D were dropped with a stated reason (headroom, plus alias A's cross-core ordering hazard). That is acceptable.
- **Bounds**: every symbol is bounded with its predicate (`rps <= 4`, `G in {2,3}`, `max_blocks_per_core*w <= 63`, ...). The total is closed-form.
- **Block-size defaults**: sharded, one block per core = the whole resident shard (input) and all owned output blocks in one tilize call. Held.
- **Data-movement budget**: present and consistent with the built split. x crosses no DRAM. W crosses DRAM once and is multicast. Rows cross the NoC once. The cheapest-traffic split is the implemented one, with the alternatives ranked in op_design.md.
- **Per-core total**: 476,960 B at 8x512 (`cb_gathered_rows` scales with R; the token CBs scale with `rps`, `G` and `ht`).

## Precision Baseline

`combine_weights` (the only float output) against fp64 `p[t, sel_op[t,k]]`. Rerouted counts are against the fp64 top-2. All exact checks pass (`helpers.check_output`). Test: `test_nomic_moe_route_dispatch_precision_baseline.py`.

| Shape / profile | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / median (fp32) | got/true ratio p5 / p50 / p95 | Rerouted |
|---|---|---|---|---|---|---|---|
| 2x512 layer0 | > 0.9999 | 1.77e-3 | 2.93e-4 | 1.38e-3 | 43423 / 12241 | 0.99784 / 0.99897 / 1.00004 | 3/1024 (0.29%) |
| 4x512 layer0 | > 0.9999 | 1.47e-3 | 2.87e-4 | 1.36e-3 | 44862 / 12058 | 0.99785 / 0.99899 / 1.00005 | 4/2048 (0.20%) |
| 8x384 layer3 | > 0.9999 | 1.85e-3 | 3.10e-4 | 1.36e-3 | 52819 / 12584 | 0.99778 / 0.99894 / 1.00007 | 2/3072 (0.07%) |
| 8x512 layer0 | > 0.9999 | 1.67e-3 | 2.91e-4 | 1.38e-3 | 45583 / 12237 | 0.99782 / 0.99897 / 1.00004 | 6/4096 (0.15%) |
| 8x512 layer5 | > 0.9999 | 7.97e-4 | 1.60e-4 | 1.14e-3 | 40861 / 9259 | 0.99805 / 0.99914 / 0.99988 | 22/4096 (0.54%) |

**Assessment**: inside the contract on every case: weight error <= 1.85e-3 against a 3e-3 bar, reroutes <= 0.54% against 0.6%, worst reroute gap <= 5.5e-4 against 6e-3. These numbers are in the stock path's measured range (3.3e-4 to 2.6e-3, 0.07 to 0.42%). The 8x512 layer-5 profile (low-gap logits) is closest to the reroute bar. The got/true ratio has a small negative bias (median about 0.999) inside a 2.2e-3 p5-p95 band. That is not the tight off-1.0 cluster of a scale bug, and it is consistent with W passing through SrcA at tf32 in the logits matmul, which stock shares. The large ULP counts reflect that same tf32 rounding against an fp64 reference, not a defect.
**Recommended tolerances**: the contract's bars (max |w - p64| <= 3e-3, rerouted <= 0.6%, gap <= 6e-3). PCC >= 0.9999 on combine_weights.

## Verifier CLI Summary
(`/tmp/nmrd_golden_v2/verifier_report.json`, committed copy: `pipeline_logs/verifier_report.json`)
- supported_pass: 21 (14 INPUTS tile_aligned cells + 7 LOOSE_CASES perf rows)
- xfail_expected: 82 (in_placement l1/dram_interleaved, out_buffer_type DRAM, alignment ragged; each covered by Refinement 1 or 2)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 41 (test_regression.py)

Harness note: under `eval_test_runner.sh`, 20 of the 41 regression tests fail with `L1ProfilingError`. The runner exports `EVAL_CAPTURE_L1=1`, and these tests either have no device fixture (`test_default_compute_kernel_config`, `test_exported_helpers`) or are refusal checks that allocate nothing (`test_malformed_calls`, `test_out_of_contract_refused`, `test_ragged_batch_refused`). Run directly (`scripts/run_safe_pytest.sh eval/golden_tests/nomic_moe_route_dispatch/test_regression.py`), all 41 pass. This is an `eval/l1_profiling.py` / golden-suite interaction, not an op defect; it needs a harness-side fix (skip capture for tests without a device or without an allocation).

## Performance (device kernel ns, warm call, model L1 occupancy)

`test_nomic_moe_route_dispatch_perf.py` under `--profile`:

| Row | Phase 0 us | Stock us | Target us | vs stock |
|---|---|---|---|---|
| 8x512 (primary) | 58.7 | 45.6 | 5.6 | **slower** |
| 6x512 | 31.4 | 32.6 | 4.3 | faster by 4% |
| 8x384 | 31.9 | 32.6 | 4.3 | faster by 2% |
| 4x512 | 29.5 | 30.3 | 2.9 | faster by 3% |
| 8x256 | 32.3 | 30.1 | 2.9 | **slower** |
| 2x512 | 24.7 | 22.6 | 1.5 | **slower** |
| 4x256 | 22.2 | 23.5 | 1.5 | faster by 6% |

Zone profile at 8x512 (profiler build, 85.7k cycles; zones added temporarily and reverted):

| Phase | Cycles (span) |
|---|---|
| W slice fetch + multicast (compute waits) | 0 to ~16k |
| matmul + partial exchange | to ~20.5k |
| logits + softmax | ~20.5k to ~32k |
| selection + count row | ~32k to ~42k |
| table gather + multicast | ~42k to ~44k |
| pushes | ~45k to ~83k (R_push median 17.9k, max 37.7k) |
| tilize | ~83k to ~85.7k |

The landing barrier waits for the slowest pusher. These findings feed Refinements 3-5.

## Recommendations
- Perf is the main open debt. The production config is already supported, so the first perf phase (Refinement 3) needs no unlock. The queue order puts it third (2:1 cadence), after the two generality refinements, which must not regress the production rows.
- Environment: `TT_METAL_HOME` points at `/localdev/kgrujcic/tt-metal`, so profiler CSVs land under that tree, and the run wrapper prints "no ops_perf_results CSV" even when one was produced.
- L1: 8x512 has > 660 KB of headroom at model occupancy. The planned aliases (about 173 KB) are the first lever if the interleaved or DRAM-output regimes need room.
- Precision: the small negative bias in combine_weights (about -0.1% median) is within contract and matches the stock path. A full-fp32 W path in the matmul would be the lever, but no failing cell asks for it.
