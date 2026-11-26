# Verification Report: gen_qkv_causal_conv1d_silu

Blackhole, 11×10 compute grid (tests run on a 1×1 `MeshDevice`, so the mesh entry path is the one
exercised). Every number below is device kernel time from the golden runner's real-time profiler
(`device_kernel_ns`, strict op window), or from a pytest run of the files named.

## Code Review

### Fixed in this pass

| # | Finding | Fix |
|---|---|---|
| 1 | **Acceptance test harness bug.** `test_repeat_is_bit_identical` parametrizes over `pytest.param` objects. pytest unpacks those into the case dict, but the test then called `case.values[0]` on that dict, so it raised `TypeError` before the op ran. The BB-06 check never executed. | Changed it to use `case` directly; the test's intent is unchanged. Both cases (RM 001, TILE 005) now pass. |
| 2 | **DRY violation in CB sizing.** Each CB capacity was computed twice from the knobs: once in `footprint()` (the L1-budget check `select_plan` uses) and once in `create_program_descriptor` (the allocation). Editing one formula without the other would make the budget check and the real allocation silently disagree. | Added a single `cb_layout(block_ch_tiles, block_rows, regime)` that returns `(cb, page bytes, pages)` for every CB. `footprint()` sums it and the descriptor allocates from it. The program is identical (checked: per-core L1 peak equals the formula on every golden cell). |
| 3 | **Incomplete program-cache key (latent BB-07).** The key did not include `page_channels`. Token-paged `[1,3,C]` and channel-paged `[1,C/P,3P]` history have the same shape when C = 3P, so once channel-paged lands, a token-paged cached program could be replayed for a channel-paged call. | Added `page_channels` to the key now. This is a no-op until Refinement 1. |
| 4 | `HISTORY_ROWS` (op file) duplicated `HALO_ROWS` (descriptor). | The op file now imports `HALO_ROWS` (one definition of `TAP_COUNT − 1`). |
| 5 | **Coverage gap.** The TILE regime with a multi-tile-row block (`block_rows ≥ 2`: several untilized tile-rows copied into one `cb_rows` slot) was never run. The default plan never selects it, but any `block_rows` co-tune in a perf refinement would. | Added `test_tile_multirow_block_pinned` (golden cell 005, `plan_limits(max_block_rows=2, max_cores_per_group=1)`) to `test_gen_qkv_causal_conv1d_silu_debug.py`. It passes. |
| 6 | `op_design.md` still listed `tap_multicast_per_group` as *deferred*, but the implementer built it (measured). Later refinements would plan against a stale design. | Added an as-built verifier note under the Regimes table: tap multicast built, SiLU on the packer thread, `OVERLAP_BLOCK_ROWS = 1`, and the measured 004 profile. |

### Reviewed and kept (no change)

- **`conv_silu_block` uses raw LLK.** This is justified and within the scope boundary. `eltwise_chain` cannot run an SFPU element on the sticky DEST-accumulated slot (`chain.inl:3397`), and `BinaryFpu` has no `bcast_row_idx` for the packed tap tile. The implementation's effects match the plan: 4 MACs accumulate in one DEST slot in fixed order p = 0..3, SiLU runs in DEST, there is one pack, and no partial sum reaches L1.
- **The reader uses raw NoC reads, not `read_sticks_for_tilize`.** That helper barriers after every stick and cannot take a 3-row halo prefix from another source. The reader issues one tile-row of stick reads and then waits on a single barrier (the catalog `double_buffer` rule).
- **The tap multicast uses the kernel_lib helper** (`ttnn.Mcast` + `McastArgs` `optional_sender/receiver`), not raw semaphores. ✓
- **Block knobs.** `block_ch_tiles`, `block_rows` and `cores_per_group` come from one `Plan` and reach every kernel as CT/RT args. The buffer depths are host constants, used only in `cb_layout`. No CB capacity scales with `T` or `C`. ✓
- **Default `block_rows` = 1.** This departs from the "coarsest block that fits" default, and the departure is justified by measurement in the ledger: 004 went from 153 to 129 µs and 020 from 75.8 to 73.4 µs. The windowed-tilize and MAC phases run one after another inside compute, so a coarse block starves the writer.
- **Constants restated in the kernels** (CB indices, `TAP_COUNT`, `HALO_ROWS`, `TAP_ROW_STRIDE`). These are fixed mechanism/math constants, not tunable block knobs. Noted, not changed.
- **Expression check.** With `block_rows = 1`, a block is one tile-row, so the per-tile-row handshakes in the reader and writer are per-block. For `block_rows > 1`, the reader reserves one slot per block and pushes per tile-row. The block factor still controls the live set (slot size, `cb_window_tiles`), so this is not an expression failure.
- **CB sync counts.** All verified, push = pop:
  - `cb_rows`: one block pushes `35 + 32(M_b−1)` and pops `1,1,1,29` per tile-row, with `1,1,1,32` on the last, which also equals `32·M_b + 3`.
  - `cb_window_tiles`: `4·w·M_b` both sides.
  - `cb_output_tiles`, `cb_input_tiles`, `cb_untilized_rows`: `w` per tile-row on both sides.
  - `cb_tap_tiles`: `w` pushed once and popped once.
- **Hygiene.**
  - Kernels use `TensorAccessor` throughout, `void kernel_main()`, and `api/dataflow/dataflow_api.h`.
  - The writer flushes per tile-row and has a final `noc_async_write_barrier()`.
  - The reader ends with read, write and atomic barriers (BB-03).

### Prompt rules (`eval/prompts/gen_qkv_causal_conv1d_silu.txt` ## Rules)

| Rule | Status |
|---|---|
| One program per call; no native-op call; no composition | ✓ one `ttnn.generic_op` (golden `device_num_programs = 1` on every cell) |
| ROW_MAJOR input read by the kernels (no tilize program / `to_layout`) | ✓ RM sticks read into `cb_rows`, tilized in-kernel |
| Inputs unchanged; q/k/v new tensors with the native spec | ✓ (acceptance + golden check history, taps, input) |
| Accept/refuse `channel_chunk_size` as native; result chunk-independent; L1 within BB-05 | ✓ chunk validated only, never reaches the program; `test_chunk_size_invariance` passes; `select_plan` budgets 1,048,576 B minus the L1-output share |
| `actual_start` not read on the host | ✓ never read anywhere |
| `compute_kernel_config=None` → `default_compute_kernel_config()` only | ✓ |
| Cores from `device.compute_with_storage_grid_size()` | ✓ |
| Single device and mesh device | ✓ tests run on a 1×1 `MeshDevice`; `sequence_parallel_axis` of size > 1 refused with `UnsupportedAxisValue` |
| BB-03, BB-06, BB-07, BB-08 | ✓ final barriers; bit-identical repeats (both regimes); cache hit rewrites only addresses (hit path checked against the reference in the supplementary test); `get_last_plan()` / `plan_limits` |

There were no soft-guidance (prefer/consider/avoid) advisories.

## Registry Conformance

- `INPUT_TAGGERS` has the six taggers verbatim from the prompt, in order, all with the `(inputs, axes)` signature. `SUPPORTED` has all 13 TARGET axes. `EXCLUSIONS = []`.
- `validate()` runs ValueError checks, then pinned axes, then taggers, then SUPPORTED, then EXCLUSIONS. Refusals are `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls `validate()` first.
- The op file does **not** declare `INVALID`. ✓
- No XPASS evidence, so no SUPPORTED auto-fixes.
- **INVALID audit** (`feature_spec.py`): one entry, `{"chunk_blocks": "single", "channel_count": "wide"}`.
  - Both axes derive from the same call's `(Q, K, V, chunk)`, so there is no cross-tensor coupling. ✓
  - **Caveat for the user:** the entry encodes the *native kernel's* L1 refusal, not a structural impossibility of the math. This op ignores `channel_chunk_size` and would accept these calls. That is defensible as "outside the native parameter space the suite mirrors", and no golden case matches it, but strictly it is a native-implementation limit rather than an INVALID cell. I did not edit it.
  - The canonical bf8b+ROW_MAJOR entry does not apply (TARGET dtype is bf16 only). The norm-style no-weight canonicalization does not apply either.

## L1 Ledger Audit

- **Ledger currency.** Six rows (two TILE-only) match the six declared CBs. The size expressions match the allocation: they are now produced by `cb_layout()`, and the golden per-core L1 peak equals the formula on every passing cell (001 18,816 B; 005 54,016 B; 016/020 150,528 B; 004 301,056 B).
- **Capacity vs live set.**
  - Over: none. `cb_window_tiles = 4·w·M` is fully live, because tilize finishes the block before the MAC consumes it.
  - `cb_tap_tiles` uses only 256 B of each 2 KB page (rows 0, 2, 4, 6). This is justified: `bcast_row_idx` needs a tile page, and packing already cut the CB from 4w to w pages.
  - Under: none. Every "spans" axis scales the capacity; the "streams" axes use fixed windows.
- **Page format vs DEST width.** All pages are `Float16_b`. With `fp32_dest_acc_en` on, each page has a mechanism reason (exact data-movement copies, a bf16 operand, or the bf16 output ABI). No partial sum is ever packed. No Float32 pages, and no truncated accumulator.
- **Disjoint lifetimes.** The one candidate pair (`cb_untilized_rows` / `cb_window_tiles`) overlaps at the untilize→tilize boundary, as justified in the ledger.
- **Bounds.** `w ≤ min(255, C_t)` with `w | C_t`. `M ≤ OVERLAP_BLOCK_ROWS` (or the `plan_limits` cap), and `footprint(w, M) ≤ budget` is checked in `select_plan`. `T` and `C` appear in no capacity. The total is closed-form: RM `w·(12288·M + 6528)` B, TILE `w·(12288·M + 14720)` B. The largest golden footprint is 301 KB of the 1 MiB budget (004), so there is plenty of room for depth co-tunes.
- **Data-movement budget.**
  - It is present and consistent with the code (tap multicast is the built regime).
  - The cheapest-traffic split (channel groups only) is recorded with a positive reason. It is the `cores_per_group = 1` value of the same built knob, so `select_plan` picks it whenever it is also the minimum makespan. For 004 it would idle 30 cores for a 1.8% byte saving.
- **Block-size defaults.**
  - Interleaved, spread over the full grid: 004 uses 110 cores.
  - 001–003 use 80 of 110 cores, because uniform channel groups must divide `C_t = 80`. This is a design perf lamp. I measured that it is **not** an occupancy loss: 001 is faster on *fewer* cores (see Performance).
  - `block_rows` departs from the default with a measured justification (above).
- **Findings filed:** none. The one cheap fix (single-source capacities) was applied in place.

## Precision Baseline

`tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/test_gen_qkv_causal_conv1d_silu_precision_baseline.py`
uses only golden cells, HiFi4, against a float64 reference over the bf16 operands, with q|k|v concatenated.

| Shape (golden cell) | fp32 DEST | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 | ULP max (off zero) | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| (1, 32, 2560) — 001 | False | 0.999987 | 0.1087 | 0.00328 | 0.00556 | 18 | 19 | 1.0008 [0.9927, 1.0113] |
| (1, 128, 2560) — 003 | False | 0.999987 | 0.1247 | 0.00324 | 0.00560 | 17 | 29 | 1.0008 [0.9927, 1.0112] |
| (1, 2048, 2048) — 020 | False | 0.999987 | 0.1688 | 0.00326 | 0.00558 | 17 | 31 | 1.0008 [0.9927, 1.0112] |
| (1, 4096, 2560) — 004 | False | 0.999987 | 0.1808 | 0.00326 | 0.00561 | 18 | 39 | 1.0008 [0.9927, 1.0112] |
| (1, 128, 2560) — 011, cancelling taps | True | 0.999998 | 0.0078 | 0.00047 | 0.00177 | 0 | 1 | 1.0000 [0.9972, 1.0029] |

**Assessment.**
- The error does not depend on shape or plan: four different plans give identical statistics.
- With a 16-bit DEST (the native default), the four products are summed in 16-bit, so ULP p99 is about 17–18. The error is absolute in Σ|x·w|, so it shows up as large ULP counts on outputs near cancellation, while relative RMS stays at 0.0056 (56% of the 0.01 budget).
- With fp32 DEST, the result is within 1 bf16 ULP of the correctly rounded value everywhere, and the per-element FP32_DEST_BOUND holds on the cancelling-taps input.
- **Scale-bug triage:** clear. The got/true ratio sits on 1.0 with a broad, symmetric spread, which is rounding noise, not a constant scale. The 16-bit-DEST median of 1.0008 is a small rounding bias (≈0.1 bf16 ULP), not a scale error.
- "ULP max (off zero)" is taken over elements with |ref| ≥ 0.1·std. Near-zero outputs whose sign differs from the reference span the whole bf16 ordering without being a large error.
- Golden LoFi cells (006, 010, 015, 021) have PCC ≈ 0.99985 and rel-RMS ≈ 0.03. They are gated on PCC only, per the precision convention.

**Recommended tolerances:** the defaults in `helpers.TOLERANCES` hold: PCC ≥ 0.999 and relative RMS ≤ 0.01 for both `(bf16, fp32_dest_acc_en)` keys. FP32_DEST_BOUND holds at fp32 DEST and HiFi4.

## Verifier CLI Summary

Final run (clean JIT cache) is at `/tmp/gqkv_verify_final/`. `verifier_report.json` is also copied next to the op.

- supported_pass: 17
- xfail_expected: 4 — conv1d-012 (channel_paged), conv1d-013 (column_window), conv1d-018 and conv1d-019 (both). All four are covered by Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Unit tests: `tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/` — **45 passed** in both `--dev` and production modes. That is the acceptance file (37), the supplementary file (3) and the precision baseline (5).

TARGET − SUPPORTED = {`history_layout: channel_paged`, `input_window: column_window`}. Both are in Refinement 1, with no omissions.

## Performance (flagged targets and measured profile)

| Case | Device time (final run) | must (native) | target | stretch | DRAM floor |
|---|---|---|---|---|---|
| conv1d-001 decode T32 | 8.70 µs | 30.5 | 16.97 ✓ | near launch cost | 0.71 |
| conv1d-002 decode16 T64 | 10.35 µs | 43.14 | 18.45 ✓ | near launch cost | 1.35 |
| conv1d-003 prefill T128 | 13.19 µs | 48.08 | 22.54 ✓ | near launch cost | 2.63 |
| conv1d-004 prefill T4096 | 130.45 µs (128–131 across runs) | 346.94 | 163.98 ✓ | 117.13 ✗ | 81.99 |

All four targets beat both the "must" (native) and "target" bars. Headroom remains on 004 towards its 117 µs stretch, and on 001–003 towards launch cost.

**Ablations** (define-gated kernel variants injected from a throwaway harness under the strict op window, 3 reps each, medians; the full variant was correctness-checked):

| Variant | 001 | 004 |
|---|---|---|
| full | 8.62 | 130.37 |
| no DRAM input reads / output writes (compute-only) | 4.42 | 108.70 |
| no compute (CB bookkeeping only: dataflow-only) | 7.04 | 99.07 |
| windowed tilize only, no DRAM | 3.22 | 15.58 |
| compute-only, SiLU removed | — | 40.50 |
| compute-only, 1 MAC instead of 4 | — | 99.47 |
| full, SiLU removed | — | 108.44 |

What the ablations show:
- **004 is compute-bound.** Compute alone is 109 µs; dataflow alone is 99 µs, which is already ~81% of DRAM peak.
- **The precise SiLU on the packer thread is ~68 µs of that compute** (about 0.7 µs per output tile).
- The 4 MACs cost about 9 µs, and the windowed tilize about 12 µs above launch.
- Full time (130) exceeds max(compute, DM) = 109 µs by about 21 µs of overlap loss.
- **001 is dataflow-latency-bound.** Dataflow alone is 7.0 of the 8.6 µs.

**`block_ch_tiles` sweep** (forced plan, same harness):

| Case | `block_ch_tiles` (cores) → device time |
|---|---|
| 001 | **w=1 (80 cores) 8.68 µs** (plan) · w=2 (40) **7.35** · w=4 (20) 8.92 · w=5 (16) 9.95 · w=8 (10) 14.0 |
| 002 | **w=2 (80) 10.06** (plan) · w=4 (40) 10.32 · w=8 (20) 15.1 |
| 003 | **w=4 (80) 13.32** (plan) · w=8 (40) 16.7 · w=16 (20) 25.1 |

`select_plan`'s tile-count makespan model picks the best `w` for 002 and 003. For 001 it is 15% off: w=1 issues about 3.4k sub-64 B DRAM requests across 80 cores, and those are request-rate-bound.

## Recommendations

- **Refinement queue:** R1 (reader addressing: column_window + channel_paged), then R2 (perf, latency-bound 001–003), then R3 (perf, SFPU-bound 004). Rationale is in `op_requirements.md`.
- **Do perf ablations through kernel defines, never by editing kernel sources.** In this pass, a source-edit ablation, once reverted, left **stale kernel binaries in the JIT cache** (`built/tt-metal-cache*/kernels/gen_qkv_causal_conv1d_silu_*`). Later runs of the unmodified op produced NaN outputs with compute-only timings (2.9 / 6.1 µs) until I deleted those directories. Define-gated variants (`KernelDescriptor.defines`, which are part of the kernel hash) are safe. If a source edit is unavoidable, delete the op's cached kernel directories afterwards and re-run the acceptance suite.
- **W not a multiple of 32.** The native op accepts this beyond the window; it is a corner outside this suite. Today it is refused only indirectly (W ≠ C makes the tag `column_window`). When R1 opens `column_window`, refuse it explicitly with `UnsupportedAxisValue`, or support it. The TILE path's `input_width_tiles = W // 32` would be wrong for a padded width.
- **No L1 pressure.** The largest footprint is 301 KB of the 1 MiB budget, so depth / `block_rows` co-tunes in R3 have room.
- **Accuracy at 16-bit DEST** sits at 56% of the rel-RMS budget, inherent to the native default (16-bit accumulation). Nothing to do unless a refinement changes the summation order. Any SiLU change in R3 must keep this baseline.
