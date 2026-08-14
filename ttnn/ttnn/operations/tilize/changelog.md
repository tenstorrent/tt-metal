# tilize — changelog

## Phase 0 — implement op_design.md (correct + measured + classified)

**Status:** acceptance suite green (52/52, `--dev` and production), regime-pinned
tests green (9/9), lever structural pins green (15/15). Perf gate run and recorded
below.

### What landed

| Piece | Where |
|---|---|
| Op file (4 registry declarations + `validate()` + entry point) | `tilize.py` |
| ProgramDescriptor + `derive_blocking()` (the single source for every knob) | `tilize_program_descriptor.py` |
| Reader / compute / writer kernels | `kernels/tilize_{reader,compute,writer}.cpp` |
| Regime-pinned tests (design §5.3) | `tests/.../test_tilize_regimes.py` |
| Lever structural pins | `tests/.../test_tilize_levers.py` |
| Perf bench + lever counterfactual arms + classification ablation | `tests/.../_bench_tilize.py` |
| Lever ledger (29/29 catalog rows, `verify_levers` clean) | `lever_ledger.json` |

**Blocking model, as implemented (design §1).** Work unit = one **block** =
1 tile-row × `WT_CHUNK` tile-columns. `derive_blocking()` owns all of it:
`WT_CHUNK` is the coarsest exact divisor of `WT` that both fills the grid
(`n_want = ceil(NUM_CORES / NT_H)`) and fits `CB_L1_BUDGET`; `NT_BLK` (=1),
`CB_DEPTH` (=2) and `NUM_CORES` are named parameters, never inlined. Both CBs are
`CB_DEPTH * NT_BLK * WT_CHUNK` pages — **no CB is a function of `WT`, `NT_H` or any
tensor dimension** (pinned by `test_cb_footprint_is_bounded_in_w`). Blocks are
W-chunk-major and split with `split_work_to_cores(..., row_wise=True)`.

**Deviations from the design** (both surfaced in the Phase-0 commit message):
one reader *file* with a compile-time regime selector instead of two files (the
task pins the kernel file list; the `R_ALIGNED` branch is the library helper
call verbatim, so the hot path compiles byte-identically); the `R_PAD` branch and
the writer are raw dataflow, justified in-file by design §7 rejections 1–2 (the
library reader cannot fill, and `write_sticks_after_untilize` is the inverse
direction).

### Golden suite (informational — the acceptance suite is the phase gate)

`eval/golden_tests/tilize/test_golden.py`: **102 passed, 246 xfail (strict), 592
skipped (INVALID), 0 failed.** Every xfail is an axis this phase deliberately did
not build (sharded placement, integer dtypes, tiny tiles, retile) or one of the
two EXCLUSIONS below.

Two gate fixes came out of that run:

1. **Gate order.** `validate()` now projects the axes from the RAW call and runs
   SUPPORTED/EXCLUSIONS *before* `_canonicalize()`'s shape-legality checks.
   Previously a rank-0 (scalar) cell hit `ValueError: output_padded_shape ... must
   have the same rank as the input` before the `rank` axis was ever consulted, so
   an out-of-rectangle cell was refused with the wrong exception type instead of
   the typed `UnsupportedAxisValue` the registry contract promises.
2. **EXCLUSION: padding + a widening cast with a non-zero fill.** The fill is
   materialized into the input CB, so it is necessarily packed in the INPUT
   element format (design §10 — packing it in `output_dtype` is garbage the
   moment a cast is requested). For `bfloat16 -> float32` that means a fill which
   is inexact in bf16 (e.g. 10.2) lands as 10.1875 in an fp32 output while the
   oracle expects 10.2 — measured as exactly the 0.0125 ATOL delta the cell
   reported. Zero fills are exact and stay supported. Refinement candidate: give
   the pad path a second fill word in the OUTPUT format applied after the cast.

`eval/golden_tests/tilize/test_regression.py` fails 12/27 — every one on an axis
outside the Phase-0 rectangle (int32/uint16 dtypes, legacy/nd sharding). That file
carries no xfail machinery, so those are the expected "future refinement" reds.

### Perf gate

Box: Wormhole B0, 8×8 compute grid, AICLK 1000 MHz (measured 0.985 GHz).
Numbers are `DEVICE KERNEL DURATION [ns]`, in-process device profiler, one warm
launch then one measured launch per variant (device kernel time has no warm-up
transient — `/perf-measure` measurement discipline).

#### 1. Bound classification — the mandatory A0 ablation (design §9.1)

Payload stubbed, synchronization (CB reserve/push/wait/pop, barriers, loop trip
counts) kept intact. bf16.

| shape | full | no-compute | no-DM | sync-only | verdict |
|---|---|---|---|---|---|
| (a) `[1,1,2048,2048]` | 92,859 | 88,712 (−4.5%) | 4,937 (−95%) | 783 | **DM-bound** |
| (b) `[1,1,32,16384]` | 13,315 | 12,894 (−3.2%) | 1,566 (−88%) | 777 | **DM-bound** |
| (c) `[1,1,8192,1024]` | 170,508 | 173,513 (+1.8%, noise) | 8,940 (−95%) | 925 | **DM-bound** |
| (d) `[1,1,32,64]` | 2,889 | 2,154 (−25%) | 864 (−70%) | 658 | **overhead-bound** (658 ns sync floor on 2 tiles of work) |

Removing compute moves the wall by ≤4.5% on every real-work shape; removing data
movement collapses it by 88–95%. **The op is data-movement-bound, so the
`/perf-ceiling-dm` target applies** — this is the baseline claim the DM levers
below rest on. The smallest regime is the exception: it is dominated by the
~660 ns dispatch/sync floor, which is why every per-core-overhead lever was also
counterfactualed there (master.md B0).

#### 2. Ceiling vs measured

`/perf-ceiling-dm` Mode B (audit), per-core groups, depth-2 CB → `max(read, write)`,
then Step 4b (cap at `dram_peak` = 288 GB/s, bracket the contention):

Shape (a), 64 cores, 1 block/core, `WT_CHUNK=64`: read = 32 sticks × 4096 B
(`ONE_FROM_ALL` 4.64 µs … `ALL_FROM_ALL` 116.8 µs), write = 64 tile pages × 2048 B
(`ONE_TO_ALL` 5.0 µs … `ALL_TO_ALL` 51.6 µs). DRAM floor = 16.78 MB / 288 GB/s =
**58.3 µs**. `op_target = max(per-core NoC bound, DRAM floor)` = **[58.3 … 116.8] µs**.

| shape | DRAM bytes | DRAM-floor target | measured | achieved (target/measured) | achieved BW |
|---|---|---|---|---|---|
| (a) `[1,1,2048,2048]` bf16 | 16.78 MB | 58.3 µs | 92.9 µs | **0.63** | 180.7 GB/s |
| (b) `[1,1,32,16384]` bf16 | 2.10 MB | 7.3 µs | 13.3 µs | 0.55 | 157.5 GB/s |
| (c) `[1,1,8192,1024]` bf16 | 33.55 MB | 116.5 µs | 170.5 µs | **0.68** | 196.8 GB/s |
| (d) `[1,1,32,64]` bf16 | 8 KB | 0.03 µs | 2.9 µs | n/a (overhead-bound) | 2.8 GB/s |

The measured numbers land **inside** the predicted bracket, near the contended
end — exactly where the skill says an interleaved round-robin op lands. 63–68% of
DRAM peak on the grid-filling shapes; the tech report's ceiling for the recipe we
have *not* built (bank-adjacent readers + per-reader VCs, ledger rows A3/B10) is
~92%, which is the largest identified headroom.

**Design reconciliation (Mode A → Phase 0).** `op_design.md` §1.3 ranked the 2-D
block split (candidate 1) the winner over the pure height split (candidate 2) on
two properties: grid occupancy on short/wide shapes, and L1 boundedness in W.
Both were confirmed on device, and candidate 2 turned out to be not merely slower
but **unbuildable** on the mandatory regime: with `w_split=0` on `[1,1,32,16384]`
the input CB becomes `WT=512` tiles and the program fails to allocate L1. That
counterfactual is pinned as a strict-xfail bench arm. No divergence that would
call the algorithm choice into question.

#### 3. Lever ledger (all 29 catalog rows — `lever_ledger.json`)

`python3 -m eval.verify_levers ttnn/ttnn/operations/tilize/lever_ledger.json --bench
tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py --phase 0` → **clean**
(0 blocking, 0 signal, 0 possibly-unlocked). Measured closures:

| lever | knob | on ns | off ns | delta | disposition |
|---|---|---|---|---|---|
| **B9** reader NOC0 / writer NOC1 | `noc_split` | 93,626 | 246,415 | **2.63×** | applied — biggest single win |
| **A0** full-grid active cores | `w_split` / `multicore` | 92,859 | 382,290 | **4.12×** | applied (4.53× on (b), 4.76× on (c)) |
| **B7** one barrier per block | `block_write` | 170,508 | 194,762 | **14.2%** | applied |
| **B5** whole-page transactions | `page_write` | 173,311 | 177,432 | 2.4% | applied (small but never negative; precondition for B7) |
| **B0** smallest-regime check | `block_write` @ (d) | 2,889 | 2,942 | noise | applied — no regression on the tiny regime, so B7 is ungated |
| **F25** fp32 DEST, gated on the dtype pair | `fp32_dest` | 191,169 | 192,594 | 0.7% cost | applied — buys bit-exact fp32→fp32 for ≤3% |
| **A1** `row_wise=True` | `row_wise` | 92,859 | 95,146 | 2.4% = noise | measured-no-payoff (kept: design-binding, free) |
| **C16** depth-2 CBs | `double_buffer` | 92,859 | 90,761 | −2.3% = noise | measured-no-payoff (kept: API default; DRAM-bound path leaves nothing to overlap) |
| **D20** compile-time regime selector | `regime_select` | 93,626 | 92,225 | noise | measured-no-payoff (kept: §5.1 correctness contract) |

Structurally closed, each pinned by a passing test in `test_tilize_levers.py`:
**A4** (no cliff-core width — `WT_CHUNK` divides `WT` exactly), **B11**
(every transaction is a 32 B multiple by construction), **B12** (no operand is
read by more than one core — no semaphores, no mcast anywhere), **C17** (RM in /
TILE out can never alias), **F23** (no caller-supplied precision knob exists),
**F26** (tilize is an FPU phase — `Fp32Mode::Fast` always), **F27** (no
arithmetic, so no fidelity to lower).

Open and carried forward (the next run's candidate list, **not** filed as work
here): **A3 + B10** bank-adjacent readers + per-reader VCs (predicted ~35% —
the gap to the 92% DRAM recipe), **C14 + C15** zero-copy sharded I/O (blocked on
`SUPPORTED[shard_api]`), **B8** trid double-issue (needs a custom reader;
measurable on (c), which is benched at 4 blocks/core), **B13** stateful writer
(must be *swept* across transaction size, not argued), **B6** one-packet fast
path (pulls against B5 — only a sweep can price the pair), **D18/D19/D21**
(applied in the kernels, off-arms not built), **F24** `bfp8_pack_precise` (bf8b
is emitted but has no bench arm), **E22** (whole-model, out of scope).

#### 4. Cumulative bench set (carried forward — every later phase re-measures ALL of these)

| shape | dtype | ns | GB/s |
|---|---|---|---|
| `[1,1,2048,2048]` | bf16 | 92,859 | 180.7 |
| `[1,1,2048,2048]` | fp32 | 191,169 | 175.5 |
| `[1,1,32,16384]` | bf16 | 13,315 | 157.5 |
| `[1,1,32,16384]` | fp32 | 27,466 | 152.7 |
| `[1,1,8192,1024]` | bf16 | 170,508 | 196.8 |
| `[1,1,8192,1024]` | fp32 | 363,140 | 184.8 |
| `[1,1,32,64]` | bf16 | 2,889 | 2.8 |
| `[1,1,32,64]` | fp32 | 3,101 | 5.3 |

Run-to-run spread across the four sessions in this phase was ≤3% on every row —
that is the noise band any later "win" has to clear.

---

## Phase 0 — Verification (verifier pass)

- **Date**: 2026-08-14
- **What was done**: code review against `op_design.md` §1 (Blocking Model) and
  `eval/prompts/tilize.txt` `## Rules`; registry-conformance + INVALID audit; full
  golden run + `eval.verify_supported`; precision baseline authored and measured;
  refinement queue written. Report: `verification_report.md`. Artifacts:
  `verifier_report.json` (this directory — trimmed to fit the repo's 500 KB file limit;
  summary + per-category counts + the xfail blocking-axis histogram + sample cells),
  `op_requirements.md`.
- **SUPPORTED at Phase 0** (unchanged by this pass — no drift to fix):
  `dtype=[bfloat16, float32]`, `output_dtype=[bfloat16, float32, bfloat8_b]`,
  `use_multicore=[False, True]`, `double_buffer=[False, True]`,
  `buffer=[dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram]`, `rank=[2,3,4,5]`,
  `pad_mode=[none, auto, explicit]`, `pad_value=[none, zero, positive, negative]`,
  `alignment=[tile_aligned, w_non_aligned, h_non_aligned, hw_non_aligned]`,
  `tile_height=[32]`, `in_layout=[ROW_MAJOR]`, `in_tile_height=[none]`,
  `shard_api=[none]`, `out_scheme=[interleaved]`, `orientation=[none]`;
  2 EXCLUSIONS (bf8b × padded modes; bf16→fp32 × non-zero fill).
- **Accuracy achieved** (`test_tilize_precision_baseline.py`, 4 shapes × 4 dtype pairs):
  no-cast paths are **bit-exact** — bf16→bf16 and fp32→fp32 give
  `PCC=1.000000, max_abs=0, mean_abs=0, rms=0`, got/true ratio identically 1.0 at every shape
  (this is the strongest possible result for a byte-permutation op and confirms the fp32
  `fp32_dest_acc_en` + `UnpackToDestFp32` configuration). Cast paths carry only representation
  error: `fp32→bf16` PCC=0.999998, max_abs=3.1e-2, mean_abs=2.2e-3, rel_rms=3.3e-3;
  `bf16→bf8b` PCC=0.999971, max_abs=4.7e-2, mean_abs=7.1e-3, rel_rms=9.3e-3. Error is
  shape-independent, as a permutation requires.
- **Golden suite at Phase 0** (per `verifier_report.json`): `supported_pass=102`,
  **`supported_fail=0`, `xpass_drift=0`, `xfail_wrong_mode=0`**, `xfail_expected=246`
  (216 unbuilt axes + 30 EXCLUSIONS cells), `invalid_skipped=568`, 24 retile cells
  arch-skipped (Blackhole-only, correct on this Wormhole box). Whole golden directory:
  344 passed / 155 failed / 2 errors / 884 skipped — every one of the 155 failures is an
  `UnsupportedAxisValue` raised by `validate()` (106 sharding, 29 integer dtype, 8 rank-0/misc),
  i.e. queued axes rather than defects; the 2 errors are a grader-harness conflict
  (`pytestmark use_module_device` + a `device_params`-parametrized trace test) that never
  reaches the op.
- **Issues encountered / fixed in this pass**:
  1. **DRY / collapsed-knob fix** — the CB-footprint formula was restated in three places, twice
     *without* the `NT_BLK` factor (`derive_blocking()`'s L1 ceiling and the never-OOM depth
     fallback), so turning that knob (lamp L3 / the trid-double-issue perf lever) would have
     silently overflowed the budget it was checked against. Now `cb_pages()` / `cb_bytes()` are
     the single source and all consumers read them; no behavioural change at `NT_BLK == 1`
     (verifier categories identical before and after).
  2. Same formula restated a fourth time inside `test_cb_footprint_is_bounded_in_w` — now
     asserts against `cb_bytes()`.
  3. New guard `test_cb_geometry_has_a_single_source` (pins fix 1).
  4. New guard `test_production_switches_ship_in_their_optimal_state` — nothing previously
     pinned that `ABLATE` is all-zero and `LEVERS` all-ON in the shipped config, even though an
     ablation arm produces deliberately wrong output.
  5. No SUPPORTED drift to repair (`xpass_drift = 0`, `supported_fail = 0`).
  - Advisories left as notes (see report): `fp32 → bf16` packs by **truncation**, not
    round-to-nearest (ratio entirely below 1.0, ≈1 bf16 ulp; inside the allowed cast tolerance,
    ruled out as a scale bug via the ratio spread) and no pack-rounding knob is exposed;
    `tt_npe.sh` is absent from this checkout, so the prompt's tt-npe pin could not be produced.
- **Tests added**: `test_tilize_precision_baseline.py` (17 cells),
  `test_tilize_levers.py::test_production_switches_ship_in_their_optimal_state`,
  `test_tilize_levers.py::test_cb_geometry_has_a_single_source`. Acceptance suite now
  **95 passed** (`scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/tilize/`).
- **Refinement queue**: 6 entries in `op_requirements.md` at the 2:1 generality:perf cadence —
  R1 sharded same-spec + crossover (knob-turn, lamp L1), R2 cross-spec reshard + padded sharded
  (scheme-change, lamp L2), **R3 perf** (A3+B10 bank-adjacent readers + per-reader VCs, the
  recorded ~35% gap to the 92%-of-DRAM-peak recipe; B8 trid double-issue as the `NT_BLK>1`
  knob-turn), R4 integer dtypes + rank 0 + both EXCLUSIONS lifted, R5 tiny tiles + retile,
  **R6 perf completeness audit** (Mode D, run-closing).
