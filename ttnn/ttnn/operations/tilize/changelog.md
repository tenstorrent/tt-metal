# tilize — changelog

Box/arch stamp for every number below: **Blackhole, compute grid 11x10 = 110 cores,
AICLK 1349.98 MHz**, bf16, metric = `DEVICE KERNEL DURATION [ns]` from the in-process
device profiler (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`),
3 launches averaged after 2 cache-warming launches. Harness:
`tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py`
(raw JSON: `generated/tilize_bench/phase0.json`).

---

## [x] A0 — Phase 0: baseline correct + measured + classified

Interleaved DRAM->DRAM, single-core, bf16, rank 4, no padding, 32x32 tiles.

### Correctness gate

| suite | result |
|---|---|
| `eval/golden_tests/tilize/test_golden.py` | **1 passed, 379 xfailed (strict), 580 skipped (INVALID)** |
| `tests/.../test_tilize.py` (acceptance, whole-contract spec) | 8 passed: 4/4 single-core identities + all 4 structural refusals. Every other case is a Phase-0 support refusal (`UnsupportedAxisValue` / `ExcludedCell`) for a capability a later refinement lands. |
| `tests/.../test_tilize_debug.py` | 34 passed (deterministic values, blocking-model invariants, multi-core identity on all regimes, lever off-arm correctness, 3 structural pins) |

Both correctness modes were run (`--dev` and production); no race.

### Bound classification — ABLATION (the claim everything else rests on)

Payload stubbed, all CB reserve/push/wait/pop and every loop trip count kept.

| shape | cores | blocks | WT_BLOCK | base | ablate_compute | ablate_read | ablate_write | ablate_read_compute | ablate_all |
|---|---|---|---|---|---|---|---|---|---|
| `(1,1,2048,2048)` | 110 | 256 | 16 | **44590** | 44324 (0.994x) | 31868 (0.715x) | 25652 (0.575x) | 31573 (0.708x) | 541 (0.012x) |
| `(1,1,32,16384)` | 32 | 32 | 16 | **7321** | 5893 (0.805x) | 4932 (0.674x) | 4042 (0.552x) | 4438 (0.606x) | 472 (0.065x) |
| `(1,1,2048,64)` | 64 | 64 | 2 | **4869** | 4541 (0.933x) | 1798 (0.369x) | 4397 (0.903x) | 1626 (0.334x) | 479 (0.098x) |
| `(1,1,32,64)` | 1 | 1 | 2 | **1893** | 1416 (0.748x) | 940 (0.496x) | 1581 (0.835x) | 678 (0.358x) | 320 (0.169x) |
| `(1,1,32,32)` | 1 | 1 | 1 | **1867** | 1427 (0.764x) | 758 (0.406x) | 1557 (0.834x) | 621 (0.332x) | 323 (0.173x) |

**Classification: DATA-MOVEMENT-BOUND, both halves, compute fully overlap-hidden.**

- **Compute is free on the grid-filling shapes** — stubbing the whole `tilize`/`fast_tilize`
  LLK moves the square by 0.6% and tall_narrow by 6.7%. So the DM ceiling *is* the right
  target here (the prompt's precondition for chasing DM levers is satisfied by measurement,
  not assumption).
- **Read and write are co-binding and overlapping.** Each alone removes 28%/42% of the
  square, and with *both* payloads stubbed only **1.2%** of the wall remains. Per
  `/perf-measure`, two single removals that each look partial plus a tiny all-stubbed floor
  is the signature of **balanced overlapping stages** — NOT overhead. There is no
  sync/dispatch floor worth attacking on the big shapes.
- **The smallest regime is different in kind**: `ablate_all` is 17% of a 1.87 us launch, so
  ~1/6 of a tiny call is fixed setup. That is why every per-core-overhead lever below was
  counterfactualed *there* too (master.md B0).
- tall_narrow is **read-dominated** (`ablate_read` 0.369x, `ablate_write` 0.903x): with
  `Wt=2` the block is 2 tiles, so reads are 128 B and the reader is transaction-rate bound —
  the design's own reason for using the full grid rather than a bandwidth-knee cap.

### Ceiling target vs measured

`/perf-ceiling-dm` audit mode (`noc_estimate`, `--arch BLACKHOLE`, `DRAM_INTERLEAVED`,
read `ONE_FROM_ALL`/`ALL_FROM_ALL`, write `ONE_TO_ALL`/`ALL_TO_ALL`, per-core transfer
counts taken from the kernel; CB depth 2 => reader/writer pipeline => `max`, not `sum`).
DRAM peak assumed **448 GB/s** (BH GDDR6 512 GB/s scaled by the 7 of 8 banks this part
reports, `NUM_DRAM_BANKS=7`); practical best per the DRAM tech report is 92% of that.

| shape | per-core NoC bound (no-contention … full-contention) | DRAM floor | practical target (92%) | measured | achieved |
|---|---|---|---|---|---|
| `(1,1,2048,2048)` | 5.0 … 75.8 us | 37.4 us | **40.7 us** | **44.6 us** | **0.91** |
| `(1,1,32,16384)` | 12.0 … 15.2 us | 4.7 us | **5.1 us** | **7.3 us** | **0.70** |

- **The square is essentially DRAM-saturated**: 376 GB/s of DRAM traffic = **84% of
  theoretical peak**, 0.91 of the practical target. The remaining ~9% is exactly the
  bank-adjacency + per-reader-VC recipe, recorded as open levers A3/B10.
- **wide_short has ~1.4x of headroom** and it is not bandwidth: 2 MB total moves in 7.3 us
  with only **32 of 110 cores** active, because at `WT_BLOCK=16` the shape has just 32
  column-blocks. Block width and grid fill trade off here (at 512 B it used 64 cores and was
  *slower*, 1.060x), so the honest reading is that this shape is partly fixed-cost-bound
  (`ablate_all` = 6.5%). Open follow-ups: B8 trid double-issue and the deferred
  per-core transaction stagger.
- The full-contention `ALL_FROM_ALL`/`ALL_TO_ALL` keys are *pessimistic* on this part
  (75.8 us vs 44.6 us measured); the binding bound is the DRAM aggregate, as Step 4b says it
  should be for interleaved round-robin.

### Reconciling against the DESIGN's Mode-A prediction

`op_design.md` §10.3 ranked candidate 1 (TILE-granularity row-block reader, 2-D linear block
split, one barrier per block, whole-tile writes, `row_wise=True`, NoC0/NoC1, `CB_DEPTH=2`)
and predicted **≈88 us** for `[1,1,2048,2048]` at ~191 GB/s.

- **The algorithm was right and is confirmed** — every one of its structural choices measured
  positive here (B7 1.24-6.13x, B9 1.27-1.86x, B5 1.03-1.10x, the 2-D split 6.03x on
  wide-short), and the three rejected candidates would each have lost for exactly the recorded
  reason (candidate 3's barrier-per-stick is measured as B7's off-arm: 1.24-6.13x slower;
  candidate 4's height-only split is `width_split=0`: **6.03x** slower on wide-short;
  candidate 2's unbounded CB is refuted by the L1 bound, and its *knob* survived — see next).
- **The predicted TARGET was mispredicted, in the safe direction, because it was a Wormhole
  number.** Measured 44.6 us at 376 GB/s vs a predicted 88 us at 191 GB/s. The design's
  anchors (`dram_peak` 288 GB/s, 191 GB/s achievable) are WH; this box is BH with ~448 GB/s
  peak. **No algorithm change is implied** — the ranking's winner is still the winner — but
  one *sub-choice* inside it did have to move, below.
- **`TARGET_READ_BYTES` moved 512 -> 1024 on measured evidence.** The design justified 512 B
  as "the one-packet NoC fast path (`NOC_MAX_BURST_SIZE` = 512 B on WH)". On Blackhole
  `NOC_MAX_BURST_SIZE` is 256 words x 64 B = **16 KB**, so that argument does not bind at
  512 B and the sweep decides: square 512 B **55.3 us** vs 1024 B **44.6 us** (1.24x),
  wide_short 1.060x, and no regime regresses (tall_narrow/smallest clamp at `min(Wt, ...)`).
  This is the design's own ordered fallback ("sweep `TARGET_READ_BYTES` toward candidate 2's
  transaction size, bounded by L1") — taken, and bounded: per-core CB L1 is 128 KiB at
  bf16, still a constant in H/W/Wt/rank.

### Used-optimization ledger (Mode C) + completeness (Mode D)

Machine-readable: `ttnn/ttnn/operations/tilize/lever_ledger.json` (24/24 catalog rows).
Render with `python3 -m eval.verify_levers ttnn/ttnn/operations/tilize/lever_ledger.json --report`.
**11 of 24 closed with evidence, 13 open** (0 closed on an argument alone; the 3
`structurally-impossible` rows are each pinned by a named passing test).

Every applied lever, off-arm measured, ratio = off/on (>1 means the lever pays):

| off-arm (knob) | square | wide_short | tall_narrow | smallest 32x64 | smallest 32x32 |
|---|---|---|---|---|---|
| `multicore=0` (A0) | **7.490x** | **6.008x** | **16.870x** | 1.026x | 0.986x |
| `width_split=0` (A0/A1) | 0.981x | **6.025x** | 1.003x | 1.035x | 0.982x |
| `row_wise=0` (A1) | 0.996x | **1.109x** | 1.022x | 1.024x | 0.978x |
| `target_read_bytes=512` (B6) | **1.240x** | 1.060x | 0.996x | 0.999x | 0.974x |
| `target_read_bytes=128` (B6) | 3.393x | 2.809x | 1.004x | 1.001x | 0.977x |
| `barrier_per_block=0` (B7) | **1.242x** | **2.242x** | **2.465x** | **6.132x** | **6.025x** |
| `coalesce_writes=0` (B5) | 1.026x | **1.061x** | **1.031x** | **1.098x** | 1.027x |
| `noc_split=0` (B9) | **1.273x** | **1.864x** | **1.118x** | 1.063x | 1.045x |
| `double_buffer=0` (C16) | 0.995x | 0.993x | 1.010x | 1.018x | 1.002x |

Findings that changed the code or the record:

- **C16 (depth-2 CBs) is refuted as a perf lever** — inside the ~2-3% noise band in *every*
  regime, which the ablation explains mechanistically: compute is already fully hidden, so a
  deeper CB has nothing left to overlap. The code keeps depth-2 as the default because
  `use_double_buffer=True` is a user-facing contract (A6), but A6 should record that depth-1
  halves per-core CB L1 (128 KiB -> 64 KiB at bf16) **for free** on these shapes.
- **`width_split` is a no-op on the square by construction and a 6.03x win on wide-short** —
  the structural no-regression property the design claimed, now measured on both sides.
- **A0's per-regime core count is asserted, not inferred**: 110/32/64/1 cores for
  square/wide_short/tall_narrow/smallest, each equal to `min(grid, total_blocks)`, pinned by
  `test_multicore_fills_the_grid_on_wide_short`.

Two `verify_levers` findings are open and explained rather than papered over: B5 and B7 are
per-core-overhead levers, and master.md B0 requires their counterfactual on the *smallest*
INPUTS shape, which is `[1,1,30,32]` — a `pad_mode="auto"` cell Phase 0 refuses. They are
measured on `[1,1,32,32]` instead (the smallest shape the op can run: 1 core, 1 block, 1
tile, identical per-core geometry), with a P1 follow-up recorded in both ledger rows to
re-measure once the padded reader exists (the pad fill adds per-block L1 stores this shape
does not carry).

### Perf-bench set carried forward (the non-regression baseline)

Every later phase re-measures **all** of these, not only the shape it targets:

| shape | regime | base ns | GB/s |
|---|---|---|---|
| `(1,1,2048,2048)` | grid-filling square, DRAM-bound | 44590 | 376 |
| `(1,1,32,16384)` | wide-short (`nt_h=1`) — does the split fill the grid | 7321 | 287 |
| `(1,1,2048,64)` | tall-narrow — the pure-height-split degenerate | 4869 | 108 |
| `(1,1,32,64)` | smallest golden-sized regime | 1893 | 4.3 |
| `(1,1,32,32)` | smallest runnable (1 core, 1 block, 1 tile) | 1867 | 2.2 |

**Note on what A0 actually ships:** Phase 0's SUPPORTED rectangle accepts only
`use_multicore=False`, so the *shipped* default path for the square is the single-core
number, **334 us**. The multi-core value of the same parameter is wired, correctness-tested
and measured (44.6 us, 7.49x) — refinement A1 flips one SUPPORTED entry and changes no kernel
code. The 44.6 us row is the number A1's gate inherits.

---

## [x] Phase 0 — verifier pass (review + golden/registry gate + precision baseline)

- **Date**: 2026-08-12
- **What was done**: independent verification of the A0 implementation — code review against
  `op_design.md` and `eval/prompts/tilize.txt` §Rules, registry-conformance and INVALID audit, the
  golden suite + `eval.verify_supported` gate, a precision baseline, and the refinement queue.
  Full write-up: `verification_report.md`; queue: `op_requirements.md`.
- **SUPPORTED at Phase 0** (unchanged by this pass — no drift to fix): dtype=[bfloat16],
  output_dtype=[bfloat16], use_multicore=[False], double_buffer=[True], shard_api=["none"],
  out_scheme=["interleaved"], buffer=["dram_to_dram"], rank=[4], pad_mode=["none"],
  pad_value=["none"], alignment=["tile_aligned"], orientation=["none"], tile_height=[32],
  in_layout=[ROW_MAJOR], in_tile_height=["none"].
- **Accuracy achieved**: PCC=1.000000, max_abs_err=0.0, mean_abs_err=0.0, rms_err=0.0,
  got/true ratio median=1.000000 spread=0.0 — **bit-identical** on all 4 shapes
  ((1,1,32,32), (1,1,64,128), (1,1,32,512), (1,1,512,512)) via
  `test_tilize_precision_baseline.py`. Correct for a byte bijection; no precision refinement exists
  to file, and the scale-bug signature (tight ratio cluster at a non-1.0 constant) is ruled out.
- **Golden suite at Phase 0**: **1 / 1 reachable cell passing** — `supported_pass=1`,
  `xfail_expected=379`, `invalid_skipped=580`, and all three loud categories **0**
  (`supported_fail`, `xpass_drift`, `xfail_wrong_mode`), per `verifier_report.json`. The 946
  `no_axes_found` rows are the non-registry files in the same directory (external grader,
  regression, translated, trace); their 216 failures were audited individually and are **all** typed
  support refusals — zero non-refusal failures, zero hangs.
- **Issues encountered / fixed in place**:
  1. The "padding is never implicit" structural check measured a **TILE-layout** (retile-path) input's
     H against the *requested output* tile height, so a legal retile (`H=16, in_tile_height=16 →
     tile 32`) raised a bogus `ValueError` instead of the honest `in_layout` support refusal. Gated on
     `layout != TILE_LAYOUT`. Effect: 5 `test_translated` retile cells moved from hard failure to
     xfail; the golden directory's hard-failure count went **221 → 216** and non-refusal failures to 0.
  2. DRY: the default tile height `32` was restated at 7 sites across the op file and the descriptor.
     Now `DEFAULT_TILE_HEIGHT`, defined once next to `TILE_WIDTH` and imported — Refinement 8 turns
     exactly this knob.
  3. `_cb_budget_bytes` probed a non-existent `ttnn.get_device_info` and so **always** used the
     400 KiB fallback, i.e. the depth-2 "use only if it fits" rule was gated on a magic number. Now
     queries `ttnn.get_max_worker_l1_unreserved_size()` (1 532 160 B here) × a named 0.5 fraction, with
     the constant kept as fallback. No behaviour change today (the op spends a constant 128 KiB/core);
     load-bearing from Refinement 2 onward.
- **Perf re-measured after the fixes** (same bench, DEVICE KERNEL DURATION): square
  `[1,1,2048,2048]` **44 153 ns / 380 GB/s / 110 cores**; wide_short `[1,1,32,16384]`
  **7 227 ns / 290 GB/s / 32 cores** — both within noise of the A0 record, so the three fixes are
  perf-neutral as intended.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`
  (4 shapes + a table-emitting test; asserts bit-identity, PCC, abs/RMS error and the got/true ratio
  spread). No other tests added — the acceptance suite, the debug/blocking-model pins, the golden
  registry suite and the perf bench already cover the Phase-0 surface; further coverage belongs to
  the refinements.
