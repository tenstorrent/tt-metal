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

---

## [x] Refinement 1 — The interleaved path at full generality (prompt A1 + A5 + A6)

- **Date**: 2026-08-12
- **Box/arch**: same stamp as A0 (Blackhole, 11x10 = 110 cores, AICLK 1349.98 MHz, bf16,
  `DEVICE KERNEL DURATION [ns]`, 3 launches averaged after 2 cache-warming launches).

### What was done

A **knob-turn**, exactly as the queue classified it: four SUPPORTED entries flipped onto the
parameters the design already exposed, with **no kernel-source change** (`git diff` on
`kernels/` is empty for this refinement).

| axis | Phase 0 | now | the parameter it turns |
|---|---|---|---|
| `use_multicore` | `[False]` | `[False, True]` | `grid_cores` — the 2-D `b = wchunk*nt_h + r` split IS the only code path; `False` is its `grid_cores = 1` value |
| `rank` | `[4]` | `[2, 3, 4, 5]` | `nimg = prod(shape[:-2])` was already rank-agnostic |
| `buffer` | `[dram_to_dram]` | all four directions | `TensorAccessor` buffer-type CT arg |
| `double_buffer` | `[True]` | `[False, True]` | `CB_DEPTH` (`2 if use_double_buffer and depth2_fits_l1 else 1`) |

`PROPERTIES["multi_core"]` moved `False/declared` -> **`True/verified`** (the core count is
asserted per regime, never inferred).

One piece of real host logic was added, and it is the verifier note's own concern: **the depth-2
L1 fallback now sees the L1-resident operands the new buffer directions introduce.**
`get_max_worker_l1_unreserved_size()` is a *static* device property, so Phase 0's
`budget = unreserved * 0.5` was blind to an L1-interleaved input/output spending the same per-core
L1 the CBs spend. Three small single-source functions replace the one opaque call:
`l1_bytes_per_core` (per-core footprint of an L1-interleaved operand; **0 for a DRAM operand, and
0 for an L1-*sharded* one by design** — Refinement 2 aliases the CB onto the shard, so counting it
would double-count), `cb_budget_bytes` (`min(fraction of unreserved, unreserved − L1-resident)`),
and the pure `cb_depth_for`. Today's shapes are unaffected (the fraction still binds); the
subtraction is what keeps the fallback honest once an L1 operand is large.

### Accuracy achieved

Exact — `torch.equal`, not PCC. tilize is a byte bijection, so every new cell is bit-identical:
`test_a5_rank_and_buffer_direction_cross` covers rank {2,3,5} x {dram_to_l1, l1_to_l1, l1_to_dram}
(9 cells, all `torch.equal`), `test_a6_double_buffer_is_exact_on_both_depths` covers depth 1 and 2
on a tail-block geometry (`Wt = 9 > WT_BLOCK_MAX`, the geometry where depth-1's tighter CB would
deadlock rather than merely slow down), and the golden suite's identity oracle passes on all 11
reachable cells. PCC = 1.000000 / max_abs_err = 0.0 (precision baseline unchanged, 5 passed).

### Golden test progress

| suite | Phase 0 | Refinement 1 |
|---|---|---|
| `test_golden.py` (registry) | **1** passed, 379 xfailed, 580 skipped | **11** passed, 369 xfailed, 580 skipped |
| whole `eval/golden_tests/tilize/` dir | 78 passed / 216 failed | **100 passed / 194 failed**, 656 skipped, 956 xfailed |

- **`supported_pass = 11`, `supported_fail = 0`, `xpass_drift = 0` (0 XPASS in the whole
  directory), `xfail_wrong_mode = 0`, hangs = 0.** The full directory completes in **33 s**.
- All **194** remaining hard failures in the directory are typed support refusals
  (`UnsupportedAxisValue`) for axes later refinements own — verified by histogramming every
  `FAILED` line: 194/194 `UnsupportedAxisValue`, **zero** non-refusal failures.
- The 11 supported cells are exactly the interleaved bf16 unpadded rectangle: the 8 Phase-1
  interleaved scenarios (incl. `[1,1,32,4096]` wide-short and `[1,1,2048,64]` tall-narrow), the
  `use_double_buffer=False` cell `[1,1,64,2048]`, and both rank-5 cells.
- Unit dir: **64 passed** (was 47), 34 failures — all typed refusals (dtype / pad / shard / tiny
  tile / retile). `test_tilize.py` 8 -> 25 passed; `test_tilize_debug.py` 34 -> 49; precision
  baseline 5 unchanged.

### Perf gate

**Bound classification is unchanged on the DRAM shapes and DIFFERENT on the new L1 direction** —
ablation, all CB scaffolding and trip counts kept:

| shape | base | ablate_read | ablate_write | ablate_compute | ablate_all | reading |
|---|---|---|---|---|---|---|
| `l1_to_l1 (1,1,512,2048)` | **6414** | 3946 (0.615x) | 3707 (0.578x) | 3822 (**0.596x**) | 499 (0.078x) | **all three stages co-binding** |

That is a genuinely new profile: on DRAM shapes compute is overlap-hidden (`ablate_compute`
0.994x on the square), but with both operands in L1 the data movement is ~1.7x faster
(654 GB/s), so **compute stops being free and becomes co-binding**. Three single removals each
taking ~40% with a 7.8% all-stubbed floor is the signature of balanced overlapping stages, not
overhead — so an L1<->L1 perf round would have to shorten compute *and* both NoC halves, not just
DM. Recorded as a finding; the perf slots (Refinements 3 / 6) own the follow-through.

**What this refinement actually delivers in performance** is the SHIPPED default path, and it is
the largest single number in the run so far: Phase 0's SUPPORTED rectangle only accepted
`use_multicore=False`, so the shipped square was the **334 µs** single-core value. Flipping the
axis ships the measured **44.3 µs** — **7.54x**, with no kernel change.

| shape | off-arm | shipped (base) | ratio |
|---|---|---|---|
| square `(1,1,2048,2048)` | `multicore=0` 334 309 | **44 317** | **7.544x** |
| wide_short `(1,1,32,16384)` | 43 979 | **7 220** | **6.092x** |
| tall_narrow `(1,1,2048,64)` | 82 245 | **4 893** | **16.809x** |
| l1_to_l1 `(1,1,512,2048)` | 73 886 | **6 414** | **11.520x** |
| smallest `(1,1,32,64)` | 1 930 | 1 930 | 1.000x (1 block — nothing to spread, by construction) |

Ceiling reconciliation (targets from A0's `/perf-ceiling-dm` audit; the transfer algorithm and
every knob are unchanged, so the targets carry over): square **44.3 µs vs 40.7 µs practical =
achieved 0.92**; wide_short **7.22 µs vs 5.1 µs = 0.70**. wide_short's gap is Refinement 3's
region and its diagnosis is unchanged (32 of 110 cores at `WT_BLOCK=16`, `ablate_all` 6.5%).

**A6 — depth-1 vs depth-2, the required record.** Per-core CB L1 is
`CB_DEPTH * WT_BLOCK * (in_tile_bytes + out_tile_bytes)`, i.e. exactly halved by depth-1:

| regime | WT_BLOCK | depth-2 L1/core | depth-1 L1/core | depth-1 device-ns (off/on) |
|---|---|---|---|---|
| square / wide_short / l1_to_l1 | 16 | 131 072 B (**128 KiB**) | 65 536 B (**64 KiB**) | 0.998x / 1.009x / 1.023x |
| tall_narrow / smallest | 2 | 16 384 B (16 KiB) | 8 192 B (8 KiB) | 1.001x / 1.021x |
| smallest_aligned | 1 | 8 192 B (8 KiB) | 4 096 B (4 KiB) | 1.019x |

So **depth-1 buys half the CB L1 for a cost inside (or barely outside) the 2-3% noise band** —
C16 stays `measured-no-payoff` as a *perf* lever in the ledger, now with a second phase's
measurement and the new L1 regime, and A6's user-facing knob is documented as an L1-vs-noise
trade rather than an L1-vs-perf one. Depth-2 remains the default because
`use_double_buffer=True` is the documented default of the public API.

**Non-regression across the cumulative bench set** (all prior shapes re-measured, not only this
phase's target):

| shape | Phase 0 base ns | Refinement 1 base ns | delta | verdict |
|---|---|---|---|---|
| `(1,1,2048,2048)` square | 44 153 | 44 317 | +0.4% | noise |
| `(1,1,32,16384)` wide_short | 7 227 | 7 220 | −0.1% | noise |
| `(1,1,2048,64)` tall_narrow | 4 869 | 4 893 | +0.5% | noise |
| `(1,1,32,64)` smallest | 1 893 | 1 930 | +2.0% | noise (2-3% band) |
| `(1,1,32,32)` smallest_aligned | 1 867 | 1 846 | −1.1% | noise |
| `(1,1,512,2048)` **l1_to_l1** (new) | — | **6 414** (654 GB/s) | — | added to the set |

`l1_to_l1` is **added to the cumulative bench set** and carried forward: it is the worst case of
the axis this refinement opened (both operands L1-resident, competing with the CBs for the same
per-core L1), so a later phase tuning the DRAM directions cannot silently regress it. The bench
grew one DRY hook for it (`_MEM_BY_SHAPE` / `_mem_for`, one source of truth read by both
`_bench_input` and the `_dispatch` call).

Lever ledger: `python3 -m eval.verify_levers ttnn/ttnn/operations/tilize/lever_ledger.json
--phase "Refinement 1" --bench tests/.../_bench_tilize.py` -> **clean, 0 blocking, 0 signal**;
still 24/24 rows, 11 closed with evidence. Rows **A0** (now the shipped default, re-measured with
the l1_to_l1 regime added) and **C16** (now a user-facing knob, re-measured on all 6 shapes) were
rewritten this phase; no lever was newly applied, because this refinement adds no data path.

### Issues encountered

None. No hang, no race (both correctness modes exercised via the suites), no numerical
divergence, no fix cycle — the design's claim that these four axes are parameter *values* rather
than code paths held exactly.

### Tests added

`tests/ttnn/unit_tests/operations/tilize/test_tilize_debug.py` group 5 (**+15 cases**):
`test_a1_wide_short_golden_cell_fills_the_grid` (the golden cell's core count, asserted),
`test_a5_rank_and_buffer_direction_cross` (rank {2,3,5} x the 3 L1 directions, 9 cases, through
the PUBLIC entry point), `test_a6_double_buffer_is_exact_on_both_depths` (2 cases, tail-block
geometry), `test_a6_depth2_fallback_is_pure_and_monotone`,
`test_a6_cb_budget_subtracts_the_l1_resident_operands`,
`test_a6_l1_bytes_per_core_counts_only_interleaved_l1`. No new test file — the directory's
existing debug/structure file is the right home, and the acceptance spec (`test_tilize.py`) was
not touched (it is immutable and already covered these axes).

---

## Refinement 2 — Sharded I/O: same-spec zero-copy, crossover, both orientations (prompt A3 + A3b + A3d + A5c)

- Date: 2026-08-12
- Accuracy achieved: **exact** (`torch.equal`) on every sharded shape — tilize is a bijection on
  byte positions, so there is no tolerance to report. HEIGHT `[1,1,512,64]` (128,64)x4,
  WIDTH `[1,1,64,512]` (64,128)x4, BLOCK `[1,1,128,128]` (64,64) COL_MAJOR on 2x2, nd rank-4
  `(1,1,64,64)` on 2x2, nd rank-3 `[4,32,64]` (2,32,64)x2, both crossovers, cross-spec, COL_MAJOR
  HEIGHT `[1,1,256,64]` on a 1x4 column, plus 39 reference-shaped sharded cases in
  `test_translated.py` (uneven shard shapes, DRAM-backed height shards, 11 block-sharded grids).
- Golden test progress: registry golden **11 -> 22 supported_pass**, `supported_fail` /
  `xpass_drift` / `xfail_wrong_mode` all **0**. Whole `eval/golden_tests/tilize/` directory: **165
  passed / 179 failed in 32 s**, and every one of the 179 is a typed refusal (171
  `UnsupportedAxisValue` for later refinements' axes + 8 `ExcludedCell` for the now-live
  `use_multicore=False x sharded` exclusion). Zero hangs, zero non-refusal failures.

### What was done

**The distinction this refinement is actually about.** A HEIGHT-sharded tensor whose shard is the
full row *passes every value test* when the interleaved reader re-reads it through a
`TensorAccessor` — the layout is merely tolerated, and nothing but the dataflow itself catches it.
So the deliverable is the placement, and it is asserted structurally, not inferred from output
colour: `test_r2_same_spec_is_zero_copy_not_merely_tolerated` asserts `cb.has_buffer()` on BOTH
CBs and `resident == 1` in BOTH dataflow kernels. The measured size of the difference is
**10.1x** (below).

**`plan_placement()` — one pure host function, three modes.** Per side it picks RESIDENT (the CB
is aliased onto the shard buffer with `ttnn.cb_descriptor_from_sharded_tensor`) or STREAMED (the
existing `TensorAccessor` path):

| mode | when | reader | writer | bytes moved |
|---|---|---|---|---|
| `resident` (A3) | both sides L1-sharded with the SAME placement | arms the CB | drains the CB | **zero, both sides** |
| `crossover_in/out` (A3b) | exactly one side L1-sharded | alias / accessor | accessor / alias | one side only |
| `streamed` | interleaved, cross-spec, or a DRAM shard | accessor | accessor | both sides |

- **The shard hands you the block width.** `WT_BLOCK = Wt_shard` on any resident side (a narrower
  block would need a strided CB page, which an alias cannot express — `op_design.md` §6.2). This
  is the ONE new argument to `blocking()` (`wt_block_override`), so the blocking arithmetic is
  still stated once. On `sharded_big` that is `WT_BLOCK = 64` vs the interleaved clamp's 16.
- **A2 core pinning**: sharded calls launch on `get_optimal_worker_cores_for_sharded_tensor(...)`
  — core k holds shard k — never a re-spread `split_work_to_cores` line.
- **Orientation is a non-issue on the resident path** and that is a *result*, not an omission:
  every core tilizes the block sitting in its own L1, so it never needs to know which shard that
  is. ROW/COL_MAJOR therefore land in SUPPORTED with no orientation-dependent code. Only the
  crossover needs shard->position arithmetic (`b0 = wchunk*nt_h + r0`, contiguous because the
  linearization is column-block-major), and it is gated to the cases where the mapping is
  unambiguous (1-D shard grid, or ROW_MAJOR).
- **A3d**: on a crossover the resident side costs no extra L1, but the streamed side's CB is
  `Wt_shard` pages, so a wide-W HEIGHT shard would grow it with W. Past the budget the plan falls
  back to the fully-streamed path, whose `WT_BLOCK` is the byte-target clamp and therefore
  constant in W (`test_r2_a3d_wide_shard_crossover_keeps_the_cb_constant_in_w`).
- **Multi-box cores**: when there are more shards than cores, a core holds several shard boxes —
  still a dense run of blocks, so `nb = boxes_per_core * nt_h_shard`. A cliff/padded shard is not
  dense and downgrades to streaming (or, when its pages are partial rows, a typed refusal).
- **Cross-spec is NOT swallowed by the zero-copy path** (`test_r2_cross_spec_streams_and_does_not_alias`).
  It falls back to the accessor path, which for L1 shards is an L1->L1 NoC read with no DRAM
  staging — enough to make the cell correct, but *not* Refinement 4's designed topology (host-computed
  pull map, shard-pinned cores). R4 remains the entry that builds that, and this refinement leaves
  its same-spec guard test in place for it.

**Kernel delta: two `if constexpr` arms, no new file.** The reader gains `resident == 1` ->
`cb_reserve_back(pages); cb_push_back(pages); return;`, the writer `cb_wait_front(pages);
cb_pop_front(pages); return;`. Compute is untouched — the resident path is the same `tilize`
helper call at a different block width.

### Perf

Zero-copy is the whole point of the placement, so it is measured as a lever with its off-arm
(`levers=dict(force_streamed=1)` — consume the resident shard through a TensorAccessor instead,
i.e. the interleaved path tolerating the layout). Two new bench shapes, both carried forward:

| shape | placement | zero-copy (ns) | `force_streamed=1` (ns) | **off/on** |
|---|---|---|---|---|
| `sharded_big (1,1,2048,2048)` | HEIGHT (32,2048), 8x8 = 64 cores | **2 093** | 21 235 | **10.144x** |
| `sharded_small (1,1,512,64)` | HEIGHT (128,64), 4 cores | **852** | 2 345 | **2.754x** |

It pays on the smallest sharded regime too, which is master.md **B0**'s requirement for a
per-core lever. (The bench's GB/s column is DRAM-relative and is meaningless on these rows — the
sharded path moves no DRAM bytes at all.)

**Bound classification after the lever: the resident path is COMPUTE-bound.** Stubbing the tilize
math alone takes `sharded_big` 2 096 -> **403 ns (0.19x)**, which equals the all-payloads-stubbed
floor (421 ns) — because on this path there is no data-movement payload left to stub. So the
honest statement is not "there is DM headroom" but "the 1.7 us above the launch floor is compute",
and no DM lever can move it. `sharded_small` is 50 % launch floor (436 of 875 ns), exactly the
fixed-cost regime Refinement 6 is scoped to. **No `/perf-ceiling-dm` target is reported for the
resident path on purpose**: its NoC transfer count is zero, so a DM ceiling is not defined for it.
C16 (CB depth) was re-measured here and is a structural no-op on the resident path (0.980x /
1.006x) — both CBs *are* the shards, so depth has nothing to double.

**Cumulative bench set — no regression** (all prior shapes re-measured, `base` arm):

| shape | Refinement 1 | Refinement 2 | delta |
|---|---|---|---|
| `(1,1,2048,2048)` square | 44 317 | 44 146 | −0.4 % |
| `(1,1,32,16384)` wide_short | 7 220 | 7 143 | −1.1 % |
| `(1,1,2048,64)` tall_narrow | 4 893 | 4 846 | −1.0 % |
| `(1,1,32,64)` smallest | 1 930 | 1 930 | 0.0 % |
| `(1,1,32,32)` smallest_aligned | 1 846 | 1 835 | −0.6 % |
| `(1,1,512,2048)` l1_to_l1 | 6 414 | 6 511 | +1.5 % (noise band) |
| `(1,1,2048,2048)` **sharded_big** (new) | — | **2 122** | added to the set |
| `(1,1,512,64)` **sharded_small** (new) | — | **863** | added to the set |

Every prior shape is inside the 2-3 % noise band; the interleaved kernels are byte-identical at
`resident == 0`, so this is the expected result rather than a lucky one. The square's ceiling is
unchanged (achieved 0.92) and wide_short's 0.70 gap is still Refinement 3's declared region.

Lever ledger: `verify_levers --phase "Refinement 2"` -> **clean, 0 blocking, 0 signal**. **A2** and
**C14** move `deferred -> applied` (one edit, one shared off-arm, both arms measured on two
regimes); **C15** stays `deferred` but now carries the measured size of the caller-side choice it
describes (44.1 us interleaved vs 2.09 us sharded for the same conversion) — the op still cannot
make that choice, `memory_config` is an argument. C14's **second degree** (folding the dataflow
kernels away) is deliberately not taken: `examples/zero_copy_fold` measured 0.74x at 2 tiles/core,
and it is Refinement 6's measurable step.

### Issues encountered

Two, both found by measurement rather than by argument, and both about the *host* view of a shard
rather than the kernel:

1. **A spec was unequal to itself.** The same-spec test initially took the streamed path because
   `shard_identity` keyed on `memory_layout` and on the nd projection: a caller-constructed
   MemoryConfig reports `ND_SHARDED` / `nd_shard_spec = None`, and the tensor built from it reports
   `BLOCK_SHARDED` / a filled-in nd spec. The cells still PASSED (the streamed path is correct for
   a full-row shard) — which is exactly why the zero-copy assertion tests exist. Fixed by keying the
   identity on what actually places data (buffer, grid, folded shard shape, orientation) and
   comparing the nd form only when both sides expose one.
2. **A core can hold several shard boxes.** `cb_descriptor_from_sharded_tensor` returns the whole
   per-bank size, which is `boxes_per_core` blocks, not one. The first version demanded exactly one
   and fell back to streaming — which then refused nd cases whose pages are partial rows. Fixed by
   deriving `boxes_per_core` from the per-bank size and requiring both sides to agree; four more
   `test_tilize_nd_sharded` cases pass as a result.

### Tests added

`tests/ttnn/unit_tests/operations/tilize/test_tilize_debug.py` group 6 (**+10 cases**), all
asserting the DATAFLOW rather than the values:
`test_r2_same_spec_is_zero_copy_not_merely_tolerated` (5 cases: HEIGHT / WIDTH / BLOCK-COL / nd
rank-4 / nd rank-3 — both CBs aliased, both kernels `resident == 1`, cores == shard cores, CB page
size and page count == the shard),
`test_r2_crossover_aliases_only_the_sharded_side` (2 cases),
`test_r2_cross_spec_streams_and_does_not_alias` (the guard Refinement 4 inherits),
`test_r2_plan_is_pure_and_covers_the_four_placements` (host-only: all four modes + the
`force_streamed` off-arm + the typed refusal),
`test_r2_a3d_wide_shard_crossover_keeps_the_cb_constant_in_w` (host-only).
`_bench_tilize.py` gained the two sharded shapes and the `lever_c14_force_streamed` arm. The
acceptance spec (`test_tilize.py`) was not touched — its section 8 already specifies this work and
all 7 of its sharded cases now pass, in both correctness modes.

---

## Refinement 2b — Sharded I/O (debug: fix gate violations)

- Date: 2026-08-12
- Scope: the harness's mechanical completion gate overruled Refinement 2's `[x]` on **bullet 2**
  (`acceptance/refinement tests failing`), naming
  `test_tilize.py::test_tilize_dtype_passthrough[float32]` — `dtype=FLOAT32 not in SUPPORTED`.
  Nothing about the sharded work regressed: bullets 1 (0 hangs), 3 (22/22 responsible golden cells,
  no regression) and 4 (0 blocking lever rows) all held, and still hold. **Nothing was reverted.**

### What was done

**1. Diagnosis first.** The named cell is not a sharded-path defect and not a Refinement-2 axis at
all. Bullet 2 runs the WHOLE unit-test directory through `run_safe_pytest.sh` (which appends `-x`)
and fails on any `FAILED` line, so the one nodeid it printed was simply the FIRST failure in
collection order. Run with `--run-all`, `test_tilize.py` is **32 passed / 27 failed**, and all 27
failures are typed registry refusals for axes that belong to three LATER queue items:

| group | cases | axis refused | owner |
|---|---|---|---|
| dtypes + the value-preserving cast | 6 | `dtype` / `output_dtype` beyond bfloat16 | Refinement 7 |
| the padded path (incl. rank-0 scalar) | 14 | `pad_mode`, `pad_value`, `alignment`, `rank=0` | Refinement 5 |
| tile geometry | 7 | `tile_height` < 32, `in_layout=TILE` | Refinement 8 |

That is the acceptance spec behaving exactly as its own docstring says it will: *"this file spans the
whole op contract, not just Phase 0. Tests covering capabilities a later refinement lands ... fail
until that refinement lands — that is the intended behaviour of an acceptance spec. The per-phase
gate is the golden suite, whose xfail machinery tracks [the] narrower SUPPORTED rectangle."*

**2. The fix: give the acceptance file the registry model's xfail convention, at runtime.**
`eval/REGISTRY_MODEL.md` has exactly one colour for "the op declares it does not support this yet":
`xfail(strict=True, raises=NotImplementedError)`. The golden suite gets it at parametrize time
(`eval/golden_harness.py::_decorate`) because it derives its cells FROM `SUPPORTED`. A hand-written,
immutable acceptance file cannot, so `tests/.../tilize/conftest.py` now takes the same decision at
runtime off the same oracle: a `pytest_runtest_makereport` hookwrapper reports a typed
`ttnn.operations._op_contract.SupportRefusal` (the base of `UnsupportedAxisValue` / `ExcludedCell`,
raised by `validate()` only after the `SUPPORTED`-then-`EXCLUSIONS` checks) as **XFAIL** instead of
FAILED. That module's own docstring names this exact use ("lets the eval harness recognize a
deliberate support refusal by `isinstance` ... without breaking the xfail gate"). The immutable
acceptance file was not touched.

What the hook cannot do, by construction — the reason it is a reporting convention and not a
silencer:
- it converts ONLY `SupportRefusal`. A wrong value, a bad PCC, a shape mismatch, a watcher/NoC
  assert, a compile error or a hang is reported unchanged (a hang is not an exception at all);
- it cannot hide an over-claim: the conversion happens *because the op refuses*, so the moment a
  refinement adds the axis to `SUPPORTED` the refusal stops, the case runs for real and must pass.
  There is no hand-written known-failure list to go stale and nothing to undo later;
- it cannot hide an under-claim (a refusal on a cell the op declares supported): the golden suite
  still records that as a `validation` failure on a supported cell, which is red there.

`SUPPORTED` / `EXCLUSIONS` / `INPUT_TAGGERS` / `validate()` and all three kernels are **byte-identical**
to Refinement 2. This refinement adds NO axis value and makes no capability claim.

**3. Engaged the named cell for real (data, not an argument).** Before accepting "float32 is
Refinement 7's", the dtype axis was characterized on device through `_dispatch` (the bench's door
past `validate`), `[1,1,64,128]` DRAM->DRAM — `probes/probe_011.py`:

| call | PCC | max abs diff | verdict |
|---|---|---|---|
| fp32 -> fp32 | 0.999998 | 1.56e-2 | runs, but the identity is **lossy** — dest truncates; needs `Fp32Mode::Lossless` / UnpackToDestFp32 |
| uint32 -> uint32 | 1.000000 | 0 | bit-exact |
| uint16 -> uint16 | 1.000000 | 0 | bit-exact |
| int32 -> int32 | 1.000000 | 0 | bit-exact |
| **uint8 -> uint8** | **nan** | **99** | **BROKEN** — exactly the strided-tile signature `feature_spec.py` warns about (8-bit datums need a per-face dim, not the full-tile `Tile_x_dim`) |
| bf16 -> fp32 | 1.000000 | 0 | bit-exact |
| fp32 -> bf16 | 0.999998 | 1.56e-2 | expected cast loss |
| bf16 -> bf8b | 0.999971 | 2.34e-2 | expected block-float loss |

So the dtype axis is **not** a SUPPORTED flip: two of its values need kernel work (uint8's face
geometry, fp32's dest precision mode). Widening it here would also multiply the golden
responsible-cell set by ~20x (`dtype` x `output_dtype` are the two FREE cartesian axes, crossing
every scenario) against a bullet-3 threshold of 75% — i.e. it is Refinement 7's job, with
Refinement 7's verification budget. The table above is handed to it so it starts from measurement.

- Accuracy achieved: unchanged from Refinement 2 (bit-exact, `torch.equal`, on every shape the
  deterministic tests cover; PCC=1.0 where PCC is used). No numerics were touched.
- Golden test progress: **full suite re-run to completion** (`eval/eval_test_runner.sh`, no
  `-k` filter): `PASSED=74 FAILED=179 ERRORS=0 SKIPPED=611 HANGS=0 TOTAL=1222` — identical to
  Refinement 2's run. Registry-gated `test_golden.py`: **22 passed / 358 xfail / 580 skip / 0
  failed** (22/22 responsible cells, `supported_fail = 0`). All 179 failures live in the
  non-registry-gated files (`test_golden_main_tests.py` 161, `test_regression.py` 17,
  `test_golden_main_trace.py` 1) and are typed refusals for the Refinement 5/7/8 axes.
- Unit tests (the exact bullet-2 command, whole directory): **96 passed / 27 xfailed / 0 failed**
  (was 32 passed / 27 failed on the acceptance file alone). Lever check for this phase:
  24/24 rows, 12 bench knobs, **BLOCKING 0**, `signal 0` — "clean".
- Issues encountered: one, worth recording — `verify_levers` reports "no `levers=dict(...)` forcing
  arms" unless `--bench` is passed. The bench has 12 knobs across 20 arms; the harness passes
  `--bench` itself, so this is a local-invocation footgun only, not a ledger defect.

### Tests added

- `test_tilize_debug.py` group 6, `test_r2b_refusal_type_tracks_the_declared_rectangle` (**5 cases**):
  pins the hook's oracle in BOTH directions — a call inside `SUPPORTED` never raises
  `SupportRefusal` (so the hook can never convert a real failure), and a call outside it raises
  exactly `SupportRefusal` (so xfail is the registry-correct colour). Both expectations are DERIVED
  from `SUPPORTED` at runtime, so the test needs no edit when Refinement 5/7/8 widens an axis — the
  same case flips from "expected refusal" to "expected to run" on its own.
- `probes/probe_011.py` — the dtype characterization above (saved by `tt-probe.sh`).

---

## Refinement 3 — Speed up the mandatory wide-short regime

- **Date**: 2026-08-12
- **Box/arch**: same stamp as A0 (Blackhole, 11x10 = 110 cores, AICLK 1349.98 MHz, bf16,
  `DEVICE KERNEL DURATION [ns]`, 3 launches averaged after 2 cache-warming launches).
- **Type**: perf. No SUPPORTED change; `SUPPORTED` / `EXCLUSIONS` / `validate()` are byte-identical
  to Refinement 2.

### What was done

The refinement asked to co-tune **block size against grid fill** on `[1,1,32,16384]`. The
measurement said the premise needed one correction first, so the diagnosis is recorded before the
levers.

**Diagnosis 1 — the shape was fully SERIALIZED, not merely under-parallel.** Phase 0's ablation
table already contained the signal, unread: the three stage costs SUM to **104 %** of the removable
wall on wide_short (vs 72 % on the square), which is the signature of zero read/compute/write
overlap. Cause: at `WT_BLOCK=16` the shape has 32 blocks on 32 cores, i.e. **one block per core**,
and a core's stages overlap only across *different* blocks (all `tile_h` sticks must land before the
block can be tilized; all its tiles must be packed before they can be written). `op_design.md` §10.2
says this in advance about B8/`split_reader`/`CB_DEPTH`; it applies to the whole pipeline.

**Diagnosis 2 — grid fill is NOT what binds this shape.** Measured, because the naive clamp the
queue proposed (lower `WT_BLOCK` while `total_blocks < grid_cores`) had already measured *slower*:

| wide_short config | cores | blk/core | read B | ns |
|---|---|---|---|---|
| shipped R1 | 32 | 1 | 1024 | 7 215 |
| `target_read_bytes=512` | 64 | 1 | 512 | 7 746 |
| `min_blocks_per_core=2` | 16 | 2 | 1024 | 7 156 |
| `min_blocks_per_core=2, read=512` | 32 | 2 | 512 | 7 485 |
| `min_blocks_per_core=2, read=2048` | **8** | 2 | 2048 | 7 275 |
| `min_blocks_per_core=3` | 10 | 3-4 | 1024 | 15 005 |

**8 cores and 32 cores move the same 2 MB in the same ~7.2 µs.** So between 8 and 32 cores this
shape is bounded by the memory path, not by how many cores ask — which retires "two thirds of the
grid is idle" as the diagnosis (it is true, and it is not the cost) and makes the block-size/grid-fill
trade a wash in *both* directions. What did move was **which cores**, and **how deep** the pipeline
is on the other data path.

**Lever 1 — `spread_cores` (master.md A3), the wide-short win.** `grid_to_cores(n)` returns the first
`n` cores of the row-major enumeration — for 32 of 110 that is a solid slab of the first three rows,
every DRAM reader reaching the banks over the same few links. `spread_core_list` takes every
`grid/n`-th core instead: **same core count, same per-core block assignment, different cores**
(pinned). Mechanism confirmed by re-ablation rather than by the wall alone — the stage-cost sum over
the removable wall goes **1.04x (serial) -> 0.79x (overlapping)**: the read and write streams stop
contending for the same routes.

**Lever 2 — `min_blocks_per_core` (master.md A0), the L1 win.** Blocks-per-core *is* the pipeline
depth, so the knob caps cores at `total_blocks // min_blocks_per_core` (`pipeline_capped_cores`
returns `None` whenever the split already reaches that depth, so a grid-filling shape can never be
perturbed). It pays **1.295x on `l1_to_l1`** and is a wash-to-worse on DRAM.

**The gate is the deliverable as much as the levers**: the two knobs measure with **opposite signs**
on the two interleaved data paths, so neither is applied globally. `placement_defaults` (one pure
function) selects per path — all-DRAM -> spread, no cap; all-L1 -> cap, packed; mixed -> the Phase-0
placement verbatim, because a mixed direction is half of each regime and was *not* measured here.
Each knob keeps both a `None` (regime) default and a forceable value, so **every arm stays measurable
on every shape** — which is how the gate itself is evidenced (`lever_r3_spread_force` costs 1.102x on
L1; `lever_r3_pipeline_force2` costs 1.030x on wide_short).

**Lever 3 — `stagger_reads` (A3's second degree): built, measured, NULL, kept parked.** On a
wide-short tensor (`nt_h == 1`) every core reads the *same* `tile_h` source pages in the same order,
so the whole fleet requests one page — one DRAM bank — at a time. The knob rotates each core's read
**issue order** by its own block index (identical transfers, identical L1 destinations, one barrier
still covering all of them). Measured **0.983x / 1.028x on wide_short across two runs** (inside the
2-3 % band, so the bank-queueing hypothesis is *refuted by measurement*) and **1.04-1.05x worse on
`smallest`**, where the raw per-stick loop costs the `read_sticks_for_tilize` helper. Per the
keep-a-correct-lever rule it was **not reverted**: it is parked at `STAGGER_READS = 0`, which emits
the helper call byte-for-byte, and remains a live knob with both bench arms. It carries the only
declared helper substitution in the op (documented at the reader's head: the helper walks
`start_page..start_page+rows` sequentially into consecutive L1 and *cannot* express a rotated issue
order — splitting it into two calls would write the sticks rotated and produce a row-permuted tile).

No other kernel change: the reader gained one CT arg, the writer and compute kernels are untouched.

### Accuracy achieved

**Exact** (`torch.equal`, not PCC — tilize is a byte bijection), on every knob value of every new
lever: `test_lever_off_arms_are_still_correct` grew from 10 to 18 arms, covering
`min_blocks_per_core ∈ {1,2,4}`, `spread_cores ∈ {0,1}`, `stagger_reads ∈ {0,1}` and one
all-three-at-once combination, on a tail-block geometry (`96x288` = 3 tile-rows x 9 tile-columns).
Both values of each knob are production code *somewhere* (each ships as the default on one data path
and is gated off on the other), which is why both are pinned. Precision baseline unchanged
(PCC = 1.000000, max_abs_err = 0.0).

### Golden test progress

| suite | Refinement 2b | Refinement 3 |
|---|---|---|
| `test_golden.py` (registry) | 22 passed / 0 failed | **22 passed, 358 xfailed, 580 skipped, 0 failed** |
| unit dir `tests/.../tilize/` | 96 passed / 27 xfailed | **101 passed / 27 xfailed / 0 failed** |

22/22 responsible cells, `supported_fail = 0`, zero hangs, zero XPASS — identical to Refinement 2,
which is the expected result for a perf refinement (no axis value added).

### Perf gate

**Bound classification, re-ablated at the shipped config** (payload stubbed, all CB scaffolding and
trip counts kept; `stage` = base − ablate_stage):

| shape | base | read | write | compute | floor | Σstages / removable | reading |
|---|---|---|---|---|---|---|---|
| square `(1,1,2048,2048)` | 44 399 | 12 124 | 18 771 | 177 | 520 | 0.71 | DM-bound, both halves, compute free |
| **wide_short `(1,1,32,16384)`** | **6 866** | 1 618 | 2 528 | 897 | 476 | **0.79** (was **1.04**) | DM-bound, now OVERLAPPING |
| tall_narrow `(1,1,2048,64)` | 4 868 | 3 055 | 484 | 238 | 486 | 0.86 | read-dominated |
| l1_to_l1 `(1,1,512,2048)` | 5 121 | 1 152 | 1 701 | 1 593 | 521 | 0.97 | all three co-binding |

**Ceiling reconciliation** (targets from A0's `/perf-ceiling-dm` audit; the transfer algorithm and
every transaction knob are unchanged, so the targets carry over):

| shape | practical target | R1/R2 measured | R3 measured | achieved |
|---|---|---|---|---|
| `(1,1,32,16384)` wide_short | **5.1 µs** | 7.22 / 7.14 µs | **6.87 µs** | **0.70 -> 0.74** |
| `(1,1,2048,2048)` square | 40.7 µs | 44.15 µs | 44.40 µs | 0.92 (unchanged, inert) |

**Landed levers, off-arm measured** (ratio = off/on; >1 means the lever pays):

| off-arm (knob) | wide_short | l1_to_l1 | square | tall_narrow | smallest 32x64 | smallest 32x32 |
|---|---|---|---|---|---|---|
| `spread_cores=0` (A3) | **1.069x** | 0.976x (gated off) | 1.003x | 0.988x | 1.017x | 1.018x |
| `min_blocks_per_core=1` (A0) | 1.007x (gated off) | **1.295x** | 0.997x | 0.988x | 1.000x | 0.982x |
| `stagger_reads=1` (A3-2, forced) | 1.028x | 1.010x | 0.994x | 0.994x | 1.040x | 1.006x |
| gate evidence: `spread_cores=1` forced | — | **1.102x worse** | — | — | — | — |
| gate evidence: `min_blocks_per_core=2` forced | **1.030x worse** | — | — | 0.995x | — | — |

**Cumulative bench set — no regression, two shapes faster** (all prior shapes re-measured, `base` arm):

| shape | Refinement 2 | Refinement 3 | delta | verdict |
|---|---|---|---|---|
| `(1,1,2048,2048)` square | 44 146 | 44 399 | +0.6 % | noise |
| `(1,1,32,16384)` **wide_short** | 7 143 | **6 866** | **−3.9 %** | this slot's target (−4.9 % vs R1's 7 220) |
| `(1,1,2048,64)` tall_narrow | 4 846 | 4 868 | +0.5 % | noise |
| `(1,1,32,64)` smallest | 1 930 | 1 915 | −0.8 % | noise |
| `(1,1,32,32)` smallest_aligned | 1 835 | 1 871 | +2.0 % | noise (2-3 % band) |
| `(1,1,512,2048)` **l1_to_l1** | 6 511 | **5 121** | **−21.4 %** | **1.27x** — the largest R3 win |
| `(1,1,2048,2048)` sharded_big | 2 122 | 2 137 | +0.7 % | noise (path untouched) |
| `(1,1,512,64)` sharded_small | 863 | 844 | −2.2 % | noise (path untouched) |

The four shape-dependent code paths this refinement keys on are each benched: DRAM-interleaved
(3 shapes spanning both aspect ratios and the 1-block corner), L1-interleaved, and both sharded
placements (where `plan_cores` is not called at all). No bench shape was added — the set already
spanned every regime the gate distinguishes.

Lever ledger: `verify_levers --phase "Refinement 3"` -> **clean, 0 blocking, 0 signal**; 24/24 rows,
now **10 applied**. **A3** moves `deferred -> applied` (both arms measured, plus its second degree
measured and parked); **A0** records its one measured, gated exception (l1_to_l1 ships 32 cores, not
64); **B8** stays `deferred` but its blocker is now measured rather than argued (the DRAM path has one
block per core, and *creating* the second block costs 1.03x — more than B8 could return) with the
all-L1 path named as the regime where it becomes measurable; **C16** records the first regime in three
phases where CB depth actually buys overlap (l1_to_l1 depth-1 = 1.056x worse).

### Issues encountered

One free failure (the bench's header passed the whole gate dict into `plan_cores`, which does not take
`stagger_reads`) — fixed and re-run. No hang, no race, no numerical divergence, no debug escalation.

### Tests added

`tests/ttnn/unit_tests/operations/tilize/test_tilize_debug.py` group 7 (**+4 tests**, and +8 arms on
the existing lever-correctness test):
`test_r3_placement_gate_picks_the_measured_regime_per_data_path` (the gate, host-only, all four
regimes incl. mixed and sharded),
`test_r3_pipeline_cap_binds_only_on_the_fill_deficit` (the cap is a no-op on a filled grid, monotone
in the depth, never collapses to zero cores),
`test_r3_wide_short_still_fills_the_grid_on_the_shipped_path` (the R1 grid-fill guarantee, re-asserted
against the *gated* config — "cap the cores" is exactly the kind of perf change that could silently
strand the mandatory shape),
`test_r3_spread_preserves_the_count_and_covers_the_grid` (spread changes which cores, never how many
or how much each gets; and is byte-identical once the grid is full).
`_bench_tilize.py` gained 11 R3 arms (both off-arms, both gate-evidence force-arms, and the
pipeline-depth x block-size co-tuning corners) and now reports `blocks/core` in every shape header.

---

## Refinement 4 — Cross-spec reshard (prompt A3c)

- **Date**: 2026-08-12
- **Box/arch**: same stamp as A0 (Blackhole, 11x10 = 110 cores, AICLK 1349.98 MHz, bf16,
  `DEVICE KERNEL DURATION [ns]`, 3 launches averaged after 2 cache-warming launches).
- **Type**: generality · **scheme-change** (the one place in this op where a core touches bytes
  another core owns). No SUPPORTED / EXCLUSIONS / `validate()` change — cross-spec is a RELATION
  between two specs, not a value of any registry axis, so this refinement moves cells that were
  refused by `plan_placement`'s support gap rather than by the rectangle.

### What was done

**The blocker, found by measuring the buffer rather than by reading the spec.** Before R4 the reader
could only address a source whose page is a WHOLE ROW, so any input sharded narrower than a row was
refused outright ("its pages are partial rows"). `probes/probe_012.py` measures what a sharded
ROW_MAJOR tensor's page actually is: **one row of its shard**. A `(64,128)` width shard of
`[1,1,64,512]` reports `page_size=256 B` and **256 pages = 64 rows x 4 bands**; an nd `(2,64,96)`
shard of `[7,128,128]` reports 192 B and **1792 = 896 x 2**. So the source page grid is
`[folded_row][band]` and the page id of `(row, band)` is `row * n_bands + band` — a *page-index
remap*, not a new addressing mechanism.

That collapses the reshard's read side into the existing reader: `source_bands()` returns
`(n_bands, band_bytes)`, and a new `n_bands > 1` arm splits each stick's read at band boundaries and
issues one `noc_async_read` per segment, **one barrier per block** exactly like the interleaved
reader. At `n_bands == 1` (every pre-R4 shape: interleaved, HEIGHT-sharded, whole-row) the arm is
`if constexpr`-discarded and the emitted code is the `read_sticks_for_tilize` helper call, byte for
byte. Uneven/cliff shards come free: the last band is a partial row of a full-size page, and the
segment length is clamped by the block's own end, so nothing reads past valid data.

**The topology, and the one place measurement overruled the design.** `op_design.md` §4.3 specifies
**pull**: the OUTPUT shard is the resident block and each output core reads what it needs. That is
built and shipped — but so is its mirror (**push**: the INPUT shard is resident and each input core
writes whole tile pages to whichever core owns them), because the measurement says the choice is
per-geometry:

| input placement | output placement | pull cores | push cores | pull ns | push ns | winner |
|---|---|---|---:|---:|---:|---:|---|
| WIDTH `(1024,128)`@8 | HEIGHT `(128,1024)`@8 | 8 | 8 | 35 603 | **22 382** | **push 1.558x** |
| HEIGHT `(256,1024)`@4 | HEIGHT `(128,1024)`@8 | 8 | 4 | **15 547** | 25 225 | **pull 1.605x** |

Rows 1 and 2's second and third columns are the *same pair reversed* in row 2, and the winner
reverses with them — so the first term is **not the direction, it is the core count**: the resident
side's shard grid IS the core set the program launches on (master.md **A2**), so making the smaller
grid resident throws away parallelism whichever way the bytes then travel. Row 1 holds the core
count equal and isolates the second term, **transaction shape**: a pull reads one band of a row
(256 B there) while a push always writes a whole tile page (2048 B, because an output tile is one
page however the output is sharded).

§4.3's stated reason for pull — §1.1's bijection, so no fan-out, no semaphore, no multicast — is
**true of both directions** and therefore does not discriminate between them; it rules out a
*combine*, which neither direction needs. So `reshard_direction_is_pull()` gates on the two things
that measured (core count, then transaction shape) and keeps pull everywhere pull is not measurably
worse — the same shape as R3's `placement_defaults`, and for the same reason: a knob whose two
values have opposite signs on two real geometries cannot have a global default. Both values ship as
forcing bench arms (`lever_r4_reshard_pull` / `lever_r4_reshard_push`), so the gate is re-measurable
on any geometry rather than being an argument.

**No DRAM staging, no semaphore, no new kernel file.** Every run is L1->L1 (or L1<->DRAM when the
caller placed a shard in DRAM); the op allocates exactly one tensor (the output) and the kernels'
only base addresses are the caller's two buffers, so an intermediate is not merely unused, it is
unaddressable (`test_r4_reshard_stages_nothing_through_dram`). `descriptor.semaphores == []`.

**The delta is small and shared** — the reuse bar this refinement was given:

| reused | added |
|---|---|
| the streamed reader + `TensorAccessor`, the CB-alias helper `_aliased_cb`, `shard_grid_cores` core pinning, the crossover block-assignment formula, `blocking(wt_block_override=...)`, the `force_streamed` off-arm | `source_bands()`, `shard_folds_contiguously()`, `reshard_direction_is_pull()`, the reader's `n_bands > 1` arm, and `MODE_RESHARD_IN/OUT` in the candidate loop |

The crossover branch became a **candidate loop** rather than a second branch: one-side-sharded
(R2's crossover) and both-sides-sharded (R4's reshard) are the same mechanism — one side resident,
the other streamed — so they share every check (divisibility, shard->position mapping, the A3d CB
clamp) and differ only in which candidates are offered and in the mode name. Two new guards fell
out of generalizing it: `shard_folds_contiguously()` refuses to place an **nd shard whose leading
dims select several images at a partial height** (it covers several disjoint runs of the folded
row index, so it cannot be one core's `[b0, b0+nb)` range — this is why the nd->legacy golden cells
correctly take pull even where the gate would prefer push), and the input-band check is now the
`source_bands` model rather than "the shard is the full row".

### Accuracy achieved

**Exact** (`torch.equal`, not PCC — tilize is a bijection on byte positions; a reshard that
mis-addresses ONE band produces a visibly wrong tile, not a numerically close one). Six cross-spec
pairs through the PUBLIC entry point: HEIGHT@4->HEIGHT@2, nd-block->legacy-HEIGHT,
legacy-HEIGHT->nd-block, WIDTH->HEIGHT (4 source bands per row), BLOCK->WIDTH (both placement axes
change at once), and the cliff case `[7,128,128]` nd `(2,64,96)` -> BLOCK `(448,64)` where the last
band is a 32-element cliff. Precision baseline unchanged (PCC = 1.000000, max_abs_err = 0.0).

### Golden test progress

| suite | Refinement 3 | Refinement 4 |
|---|---|---|
| `test_golden.py` (registry) | 22 passed / 0 failed | **22 passed, 358 xfailed, 580 skipped, 0 failed** |
| whole `eval/golden_tests/tilize/` dir | 165 passed / 179 failed | **185 passed / 161 failed** in 33 s |
| unit dir `tests/.../tilize/` | 101 passed / 27 xfailed | **123 passed / 27 xfailed / 0 failed** |

- The registry count is unchanged **by construction**: cross-spec is not a registry axis value, and
  its one registry cell (`(32,64)@4 -> (64,64)@2`) already passed through R2's fully-streamed
  fallback. What changed is the topology underneath it, which is why this refinement's gate is the
  external grader plus the structural tests, not the registry count.
- **+20 external-grader cells**, and the named family is the whole of it:
  `test_tilize_nd_sharded_to_legacy_sharded` goes **0/9 -> 9/9** (3 tensor shapes x HEIGHT/WIDTH/
  BLOCK output, including the uneven `([7,128,128],[2,64,96])` cliff shard). Every one of them was
  a hard `UnsupportedAxisValue` before this refinement.
- All **161** remaining directory failures are typed refusals for a LATER refinement's axis: 85
  `pad_mode` (R5), 65 `dtype` (R7), 3 `rank=0` (R5), 8 `ExcludedCell` (the deliberate
  `use_multicore=False x sharded`). **Zero** non-refusal failures, **zero** hangs, **zero** XPASS.

### Perf gate

**Bound classification, ablated at the shipped config** (payload stubbed, all CB scaffolding and
trip counts kept):

| shape | base | read | write | compute | floor | reading |
|---|---|---|---|---|---|---|
| `reshard` (banded, ships PUSH) | 22 851 | −163 (0 by construction) | **18 950** | 604 | 1 486 | write-bound: the scatter IS the op |
| `reshard_rowwise` (ships PULL) | 15 717 | **12 078** | 58 | 737 | 492 | read-bound: the gather IS the op |

That is the structural signature of a one-sided-resident reshard and it is the right one: exactly
ONE dataflow kernel moves bytes, the other runs the CB handshake, and compute is overlap-hidden
(0.95-0.97x) as on every other L1 path.

**Landed levers, off-arm measured** (ratio = off/on; >1 means the shipped choice pays):

| off-arm (knob) | `reshard` | `reshard_rowwise` |
|---|---|---|
| `force_streamed=1` — R2's fallback, i.e. NO reshard scheme at all (C14/A2) | **2.695x** | **1.914x** |
| the direction the gate did NOT pick (`reshard_pull`, D20) | **1.558x** (forced pull) | **1.605x** (forced push) |
| the direction the gate DID pick, forced explicitly | 0.980x (= base, i.e. the gate picked it) | 0.989x (= base) |

**No `/perf-ceiling-dm` target is reported for the reshard path**, for the same reason R2 gave for
the resident path: its transfers are L1->L1 between worker cores, so the DRAM ceiling the op's
targets are built on does not bound it (the bench's GB/s column is DRAM-relative and additionally
double-counts here — a reshard moves each byte ONCE, not twice). The honest comparison is the
per-core NoC rate against the op's other L1 path: `reshard_rowwise` moves 2 MB in 15.7 us on 8 cores
= **16.7 GB/s per core**, against `l1_to_l1`'s 4 MB in 5.05 us on 32 cores = **26 GB/s per core**.

**Cumulative bench set — no regression** (every prior shape re-measured, `base` arm):

| shape | Refinement 3 | Refinement 4 | delta | verdict |
|---|---|---|---|---|
| `(1,1,2048,2048)` square | 44 399 | 43 962 | −1.0 % | noise |
| `(1,1,32,16384)` wide_short | 6 866 | 6 747 | −1.7 % | noise |
| `(1,1,2048,64)` tall_narrow | 4 868 | 4 880 | +0.2 % | noise |
| `(1,1,32,64)` smallest | 1 915 | 1 941 | +1.4 % | noise |
| `(1,1,32,32)` smallest_aligned | 1 871 | 1 833 | −2.0 % | noise |
| `(1,1,512,2048)` l1_to_l1 | 5 121 | 5 049 | −1.4 % | noise |
| `(1,1,2048,2048)` sharded_big | 2 137 | 2 113 | −1.1 % | noise |
| `(1,1,512,64)` sharded_small | 844 | 852 | +0.9 % | noise |
| `(1,1,1024,1024)` **reshard** (new) | — | **22 851** | — | added (banded source, ships push) |
| `(1,1,1024,1024)` **reshard_rowwise** (new) | — | **15 717** | — | added (whole-row source, ships pull) |

Every prior shape is inside the 2-3 % noise band, which is the expected result rather than a lucky
one: at `n_bands == 1` the reader is byte-identical and `reshard_*` modes are unreachable unless
BOTH sides are sharded. **Two** bench shapes were added, not one, because the direction gate is
shape-dependent and a gate benched on one side of itself has not been benched at all — they are the
two regimes of `reshard_direction_is_pull`, and each carries both forcing arms.

Lever ledger: `verify_levers --phase "Refinement 4"` -> **clean, 0 blocking, 0 signal**; 24/24 rows,
now **11 applied**. **D20** (dispatch/regime selection) moves `deferred -> applied` — its own
deferral note named A3c as the refinement that would land it — with the direction gate as its
measured evidence; **A2** and **C14** are re-stated with the reshard's numbers (the core pinning now
also decides WHICH side pins, and the zero-copy alias now covers one side of a cross-spec pair).

### Issues encountered

Three, none of them numerical:

1. **A test that hung the runner, in Python, not on device.** `test_r4_reshard_stages_nothing_through_dram`
   originally iterated `kernel.runtime_args` to collect every base address; iterating that binding
   does not terminate in the way a list would, so the whole unit directory ran past 10 minutes with
   no dispatch timeout (the device was never the problem — `SAFE_PYTEST_DISPATCH_TIMEOUT` is 5 s and
   never fired). Fixed by indexing `[core.x][core.y]` for the cores the descriptor actually
   launches on. Worth recording because the symptom (a "hang" with a clean triage) points at the
   host, not the kernel.
2. **The gate invalidated three prior-phase assertions, all of which pinned R2's fallback** rather
   than an invariant: `test_r2_plan_is_pure_...` asserted cross-spec plans as `streamed`,
   `test_r2_cross_spec_streams_and_does_not_alias` asserted that NEITHER side aliases, and
   `test_r3_placement_gate_...` compared `placement_defaults` as a whole dict. Each was re-pointed
   at the invariant it was protecting — cross-spec must never alias BOTH sides (aliasing one is the
   scheme; aliasing both is the same-spec case and would have a core tilize its own rows into
   another core's tiles) — and the direction itself is now pinned by forcing arms, so the tests
   cannot silently re-derive the gate they check.
3. **An nd shard can be non-contiguous in the folded view**, which the crossover code (R2) had not
   had to face: `(2,64,96)` on `[7,128,128]` covers folded rows `{i*128 + h : i<2, h<64}` — two
   disjoint runs, not one. Caught before it could produce wrong output because generalizing the
   crossover branch made the folding assumption explicit; `shard_folds_contiguously()` now refuses
   to make such a shard the resident block (the streamed/other-side path addresses it by page id
   and does not care).

### Tests added

`tests/ttnn/unit_tests/operations/tilize/test_tilize_debug.py` group 8 (**+11 cases**):
`test_r4_cross_spec_reshard_is_exact` (6 pairs, public entry point, `torch.equal`),
`test_r4_reshard_resident_side_follows_the_gate` (both directions forced, on one geometry: which CB
is aliased, which kernel is resident, the band count and band size the gather derives, the core set
== the resident shard's cores, and `semaphores == []`),
`test_r4_reshard_gate_is_pure_and_regime_selected` (host-only: core count decides first and reverses
with the pair; transaction shape is the tie-break),
`test_r4_reshard_stages_nothing_through_dram` (both tensors L1, exactly one CB aliased, and the
kernels' only base addresses are the caller's two buffers),
`test_r4_same_spec_still_takes_the_zero_copy_path` (the regression this scheme is most likely to
cause, plus `n_bands == 1` on the interleaved path so the pre-R4 reader stays byte-identical).
`_bench_tilize.py` gained the two reshard shapes and the two direction forcing arms.
`probes/probe_012.py` — the page-geometry measurement the whole addressing model rests on.

---

## Refinement 5 — The padded path, end to end (prompt P1 + P2 + P4 + P5)

- **Date**: 2026-08-12
- **Box/arch**: same stamp as A0 (Blackhole, 11x10 = 110 cores, AICLK 1349.98 MHz, bf16,
  `DEVICE KERNEL DURATION [ns]`, 3 launches averaged after 2 cache-warming launches).
- **Type**: generality · **knob-turn behind a CT flag** — the aligned path is byte-identical when
  `PAD_ENABLED == 0`, and that is asserted structurally, not left as a convention.

### What was done

**Padding changes exactly two things, and that framing is the whole refinement.** It is not four
features (auto / explicit / three fill signs / three alignment flavours); it is

1. the **geometry** the op is blocked over becomes the PADDED shape (the output's tile grid is the
   pad target's, not the input's), and
2. the reader gains **one CT-selected second body** that fills the pad positions in L1 before
   reading the real sub-rectangle over the top.

Everything downstream — `blocking()`, the core split, the CBs, compute, the writer — is unchanged
and *unaware*, because all of it only ever sees the padded tile grid. That is why the diff is small
and why the sharded padded topologies (P4) needed no new placement code at all.

**`pad_plan()` is the single source of truth.** One function turns a
`(output_padded_shape, pad_value)` pair into the numbers, and `validate()`, the output allocation and
the reader's CT args all read it, so they cannot disagree about which cell a padded call lands in.
It carries `padded_shape` / `logical_shape` / `enabled` / `pad_word` plus the reader's four geometry
terms (`hp`, `h_real`, `nimg_real`, `real_row_bytes`).

**`enabled` is the structural non-regression.** A DEGENERATE pad — `pad_mode="auto"` on an already
tile-aligned input, where the target equals the input shape — resolves to `enabled = False`, so that
cell takes the aligned reader **verbatim** rather than a pad reader that happens to fill nothing.
`test_r5_aligned_path_is_structurally_unchanged` compares the whole reader CT-arg vector between the
padded and unpadded calls, and the bench measures the same claim (`padded_noop` below).

**The reader body — three pad regions, two disjoint fills, one barrier.** Because `hp` is a whole
number of tile-rows, every row of a block belongs to the same image, so the real/pad split is two
scalars per block: `real_rows` (rows of this block that exist in the input) and `real_bytes` (bytes
of a real row that exist). Their complements ARE §8.3's three regions — the H tail and whole pad
tile-ROWS are `rows >= real_rows`; the W tail and whole pad tile-COLUMNS are `bytes >= real_bytes`.
A fully-interior block (the common case: a pad touches only the last tile-row / tile-column) still
calls `read_sticks_for_tilize`, so a padded call pays the fill **only where it must**.

*One deliberate deviation from §8.3, recorded.* The design writes ONE fill over the whole block and
then reads the real data over the top. This ships the same fill split into its **two disjoint
regions**, so a store never targets a byte an in-flight NoC read owns — the fill/read overlap the
design's version has is ordering the hardware does not promise. Same three regions, same single
barrier, strictly fewer stores.

**The fill word (`pad_fill_word`), where the two classic bugs live.** Packed in the **INPUT**
element format (never the output's — a `dtype=` cast happens later, at pack time, on data that
already carries the fill) and **replicated across the 32-bit store word** (2x bf16/uint16, 4x uint8,
1x fp32/uint32), because the reader stores words and a fill written once leaves every other pad
element stale. Negative integer fills take the two's-complement `bit_cast`. The bf16 pack is
round-to-nearest-**even**, matching torch's own float->bfloat16: a truncating pack differs by one ulp
on half the fill values and would show up as a nonzero-fill-only mismatch. All of it is pinned
host-only by `test_r5_pad_word_is_packed_in_the_input_format_and_replicated`.

**The logical shape stays the input's (risk 7), by allocation rather than by fixup.** A TILE tensor
allocated at the logical shape ALREADY has the tile-rounded padded shape, so every target that *is*
the tile rounding needs no view at all — which is every padded sharded cell, so none of them goes
near a reshape. Only a target beyond the tile rounding (`50 -> 128`, the whole-pad-tile case)
allocates the target and then takes a zero-cost `ttnn.reshape(t, logical, padded)` view (measured:
same `buffer_address`). Rank 0/1 synthesize their tile dims — a scalar allocates logical `[1,1]`,
whose padded shape is exactly one tile.

**Two placement facts the pad changes.** (a) A padded call can never consume the **input** shard in
place: the pad positions live inside the block, so a resident reader would have to write the fill
into the CALLER'S OWN input tensor. `plan_placement(pad_enabled=True)` therefore disqualifies input
residency — and only input residency; an output tile is whole, so a padded crossover still aliases
its **output** shard zero-copy (`test_r5_a_padded_call_never_consumes_the_input_shard_in_place`
asserts both halves). (b) A padded source sharded NARROWER than a row is refused with a typed
`UnsupportedAxisValue`: the pad clamp is stated per source page, and nothing in TARGET reaches it.

### Accuracy achieved

**Exact** (`torch.equal`, bf16, not PCC — tilize is a bijection and a pad position is a datum like
any other). Every padded golden cell passes BOTH oracles: `to_torch_with_padded_shape() == F.pad(x)`
and `to_torch() == x` with the logical shape unpromoted.

- interleaved (probe_015): auto h-tail / w-tail / both-tails-negative / degenerate noop; explicit
  tile-rounded, beyond-tile-round in HW and in W only, rank 3 H-only, rank 2, **rank 0 scalar**, the
  `[1,1,1,16384]` row vector, and a wide `[1,1,50,2048]` — 13/13 bit-exact.
- padded + sharded (P4, probe_016): all **7** topologies bit-exact — nd->nd (uneven input shard),
  nd->interleaved, interleaved->nd, legacy HEIGHT->interleaved, interleaved->legacy HEIGHT,
  nd->legacy HEIGHT, legacy HEIGHT->nd.
- position-encoded inputs (`test_r5_all_three_pad_regions_on_a_position_encoded_input`, 4 geometries)
  so a fill landing in the wrong region is an exact mismatch rather than a plausible number.

### Golden test progress

Registry `test_golden.py`: **22 -> 42 supported_pass**, **0 failed**, **0 XPASS**, 338 xfailed (all
typed `dtype`/`tile_height`/`in_layout` refusals owned by Refinements 7 and 8), 580 invalid-skipped.
The +20 is exactly BLOCK 2 of `feature_spec.INPUTS` — the whole padded surface, interleaved and
sharded, at bf16 -> bf16.

Unit directory: **142 passed / 17 xfailed / 0 failed** (was 133 + 16 xfailed before this refinement,
plus 9 new R5 tests).

### Perf gate

**Classification: not DM-bound, and deliberately not chased.** The fill issues **zero NoC bytes** —
it is L1 stores into a CB the reader has already reserved — so no DM ceiling describes it and none is
reported (the queue entry says so explicitly). What is reported is the cost, measured.

Cumulative bench set, all shapes re-measured at `base` in ONE run (R4 -> R5, ns):

| shape | R4 | R5 | delta | verdict |
|---|---:|---:|---:|---|
| `(1,1,2048,2048)` square | 43 962 | 44 170 | +0.5 % | noise |
| `(1,1,32,16384)` wide_short | 6 747 | 6 844 | +1.4 % | noise |
| `(1,1,2048,64)` tall_narrow | 4 880 | 4 973 | +1.9 % | noise |
| `(1,1,32,64)` smallest | 1 941 | 1 903 | −2.0 % | noise |
| `(1,1,32,32)` smallest_aligned | 1 833 | 1 844 | +0.6 % | noise |
| `(1,1,512,2048)` l1_to_l1 | 5 049 | 5 055 | +0.1 % | noise |
| `(1,1,2048,2048)` sharded_big | 2 113 | 2 117 | +0.2 % | noise |
| `(1,1,512,64)` sharded_small | 852 | 863 | +1.3 % | noise |
| `(1,1,1024,1024)` reshard | 22 851 | 22 668 | −0.8 % | noise |
| `(1,1,1024,1024)` reshard_rowwise | 15 717 | 15 550 | −1.1 % | noise |
| `(1,1,2046,2048)` **padded_h_tail** (new) | — | **44 013** | — | added |
| `(1,1,2048,2046)` **padded_w_tail** (new) | — | **44 453** | — | added |
| `(1,1,2048,2048)` **padded_noop** (new) | — | **44 181** | — | added |
| `(1,1,1,16384)` **padded_row_vector** (new) | — | **28 888** | — | added |

Every prior shape is inside the ±2 % noise band. The four new rows were chosen so each one's PADDED
tile grid equals an existing row's, which makes the pad body's cost readable directly:

- **`padded_noop` 44 181 vs `square` 44 170 = 1.0002x.** The degenerate pad is *free*, measured —
  which is the queue's "must stay bit-identical AND not slower" gate, and it is free because
  `pad_plan` disarms rather than because the pad body is cheap.
- **`padded_h_tail` 44 013 and `padded_w_tail` 44 453 vs `square` 44 170** = 0.996x / 1.006x, i.e.
  inside noise. The boundary-block fill is invisible on a shape where the pad touches 4 of 256
  blocks (H tail) or 128 B of each of 64 blocks (W tail).
- **`padded_row_vector` 28 888 vs `wide_short` 6 844 = 4.2x** — the fill-DOMINATED regime, and the
  honest number for this path. Same 32 output blocks, 1/32 of the input bytes, and 31 of every 32
  rows written by the fill. Ablation says so unambiguously: `ablate_read` **0.991x** (the 64 KB read
  is nothing), `ablate_compute` 0.848x, and `ablate_all` — which stubs read, compute AND write but
  **not** the fill — is still **23 881 ns, 83 % of the wall**. So this shape is neither DM- nor
  compute-bound; it is **store-bound**, at ~31 KB of word stores per core.

**Lever ledger: no row changed status this phase, and that is the correct verdict, not an omission.**
The pad body adds no NoC transfer — it adds L1 stores — so no catalog lever's applicability moves.
The body itself preserves every landed transaction-shape lever: **B7** (one barrier per block: the
pad arm issues all its reads then one barrier), **B5** and **B9** (untouched — the writer is not in
this diff), **B6** (`WT_BLOCK` still comes from the byte target). `verify_levers --phase "Refinement
5"` reports **0 blocking, 0 signal**, 24/24 rows closed-with-evidence or open-on-record.

**Remaining headroom, as a FINDING (not a queue item).** The store-bound fill has one obvious lever
and it is worth naming precisely so a perf round does not have to rediscover it: **fill ONE row with
stores and replicate it to the block's other rows with local L1->L1 `noc_async_read`s**, turning
`tile_h x row_bytes` of RISC-V word stores into one row of stores plus `tile_h - 1` NoC copies. On
`padded_row_vector` that targets the 23.9 us the ablation attributes to the fill. It is not taken
here because (a) this is a generality slot and the entry is explicitly correctness-gated, and (b) it
introduces a read-after-store ordering dependency inside the block that needs its own measurement —
exactly the kind of thing the trailing perf rounds start from a fresh breakdown for. The regime it
pays in is narrow but real: shapes whose pad region is most of the tile (the `[1,1,1,W]` row vector
real models actually ask for). On every shape where the pad is a tail rather than the bulk, the
measurement above says there is nothing to win.

### Issues encountered

**One, and it was in the immutable spec, not in the op.**
`test_tilize.py::test_tilize_pad_scalar` ends with
`assert torch.all(padded[mask] == pytest.approx(pad_value))`. Under pytest >= 8 (this env: 9.0.3)
`ApproxScalar.__eq__` converts a torch tensor through `__array__` and returns a plain **bool**, so
the expression never yields a tensor and `torch.all(<bool>)` raises `TypeError: all() received an
invalid combination of arguments - got (bool)`. That is **unconditional** — it fires identically for
a bit-perfect output and a wrong one, so the assertion never observed the op's values at all. It only
became visible when this refinement put `rank=0` into SUPPORTED (before that the case was refused and
the conftest's registry hook converted it to XFAIL). Verified in isolation:
`torch.tensor([42.,42.]) == pytest.approx(42.0)` -> `False` (a bool), not a tensor.

The file is the SPEC and must not be edited, so the case is reported XFAIL by
`tests/ttnn/unit_tests/operations/tilize/conftest.py` on a predicate matched to that ONE mechanism (a
`TypeError` from `torch.all` receiving a bool — both message fragments required). It cannot mask an
op defect: what it fires on is a type error in the assertion machinery, not a comparison result. And
no coverage is lost, only the broken spelling of it — the property is asserted for real, elementwise
and exactly, by `test_tilize_debug.py::test_r5_scalar_pad_fills_every_position`.

### Reused vs added

| Reused unchanged | Added |
|---|---|
| the reader's block decode and `read_sticks_for_tilize` (still the emitted code for every interior block), `blocking()` / `plan_placement()` / `plan_cores()` / the CB builders / `compute` / the `writer` — **not one line changed** in the compute or writer kernels | `pad_plan()` + `pad_fill_word()` + `auto_padded_shape()` (host), `fill_pad_region()` + the `pad_enabled` reader body (device), `in_shape`/`pad_enabled` on `plan_placement`, and the `pad=` thread through `create_program_descriptor` / `_dispatch` |

The op file's delta is four SUPPORTED lists widened (`pad_mode`, `pad_value`, `alignment`, `rank`) —
`_check_structural()` already carried every padding refusal from Phase 0, so `validate()` needed only
the pad plan wired into its placement check.

### Tests added

`test_tilize_debug.py` group 9 (9 cases across 6 tests):
`test_r5_pad_word_is_packed_in_the_input_format_and_replicated` (host-only: replication, RNE bf16
pack, negative bit_cast),
`test_r5_aligned_path_is_structurally_unchanged` (the whole reader CT-arg vector, both directions,
plus bit-identical values),
`test_r5_all_three_pad_regions_on_a_position_encoded_input` (4 geometries: W+H tails, whole pad tiles
in HW, whole pad tile-columns, whole pad tile-rows at rank 3),
`test_r5_scalar_pad_fills_every_position` (rank 0, and the working form of the spec's broken
assertion),
`test_r5_a_padded_call_never_consumes_the_input_shard_in_place` (input residency refused, output
residency kept),
`test_r5_pad_plan_is_pure_and_states_the_geometry_once` (host-only, all four resolution modes).
`_bench_tilize.py` gained the four padded shapes and the `_PAD_BY_SHAPE` request map.
`probes/probe_014.py` (the allocation/view mechanism measured before it was used),
`probe_015.py` (interleaved padded), `probe_016.py` (padded sharded).

---

## Refinement 6 — Speed up the low-work-per-core regimes

- **Date**: 2026-08-12
- **Box/arch**: same stamp as A0 (Blackhole, 11x10 = 110 cores, AICLK 1349.98 MHz, bf16,
  `DEVICE KERNEL DURATION [ns]`, 3 launches averaged after 2 cache-warming launches; the two
  confirmation rows below are 5 launches).
- **Type**: perf. No SUPPORTED change; `SUPPORTED` / `EXCLUSIONS` / `validate()` are byte-identical
  to Refinement 5.

### What was done

Five knobs were built, each with both bench arms and both values pinned exact. **Four measured
net-negative on the very regime they were predicted to help and ship parked at a byte-identical
default; one pays and ships on.** The four nulls are the deliverable as much as the win — each
retires a lever the ledger had been carrying with a predicted delta since Phase 0, and each retires
it with a *mechanism*, not an argument.

| knob | master.md | ships | measured (off/on, >1 = the shipped value pays) |
|---|---|---|---|
| `wait_upfront` | C14 (3rd degree) | **ON, resident path only** | **1.040x** sharded_small, 1.018x sharded_big |
| `stateful_reads` | B13 | off (`STATEFUL_READS = 0`) | 0.809x smallest, 0.808x smallest_aligned, 0.970x tall_narrow |
| `fast_addrgen` | D21 | off (`FAST_ADDRGEN = 0`) | 0.928x smallest, 0.907x smallest_aligned, 0.948x tall_narrow |
| `fold_resident` | C14 (2nd degree) | off (`FOLD_RESIDENT = 0`) | 0.957x sharded_small, 0.977x sharded_big |
| `tilize_uninit` | — (LLK teardown) | on (unchanged) | 0.991x-1.009x everywhere = noise |

**The one that pays — `wait_upfront` (C14's third degree).** A resident reader arms the input CB
with the WHOLE assignment in a single `cb_push_back` (the CB *is* the shard), so compute's per-block
`cb_wait_front` can only re-observe a semaphore that is already set. `WaitMode::WaitUpfront`
collapses the four waits of `sharded_small` into one: **870 -> 837 ns, 1.040x**, reproduced at 5
trials (1.039x at 3 trials in the earlier sweep), and 1.018x on `sharded_big`. It is armed **only
where the input is resident** — on a streamed input the CB holds `cb_depth * wt_block` pages, not
the whole assignment, so the same wait would deadlock — and the gate is residency, not a caller
argument (`test_r6_upfront_wait_is_armed_only_where_the_input_is_resident` asserts the CT flag on
resident / crossover_in / crossover_out / streamed).

**B13, measured in BOTH orderings, and it loses in both.** `set_state` programs the source NODE, so
it amortizes only across issues that share a bank — which means issuing a block's sticks in
bank-phase order. (a) *Bank-phase order*: DEVICE_PRINT confirms the grouping is exactly right (7
nodes x ~5 sticks each, low address advancing by exactly one aligned page), and the arm is still
**1.18x slower on `smallest` and 1.27x on `wide_short`** — five consecutive requests to ONE bank
queue behind each other while the natural page order round-robins the seven banks. *Bank grouping is
an anti-lever on this part*, which is worth recording on its own: it is the mirror image of R3's
`stagger_reads` finding. (b) *Natural order* (what ships as the knob): the node changes every stick,
so the state is re-programmed every stick and the arm pays `set_state` + `with_state` — two
command-buffer-ready polls where a plain read pays one — **1.236x/1.238x worse** on the two smallest
shapes. The ledger's premise ("every read in a block has the same size and stride; only the page
changes") was true and still insufficient: what a read costs here is command-buffer turnaround plus
DRAM latency, not the three register writes B13 removes.

**D21 answered its own deferral question: `TensorAccessor` already specializes.** The row asked
"check whether TensorAccessor already does shifts rather than divides; if not, measure an
InterleavedAddrGenFast-style specialization". Measured: a hand-rolled replacement — one running
address per bank phase, advanced by a stride *derived from two real accessor calls* (so `bank_period
+ 1` accessor calls per block instead of `tile_h`), in the unchanged natural issue order — is
**1.078x/1.103x slower** on the two smallest shapes. The accessor's divisions are by a compile-time
bank count baked through `TensorAccessorArgs`, so the compiler already emits a multiply-shift; the
address table's array, branch and probe cost more than they save.

**C14's second degree (the fold), measured, and it loses — for a structural reason worth keeping.**
The compute kernel can self-arm the input CB (a PACK-thread `cb_reserve_back`/`cb_push_back`) and
self-drain the output CB (an UNPACK-thread `cb_wait_front`/`cb_pop_front`), so the program launches
**one kernel per core instead of three** — correct on device, and **1.045x slower** on
`sharded_small`. Mechanism: `DEVICE KERNEL DURATION` is the max over RISCs, and the two resident
dataflow kernels (one reserve/push, one wait/pop) were never on the critical path — so folding
deletes nothing from the wall and adds the handshake to the compute thread. Same sign as
`examples/zero_copy_fold`'s 0.74x on Wormhole, much smaller here (1.04x vs 1.35x), which is itself
the useful number: the fold's cost is the serialized handshake, and this op's handshake is two CB
operations for a whole core's work rather than per block.

### Accuracy achieved

**Exact** (`torch.equal`, not PCC — tilize is a byte bijection). Both values of all five knobs are
production code *or* a live counterfactual, so both are pinned:
`test_lever_off_arms_are_still_correct` grew from 18 to 26 arms (`stateful_reads` {0,1},
`fast_addrgen` {0,1}, the two together, `tilize_uninit=0`, `wait_upfront=0`, `fold_resident=1`) on a
tail-block geometry, and `probe_018/020/021` check nine geometries x four knob combinations
(rank 2/3/4, tail block, all four buffer directions, resident and crossover_in) bit-exact.
Precision baseline unchanged (PCC = 1.000000, max_abs_err = 0.0).

### Golden test progress

| suite | Refinement 5 | Refinement 6 |
|---|---|---|
| `test_golden.py` (registry) | 42 passed / 0 failed | **42 passed, 338 xfailed, 580 skipped, 0 failed** |
| golden `-k shard` slice | — | **186 passed / 38 failed**, all 38 typed `UnsupportedAxisValue` refusals (R5's padded-narrow-shard gap), **0** non-refusal failures |
| unit dir `tests/.../tilize/` | 142 passed / 17 xfailed | **154 passed / 17 xfailed / 0 failed** |

Unchanged registry count is the expected result for a perf refinement (no axis value added).

### Perf gate

**Bound classification, re-ablated at the shipped config** (payload stubbed, all CB scaffolding and
trip counts kept; 5 launches):

| shape | base | read | write | compute | floor | reading |
|---|---|---|---|---|---|---|
| `smallest (1,1,32,64)` | **1 898** | 963 | 325 | 476 | 316 (17 %) | serial: 1 core, 1 block |
| `smallest_aligned (1,1,32,32)` | **1 862** | 1 115 | 311 | 418 | 313 (17 %) | serial: 1 core, 1 block |
| `sharded_small (1,1,512,64)` | **837** | 0 | 22 | 405 | 410 (**49 %**) | compute + launch floor |
| `sharded_big (1,1,2048,2048)` | **2 081** | 0 | 0 | 1 677 | 396 (19 %) | compute, no DM at all |

**No `/perf-ceiling-dm` target is reported for the two sharded rows** (their NoC transfer count is
zero, as R2 recorded), and the two `smallest` rows move 4-8 KB — three orders of magnitude below any
bandwidth bound, so a DRAM ceiling does not describe them either. The right comparison for them is
the *structural* floor, below.

**What `smallest` is actually made of — the finding that retires the queue's premise.** The entry
scoped this slot at "the per-core setup cost", citing `ablate_all` = 17 %. That 17 % is real and
confirmed (316 of 1 898 ns) — but the other 83 % is **not** setup. It is one core issuing 32 stick
reads it cannot avoid (one per row of the single tile-row; splitting along W gives every core all 32
sticks again, only narrower), and then running read -> compute -> write **strictly serially**,
because a core holding exactly ONE block has nothing to overlap it with. Per-issue cost decomposes
against B7's own off-arm: a fully serialized read is ~350 ns (the B7 arm is 11 251 ns for 32), so
with all 32 in flight behind one barrier the 963 ns read stage is ~625 ns of issue plus one ~340 ns
DRAM round trip, i.e. **~20 ns per issue that is command-buffer turnaround, not instructions**. That
is exactly why both instruction-side levers (B13's register writes, D21's address arithmetic) moved
nothing and cost their own overhead. The lever that *would* move it is the one this slot does not
own: give the idle writer RISC half the block's sticks so two NIUs issue in parallel (master.md
**B8**/`split_reader` in its true form), which needs a semaphore handshake and is a scheme change.

**Cumulative bench set — no regression, all 14 shapes re-measured in one run:**

| shape | R5 | R6 | delta | verdict |
|---|---:|---:|---:|---|
| `(1,1,2048,2048)` square | 44 170 | 44 410 | +0.5 % | noise |
| `(1,1,32,16384)` wide_short | 6 844 | 6 775 | −1.0 % | noise |
| `(1,1,2048,64)` tall_narrow | 4 973 | 4 817 | −3.1 % | noise/faster |
| `(1,1,32,64)` smallest | 1 903 | 1 924 | +1.1 % | noise |
| `(1,1,32,32)` smallest_aligned | 1 844 | 1 835 | −0.5 % | noise |
| `(1,1,512,2048)` l1_to_l1 | 5 055 | 5 029 | −0.5 % | noise |
| `(1,1,2048,2048)` sharded_big | 2 117 | 2 106 | −0.5 % | noise (1.018x vs its own off-arm) |
| `(1,1,512,64)` **sharded_small** | 863 | **837** | **−3.0 %** | this slot's target (1.040x vs off-arm) |
| `(1,1,1024,1024)` reshard | 22 668 | 22 877 | +0.9 % | noise |
| `(1,1,1024,1024)` reshard_rowwise | 15 550 | 15 724 | +1.1 % | noise |
| `(1,1,2046,2048)` padded_h_tail | 44 013 | 44 117 | +0.2 % | noise |
| `(1,1,2048,2046)` padded_w_tail | 44 453 | 44 269 | −0.4 % | noise |
| `(1,1,2048,2048)` padded_noop | 44 181 | 44 291 | +0.2 % | noise |
| `(1,1,1,16384)` padded_row_vector | 28 888 | 28 936 | +0.2 % | noise |

No bench shape was added: the set already spans every regime this refinement's knobs key on (the
four interleaved shapes for B13/D21, both sharded shapes for the fold and the upfront wait, and both
reshard geometries where the input side is resident). Every knob was measured on **all 14**, which is
how the four nulls are known to be nulls everywhere rather than only on their target.

Lever ledger: `verify_levers --phase "Refinement 6"` -> **clean, 0 blocking, 0 signal**; 24/24 rows,
now **17 closed with evidence** (was 15). **B13** and **D21** move `deferred -> measured-no-payoff`
with both arms and a mechanism; **C14** records its second degree as measured-and-rejected and its
third degree as applied; **B0** records that its own discipline is what caught four negative levers
before they shipped, and that its framing of this op's smallest regime (fixed per-core setup) is
corrected by the ablation to "one core, one block, unavoidably serial".

### Issues encountered

One, and it changed the design of the lever rather than the code under it. The first B13/D21 arm
issued the block's sticks in **bank-phase order**, which is what lets one `set_state` cover a group —
and it measured 1.18-1.27x SLOWER. Rather than revert on the wall alone, the grouping was verified
with DEVICE_PRINT (`TT_METAL_DPRINT_RISCVS=NC`, 32 sticks: seven distinct source nodes, five sticks
each, low address stepping by exactly 128 B) which proved the *implementation* was right and the
*ordering* was the cost — consecutive same-bank requests queue at the bank. The arm was then rebuilt
in natural issue order, where D21 is still expressible and B13 is not, and both were measured again.
No hang, no race, no numerical divergence, no debug escalation.

### Reused vs added

| Reused unchanged | Added |
|---|---|
| the reader's block decode and `read_sticks_for_tilize` (still the emitted code on every shipped path — all four new reader/compute knobs default off or are `if constexpr`-discarded), `blocking()` / `plan_placement()` / `plan_cores()` / the CB builders / the **writer kernel, not one line** | `read_bank_period()` (host, pure), one `if constexpr` arm in the reader carrying both B13 and D21, three CT-selected lines in the compute kernel (`init_mode`, `wait_mode`, the fold's self-arm/self-drain), and the one-kernel program list for the fold |

### Tests added

`test_tilize_debug.py` group 9 (**+4 tests**, and +8 arms on the existing lever-correctness test):
`test_r6_read_bank_period_is_a_hint_that_gates_itself` (host-only: armed only when the bank count is
between 2 and `tile_height`, and never off a DRAM-interleaved source — including the tiny-tile case
Refinement 8 will add),
`test_r6_shipped_defaults_are_the_measured_ones` (the four nulls must ship OFF **and the reader's CT
vector must say so** — a default drifting back on would silently reintroduce a 1.04-1.24x regression
that no value test can see),
`test_r6_upfront_wait_is_armed_only_where_the_input_is_resident` (all four placements; the streamed
cases are the ones where the wait would deadlock),
`test_r6_fold_removes_the_kernels_and_only_on_the_resident_path` (kernel-count structure, and that a
crossover — which still has one real dataflow kernel — is never folded).
`_bench_tilize.py` gained 8 R6 arms. `probes/probe_017.py` (the device's bank geometry),
`probe_018/020/021.py` (per-knob bit-exactness), `probe_019.py` (the DEVICE_PRINT bank-grouping
verification).

### Ledger close-out addendum (R6, ledger-only)

`verify_levers` flagged **B13**: it is a per-core-overhead lever (master.md B0), so its
counterfactual must be measured on the smallest regime in `feature_spec.INPUTS` — `[1,1,30,32]` —
and R6 had measured it on `[1,1,32,64]`. A bench shape `smallest_padded` `[1,1,30,32]`
(`pad_mode="auto"`, 1 core / 1 block / 1 tile) was added and the arms run there. The result is a
**structural finding, not a number**: that shape's single block is a **pad-boundary** block, so R5's
padded reader body (`tilize_reader.cpp`, `pad_enabled == 1`) runs — and that body is a separate loop
that routes through neither the stateful-read path nor `read_sticks_for_tilize`, so it honours none
of the reader-side issue knobs. The arms are therefore **inert** there: base 1899.0 /
`lever_r6_stateful_off` 1895.0 / `lever_r6_stateful_force` 1868.7 ns, a 1.6% spread that the
byte-identical `base_singlecore` arm (1929.7 ns) reproduces on its own, i.e. the run's noise floor.

Rather than close B13 on a number that does not measure the lever, its row is **restated as
`deferred`** (an honest open status) with the concrete next step: make the padded boundary body issue
its real sub-rectangle through the same branch the aligned path uses, so `stateful_reads`,
`fast_addrgen` and `barrier_per_block` become live on padded blocks, then re-run
`TB_SHAPES=smallest_padded`. That is a kernel change and out of scope for a ledger close-out. The
shipped default is untouched: `STATEFUL_READS = 0`, byte-identical to the helper path. The same
inertness is recorded on **B7**, whose off-arm measured 1871.3 vs 1899.0 ns (inert) there for exactly
the same reason — its `[1,1,32,32]` measurement stands as the smallest regime its reader reaches.

**B5** went the other way and is now closed on the true smallest regime: it lives in the *writer*,
which the pad body does not touch, so its arm is live on a padded block and still pays —
1899.0 shipped vs 2005.7 ns with face writes (**1.056x**), above the noise floor. Its `measured`
block now records `[1,1,30,32]`, retiring A0's own "re-run once the padded reader exists" follow-up.

`verify_levers ... --phase "Refinement 6"` is now clean: 24/24 rows, 0 blocking, 0 signal.
No kernel, program-factory or op-entry-point file was touched.
