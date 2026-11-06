# Verification Report: nomic_sdpa

Verifier pass on Phase 0 (2026-11-05, Blackhole p300c, 11x10 grid). Golden results are in
`verifier_results/` (`verifier_report.json`, `test_results.json`).

**Headline.** The op is correct and the registry is honest: 45/45 supported cells pass, and every loud category is 0.
It is **not yet fit for purpose on speed**. The perf focus 8x512 runs at **~130 µs**, against the incumbent's
**72.1 µs** and a **58 µs** acceptance ceiling (62.6 µs model gate). The guards 2x512 / 4x512 run at 54 / 79 µs,
against 24.5 / 38.2 µs for the incumbent (they must never be slower). Closing that gap is what the perf refinements in
`op_requirements.md` are for.

## Code Review

### Fixed in place
1. **Precision contract: `validate()` honored a caller config that cannot meet the bar.** The prompt says to refuse,
   with a SupportRefusal, any compute config that cannot meet the golden bar. bf16 DEST + approximate exp (the
   incumbent's own setting) was accepted, and it misses the bar at S = 512. Measured: real rel-RMS 0.0269–0.0273 >
   0.0265 at 2x512 / 4x512 / 8x512, and flat 0.0194–0.0196 > 0.0175, still failing at HiFi3 / HiFi4. It passes at
   S ≤ 384 (8x384 0.0255, 16x256 0.0242, 32x128 0.0220). `validate()` now refuses that combination at
   `S > BF16_DEST_APPROX_EXP_MAX_SEQLEN = 384`, with the measurements cited at the constant. bf16 DEST + exact exp
   meets the bar on all three profiles at every measured shape and stays honored. New tests in
   `test_nomic_sdpa_precision_baseline.py`: `test_nomic_sdpa_refuses_bf16_dest_approx_exp_at_512`, and
   `test_nomic_sdpa_bf16_dest_exact_exp_meets_bar` (8x512 × 3 profiles).
2. **DRY: the common-RT table offset was restated as kernel literals.** The reader hard-coded `kTableArg = 3` and the
   writer `kTableArg = 1`, while the host's `READER_ADDRESS_ARGS` / `WRITER_ADDRESS_ARGS` were declared but unused.
   Now each kernel gets the named CT arg `table_arg` from those constants (compute gets `COMPUTE_ADDRESS_ARGS = 0`),
   and `bind_call` asserts the address count against the same constant. Adding an io address (for example key-mask
   data in Refinement 1) now lands in one place.
3. **Perf harness** (`test_nomic_sdpa_perf.py`): its `bf16dest_hifi2` config is now refused, so it is replaced by the
   honored bf16 DEST + exact-exp config. An fp32 DEST full-sync config was added (measured below).

### Reviewed, no change needed
- **Registry conformance**: see below.
- **Prompt MUST rules** (all hold):
  - one `generic_op` per call;
  - kernel names `nomic_sdpa_{reader,compute,writer}.cpp`, io order `[q, k, v, out]`;
  - no host-side reshape / pad / to_layout, and nothing staged in DRAM; the score matrix lives only in `cb_scores`
    (one core's block);
  - no C++ host changes.
  The `key_lengths`-per-call rule is armed for Refinement 1 (today `key_lengths` is rejected by the SUPPORTED gate).
  The soft rule "prefer kernel_lib helpers" is followed: matmul_block ×2, reduce, eltwise_chain ×3, the
  reduce-scaler helper and the local-copy helper. The raw rows (ring pads, K/V release, page I/O) are justified in
  `op_design.md` § API Mapping.
- **CB sync**: push and wait/pop counts were traced for all 8 CBs, including the nominal-quanta ring pads and the
  in-place `cb_scores` rotation. They balance for ragged `block_rows` (every perf shape has 13/14- or 1–4-row
  blocks, and they pass).
- **APIs**: `TensorAccessor` throughout, `void kernel_main()`, `api/...` include paths.
- **Broadcasts**: the `Sub` in `probs_block` and the `Mul` in `normalize_block (b)` both use `BroadcastDim::Col`
  against REDUCE_ROW / column-0 data. The ones column holds 1.0 in column 0 only, which is enough because only
  column 0 of `l` is read. The `recip` of the other 31 columns gives `inf` in unread lanes; that is harmless, and
  lamp L8 (`VectorMode::C`) removes the waste.
- **Helper usage**: there is no unfused phase that a helper could fuse today without the restructure in lamp L5
  (normalize in the matmul `PostComputeFn`). That is filed in Refinement 4 because it also removes `cb_pv_out`.
- **Host time**: 54.5 µs a call at 8x512 with a cached descriptor (target ≤ 100 µs). The two live-allocator scans
  and the per-call `plan()` are inside that budget.

### Design conformance (binding dimensions)
| Dimension | Verdict |
|---|---|
| Algorithm | Matches: exact two-pass softmax over the whole resident key range, scores→P in place, row sum through the ones column, `O·(1/l)`. |
| Pipeline / RISC ownership | Matches: the reader does K (K^T order), Q, then V; compute runs 5 block phases; the writer handles the scaler and concatenated-heads pages. |
| Work distribution | Matches: all 110 cores for every shape with ≥ 110 query rows (contiguous balanced ranges from a common-RT table indexed by core). Both dataflow halves are batched: one barrier per operand block in the reader, one write barrier per block in the writer. |
| Inter-core communication | None, as designed for the built regime. The deferred multicast's premise is falsified (see L1 audit), so it is filed as Refinement 2. |
| Blocking-model fidelity | Knobs are parameters with one source (`plan()` → CT args): `q_block_rows`, `key_block_tiles`, the three depths, and the core table. `kv_depth = 2` was exercised by the implementer and runs correctly, so the knob is live. There is no half-turned split: the per-core block is the whole per-core assignment. Expression is block-shaped in all three kernels: the writer's per-row `cb_wait_front` inside a block issues writes as rows arrive, with one barrier per block, so there is no per-unit completion boundary. |
| **Deviation** | The precision default moved from the design's bf16 DEST to **fp32 DEST** (the implementer measured that the design default misses the bar). This halves DEST to 4 tiles, so the subblocks become Q·K^T 1×4 and P·V 1×3. The deviation is documented in `default_compute_kernel_config()`, `l1_ledger.md`, and now in a note in `op_design.md` § Precision. |
| **Performance conformance** | **Not met.** The design's targets (58 µs at 8x512; guards never slower than the incumbent) are missed by 2.2× and 2.2–2.6×. The implementer's ablation (`l1_ledger.md` § Measured data movement) puts the DM floor at 70–80 µs and the compute floor at 73 µs at 8x512, and the two barely overlap (full run ≈ 125–133 µs). The data-movement and compute fixes are Refinements 2–4. They are not quick in-place fixes: one is a new topology, one a compute restructure. |

## Registry Conformance
- `INPUT_TAGGERS = {"alignment": tag_alignment}`, with signature `(inputs, axes)`.
- `SUPPORTED`: dtype [bfloat16], buffer [l1, dram], mask_mode [none], alignment [tile_aligned]. These are TARGET's
  axis names; `mask_mode` is derived inside `validate()` from `key_lengths`, as the prompt specifies.
- `EXCLUSIONS = []`.
- `validate()` is the first line of `nomic_sdpa`. Its order: malformed → ValueError, SUPPORTED → `UnsupportedAxisValue`,
  EXCLUSIONS → `ExcludedCell`, then domain / grid / scale / compute-config SupportRefusals. Live-L1 is checked
  before and after the output allocation.
- The op file does **not** declare `INVALID`.
- Drift auto-fixes: none needed (0 xpass_drift).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. There is one dtype (bfloat16) and one
  layout (TILE), so the bf8b+ROW_MAJOR canonical entry does not arise. No weight axes; no cross-tensor couplings.
  No change requested.

## L1 Ledger Audit
The full audit is appended to `l1_ledger.md` § Verifier audit. Summary:
- **Currency**: all 8 rows match `circular_buffer_specs()`, the single sizing source.
- **Size vs live set**: `cb_out` over-capacity is justified as decoupling. `cb_pv_out` is removable through the L5
  fusion, folded into Refinement 4. No collapsed extents.
- **Page format vs DEST**: fp32 DEST with all-bf16 pages is justified. Both matmuls finish their K accumulation in
  DEST (one K block, no packer L1 accumulation), so no partial sum is truncated at a phase boundary.
- **Disjoint lifetimes**: the row max and `1/l` already share `cb_row_stats`. Every other pair is concurrent, with a
  reason.
- **Block-size defaults**: held (full grid; block = the whole per-core assignment).
- **Per-core footprint**: `2048 · (Q·N + 8Q + 5N + 1)` bytes (`Q·N` from `cb_scores` dominates).
  - 834 KiB at 8x512 (Q = 14, N = 16), against 1055 KiB free at the call.
  - `Q` terms scale with `q_block_rows`, `q_depth` and `out_depth`.
  - `N` terms scale with `key_block_tiles` and `kv_depth` (`kv_depth = 2` adds `5N` pages: 994 KiB, still fits).
- **Traffic budget**: present and consistent with the code (K/V once per segment). **Finding:** the cheapest-traffic
  split (row-split + pair multicast) was deferred for a reason the implementer's measurement disproved. That makes
  it a **Refinement 2** perf entry, a standalone scheme-change.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nomic_sdpa/test_nomic_sdpa_precision_baseline.py`, default config (HiFi2, fp32 DEST,
approximate exp), L1, against the float64 golden reference. Ratio = got/true over elements with
|ref| > 0.1·RMS(ref).

| Shape | Profile | PCC | Max Abs Err | Mean Abs Err | Rel RMS | bf16 ULP p50 / p99 | got/true median (p5 / p95) |
|---|---|---|---|---|---|---|---|
| 1x128 | real | 0.999880 | 0.0402 | 0.00374 | 0.0220 | 4.1 / 186 | 0.984 (0.926 / 1.036) |
| 1x128 | flat | 0.999923 | 0.0034 | 0.00049 | 0.0142 | 2.6 / 132 | 0.993 (0.947 / 1.038) |
| 1x128 | peaky | 0.999972 | 0.0598 | 0.00342 | 0.0112 | 2.0 / 69 | 0.992 (0.961 / 1.006) |
| 4x288 | real | 0.999851 | 0.0493 | 0.00347 | 0.0244 | 4.7 / 209 | 0.982 (0.920 / 1.040) |
| 4x288 | flat | 0.999933 | 0.0023 | 0.00031 | 0.0133 | 2.4 / 119 | 0.993 (0.953 / 1.035) |
| 4x288 | peaky | 0.999918 | 0.1512 | 0.00429 | 0.0151 | 1.7 / 83 | 0.993 (0.960 / 1.016) |
| 2x512 | real | 0.999838 | 0.0644 | 0.00336 | 0.0251 | 4.8 / 216 | 0.981 (0.918 / 1.040) |
| 2x512 | flat | 0.999940 | 0.0016 | 0.00022 | 0.0127 | 2.2 / 105 | 0.993 (0.957 / 1.031) |
| 2x512 | peaky | 0.999851 | 0.1311 | 0.00508 | 0.0192 | 1.9 / 130 | 0.992 (0.943 / 1.032) |
| 8x512 | real | 0.999835 | 0.0641 | 0.00331 | 0.0255 | 4.9 / 214 | 0.981 (0.918 / 1.040) |
| 8x512 | flat | 0.999942 | 0.0018 | 0.00022 | 0.0126 | 2.3 / 116 | 0.993 (0.955 / 1.032) |
| 8x512 | peaky | 0.999852 | 0.1780 | 0.00510 | 0.0191 | 1.9 / 123 | 0.992 (0.945 / 1.029) |

ULP max is huge only where the reference is near 0, which is expected for a softmax-weighted average.

**Bar** (`PROFILE_TOLERANCES`): real (0.99965, 0.0265), flat (0.99985, 0.0175), peaky (0.99925, 0.0375). Every
cell passes. **The tightest margin is real at S = 512: 0.0255 against 0.0265 (4%).**

**Scale-vs-precision triage.** The got/true median sits at 0.981 on real and about 0.993 on flat / peaky: a uniform
shrink. Fitting `got ≈ c·ref` gives c = 0.982 and leaves 0.0177 rel-RMS on real at 2x512. **The ~1.8% scale term
is the largest single component of the error.** I attributed it before deciding how to route it (rel-RMS at 2x512):

| Config (2x512) | real rel-RMS | real ratio | flat ratio | peaky ratio |
|---|---|---|---|---|
| default: fp32 DEST, HiFi2, approx exp | 0.0251 | 0.981 | 0.993 | 0.992 |
| fp32 DEST, HiFi2, **exact exp** | **0.0114** | 0.994 | 0.997 | 1.000 |
| fp32 DEST, **HiFi4**, approx exp | 0.0194 | 0.989 | 0.997 | 0.995 |
| bf16 DEST, HiFi2, exact exp | 0.0210 | 1.000 | 1.004 | 1.000 |

The shrink follows the **approximate exp** (~1.2% of it) and **HiFi2** (~0.5%), not the kernel structure. The
structure is exact when both are lifted, and the spread is broad and centred near 1 once the exp is exact. This is
the incumbent's precision class: the incumbent uses the same approximate exp, and model_config.py records "most of
the error is the approximate exp". The golden bar was measured on that class. **It is not a structural scale bug,
so I filed no fix and no precision refinement.** The baseline test keeps a structural-scale tripwire at
|median − 1| < 3%.

Lever for later: the exact exp halves the error but measured **+60% device time** (below), so it is not a free
default.

**Recommended tolerances** for later refinements: the golden PROFILE_TOLERANCES as-is. Any perf change must keep
real at S = 512 ≤ 0.0265; the measured value is 0.0255, so there is little room.

## Verifier CLI Summary
(`verifier_results/verifier_report.json`, after the fixes)
- supported_pass: 45
- xfail_expected: 40. These are all `mask_mode=key_padding` or `alignment=ragged` cells, and all go to Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 5. These are the regression tests that self-xfail on the same missing axis values:
  `test_key_padding[2-512|8-512|3-400]`, `test_key_padding_lengths_are_per_call`,
  `test_finite_padding_rows_of_a_ragged_sequence`. All go to Refinement 1.

Acceptance: 39/39. Precision baseline + new config tests: 16/16.

### TARGET − SUPPORTED coverage
| Axis value | Disposition |
|---|---|
| `mask_mode = key_padding` | Refinement 1 |
| `alignment = ragged` | Refinement 1 (same mask mechanism, with `L_b = S`) |
| dtype, buffer | complete |

## Measured performance (Phase 0)

Device kernel time (µs, L1 placement). The golden loose cases include the 176 KB L1 reservation; the harness runs are
one call each and noisy.

| Shape | golden loose case | default (harness) | fp32 DEST HiFi3 | fp32 DEST full-sync | bf16 DEST + exact exp | incumbent | acceptance |
|---|---|---|---|---|---|---|---|
| 2x512 | 54.3 | 63.5 | 60.9 | 65.3 | 78.3 | 24.5 | 18.0 |
| 4x512 | 78.7 | 81.7 | 78.4 | 87.6 | 117.5 | 38.2 | 30.0 |
| 8x512 | 129.8 | 128.5 | 156.3 (noise) | 130.3 | 203.6 | 72.1 | 58.0 |

Reading:
- **Fidelity is not binding** (HiFi3 ≈ HiFi2).
- **Doubling DEST capacity buys nothing** (full-sync gives 8 fp32 tiles, the same time).
- **The exp is expensive**: the exact exp adds ~60%.

Together with the implementer's ablation, this gives:
- (a) Data movement is bound by **aggregate NoC bytes** (~450 GB/s chip-wide, the same for L1 and DRAM), and K/V
  are read 2.00× / 3.15× / 4.88× the minimum at 8x512 / 4x512 / 2x512.
- (b) The compute floor (~73 µs at 8x512) is ~2× the speed of light, with the SFPU exp serialized against the FPU
  phases.
- (c) Data movement and compute barely overlap: every core loads its first segment simultaneously, so the first load
  is exposed chip-wide.

## Recommendations
- **Queue order** (`op_requirements.md`):
  - R1 adds `key_padding` + `ragged`. It is the only generality bundle; the perf-focus contract is already fully
    supported.
  - R2–R4 are perf: R2 K/V reuse multicast (T3, scheme-change), R3 compute-side exp/FPU overlap (T3), R4 block,
    depth and cheap-lever co-tune (T1/T2).
  - The guards (2x512 / 4x512) are DM-dominated and should gain most from R2.
- **Precision headroom is thin** (4% at real S = 512). Every perf refinement must re-check all three profiles at
  S = 512. A perf lever that changes the exp form (lamp L3) or the row-sum route (lamp L4: `reduce<SUM>` over P
  loses the HiFi2 truncation cancellation) is the most likely to break the bar.
- **Ragged / prime key extents** (S = 400 gives N = 13): under fp32 DEST the Q·K^T subblock width falls to 1 (the
  largest divisor of 13 that is ≤ 4). That is correct but slow. It is not a perf target; Refinement 1 should not let
  it leak into tile-aligned shapes.
- **Planner feedback** (already in the implementer's friction log, confirmed here):
  - deferral decisions for byte-reducing regimes need an aggregate-NoC-bytes check, not only a per-core-ingress
    check;
  - the design's default precision class was not validated against the bar on device.
