# Verification Report: rms_norm

Terminology (qualified once, bare afterwards): *core* = Tensix core; *layout* = `ttnn.Layout`
(`TILE_LAYOUT` / `ROW_MAJOR_LAYOUT`); *placement* = `TensorMemoryLayout`; *tile* = 32×32 elements; *block* = the
blocking-model block of `op_design.md` §Blocking Model; *R1 / R2* = the design's regimes `row_local` /
`w_split_root_combine` (never the refinement numbers, which are written "Refinement N").

Board: Blackhole p150b, 12×10 = 120-core compute grid, CB budget 1 461 120 B per core (from `_l1_budget_bytes`).

## Code Review

Every item below was **fixed in place**. Nothing was deferred for architectural reasons.

1. **Registry bug: `INPUT_TAGGERS` collapsed the whole cartesian gamma path (fixed).** The op registered
   `tag_gamma_dtype` / `tag_gamma_layout` in `INPUT_TAGGERS`. `eval/feature_matrix.cartesian` does **not iterate**
   an axis that has a registered tagger ("the tagger wins"), and a tagger only sees the input shape. So every
   cartesian cell with `gamma_mode="gamma"` was tagged `gamma_dtype="none"` and skipped by the INVALID entry
   `{gamma_mode: gamma, gamma_dtype: none}`: **1680 of 1680 cartesian gamma cells were skipped, so the cartesian
   suite never ran gamma at all** (only the 9 `_WIDE` / sharded loose cases did). As a result, `gamma_dtype=bfloat8_b`
   was unreachable, and the "supported" gamma × dtype × layout × alignment surface was unverified.
   Fix: `INPUT_TAGGERS = {alignment, rank}` (the shape-derived axes only). The two gamma functions keep the prompt's
   `"none"`-sentinel contract in a separate `GAMMA_TAGGERS` dict that `_axes_for()` uses for `validate()`. The golden
   runtime classifier (`eval/golden_tests/rms_norm/axes.py`) already pins the sentinel itself.
   Effect: `supported_pass` 177 → **737**. All 560 newly exercised gamma cells pass, and the loud categories stay 0.
   A note was added to `op_design.md` so a later stage does not re-register them.
2. **Blocking-model DRY violation in the descriptor (fixed).** `_footprint()` (the L1 solve) and `_build_cbs()` (the
   allocation) each restated every CB's page count. `_footprint` used the module literal `X_DEPTH_BLOCKS` while
   `_build_cbs` used `plan.x_depth_blocks`, and `out_depth` (1 for RM, 2 for TILE), the 2-deep stick windows `2·c` /
   `2·b·c`, and every intermediate term were written twice. Turning the x-depth knob (perf lamp L1) would have
   changed the real allocation without changing the budget the selector solved against, which would cause a silent
   L1 overflow. Fix: one function, `_cb_specs(geo, b, n, c, k, L, gw, x_depth) -> [(cb, pages, page_bytes, format
   role)]`, is now the only definition of every CB. `_footprint` sums it, `_build_cbs` allocates from it, and the
   stream depth is a named local (`stream_depth = 2`). Golden and acceptance results are unchanged.
3. **`epsilon` contract not enforced (fixed).** The design's parameter table says `epsilon` must be finite and ≥ 0.
   `_structural_checks` now raises `ValueError` for non-finite or negative values.
4. **Prompt rules.** `eval/prompts/rms_norm.txt` has no `## Rules` section. The stock native-support policy holds:
   no `to_layout` / `tilize` / `pad` / `slice` in the entry point, and exactly one `ttnn.generic_op` dispatch per
   call. RM and non-aligned shapes are handled in-kernel (tilize-wrapped reader path, partial reduce scaler on the last
   tile, M9 zero-fill of the RM tail). The prompt's required `ValueError`s for rank < 2 and gamma-W mismatch are
   present.

Checked and found conformant (no change needed):

- **Kernel hygiene.** `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor`
  everywhere (no `InterleavedAddrGen`), and the inverse-RMS multicast uses the `ttnn.Mcast` + `McastArgs` helper.
  The many-to-one partial gather is raw unicast + `noc_semaphore_inc`, justified by design rejection R-d (the helper
  is one-to-many).
- **CB push/pop balance**, verified for all 13 CBs. Examples: `cb_x_tiles` gets `b·s` pushed per block and
  `k × b·c` popped (popped by the γ pass, or by normalize when there is no gamma). `cb_rowsum_gather` is pushed and
  popped `gw·b` on roots only. `cb_reduce_scaler` gets 2 pushed and 2 popped at kernel end.
- **R2 gather/multicast race safety.** The cumulative `noc_semaphore_wait_min` count is never reset. Each member
  reserves `cb_inv_rms` before sending its partial, and the gather capacity equals one block's push, so the write
  pointer on every core is the CB base (M1/M6).
- **Broadcast dims.** γ uses `BroadcastDim::Row` (row 0 only, rows 1–31 never read), inverse-RMS uses
  `BroadcastDim::Col`, and sum-of-squares is `Mul` with PerRow DEST accumulation (no `x²` buffer).
- **Helper usage.** Every compute phase is a kernel-lib helper (`eltwise_chain`, `mul`, `reduce` with
  `ReducePartialScaler`, `tilize`, `untilize`). The combine is `reduce<AccumulateViaAdd, ReduceWithinTile::Skip>`.
  The mean / +ε / rsqrt finalize is fused as the reduce post-op, so it never lands in L1.

### Design conformance

- **Algorithm.** It matches the design: sum of squares accumulates in DEST, the masked last tile is kept separate
  (M8), `1/W` uses the logical W, and γ is applied before the inverse-RMS (stall-shadow S1).
- **Topology / RISC-V ownership.** It matches: the reader loads scalers and x; the writer loads γ once, runs the R2
  combine and stores the output.
- **Work distribution (performance conformance).** R1 spreads tile-rows over up to all 120 cores. R2 spreads W
  across consecutive row-major W-groups when tile-rows < cores. Measured core counts: 120 cores on every prefill
  and 640-row profile, 4 / 9 / 20 / 28 cores on the 32-row decode widths 1024 / 2304 / 5120 / 7168. That follows
  `MIN_W_TILES_PER_CORE = 8`; see design lamp L2. Both dataflow halves are batched: the reader issues one barrier
  per chunk of `b·c` tiles (TILE) or per 32-stick slab (RM), and the writer flushes once per chunk.
- **Blocking-model fidelity.** `block_rows`, `chunk_w_tiles`, `chunks_per_row`, `xg_lookahead_chunks`,
  `w_group_size` and `x_depth_blocks` come from one `BlockingPlan`, are passed as compile-time args, and derive
  every CB and trip count. After fix 2, depth is single-sourced too.
  - No CB unconditionally scales with a whole-op dimension. `cb_x_tiles` spans the core's W slice
    (`core_w_tiles`), which is bounded by the L1 solve; past that bound the selector widens the W-group (R2), so
    there is no unbounded op dimension.
  - Expression is fine. Reader, compute and writer all handshake at chunk or block granularity, and per-tile loops
    exist only inside a chunk's single barrier or inside helpers.
  - Interleaved block-size default held: spread across the grid, then the coarsest `block_rows` that fits.

## Registry Conformance

- `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` and `validate()` are present. All taggers use the `(inputs, axes)`
  signature. `validate()` checks the output / gamma placement, then `SUPPORTED` per axis, then `EXCLUSIONS`, and
  raises `UnsupportedAxisValue` / `ExcludedCell`. `rms_norm()` runs structural `ValueError` checks, resolves
  `compute_kernel_config` through `default_compute_kernel_config()`, then calls `validate()` before any device work.
- The op file does **not** declare `INVALID`.
- No `SUPPORTED` auto-fix was needed: there was no `xpass_drift`. The only registry fix is the tagger fix above.
- `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]` is the precision-convention refusal the prompt
  mandates ("float32 input with fp32_dest_acc_en=False is rejected natively"). It is a **documented permanent
  omission**, not a refinement candidate.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`). Not edited; for the golden-test owner.

| Entry | Verdict |
|---|---|
| `{dtype: bf8b, layout: ROW_MAJOR}` | ✓ canonical, single-tensor, structural |
| `{gamma_dtype: bf8b, gamma_layout: ROW_MAJOR}` | ✓ same impossibility on the gamma tensor |
| 7 × `gamma_mode` ↔ `"none"` sentinel couplings | ✓ canonicalization of the no-weight cells (rule 3) |
| `{layout: RM, memory_layout: *_SHARDED, gamma_layout: TILE}` × 3 | ✗ **cross-tensor** (activation layout × activation placement × gamma layout) and self-described as "for now". This is an EXCLUSIONS-style scope decision parked in INVALID. Recommend moving it out of INVALID once Refinement 2 lands; RM-sharded input with a TILE gamma is not structurally impossible (gamma is loaded independently by the writer). |
| `{dtype: bf8b, alignment: w_non_aligned}`, `{dtype: bf8b, alignment: h_non_aligned}` | ✗ self-described "out of scope for now", not structural (bf8b tiles carry padding like any tile; the partial-scaler masking is format-agnostic). Recommend converting these to op-side `EXCLUSIONS` if they are ever to be built. Refinement 1 mirrors them as `EXCLUSIONS` (see its notes) because `test_translated.py` does **not** honour INVALID; that is where the 2 `invalid_unexpected` cells come from. |

## L1 Ledger Audit

- **Ledger currency.** All 13 CBs the code declares have a row, every row is a live CB, and every capacity
  expression matches `_cb_specs` (now the single source; the ledger §Footprint notes this). I added one missing
  concurrency fact: `cb_x_rm_sticks` is live at the same time as `cb_out_rm_sticks`, because the reader lands block
  b+1 slabs while compute untilizes block b.
- **Capacity vs. live set, over-capacity.**
  - `cb_gamma_tiles` capacity is `s`, live set `n` (≤ `k−1` pages of chunk rounding, which is M6 nominal quanta).
    Justified.
  - `cb_rowsum_gather` / `cb_inv_rms_root` are unused on non-root cores. Justified by M1: symmetric addressing is
    the remote-write mechanism.
  - `cb_sumsq` carries `2·b` pages where the live set is `b` when the slice holds one tile. Negligible (≤ 8 KiB) and
    nominal.
  - No buffer is sized to a block when its live set is one tile.
- **Capacity vs. live set, under-capacity (axis accounting).** `cb_x_tiles` spans `row` (`b`) and `w` (`s`), and
  its capacity scales with both. `cb_x_gamma` spans `w` only in R2 (`L = k`) and scales with `L`. No collapsed
  extents.
- **Page format vs. DEST width.** Every DEST-produced intermediate (`cb_sumsq`, `cb_rowsum_*`, `cb_inv_rms*`,
  `cb_x_gamma`) is a `Float32` page while `fp32_dest_acc_en=True`, which is consistent. **The format is hard-coded
  `ttnn.float32` (`_FMT_F32`), not derived from `fp32_dest_acc_en`.** That is correct today because only `True` is
  in `SUPPORTED`, but it becomes an *over* finding (2× packer / unpacker / L1 bytes for no precision) the moment
  `fp32_dest_acc_en=False` lands. **Folded into Refinement 1.**
- **Disjoint lifetime.** `cb_sumsq` ↔ `cb_x_gamma` and `cb_gamma_rm_stick` ↔ the output CBs are disjoint, and
  both pairs carry a stated reason in the ledger. No blank `Shares with` cells.
- **Bounds / closed form.** Every symbol has a bound and an establishing predicate in the ledger. The total is
  closed-form. `_select_blocking` raises `RuntimeError` (M3) rather than allocating past the budget.
- **Block-size defaults.** Held. The one departure (R2 lookahead `L` dropping to 1 under pressure) is the
  documented selector order, not a measurement-free deviation.
- **Per-core footprint.** In bytes:
  `2c·T_in[RM] + d_x·b·k·c·T_in + round_up(n·32·e_g, align)[RM γ] + k·c·T_g[γ] + 4096 + 2b·F32 + (2+gw)·b·F32[R2] + b·F32 + L·b·c·F32[γ] + o·b·c·T_out + 2b·c·T_out[RM]`.
  `b` scales the block terms, `c` and `L` scale the streaming windows, `gw` scales the gather, and `d_x` scales x.
- **Data-movement budget.** Present and consistent with the code. x crosses DRAM once and out once. γ is read once
  per row group (≤ 6.25 % of x bytes). R2 adds `(2·gw − 1)·b` fp32 tiles of cross-core NoC per block.
  Cheapest-traffic split: R1 is implemented wherever it fills the grid. Gamma multicast (R5) is deferred with a
  positive reason: it saves ≤ 6.25 % and sits below the catalog's handshake crossover.

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`, 16 cases (4 shapes × {bf16, fp32}
× {TILE, RM}). Inputs are N(0,1) x, an N(0,1) ROW_MAJOR gamma of the same dtype, ε = 1e-6, at the Phase 0 corner
(HiFi4, `fp32_dest_acc_en=True`). Errors are measured against the fp32 torch reference. ULP is in units of the
output dtype's last place. The ratio is median [p5, p95] of actual / expected. TILE rows are shown; the RM rows are
identical for bf16 and up to ~20 % larger in fp32 RMS (see note 3).

| Shape (regime) | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP p50 / p99 / max | ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|
| (1,1,32,64) R1 | bf16 | 0.9999985 | 0.0148 | 0.00112 | 0.00196 | 0 / 1 / 1 | 0.99905 [0.99638, 1.00183] |
| (1,1,128,512) R1 | bf16 | 0.9999985 | 0.0303 | 0.00097 | 0.00186 | 0 / 1 / 1 | 0.99934 [0.99653, 1.00206] |
| (1,1,2048,1000) R1, W non-aligned | bf16 | 0.9999985 | 0.0423 | 0.00103 | 0.00191 | 0 / 1 / 1 | 0.99920 [0.99636, 1.00190] |
| (1,1,32,16384) R2 | bf16 | 0.9999985 | 0.0403 | 0.00098 | 0.00188 | 0 / 1 / 1 | 0.99931 [0.99648, 1.00205] |
| (1,1,32,64) R1 | fp32 | 0.99999975 | 0.0170 | 0.00099 | 0.00159 | 1.6e4 / 4.0e4 / 5.2e4 | 0.99861 [0.99733, 0.99968] |
| (1,1,128,512) R1 | fp32 | 0.99999973 | 0.0186 | 0.00080 | 0.00147 | 1.4e4 / 3.7e4 / 5.5e4 | 0.99878 [0.99753, 0.99990] |
| (1,1,2048,1000) R1, W non-aligned | fp32 | 0.99999975 | 0.0253 | 0.00093 | 0.00157 | 1.6e4 / 3.9e4 / 6.2e4 | 0.99862 [0.99739, 0.99970] |
| (1,1,32,16384) R2 | fp32 | 0.99999975 | 0.0255 | 0.00080 | 0.00143 | 1.5e4 / 3.6e4 / 5.6e4 | 0.99877 [0.99755, 0.99986] |

**Assessment.**

1. **bf16 is essentially exact.** Every output is within 1 bf16 ULP of the correctly rounded result (p50 = 0), on
   both regimes, both layouts and the non-aligned shape. The R2 cross-core combine adds no measurable error.
2. **fp32 is TF32-class, not fp32-class.** The error is ~1.5e4 fp32 ULP at p50 (≈ 2⁻⁹ relative), with a systematic
   **−0.12 … −0.14 % bias** in the ratio median. **Scale-vs-precision triage: this is not a scale bug.**
   - The bias is the same at W = 64, 512, 1000 (non-aligned) and 16384, so it is not a wrong-W count or padding
     leak (that would scale with `W_pad / W`).
   - The spread (p5 to p95 ≈ 0.25 %) is wider than the bias, rel-RMS is low (0.0015), and PCC is 0.9999997.
   - Cause: the FPU source registers (`SrcA` / `SrcB`) hold at most TF32 (10-bit mantissa) and truncate fp32
     operands toward zero. x, γ, x·γ and the inverse-RMS each lose up to 2⁻¹⁰ (mean −2⁻¹¹) before each FPU
     multiply. A CPU emulation of exactly that truncation (`tf32_trunc` on x, γ, x·γ and inv-RMS) gives a ratio
     median of **0.99897**, the same sign and magnitude as measured (0.99878). The device has a few more truncation
     points (reduce-scaler multiply, Col-broadcast operand).
   - The golden fp32 gate (PCC 0.999, rel-RMS 0.02) passes with a 13× margin.
3. **RM fp32 is ~15–20 % worse in RMS than TILE fp32**, consistent with one more truncation point
   (`tilize<…>` runs at its default `Fp32Mode::Fast`, which the helper documents as lossy for fp32 versus `Fp32Mode::Lossless` / `UnpackToDestFp32`). It is within the same TF32 class.
4. **bf16 shows the same direction of bias (−0.07 … −0.10 %).** Its fp32 intermediates `cb_x_gamma` and
   `cb_inv_rms` are truncated to TF32 when unpacked for normalize. It is invisible at bf16 output resolution.

**Recommended tolerances** (Phase 0 corner, N(0,1) data):

- bf16: PCC ≥ 0.9999, rel-RMS ≤ 0.005, atol 0.05, rtol 1e-2.
- fp32: PCC ≥ 0.99999, rel-RMS ≤ 0.005, atol 0.03, rtol 5e-3.

**Do not** set fp32 tolerances at fp32-ULP scale until the TF32 operand path is replaced (see Recommendations).

## Verifier CLI Summary

Final run on the committed code: `/tmp/rms_norm_verify/golden_final/`. A compact copy is in
`verifier_report.json` in this directory.

- supported_pass: **737** (was 177 before the tagger fix)
- xfail_expected: 6180
- invalid_skipped: 33900
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**
- supported_marked_xfail: 0
- invalid_unexpected: 2. These are `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[*-bfloat8_b]`
  cells (bf8b × `w_non_aligned` × WIDTH_SHARDED). The translated suite does not apply INVALID; the cells xfail
  correctly today because bf8b is not in `SUPPORTED`. Handled in Refinement 1's notes.
- infeasible_skipped: 2. These are `99991×64` WIDTH_SHARDED (TILE and RM): the auto shard spec does not fit L1,
  which is a harness-side skip and not op-charged.
- no_axes_found: 15. These are `test_regression.py` numerics tests (small / large magnitude, uniform, positive-only,
  negative-only); **all 15 pass**.
- Acceptance: `tests/ttnn/unit_tests/operations/rms_norm/` passes **137 / 137** with `--dev` (121 acceptance and
  debug tests plus the 16 precision-baseline cases).

`xfail_expected` broken down by refusing axis value (a cell can count under several):

| Refusing value | Cells | Disposition |
|---|---|---|
| `fp32_dest_acc_en=False` | 3615 | Refinement 1 (bf16 / bf8b). `{float32, False}` stays the mandated EXCLUSION. |
| `memory_layout=BLOCK_SHARDED` | 1623 | Refinement 2 |
| `memory_layout=WIDTH_SHARDED` | 1622 | Refinement 2 |
| `memory_layout=HEIGHT_SHARDED` | 1593 | Refinement 2 |
| `dtype=bfloat8_b` | 962 | Refinement 1 |
| `gamma_dtype=bfloat8_b` | 860 | Refinement 1 |

The same 6180 cells, partitioned by the refinement that unlocks each one:

| Unlocked by | Cells |
|---|---|
| Refinement 1 only | 922 |
| Refinement 2 only | 1733 |
| Refinements 1 and 2 | 2160 |
| Permanent `{float32, False}` EXCLUSION (420 interleaved, 945 sharded) | 1365 |

That leaves zero `xfail_expected` cells without a refinement or a documented omission.

Gap coverage (`TARGET − SUPPORTED`):

- `dtype` {bf8b} → Refinement 1.
- `gamma_dtype` {bf8b} → Refinement 1.
- `fp32_dest_acc_en` {False} → Refinement 1, except `{float32, False}`, which is a permanent mandated EXCLUSION.
- `memory_layout` {HEIGHT, WIDTH, BLOCK} → Refinement 2.
- `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_layout`: no gap.

## Performance snapshot

Measurement method: device-kernel ns from the eval real-time profiler window (the same capture the golden suite
uses). Each value is the per-call mean of 2 dispatches. Config: bf16 TILE input, bf16 TILE gamma, HiFi2,
interleaved DRAM. **`fp32_dest_acc_en=True`**, because the profiles' own `False` is not supported yet.

**This is a proxy for headroom triage only. It is never an acceptance number**: the perf refinements must
re-measure at the exact profile config once Refinement 1 lands. `achievable_ns` is the unscaled reference at
1350 MHz. Idle AICLK read 800 MHz (`tt-smi`), which is not valid under-load evidence; the perf pass must scale from
profiler evidence. With AICLK ≤ 1350 MHz the scaled reference is ≥ the value shown, so these comparisons are
conservative.

| Profile (rows × W) | Regime | Cores | TILE ns / call | RM ns / call | Interleaved `achievable_ns` | Sharded `achievable_ns` |
|---|---|---|---|---|---|---|
| 32 × 1024 | R2 | 4 | 5 760 | 7 773 | 9 149 | 4 110 (WIDTH) |
| 32 × 2304 | R2 | 9 | 6 348 | 8 166 | 17 003 | 4 617 (WIDTH) |
| 32 × 5120 | R2 | 20 | 7 757 | 9 494 | 75 825 | 5 267 (WIDTH) |
| 32 × 7168 | R2 | 28 | 9 279 | 10 312 | 14 894 | 5 481 (WIDTH) |
| 640 × 6144 | R2 | 120 | 47 782 | 49 855 | 101 968 | 19 039 (BLOCK) |
| 640 × 7168 | R2 | 120 | 54 292 | 55 738 | 119 307 | 21 669 (BLOCK) |
| 7168 × 1024 | R1 | 120 | 78 991 | 80 436 | 89 392 | 18 871 (BLOCK) |
| 7168 × 2048 | R1 | 120 | 158 015 | 164 450 | 172 892 | 25 947 (BLOCK) |
| 8192 × 1024 | R1 | 120 | 90 374 | 93 259 | 96 744 | 25 640 (BLOCK) |
| 8192 × 2304 | R1 | 120 | 202 325 | 219 031 | 211 345 | — |
| 8192 × 5120 | R1 | 120 | 425 456 | 458 517 | 738 307 | — |
| 8192 × 7168 | R1 | 120 | 594 524 | 607 074 | 1 032 281 | — |

Readings that drive the perf refinements:

- **Interleaved prefill (8192 × 7168) is at the DRAM ceiling.** It moves 235 MB in + out at ≈ 395 GB/s, against
  the catalog's achieved ~415–430 GB/s on BH p150b. That is within 8 % (close-to-ceiling), so it gets **no** perf
  refinement. 8192 × 1024 runs at ≈ 371 GB/s (~13 % above the ceiling): marginal.
- **R2 regime cells have the interleaved headroom.**
  - Decode (32 rows) is fixed-cost and combine bound: 32 × 7168 moves 0.9 MB in ~9.3 µs, against a DRAM time of
    ~2 µs.
  - 640-row cells run one block per core with no read/compute overlap. 640 × 6144 moves 15.7 MB in 47.8 µs, about
    29 % above the DRAM ceiling.
- **The sharded references are 2–4× tighter** than anything an interleaved input can reach. That placement class is
  the biggest gap, so it is perf refinement #1 once Refinement 2 builds it.
- **RM narrow-W is a separate class.** 8192 × 64: RM 24.5 µs vs TILE 10.8 µs (2.3×, implementer measurement); the
  per-call tilize/untilize cost dominates.

## Recommendations

- **fp32 precision lever (no failing cell, so not queued).** To make the fp32 path fp32-class rather than TF32-class:
  route the three multiplies whose operands are fp32 (x·x, x·γ, (x·γ)·inv-RMS) through the SFPU on fp32 DEST
  (`copy_tile` with `UnpackToDestFp32` into DEST, then `mul_binary_tile`), and run `tilize` with
  `Fp32Mode::Lossless`. The cost is SFPU throughput (lower than the FPU per tile; not measured here). Only take it on if a consumer needs it;
  measure and gate it with the precision-baseline test, which would then tighten to an fp32 ratio median of 1 ± 1e-5.
- **Interleaved prefill is DRAM-saturated** (see above). Do not spend a perf phase on it.
- **Gamma multicast (design R5) and tree combine (R4)** stay deferred, for the reasons recorded in `op_design.md`.
- **L1 pressure (no OOM).** The fp32 `4096 × 8192` + fp32 RM gamma case already solves to a 1 433 600 B footprint
  against the 1 461 120 B budget (R2 is L1-forced). Sharding (Refinement 2) puts the shard itself in L1. The
  selector must subtract it, which `_l1_budget_bytes` already does by clipping at the lowest allocated L1 block.
  But once x is zero-copy, `cb_x_tiles` must also drop out of `_cb_specs`, or the same bytes are counted twice.
- **Harness.** `test_translated.py` ignores `INVALID` (the 2 `invalid_unexpected` cells), and the cartesian
  harness's "tagger wins" rule should probably be documented in `REGISTRY_MODEL.md` next to the optional-weight
  sentinel guidance. The prompt's wording ("INPUT_TAGGERS: tag_gamma_dtype …") led straight into the collapse fixed
  here.
