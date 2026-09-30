# Verification Report: mhc_pre

Hardware: Blackhole p150 (11×10 compute grid). Golden artifacts: `verifier_report.json` (this directory),
produced from `eval/eval_test_runner.sh eval/golden_tests/mhc_pre/` + `python3 -m eval.verify_supported`.

## Code Review

What was fixed:

- **DRY: CB sizes had two sources (fixed).** `_l1_bytes()` in `mhc_pre_program_descriptor.py` restated every
  CB page count by hand (`2 + 2·G + 2 + 1 + 1 + 1 + 1 + n` fp32 pages per token tile, and so on). The
  ProgramDescriptor's CB list restated them again. A block-knob turn could change one without the other, so
  the L1 fit test would drift from what actually gets allocated. Both now read one table,
  `_cb_table(bt, depth, kmax, G, y_chunk, n, tiles, dtypes)`. `_l1_bytes` is the sum over that table, and the
  selection function takes the fixed and per-block-row terms from it (the footprint is affine in
  `block_token_tiles`). The depth-1 fallback is a loop over `(X_BLOCK_DEPTH_DEFAULT, 1)` instead of a
  duplicated block. Behaviour is unchanged: 26/26 unit tests pass, and the golden L1 peak is identical
  (1,255,424 B at C=7168 fp32).
- **SUPPORTED widened on evidence (drift fix).** `weight_dtype = bfloat16` is now claimed. Every golden cell
  with a bf16 W passes (fp32 X × bf16 W: 25/25 `test_op`). CB formats and page sizes already follow the tensor
  dtypes, and every helper reconfigures per operand, so no kernel change was needed.
- **bf16 streams tried and not claimed.** I also tried `dtype = bfloat16`. The kernel runs it: bf16 X × bf16 W
  passes everywhere. But bf16 X × fp32 W fails the coefficient gate on 11 golden/loose cells plus
  `test_comb_depth_chain[bf16]` (rel-RMS 5.0–5.5e-4 vs a 5e-4 target, `severity=precision`). Per policy,
  precision failures are not hidden behind EXCLUSIONS. bf16 streams therefore stay out of SUPPORTED and move
  to Refinement 1 with the root cause below.
- Unused local `f32` removed from `create_program_descriptor`.

Reviewed and found correct (no change needed):

- **One dispatch.** The entry point is `validate()`, then output allocation, one `ProgramDescriptor`, and one
  `ttnn.generic_op`. No other device op runs.
- **Kernels.** All three use `void kernel_main()`, `api/...` include paths, and `TensorAccessor`.
- **CB push/wait balance.** Checked per CB. Pushes and pops use the nominal `block_token_tiles ·
  core_k_tiles_max` or `2·bt` page counts on both sides, so the ring stays aligned for uneven ranks and a
  ragged last block. `cb_y_out` is drained wrap-aware.
- **Helpers.** `matmul_block` (in0 retained, `num_k_blocks = 1`), `eltwise_chain` for Σx² (DEST
  WholeShape) and for the y-mix (bcast Col, PerRow DEST accumulation), `reduce<..., Accurate>`,
  `prepare_reduce_scaler`, and `mcast_pipe` Sender/ReceiverPipe for the combine broadcast. The raw-LLK pieces
  are the combine fold and the two custom SFPU block ops (coefficients, Sinkhorn). The design justifies each
  one against a concrete helper mismatch, and the scope boundary puts mechanism choice outside this review.
- **NoC assignment.** The reader uses NOC_0 and the writer NOC_1. `McastConfig(noc=NOC_1)` matches the
  writer's NoC (`WriterDataMovementConfig` defaults to NOC_1 on every arch).
- **Broadcast.** The y-mix uses `BroadcastDim::Col` on column-0-valid pre tiles, which is correct. The bias is
  written once into a coefficient-major tile and never refilled.

### Prompt rules (`eval/prompts/mhc_pre.txt ## Rules`)

| Rule | Status |
|------|--------|
| MUST: one device program per call | ✓ one `generic_op` |
| MUST NOT: limit the nC reductions to one core per token tile-row | ✓ nC split over group ranks, with an on-device combine. 110 cores busy at T=640 |
| prefer: read X once | ✓ X block resident from the projection to the y-mix |
| MUST: exact Sinkhorn order, overflow-safe softmax | ✓ row-max softmax + eps → col → (iters−1)×(row, col). **Measured in isolation** (W = 0, so logits = 30·bias; 40 random matrices): max\|comb − fp64\| = 1.9e-7, the same as torch fp32 (1.2e-7). The Sinkhorn is fp32-exact. |
| MUST: coefficients fp32 | ✓ all SFPU fp32. Every coefficient CB is Float32 / UnpackToDestFp32 |
| MUST: refuse `fp32_dest_acc_en=False` | ✓ `UnsupportedAxisValue` (SUPPORTED axis) |
| MUST: honour `math_fidelity` / `math_approx_mode`; None → `default_compute_kernel_config()` | ✓ |
| MUST NOT: specialise to one hidden size | ✓ group geometry, blocks and windows are all derived at call time from (T, C, dtypes, grid) |
| MUST: bitwise-identical repeat calls | ✓ rank-ordered fold. `test_mhc_pre_precision_baseline.py` asserts `torch.equal` over two calls on 4 shapes |
| MUST: padded rows must not contaminate | ✓ all math is row/lane-local. Ragged shapes pass |

No MUST violations. No prefer/consider advisories are left unfollowed.

### Design conformance (`op_design.md`)

- **Algorithm, pipeline and RISC ownership** match the Block schedule. The Sinkhorn is reordered after the
  y-mix (stall-shadow), as designed.
- **Work distribution** fills the machine. At T=640 every C uses 110/110 cores; T=256 uses 88. Both dataflow
  halves are batched: the reader issues one barrier per X block, and the writer issues one barrier per ≤ 8-tile
  y window.
- **Blocking-model fidelity.** `block_token_tiles`, `x_block_depth`, `y_chunk_tiles`/`y_depth`, `group_w/h`
  and `GROUP_CORES_CAP` are host parameters, derived once and passed as CT/RT args. No CB scales with a whole-op
  dimension: `cb_x_resident` / `cb_weight` scale with the rank slice `core_k_tiles_max`, which is bounded by the
  group split. The one departure from the design default is `BLOCK_TOKEN_TILES_CAP = 1` instead of the
  coarsest block that fits. It was justified by measurement (fp32; see the ledger), so it is accepted.
- **Expression.** At the measured defaults (bt = 1) every phase acts on the whole block. With bt > 1,
  `sumsq_block` does one Q tile per row through the 1-tile `cb_sq_acc`. That CB is compute→compute, one
  compute thread has no pipelining to buy, and the per-row Q is what the Accurate row collapse consumes. It is
  not flagged. It should be revisited if a perf refinement raises bt.
- **R2 (W column broadcast)** remains `deferred`. It is filed as the first perf refinement (Refinement 3).

## Registry Conformance

- `INPUT_TAGGERS` has one tagger, `alignment`, with signature `(inputs, axes)` over X dim −2. ✓
- `SUPPORTED` covers every axis the op gates on: dtype, layout, weight_dtype, fp32_dest_acc_en, alignment. ✓
- `EXCLUSIONS = []`. `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS
  (`ExcludedCell`), then structural `ValueError`s. It is called first in the entry point. ✓
- The op file declares no `INVALID`. ✓
- **Auto-fix applied:** `weight_dtype += [bfloat16]` (evidence above).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. X, W and the outputs are TILE-only,
  there is no bfloat8_b in TARGET (so the bf8b + ROW_MAJOR entry is not applicable), there is no norm weight
  (so no no-weight canonicalization), and X / W dtypes are independent tensors that no entry couples. Nothing
  to flag.

## L1 Ledger Audit

- **Currency.** 15 rows for 15 CBs. After the DRY fix every size comes from `_cb_table`, which mirrors the
  ledger row for row. The measured peak equals the closed form (1,255,424 B, C=7168 fp32).
- **Capacity vs live set.** No unjustified over-capacity:
  - `cb_gathered` is `G·2·bt` pages on every core but live only on the root. That is the stated cost of the
    uniform remote-write address, bounded by `G ≤ 32`.
  - `cb_pre_cols` is n full fp32 tiles per row, of which only column 0 is valid. That is required by the FPU
    col-broadcast.
  - `cb_y_out` is a fixed streaming window, correctly not spanning C.
  - No under-capacity: the axis accounting spans token × K for `cb_x_resident` and scales with both.
- **Page format vs DEST.** `fp32_dest_acc_en` is always on. Every Float32 page carries a DEST-produced fp32
  value, or is a dataflow-only fp32 payload (the partials, S, coefficient tiles). No 16-bit intermediate page
  truncates an fp32 accumulation. The bf16 scaler is the reduce convention.
- **Disjoint lifetime.** Every pair has a `Shares with / why not` answer: `cb_combined` is genuinely shared
  (root output and non-root landing); the ownership rule separates `cb_coef_in`/`cb_coef_out` and
  `cb_logits_coef`/`cb_comb_coef`; the bias is staged in the not-yet-pushed first X slot.
- **Bounds.** The symbol table bounds `n`, `group_cores`, `core_k_tiles_max`, `bt`, `x_block_depth` and the
  window knobs by their predicates, and the total is closed-form.
- **Data-movement budget.** Present and consistent with the code: X once, W once per group (`G_t = 10`×),
  outputs once, combine NoC ≈ 3.4 MB.
  - **Finding: the cheaper split has no positive deferral reason.** The cheapest-traffic split (R2, W column
    broadcast: −33 MB at 640×7168) is a `deferred` row. Its reason is sequencing ("validate one cross-core
    mechanism first"), not unreachability. The measurement now shows the W re-read is ~1/3 of fp32 bytes and
    ~45 % of the bf16 perf-focus bytes. **Disposition:** folded into Refinement 3 (the first perf refinement),
    not a separate entry.
- **Block-size defaults.** Interleaved across the full grid: held. The block departs from "coarsest that
  fits" by measurement only (BLOCK_TOKEN_TILES_CAP = 1, fp32 shapes). Refinement 5 re-tunes it for bf16, where
  X occupies half the L1.
- **Per-core footprint.** `x_block_depth·bt·kmax·xT + kmax·wT + bt·fT·(8 + 2G + n) + y_depth·y_chunk·yT +
  3·fT + hT`:
  - X scales with bt, depth and kmax (C / group split).
  - W scales with kmax.
  - The fp32 coefficient terms scale with bt and G.

## Precision Baseline

`ttnn/ttnn/bringup/mhc_pre/tests/unit/test_mhc_pre_precision_baseline.py` (fp32 X, tf32-rounded fp32 W,
golden reference `pytorch_mhc_pre`, seed 7). ULP is float32 ULP of the reference. Ratio = got/true over
\|ref\| > 1e-3·max.

| Shape | Out | PCC | Max Abs | Mean Abs | Rel RMS | ULP p50 / p99 | ratio median | ratio p95−p5 |
|-------|-----|-----|---------|----------|---------|---------------|--------------|--------------|
| (1,1,64,4096) | y | 0.99999987 | 6.66e-3 | 8.22e-4 | 8.64e-4 | 9514 / 288130 | 0.999323 | 4.8e-3 |
| | post | 0.99999991 | 7.37e-4 | 1.65e-4 | 4.12e-4 | 1694 / 12529 | 1.000008 | 1.1e-3 |
| | comb | 0.99999993 | 2.90e-4 | 5.60e-5 | 3.66e-4 | 2808 / 19678 | 1.000028 | 1.6e-3 |
| (1,100,4096) ragged | y | 0.99999987 | 6.68e-3 | 6.48e-4 | 8.81e-4 | 9360 / 263938 | 0.999319 | 4.1e-3 |
| | post | 0.99999991 | 8.09e-4 | 1.54e-4 | 4.17e-4 | 2198 / 13302 | 0.999972 | 1.1e-3 |
| | comb | 0.99999990 | 3.02e-4 | 6.21e-5 | 4.52e-4 | 3052 / 17445 | 1.000017 | 1.5e-3 |
| (1,1,640,7168) | y | 0.99999988 | 9.39e-3 | 8.74e-4 | 8.60e-4 | 9507 / 293063 | 0.999317 | 4.6e-3 |
| | post | 0.99999993 | 8.47e-4 | 1.50e-4 | 3.76e-4 | 1878 / 13467 | 0.999997 | 1.1e-3 |
| | comb | 0.99999991 | 3.90e-4 | 6.10e-5 | 4.31e-4 | 3056 / 18765 | 1.000016 | 1.6e-3 |
| (1,1,640,28672) | y | 0.99999988 | 9.49e-3 | 8.85e-4 | 8.64e-4 | 9501 / 289840 | 0.999316 | 4.6e-3 |
| | post | 0.99999986 | 9.40e-4 | 1.76e-4 | 5.27e-4 | 1826 / 13131 | 1.000001 | 1.0e-3 |
| | comb | 0.99999990 | 3.99e-4 | 6.18e-5 | 4.47e-4 | 3088 / 18971 | 1.000023 | 1.6e-3 |

**Assessment.** The errors are tf32-class and shape-independent. Two outputs need explanation:

- **y has a small systematic bias** (ratio median 0.99932). I triaged it as a possible scale bug and rejected
  that: the spread (4.6e-3) is about 7× the offset, so it is not a tight cluster. It matches a CPU emulation
  where the FPU reads fp32 X **truncated to 9 explicit mantissa bits** (drop 14 bits → 0.999314; tf32's
  13-bit drop would give 0.99965). This is the contract-allowed tf32-class FPU y-mix (design lamp L5).
- **post/comb error (~4e-4) is not only the X truncation.** Two device probes (probes 004–006) isolated the
  other sources:
  1. **The FPU consumes fp32 W with ~9 explicit bits, not full tf32.** For bf16 X, a tf32 W gives rel-RMS
     4.35e-4 / 5.03e-4 (post / comb). The same W pre-truncated to 9 bits, or to bf16, gives 2.5e-4 / 2.9e-4.
  2. **The FPU matmul's in-tile accumulation is not fp32-exact, even with fp32 DEST.** With ±1 X and a
     bf16-valued W, the post logits are off by 1.8e-4 even on a single 4-tile K. With a one-hot W, where every
     sum is a single product, the error is 1.8e-6. So the SFPU coefficient path and Σx² are exact, and the
     residual is the FPU's dot-product accumulation.

**Recommended tolerances (fp32 streams).** PCC ≥ 0.99999, rel-RMS < 2e-3 (y, post, comb), and \|ratio median
− 1\| < 1.5e-3 with the spread wider than 2× the offset (scale-bug guard). These are the same as the golden
`TOLERANCES[(*, float32)]`, which pass with ≥ 2× margin.

## Verifier CLI Summary (final run, `verifier_report.json`)

- supported_pass: **106**
- xfail_expected: **98**. These are exactly the `dtype = bfloat16` cells: 25 shapes × 2 weight dtypes + 48
  loose cases. All are queued in Refinement 1.
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: **1**. `test_regression.py::test_large_sinkhorn_logits[T64_nC4096]` (fp32 X, a_res = 30).
  It is labelled `numerical-bug` only because the doubly-stochastic gate raises `severity=bug`. Its metrics are
  precision-class: PCC 0.9999982, rel-RMS 1.9e-3 (< 2e-3), column sums exact (mean −9.9e-7). The failing clause
  is the worst row sum: 0.0674 vs the reference's 0.0672 + 5e-5 slack. The Sinkhorn itself is fp32-exact
  (probe above), so the gap is the logits' tf32-class noise (×30), amplified by a near-permutation Sinkhorn
  that has not converged. This is a tracked precision failure, not a structural bug, and not EXCLUDED. It moves
  to passing in Refinement 2.
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1. `test_comb_depth_chain[bf16]` self-skips while bf16 is out of SUPPORTED. That is expected,
  and it turns on in Refinement 1.

History within this verification:

- **Initial run:** 81 pass / 123 xfail / 1 fail.
- **With bf16 streams and bf16 W claimed:** 193 pass / 13 fail. The 11 bf16-X × fp32-W coefficient cells, the
  bf16 depth chain, and the fp32 regression failed.
- **Final:** bf16 W only.

## Recommendations

- **Golden-owner advisory (not edited by the verifier).**
  - `helpers.py` says `TOLERANCES` were "set from host emulation, not yet calibrated on device".
  - The precision contract assumes a tf32 W is lossless in the FPU. On BH at HiFi4 it is not: the FPU keeps
    about 9 explicit bits, and its in-tile accumulation is not fp32-exact.
  - With an FPU projection, the `("coeff", bfloat16)` rms 5e-4 gate is reachable only with the W hi/lo split
    of Refinement 1. The bf16-W floor measured today is 2.5–2.9e-4.
  - `ROW_SUM_SLACK = 5e-5` under a_res = 30 is sensitive at the 1e-4 level to any tf32-class noise in the
    logits. Even an emulated exact-tf32 projection moves the worst row by ~1e-4.
  - Please confirm these gates are intended. If they are, Refinements 1–2 are the only ways to meet them.
- **y bias for fp32 streams** (−6.8e-4 relative, from FPU truncation of X): harmless under the contract (y
  feeds a re-normalising sublayer) and within tolerance. The in-scope lever is L5 (SFPU or hi/lo y-mix), which
  Refinement 2's X split makes cheap if the owner ever gates y tighter.
- **Perf headroom (bf16 perf focus, fp32 W), measured with bf16 temporarily enabled:**

  | Shape | Measured | DRAM roofline | Ratio |
  |-------|----------|---------------|-------|
  | 640×7168 | 270.9 µs | 121.1 µs | 2.24× |
  | 640×1792 | 104.7 µs | 30.4 µs | 3.45× |
  | 1280×4096 | 264.1 µs | 133.4 µs | 1.98× |

  For comparison, fp32 640×7168 is 383 µs vs 233 µs and the composite baseline is 2396 µs. Refinements 3–5
  target these shapes.
- **L1 is not tight** for any TARGET shape: the fp32 C=7168 peak is 1.26 MB of 1.47 MB usable. bf16 halves the
  X term, which is the room Refinement 5's block/depth co-tune spends.
