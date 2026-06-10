# Verification Report: groupnorm_sc_N_1_HW_C

Verified 2026-06-10. Phase-0 single-core GroupNorm over (N, 1, HW, C), three streaming
passes per (n, g) group, kernel-lib helpers end-to-end.

## Code Review

All findings were fixed in place; nothing deferred except the L1 observation (see Recommendations).

1. **Reader: per-tile NoC barrier serialized every read** (`groupnorm_sc_N_1_HW_C_reader.cpp`).
   The inner loop issued `cb_reserve_back(1) → noc_async_read_tile → noc_async_read_barrier → cb_push_back(1)`
   per tile, killing NoC pipelining. **Fixed**: batched per row chunk — reserve `Wg`, issue `Wg`
   reads, one barrier, push `Wg`. `cb_input_tiles` is sized `2*Wg` pages, so chunk batching keeps
   the double-buffered reader/compute overlap. Compute side is unaffected (reduce uses
   `WaitAndPopPerTile`, eltwise lifecycles wait per tile; cumulative push counts satisfy both).
2. **Writer: same per-tile barrier pattern** — **fixed** identically (wait `Wg`, `Wg` writes,
   one barrier, pop `Wg`); `cb_output_tiles` is also `2*Wg` pages.
3. **`affine_layout` under-claim (drift fix-in-place)**: the entry point converts gamma/beta to
   TILE host-side, so TILE-given affine tensors traverse the identical path. Probed bf16 TILE
   gamma+beta at three shapes (PCC ≥ 0.99999), then added `ttnn.TILE_LAYOUT` to
   `SUPPORTED["affine_layout"]`, re-ran the golden suite, supported_pass 180 → 300 cells.

Helper usage matches the design table 1:1 (reduce<SUM,SCALAR>, sub/square/mul convenience
wrappers, `accumulate_reduce_block` for pass-2 interleaving, `transform_in_place` for eps+rsqrt,
`eltwise_chain` BinaryFpu Row+TileOffset::Set for gamma/beta). No further fusion available:
mean/rstd/gamma/beta all need Scalar/Row broadcast, and `DestReuseBinary` is elementwise-only —
stages must round-trip CBs.

CB sync verified per group: cb_input 3·Ht·Wg push = pop; cb_output Ht·Wg; cb_mean/cb_var
push-1/pop-1; scaler + gamma/beta drained at kernel end. TensorAccessor everywhere; correct
`api/dataflow/dataflow_api.h` includes; `void kernel_main()` in all three kernels.

## Design Conformance

Matches `op_design.md` on all binding dimensions: two-pass centered-variance statistics with
streaming re-read (slab never resident), per-(n,g) group loop on 1 core, no inter-core
communication, broadcast dims (Scalar for mean/rstd, Row for gamma/beta) as designed.

## Registry Conformance

- INPUT_TAGGERS (`alignment`, `groups_alignment`, both `(inputs, axes)`), SUPPORTED, EXCLUSIONS
  (empty), validate() (per-axis then exclusions, NotImplementedError) — present and wired
  correctly. validate() is the entry point's first call. Op file declares **no** INVALID. ✓
- Auto-fix applied: `SUPPORTED["affine_layout"] += [ttnn.TILE_LAYOUT]` on XPASS-equivalent
  probe evidence (see Code Review #3).
- **INVALID audit** (feature_spec.py, 5 entries): all well-formed. Two bf8b+ROW_MAJOR entries
  (activation and affine), each single-tensor coupling — canonical entries present. Three
  no-affine canonicalization entries correctly mark exactly 5 of the 6 redundant
  (affine_dtype × affine_layout) cells, keeping (bfloat16, ROW_MAJOR) canonical. No
  cross-tensor coupling, no "not implemented yet" miscategorized as INVALID.

## Precision Baseline

bfloat16, gamma_beta, eps=1e-5, via `test_groupnorm_sc_N_1_HW_C_precision_baseline.py`:

| Shape | G | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|---|-----|-------------|--------------|------------------|
| (1,1,32,32) | 1 | 0.999997 | 0.0275 | 0.0018 | 0.0025 |
| (1,1,128,128) | 4 | 0.999995 | 0.0804 | 0.0026 | 0.0031 |
| (2,1,64,128) | 4 | 0.999994 | 0.0676 | 0.0030 | 0.0036 |
| (1,1,512,256) | 8 | 0.999992 | 0.0639 | 0.0028 | 0.0034 |

**Assessment**: relative RMS ~0.003 across all sizes — ~6× inside the bf16 tolerance band
(0.02); no drift with slab size, confirming the centered-variance accumulation is stable.
**Recommended tolerances**: PCC ≥ 0.995, rel-RMS ≤ 0.02 (current golden TOLERANCES are correct).

## Verifier CLI Summary

7236 golden cells (`verifier_report.json`):

- supported_pass: 300
- xfail_expected: 3385
- invalid_skipped: 3551
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Acceptance: 32/32 · regression: 35/35 · precision baseline: 4/4 · extended: 3/3.

## xfail_expected coverage (TARGET − SUPPORTED → queue)

| (axis, missing value) | marginal xfails | Disposition |
|---|---|---|
| dtype=FLOAT32 / BFLOAT8_B | 1474 / 737 | Refinement 1 |
| affine_dtype=FLOAT32 / BFLOAT8_B | 1340 / 670 | Refinement 1 |
| affine_layout=TILE | 0 (was 2010) | fixed at Phase 0 |
| alignment=hw_non_aligned / c_non_aligned | 220 / 605 | Refinement 2 |
| groups_alignment=non_aligned | 1815 | Refinement 3 |

Every missing pair is queued or fixed — no gaps, no documented omissions needed.

## Recommendations

- **L1 budget boundary (no failing golden cell — observation only)**: 5 stream CBs scale as
  2·Wg + gamma/beta hold 2·Wt pages. With gamma_beta affine, G=1 fits up to C=1024 (Wg=32);
  C=2048 OOMs (measured: 1.68 MB > 1.5 MB). All golden wide-C cells use G=32 (Wg=4), so
  nothing in the suite fails. If wide single-group shapes ever enter feature_spec.INPUTS,
  that refinement is `/memory-budget-metal` scope (chunk the pass-3 affine pipeline; gamma/beta
  could stream per group block instead of holding Wt).
- Refinement order: numeric config first (machinery reused by both alignment refinements),
  then edge alignment, then group straddle — the hardest INPUTS cells couple both alignment
  regimes, so Refinement 3 builds on Refinement 2's masking.
- For Refinement 1, `prepare_reduce_scaler` already pushes a bf16 scaler tile regardless of
  input dtype; intermediate stat CBs in input dtype should switch to fp32 when
  `fp32_dest_acc_en` is set.
