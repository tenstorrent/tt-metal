# Verification Report: rms_norm

Device: Blackhole p150 (12×10 = 120-core compute grid). Date: 2026-11-12.
Golden run artifacts: `verifier_report.json` (trimmed copy — summary + non-bulk categories; the full 25 MB report is in the results dir `/tmp/rms_norm_verify/golden2/`).

## Code Review

**Fixed in place**
- **DRY — `n_sq` (sumsq tiles per row) derived from the wrong source on the host.** `_Geometry` computed
  `n_full` / `n_sq` from `tensor_w_tiles`, while the compute kernel derives the same quantity from the
  `core_w_tiles` knob (CT arg). They agree only while R1 pins `core_w_tiles = tensor_w_tiles`; the first
  W-split (Refinement 4) would have sized `cb_sumsq` from the full row while the kernel pushed per-slice
  counts. Replaced with one host helper `_n_sq(g, core_w_tiles)` used by both the footprint function and
  the `cb_sumsq` allocation (`rms_norm_program_descriptor.py`).
- **L1 ledger currency** (`l1_ledger.md`): removed the stale `s_m` (bf16 mask) symbol — the mask CB is
  Float32 (documented implementer deviation); replaced the hand-estimated worked examples (one of which
  still described the dropped `cb_sumsq`↔`cb_normed` alias, the other a `c = 8, K = 32` choice the
  selector does not make) with measured `_select_blocks` output for six cells.

**Reviewed, no change needed**
- Kernels: `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` everywhere (no
  `InterleavedAddrGen`). CB sync balanced on every CB (cumulative `cb_x` waits vs one block pop; per-row
  `cb_sumsq` push vs BulkWaitBulkPop reduce; `cb_gamma` Upfront/AtEnd per chunk; mask push 1 / pop 1).
- Helpers: tilize/untilize, `eltwise_chain` (`BinaryFpu` Mul with Col/Row broadcast, packer L1
  accumulation), `reduce<SUM, REDUCE_ROW, AccumulateViaAdd>` with the masked partial fold,
  `prepare_reduce_mask`, `read_sticks_for_tilize` / `write_sticks_after_untilize`. The raw uses (post-op
  SFPU calls inside the reduce callback, caller-managed `cb_x`/`cb_rstd` windows, RM tail chunk, gamma
  row-0 relayout, TILE page loops) each carry a stated reason; none has a helper that expresses it.
- Broadcasts: rstd is Col-broadcast (REDUCE_ROW output, col 0 valid), gamma Row-broadcast (row 0 valid);
  no CB is filled with replicated data.
- Prompt rules: `eval/prompts/rms_norm.txt` has no `## Rules` section; the stock constraints hold — no
  host-side `to_layout`/`tilize`/`pad`/`slice`, one `ttnn.generic_op` dispatch, ValueError for rank < 2 and
  gamma width mismatch, `default_compute_kernel_config()` is the only default, config passed as-is.

**Design conformance** (`op_design.md`)
- Algorithm: one-pass resident R1 — Σx² by packer L1 accumulation + masked tail fold, rstd by reduce
  post-op (·1/W with logical W, +eps, rsqrt), normalize/scale/untilize per W chunk. Matches.
- Work distribution: `split_work_to_cores(grid, Rt, row_wise=True)` over the full compute grid. Golden
  `device_num_cores` = `min(Rt, 120)` on every passing cell (e.g. 2048×256 → 64, 4×8×47×256 → 64,
  1024×1024 → 32). Few-row shapes (decode 32×W) use `Rt` cores — the design's documented R1 occupancy
  limit, deferred to R3 (Refinement 4/5), not a contract violation.
- Both dataflow halves are batched: reader one barrier per `c`-tile row-chunk (TILE) / per 32-stick
  row-chunk (RM); writer the same; streaming CBs at depth 2.
- Blocking-model fidelity: `block_rows`, `w_chunk_tiles`, `num_w_chunks`, `core_w_tiles`, `w_tile_start`,
  grid, `STREAM_DEPTH` are each defined once on the host and passed as CT/RT args; every CB page count
  and loop bound derives from them (one `blocking_ct` list feeds all three kernels; CB indices passed as
  CT args). `cb_x` is `r·Wp` — sized by block knobs within the proven R1 residency predicate, with the
  `NotImplementedError` fallback naming R3; not a collapsed knob. Expression: block-scoped phases; the
  per-row-chunk handshakes are the design's planned synchronization grouping (they let sumsq run under
  the read), not per-tile retirement.
- Deviations from the design, both justified and recorded in the ledger: (1) `cb_sumsq` is not aliased
  with `cb_normed` (indexed reduce window would straddle `fifo_limit`); (2) the reduce mask is Float32
  (the helper's partial fold unpacks the mask as srcB in `cb_sumsq`'s format).

## Registry Conformance

- `INPUT_TAGGERS` (`alignment` three-valued, `rank`) — both `(inputs, axes)` signatures. Gamma axes are
  derived in `validate()` with the `"none"` sentinel, matching `axes.classify_call`.
- `SUPPORTED` covers every gated axis: dtype, fp32_dest_acc_en, layout, alignment, rank, gamma_mode,
  gamma_dtype, gamma_layout, memory_layout.
- `EXCLUSIONS`: `{dtype: float32, fp32_dest_acc_en: False}` (precision convention).
- `validate()`: structural ValueErrors → SUPPORTED per axis (`UnsupportedAxisValue`) → EXCLUSIONS
  (`ExcludedCell`) → output memory_config interleaved check. Called first in `rms_norm()`.
- Op file declares no `INVALID`. No XPASS drift → no SUPPORTED auto-fixes needed.

**INVALID audit** (`eval/golden_tests/rms_norm/feature_spec.py`, not edited here):
- Canonical `bf8b + ROW_MAJOR` entry present for the activation (and for gamma). No-gamma
  canonicalization (`"none"` sentinel both ways) present and correct.
- No cross-tensor-axis couplings among the structural entries.
- **Flag — "author-scoped exclusions" in INVALID violate the universe-must-change rule**:
  `{dtype: bf8b, alignment: w_non_aligned|h_non_aligned}` and `{layout: ROW_MAJOR, memory_layout:
  *_SHARDED, gamma_layout: TILE}` are explicitly "for now" (the file says so). They are op-side
  EXCLUSIONS by definition; parked in INVALID they are removed from the refinement backlog. The second
  family also couples axes of two tensors (activation layout × gamma layout). Recommend the user move
  them out of INVALID (via `/golden-tests`) when the op is ready to grow there.
- `invalid_unexpected = 2`: `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width
  [w72_c2_nonaligned-bfloat8_b | w200_c3_nonaligned-bfloat8_b]` exercise the bf8b + w_non_aligned INVALID
  cell (they xfail today on `gamma_layout=TILE` / WIDTH_SHARDED). A consequence of the author-scoped
  INVALID entry above — the reference suite does test those cells.

## L1 Ledger Audit

- **Currency**: 9 CB rows ↔ 9 CB slots in `rms_norm_program_descriptor.py`; each row's capacity
  expression equals the allocation (`cb_x r·Wp`, `cb_x_rm/cb_gamma/cb_out/cb_out_rm 2c`, `cb_sumsq r·n_sq`,
  `cb_rstd r`, `cb_normed r·c`, mask 1). `_footprint()` mirrors the ledger's `F(r, K)` term by term.
  Stale symbol + examples fixed (above).
- **Capacity vs live set**: no "over" findings — every full-block CB (`cb_sumsq`, `cb_rstd`, `cb_normed`)
  holds a live set that is produced in full before its single consumer (sequential helpers on one
  thread); streaming CBs are 2× live for double buffering. No "under": `cb_x` spans rows and W and scales
  with both; `cb_normed` spans rows × chunk and scales with `r·c`.
- **Page format vs DEST width**: Phase 0 gates `fp32_dest_acc_en=True`, so Float32 intermediates are
  correct (no "under"). When Refinement 2 adds `fp32_dest_acc_en=False`, the Float32 `cb_sumsq`/`cb_rstd`/
  `cb_normed`/mask pages become "over" — folded into Refinement 2's verifier notes.
- **Disjoint lifetimes**: `cb_sumsq` ↔ `cb_normed` are disjoint and unshared, with a concrete mechanism
  reason (indexed-window ring wrap). `cb_x_rm` ↔ `cb_out_rm` concurrent (pipelining). Clean.
- **Bounds**: every symbol has a bound + predicate (`r ≤ rows_cap`, `min(Wc,8) ≤ c ≤ Wc`, `Wc` bounded by
  the R1 legality predicate with a `NotImplementedError` fallback). Closed form.
- **Data-movement budget**: present and consistent with the code — x 1 crossing, out 1, gamma once per
  block per core. Cheapest-traffic split (rows, R1) is the one implemented; R4 (gamma mcast) and R3
  (W-split) are `deferred` regime rows with positive reasons (≤ 1/(32·r) gamma saving; R3 equal DRAM
  traffic, gain is occupancy/residency → now Refinements 4/5).
- **Block-size defaults**: held — rows spread across the full grid, then the coarsest block that fits
  (`r = rows_cap` whenever `F(rows_cap, K_floor) ≤ B`; measured `r = 3 = rows_cap` on 8192×{1024,4096},
  `r = 2 < rows_cap` only on 8192×7168 where L1 binds).
- **Per-core footprint**: `F(r,K) = r·Wp·s_x + [rm]·2c·s_x + [p]·s_f + r·s_f + r·n_sq·s_f +
  ([g]∨[rm])·r·c·s_f + [g]·2c·s_g + 2c·s_o`. Scales with `r` (cb_x, cb_rstd, cb_sumsq, cb_normed), `W`
  (cb_x via `Wp`), `c` (streams, cb_normed). Budget `B = 1 427 520 B` on this board.

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` — gamma on (same dtype),
HiFi4, `fp32_dest_acc_en=True`, DRAM interleaved; errors vs the fp32 reference rounded to the output dtype.
TILE and ROW_MAJOR give identical numbers for bf16 (shown once); fp32 RM is listed separately.

| Shape | dtype / layout | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 | got/true median [p5, p95] |
|-------|----------------|-----|-------------|--------------|------------------|---------|---------------------------|
| (1,1,32,64) | bf16 | 0.9999982 | 0.0148 | 0.00113 | 0.00207 | 1 | 0.99902 [0.99635, 1.00177] |
| (1,1,100,1000) | bf16 | 0.9999981 | 0.0386 | 0.00106 | 0.00211 | 1 | 0.99917 [0.99643, 1.00194] |
| (2,1,256,4096) | bf16 | 0.9999979 | 0.0440 | 0.00102 | 0.00220 | 1 | 0.99915 [0.99639, 1.00194] |
| (1,1,2048,8192) | bf16 | 0.9999980 | 0.0895 | 0.00102 | 0.00217 | 1 | 0.99916 [0.99635, 1.00192] |
| (1,1,32,64) | fp32 TILE | 0.9999997 | 0.0192 | 0.00099 | 0.00160 | 42 409 | 0.99861 [0.99728, 0.99968] |
| (1,1,100,1000) | fp32 TILE | 0.9999997 | 0.0260 | 0.00096 | 0.00160 | 43 854 | 0.99862 [0.99730, 0.99973] |
| (2,1,256,4096) | fp32 TILE | 0.9999997 | 0.0367 | 0.00090 | 0.00158 | 43 011 | 0.99864 [0.99733, 0.99973] |
| (1,1,2048,8192) | fp32 TILE | 0.9999997 | 0.0389 | 0.00089 | 0.00155 | 42 412 | 0.99865 [0.99740, 0.99974] |
| (1,1,2048,8192) | fp32 RM | 0.9999997 | 0.0413 | 0.00111 | 0.00188 | 46 968 | 0.99830 [0.99701, 0.99945] |

Golden suite (436 passing cells): fp32 min PCC 0.9999997, max rel-RMS 0.0020; bf16 min PCC 0.99997,
max rel-RMS 0.0132 (no-gamma bf16 TILE 1×24×4096).

**Assessment**: bf16 is at the output-rounding floor (ULP p99 = 1). fp32 carries a small systematic
negative bias (median ratio 0.9986 with gamma, p95 < 1.0). Scale-vs-precision triage, done with a probe:
no-gamma random fp32 → 0.99968; bf16-exact x → 0.99987; constant rows of 1.5 → exactly 0.999756 for
every element (the tf32 truncation of `rstd = 2/3`). The bias is the **FPU's tf32 operand truncation**: each fp32 operand that
passes through srcA/srcB (rstd, the Float32 `cb_normed`, fp32 gamma) loses ~2.5–5·10⁻⁴ toward zero.
That is precision, not a structural/scale bug: the bias is 0.14%, far inside tolerance, and rel-RMS ≈ 0.0016.
The fp32 path's ULP p99 (~43 000 fp32 ULP ≈ 0.5% relative) reflects it.

**Recommended tolerances**: defaults hold — `helpers.TOLERANCES` (fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02,
bf16 0.995 / 0.04) leave > 10× margin.

## Verifier CLI Summary

- supported_pass: 436
- xfail_expected: 6481 (all mapped to Refinements 1, 2, 4 — see below)
- invalid_skipped: 33900
- invalid_unexpected: 2 (translated tests on an author-scoped INVALID cell — see INVALID audit)
- infeasible_skipped: 2 (99991×64 WIDTH_SHARDED shard geometry, harness-side)
- no_axes_found: 15 (`test_regression.py`, all passed)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

**TARGET − SUPPORTED coverage**

| Axis | Missing value | Covered by |
|------|---------------|------------|
| gamma_layout | TILE | Refinement 1 |
| memory_layout | HEIGHT_SHARDED | Refinement 1 (local, zero-copy) |
| fp32_dest_acc_en | False | Refinement 2 (`{float32, False}` stays an EXCLUSION) |
| dtype | bfloat8_b | Refinement 2 (RM and non-aligned cells are INVALID) |
| gamma_dtype | bfloat8_b | Refinement 2 (after Refinement 1's TILE gamma) |
| memory_layout | WIDTH_SHARDED, BLOCK_SHARDED | Refinement 4 (cross-core W-split, R3) |

## Recommendations

- **Queue shape**: generality R1 (TILE gamma + HEIGHT_SHARDED) → R2 (numeric) land the full config of the
  `_PERF_BASE` cases (bf16, TILE gamma, `fp32_dest_acc_en=False`, HiFi2) before perf R3 optimizes them.
  R4 (cross-core W-split + WIDTH/BLOCK) is the one scheme-change and the hardest. After it, no generality
  candidates remain, so R5/R6 are perf (decode/few-row on the W-split; block-sharded prefill).
- **Phase 0 perf vs the references** (implementer profile, HiFi4/fp32-DEST, not the perf cases' exact
  config): 8192×1024 TILE 88.0 µs vs interleaved reference 96.7 µs at 1350 MHz → already competitive on
  DRAM-bound prefill. Decode 32×4096 at 33.9 µs on 1 core is the clear gap (references 9–15 µs interleaved,
  4–5.5 µs width-sharded) → Refinements 4/5.
- **fp32 precision lever (no failing cell, so not queued)**: the tf32 truncation bias could be removed
  for fp32 inputs by doing the two broadcast multiplies on the SFPU in fp32 (`UnpackToDestFp32` + SFPU
  binary mul) instead of the FPU. That costs throughput; take it only if an fp32 consumer needs better
  than ~5·10⁻⁴ relative bias.
- **R3 hand-off note**: `cb_sumsq`/`cb_rstd` per-core W slices will need per-core `core_w_tiles` and partial
  flags (currently CT) — called out in Refinement 4.
- **L1 pressure (no OOM)**: the widest R1 cells sit at 98% of `B` (fp32 RM 32×8192: 1 404 928 / 1 427 520 B
  with `c = 11`); any new CB on the R1 path (e.g. a TILE-gamma staging CB, a deeper `x_block_depth`) must
  go through `_footprint()` or these cells will OOM — the selector already shrinks `c`, so keep every
  new term inside `F(r, K)`.
- Infrastructure: the up-front precompile pass fails on this host (remote JIT farm compiler mismatch,
  `__xtt_vector8`) and falls back to inline compile — results unaffected, just slower.
