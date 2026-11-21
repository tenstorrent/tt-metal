# Verification Report: scaled_dot_product_attention

Target: Quasar (`quasar`) on the ttsim simulator. Phase 0 as built by the planner and implementer (commits
`3d6ff0d`..`e97f9fc`). Specification: `eval/specs/scaled_dot_product_attention/spec.py`. The specification lint
(`python3 -m eval.validate_contract scaled_dot_product_attention --phase0 --text`) passes with 0 errors; its 6
UNRESOLVED items belong to the specification (open items, no float32 numeric-list entry), none to the op.

## Code Review

Every item below was fixed in place.

1. **DRY: the fit predicate restated every ring capacity** (`scaled_dot_product_attention_program_artifacts.py`).
   The ring capacities were written twice: once in `SdpaPlan.ring_entries()`, which builds the
   `DataflowBufferSpec`s, and again as separate closed forms (`_fixed_bytes`, `_kc_fit`, `_qc_ring_caps_ok`)
   that chose `qc` / `kc`. A knob or ring change in one place would have silently desynchronized the fit from
   what is allocated. Now a module-level `ring_entries(R, C, qc, kc, Dt, causal, dtypes…)` is the one
   statement. The specs, `l1_bytes()` and the fit (`_fits`: every ring ≤ `DFB_MAX_RING_BYTES` and the sum
   ≤ budget; `_kc_fit`: bisect the coarsest `kc ≤ Skt` that fits) all evaluate it. The old and new solves pick
   the same `(qc, kc)`, and refuse the same calls, on all 322,560 cases of a sweep over B, Hq, S, D, four thread
   plans, causal, both DEST widths, both dtypes, knob overrides, cluster counts and budgets. The separate
   kc-independent ring-cap check fell away: `_kc_fit` returns 0 when the Q or O ring already breaks a limit.
2. **DRY: the constants ring's entry order lived in three places.** It was dead `SCALER_TILE` / `DIAG_MASK_TILE`
   constants in Python (never read), a second copy in the compute kernel, and the reader's push order implied by
   neither. Both indices now live once in `kernels/sdpa_geometry.hpp`, which the reader, compute and writer
   kernels all include. The reader `static_assert`s that its pushes follow that order.
3. **Stale L1 ledger** (`l1_ledger.md`). The worked footprint table predated the implementation. It used the
   design's 3-tile bfloat16 constants ring, not the built 2-tile Float32 one, and it counted one K/V ring where
   the expression has two (e.g. `(1,1,128,64)` was 352,256 B; it is 409,600 / 425,984 B full / causal). I
   regenerated it from `ring_entries()` for every first-phase shape and noted the refused `(2,1,32,4096)`. I
   also recorded that the fit now evaluates the ring table, and that `cb_rescale`'s live set is 0 in the
   resident regime.

Not changed, deliberately (mechanism is the implementer's, and none of these has an observable cost on the
Phase 0 shapes):
- `row_max_block` / `row_sum_block` issue `reduce_init` + `pack_init` once per query tile-row, because the
  chunk > 0 fold interleaves copies and SFPU ops in each row's DEST section. At `qc = 1` (every Phase 0 shape)
  that is once per block. It costs only in the streamed regime at `qc > 1`, which Refinement 4 targets.
- `rescale_accum_block` is a separate pass over O, as the design's Block schedule specifies; fusing it into
  `pv_block` is the design's own perf lamp (Refinement 4).
- `invalidate_alu_formats()` before each FPU op init clears the math thread's ALU-format latch. It
  is correct and documented.

### Specification rules

| Rule | Status |
|------|--------|
| R1 mask | met: diagonal mask tile + `fill_tile_bitcast(-inf)` on tiles past the diagonal (their matmul skipped); cells pass causal and not |
| R2 heads | met: `q_head = c·g + u % g`, K/V of head `c` shared through the ALL ring |
| R3 scale | met: `mul_unary_tile(scale_bits)` in DEST **before** the mask; `None` → `1/sqrt(D)` of Q's logical D; INPUTS scales 0.25 / 0.05 pass |
| R4 float32 + 16-bit DEST | met: refused in `validate()` in every phase (currently also outside SUPPORTED) |
| R5 default factory | met: `default_compute_kernel_config()` is the one source; `_resolve_compute_config` goes through it |
| R6 fidelity / approx not gated | met: `validate()` reads neither; `test_traced_call_arguments` runs LoFi–HiFi4 with `math_approx_mode=True` |
| R7 device config accepted | met: fields read as attributes, a `ComputeConfigDescriptor` rebuilt; tested with `WormholeComputeKernelConfig` |
| R8 program_config | met: read by nothing; tested with the traced `SDPAProgramConfig` |
| R9 placement axes | met: `eval.memory.memory_axes(input_tensor_q)`; L1 queries refused by axis today (they work: Refinement 2) |
| R10 K/V placement | met: DRAM-interleaved, TILE, Q's dtype, else `NotImplementedError` naming the parameter |
| R11 output in DRAM | met: `ttnn.DRAM_MEMORY_CONFIG` in the output spec; the spec's check hook passes in every cell |
| R12 overflow-safe softmax | met: `exp(S − m)` with `m` the row max of the scaled, masked scores (online form) |
| R13 key padding | **not yet applicable** (h_non_aligned is outside SUPPORTED). It re-arms in Refinement 1: the non-causal ragged cells fail today without a padding mask (measured below) |
| R14 refusal | met: every non-PHASE0 cell refused (210 xfail_expected, 0 wrong mode) |

GUIDANCE advisories: none unfollowed. The op uses the online softmax, a non-tensor mask with skipped
fully-masked blocks, K/V read once per group, and a compute descriptor built from the three fields. The
exponential is always exact even under `math_approx_mode=True` / `exp_approx_mode`, which R8 allows.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local tagger.
- `SUPPORTED` equals `PHASE0` axis for axis (the lint checks it); every gated axis is present.
- `EXCLUSIONS = []`.
- `validate()` runs REJECTIONS X1–X6 (`ValueError`), then R10 on K/V, then R4, then SUPPORTED per axis
  (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
- The public entry point calls `run_attention`, whose first statement is `validate()`.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED: no XPASS drift. SUPPORTED stays at PHASE0 even though most TARGET values already
  work (probe below), because the first phase must equal PHASE0. Each such value is a queued refinement.
- INVALID audit (specification): `INVALID = []`. It needs no bf8b+ROW_MAJOR entry: TARGET has no ROW_MAJOR, and
  Quasar has no `bfloat8_b`, whose cells skip as unbuildable on the family. It is not a norm-like op, so it needs
  no no-weight canonicalization. Nothing to flag.

## Design Conformance

- **Algorithm**: FlashAttention online softmax as designed. The resident and streamed regimes run the same
  kernels, and the S·S scores are never materialized.
- **Pipeline / RISC ownership**: as designed. One reader kernel at R = 4 threads (Q STRIDED, K/V ALL with R
  producer regions, explicit sync, trid 0). One compute kernel at C = 4 NEOs. One writer kernel at W = 2
  threads (implicit sync, tag handed back).
- **Work distribution (performance conformance)**: the independent axes are fully spread. Every query
  tile-row of every head is its own NEO unit, units are dealt C per cluster round, and whole rounds go to
  clusters with `row_wise=True`. Clusters engaged on the first-phase shapes: 1, 1, 1, 4, 6, 2, 8; the idle rest
  is the deferred key-split regime (Refinement 5). Both dataflow halves are batched: Q, K and V each take one
  reserve, one barrier and one push per block. The writer uses implicit sync, one tagged write per tile, which
  is the ring protocol for a NoC-drained ring and not a per-tile barrier.
- **Thread plan**: `THREAD_PLAN = (4, 4, 2)`, one host constant; its counts reach all three kernels as CT args
  (`reader_threads`, `compute_threads`, `writer_threads`). Rings are sized per NEO (STRIDED) or per producer
  region (ALL). Reader rings are opted out of implicit sync and every transfer is tagged; the writer's ring
  stays implicit. The tests are present and pass:
  - `test_each_neo_computes_its_own_unit` (`tag_threads`), on a shape with ≥ C units and on a padded one;
  - bit-identical results across 5 other plans;
  - `test_one_device_session` (a tilize program, the op, `ttnn.to_layout`).

  The design states the op as compute-bound, the page's reason for (4, 4, 2). Backend: ttsim has no timing
  model and `pipeline_logs/model_cycles.jsonl` does not exist, so the plan is not measured against its
  alternatives here. Finding for Refinement 1: C = 4 pads NEOs on shapes with `units_per_head < 4`, and the
  target page says to engage fewer threads there.
- **Blocking-model fidelity**: `q_chunk_tiles`, `k_chunk_tiles`, `d_block_tiles`, `Q_BUFFER_DEPTH`,
  `KV_BLOCKS_PER_READER` and `OUT_BUFFER_DEPTH` are host parameters, each defined once and passed as CT args.
  The DRY breaks found are fixed (Code Review 1–2). No CB scales unconditionally with a whole-op dimension: the
  K/V ring is `R·kc·Dt`, with `kc = Skt` only when the fit admits it, which is the sanctioned predicate-guarded
  resident regime with the streamed fallback. `d_block_tiles` is pinned to `Dt`, as the design states for
  Phase 0 (`wide_head_dim` deferred). `qc = 1` on the first-phase shapes is the occupancy-first derivation
  (one unit per NEO), not a half-turned knob.
- **Expression**: the reader handshakes per block, not per tile. Compute holds each block's live set (Q for
  the round; K, S, P and V per chunk), with DEST sections only inside a block operation. The writer's per-tile
  implicit write is the ring protocol, not a sync boundary per unit.
- **Deviation from the design, accepted**: the design's third constant tile (the full mask) is replaced by
  `fill_tile_bitcast(-inf)` in DEST with the matmul skipped. This means fewer constants and less work.

## L1 Ledger Audit

- **Currency**: all 11 rings in the code have rows, and every row is a live ring; capacity expressions match
  `ring_entries()`. The worked table was stale and is now regenerated from the code (Code Review 3).
- **Capacity vs live set**:
  - Q ring = 2× live block: the `Q_BUFFER_DEPTH` double buffer, justified.
  - K/V rings = R × live block: the ALL ring's R producer regions (prefetch depth R), a mechanism requirement.
  - `cb_rescale`: live set 0 in the resident regime, capacity `C·qc` (16,384 B at qc = 1). Accepted, since both
    regimes bind the same rings; noted in the ledger.
  - No collapsed extent: S and P span `qc × kc`, O spans `qc × Dt`, and both scale with their knobs.
- **Page format vs DEST width**: internal rings are Float32 with `fp32_dest_acc_en` on, which is correct. The
  constants ring is Float32 for a stated mechanism reason (the
  row-max reduce moves DEST into SrcB). Q/K/V/out are the input and output dtypes (the ABI). With the 16-bit
  accumulator all internal rings follow it to bfloat16 (`acc_dtype`), so no under-width page exists.
- **Disjoint lifetimes**: every pair has a recorded reason (S and P are concurrent in `exp_block`; m, α and l
  are concurrent in `row_sum_block`; O and out are concurrent in `normalize_block`).
- **Bounds**: every symbol has a bound and the predicate that sets it; the total is closed-form in
  `(qc, kc, Dt, R, C)`.
- **Block-size defaults**: held. The independent q axis is spread first (occupancy), then the coarsest `kc`
  that fits; `kc = Skt` (resident) on every first-phase shape.
- **Per-core footprint**:
  `L1 = C·Q_BUFFER_DEPTH·qc·Dt·T_in + 2·R·KV_BLOCKS_PER_READER·kc·Dt·T_in + C·CONST_TILES·T_acc + 2·C·qc·kc·T_acc + 3·C·qc·T_acc + C·qc·Dt·T_acc + C·OUT_BUFFER_DEPTH·T_out`.
  The `qc` and `Dt` terms scale with the query block, the `kc·Dt` terms with the key chunk and R, and the `qc·kc`
  terms with the score block. The largest first-phase shape is 1,212,416 B of 3,963,328 (30.6 %).
- **Data-movement budget**: present and consistent with the code. K/V are counted once per round
  (`rounds_per_head` per head), and the causal shrink to the round's diagonal is counted. The cheapest-traffic
  split (read-once K/V + multicast) is a deferred regime row with its delta, and it is now queued (Refinement 3).
  No ledger finding needed a refinement of its own.

## Precision Baseline

`tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_precision_baseline.py`:
bfloat16 inputs, default config (HiFi4, `fp32_dest_acc_en=True`), standard-normal inputs (seed 42), against
the specification's float64 `reference`. ULP is the bfloat16 spacing at the reference value. The ratio is
actual / expected over the elements with |expected| ≥ 1e-3.

| Shape (q + k/v) | causal | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ULP p99 | ratio median [p5, p95] |
|-----------------|--------|-----|-------------|--------------|---------|---------|-------------------------|
| 1x1x32x32 + 1x1x32x32 | no | 0.9999985 | 0.0024 | 0.00031 | 0.00182 | 4.6 | 0.9994 [0.9959, 1.003] |
| 1x1x32x32 + 1x1x32x32 | yes | 0.9999988 | 0.0046 | 0.00045 | 0.00163 | 2.4 | 0.9997 [0.9964, 1.003] |
| 1x1x128x64 + 1x1x128x64 | no | 0.9999984 | 0.0019 | 0.00019 | 0.00187 | 6.4 | 0.9994 [0.9957, 1.003] |
| 1x1x128x64 + 1x1x128x64 | yes | 0.9999986 | 0.0085 | 0.00030 | 0.00171 | 4.6 | 0.9996 [0.9959, 1.003] |
| 1x2x64x128 + 1x1x64x128 | no | 0.9999984 | 0.0040 | 0.00025 | 0.00189 | 5.3 | 0.9995 [0.9957, 1.003] |
| 1x2x64x128 + 1x1x64x128 | yes | 0.9999986 | 0.0052 | 0.00038 | 0.00171 | 4.0 | 0.9996 [0.9961, 1.003] |
| 1x8x96x64 + 1x2x96x64 | no | 0.9999983 | 0.0077 | 0.00050 | 0.00197 | 7.9 | 0.9994 [0.9944, 1.004] |
| 1x8x96x64 + 1x2x96x64 | yes | 0.9999985 | 0.0107 | 0.00059 | 0.00182 | 6.4 | 0.9995 [0.9953, 1.004] |
| 1x2x512x64 + 1x1x512x64 | no | 0.9999984 | 0.0016 | 0.00010 | 0.00192 | 6.2 | 0.9994 [0.9954, 1.003] |
| 1x2x512x64 + 1x1x512x64 | yes | 0.9999986 | 0.0070 | 0.00018 | 0.00175 | 5.6 | 0.9995 [0.9956, 1.003] |

**Assessment**: the op is at the bfloat16 output-rounding floor. The specification measures rounding the
float64 result once to bfloat16 at relative RMS ≈ 0.0016, and the op is at 0.0016–0.0020, independent of
sequence length (32–512 tokens) and of the mask.

The ratio's median sits 0.03–0.06 % below 1. That is a sub-ULP downward bias (one bfloat16 ULP is 0.4–0.8 %
of the value), consistent with the FPU reading the Float32 S, P and O rings through its 19-bit (tf32) sources
and with the reduce's tf32 accumulation. It is not a scale bug: the p5–p95 spread
(±0.4 %) is ordinary rounding noise around it, and PCC and RMS move together.

The large ULP maxima (up to ~2·10⁴) are on near-zero reference elements, where a bfloat16 ULP is tiny; the max
absolute error stays ≤ 0.011. Every gate (PCC ≥ 0.999, rel-RMS ≤ 0.05) holds with more than 20× margin on RMS.
The golden cells measured the same: PCC ≥ 0.999998, rel-RMS ≈ 0.002.

**Recommended tolerances**: defaults hold. The specification's rows are (bfloat16, True) 0.999 / 0.05,
(bfloat16, False) 0.99 / 0.13 and (float32, True) 0.999 / 0.01. The verifier probe measured the not-yet-
supported corners well inside them: (bfloat16, False) rel-RMS 0.005–0.011, (float32, True) rel-RMS 0.001–0.002.

## Verifier CLI Summary

`golden_results/verifier_report.json` (golden run after the fixes; the run before them was identical):
- supported_pass: 18 (14 cells, 4 numeric regression cases)
- xfail_expected: 210
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 7 (6 rejections X1–X6, all passing; the empty `LOOSE_CASES` placeholder, skipped)
- Queue coverage (`--queue op_requirements.md`): 5 values in TARGET − SUPPORTED, 0 gaps.

Unit tests (`scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/scaled_dot_product_attention/`):
72 passed. These are acceptance, regime-pinned, thread-plan, one-session, traced-argument, two-chunk debug and
the new precision baseline.

### What the out-of-phase cells do today (verifier probe, SUPPORTED widened in-process, not committed)

| Value | Cells tried | Result |
|-------|-------------|--------|
| `batch = batched` | (2,2,64,64), (4,4,32,32), causal and not | pass, PCC ≥ 0.999998 |
| `batch = batched` | (2,1,32,4096) | refused by the fit at plan (4,4,2) (rings > 1 MiB). **Passes at plan (1,1,1)**, PCC 0.999998 / 0.999999 |
| `alignment = h_non_aligned`, causal | all four ragged entries | pass (the diagonal hides every padding key from every real row) |
| `alignment = h_non_aligned`, non-causal | all four ragged entries | **fail, numerical-precision signature** (rel-RMS 0.067–0.36, PCC 0.991–0.99996): no key-padding mask, rule R13 |
| `memory_config = L1` | first-phase shapes, causal and not | pass, identical to DRAM |
| `fp32_dest_acc_en = False` (bfloat16) | first-phase shapes | pass, PCC ≥ 0.99994, rel-RMS 0.005–0.011 |
| `dtype = float32` (fp32 acc) | first-phase shapes | pass, PCC ≥ 0.999999, rel-RMS 0.001–0.002 |

The non-causal ragged failure is not the scale-bug signature (high PCC, high RMS, ratio tightly off 1). The
error is the padding keys' weight, exp(0 − m) per padding column, spread across rows. It is a missing mask (a
structural gap), and Refinement 1 fixes it in the kernel.

## Recommendations

- **Queue** (`op_requirements.md`): two generality refinements cover all five TARGET − SUPPORTED values, then
  three structural-count perf refinements.
  - R1 (batched + h_non_aligned: key-padding mask, per-call C from the fit and the unit count) comes first,
    because it changes kernel code and the plan.
  - R2 (float32 + 16-bit DEST + Q in L1, `/numeric-formats-metal`) is mostly declaration plus tests.
  - R3 (K/V read-once + multicast, T3) is the cheapest-traffic split.
  - R4 (fused rescale, hoisted inits in the streamed regime, T2).
  - R5 (key-axis split across idle clusters, T3).

  There are only two generality refinements, not three to five, because the five missing values are mostly
  free today. Splitting them further would make refinements far smaller than the build.
- **Perf, unconfirmed in time**: ttsim has no clock. Every perf gate is a structural count, and no simulator
  timing is reported as a speedup.
- **Padded NEOs**: with C fixed at 4, shapes with `units_per_head < 4` recompute NEO 0's unit on the spare NEOs:
  3 of 4 on (1,1,32,32), 2 of 4 on (1,4,64,64) and (1,2,64,96). This is folded into R1 (C ≤ units_per_head).
- **L1 headroom**: the largest first-phase shape uses 30.6 % of L1. The first ceiling a longer sequence or a
  wider head meets is the per-ring limit (1 MiB on the S/P and O rings, `C·qc·kc·T_acc`), not the budget. The
  fit handles it by shrinking `kc` (streamed regime), so there is no OOM path.
