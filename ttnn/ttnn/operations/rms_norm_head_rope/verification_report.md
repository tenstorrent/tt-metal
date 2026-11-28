# Verification Report: rms_norm_head_rope

Verifier pass on Phase 0 (commits `f9c098cf288`, `fb3229e95ad`). Blackhole, 12×10 compute grid.

## Code Review

### Fixed in place

1. **FPU operand placement biased the weight / RoPE output (precision).** In the original kernel, the weight multiply
   was `DestReuseBinary<Mul, cb_weight, DEST_TO_SRCA>`, which moves the fp32 DEST value `y` onto srcA. The rope chain
   was `BinaryFpu<Mul, normed|rotated (DEST-width page), cos|sin (bf16)>`, which also puts the wide operand on
   srcA. Wide values on srcA lose low mantissa bits under the fidelity phases. The precision baseline's got/true
   ratio caught it: the ratio median was **0.9964** for weight+rope under fp32 DEST, against 0.9998 for the plain
   path. Attribution probe at 256×3072: weight-only 0.9971, rope-only 0.9991.
   Fix: `DEST_TO_SRCB` for the weight multiply, and swapped A/B in both rope multiplies (`cos·y`, `sin·rot`), so the
   bf16 operand is always on srcA. Effect (fp32 DEST, weight+rope, 2304×3072):
   - rel RMS 0.0044 → **0.0019**
   - max abs 0.177 → **0.091**
   - ratio median 0.9964 → **0.9991**

   bf16-DEST numbers are unchanged. The change is perf-neutral (A/B profile, 3 shapes, all within ~2% noise; see
   the Perf section). This was not a scale bug: rel-RMS was 0.0044, far from the ≳0.1 scale-bug signature. It was a
   systematic rounding skew, removed in place.
2. **`select_grid_split` could return `None` → `TypeError`.** This happened when no `G` satisfied
   `ceil(H/G) ≤ MAX_RESIDENT_HEADS`, i.e. `H > 24·num_cores` (W > ~368K). It now raises a clear `ValueError` that
   names the resident-weight L1 cap. It can't be reached from the corpus, but it was an unguarded crash.
3. **`_check_aux` had an unused `device` parameter.** Removed, along with the dead `device = ...` local in `validate()`.
4. **Compute-kernel header** now documents the operand-placement rule and why `cb_sumsq` needs no tail realign:
   `reduce<REDUCE_ROW>` over `of(hb, 1)` waits and pops one tile per row, so its window never wraps. I verified this
   on device, see 5.
5. **Coverage gap closed (new test file `test_rms_norm_head_rope_tail_blocks.py`).** No acceptance or golden shape
   reaches two block-schedule paths:
   - `heads_per_core > MAX_BLOCK_HEADS`: several blocks per row-tile, with cos/sin held across blocks, plus a tail
     block. Covered by `(1,1,1472,5504)`, 43 heads.
   - Uneven head split where every block on some cores is a multi-tile tail (`hb=2 < bh=3`). This exercises the
     compute-internal CB realign. Covered by `(1,1,1952,1024)` with rope, `G=3` over 8 heads.

   Each test asserts the `(G, hpc, bh)` it lands on, so a change to `select_grid_split` can't silently drop the path.
   All 4 cases pass.

### Reviewed, no change needed

- **Helper usage.** sum_squares, normalize and apply_rope run as `eltwise_chain`, reduce as `reduce<>`, and the
  weight expansion as `unary_bcast`. The one raw-LLK region, `rotate_pairs_block` (matmul vs. the 32×32 rotation
  tile), is justified: `ttnn/cpp/ttnn/kernel_lib/` still has no matmul helper on this branch (re-checked), and no
  chain element does an intra-tile column permutation. Its init is issued once per block and it packs up to
  `DEST_AUTO_LIMIT` tiles per acquire.
- **CB sync.** Every push count matches its wait/pop count, including tail-block padding: reader and writer move
  whole `BT` windows, and compute pads the output window and realigns `cb_rstd`/`cb_normed`/`cb_rotated`.
- **TensorAccessor.** Used for all interleaved I/O.
- **Kernel API conventions.** `void kernel_main()` and `api/...` include paths are used throughout.
- **Broadcasts.**
  - rstd: Col broadcast from the col-0 reduce output.
  - weight: Row broadcast from row-0-only reads, 2×64 B per tile.
  - cos/sin: `InputTileMapping::Row`, reused across all heads with no replication.
  - No CB is filled with redundant repeated data.
- **Blocking-model fidelity / DRY.**
  - `block_heads`, `MAX_BLOCK_HEADS`, `MAX_RESIDENT_HEADS`, the depth knobs and `STORE_CHUNK_HEADS` are each
    defined once in the descriptor. Kernels receive them as CT args, and every CB page count and loop bound is
    derived from them.
  - `HEAD_DIM` is defined once and the op file imports it.
  - **One collapsed knob:** `BLOCK_ROW_TILES` is a host constant asserted `== 1` and never passed to the kernels,
    which hard-code one row-tile per block. The design records this as a Phase 0 limit (cos/sin would need
    Strided addressing). Re-exposing it is kernel rework, so it is filed inside **Refinement 1** (first refinement),
    not fixed here.
- **Expression check.**
  - Reader: one barrier per block over `4·bh` tile reads.
  - Compute: block-scoped chains, DEST-blocked inside each.
  - Writer: streams the block in `STORE_CHUNK_HEADS`-head chunks. Each chunk is flushed, with one barrier per block.
    That per-chunk handshake is a measured knob (it helped at 256×3072), not a collapsed size.
- **Performance conformance.** The design's 2-D (row-group × head-group) split fills the grid: 120/120 cores at
  R ≥ 2048 and 96/120 at R=256. Both dataflow halves are batched per block, and the I/O CBs are double-buffered.

### Prompt rules (`eval/prompts/rms_norm_head_rope.txt`)

| Rule | Status |
|---|---|
| MUST handle TILE natively; no to_layout/tilize/untilize/pad/slice/op composition in the entry point | ✓ entry point = validate + allocate + one `ttnn.generic_op` |
| Mean square in fp32 where the config allows | ✓ fp32 DEST accumulation of x² and a Float32 `cb_sumsq` page |
| rsqrt accurate to bf16 round-off | ✓ `rsqrt_tile<Default>`; the plain path's ratio median is 0.9998 and ULP mean 0.25 |
| MUST NOT cache anything derived from weight/cos/sin | ✓ descriptor and weight expansion are rebuilt per call; `test_..._repeat_calls_no_cache` passes |
| One native dispatch per call | ✓ |
| ValueError/RuntimeError on bad inputs | ✓ (`SupportRefusal` ⊂ `NotImplementedError` ⊂ `RuntimeError`) |

## Registry Conformance

- **INPUT_TAGGERS:** `alignment` and `rank`, both with the `(inputs, axes)` signature.
- **SUPPORTED:** covers every gated axis, including `fp32_dest_acc_en`, the affine and rope axes, and `memory_layout`.
- **EXCLUSIONS:** `[]`.
- **validate():** checks SUPPORTED per axis, then EXCLUSIONS, then the structural checks. It is the first statement
  of the entry point.
- **INVALID:** not declared in the op file ✓.
- **SUPPORTED vs. TARGET:** SUPPORTED is a superset of TARGET. The only extra is `fp32_dest_acc_en=False`, which the
  op spec allows; the golden suite doesn't sweep it, but the acceptance test and the precision baseline cover it.
  **`TARGET − SUPPORTED` is empty**, so the queue has no generality candidates.
- No auto-fixes to SUPPORTED were needed (`xpass_drift = 0`).
- **INVALID audit** (`feature_spec.py`): six entries.
  - Each couples two axes that describe the **same** tensor (`affine` × `affine_dtype` / `affine_layout`,
    `rope` × `rope_dtype`), so they are pure "absent tensor ⇒ `none`" canonicalizations. That satisfies all three
    rules.
  - The bf8b+ROW_MAJOR canonical entry is not applicable: TARGET has neither bf8b nor ROW_MAJOR.
  - Nothing to change.

## L1 Ledger Audit

- **Currency.** All 12 CBs match `l1_ledger.md`: index, size expression, and format (`DF` = Float32 iff fp32 DEST).
- **Capacity vs. live set.**
  - Only the I/O CBs and the rope CBs exceed their live set, each by its declared depth-2 double buffer.
  - Compute-internal CBs are sized to exactly one block.
  - `cb_weight`/`cb_weight_row` span the core's head group (`HC·4`), bounded by `MAX_RESIDENT_HEADS = 24` through
    `select_grid_split` (a predicate-guarded residency). The new `ValueError` now enforces that bound instead of
    crashing.
  - No under-sized spanning buffer.
- **Page format vs. DEST.**
  - Over: none. Float32 pages appear only under fp32 DEST, on CBs whose values left DEST.
  - Under: none. `cb_weight` is bf16 under fp32 DEST, but it holds an exact bf16 copy.
  - The design's perf lamp "bf16 `cb_normed`/`cb_rotated` under fp32 DEST" would be an *under* violation, so it is
    out of bounds as a perf lever. This is now noted in the ledger.
- **Disjoint lifetimes.** One justification was missing: `cb_sumsq` ↔ `cb_normed`/`cb_rotated`. I added it to the
  ledger: there is no alias mechanism on the descriptor path, and the saving is ≤ 32 KB. `cb_weight_row` ↔
  `cb_rotated` was already justified.
- **Bounds / closed form.** Every symbol is bounded with its establishing predicate, and the total is closed-form.
  No CB scales with R or W except through bounded `HC`.
- **Data-movement budget.**
  - Present, and consistent with the implemented split. cos/sin are counted ×G, and weight row-0 reads ×Rg.
  - The cheapest-traffic split (rows-only, −2.0 MB) is recorded with a positive reason (per-core max 25 vs 18
    units, 48 cores idle).
  - The cos/sin multicast (R2) is a `deferred` regime with a measured-threshold reason. Because cos/sin is a
    **reuse-shared operand with a deferred broadcast regime**, the verifier rules require filing it as its own
    measured refinement. It is filed as **Refinement 3**: it must be built and measured, not ruled out from catalog
    thresholds alone.
- **Block-size defaults.** Interleaved split across the full grid, then the coarsest block that fits:
  `bh = min(HC, 8)`, which is the whole head group at every FLUX shape. The implementer measured
  `bh ∈ {1,2,4,8}` as flat (119–122 µs at 2304).
- **Per-core footprint:**
  `2KB·(input_depth·BT + 2·rope_depth·4 + 2·HC·4 + 2 + output_depth·BT) + DF·(2·bh + 2·BT)`.
  - Terms scaling with `bh`: x, out, sumsq, rstd, normed, rotated.
  - Terms scaling with `HC`: the two weight CBs.
  - Worst case 996 KB; FLUX R ≥ 2048: 740 KB; R=256: ~190 KB.

## Precision Baseline

`test_rms_norm_head_rope_precision_baseline.py`, HiFi4, after the operand fix. Reference is fp32 golden, not rounded.
ULP and ratio are computed over |ref| > 1e-3.

**fp32 DEST (default config):**

| Shape | Variant | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP mean | ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|
| (1,1,32,128) | weight+rope | 0.999998 | 0.040 | 0.00164 | 0.00199 | 0.32 | 0.9991 [0.9963, 1.0020] |
| (1,1,256,1024) | weight+rope | 0.999999 | 0.052 | 0.00167 | 0.00192 | 0.33 | 0.9991 [0.9962, 1.0020] |
| (1,1,256,3072) | weight+rope | 0.999999 | 0.089 | 0.00168 | 0.00192 | 0.33 | 0.9991 [0.9962, 1.0020] |
| (1,1,2304,3072) | weight+rope | 0.999999 | 0.091 | 0.00168 | 0.00191 | 0.33 | 0.9991 [0.9962, 1.0020] |
| (1,1,32,128) | plain | 0.999998 | 0.0099 | 0.00118 | 0.00174 | 0.26 | 0.9999 [0.9969, 1.0024] |
| (1,1,2304,3072) | plain | 0.999999 | 0.0173 | 0.00114 | 0.00169 | 0.25 | 0.9998 [0.9972, 1.0025] |

**bf16 DEST:**

| Shape | Variant | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP mean | ratio median |
|---|---|---|---|---|---|---|---|
| (1,1,256,3072) | weight+rope | 0.999988 | 0.189 | 0.00445 | 0.00504 | 1.49 | 1.0010 |
| (1,1,2304,3072) | weight+rope | 0.999988 | 0.254 | 0.00446 | 0.00502 | 1.49 | 1.0010 |
| (1,1,2304,3072) | plain | 0.999995 | 0.048 | 0.00224 | 0.00350 | 0.50 | 1.0013 |

**Assessment:**
- **fp32 DEST:** essentially bf16 round-off. The plain path's ULP mean is 0.25, i.e. output rounding only.
  Weight+rope reaches 0.33 ULP mean.
- **Ratio spread:** broad and centred on ~1.0, so there is no scale bug.
- **ULP max outliers:** up to ~470 under fp32 DEST, from cancellation in `y·cos + rot·sin` near zero. They are
  absolute-error-bounded (max abs 0.09), not relative.

**Recommended tolerances:** the defaults hold, with a large margin: `(bf16, True)` PCC ≥ 0.995 / rel RMS ≤ 0.04,
and `(bf16, False)` 0.99 / 0.13.

## Verifier CLI Summary

`verifier_report.json` (copy in this directory; run dir `/tmp/rnhr_golden`):
- supported_pass: **32**
- xfail_expected: 0 (no gap: `TARGET ⊆ SUPPORTED`)
- invalid_skipped: 224
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**
- no_axes_found: 19 (the 18 `@numerics` regression tests, all passing, plus the empty `LOOSE_CASES` placeholder)

Other suites:
- Acceptance: 48/48.
- Tail-block (new): 4/4.
- Precision baseline (new): 16/16.
- Unit directory total: 75 passed.

## Perf (device kernel duration, BH, weight+rope, HiFi4 fp32 DEST)

| Shape | Calls/step | Phase 0 + verifier fix | Reference op | Roofline |
|---|---|---|---|---|
| 256×3072 | 16 | 20.0–20.2 µs (96 cores) | 99.0 | 8.3 |
| 2048×3072 | 16 | 114–115 µs | 129.8 | 58.1 |
| 2304×3072 | 96 | 118.6–119.4 µs | 137.5 | 65.2 |

- **A/B of the operand fix at 2304×3072:** before 118.2 / 121.6 µs, after 118.6 / 119.4 µs. That is noise
  (`perf_rules`: < 2–3%).
- **Headroom:** 2304×3072 is ~1.8× the roofline and dominates the step (96 × ~119 µs ≈ 11.4 ms/step vs 1.8 ms for
  2048 and 0.3 ms for 256).
- **Implementer attribution at 2304:**
  - DM-only 89 µs; reads alone ~290 GB/s, writes alone ~250 GB/s.
  - Compute-only 76 µs.
  - Full 120 µs.

  Neither DM nor compute is saturated, and they overlap poorly. Both are real levers, which sets the perf queue.

## Recommendations

- **The queue is all perf.** There are no generality candidates. The three perf refinements are ordered by
  step-time impact on 2304×3072 (DM efficiency → compute → cos/sin broadcast).
- **The perf catalog `ttnn/ttnn/operations/examples/master.md` is absent on this branch** (the design notes it too).
  The refinements name their levers by the catalog-entry names the design cites (`noc_placement`, `double_buffer`,
  `shared_input_reuse`, `compute_fusion`, `compute_block_size`) and leave lever selection to the implementer.
- **The "Intermediate page format" perf lamp is out of bounds:** a bf16 `cb_normed`/`cb_rotated` under fp32 DEST is
  an under-width page.
- **Any operand reorder or formulation change must keep the srcA = bf16 / srcB = wide placement**, and must be
  re-gated on the precision baseline's ratio median. The fix above shows the cost of getting it wrong.
- **Tooling friction:** `run_safe_pytest.sh` auto-precompile crashes with
  `ttnn.graph has no attribute up_front_clear` (the plugin is ahead of the binding). All runs here used
  `--no-precompile`.
