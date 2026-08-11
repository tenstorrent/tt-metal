# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-08-11
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Multi-core from day 1: the grid is partitioned into row-groups; the independent `row`
  axis is split across groups and the **dependent** `hidden` axis is split *within* a group, with
  partial `Σ x²` combined over the NoC (gather-to-root + `mcast_pipe` multicast-back). Both regimes
  ship (R1 `w_group_size == 1` row-parallel, R2 `w_group_size > 1` cross-core). Native TILE and
  ROW_MAJOR I/O (in-kernel tilize/untilize, no host-side `to_layout` / `pad` / `slice`), and the
  ragged hidden tile is masked numerically before squaring so the RMS denominator counts only valid
  elements.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED].
  EXCLUSIONS=[{dtype: float32, fp32_dest_acc_en: False}].
- **Accuracy achieved** (measured on 5 shapes × 2 dtypes × 2 layouts × gamma/no-gamma via
  `test_rms_norm_precision_baseline.py`; TILE + gamma rows quoted):
  bfloat16 — PCC = 0.999997, max_abs ≤ 0.066, mean_abs ≈ 0.0012, rel_rms ≈ 0.0023–0.0026,
  got/true ratio median 0.9999–1.0002 (p5/p95 ≈ ±0.4 %);
  float32 — PCC = 0.9999997, max_abs ≤ 0.027, mean_abs ≈ 0.0007, rel_rms ≈ 0.0012–0.0015,
  ratio median 0.9989–0.9991 (p5/p95 ≈ ±0.13 %; a systematic ~0.1 % shrink from the FPU's
  truncating fp32 mantissa, not a scale bug).
- **Golden suite at Phase 0**: 737 / 737 supported cells passing (`supported_pass` = 737,
  `xfail_expected` = 6174, `invalid_skipped` = 33900, and **0** in each of `supported_fail`,
  `xpass_drift`, `xfail_wrong_mode`), per `verifier_report.json`.
- **Issues encountered**: no drift and no functional failures — the SUPPORTED block was already
  honest. Verifier code-review fixes: (1) the CB inventory was stated **twice** (the descriptor's CB
  list and the L1-residency solve's `fixed_bytes` / `per_row_bytes` expressions), so a page-count
  change could silently drift the solve — the inventory is now one `_cb_specs()` table and the solve
  *derives* the footprint from it by differencing (verified byte-identical to the previous closed
  form over 4608 configurations); (2) `cb_input_tiles` reserved a depth-2 prefetch slot even when the
  busiest core's whole row assignment is a single block (e.g. `(1,1,8192,1024)`), where no block
  `b+1` exists — the second buffer is now not allocated in that case; (3) the reader / writer read
  their `TensorAccessorArgs` / `McastArgs` from hardcoded arg offsets that no assertion pinned, so an
  added host arg would have silently fed a peer's args — the three offsets are now named host
  constants with build-time asserts. `l1_ledger.md` updated for all of the above plus a stale
  measured-selection row.
- **Tests added**: `test_rms_norm.py` (acceptance, 82 cases — immutable spec),
  `test_rms_norm_shapes.py` (adversarial-shape sweep, 92 cases),
  `test_rms_norm_perf.py` (device-ns harness + the P1/P2 knob sweeps),
  `test_rms_norm_precision_baseline.py` (**new** — PCC / abs / rel-RMS / got-true ratio spread,
  40 cells).

## Refinement 1 — Numerical configurability expansion (`fp32_dest_acc_en=False` + `bfloat8_b`)
- **Date**: 2026-08-11
- **What was done**: extended SUPPORTED to `fp32_dest_acc_en=[True, False]`,
  `dtype += bfloat8_b`, `gamma_dtype += bfloat8_b`, keeping
  `{dtype: float32, fp32_dest_acc_en: False}` in EXCLUSIONS (now REACHABLE and actively enforced
  rather than a declared-but-dead entry). **Zero kernel changes** — all three `.cpp` files are
  byte-identical to Phase 0; `/numeric-formats-metal`'s pass condition held, so the entire diff is
  the op file + the program descriptor + tests.
  Reused rather than rebuilt: (1) DEST capacity is only ever read through `DEST_AUTO_LIMIT` /
  `Dst::D0`, so the 4 → 8 doubling when fp32 DEST accumulation is off needed nothing — no literal
  in the kernels or the L1 solve assumed the halved value; (2) the whole `cb_stat_*` chain
  (`cb_stat_sq`, `_partial`, `_gather`, `_sum`, `cb_rstd_send`, `cb_rstd`) plus `cb_zero_tile` were
  already pinned to fp32 **unconditionally** instead of following `fp32_dest_acc_en`, and
  `cb_scaler` / `cb_wmask` already stay bf16 (`reduce_helpers_dataflow.inl:185-187` `static_assert`s
  the scaler format) — so `Σ x²` still crosses L1 in fp32 and only the *in-DEST* accumulation
  narrows, which is what keeps `row_reduce_accumulate`'s failure mode away; (3) `_cb_specs()`
  already derived every input/output/gamma CB format from the corresponding tensor's dtype, so
  `bfloat8_b` flowed through as a third `tile_size` value (1088 B) with no new branch — and because
  a bfp8 tile is *smaller* than bf16/fp32, the residency predicate only gets easier and the selected
  `(G, C, R)` can only get coarser.
  The one real code change: `_elem_bytes()` / `_BLOCK_FLOAT_DTYPES` in the descriptor, because
  `Tensor.element_size()` **raises** for block-float ("datum for bfp2, bfp4, bfp8 is invalid"). It
  returns 0 for `bfpN` — the only consumers of an element size are the ROW_MAJOR stick legs, and a
  block-float tensor has no ROW_MAJOR form (ttnn itself `TT_FATAL`s `layout == Layout::TILE`), which
  independently confirms `feature_spec.INVALID` is right to *skip* those cells rather than xfail them.
- **Accuracy achieved** (vs the golden suite's own per-dtype `TOLERANCES`, on shapes
  `(1,1,32,64)`, `(1,1,64,128)`, `(4,8,32,256)`, `(1,1,32,4096)`, `(1,1,32,8192)`, `(2,4,128,512)`,
  `(1024,1024)`, plus the cross-core-forcing `(1,1,32,16384)`, `(1,1,32,32768)`, `(1,1,64,12288)`,
  `(1,1,160,11008)`):
  bfloat16 @ `fp32_dest_acc_en=False` — PCC ≥ 0.999937, rel-RMS ≤ 0.012 (gate 0.995 / 0.04);
  bfloat8_b (both DEST settings, incl. the wide cross-core combine) — PCC ≥ 0.999854,
  rel-RMS ≤ 0.021 (gate 0.99 / 0.10);
  `gamma_dtype=bfloat8_b` against bfloat16 and float32 activations — PCC ≥ 0.999937,
  rel-RMS ≤ 0.0117;
  `pad_poison` (poison 1000.0) @ `fp32_dest_acc_en=False` + HiFi2 — PCC ≥ 0.999979 on all 6 shapes,
  i.e. the mask-before-square identity `(x·mask)² == x²·mask` holds under a 16-bit DEST accumulator.
  rtol/atol: the precision matrix reports max/mean-abs and the got/true ratio spread per cell rather
  than gating on a single rtol/atol pair; the scale tripwire is `|ratio_median − 1| < 0.02`
  (0.05 for bfloat8_b, 0.08 at LoFi — see Issues).
- **Golden test progress**: targeted slices, all clean (the harness re-runs the full suite).
  `1x1x64x128 × INTERLEAVED` = **48 passed / 12 xfailed** — and 48 is exactly what SUPPORTED now
  predicts (bf16 × 2 DEST settings × 6 gamma combos × 2 layouts = 24, fp32 × 1 setting = 12,
  bfp8 × 2 settings × 6 gamma combos at TILE only = 12), with the 12 xfails being precisely the
  `{float32, fp32_dest_acc_en=False}` EXCLUSIONS cells → **no XPASS drift**. Non-aligned shapes
  `1x1x32x50` + `1x1x17x64` × INTERLEAVED = 72 passed / 24 xfailed. `-m pad_poison` = 6 passed
  (all interleaved) / 18 xfailed (the 3 sharded placements — Refinement 2). `-m perf` = **8 passed**
  (every interleaved perf case, now running at its pinned `fp32_dest_acc_en=False` + HiFi2) /
  5 xfailed (sharded — Refinements 2/5). This is the gate Refinements 3 and 4 were waiting on:
  they can now measure their specified configuration instead of proxying it at
  `fp32_dest_acc_en=True`.
- **Issues encountered**: two, neither an op defect.
  (1) **LoFi systematic shrink.** `test_rms_norm_precision_matrix_gamma_dtype` initially failed 2 of
  140 cells (LoFi + bf16) on the got/true-ratio tripwire at ratio median 0.965 — a ~3.5 % *uniform*
  shrink. Cause is hardware, not the kernel: LoFi truncates srcA/srcB to a 5-bit mantissa and the
  FPU **truncates rather than rounds**, so each of the `x · rstd` and `· gamma` multiplies biases
  low and the two compound. PCC stayed ≥ 0.9995 throughout. Same mechanism Phase 0 already recorded
  as a ~0.1 % shrink at fp32/HiFi4. Fixed in the *test*: the scale-bug band is now derived from
  `math_fidelity` and dtype (0.02 base → 0.05 for bfloat8_b → 0.08 at LoFi) so the tripwire keeps
  detecting real scale bugs instead of firing on documented hardware behavior.
  (2) **`element_size()` raises for block-float** — see above. Found on the first bfp8 probe, before
  any kernel work, which is why bfloat8_b needed no kernel changes at all.
  Nothing was added to EXCLUSIONS: every cell this refinement named measured green.
- **Tests added**: `test_rms_norm_precision_matrix` + `test_rms_norm_precision_matrix_gamma_dtype`
  in `test_rms_norm_precision_baseline.py` (140 cells: 5 shapes × 3 dtypes × 2 `fp32_dest_acc_en`
  × {HiFi4, HiFi2} × {normal, uniform}, plus an independent gamma_dtype × {HiFi4, LoFi} sweep;
  uniform inputs are kept deliberately — all-positive data is the monotonically-growing-sum regime a
  narrowed DEST accumulator degrades on). Skips are only the two declared-impossible sets
  (`{float32, accFalse}` = EXCLUSIONS, `bfloat8_b × non-tile-aligned` = `feature_spec.INVALID`).
  `test_rms_norm_shapes.py` gained an `fp32_dest_acc_en` axis (92 → 184 cases): its whole premise was
  "the golden resilience / pad_poison cases pin `fp32_dest_acc_en=False`, which is unsupported, so
  replay them at `True`" — that premise is now obsolete, so it sweeps both DEST datapaths and its
  docstring says why. Unit directory: **464 passed / 36 skipped / 0 failed** (was 372/36/0).

## Refinement 2 — Sharded input/output placements (all three schemes)

- **Date**: 2026-08-11
- **What was done**: added `HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to
  `SUPPORTED["memory_layout"]`, consuming the resident shard **in place** and producing a matching
  sharded output. This was a *placement* unlock of the scheme Phase 0 already built (lamp S1) — the
  cross-core combine, `_cb_specs`/`_cb_bytes`/`_max_block_row_tiles`, `read_slice_rows` /
  `write_slice_rows`, the `Mcast2D` wiring and the **entire compute schedule** were reused unchanged.
  What was added:
  - `_ShardView` — the shard spec read AS the block geometry, so `_select_regime` is never called for
    a sharded input: HEIGHT cuts the independent `row` axis ⇒ `w_group_size = 1` and the combine
    degenerates to a local copy; WIDTH cuts the dependent `hidden` axis ⇒ the whole shard grid is
    **one** reduction group; BLOCK cuts both ⇒ one grid row of the shard rectangle is a group.
    `_shard_core_infos` clamps every extent to the tensor, because a shard grid need not divide it
    (`auto_shard_config` ceil-splits and pads the last shard).
  - `_sharded_cb` — `cb_input_tiles` / `cb_output_tiles` pinned **zero-copy** over the resident L1
    buffer via `ttnn.cb_descriptor_from_sharded_tensor` on the TILE legs. `load_block` and
    `store_block` become the CB handshake alone: **zero NoC traffic** for the activations. The
    shard's pages are already tile-row-major at row stride `C`, and total pushes
    (`core_row_tiles · C`) never exceed the shard's page count, so the CB pointer never wraps.
  - `LocalShardAccessor` + `read_shard_rows` / `write_shard_rows` — the ROW_MAJOR legs cannot pin
    (the block CB there is the `tilize`/`untilize` staging buffer, whose group-uniform tile-row
    stride is not the shard's stick stride), so they re-stride the sticks **core-locally**, L1 → L1,
    with a single bulk transfer per 32-row group when the strides already agree. Still zero DRAM
    crossings. `LocalShardAccessor` exposes the same `get_noc_addr(page, offset)` shape as
    `TensorAccessor`, which is what lets one `read_slice_rows` body serve both legs.
  - **Per-core `partial_w`.** A ROW_MAJOR WIDTH/BLOCK shard's width granule is the **L1 alignment**,
    not a tile, so a core's hidden slice can be 8–16 elements of a 32-column tile: *every* core can
    carry a ragged tail, not just the one owning the tensor's last hidden tile. The CT constant
    `PARTIAL_W` became the runtime `core_partial_w` (plus a CT `HAS_ANY_TAIL` to gate the mask CB);
    on every other path this reduces to exactly the old `owns_last_w_tile && W % 32` condition.
  - **Mcast-box filler cores.** A WIDTH shard grid is often *not* a rectangle (16 slices on an
    11-wide grid = a full row + a 5-core row) while the `rstd` broadcast needs one. The group
    rectangle is now the shard grid's bounding box, and the cores in it that own no shard stay
    program cores carrying the identical CB map — so the broadcast lands in a reserved `cb_rstd`
    instead of unowned L1 — with `num_blocks = 0` returning them immediately from all three kernels.
    They receive and never ack, which is why `Mcast2D` is now given an explicit `num_active = G − 1`
    instead of the dense fan-out (`mcast_pipe`'s documented divergent-ack case).
  - Two L1 levers: `_DEPTH_LADDER` (a shard spec pins both `G` and `C`, so `R` and the buffer depths
    are the only residency knobs left; depth is spent first and surrendered last, and no interleaved
    geometry ever leaves step 0, so the default path is byte-identical), and the `cb_gamma_rm` →
    `cb_input_rm` alias (Rule 3 pattern 3: disjoint lifetimes, same producer *and* consumer, guarded
    on equal page formats so a mixed-precision gamma keeps its own buffer). Together they took the
    loose-sharded slice from 263 to 269 passing.
- **Accuracy achieved**: PCC ≥ 0.99996 (bf16 rtol/atol well inside the golden gate 0.995/0.04) across
  all three schemes × {TILE, ROW_MAJOR} × {bfloat16, bfloat8_b, float32} on shapes
  `(1,1,256,512)`, `(1,1,32,2048)`, `(1,1,100,736)`, `(1,1,224,1000)`, `(1,1,32,4095)`,
  `(1,1,32,4096)`, `(1,1,32,8192)`, `(128,8192)`, `(1,1,160,11008)`, `(2047,2047)`, `(99991,64)`.
  `bfloat8_b` sharded PCC ≥ 0.99986 (gate 0.99/0.10). The six `pad_poison` shapes are green on all
  three placements: PCC ≥ 0.99998 with got/true ratio ≈ 1.0, i.e. the mask-before-square identity and
  the true-`W` divisor both survive sharding. All five pinned `group="perf"` shard geometries run at
  PCC ≥ 0.9998, which **unblocks Refinement 5**.
- **Golden test progress**: loose-sharded slice **269 / 281** passing (3 skipped). Cartesian slices:
  **516 / 516** on `{1x1x32x64, 1x1x32x50, 1x1x17x64, 4x8x32x47}` (all four placements, all dtype /
  precision / gamma combinations) and **761 / 777** on
  `{2x4x128x512, 1x1x2048x256, 1024x1024, 1x1x32x4096, 1x17x128}`. No xpass-strict drift. Prior
  phases unchanged: unit directory 82 + 324 passed, 0 failed.
- **Issues encountered**:
  1. **A DRAM read TRUNCATES its source address to the DRAM alignment** (64 B on Blackhole) — no
     assert, no hang, just a neighbouring slice. Every core whose gamma slice started mid-tile was
     reading gamma from the aligned offset *below* its own: PCC 0.28–0.50 on RM WIDTH/BLOCK. Isolated
     by a deterministic probe (all-ones input + a repeating gamma ramp): the no-gamma output was
     *exactly* 1.0 everywhere, so the reduction was provably right and only gamma was misplaced, and
     the observed gamma offsets were the true ones quantized to 64 B. Fixed by having the host pass a
     DRAM-**aligned** read offset plus the leading bytes to drop; `lead == 0` on every previously
     working path, so those reads are byte-identical.
  2. **The shifted re-read then hit a sub-L1-alignment offset.** With fp32 activations and a bf16
     gamma the offset is a multiple of the *gamma* element size only (8 B against the fp32 ROW_MAJOR
     4-element granule), so the L1 → L1 hop that performs the shift was itself misaligned: PCC 0.44
     on 4 cartesian cells. Fixed by doing the shift with the CPU (`copy_l1_bytes`) — the one transfer
     whose source alignment is not ours to choose.
  3. **`_cb_bytes` was dropping its `depths` argument**, so the depth ladder computed identical
     numbers at every step and rescued nothing. Caught by instrumenting the solve on the failing
     shapes rather than by reading the code.
  4. Pad columns of a sub-tile slice are now zeroed in `read_slice_rows` (free when the slice fills
     the block row, i.e. every tile-aligned case). `garbage · 0` is NaN if the garbage is non-finite,
     and a ROW_MAJOR WIDTH shard makes half of every tile a pad column, turning a latent risk into
     the common case.
  5. **Deferred (Refinement 2b), one class, 12 loose + 16 cartesian cells:** `HEIGHT_SHARDED` at
     large `W`. HEIGHT cuts the independent axis, so `G = 1` and `C = tensor_w_tiles` — the caller
     pinned the knob the solve uses to bound `C`. Past `C ≈ 127` (bf16) / `C ≈ 64` (fp32) the two
     resident shards plus `cb_gamma_tiles` alone exceed the budget, and the descriptor raises. This
     is a **capacity** limit, not a tuning one: verified at the ladder's last step with `R` already 1.
     Left failing rather than silenced in EXCLUSIONS. Refinement 2b names the lever — chunk the
     hidden axis inside a core, which for a *resident* shard is a nearly free regime R3, since R3's
     fatal cost (a second whole-tensor DRAM read for the apply pass) is zero when the input is
     already in L1.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded.py` (31 cases) —
  correctness per scheme × layout, no-gamma, output-inherits-shard-spec, the sub-tile ROW_MAJOR
  slices and the mixed gamma dtype (regressions for issues 1 and 2), the five pinned perf geometries,
  and `test_rms_norm_tile_shard_is_consumed_in_place`, which asserts the **zero-copy contract
  structurally** on the program descriptor. That last one matters: an accessor read of a core's own
  shard returns the same bytes, so no numerical test can tell "sharding implemented" from "sharding
  tolerated" — only the descriptor can.
