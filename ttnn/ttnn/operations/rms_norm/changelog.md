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

## Refinement 2b — HEIGHT_SHARDED wide-`W`: chunk the hidden axis inside a core
- Date: 2026-08-11
- **What was done**: added **one block factor**, `w_chunk_tiles` (WC), the block extent along
  `hidden` of every buffer that only *streams* over that axis (`cb_gamma_tiles`, `cb_normed`,
  `cb_output_tiles`, `cb_output_rm`, `cb_input_rm`). It is a single compile-time arg
  (`CB_CHUNK_TILES`) in all three kernels, from which `NUM_CHUNKS = ceil(C / WC)` is derived; at its
  default `WC == C` every loop it introduces runs exactly once, so the interleaved and
  already-fitting sharded schedules are unchanged. `cb_input_tiles` is deliberately **not** chunked:
  the block stays whole-resident, which is what makes this a nearly free regime R3 — R3's fatal cost
  in the traffic ranking is a second whole-tensor read for the apply pass, and a resident shard is
  already in L1, so that read does not exist.
  - The residency solve turns the knob only as a last resort and takes the **coarsest** chunk that
    fits: the depth ladder runs first at `WC == C` (so nothing that fits today changes), then WC
    walks down from `C - 1`. Gated by `_chunking_supported()` on a resident shard with a *uniform*
    per-core hidden geometry (the chunk count sets `cb_stat_sq`'s column stride and every chunked
    CB's push/pop quantum, which must divide a capacity shared group-wide) and on input and output
    being both pinned or both staged.
  - **Per-chunk statistics without an L1 accumulator**: each chunk's partial `Σ x²` packs into its
    OWN `cb_stat_sq` column, and `reduce_stat_block` — which already folds a tile-row's `nc` columns
    (the ragged-tail column proved the pattern) — sums them for free. `eltwise_chain` forbids
    `L1Accumulation` together with `DestAccumulation` (`eltwise_chain.inl:1034`), so the DEST
    accumulation that `sumsq_block` is built on is preserved rather than traded away.
  - **Two block layouts, one decoder**: a ROW_MAJOR block is chunk-major (`tilize<WC>` emits chunks
    back to back), a TILE block (interleaved or pinned shard) stays row-major at stride `C`.
    `in_ref(g, rows)` in the compute kernel is the only place that knows which, and the two coincide
    at `NUM_CHUNKS == 1`.
  - **Pinned output**: with the output shard pinned, the apply packs at a strided offset under a
    caller-managed reserve/push (the shard's layout is not the compute order), which also costs the
    CB arena nothing. The writer is untouched on that leg.
  - Gamma is re-fed per chunk from inside the block loop when chunked (`W` bytes per block, on
    geometries whose whole slice would not fit at all); the `cb_gamma_rm → cb_input_rm` alias is
    disabled there, since the two lifetimes then interleave instead of being disjoint.
- **Accuracy achieved**: PCC ≥ 0.99987 on every recovered cell — `(1,1,32,4064)` 0.999996 /
  `(1,1,96,6144)` 1.000000 / `(1,1,992,3000)` 0.999924 / `(3,1,736,5119)` 0.999866 /
  `(1,1,32,4095)` 0.999998 / `(100,5120)` 0.999995 / `(3104,4064)` 0.999894 (TILE and ROW_MAJOR),
  and `(1,224,11008)` / `(1,1,160,11008)` ROW_MAJOR 1.000000 / 0.999963. Chunking with a real
  cross-core combine (pinned `[32,8192]` WIDTH shard on 2 cores, and the BLOCK equivalent) is
  PCC 1.000000.
- **Golden test progress**: HEIGHT loose slice **91/93** (was 81/93 — 10 of the 12 named failures
  recovered); cartesian `1x1x32x4096` HEIGHT column **39/39**, including all 19 fp32-activation /
  fp32-gamma cells that previously refused; WIDTH+BLOCK loose 187/188 and INTERLEAVED loose 103/103
  unchanged. Interleaved device-kernel perf re-measured on all 8 perf shapes: within ±2 % of the
  recorded numbers (8192×7168 594 707 ns vs 597 240 ns), i.e. the strided apply costs nothing.
- **Issues encountered**:
  1. `eltwise_chain` rejects `L1Accumulation` + `DestAccumulation` in one chain, which ruled out the
     obvious "accumulate each chunk into one stat tile" shape — resolved by the per-chunk stat column
     above, which is cheaper anyway (one reduce per block, not per chunk).
  2. The CB wrap rule (`dataflow_api.h:216-221`) makes a ragged chunk illegal for a depth-limited
     streaming CB, so the chunk quantum is uniform `WC` and the block width the chunked buffers span
     is `ceil(C/WC)·WC` — the same pad convention the ragged hidden split already carries. The one
     place a uniform chunk would overrun is a *strided* pack into a row-major-`C` pinned shard, where
     the last chunk is clamped to `C - k·WC` instead.
  3. Two `W = 11008` **TILE** cells still fail, now with an explicit byte accounting and for a
     different reason than chunking addresses: the input and output shards take 1 409 024 B of the
     1 441 792 B budget, leaving 32 768 B against a **fixed** 26 624 B statistics pipeline and a
     chunked working set that bottoms out near 182 KiB (`26 624 + 4096·(⌈344/WC⌉ + WC)`, optimum
     `WC ≈ 19`). No chunk size closes a 5.5× gap. The lever that would is collapsing the degenerate
     `G == 1` combine — at `w_group_size = 1` the gather is a self-write and the multicast a local
     copy, so four of those seven fp32 buffers are copies of one another — which is a combine-topology
     change on the much-travelled interleaved R1 path, not a chunking one. Recorded as a finding.
  4. Known bound, no cell needs it today: on the ROW_MAJOR legs `cb_input_tiles` stays `O(R·C)`, so a
     ROW_MAJOR shard whose tilized block alone overruns L1 is still out of reach. The true two-pass R3
     (re-stride each chunk from the resident shard once per pass) would make it `O(WC)` at the cost of
     the block residency.
- **Tests added**: `test_rms_norm_sharded.py` grows to 39 cases —
  `test_rms_norm_height_shard_wide_w` (3 wide shapes × TILE/ROW_MAJOR: both layouts are covered
  because they chunk *different* buffers) and `test_rms_norm_hidden_chunking_is_a_live_knob`, a
  structural check that a narrow geometry keeps the single-chunk schedule while a wide HEIGHT shard
  comes back with a strictly smaller chunk — invisible to any numerical check, exactly like the
  zero-copy assertion next to it.

## Refinement 3 — Speed up the perf-flagged wide interleaved decode profile
- **Date**: 2026-08-12
- **What was done**: a perf-only phase (no SUPPORTED change) on `(1,1,32,7168)` INTERLEAVED at its
  pinned config (bf16 / TILE / gamma bf16 TILE / `fp32_dest_acc_en=False` / HiFi2). Every step was
  driven by a temporary `DeviceZoneScopedN` timeline (removed before the final commit; the numbers
  are recorded in the kernels' comments), which showed the op is a **latency chain**, not a
  bandwidth wall: the input block only reached compute at 5.9 µs of a 12.5 µs op.
  Three levers, in the order they were measured:
  1. **`cb_zero_tile` off the reader's critical path.** The fp32 zero tile — the identity B operand
     of the combine's `Add` accumulation, read by combine LEADERS and nobody else — was filled by a
     4096-byte scalar store loop in *every* core's reader, *ahead* of the input block: 2363 ns of
     pure critical path on all 22 cores. It moved to the leader's **writer**, whose BRISC idles ~6 µs
     waiting for the first partial. Still exactly one producer kernel per CB, just a different one.
  2. **Gamma read behind the input block.** Gamma is first consumed by the apply pass, which cannot
     start until the combine has returned `rstd` — microseconds after the input block, which gates
     everything. Moving it behind the first block read removed another 1447 ns from the critical
     path and the read is now fully hidden under the combine round. **Gated on a TILE gamma**: see
     Issues.
     After 1+2: 12467 → 9721 ns.
  3. **The scheme-change: a two-stage grid combine** (`tensix_all_reduce`'s `two_stage_grid_reduce`)
     replacing flat root-gather. The combine is now ALWAYS a two-level tree (`_tree_for_box`):
     level 1 folds one grid ROW of the group on that row's leader, level 2 folds the row totals on
     the root; `stage2_span == 1` is the degenerate flat gather and is the Phase 0 code verbatim,
     which every group that is not a fully populated multi-row rectangle keeps (a shard grid with
     filler cores, any 1-D group, `G == 1`, and `nx == 1` where level 1 would be a self-write). Two
     new CBs (`cb_stat_gather2`, `cb_branch_sum`) exist only on that path, and `cb_stat_gather`
     shrinks from `R·G` to `R·nx`, so the tree is a net **L1 saving** (`R·(nx+ny+1)` against
     `R·nx·ny`) and `MAX_GATHER_TILES` now caps `R·max(S1,S2)`.
  4. **`MAX_W_GROUP_SIZE` 32 → 0** (the knob stays live). The cap existed only because the flat
     root's combine was O(G) on one core; with the tree, the full-grid group is the *fastest* pick
     instead of 1.5× the slowest. This also removes the idle-core artefact the verifier flagged on
     the cap's unmeasured neighbours (44 of 110 cores active at `tensor_row_tiles = 2`).
- **Measured (device kernel ns, blackhole_p150b, pinned perf config)**:
  decode `(1,1,32,W)` — 1024 **9101 → 6882** (1.32×), 2304 **9730 → 7299** (1.33×),
  5120 **11219 → 8350** (1.34×), **7168 12467 → 8987 (1.39×)**, against the phase's goal of
  `achievable_ns/7 ≈ 14894 ns`, now cleared by 1.66×.
  prefill `(1,1,8192,W)` — 103076 → 99559, 220005 → 212595, 425343 → 425990, 591707 → 592081, i.e.
  unchanged to within the 2–3 % noise band (the cap never bound there).
  Component numbers: the root's combine chain 3630 → 1360 ns; the same `G = 110` geometry that cost
  19133 ns under the flat combine costs 9210 ns under the tree.
- **Accuracy achieved**: unchanged — the tree sums the same partials in a different association
  order. `-m perf` 13/13 (soft `pcc_threshold = 0.9995` holds), `-m pad_poison` 24/24
  (PCC ≥ 0.99998, so the mask-before-square identity and the true-`W` divisor survive the new
  combine), the acceptance suite at its own per-dtype PCC gates 82/82.
- **Golden test progress**: targeted slices (the harness re-runs the full suite). `-m perf` 13/13,
  `-m pad_poison` 24/24, cartesian `1x1x64x128` 165 passed / 39 xfailed / 0 xpass-drift,
  `test_op_loose` 381 passed / 3 failed — the 3 are exactly the cells Refinement 2b recorded as
  remaining (`1x1x160x11008` + `1x224x11008` TILE HEIGHT, `13x777x1023` WIDTH), unchanged.
- **Issues encountered**:
  1. **The gamma reorder is illegal for a ROW_MAJOR gamma** — PCC ~ 0 on every RM-gamma cell (132
     unit failures). A RM gamma has two consumers that both run before the block loop can: compute
     tilizes the stick into `cb_gamma_tiles` up front, and under `ALIAS_GAMMA_RM` the staging buffer
     IS `cb_input_rm`, whose alias is justified precisely by "gamma dies before the input's first
     push". Reordering made compute tilize input sticks as gamma. Fixed by gating the reorder on
     `!IS_RM_GAMMA`; the RM path keeps the old order byte-for-byte. Found by bisecting the two WIP
     commits against the unit suite (no_gamma RM passed, gamma RM failed — which named the cause).
  2. **The tree must not be taken when `nx == 1`** (a vertical line of cores): level 1 would be a
     self-write, buying nothing and paying a full extra hop. Caught by the `G = 5` (1×5) point of the
     cap sweep.
  3. **A leader that is not the root still runs a combine chain**, so `cb_zero_tile` is filled on
     every LEADER, not only the root — the first version filled it on the root alone and would have
     hung every non-root leader on `cb_wait_front(cb_zero_tile)`.
- **What is left (a finding, not a queued task)**: the decode profile is now a latency chain whose
  two largest single items are both hop/latency rather than arithmetic — the level-2 rendezvous
  (~1.1 µs for one 4 KiB tile + a semaphore) and the root's finalize chain
  (`CopyTile → MulUnary → AddUnary → Rsqrt` on ONE fp32 tile, ~1.3 µs to produce 32 useful numbers,
  since a REDUCE_ROW result is column-0-valid). The next levers would be a narrower stat payload
  (bf16 `cb_stat_gather`, design lamp P6) or broadcasting the SUM so every core finalizes in
  parallel; both are new schemes with their own precision / serialization questions. See the
  `**Outcome**` line in `op_requirements.md`.
- **Tests added**: `test_rms_norm_perf.py` grows two structural cases —
  `test_rms_norm_combine_tree_shape` (the `_tree_for_box` topology table, including the flat
  fallbacks) and `test_rms_norm_combine_tree_is_selected` (a decode shape must come back with a
  two-stage writer CT block and a prefill shape with the flat one). Structural because the two
  combines produce identical numbers, so no value check can tell them apart — the same argument as
  the zero-copy shard assertion and the chunking knob. The harness itself now runs the **exact**
  pinned perf config (`fp32_dest_acc_en=False`), which Refinement 1 unblocked and which it was still
  proxying at `True`.

## Refinement 4 — Speed up the prefill (bandwidth-bound) profiles

- **Date**: 2026-08-12
- **What was done**: one lever landed — a **critical-path admissibility band** on the work split
  (`_admissible_by_balance` in `rms_norm_program_descriptor.py`) — plus two named knobs re-measured
  and kept at their byte-identical defaults. No kernel *logic* changed; the diff is the host
  selection function, two new host constants, and a small DRY fix.
  - **The ablation came first, and it redirected the phase.** The heading is named
    "bandwidth-bound", so I classified the bound before touching a knob, by stubbing each payload
    while keeping every barrier and CB handshake (the three `RMSN_ABLATE_*` switches, left in the
    kernels at `0`, i.e. compile-time inert):

    | (1,1,8192,W) | W=1024 | W=2304 | W=5120 | W=7168 |
    |---|---|---|---|---|
    | baseline | 100047 | 218643 | 425780 | 588950 |
    | input read stubbed | 71803 | 153281 | 312564 | 439221 |
    | output write stubbed | 74229 | 157832 | 287079 | 421081 |
    | both DRAM legs stubbed (compute only) | 30508 | 64558 | 99126 | 134253 |
    | + apply math stubbed (floor) | 21390 | 45241 | 62428 | 83047 |

    `baseline − compute_only` gives **483 / 490 / 514 / 517 GB/s over 2N bytes**: the DRAM stream is
    already at peak, so there is no bandwidth to recover and neither named bandwidth knob can help.
    What *is* recoverable: **per-core compute is 23–30 % of the wall and essentially ADDITIVE with
    the DRAM stream** (near-zero overlap), so the wall is the *busiest core's* tile count. The
    single-leg ablations are also **super-additive** (149.7 + 167.9 µs < 454.7 µs at W=7168) — the
    read and write streams contend with each other, which is why removing either alone under-counts.
  - **The Goal's "tile-row imbalance" pointer was the right target for the wrong reason.** I tested
    it directly first, with no code change, by profiling perfectly-balanced row twins of each perf
    shape (7040 and 10560 rows at W=1024 — 220 and 330 tile-rows over 110 groups, exactly 2 and 3
    each — and 8800 rows at W=7168): 328.6 / 361.4 / 403.6 GB/s against the *imbalanced* 336.2 /
    394.5. Balance alone moves **nothing** as a bandwidth effect. It is a **critical-path** effect,
    which the ablation is what explains.
  - **The lever.** `row` splits in whole TILE-ROWS, so 256 tile-rows over 110 row-groups is 3-vs-2
    and the critical core carries 1.29× the mean. Splitting `hidden` as well re-quantises the row
    split finely (`G = 2` ⇒ 55 groups ⇒ 5-vs-4 ⇒ 1.07×) at the price of one combine round per block
    and a narrower per-core DRAM run. `_admissible_by_balance` keeps, among the candidates that tie
    on occupancy, only those whose `max_tiles_per_core = core_row_tiles · C` is within
    `BALANCE_SLACK_PCT = 15 %` of the best — and only among those with `C ≥ MIN_CORE_W_TILES = 16`.
    The Phase 0 keys (`−G`, then `R`) choose among the survivors, unchanged.
  - **Why a slack BAND and not a score key** (this is the whole design of the lever): minimising
    `max_tiles_per_core` outright picks the *widest* group and loses badly. Measured by forcing `G`
    with the new `MIN_W_GROUP_SIZE` sweep handle:

    | shape | G=1 | G=2 | G=5 | G=10 | G=11 |
    |---|---|---|---|---|---|
    | (1,1,8192,1024) | 99123 | **91316** | 99687 | 103695 | 123542 |
    | (1,1,8192,2304) | 214959 | **196059** | 208107 | 205636 | 214383 |
    | (1,1,8192,5120) | n/a (L1) | 423343 | 437224 | 426409 | 415553 |
    | (1,1,8192,7168) | n/a (L1) | 591909 | 588487 | 600031 | 584071 |

    `G = 11` has the *shortest* critical path at W=1024 (24 tile-rows × C=3 = 72 tiles, against
    `G = 2`'s 80) and is **25 % slower**, because those 8 tiles are bought with 5 combine rounds on
    an 11-core group and a 6 KiB per-tile-row DRAM run. `MIN_CORE_W_TILES` is that second cost read
    off the geometry: `C` *is* the reader's per-tile-row DRAM run and the writer's per-barrier batch,
    and `double_buffer`'s measured plateau is 4–8 tiles. The admissible window for the slack is
    [11.2 %, 20 %) (below 11.2 % only `G=11` survives at W=1024; at 20 % `G=1` re-enters at W=2304
    and wins on `−G` again) — 15 % is its midpoint.
  - **Two knobs re-measured and kept at their defaults, as measured nulls.** `MIN_PIPELINE_BLOCKS`
    1/2/3/4, now at `fp32_dest_acc_en=False` *and* at the new geometry (the verifier notes asked for
    exactly this re-measurement): 92796/91009/94927/92082 · 198618/197725/195813/196209 ·
    423492/424718/419275/424435 · 593148/595673/589124/587184 — flat, and now for the *measured*
    reason (the read is not the un-overlapped stage; the compute is).
    `(input_cb_depth, output_cb_depth)` over {(2,2),(2,3),(2,4),(3,2),(3,3),(4,4)}: every shape's
    best is within 1.8 % of (2,2) and the winner differs per shape — noise, not signal. The writer's
    TILE drain already batches `C = 16…112` tiles per barrier, an order above the plateau, so the
    notes' "batch several tile-rows behind one barrier" has nothing left to buy here.
  - **DRY fix the sweep exposed**: `_DEPTH_LADDER` was a module constant built at import, so
    `INPUT_CB_DEPTH`'s import-time value was frozen into the residency solve's *default argument*.
    Re-tuning the depth would then have been **allocated without being solved for** (a silent L1
    overrun). It is now `_depth_ladder()`, rebuilt per call, and the two solve entry points resolve
    `depths=None` internally — so both depths are genuinely live knobs.
- **Accuracy achieved**: unchanged and unaffected — the band changes only which `(G, C, R)` a shape
  is computed with, and both regimes were already exercised. `test_rms_norm_perf`'s PCC gate (0.995)
  holds on all 8 perf shapes; golden `-m perf` 13/13, `-m pad_poison` 24/24 (so the ragged-hidden
  mask and the true-`W` divisor survive the geometry change), the `1x1x64x128` cartesian slice
  165/165 with 39 xfailed and no xpass drift.
- **Golden test progress**: `test_op_loose` **381/384** — byte-for-byte the same 3 failures
  Refinement 2b recorded and Refinement 3 confirmed (two `W = 11008` TILE HEIGHT cells +
  `13x777x1023` WIDTH, all three the documented capacity limit), so no regression. Unit directory
  green (266 + sharded/precision + 114 perf-file cases).
- **Issues encountered**: none of substance. Two free failures: the first depth-sweep test
  monkeypatched `_DEPTH_LADDER` after I had replaced it with a function (`AttributeError`), and the
  first ablation runs collected only one case each because `run_safe_pytest.sh` appends `-x` and an
  ablated kernel fails its PCC check — `--run-all` fixes it.
- **What is left (a finding, not a queued task)**: the wall is now
  `bytes/500 GB/s + per-core compute`, and perfect DRAM/compute overlap would be worth another
  1.25–1.49×. The compute term is three block-wide FPU passes — `sumsq` (`x·x`), `scale`
  (`x·rstd`), `gamma` (`normed·γ`) — plus `cb_normed`'s whole-block round trip between the last
  two. The next lever is design lamp **P4**: fuse `scale_block` and `gamma_block` into one DEST
  window, deleting a pass *and* `cb_normed`. I did not take it: `DestReuseBinary` has **no
  broadcast-dim parameter**, so gamma would first have to be expanded from its row-0-valid form to
  full tiles (once per kernel, `C` tile-ops against `rows·C` saved per block ≈ 3 % of the wall), and
  dest-reuse routes DEST through a Src register at bf16 — exactly `cb_normed`'s format for a bf16
  input, but a real precision loss for `float32`, so the path needs a dtype gate. `compute_fusion`
  also measured FPU-consumer dest-reuse at 0.94× / 0.82× *isolated*, so a ~3 % predicted win
  against a measured-negative primitive was not worth the remaining budget. See the `**Outcome**`
  line in `op_requirements.md`.
- **Tests added**: `test_rms_norm_perf.py` grows four sweeps/guards, all reusing the existing
  `test_rms_norm_perf` body — `test_rms_norm_perf_row_balance` (perfectly-balanced row twins, the
  measurement that refuted the imbalance-as-bandwidth hypothesis),
  `test_rms_norm_perf_wgroup_min` (the `MIN_W_GROUP_SIZE` sweep that produced the G table above),
  `test_rms_norm_perf_cb_depths` (the `(input, output)` depth co-tune), and
  `test_rms_norm_perf_balance_collateral` (the two non-perf shapes whose geometry the band changes,
  measured `phase0` vs `banded` in one run: `(3,1,736,5119)` 155493 → 135613, `(1,1,4096,4096)`
  195524 → 184774 — both faster). Kernel-side, the three `RMSN_ABLATE_*` compile-time switches are
  left in place at `0`: they are inert, and they are the harness that produced the table above.
