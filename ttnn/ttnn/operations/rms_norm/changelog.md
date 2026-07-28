# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1350 MHz

- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  Generic-op (`ProgramDescriptor`) RMSNorm over the last dimension. The independent tile-row axis
  is split across the whole compute grid (`ttnn.split_work_to_cores(..., row_wise=True)`); each core
  owns whole rows and reduces the dependent `W` axis sequentially in-core over `NW` coarse chunks,
  holding `x` (and `gamma`) resident in L1 whenever a size predicate says they fit, with a bounded
  streaming fallback. TILE and ROW_MAJOR are both native (in-kernel `tilize`/`untilize`), and
  non-tile-aligned `H` and `W` are handled natively by a masked reduce plus a valid-stick writer —
  no host-side `to_layout` / `pad` / `slice` anywhere.

- **SUPPORTED at Phase 0**:
  - `dtype = [float32, bfloat16]`
  - `fp32_dest_acc_en = [True]`
  - `layout = [TILE, ROW_MAJOR]`
  - `alignment = [tile_aligned, w_non_aligned, h_non_aligned]`
  - `rank = [2, 3, 4]`
  - `gamma_mode = [gamma, no_gamma]`
  - `gamma_dtype = [float32, bfloat16, "none"]`
  - `gamma_layout = [TILE, ROW_MAJOR, "none"]`
  - `memory_layout = [INTERLEAVED]`
  - `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]` (permanent op-side refusal)

- **Accuracy achieved** (measured on 16 cells = 4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR} via
  `test_rms_norm_precision_baseline.py`; gamma present, `epsilon=1e-6`, HiFi4 + `fp32_dest_acc_en=True`):
  - `bfloat16`: PCC ≥ 0.995 on every cell, max_abs_err ≤ 4.8e-02, mean_abs_err ≈ 1.2e-03,
    relative RMS err 2.34e-03 … 2.46e-03 (gate 4.0e-02 → ~17× headroom)
  - `float32`: PCC ≥ 0.999 on every cell, max_abs_err ≤ 2.2e-02, mean_abs_err 4.1e-04 … 1.0e-03,
    relative RMS err 8.28e-04 … 1.84e-03 (gate 2.0e-02 → ~13× headroom)
  - got/true ratio spread: median 0.9984 … 1.0003, p5/p95 within ±0.5 %. The mild fp32 low bias
    was traced (`probes/probe_001.py`) to FPU SrcA/SrcB 19-bit truncation on the two multiplies,
    **not** a scale/structural bug — the reduce itself is unbiased and the offset is smaller than
    the random spread.
  - Error does not grow with `W`: the widest and only chunked-reduce cell `(1,1,32,4096)` (`NW = 5`)
    is the most accurate fp32 row.

- **Measured device performance** (bf16 / HiFi4 / `fp32_dest_acc_en=True`, DEVICE KERNEL DURATION ns,
  from `test_rms_norm_perf.py`; "first commit" = the correctness-only baseline at 56cbc4f8):

  | shape | cores | first commit | Phase 0 | speedup |
  |---|---:|---:|---:|---:|
  | (1,1,32,1024) | 1 | 13 559 | 12 172 | 1.11× |
  | (1,1,32,2304) | 1 | 25 404 | 21 382 | 1.19× |
  | (1,1,32,5120) | 1 | 51 434 | 42 439 | 1.21× |
  | (1,1,32,7168) | 1 | 63 313 | 57 474 | 1.10× |
  | (1,1,8192,1024) | 110 | 103 238 | 102 114 | 1.01× |
  | (1,1,8192,2304) | 110 | 214 633 | 215 181 | 1.00× |
  | (1,1,8192,5120) | 110 | 482 655 | 483 074 | 1.00× |
  | (1,1,8192,7168) | 110 | 835 407 | 831 973 | 1.00× |

  Prefill is reader-bound (NCRISC 90–99 % of kernel time) — at the interleaved-DRAM read floor.
  Decode is latency-bound on **1 of 110 cores** (`ht_total == 1`), which is the standing gap and the
  subject of Refinements 2 + 3.

- **Golden suite at Phase 0**: **755 / 755** supported cells passing
  (`supported_pass` 755, `xfail_expected` 5 768, `invalid_skipped` 33 900, `no_axes_found` 15 uncharged;
  `supported_fail` = `xpass_drift` = `xfail_wrong_mode` = `supported_marked_xfail` = `invalid_unexpected` = **0**).
  Runner line: `770/40438 passed (0 failed, 0 errors, 33900 skipped, 0 hangs)`.
  Per-core CB totals 408–920 KB against `L1_CB_BUDGET_BYTES = 1 100 000` — no OOM anywhere in the suite.

- **Issues encountered**:
  - *Found and fixed by the implementer before verification*: `cb_scaler`'s data format was a fixed
    `Float16_b` per `op_design.md` R4, but `reduce_accumulate_via_add`'s `fold_partial_last` reads the
    partial-W mask through srcB **without** reconfiguring srcB (reduce entry already programmed it at
    the input format), so an fp32 input reinterpreted the bf16 mask as fp32 — 82 golden failures, all
    `float32 × w_non_aligned`. Fixed by deriving `cb_scaler`'s format from the input dtype; pinned by
    `test_rms_norm_debug.py`'s hand-calculable all-ones cases.
  - *Found and fixed during this verification pass* (three single-source / DRY repairs in
    `rms_norm_program_descriptor.py`, no math change):
    1. The reader's and writer's 18-entry compile-time-arg blocks were two separate literal
       expressions building the identical list, while both kernels hard-code the same CT indices and
       `TensorAccessorArgs<18>`. Now built once as `dataflow_ct_args`, with an assert pinning the
       block length so host and kernels cannot desync.
    2. `X_RESIDENT_DEPTH` was raised to 2 on the ROW_MAJOR path, where `cb_input_tiles` is produced
       *and* consumed by the compute RISC — a second strip can never be filled ahead, so it only
       spent L1 (and could evict `GAMMA_RESIDENT`). Now gated on `not is_rm`.
    3. The compute grid was derived twice (once to size the blocking, once inside
       `_core_assignment`), giving two sources for the quantity `grid_full` keys off. `_core_assignment`
       now takes the caller's grid.
  - *Reported, not edited* (`eval/golden_tests/rms_norm/feature_spec.py` is the golden-test author's):
    three INVALID entries couple the activation's `layout`/`memory_layout` with the **gamma tensor's**
    `gamma_layout` (cross-tensor coupling — 1 260 cells skipped that should be xfailing), and two
    `{bfloat8_b, *_non_aligned}` entries encode "out of scope for now" rather than impossibility
    (720 cells; they belong in the op's `EXCLUSIONS`). Both shrink the exercised universe, so nothing
    is over-claimed. See `verification_report.md` → "INVALID audit".
  - *Design-doc drift noted, not blocking*: `op_design.md` R4 (`cb_scaler` format) and §6
    (`cb_output_rm` page size) no longer match the shipped, helper-mandated behaviour; the
    implementation is correct in both cases.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py` (89 — immutable acceptance spec)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_debug.py` (31 — deterministic bug repros)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py` (9 — on-device perf harness + blocking report)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` (16 — **new this pass**:
    PCC + max/mean abs error + relative RMS + got/true ratio-spread scale-bug tripwire)
  - `tests/ttnn/unit_tests/operations/rms_norm/probes/probe_001.py` (**new this pass** — the fp32
    bias attribution probe referenced in the precision baseline)
  - Total: **145 passed**, `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/rms_norm/`

- **Artifacts**: `verification_report.md`, `verifier_report.json` (trimmed: summary + loud-category
  node lists + xfail axis histogram), `op_requirements.md` (5 refinements: 3 generality + 2 measured
  perf, at the 2:1 cadence), this changelog.

## Refinement 1 — Numerical configurability expansion (unlocks every perf target)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1350 MHz

- **What was done**: opened the full float surface `{bfloat16, float32, bfloat8_b} ×
  {fp32_dest_acc_en True, False}`. **Zero kernel changes** — the compute/reader/writer triple was
  already helper-based and dtype-agnostic (runtime `get_tile_size`, host-computed byte counts), so
  the skill's pass condition held and every edit is in the op file + program descriptor.

  - `SUPPORTED`: `dtype += bfloat8_b`, `gamma_dtype += bfloat8_b`, `fp32_dest_acc_en = [True, False]`.
  - `default_compute_kernel_config()` **unchanged** (`fp32_dest_acc_en=True`), so
    `axes.py:40-43` and `test_rms_norm_default_config_matches_factory` still hold.
  - The existing `{float32, fp32_dest_acc_en=False}` `EXCLUSIONS` entry became *reachable* — kept,
    as the queue requires; it is now exercised (64 skips in the precision matrix).
  - `_row_elem_bytes()`: `Tensor.element_size()` **raises** for block-float dtypes (no per-datum
    width). It is consumed only on the ROW_MAJOR stick path, which a block-quantized tensor can
    never take, so it returns a documented structurally-unused placeholder rather than letting the
    exception escape a branch that discards it.
  - `x_squared_dtype`: the reduce programs srcA/srcB from its **input CB** (`cb_x_squared`), so that
    — not the input tensor's dtype — is the format `fold_partial_last` reads the partial-W mask at.
    Phase 0 conflated the two because they were always equal. Block-float inputs now square into a
    **bfloat16** `cb_x_squared`; `cb_scaler` derives from it, single-source.

- **Accuracy achieved** (`precision_matrix_results.md`, 320 cells = 8 shapes × 3 dtypes ×
  4 math_fidelity × 2 fp32_dest_acc_en × 2 distributions; gate PCC ≥ 0.99):
  - **320 / 320 pass**; worst PCC anywhere is **0.999325** (bfloat8_b / LoFi / bf16 DEST), ~75×
    inside the gate. 64 cells skipped via the op's own `EXCLUSIONS`.
  - `bfloat16`: PCC ≥ 0.999980 / rel-RMS ≤ 6.2e-03 at HiFi4 with `fp32_dest_acc_en=False`
    (vs 0.999992 / 2.5e-03 with it on) — the new DEST mode costs ~2.5× rel-RMS at HiFi4 and
    **nothing** at HiFi2/LoFi, where fidelity already dominates.
  - `bfloat8_b`: PCC ≥ 0.99979 / rel-RMS ≤ 2.2e-02 at HiFi4 (both DEST modes), flat across
    fidelity — its error is input/output block-float quantization, not the compute pipeline.
  - `float32`: unchanged from Phase 0 (PCC ≥ 0.999999 / rel-RMS 1.6e-03 at HiFi4).
  - **Accuracy watch item resolved**: the 8 interleaved perf loose cases (bf16 / HiFi2 /
    `fp32_dest_acc_en=False`, W up to 7168, `NW = 7`) pass their tighter `pcc_threshold = 0.9995`
    soft gate. The pairwise-add `AccumulateViaAdd` datapath was the right one; no SFPU finalize
    needed, no gate widened.

- **Golden test progress**: the refinement's gating criterion is met — **all 8 interleaved
  perf-flagged loose cases now run as `supported_pass`** (they were 8/8 xfail before, because every
  one pins `fp32_dest_acc_en=False`). Nothing downstream is blocked on measurement any more. The
  remaining 8 loose xfails are the sharded schemes (Refinements 2 / 4). Targeted cartesian slice
  over `1x1x64x128` (aligned) + `1x1x64x17` (W non-aligned) + `1x1x17x64` (H non-aligned):
  **120 passed, 0 failed, 0 XPASS-drift**, with bfloat8_b cells passing at every `gamma_dtype`
  including bf8 gamma. Unit suite: **456 passed** (136 prior + 320 new matrix cells) + 25 element-count
  cells, no regression.

- **Issues encountered**:
  1. **A silently-wrong bfloat8_b cell that PCC could not see.** With a `Bfp8_b` reduce datapath the
     `Float16_b` partial-W mask decoded as all-zeros (a bf8 tile's leading bytes are the
     shared-exponent header), so the final reduce-dim tile contributed **nothing**: an all-ones
     `W=49` row summed to **32**, not 49. Random-data PCC was **0.9998** and the golden gate
     (0.99 / 0.10) passed anyway — because dropping elements only *rescales* each row and PCC is
     scale-invariant. Caught by a deterministic all-ones probe (`probes/probe_005.py`), fixed by the
     `x_squared_dtype` rule above, and pinned by
     `test_partial_w_reduce_counts_every_element`, which inverts the kernel's own output back into
     the element count it actually summed. Post-fix the recovered sum is exact at
     `W = 33 / 49 / 63 / 100 / 4097`. **No `EXCLUSIONS` entry was needed for bf8 non-aligned** —
     both buckets are genuinely correct now (they remain `feature_spec.INVALID`-skipped, so this is
     honesty rather than new golden coverage).
  2. **`UnpackToDestFp32` is unavailable on the generic-op path** — investigated and reverted, see
     "Measured null" below.

- **Measured null — the `UnpackToDestFp32` upside the queue flagged**: the verifier suggested
  tagging the fp32 *inputs* to the two FPU multiplies. That is forbidden outright by the tag's
  exclusivity rule (a tagged CB can never be an FPU operand), so the only legal candidates here are
  `cb_partials` (the `AccumulateViaAdd` running sum, reloaded via `copy_tile`) and `cb_rms_sum`
  (read only by `CopyTile` → SFPU). Those looked genuinely promising: the reduce path's accuracy
  degrades exactly with the reload count — **12.02 effective mantissa bits at `NW=2`, 11.11 at
  `NW=8`, 9.89 at `NW=16`** (`probes/probe_007/008.py`) — the signature of per-reload TF32
  truncation compounding. The tag was implemented and correctly reached the descriptor (CBs 25/26
  tagged, vector length 64, verified by readback). It produced **bitwise-identical output**.
  Falsification test (`probes/probe_010.py`, plus three *separate processes* on a fresh shape to
  rule out the program cache): tagging `cb_rms_recip` — an **FPU srcB operand**, which by the
  exclusivity rule *must* corrupt the result if the tag were honoured — is also bitwise identical
  (`md5=a7c265c57d4719bb` for off / legal / illegal alike). Conclusion:
  `ComputeConfigDescriptor.unpack_to_dest_mode` is **inert on the `ttnn.generic_op` ProgramDescriptor
  path**. Reverted rather than left in as dead code, because the wrapper had to rebuild the caller's
  config field-by-field and would silently drop any future `ComputeConfigDescriptor` field — a real
  latent hazard bought for a measured-zero benefit. fp32 retains ≥13× tolerance headroom regardless.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` (**new** — 320
    matrix cells + 25 partial-W element-count cells; imports the op's `EXCLUSIONS` rather than
    copying it, so the skip list cannot drift)
  - `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md` (**new** — the §10
    results file)
  - `probes/probe_003..013.py` (bf8 bring-up, the deterministic mask probe that found the bug, the
    fp32 error attribution, and the `UnpackToDestFp32` falsification)

---

## Refinement 2 — Cross-core `W`-split: partial-sum combine + `1/rms` multicast

- **Date**: 2026-07-28

- **What was done**: built op_design.md's Lamp L1 / §4.2 — the **dependent** `W` axis is now split
  across cores. Each core reduces its own `W`-slice to a raw `Σx²`, its **writer** unicasts that tile
  into slot `slot` of the group root's `cb_group_partials` and bumps the root's gather counter; the
  root's **reader** waits for all `CW` of them, hands them to compute, which folds them into one
  `mean(x²)` per tile-row with `n_reduced = W` (the grand total); the reader then **multicasts** that
  tile back over the group via `kernel_lib/mcast_pipe.hpp` (`SenderPipe`/`ReceiverPipe` +
  `ttnn.Mcast2D` on the host), and every core finalizes `rsqrt(mean+ε)` and scales its own slice.
  `SUPPORTED["memory_layout"]` gains `WIDTH_SHARDED` and `BLOCK_SHARDED`, consumed **natively**:
  `ttnn.cb_descriptor_from_sharded_tensor` backs `cb_input_tiles`/`cb_output_tiles` on the core's own
  L1 shard, so the reader issues **no input read at all** and the writer **no output write**.

  The delta is deliberately small. `_Blocking` is untouched in shape — the per-core `W` extent is
  simply the axis every knob now derives against, so `WT_CHUNK` / `NW` / `HT_BLOCK` / the residency
  predicates / every CB page count keep their existing single source. The new structure is one class
  (`_Placement`: row-split | W-split | shard-pinned), four fp32 CBs, and a combine block that slots
  between compute phases 3 and 4 exactly as the design promised. Compute phases 1/2/5/6/7 are
  byte-identical; only the *producer* of `cb_rms_sum` changes (reader multicast instead of phase 3).

- **Accuracy achieved**: PCC ≥ 0.9999 on every shape measured. Golden `test_op`: **3258 passed,
  0 failed, 0 XPASS, 3181 xfail** (was 3240 pass before the axis values; the sharded cells are the
  new pass mass). All 18 supported loose cases pass, including all 5 sharded perf-flagged geometries
  and the three interleaved `_WIDE` cases. Worst PCC seen across the sharded probe matrix: 0.999850
  (bfloat8_b BLOCK_SHARDED).

- **Perf** (blackhole_p150b, 110-core grid, bf16 / HiFi4 / fp32-on, DEVICE KERNEL DURATION,
  `RMS_NORM_W_SPLIT=0` vs default, one fresh-cache run each):

      shape             row-split  cores | W-split  cores | speedup
      (1,1,32,1024)        12_271      1 |   6_999     32 |  1.75x
      (1,1,32,2304)        21_429      1 |   7_663     36 |  2.80x
      (1,1,32,5120)        42_484      1 |   9_611     40 |  4.42x
      (1,1,32,7168)        57_608      1 |  11_279     56 |  5.11x
      (1,1,8192,*)                   110 |             110 |  0.99-1.01x (noise band)

  The decode column was 1 busy core of 110; it is now 32-56. Prefill is not engaged (its row axis
  already fills the grid) and is unchanged.

- **Golden test progress**: 3258/3258 supported cross-product cells + 18/18 supported loose cases.
  `WIDTH_SHARDED` and `BLOCK_SHARDED` are in SUPPORTED; `HEIGHT_SHARDED` remains the only
  `memory_layout` xfail (Refinement 4).

- **Issues encountered** — three real bugs, each invisible to at least one obvious check:

  1. **The combine must gather the RAW accumulator, not the reduced tile.** `AccumulateViaAdd`'s
     finalize writes the row sum into column 0 and leaves the surviving `x²` lanes in columns 1..31,
     so a second `REDUCE_ROW` over *finalized* partials double-counts them. Measured with DEVICE_PRINT:
     an all-ones `W=64` produced `mean(x²) = 8.75` instead of `1.0` — and **PCC scored 0.9999**,
     because rescaling every row by one factor is invisible to a scale-invariant metric. Fixed by
     running every chunk with `Accumulate::at` (never `at_last`) so `cb_partials` keeps the raw
     elementwise accumulator, and letting the root's `reduce_mean` do the one within-tile fold. The
     combine is then *literally* the local chunk-accumulate, done across cores instead of across
     chunks. Pinned by `test_cross_core_combine_counts_every_element` (absolute, not correlational).

  2. **A multicast rectangle is a VIRTUAL rectangle, and the logical grid is not virtually
     contiguous.** On this box logical x 0..6 → virtual 1..7 but logical x 7..10 → virtual 10..13, so
     virtual columns 8-9 are **not worker cores**. A group whose bounding box straddles that seam
     multicasts into non-worker endpoints — a hard device hang on the WIDTH_SHARDED cells whose shard
     grid is wider than one run. Fixed by splitting each group's broadcast into **one multicast family
     per virtually-contiguous run** (`_virtual_x_runs` / `_split_rect_by_runs`, two `Mcast2D` per
     group, two `McastArgs` in the reader) and by confining the *interleaved* split's group
     rectangles to a single run. Pinned by
     `test_every_combine_group_is_one_virtual_rectangle_per_family`.

  3. **`noc_semaphore_inc` is a non-posted atomic and needs draining.** The gather leg's counter bump
     left an outstanding atomic at kernel exit; on the sharded path there is no trailing output-write
     barrier to absorb it, so the core's NoC transaction counters never balanced and dispatch
     completion stalled. Fixed with a `noc_async_atomic_barrier()` at the end of the reader and the
     writer under `W_SPLIT`. **Only `--dev` caught this** — plain mode passed the same cells.

  Two geometry cases `auto_shard_config` really emits also needed handling: a **ragged** shard core
  grid (full rows + a partial last row), padded up to its bounding box with zero-work *filler* cores
  so the broadcast rectangle stays legal and dense-ack accounting stays right; and a shard grid that
  **over-covers W** (86 cores × 3 tiles vs 256 real tiles), whose trailing padding tiles are
  uninitialized L1 — the reader zeroes them so they contribute nothing to `Σx²`.

  Deferred to EXCLUSIONS with a note: `{ROW_MAJOR, WIDTH_SHARDED}` and `{ROW_MAJOR, BLOCK_SHARDED}`.
  `eval.sharding`'s ROW_MAJOR granule is `(1 row, L1_align/elem_bytes columns)`, so an RM shard is
  e.g. `[1, 128]` or `[64, 8]` — a single stick, or 8 of a tile's 32 columns. The kernels tilize a
  32-stick × 32-column block in place, which a core physically does not hold under those shards;
  forming one needs a cross-core *stick* gather, a different scheme from this refinement's W-split.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_wsplit.py` (**new**, 29 cases) — the
    split engages on wide/few-row shapes and does *not* on prefill or narrow W; every group's
    broadcast sub-rectangle is virtually contiguous; the absolute element-count assertion for the
    combine; the pinned + auto sharded geometries incl. ragged / padding-tile / degenerate-1-core;
    and `test_sharded_input_is_consumed_in_place`, which asserts the input and output CBs are
    **aliased onto the tensor buffers** — an accessor read of a core's own shard passes every
    numerical gate, so this is checked on the descriptor, not on the test colour.
  - `test_rms_norm_perf.py::test_report_blocking` now reports `CW` and the real core count, and the
    module docstring carries the row-split vs W-split A/B table.

---

## Refinement 3 — Speed up the perf-flagged **decode** column

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK
  **1349.98 MHz** — the references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns`
  and no clock scaling applies to any number below.

- **What was done**: a perf-only refinement (no SUPPORTED change) on the four pinned decode
  profiles `(1,1,32,W)`, `W ∈ {1024, 2304, 5120, 7168}`, measured at the config
  `feature_spec` actually pins for them — bf16 / TILE / `fp32_dest_acc_en=False` /
  `math_fidelity=HiFi2` / bf16 TILE gamma — never at the op's default HiFi4 / fp32-on corner.

  **Measure first.** A new `test_rms_norm_perf_decode_pinned` re-baselined the column at that
  config, and a new opt-in ablation (`test_rms_norm_ablate`, `RMS_NORM_ABLATE=combine[,gamma]`)
  attributed the time. The ablation holds the cross-core placement byte-for-byte — same core
  count, same per-core W slice, same DRAM reads and writes, same per-core square/reduce/scale —
  and removes only the named stage, so the delta is that stage's real cost on the critical path:

      shape           cores    full   no-combine   combine   share
      (1,1,32,1024)      32   6_938        3_524     3_414     49%
      (1,1,32,2304)      36   7_555        3_827     3_728     49%
      (1,1,32,5120)      40   9_309        5_152     4_157     45%
      (1,1,32,7168)      56  10_929        5_759     5_170     47%

  The **combine was the bottleneck**, not the data movement (NCRISC 1.0–2.2 µs) and not gamma
  (117–468 ns total, i.e. noise). It also grew ~73 ns per contributor — the signature of one
  core absorbing `CW × 4 KB` of NoC writes and then running `CW` fp32 tile-adds back to back.

  **Lever 1 — the two-stage combine** (`examples/tensix_all_reduce`'s measured
  `two_stage_grid_reduce`, the topology R2 deliberately deferred to a tuning phase). Row members
  gather to their grid-row leader, the row leaders gather to the group root: the serial fan-in
  becomes `cx + cy` instead of `cx · cy`. Deliberately small delta — one CB
  (`cb_group_partials2`), one semaphore, and a `CW1 × CW2` factorization of the existing `CW`.
  The leader's fold is *literally the worker's own chunk-accumulate* (`Accumulate::at`, never
  `at_last`, so the raw elementwise accumulator survives and the root's single finalize cannot
  double-count the surviving x² lanes), and its row sum rides the **same** `cb_partial_out` the
  slice partial did — one producer, one consumer, two sequential pushes, so no CB was added for it.
  Fan-in 32/36/40/56 → **12/12/13/15**; per-core gather L1 174/204/248/312 KB → **94/108/140/148 KB**.

  **Lever 2 — delete the republishing `copy` pass.** When a gather source comes from a *single*
  accumulate call (`NW == 1` workers, and every leader's stage-1 fold) the accumulator is written
  once and never reloaded, so the reduce packs straight into `cb_partial_out` instead of into
  `cb_partials` + a `copy`. That is a whole compute pass (~320 ns fixed, `examples/compute_block_size`)
  off the combine's serial path, twice over on a leader. `NW > 1` keeps the copy — there the
  accumulator genuinely is a compute→compute read-modify-write.

  Both levers are single-source knobs, not inlined constants: `COMBINE_MAX_FLAT_FANIN` selects the
  topology at fixed `CW` (raise it past any reachable group area to get R2's flat root back), and
  the existing `L1_GATHER_BUDGET_BYTES` still caps `CW` — now interpreted through `_gather_tiles`,
  which charges `cx + cy` for a staged rectangle and `cx · cy` for a flat one, so the same budget
  reaches a much wider group once the gather is staged.

- **Measured result** (pinned perf config, DEVICE KERNEL DURATION ns, one fresh-cache run per
  variant; a repeat of the winning configuration reproduced to 0.4–2.3 %, consistent with the
  op's established 1–2 % noise band):

      shape          R2 flat   +two-stage   +no-copy   speedup   ceiling   margin
      (1,1,32,1024)    6_938        5_987      5_709     1.22x     9_149    1.60x inside
      (1,1,32,2304)    7_555        6_513      6_219     1.21x    17_003    2.73x inside
      (1,1,32,5120)    9_309        8_320      7_933     1.17x    75_825    9.56x inside
      (1,1,32,7168)   10_929        9_200      8_917     1.23x    14_894    1.67x inside

  The headline `(1,1,32,7168)` ceiling is `104_259 / minimum_expected_speedup 7.0`; the op now
  delivers **104 259 / 8 917 = 11.7×** against that 7.0× requirement.

  At the op's *default* config the same shapes go 6_999 → 5_762 · 7_663 → 6_318 · 9_611 → 8_273 ·
  11_279 → 9_273 (1.16–1.22×), and **prefill is unmoved** — 101_152 / 219_255 / 479_801 / 831_397 ns
  against R2's 101_922 / 215_814 / 487_581 / 834_799, all inside the noise band. The W-split is
  not engaged at prefill (its row axis already fills the grid), so `CW == 1` and none of this code
  runs there.

- **Accuracy achieved**: `pcc_threshold = 0.9995` holds on all four pinned decode cases (asserted
  by `test_rms_norm_perf_decode_pinned`). Staging changes the *association order* of an exact
  elementwise fp32 add tree, nothing else — worst PCC across the golden slices is unchanged, and
  the all-ones absolute check recovers `mean(x²) = 1.0` exactly on both topologies.

- **Golden test progress**: unchanged by design (perf refinement, no SUPPORTED change).
  `test_op_loose` **18 passed / 1 xfail** (the xfail is HEIGHT_SHARDED — Refinement 4), including
  all five sharded perf geometries; the `(8,4)` and `(7,4)` WIDTH_SHARDED ones are dense
  rectangles and now take the staged combine too. Golden cross-product slices, 0 failed and
  0 XPASS-drift throughout: `1x1x32x8192` 108 passed (wide → staged combine, across TILE/RM ×
  gamma/no-gamma × every dtype × interleaved/WIDTH/BLOCK sharded), `2x1x64x4096` 108, `1x1x17x50`
  72 (both-non-aligned, no split), `4x8x32x47` 72 (W-non-aligned, row-split with the grid full).
  Unit suite **527 passed / 73 skipped**; the W-split suite is green under `--dev` (watcher + NoC
  sanitizer), which R2's changelog flags as non-optional for this scheme — the second gather leg's
  `noc_semaphore_inc` is drained by the existing `noc_async_atomic_barrier()`.

- **Issues encountered**: none — no hang, no numerical failure, no debug cycle. The two structural
  hazards were anticipated from R2's findings rather than hit: (a) a leader must fold with
  `Accumulate::at` and never `at_last`, or the root's finalize double-counts the surviving x² lanes
  exactly as R2 measured (8.75 instead of 1.0) with PCC scoring it 0.9999; and (b) two-stage needs a
  **dense** rectangle — every grid row needs a real core at column `x0` to lead it — so a ragged
  shard grid (auto_shard_config's full rows + partial last row, whose filler cores own no work) and
  any single-row group keep the flat topology by construction.

- **Knob defaults are measured, not assumed**:
  - `COMBINE_MAX_FLAT_FANIN = 24`. The second stage buys back ~73 ns per contributor removed from
    the root but costs one extra gather round (~1.3 µs, fitted from the matched-CW A/B), so it only
    pays above a wide fan-in. The cap is set at the widest flat gather still measured to be
    competitive rather than at the fitted break-even (~28): every group at or below it — notably the
    24-core `(1,1,64,12288)` loose case — keeps R2's flat topology byte-for-byte, and staging engages
    only in the 32–56 range where it is measured to win.
  - `CW` stays at *widest that fits*. Re-swept under the staged topology, that is now optimal
    (`(1,1,32,7168)`: 8 cores 12_016 ns, 32 cores 10_116, widest-56 **8_917**) — the reverse of the
    flat topology, where the same sweep had shown narrowing to be a 1.06–1.17× *win*. That inversion
    is the cleanest evidence the fan-in cost is what got removed. `CW` is in any case already at its
    structural maximum for these shapes: it must divide `Wt` and its rectangle must fit one
    virtually-contiguous column run (≤ 7 wide on this part).

- **Remaining headroom — a finding, not a queued task**. The combine is still the largest single
  item (~2.4–3.4 µs, 30–38 %) but its *shape* changed: it now fits ≈ **2.35 µs fixed + 73 ns ×
  (cw1+cw2)**, i.e. it is dominated by the fixed cost of two semaphore rounds plus the multicast
  handshake, not by fan-in — so a third gather stage would add another ~1.3 µs round and lose. The
  rest is a ~2.7 µs compute floor that is **per-pass-overhead bound**, not tile-work bound (1
  tile/core still costs 3.5 µs across 5–6 helper passes, ≈ 540 ns each). The next lever is therefore
  pass elimination, not topology: fold `AddUnary(eps) → Rsqrt` into the root's stage-2 reduce as a
  `post_reduce_op` so the multicast carries `1/rms` instead of `mean(x²)`, deleting one whole compute
  pass from *every* core on the serial tail after the broadcast (~5 %, and the value becomes
  bit-identical across the group instead of recomputed per core). Not done here because `reduce_mean`
  already occupies the `post_reduce_op` slot with its 1/N multiply, so it means dropping to
  `reduce<SUM>` and re-inlining that normalization — duplicating helper logic on the numerics path —
  for ~5 % on top of a result that already clears its ceiling by 1.67×. Fusing phases 5 and 6 (the
  two broadcast multiplies) into one `eltwise_chain` was investigated and is **not** expressible:
  `DestReuseBinary` combines DEST with a CB tile but carries no `BroadcastDim`, and the gamma
  multiply needs a row broadcast.

- **Tests added**:
  - `test_rms_norm_perf.py::test_rms_norm_perf_decode_pinned` (**new**, 4 cases) — the decode column
    at the pinned perf config, with the tighter `pcc_threshold = 0.9995` soft gate those loose cases
    carry. `perf_compute_kernel_config()` is the single source for that config so a knob A/B can
    never be taken on the wrong datapath.
  - `test_rms_norm_perf.py::test_rms_norm_ablate` (**new**, opt-in via `RMS_NORM_ABLATE`) — the
    stage-peeling ablation harness. Asserts nothing by design (never PCC-gate an ablated kernel) and
    is skipped unless explicitly requested, so a normal suite run never executes it.
  - `test_rms_norm_wsplit.py::test_combine_topologies_agree` (**new**, 4 cases) — an *absolute*
    all-ones check on **both** topologies plus an agreement check between them at matched `CW`.
    Staging is exactly where a premature within-tile fold could creep back one level up, and PCC is
    blind to it (one scale factor per row), so this is the R2 element-count discipline extended to
    the new fan-in tree.
  - `test_rms_norm_perf.py` knob overrides `RMS_NORM_GATHER_BUDGET_KB` / `RMS_NORM_MAX_FLAT_FANIN`,
    so `CW` and the combine topology stay A/B-measurable without editing the op.
  - `probes/probe_026.py`, `probes/probe_027.py` — clock/placement readback and the topology
    resolution check.

---

## Refinement 4 — `HEIGHT_SHARDED` placement (local shard, zero-copy)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1349.98 MHz,
  per-core L1 bank **1 461 504 B** (read from the live device, not assumed)

- **What was done**: op_design Lamp L3 — the Phase-0 row split made *physical*. `SUPPORTED["memory_layout"]`
  gains `HEIGHT_SHARDED`. It really is a knob-turn: **zero kernel changes**, and the whole placement is one
  ~10-line branch in `_placement_sharded` that hands back `cw = 1`, no groups, no semaphores and no multicast,
  with the shard grid as the core assignment and the shard height as `rows_core_max`. Everything downstream
  was already generalized by Refinement 2 and is reused verbatim: `ttnn.cb_descriptor_from_sharded_tensor`
  backs `cb_input_tiles` **and** `cb_output_tiles` on the core's own L1 shards, so the reader issues no input
  read and the writer no output write, and the compute phases are byte-identical to the interleaved resident
  regime. Because each core holds WHOLE rows, the reduce stays entirely local — none of the R2/R3 combine
  machinery is built, allocated or executed.

  Two supporting changes, both single-source:

  1. **`ROW_MAJOR` shards route to the accessor path** (`tile_shard` gate, 3 lines in
     `create_program_descriptor`). "Sharded" in this op means *zero-copy*, which is only meaningful for a TILE
     shard. `eval.sharding`'s RM granule is `(1 row, L1_align/elem_bytes columns)`, so an RM height shard is
     `[1..3, W]` — a handful of sticks. That is **not** this core's block, and no amount of CB placement makes
     it one.
  2. **The L1 budget is now two budgets** (see "Issues encountered" — this is the R2 verifier note coming due).

- **Accuracy achieved**: PCC ≥ 0.999 on every cell measured; PCC = **1.000000** on the fp32 and the
  multi-tile-row shards, 0.999870 worst case (bfloat8_b). The all-ones **absolute** element-count check
  (`test_height_shard_counts_every_element`) recovers `mean(x²) = 1.0` exactly on
  `(1,1,256,512)` / `(1,1,2048,256)` / `(1,1,32,4096)` / `(1,1,17,64)` / `(1,1,32,50)` — chosen over a PCC gate
  because a wrong row→core map, a dropped W-tile or a wrong `n_reduced` only *rescales* rows, which PCC scores
  0.9999 (the R1/R2 lesson).

- **Golden test progress**: `test_op_loose` is **19 passed / 0 xfail** — the `(1,1,256,512)` HEIGHT_SHARDED case
  was the last remaining loose xfail in the suite. Cross-product slice over five shapes spanning all four
  placements (`1x1x256x512`, `1x1x17x64`, `4x8x32x256`, `1x1x32x8192`, `2x512x1024` — TILE/RM × gamma/no-gamma ×
  every dtype × INTERLEAVED/HEIGHT/WIDTH/BLOCK): **536 passed, 0 failed, 0 XPASS**, 228 xfailed.
  `memory_layout` is now complete against TARGET — no value is left queued.

- **Issues encountered** — one real bug, found exactly where Refinement 2's verifier note pointed
  ("make sure the block shrink is not what is paying for the shard"):

  **The resident shard was being charged against the CB budget.** A zero-copy sharded CB is *aliased* onto the
  tensor's own buffer, which the **buffer** allocator already reserved out of the same L1 bank — it is not part
  of the program's CB region. Charging it against `L1_CB_BUDGET_BYTES` (a budget calibrated for
  *program-allocated* CBs on the interleaved path, where tensors live in DRAM) double-counts it, and the
  halve-and-re-derive loop then pays for the shard by shrinking the block. HEIGHT shards are full-W and so the
  largest of the three schemes, which is what made this load-bearing rather than cosmetic:

  - bf16 `(1,1,32,8192)`: `WT_CHUNK` collapsed **32 → 1** (`NW = 256` on a single core) — a 256-pass compute
    loop where 16 passes fit;
  - every fp32 `W = 4096` HEIGHT cell (`(1,1,32,4096)`, `(1,1,128,4096)`, `(2,1,64,4096)`, `(1,32,4096)`,
    `(32,4096)`) was **refused outright** — `AssertionError`, i.e. hard `supported_fail` — by **10 KB**, with
    **361 KB of the 1 461 504 B bank still free**.

  Fixed by budgeting the two quantities separately in `_Blocking._fits`:

      program CBs                  <= L1_CB_BUDGET_BYTES              (unchanged)
      program CBs + resident shard <= bank size - L1_ALLOC_HEADROOM_BYTES

  The bank size is read from the live device (`ttnn.get_memory_view(...).total_bytes_per_bank`), never
  hardcoded — it is arch- and dispatch-config-specific. After the fix: fp32 `W = 4096` runs at `WT_CHUNK = 8`,
  bf16 `W = 8192` at `WT_CHUNK = 16`.

  **No regression, and it is provable rather than merely measured.** On an interleaved tensor the shard term is
  0, so the second wall can never bind and the derivation is *byte-identical* to the single-budget model
  (pinned by `test_interleaved_blocking_is_unchanged_by_the_two_budget_split`). On the sharded side exactly one
  of the geometries the suites exercise moves — surveyed across all six pinned + eight auto geometries — and it
  gets **faster** (pinned perf config, one fresh-cache run per variant via `RMS_NORM_L1_HEADROOM_KB`):

      geometry                                single-budget   two-budget
      (1,1,32,1024)   WIDTH [32,128]  (8,1)           5_007        5_016
      (1,1,32,2304)   WIDTH [32,256]  (9,1)           5_684        5_667
      (1,1,32,5120)   WIDTH [32,160]  (8,4)           5_878        5_890
      (1,1,32,7168)   WIDTH [32,256]  (7,4)           6_295        6_284
      (1,1,8192,1024) BLOCK [1024,128](8,8)          89_413       85_107   HT_BLOCK 4 -> 8, 1.05x

  **No EXCLUSIONS entry was needed**, which was not the expected outcome. `ROW_MAJOR × HEIGHT_SHARDED` looked
  like R2's `{ROW_MAJOR, WIDTH/BLOCK_SHARDED}` precedent, but the two differ: an RM height shard's tile-row is
  spread over up to 32 *different* cores, so the read is genuinely **non-local** — precisely the case
  `TensorAccessor` exists for, and the opposite of re-reading a core's own block. Routing it there costs 3
  lines and it passes at PCC ≥ 0.99999, clean under `--dev`. The same mechanism very likely unblocks R2's two
  RM exclusions; deliberately **not** attempted here, as it is outside this heading's scope.

  One case is refused and correctly so: fp32 `(1,1,32,8192)` HEIGHT_SHARDED needs 2 MB of shards on one core
  against a 1.46 MB bank. It is unreachable through the public entry point — `allocate_tensor_on_device` OOMs
  on the output shard first, which the harness classifies as an infeasible skip.

- **Perf note, a finding for Refinement 5 rather than a task it must rediscover**: the A/B above also
  re-measures the five pinned sharded geometries against their `achievable_ns` references. The four WIDTH ones
  are close (5 016–6 284 ns vs 4 110–5 481, i.e. 1.15–1.22× away), but `(1,1,8192,1024)` BLOCK_SHARDED sits at
  **85 107 ns against a 25 640 ns reference — a 3.3× gap**, by far the largest of the five and the obvious
  first target for R5's sharded column.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_height_sharded.py` (**new**, 47 cases) — the
    placement is pinned by the shard and the reduce stays local (core map, per-core row count, `cw == 1`,
    empty `groups`, no semaphores in the descriptor, tile-rows covered exactly once); the input **and** output
    CBs are aliased onto the shards (descriptor-level, because an accessor read of a core's own shard passes
    every numerical gate in the file); the RM shard is asserted **not** aliased, since it is the deliberate
    non-local exception; the all-ones absolute element-count check; the reference across
    dtype × gamma × 5 shapes; and the two budget-model guards described above.
  - `test_rms_norm_perf.py::test_rms_norm_perf_sharded_pinned` (**new**, 5 cases) — the five pinned sharded
    geometries at the pinned perf config, one dispatch each with the 0.9995 soft gate. Added as this
    refinement's no-regression guard for the budget change; it is also R5's measurement surface.
  - `RMS_NORM_L1_HEADROOM_KB` env override (read in the op, same style as `RMS_NORM_W_SPLIT`) so the
    single-budget vs two-budget A/B stays re-runnable without editing the op.
  - `probes/probe_028..034.py` — HEIGHT bring-up, the RM-shard granule check, the L1-tight corner, and the
    old-vs-new blocking survey across every sharded geometry the suites emit.
  - Unit suite: **582 passed, 73 skipped** (`--run-all`), and the sharded suites green under `--dev`.

---

## Refinement 5 — Speed up the perf-flagged **prefill** column and the sharded geometries

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK **1349.98 MHz** —
  the references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns` and nothing below is scaled.

- **What was done**: a perf-only refinement (no SUPPORTED change) on the four interleaved prefill
  profiles `(1,1,8192,W)` and the five pinned sharded geometries, all at the config `feature_spec`
  pins for them (bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2 / bf16 TILE gamma).

  **Measure first, and the measurement redirected the phase.** A new `test_rms_norm_perf_prefill_pinned`
  re-baselined prefill at the pinned config (the Phase-0/R2/R4 prefill numbers were all taken at the
  op's *default* HiFi4 / fp32-on corner — the wrong datapath), and a new **sharded** twin of R3's
  ablation harness (`test_rms_norm_ablate_sharded`) attributed the sharded time. The ablation holds
  the shard placement, the core count and the per-core slice byte-for-byte and removes only the
  gather + root fold + multicast:

      geometry            full   no-combine   combine   share   BRISC   NCRISC
      WIDTH 32x1024      4_913        3_088     1_825     37%      95    1_139
      WIDTH 32x2304      5_503        3_658     1_845     34%     282    1_476
      WIDTH 32x5120      5_664        3_288     2_376     42%     220    1_871
      WIDTH 32x7168      6_079        3_562     2_517     41%     191    2_286
      BLOCK 8192x1024   74_838       55_522    19_316     26%     293    2_801

  `BLOCK_SHARDED` is **MATH-bound**: with the combine removed it is 55.5 µs of pure TRISC time
  against **0.3 µs of BRISC and 2.8 µs of NCRISC**. So the verifier's suggested placement/NoC tune
  could not have moved it — only FPU-op count and DEST-sync overhead can. (The NoC pairing was in
  any case already correct by construction: reader = NCRISC/NOC_0 (+x,+y), writer = BRISC/NOC_1
  (−x,−y), and every gather root sits at the *low* corner of its group, so the gather's −x/−y
  traffic and the multicast's +x/+y traffic are each on the NoC that routes them natively.)

  **Lever (a) — one L1 wall instead of two.** `L1_CB_BUDGET_BYTES = 1_100_000` was a Phase-0 guess
  at "worker L1 minus firmware headroom". R4 then read the real bank from the live device
  (1 461 504 B) and made `prog + shard <= bank − headroom` the second wall, but kept the guess as a
  first wall on program-allocated CBs alone. That first wall is a *proxy* for the quantity the second
  one measures, and strictly more conservative, so it could only ever cost block size. Retired:
  one condition, `program CBs + resident shards <= live bank − L1_ALLOC_HEADROOM_BYTES`.

  Blast radius verified by A/B over the whole perf set (`probes/probe_036.py`): **exactly one cell's
  derivation changes** — `(1,1,8192,7168)` gains a **resident gamma** (needs 1 195 648 B; the guessed
  wall refused it, the real 1 330 432 B one accepts). gamma is the *same bytes on every core* and is
  re-read once per row-block when not resident, so that deletes 2 of 3 gamma passes ≈ **100 MB of the
  shape's 386 MB of DRAM traffic**. Every other cell is byte-identical.

  **Lever (b) — `FUSE_SQ`, the fused square-accumulate.** Phases 2 and 3 were two FPU passes over the
  same block: square into `cb_x_squared`, then elementwise-accumulate that block back out. The second
  one's operation is an *add*, so the FPU's accumulate-into-DEST mode collapses them —
  `mul_tiles(x, x, acc_to_dest)` over a sticky D0 leaves `Σ_w x_w²` in DEST, which is **exactly** the
  raw elementwise accumulator both the local finalize and the cross-core combine already consume
  (op_design's "the combine is literally the local chunk accumulate"). `eltwise_chain`'s
  `DestAccumulation` walk expresses it directly: D0 stays acquired across an outer row's whole `Wt`
  and is packed once per row. Removes one FPU op per input tile of four, the whole `cb_x_squared` L1
  round trip, and `W−1` of every `W` packs. Two preconditions, decided on the host and
  `static_assert`ed in the kernel: `NW == 1` (a DEST accumulator dies at the next `tile_regs_acquire`,
  and the chain forbids composing DEST with L1 accumulation) and no partial-W mask (that rides the
  reduce helper's partial-scaler hook, which this path does not go through). `scaler_dtype` follows
  the reduce's input CB to Float32 — R1's rule, applied, not a new one.

  **Lever (c) — `DEST_BLOCK`.** `EltwiseShape`'s `block_size` defaults to **1**, and at 1 the chain
  runs a whole `tile_regs_acquire/commit/wait/release` round *plus a pack phase* around **every single
  tile** (`examples/compute_block_size`: ~1.6 µs per extra pass, 1.65× end to end). Every chain in the
  compute kernel was doing this. It now asks for `DEST_AUTO_LIMIT` and lets `eltwise_chain` clamp to
  its own compile-time DEST capacity (`chain_max_block_v`) — so it is "the coarsest block that fits
  DEST", re-derived per chain and per `fp32_dest_acc_en` / `dst_full_sync_en` setting, not a constant.

- **Measured result** (pinned perf config, DEVICE KERNEL DURATION ns, best of two fresh-cache runs
  that agreed to 0.02–1.6 %; the `(a)+(c)` column is `RMS_NORM_FUSE_SQ=0`, so both levers are shown
  pulling separately and together):

      case              R4 base   (a)+(c)   +FUSE_SQ  speedup      ref   vs ref
      WIDTH 32x1024       5_002     4_975      4_900   1.021x    4_110    1.19x
      WIDTH 32x2304       5_730     5_735      5_503   1.041x    4_617    1.19x
      WIDTH 32x5120       5_895     5_776      5_664   1.041x    5_267    1.08x
      WIDTH 32x7168       6_344     6_217      6_079   1.044x    5_481    1.11x
      BLOCK 8192x1024    85_245    77_561     74_813   1.139x   25_640    2.92x
      prefill 8192x1024  97_097    97_632     98_229   ~1.00    96_744    1.02x
      prefill 8192x2304 222_213   219_159    217_673   1.021x  211_345    1.03x
      prefill 8192x5120 480_766   471_674    470_143   1.023x  738_307    0.64x
      prefill 8192x7168 810_299   652_079    659_050   1.229x  1_032_281  0.64x
      decode x4                                        ~1.00            all inside

  On `BLOCK_SHARDED` the split is `DEST_BLOCK` 1.099× then `FUSE_SQ` a further 1.037×. `8192×1024`,
  `8192×5120` and the four decode shapes are flat *inside the noise band* — which this phase also
  measured properly: re-running a **byte-identical** `8192×5120` program gave 480 766 / 497 328 /
  471 787 / 470 143 / 473 456, a ±3.4 % spread, so the large prefill shapes are noisier than the
  1–2 % the decode column showed.

- **The prefill column has no data-movement headroom, and that is measured rather than argued.** The
  verifier note said to check the roofline before manufacturing a change. `/perf-ceiling-dm`'s NPE CLI
  is a test-build target not configured in this tree, so a stronger empirical form was used instead —
  a real op on the real box rather than an interpolated model. `test_prefill_dm_ceiling_reference`
  dispatches **`ttnn.clone`** on the same tensor with the same interleaved placement: every tile read
  once, written once, no reduction, no gamma. Nothing that reads and writes this tensor can beat it.

      W       clone ns   clone GB/s   rms_norm ns   rms_norm GB/s
      1024      83_100          404        98_229             415
      2304     192_568          392       219_966             417
      5120     408_984          410       470_143             434
      7168     586_752          400       659_050             433

  rms_norm moves its bytes **4–6 % more efficiently than a plain DRAM copy of the same tensor**, and
  comes in *under* `clone × 1.215` — its byte-scaled floor, since gamma is
  `110·Wt / (2·256·Wt + 110·Wt)` = a flat **17.7 %** of traffic independent of `W` — on all four
  shapes. Two of the four prefill shapes are 1.57× *ahead* of their references; the other two are at
  theirs within the noise band.

- **Accuracy achieved**: PCC ≥ 0.9995 on every pinned shape (asserted by the three `*_pinned` tests,
  which carry the loose cases' tighter soft gate). `FUSE_SQ` changes *where* the running sum lives —
  a sticky DEST register instead of an fp32 L1 accumulator reloaded per pair — so it was pinned with
  an **absolute** check and an equivalence check, never a PCC: all-ones input recovers
  `mean(x²) = 1.0` exactly on every fused shape, and fused vs pairwise agree to a median ratio of
  1.000 on random data.

- **Golden test progress**: unchanged by design (perf refinement, no SUPPORTED change).
  `test_op_loose` **19 passed / 0 xfail**. Cross-product slices, **0 failed and 0 XPASS-drift
  throughout**: `1x1x64x128` + `1x1x32x8192` + `1x1x17x50` + `4x8x32x256` = **534 passed**
  (fused and non-fused, aligned and both-non-aligned, TILE/RM × gamma/no-gamma × every dtype ×
  interleaved/WIDTH/BLOCK/HEIGHT), and `1x1x256x512` + `2x512x1024` + `1x1x32x4096` = **296 passed**
  (the sharded placements). Unit suite **605 passed / 78 skipped**; the W-split and HEIGHT suites
  green under `--dev` (watcher + NoC sanitizer), which R2's changelog flags as non-optional for this
  scheme since the compute kernel that feeds the gather changed.

- **Issues encountered** — one real bug, in the new lever, caught immediately by the pinned gates:

  **`block_size > 1` with `OutputLifecycle::Streaming` gives PCC 0.0.** `PackTile` is not a CB-reader
  element, so it does **not** constrain `block_size` (`chain_supports_block` only inspects the input
  side) — but `Streaming` is `{ReservePolicy::PerTile, PushPolicy::PerTile}`, reserving and pushing
  **one** tile per block-iteration while the pack loop writes `inner_count` of them. `Chunked` is the
  matching policy. Fixed on the three blocked pack sites (phases 2, 5, 6, both their resident and
  streaming spellings); the two chains whose operands are per-tile (phase 4's `CopyTile`, the
  republishing `copy`) are clamped to `block_size 1` regardless, so they keep `Streaming` and an
  unblocked shape rather than carrying a knob that can never turn. Catastrophic and immediate, so it
  cost one cycle — but it is exactly the class of thing that would have been invisible had the block
  happened to be 1 on the shapes under test (`(1,1,32,1024)`, whose `WT_CHUNK` is 1, was the single
  case that passed while the other 12 failed).

  Two of my *test* premises were also wrong and worth recording, since both would mislead a reader:
  `(1,1,32,8192)` is **fused** even though its whole-tensor `Wt` is 256 — the W-split hands each core
  only 8 W-tiles, so the precondition is on the *per-core* chunk count, not on `W`; and R4's
  `test_resident_shard_is_not_charged_against_the_cb_budget` A/B has to accept **outright refusal**
  as well as a shrunk block, because that is the form R4 measured on fp32 `W=4096`.

- **Remaining headroom — findings, not queued tasks.** `BLOCK_SHARDED` is still 2.9× from its
  25 640 ns reference and is math-bound, so the next lever there is **fusing phases 5 and 6**
  (`x·(1/rms)` then `·gamma`): `DestReuseBinary` carries no `BroadcastDim` (R3 already established
  this), but pre-expanding gamma to full tiles *once per core* makes it expressible — `Wt` tiles,
  cheap at the sharded `Wt = 4…8`, and it can reuse the L1 that `cb_scaled` vacates. It deletes the
  whole `cb_scaled` round trip (128 packs + 128 unpacks per core) but **no FPU ops**, so on a
  math-bound kernel I estimate only ~5 %, which is why it lost to spending the remaining budget on
  the regression net. The combine's residual 19.3 µs (26 %) is **not** topology-bound: a
  `BLOCK_SHARDED` group is one grid row (`cw_y == 1`), so R3's two-stage tree cannot apply by
  construction, and `master.md`'s `two_phase_reduce_mcast` — measured 1.69× on a 1×8 line at exactly
  this payload — is a T3-shaped restructure of which core owns which tile-row. For **prefill**, the
  one remaining byte reduction is a **grid-wide gamma multicast** (op_design's Lamp L2), worth ~1.2×
  on the column; not built because it adds a global sync to the highest-volume path,
  `examples/shared_input_reuse` measures such a broadcast at only 1.71× for an 11× read reduction,
  and the two shapes it would help are already at their references within the noise band.

- **Tests added**:
  - `test_rms_norm_fused_reduce.py` (**new**, 14 cases) — the `FUSE_SQ` datapath: it *engages* where
    expected (else every assertion below would pass on the fallback path), it *refuses* where
    structurally wrong (`NW > 1`, partial W), all-ones recovers `mean(x²) = 1.0` **absolutely**, and
    fused agrees with pairwise on random data by median ratio. The file's docstring records why a PCC
    gate is not acceptable here: R1, R2 and R4 each shipped an accumulation bug that PCC scored
    ≥ 0.9998 because miscounting elements only rescales rows.
  - `test_rms_norm_perf.py::test_rms_norm_perf_prefill_pinned` (**new**, 4 cases) — prefill at the
    pinned perf config, the datapath its `achievable_ns` references were taken at.
  - `test_rms_norm_perf.py::test_prefill_dm_ceiling_reference` (**new**, 4 cases) — the measured DM
    ceiling (`ttnn.clone` on the same tensor), so "prefill is saturated" stays a re-runnable
    measurement instead of a claim in a changelog.
  - `test_rms_norm_perf.py::test_rms_norm_ablate_sharded` (**new**, 5 cases, opt-in via
    `RMS_NORM_ABLATE`) — the sharded twin of R3's ablation harness.
  - `test_rms_norm_height_sharded.py::test_live_bank_budget_buys_gamma_residency` (**new**) — pins
    lever (a) exactly where it is load-bearing, including that the *block factor* must not move.
  - `RMS_NORM_FUSE_SQ` env override, so lever (b) stays A/B-measurable without editing the op.
  - `probes/probe_035.py`, `probes/probe_036.py` — the blocking survey across every perf geometry and
    the one-cell-changes A/B for the budget retirement.
