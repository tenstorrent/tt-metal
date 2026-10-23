# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-22
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), built on
  the base op `rms_norm`'s local regimes. One `ttnn.generic_op` per call (one MeshProgramDescriptor, one
  ProgramDescriptor per chip):
  - R1-X: rows split row-wise over the whole compute grid. R3-X: W-slice row-groups plus an in-chip root combine,
    selected when it buys ≥ 2× the cores.
  - Each row-group root publishes its compact fp32 per-row Σx² (128 B per tile-row) to dedicated Fabric port cores.
    The ports all-gather the partials with a one-hop store-and-forward relay (Linear and Ring) behind a per-call ready
    fence. Every device sums all G partials in device order (exact fp32 SFPU adds) and finalizes on column 0.
  - The γ pass runs in the exchange's stall shadow.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned],
  rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, "none"],
  memory_layout=[INTERLEAVED], cluster_axis=[0, 1], topology=[Linear, Ring], num_links=[None, 1, 2];
  EXCLUSIONS=[{float32, fp32_dest_acc_en: False}]
- **Accuracy achieved** (4 shapes × 2 dtypes, `test_distributed_rms_norm_precision_baseline.py`, worst device):
  - bf16: PCC=0.999998, max_abs_err=0.0625, rel_rms_err ≤ 2.2e-3, ≤ 1 ULP.
  - fp32: PCC=1.000000, max_abs_err ≤ 0.025, rel_rms_err ≤ 1.25e-3; median got/true 0.9991 from FPU TF32 truncation.
- **Golden suite at Phase 0**: 821 passed / 0 failed / 0 hangs of 47,025 (`verifier_artifacts/golden_results.txt`).
  verify_supported: supported_pass 784, supported_fail 0, xfail_wrong_mode 0, xpass_drift 1 (a documented classifier
  false positive: an expected-raise test of the {fp32, no-fp32-DEST} exclusion).
- **Performance** (traced device ns, slowest device, 1×4 ring, 2 links): 640×1792 ≈ 26.5 µs (target 14.1),
  32×1792 ≈ 11.4 µs (target 2.7), 2048×1280 ≈ 43.1 µs (target 29.4).
- **Issues encountered** (verifier fixes, details in `verification_report.md`):
  - Wave race in the stats publish (`SEM_OWN_STATS` cumulative across waves): R1 with `num_waves = 2` shipped ~40/108
    stale rows per call. Fixed with the `SEM_OWN_COUNTED` gate.
  - Writer stored output quanta in the wrong order when `R_b ≥ 2` and `nch ≥ 2` (rel-RMS 0.56 on 4096×4096). Fixed.
  - Persistent landing-scratch cache: unbounded per-size L1 growth, tensors exposed to the golden failure cleanup
    (159 cascaded translated failures), and address-0 mock allocations cached under no-dispatch collection. Fixed
    with bucketed, liveness-checked, int-only caching.
  - Drift: `num_links=None` added to SUPPORTED. Ledger currency (`WS`), `cb_output` page from the output dtype,
    DRY `TILE_DIM` / `COMMON_CT_ARGS`, reader compiler barrier.
  - Harness findings (not op defects): golden L1 capture blind to heterogeneous mesh workloads (run with
    `EVAL_CAPTURE_L1=0`); the acceptance file is incompatible with inline precompile; fabric mesh reopen under
    `--dev` is flaky (reproduced with `ttnn.all_gather`).
- **Tests added**: test_distributed_rms_norm.py (planner), test_distributed_rms_norm_debug.py,
  test_distributed_rms_norm_perf.py (implementer), test_distributed_rms_norm_precision_baseline.py,
  test_distributed_rms_norm_verifier_regressions.py (verifier), probes/ (verifier investigations)

## Refinement 1 — Sharded placements: HEIGHT_SHARDED and WIDTH_SHARDED, native zero-copy
- **Date**: 2026-10-22
- **What was done**:
  - `SUPPORTED["memory_layout"]` += HEIGHT_SHARDED, WIDTH_SHARDED. Native zero-copy: `cb_x_tiles` / `cb_output` are
    backed on the input / output shard (`ttnn.cb_descriptor_from_sharded_tensor`); the reader hands each resident block
    to compute with no NoC read (no own-shard `TensorAccessor`); compute squares the slice's valid tiles strided by the
    shard row stride and packs strided into the output shard. Output placement defaults to the input's own; any other
    placement (DRAM, or a different shard grid — translated width-sharded cells) streams through the writer.
  - HEIGHT → R1-X groups read off the shard spec (the shard is the block). WIDTH → one R3-X row-group over the shard
    cores with **R3-RS owners** (the design's deferred reduce-scatter): each owner sums / publishes / finalizes `R /
    owners` rows and unicasts their inv tails to every member; owners are independent roots at the ports, so WIDTH
    uses both links. Interleaved R3-X keeps a single root (byte-identical).
  - Port cores move to the last free cores of the grid when the shard grid covers logical row 0.
  - L1: CB budget = the bottom free L1 block (was the largest free block, which over-reported next to shards);
    persistent landing scratch / semaphores relocated upward when they cap the CB region; L1 fallbacks (streamed γ,
    smaller chunk C) for wide shards; gather / inv depth 1 (a second slot could never fill early); balanced blocks;
    `cb_x_gamma` ragged-block cycle padding (also closes a latent straddle on the interleaved ragged-wave path).
  - Reused: R1-X / R3-X plans, every compute helper chain, the exchange and port kernel unchanged. Added: CT args
    `X_SHARD`, `OUT_SHARD`, `GAMMA_RESIDENT`, `OWNERS` (COMMON_CT_ARGS 18 → 22), owner bookkeeping in the reader.
- **Accuracy achieved**: golden tolerances (helpers.TOLERANCES, worst device) on every sharded shape in INPUTS; bf16 and
  fp32, with / without γ (bf16 / fp32), both cluster axes, 1 / 2 / None links, Linear and Ring (1×4 snake).
- **Golden test progress**: all 1152 valid supported sharded cells: 1124–1128 passed, 24–28 skipped INFEASIBLE_L1
  (fp32 shards of 0.5–1 MB per core), 0 failed; interleaved sample (60) + test_regression (12) green; translated
  width-sharded RMS cells 8/8 (bf8b / w_non_aligned cells xfail as refused).
- **Performance** (traced, slowest device, 1×4 ring, 2 links, bf16 x / γ): HEIGHT 640×1792 20.0 µs, 32×1792 15.8,
  2048×1280 22.3, 4096×1024 26.1; WIDTH 640×1792 171 → 26.3 µs with R3-RS, 32×1792 15.1, 2048×1280 63.0,
  4096×1024 168 → 139 µs (depth-1 gather). Interleaved unchanged (640×1792 27.9, 32×1792 11.6, 2048×1280 43.0).
- **Issues encountered**:
  - Persistent L1 created under a live shard capped every later program's CB region (bank-uniform allocator; 1.46 →
    0.42 MB) → `_settle_persistent` + bottom-block budget. Trade-off: moved buffers can split free L1 for later
    borderline allocations (4 fp32 32×4096 HEIGHT cells skip INFEASIBLE_L1 at input creation in a sharded-only run).
  - The plan depends on live L1 (a still-held L1 output lowers the budget): a trace must be warmed up with the
    captured call sequence (perf test updated).
  - Remaining headroom: tall WIDTH is L1-bound to several blocks (four fp32 intermediate tiles per row of a 1-tile
    slice), each paying an exchange round; next: cross-block pipelining, 16-bit intermediates (Refinement 4).
- **Tests added**: `test_distributed_rms_norm_sharded.py` (HEIGHT / WIDTH shapes incl. ragged slice and R3-RS q > 1,
  fp32 + Ring + default links, sharded → DRAM output, waves with alternating inputs incl. a ragged R3-RS block, 1×4
  Ring / Linear); `test_distributed_rms_norm_perf.py::test_perf_ring_sharded`; probes for the L1 budget /
  fragmentation / plan determinism.

## Refinement 1b — Sharded placements: HEIGHT_SHARDED and WIDTH_SHARDED, native zero-copy (debug: gate violations)
- **Date**: 2026-10-22
- **What was done**: no kernel or op-file change; Refinement 1's sharded paths stand as shipped. Both gate violations
  were in the test and measurement plumbing:
  - **Bullet 2**: `test_shapes_linear[1x1x32x32-axis0]` rel-RMS 1.0. Reproduced only as the gate runs it (every file,
    default inline precompile). Root cause: the acceptance file's own `_MeshCache` opened fabric meshes inside test
    bodies, and after the precompile collect pass the real-pass mesh dropped every host write. The input, a replicated
    `ones` probe and the output all read back as zeros *before / independently of* the op. The file now takes its
    meshes from the root `mesh_device` fixture (`device_params` = FABRIC_2D_TORUS_XY + router payload). System-mesh tests
    are parametrized to `system_mesh_shape()`, ring tests to their place's mesh shape. Test bodies and assertions are
    unchanged.
  - **Bullet 3**: 593/593 golden failures (563 interleaved) were `L1ProfilingError: peak_total=0`, the same as at
    Phase 0. Graph tracking is thread-local, and a heterogeneous MeshWorkload compiles and allocates CBs on the mesh
    thread pool, so the capture never saw this op's CBs (only sharded outputs and fresh landing scratches registered).
    Once the CBs were visible, the peak was 4–16× too high (2.3–8.4 MB "per core"): each program reported its CBs to
    every sub-device, and the extractor summed per bare (x, y) across chips. Infra fixes, rebuilt via
    `cmake --build build_Release --target install`:
    - `tt_metal/distributed/mesh_workload.cpp`: compile on the calling thread while a graph capture is active, and give
      each program of a multi-program workload its own range's devices for CB tracking.
    - `tt_metal/impl/program/program{_impl.hpp,.cpp}`: `set_cb_tracking_devices()`; `allocate_circular_buffers`
      reports to those devices when set.
    - `ttnn/core/graph/graph_trace_utils.cpp`: `extract_resource_usage_per_core` accounts per (device, core).
- **Accuracy achieved**: unchanged (golden tolerances, worst device).
- **Golden test progress**: full suite, default `EVAL_CAPTURE_L1=1` + precompile: 1957 passed / 1 failed / 33,771
  skipped / 11,296 xfailed / 0 hangs of 47,025 (was 1150 / 593). Gate (`evaluate_completion`): bullets 1–3 pass,
  1921/1945 responsible cells, no regression. Per-chip L1 peaks now plausible (1×4 ring: 32×1792 145 KB,
  640×1792 285 KB, 2048×1280 555 KB).
- **Issues encountered**: 1 golden failure: translated `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`.
  It is an expected-raise test; the op refuses before dispatch, so the capture has no allocation and reports
  peak_total = 0. It is not an op defect and not in the gate's set. `tests/ttnn/unit_tests/base_functionality/test_graph_trace_utils.py`
  is still green (19/19).
- **Tests added**: none. `test_distributed_rms_norm.py` moved to the root `mesh_device` fixture. Whole unit directory,
  default precompile: 118 passed, 1 skipped.

## Refinement 2 — Row-major gamma sticks + non-tile-aligned shapes (W and H)
- **Date**: 2026-10-22
- **What was done**: `SUPPORTED["gamma_layout"] += ROW_MAJOR_LAYOUT`, `SUPPORTED["alignment"] += w_non_aligned, h_non_aligned`,
  for every supported placement (INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED), natively in the kernels. The compute
  kernel, the CB graph and the plan are unchanged.
  - **RM γ sticks** (writer, `GAMMA_RM` CT arg 23): stick `w0 + j` is read whole (one aligned 64 B / 128 B DRAM page)
    into face 0 row 0 of `cb_gamma_tiles` page j. Its elements 16..31 are then moved to face 1 row 0 with an 8 / 16-word
    L1 copy. A second NoC read of the stick's second half would start 32 B into a DRAM page, which is off the DRAM
    read alignment. Same `load_gamma` for the resident and streamed γ layouts. `BroadcastDim::Row` reads row 0 only.
  - **w_non_aligned** (reader, `w_last_valid` CT arg 22): the core whose slice holds tile `Wt-1` zeroes columns
    `[w_last_valid, 32)` of that tile in every block row before the push. Whole pad faces use the NoC zero loopback,
    and partial face rows use element stores (adapted from the base op's `zero_tile_w_padding`).
    - Interleaved: after the read barrier.
    - Sharded: in the resident shard before handing it to compute. The shard's tile padding belongs to the tensor and
      is not guaranteed to be zero.

    The divisor stays the logical `G·W_local`. The writer stores whole tiles.
  - **h_non_aligned**: no kernel change. Pad rows are independent rows and land in the output's tile padding. The
    tile-row geometry (`prod(shape[:-2]) · ceil(H/32)`) is verified on rank-3 / rank-4 shapes with batch > 1.
  - `COMMON_CT_ARGS` went from 22 to 24, so the reader and writer `TensorAccessorArgs` offsets moved to 24.
  - Reused: the reader's `load_block` (both placements), the writer's `load_gamma`, and the base op's W-pad zeroing
    routine.
  - Added: 2 CT args, the stick re-lay, and the pad-zeroing call sites.
- **Accuracy achieved**: every case passes the golden `check_output` gate (bf16 PCC ≥ 0.999 / rel-RMS ≤ 1e-2; fp32 tighter)
  on shapes [1x1x32x50, 1x1x64x1000, 2x1x128x100, 4x128x47, 32x17, 1x1x17x50, 1x1x100x512, 1x1x1x1792, 1x17x128,
  17x64, 2x3x47x96, 1x1x32x1792, 1x1x640x1792, 1x1x32x8192], interleaved / HEIGHT / WIDTH. This includes inputs whose
  tile padding was filled with 1000.0 before the call, which shows the pad columns never reach Σx².
- **Golden test progress**: stratified slice of 186 cells (one per shape × placement × γ layout over all 180 groups of
  the target cells, plus 6 prior-passing tile-aligned TILE-γ cells): 186/186 passed.
  `test_translated.py` + `test_regression.py`: 248 passed, 51 skipped, 43 xfailed, 0 failed. The translated RM-weight
  cases now pass.
- **Issues encountered**: none in the op. Test-side, `ttnn.fill_implicit_tile_padding` refuses multi-slice shards
  (`N_slices == 1`), so the dirty-padding cases skip rank-3 sharded inputs. fp32 32×8192 HEIGHT cannot allocate its
  1 MB output shard next to the 1 MB input shard, which the golden suite skips as INFEASIBLE_L1, as it did in
  Refinement 1.
- **Tests added**: `test_distributed_rms_norm_layouts.py`:
  - `test_non_aligned`: 11 shapes × 3 placements, RM γ.
  - `test_dirty_w_padding`: bf16 / fp32, pad filled with 1000.
  - `test_rm_gamma`: mixed x / γ dtypes, 3 placements.

  `test_unsupported_cells_refused` no longer expects refusals for the two newly supported values. Whole unit
  directory: 194 passed, 6 skipped.

## Refinement 3 — Speed up the perf-flagged 640×1792 ring case (knob co-tune)
- **Date**: 2026-10-22
- **What was done**: the waves knob now does what the design meant it to do (wave w's exchange overlaps wave w+1's
  DRAM reads), the plan uses it by itself on interleaved R3-X, and the other three named levers were measured.
  Measurements: traced, 6 back-to-back calls, `test_distributed_rms_norm_perf.py` under `--profile`, 1×4 ring, 2 links,
  bf16, slowest device of the group, median of the steady calls.
  - **Phase-0 timeline (640×1792, one block, 100 cores)**: launch 2.3 µs, x DRAM read about 5–7 µs, in-chip combine +
    publish about 2.5 µs, Fabric exchange about 5 µs (including the inter-chip launch skew at the ready fence), slot
    gather + finalize + inv distribution about 5 µs, then the output DRAM writes about 7 µs. All of it ran in series.
  - **Waves alone (`BLOCK_ROWS_CAP = 1`) were slower: 30.9 µs vs 27.3 µs.** The reader ran load → combine → exchange →
    inv per wave, back to back, so a second wave doubled the latency chain. `x_depth = 2` prefetched nothing.
  - **Lamp L1 — wave look-ahead (`WAVE_PIPELINE`, new knob, default on).**
    - Reader: `front(w)` (load, combine, publish) and `back(w)` (slot gather, inv distribution), run as `front(0)`,
      then `[front(w+1), back(w)]`.
    - Compute: `head(w)` (square, collapse, combine) and `tail(w)` (γ, finalize, scale), same order.
    - Safety mechanisms it needs, each documented at the code:
      - two `cb_gather` regions and two `cb_stats_slots` regions;
      - per-parity gather semaphores (`SEM_GATHER_ODD`): a member may publish block w+1 before a slower member
        publishes block w;
      - per-direction landed semaphores (`SEM_STATS_LANDED_BWD`): one port may notify wave w+1 before the other
        notifies wave w;
      - an interleaved x ring that holds every wave (indexed unpack reads may not wrap, an LLK assert).
    - The solve picks the look-ahead only where its footprint fits. Waves that L1 forced keep the back-to-back path.
    - With the look-ahead, 640×1792 at 2 waves: 23.6 µs.
  - **Decoupled port exchange**: the port advances its in-order sends and its per-wave notifies independently. Wave
    w+1's own segment no longer waits for wave w's last arrival. A single wave keeps the blocking schedule.
    640×1792: 22.2 µs.
  - **`R3_MIN_WAVES = 2` (new knob)**: interleaved R3-X row-groups are split into at least 2 blocks by the plan itself.

    | Case | Effect |
    |---|---|
    | 640×1792 | 27.3 → 22.3–22.8 µs |
    | 1024×2048 | 40.3–41.6 → 32.9–34.5 µs |
    | 2×4×128×512 | 19.2–19.7 → 18.5–18.8 µs |
    | interleaved R1-X (not applied) | 4096×1024 57.2 → 59.2 µs, 8192×1024 99.2 → 102.0 µs |
    | WIDTH_SHARDED R3-RS (not applied) | 640×1792 25.8 → 31.0 µs |

    Path-gated to interleaved R3-X for those reasons.
  - **Lamp L2 — R3 group shape**: sweep at one and two waves, all flat or worse. Base values kept.

    | Setting | 1 wave | 2 waves |
    |---|---|---|
    | `R3_ROWS_PER_GROUP` 1 | 27.2 µs | — |
    | `R3_ROWS_PER_GROUP` 4 | 33.2 µs | 33.7 µs (cap 1), 28.2 µs (cap 2) |
    | `R3_MAX_GROUP_CORES` 8 | 27.6 µs | 23.0 µs |
    | `R3_MIN_SLICE_TILES` 2 | 27.5 µs | — |
  - **Lamp L3 — port placement (`PORT_X0`, new knob, default centred)**: x0 = 0 gave 22.7 µs, the right end 23.1 µs,
    centred 22.6 µs. A null result: the exchange packets (about 1 KB) are latency-bound, not bandwidth-bound. Kept as a
    knob.
  - **Lamp L8 — NoC balance (`DM_NOCS`, new knob, default (0, 1))**: swapping to reader NoC1 / writer NoC0 gave
    32.2 µs, against 22.6 µs. Both kernels on one NoC is not a valid dedicated-NoC configuration and is refused by an
    assertion.
  - **A regression found and fixed on the way**: splitting the reader into front/back cost the inv distribution loop on
    WIDTH 32×1792 (55 members) 3.28 → 3.77 µs. The compiler had proved `my_index == 0` and `my_rows == 1` in the
    inlined original, which turned the loop into a pointer walk. `send_inv_tails` now walks the rotated member order as
    two contiguous runs, with a compile-time single row when Q == 1. Back to 3.29 µs.
  - Reused: every kernel phase and helper call, `BLOCK_ROWS_CAP`, `x_depth`, the R3 / R3-RS ownership, the landing
    layout and the `SEM_OWN_COUNTED` gate.
  - Added: the front/back and head/tail split, the look-ahead order, 2 semaphores, 1 CT arg (`slot_depth`, with
    `COMMON_CT_ARGS` going 24 → 26 together with `SEM_GATHER_ODD`), the port's two-sequence loop, and the knobs
    `WAVE_PIPELINE`, `R3_MIN_WAVES`, `PORT_X0` and `DM_NOCS`.
- **Accuracy achieved**: the math is unchanged, only the schedule moved. The golden `check_output` gate passes: bf16
  PCC ≥ 0.999 / rel-RMS ≤ 1e-2, fp32 tighter. This holds on the perf case (`test_op_loose`, every loose case), on
  1×1×640×1792 interleaved across all dtype / link / topology / axis variants, and on 2×4×128×512 bf16 interleaved.
  The perf test now also checks its traced output.
- **Guard set** (pre-R3 → now, µs):

  | Case | pre-R3 | now |
  |---|---|---|
  | 640×1792 | 27.3 | 22.3–22.8 |
  | decode 32×1792 | 11.4 | 11.4–11.5 |
  | R1-X 2048×1280 | 43.1 | 42.6–43.6 |
  | R1-X 4096×1024 | 57.5–59.3 | 57.1–57.6 |
  | 1024×2048 | 40.3–41.6 | 32.9–34.5 |
  | 2×4×128×512 | 19.2–19.7 | 18.5–18.8 |
  | HEIGHT 640×1792 | 20.3 | 20.2–20.3 |
  | HEIGHT 2048×1280 | 22.7 | 22.6–22.7 |
  | HEIGHT 4096×1024 | 26.3 | 26.1–26.7 |
  | HEIGHT 32×1792 | 15.75 | 15.78 |
  | WIDTH 640×1792 | 25.8–26.4 | 25.9–26.2 |
  | WIDTH 2048×1280 | 62.9 | 62.8 |
  | WIDTH 4096×1024 | 138.8 | 138.0 |
  | WIDTH 32×1792 | 14.6–15.4 | 14.8–14.9 |

  The γ-less, fp32, Linear and Ring paths are covered by the unit directory and the golden slice for correctness.
- **Golden test progress**: targeted slice (`test_op_loose`, 640×1792 interleaved, 2×4×128×512 bf16 interleaved):
  71 passed, 0 failed. `test_regression.py`: 12/12. Unit directory: 201 passed, 6 skipped.
- **Issues encountered**:
  - Indexed unpack reads cannot wrap a CB (LLK assert), hence the whole-slice x ring under the look-ahead.
  - Same-NoC reader / writer fails at dispatch.
  - The measured number includes inter-chip launch skew: the port's ready fence waits up to 4–5 µs on the device that
    started first, and slowest-device durations drift up about 0.4 µs per traced call.
- **Tests added**:
  - `test_waves_alternating_inputs` is parametrized over the look-ahead (`lookahead` / `back_to_back`) and asserts the
    plan takes it.
  - `test_perf_ring` gained 4096×1024 (R1-X, 2 rows per core), 2×4×128×512 and 1024×2048 (interleaved R3-X), a
    `DRMS_KNOBS` knob-override fixture, and an output check.

## Refinement 4 — Numerical configurability: bfloat8_b input and 16-bit DEST
- Date: 2026-10-23
- What was done:
  - `SUPPORTED["dtype"]` += `bfloat8_b`; `SUPPORTED["fp32_dest_acc_en"]` += `False`. The `{float32, False}` EXCLUSION
    stays. No new EXCLUSION was needed: bf8b × w_non_aligned passes, see the reader change below.
  - Program descriptor: `INTERM_FORMAT` is gone. `dest_format(fp32_dest_acc_en)` is now the one source for the
    DEST-following pages (`cb_sumsq_partial`, `cb_x_gamma`: Float32 / Float16_b). The L1 solve (`make_plan(...,
    fp32_dest_acc_en)` → `footprint(T_dest=...)`) prices them at that width, so 16-bit DEST frees `R·(1 + C)·2 KB`.
    `STATS_FORMAT = float32` covers the pages the reader moves as 4-byte words (`cb_partial_col`, `cb_gather`,
    `cb_device_partial`, `cb_stats_slots`, `cb_inv_root`, `cb_inv_rms`). This is the "keep every tile the reader touches
    Float32" option, so the wire format and `col0_word` are unchanged. These pages hold one tile per tile-row and are
    under 1 % of the footprint. `cb_x_tiles` / `cb_output` already took the tensor dtype, so Bfp8_b pages needed no
    change.
  - `UnpackToDestFp32` on `cb_stats_slots` is now set only when `fp32_dest_acc_en`. With 16-bit DEST the tag made the
    unpacker write 32-bit words into a DEST that math / pack read as 16-bit, and every output was inf (measured,
    pcc = 0). Untagged, the fp32 slots enter DEST through srcA, and `AddCol0` sums them in the SFPU's fp32 lanes and
    rounds to bf16. The sum order is still fixed, so `inv` stays bit-identical group-wide. The page and the wire stay
    fp32.
  - Reader W-pad zeroing now covers block-float: `zero_tile_w_padding<es, data_off>` with `es = T_in / 1024` and
    `data_off = T_in − 1024·es`. For Bfp8_b that is 1 byte per element and 64 B of shared exponents. Only the mantissa
    bytes are zeroed, and a zero byte is an exact 0 whatever the shared exponent. For Float32 / Float16_b,
    `data_off = 0`, so the code is the same as before.
  - Compute kernel: unchanged. `FinalizeCol0` / `AddCol0` already round when `!DST_ACCUM_MODE`, and the helpers
    reconfigure the formats.
- Accuracy achieved (precision matrix, worst device; 8 shapes × uniform / normal; full table in
  `tests/.../precision_matrix_results.md`):

  | dtype | DEST | HiFi4 min PCC / max rel-RMS | LoFi min PCC / max rel-RMS |
  |---|---|---|---|
  | float32 | fp32 | 1.000000 / 1.27e-3 | 0.999749 / 4.2e-2 |
  | bfloat16 | fp32 | 0.999997 / 2.8e-3 | 0.999743 / 4.3e-2 |
  | bfloat16 | 16-bit | 0.999990 / 6.7e-3 | 0.999756 / 4.0e-2 |
  | bfloat8_b | fp32 | 0.999936 / 1.13e-2 | 0.999712 / 4.0e-2 |
  | bfloat8_b | 16-bit | 0.999934 / 1.32e-2 | 0.999702 / 3.7e-2 |

  HiFi3 matches HiFi4, and HiFi2 sits between HiFi4 and LoFi. The fp32 rel-RMS of 1.27e-3 (baseline 1.1e-3) comes
  from the known TF32 operand truncation; it has not regressed.
- Golden test progress (targeted slices):
  - New cells, interleaved, γ bf16 TILE, axis 1, 1 link, Linear, all 36 shapes, {bf8b × both DEST, bf16 × 16-bit
    DEST}: 108/108.
  - New cells on HEIGHT / WIDTH_SHARDED, axis 0, 2 links, γ none or fp32 ROW_MAJOR: 432 passed, plus 432 INVALID
    skips.
  - Before the two fixes the first slice was 31/108: every 16-bit-DEST cell gave inf, and bf8b × w_non_aligned had
    pcc 0.945.
- Issues encountered: the `UnpackToDestFp32` × 16-bit DEST incompatibility and bf8b W-pad zeroing (both fixed, see
  above).
- Tests added: `test_distributed_rms_norm_precision_matrix.py` covers shape (8) × dtype (3) × DEST (2) × fidelity (4) ×
  distribution (2): 320 passed, 64 skipped (the EXCLUSION). Results are in `precision_matrix_results.md`. The unit
  directory still passes: main 66, layouts 76, sharded 31, verifier regressions 11, precision baseline 2.

## Refinement 5 — Exchange transport scheme for the perf-flagged case (T3)
- Date: 2026-10-23
- What was done: the three named transport levers are implemented and on by default. Each is a live knob in the program
  descriptor. No SUPPORTED change.
  - **L6 direct multi-hop sends (`DIRECT_RING_SENDS`)**: on a Ring of G ≥ 3, every chip sends its own segment straight
    to every peer. The packet is routed by destination node (by hop count on 1D fabrics) and leaves on the port whose
    direction matches `ttnn.get_eth_forwarding_direction`'s first hop, so there is no relay-kernel turnaround. Linear,
    and any group whose routes leave through neither ring neighbour, keep the store-and-forward relay. The new
    `_exchange_plan` builds both schedules as one record per chip: send items (own segment, or relay of entry j),
    entries per port, and ready targets. The port kernel runs either schedule.
    - Since entries from different sources can now land in any order, each entry has its own arrival counter: a
      persistent global semaphore on the port cores, `1 + (max mesh dim − 1)` per core.
    - The ready fence now takes one ready from every destination chip: a 2-hop ready on the direct Ring.
  - **L4 per-entry notify (`NOTIFY_PER_ENTRY`)**: a port increments its roots' landed semaphore per landed entry, by
    the newly landed run at once. A root reads and expands each slot as soon as it lands, while later slots are still
    on the Fabric. The sum order is still fixed by the page layout (slot g at page t·G + g, one fixed-order compute
    chain), so inv is bit-identical group-wide. The compute kernel is unchanged.
  - **L5 inv multicast (`INV_MCAST`)**: a single-root R3-X row-group's root multicasts each finalized row's compact
    tail through the kernel_lib `McastFamily` / `McastArgs` (Counter data-ready, no receiver handshake). Interleaved
    R3-X row-groups are now built as single hole-free rectangles (`_rect_groups`: row segments first, then column
    segments of what is left; port cores are never members). On the 11×10 grid that is 9 row segments plus column
    x = 10. The mcast is used only when every group needs ≤ 3 rectangles. R3-RS (WIDTH, several owners) keeps the
    unicast all-gather.
  - Fix found while measuring: the first version computed the 2D route per packet and lost about 0.8 µs on the relay /
    notify chains. Headers are now pre-routed once, one per destination chip, and only their command fields are
    rewritten per packet.
  - Reused: the port's raw-fabric send path, the landing layout, the `SEM_OWN_COUNTED` wave gate (counted(w) still
    fires on send item 0 = own segment), the reader's slot reads / expand / inv copy, and the kernel_lib multicast
    helper (as the base op used it).
  - Added: `_exchange_plan`, `_first_hop_dir`, `_rect_groups` / `_segments` / `_row_runs`, `_inv_mcast_config`, the
    port's per-entry counters and pre-routed headers, the reader's per-entry gather loop and multicast branch, and the
    knobs `DIRECT_RING_SENDS`, `NOTIFY_PER_ENTRY`, `INV_MCAST`.
- Performance (traced, 6 back-to-back calls, `test_distributed_rms_norm_perf.py` under `--profile`, 1×4 ring, 2 links,
  bf16, slowest device of the group, median of the traced calls; full six-shape runs, µs):

  | Case | baseline (2 runs) | Refinement 5 (3 runs) |
  |---|---|---|
  | 640×1792 (perf case) | 23.20, 22.61 | 22.04, 21.52, 21.81 |
  | decode 32×1792 | 11.60, 11.40 | 10.84, 11.07, 10.85 |
  | 2×4×128×512 | 17.94, 18.17 | 17.27, 17.36, 17.66 |
  | 1024×2048 | 32.74, 33.42 | 31.69, 32.02, 31.62 |
  | R1-X 2048×1280 | 43.84, 43.11 | 42.81, 43.42, 42.32 |
  | R1-X 4096×1024 | 57.37, 58.36 | 58.12, 59.31, 57.56 (inside the ±1 µs run-to-run band) |

  - The perf case measured alone (640×1792 only, 4 runs each): 22.18 / 22.51 / 22.29 before, 21.76–22.41 after.
  - Lever ablations in the same six-shape harness:

    | Variant | 32×1792 | 640×1792 |
    |---|---|---|
    | all levers on | 10.84–11.07 | 21.5–22.0 |
    | relay instead of direct sends | 11.55 | 21.74 |
    | per-wave notify | 11.31 | 22.53 |

    L6 and L4 both carry the decode win.
  - L5 on the old row-major groups (2–3 rectangles each) was slower than unicast: `rd_dist_send` 1.10 vs 0.80 µs,
    since Counter mode pays a write barrier per rectangle. With single-rectangle groups it is 0.73 µs, and a member
    sees inv 0.54 µs after the root's copy.
  - Sharded guard (`test_perf_ring_sharded`, before → after):

    | Case | before | after |
    |---|---|---|
    | HEIGHT 640 | 19.91 | 19.38 |
    | WIDTH 640 | 26.15 | 25.55 |
    | HEIGHT 32 | 15.80 | 15.72 |
    | WIDTH 32 | 14.89 | 14.09 |
    | HEIGHT 2048 | 22.85 | 23.37 |
    | WIDTH 2048 | 62.77 | 62.16 |
    | HEIGHT 4096 | 27.18 | 25.47 |
    | WIDTH 4096 | 138.09 | 136.09 |

    No regression beyond noise.
- Bottleneck analysis (zones):
  - On the 640×1792 ring the chips launch with 2–4 µs of skew, and the skew grows about 0.4 µs per traced call.
  - On the last-launched chip every peer slot has already landed when its roots reach `back(0)` (landed wait 0.13 µs).
    Its chain is load → combine → publish → front(1) → back(0) (slot reads 0.36 + expand 0.48 + finalize 0.6 + inv
    copy 0.2 + inv mcast 0.5 µs) → output DRAM writes (about 8 µs at about 270 GB/s).
  - The earlier chips wait for that chip's data. In steady state their durations track its self-paced chain plus the
    dispatch-gap drift, so transit latency mostly cancels out of the slowest-device number. The transport levers show
    most where the exchange is the whole chain: decode, and the R3-X shapes with more rows.
- Accuracy achieved: the math is unchanged (bit-identical inv by construction). Golden `check_output` (bf16 PCC ≥ 0.999,
  rel-RMS ≤ 1e-2; fp32 tighter) on every shape in the slice below and on all six perf shapes (traced output checked).
  The new regression test asserts bit-identical outputs for a repeated input on the Ring.
- Golden test progress: targeted slice of 123/123 passed: 108 `test_op` cells (bf16, TILE γ, 2 links, axis 1, every
  shape × {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED}), every `test_op_loose` case (incl. the perf case on the Ring),
  and `test_regression.py` (incl. `test_deterministic`).
- Issues encountered:
  - The per-packet route computation regression, fixed as above.
  - Under `--dev`, re-opening the 1×4 fabric mesh between parametrizations hits the known infrastructure flake
    (`open_mesh_device`, see op_requirements). Each direct-send case passes watcher-clean when run on its own.
- Tests added: `test_distributed_rms_norm_verifier_regressions.py::test_ring_transport_alternating_inputs` (1×4 Ring,
  2 links, forced waves, R1-X and R3-X shapes × look-ahead on/off × direct/relay × inv mcast/unicast; alternating
  inputs plus bit-identical repeats): 16/16. Unit directory (minus the precision matrix): 187 passed, 6 skipped before
  the new test.

## Perf 1
- Date: 2026-10-23
- **Scope**: a perf tournament (round 1 of 2, cap of 2 experiments). Nothing changes in SUPPORTED. The math is unchanged:
  same chains, same fixed g order, and inv is bit-identical group-wide by construction.
- **Focus case**: `LOOSE_CASES[0]`, run with its exact config (all supported): 1×1×640×1792 bf16 TILE, interleaved, γ
  bf16 TILE, fp32_dest_acc_en=True HiFi4, 1×4 Ring, FABRIC_2D_TORUS_XY, 2 links. The plan is R3-X, 100 cores = 10
  row-groups × 10, 2 waves.
- **Measurement**: `test_distributed_rms_norm_perf.py::test_perf_ring` under `--profile`. Each number is the slowest
  device per traced call, median of the 6 traced calls (`perf_experiments/_tools/traced_summary.py`). Zone timelines come
  from `perf_experiments/_tools/zone_report.py`.

### Measured breakdown (baseline, dev 0 of one traced call, µs from kernel start)
| stage | when / cost |
|---|---|
| x DRAM load (reader, NoC0) | 0.5 → 7.2. Wave 0 takes 2.6–3.5 µs per core, evenly across cores (≈ 380 GB/s aggregate burst); wave 1 takes ≈ 1 µs |
| wave-1 publish (root) | 8.4 |
| wave-1 slots landed | 12.1–12.5 (gated by the peer chips: 2–5 µs inter-chip launch skew) |
| landed → inv on every member, per wave | ≈ 2.9: slot reads 0.36, expand 0.48, finalize 0.58, dist copy 0.20, inv multicast 0.73 |
| wave-1 inv on members | 14.3–15.6 |
| output DRAM writes (writer, NoC1) | wave-0 writes take 0.3 µs on west cores but 6–7 µs on the east-most column (phys x = 15, the column row-group). The program ends at 20.0, i.e. 4.4 µs after the last inv |

Cumulative ablation (payload stubbed, synchronization kept): full 20.6 µs → no output writes 17.2 → no writes and no
x reads 13.6.
- Writes cost ≈ 3.5–4 µs and reads ≈ 2.5 µs on the wall.
- The 13.6 µs floor is compute head, in-chip combine, the Fabric exchange including launch skew, and the post-landing
  chain.

**Ranked bottlenecks:**
1. The NoC1 output-write congestion tail (≈ 4 µs).
2. The post-landing stats → inv chain (≈ 2.9 µs per wave, serial).
3. x reads (≈ 2.5 µs; even across cores, near DRAM aggregate, not floated).
4. Exchange / launch skew (not addressable from a kernel).

### Portfolio (2 ideas, one perf-part-optimizer each)
| idea | verdict | bench baseline → candidate | domain |
|---|---|---|---|
| `noc_balanced_writes`: spread the DRAM traffic over both NoCs | **WIN** | single chip, the op's 640×1792 geometry: 15,065 → 11,753 ns (slowest-core write tail 8.9 → 6.3 µs); 2048×1280 32,605 → 26,236 (−20 %); 4096×1024 50,867 → 40,019 (−21 %); 1024×4096 49,601 → 40,416 (−19 %) | every interleaved geometry. Exceptions: the east column holding a single core (decode 32×1792, measured +17 % when swapped); multicast senders (inexpressible) |
| `replicated_finalize`: every member finalizes, no inv hop | **WIN** (variant `fwd2`) | landed → inv on every member, max per wave: 3,420 / 3,231 → 2,780–2,880 / 2,571 ns; overlapping waves 12,179 → 10,670 ns bench; decode 4,187 → 3,176 | every single-root R3-X row-group (one or several rectangles, 1–4 rows per wave). R1-X: no-op. R3-RS: see carve-outs |

Options measured inside the ideas:
- **Nulls / regressions (recorded, not graduated)**, `noc_balanced_writes`:
  - per-tile NoC split (odd tiles on NCRISC / NoC0): 14,173. The west cores become the slowest.
  - per-DRAM-bank shortest hop: 13,323.
  - inverse bank split: 16,345, a regression.
  - read twin (odd x tiles on BRISC / NoC1): null.
  - swapping the multicast roots too: inexpressible.
- **Regressions**, `replicated_finalize`:
  - ports notify every member — multicast inc (`pull`): 5,021 / 4,646 ns; multicast set: 4,497 / 4,662; unicast: 4,357 / 4,436.
  - ports multicast the slot data (`push`): 8,649 / 10,036.
  - `relay+p` (the root relays landed counts, members read the ports): 2,832 / 2,773. A WIN, but slower than `fwd2`.
  - Pipelined root gather alone (`baseline+p`): 3,160 / 2,992. It is subsumed by `fwd2`.
- **No precision-trading option was floated or graduated.**

### Graduated
Both wins touch different stages (output-write NoC vs. the stats → inv chain), so both graduated, combined.

1. **`replicated_finalize` (`fwd2`).** It is the one path for every single-root R3-X row-group:
   - interleaved R3-X;
   - single-owner WIDTH_SHARDED, e.g. 32×1792;
   - groups needing several rectangles: the forward takes any number of rectangles, bounding box or row runs.

   Code deleted:
   - the root's finalize-for-the-group → `dist_copy` → kernel_lib McastFamily inv multicast → member expand path;
   - the `INV_MCAST` knob, `_inv_mcast_config` and `_row_runs`;
   - the McastFamily attach / `attach_absent`.

   Code added:
   - `SEM_OWN_FWD` / `SEM_SLOT_FWD` (`COMMON_CT_ARGS` 26 → 28; reader / writer accessor offsets 28);
   - the root's linked multicast forward (`forward_and_signal`) and the member gather;
   - `_member_rects` / `_fwd_rect_args`, which order rectangles for the root reader's NoC;
   - compute `fin_rows`: every core finalizes all `R_b` rows into `cb_inv_rms`;
   - every root's slot gather is now pipelined (reads issued on notice, batch expanded once its reads complete), R1-X included;
   - the staging words for the semaphore sets live in the now-unused `cb_inv_root` page (no new L1).

   `RECT_GROUPS` (default True) keeps the single-rectangle group shape. `test_ring_transport_alternating_inputs` is
   re-parametrized from `INV_MCAST` to `RECT_GROUPS`, so the multi-rectangle forward stays covered.

   **Carve-out: R3-RS (several owners, WIDTH_SHARDED with `OWNERS > 1`)** keeps the owners' unicast inv all-gather.
   - Reason: the graduated protocol has ONE forwarder per group (one slot mask, one own-publish count). With several
     owners, every owner would forward its own rows' slots, which is a different protocol: per-owner semaphores, and an
     owner inside its own rectangles needs loopback-excluded multicasts.
   - It is inexpressible in this protocol, and unmeasured. Follow-up: measure a multi-forwarder variant (round 2 or
     later).
2. **`noc_balanced_writes`.**
   - The compute cores of the 3 westmost logical columns (`NOC_SWAP_COLUMNS = 3`) and of the east-most column get a
     second reader / writer descriptor pair (`noc_swap_cores`, `_split_noc_swapped`), with the reader on NoC1 and the
     writer on NoC0. Kernel sources and args are unchanged, DM_DEDICATED_NOC.
   - Adapted from the base op's Perf 2 `dram_noc_balance`.
   - Carve-outs:
     - (a) The east column swaps only with ≥ 2 compute cores (`NOC_SWAP_EAST_MIN_CORES = 2`). Measured: a lone core
       there (decode 32×1792) was +17 % swapped, vs −3.6 % without it.
     - (b) FWD_INV roots never swap (inexpressible). Their multicast rectangles are packed for the reader's `DM_NOCS`
       NoC.
     - (c) No swap when x AND the output are both shard-resident: there is no DRAM traffic, so the swap has no meaning.
   - Untested but covered by the unified path: L1-interleaved inputs and sharded inputs with a DRAM output.

### Whole op, before → after (traced median, slowest device, µs)
| case | baseline | fwd2 only (swap disabled by knob) | **Perf 1 (both)** |
|---|---|---|---|
| **640×1792 (focus)** | 22.08 (21.37–23.85) | 21.33 | **19.15** (18.34–20.95); final re-run 18.80 (17.92–20.59) |
| decode 32×1792 | 10.92 | 10.00 | 10.30 (9.87–10.40) |
| R1-X 2048×1280 | 43.20 | 42.37 | 40.70 |
| R1-X 4096×1024 | 58.34 (55.8–61.1) | 60.10 (56.0–61.2, inside the run-to-run band) | 55.00 |
| 2×4×128×512 | 17.19 | 16.48 | 16.15 |
| 1024×2048 | 31.97 | 30.69 | 27.36 |

The focus case is −13 % to −15 %, against a 14.1 µs target. The output-write wave-0 hotspot moved: the east column now
takes 0.3–0.7 µs, and the worst remaining core is about 4.6 µs (phys x = 12–13, y = 5–9), a candidate for round 2.

**Sharded guard** (`test_perf_ring_sharded`, before → after, µs):

| case | before | after |
|---|---|---|
| HEIGHT 640 | 19.27 | 19.33 |
| WIDTH 640 | 26.04 | 25.46 |
| HEIGHT 32 | 15.75 | 15.72 |
| WIDTH 32 (single owner: now `fwd2`) | 14.45 | **12.55** |
| HEIGHT 2048 | 23.21 | 22.40 |
| WIDTH 2048 | 62.22 | 61.66 |
| HEIGHT 4096 | 25.58 | 25.94 |
| WIDTH 4096 | 136.26 | 134.78 |

No regression beyond noise.

### Correctness
- **Golden `test_op`**: all 36 shapes, run per shape. 10,760 passed, 0 failed, 2,160 strict xfails as expected,
  33,736 skipped (INVALID / infeasible, including the known fp32 32×8192 / 128×8192 INFEASIBLE_L1 cells).
- **`test_op_loose` (incl. the perf case) + `test_regression` + `test_ring_mock` + `test_translated`**: 317 passed, 51
  skipped, 1 failed. The failure is `test_translated::test_ds_distributed_norm[random_weights-prefill-131072-0.98]`
  (L1: footprint(R=1) > budget). It is **pre-existing**: it fails identically on HEAD with Perf 1 stashed.
- **Unit directory** (main, sharded, layouts, verifier regressions, precision baseline): 202 passed, 6 skipped.
  `test_ring_transport_alternating_inputs` (R1 / R3 × look-ahead × direct / relay × rect / run groups, alternating
  inputs, bit-identical repeats): 16 / 16.

### Instrumentation
`DeviceZoneScopedN` zones; `perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree, so the op
keeps the equivalent macro.
- New: `rd_gather` (the root's whole pipelined gather), `rd_fwd` (each forward chain), `rd_slot_expand` (root expand
  work), `rd_mbr_gather` (the member's wait plus expand), `rd_mbr_expand` (member expand work).
- Retired with their code: `rd_landed_wait` / `rd_slot_reads`, which are folded into `rd_gather`; the R3-RS path keeps
  `rd_wait_invroot` / `rd_dist_copy` / `rd_dist_send` / `rd_inv_wait`.
- The NoC-swapped cores run the same kernel sources, so they carry the same zones.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| kernel_lib `McastFamily` / `McastArgs` (sender) | capability | The data-ready signal cannot carry a growing value. Counter mode issues a write barrier before its per-round atomic, so it can't close a linked data chain without that stall. Flag mode writes only a fixed VALID word, so it can't publish the per-block slot mask `(block+1) << 24 \| mask` that lets members expand slots as they arrive. The forward needs "N linked multicast writes, then a multicast SET of an arbitrary caller value to the same rectangle", per rectangle. | 3,420 / 3,231 (landed → inv on every member, per wave, previous McastFamily inv distribution) | 2,780–2,880 / 2,571 (`fwd2` raw forward) | `kernels/distributed_rms_norm_reader.cpp:362` (`forward_and_signal`, multicasts at :377 / :379) |

The two ns columns are protocol-level: the helper-based inv distribution against the raw slot forward, on the same
bench. A same-protocol helper version does not exist (that is the gap). The kernel-head comment at
`distributed_rms_norm_reader.cpp:54` carries the same justification.

- **Artifacts**:
  - `perf_experiments/noc_balanced_writes/`: bench, variant kernels, logs. Test:
    `tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_experiments/noc_balanced_writes/`.
  - `perf_experiments/replicated_finalize/`: bench, `rf_*` kernels. Test: `tests/.../perf_experiments/replicated_finalize/`.
  - `perf_experiments/_tools/`: zone timeline and traced-call summarizers.
- **Result**: 2 ideas measured, 2 graduated, 0 null at the idea level (5 sub-options null or regressed, recorded above).
  The op is 13–15 % faster on the focus case (22.08 → 18.80–19.15 µs) and 5–14 % faster on the rest of the interleaved
  guard set. No regression.

## Perf 2

**Focus:** the perf-flagged LOOSE case, exactly as declared:
- per device (1, 1, 640, 1792);
- bf16 x and γ, TILE, `fp32_dest_acc_en=True`, HiFi4;
- DRAM interleaved, `cluster_axis=1`, Ring, 2 links, (1, 4) `FABRIC_2D_TORUS_XY`.

Regime: R3-X, 10 row-groups × `G_w` = 10 cores, S = 6, two waves of one tile-row each. Every number below is the
slowest device per traced call, median of 6 calls, from `test_perf_ring` / `test_perf_ring_sharded` under `--profile`.

### Measured breakdown (baseline = Perf 1 head)
**Whole op:** 18.94 µs, range 17.97–20.68 (a repeat run gave 18.91). Device 0's span is 17.4–17.8 µs.

**Cumulative ablation** (payload stubbed, synchronization kept):
| ablation | µs |
|---|---|
| full | 18.94 |
| output writes stubbed | 16.80 |
| output writes and x reads stubbed | 14.09 (device 0: 11.4) |
| γ reads stubbed alone | 18.71 (flat) |

So DRAM writes cost ≈ 2.1 µs on the wall and x reads ≈ 2.7 µs. The remaining ≈ 14 µs is latency: compute head, in-chip
combine, Fabric exchange, and the per-wave post-landing chain.

**Device 0 timeline, full op** (µs from kernel start):
- **wave-0 x load:** 0.5–3.0, DRAM-bound at ≈ 480 GB/s.
- **root's wave-0 publish:** 5.4.
- **root's wave-1 x load:** 5.6–6.3, while members load wave 1 at 4.2–5.1. The root's NCRISC runs `front(1)` only
  after `front(0)`'s combine and publish.
- **root's wave-1 publish:** 8.3.
- **wave-0 slots landed:** ≈ 7.4. The root's wave-0 gather nevertheless runs 8.5–10.0, and its wave-1 gather 10.1–11.6.
- **first output write:** 11.7.
- **output writes:** 11.7 → 17.4, a 5.7 µs burst at ≈ 400 GB/s. The east columns (phys x = 12–13) take 4.8 µs per
  6-tile quantum.

**Per-wave post-landing chain, no-DRAM run:**
- root gather 1.5 µs: reading the landed slots 0.5, forwarding them 0.4, expanding them 0.45;
- member expand: 0.21 µs per slot;
- compute: finalize 0.6 + inv round trip 0.35 + scale / pack 0.7.

**Ranked bottlenecks:**
1. Root NCRISC serialization: `front(w+1)` delays the wave-1 publish and the wave-0 gather.
2. The per-wave post-landing chain (≈ 3.3 µs per wave).
3. The DRAM write burst after inv, which nothing overlaps.
4. x reads.
5. γ reads: ablation flat, so not floated.

**Roofline gate** (`roofline.py`: DRAM 9.0 µs + exchange 1.6 µs = 10.6 µs; target 14.1 µs):
- x reads run at about the DRAM ceiling.
- The write burst runs at ≈ 80 % of it, so only overlap or an earlier start can move it.
- Neither half was floated as a coalescing or NoC idea.

### Portfolio (cap 2; one perf-part-optimizer each)
| idea | verdict | focus µs, base → candidate | domain / notes |
|---|---|---|---|
| `early_next_load`: issue wave w+1's x reads ahead of wave w's in-chip combine / publish | **NULL** | 18.96 (12 runs) → 19.15 (`aftpart`, 10 runs); `aftdefer` 18.82; `hoist` 21.19 (regression); `async` 19.73 (regression); `rootasync` 19.0–19.4 | Device 0's root wave-1 publish moved 7.6 → 7.1 and its wave-0 gather 8.6 → 8.1, and device 0 ends 17.5 → 16.6. The slowest device is still gated by the last peer's wave-0 publish plus the DRAM write tail. Early wave-1 reads also steal bandwidth from wave 0. 2×4×128×512: −3 %; the rest null. Not graduated |
| `no_col0_expand`: stats slots stay in their compact wire form; compute sums / finalizes 2 SFPU vectors, then moves inv to column 0 itself | **WIN** | 19.21 → 18.83 (n = 24 each, same session) | Every shape faster: 32×1792 10.23 → 9.15 (−10.6 %); 2048×1280 41.75 → 40.48; 4096×1024 55.35 → 54.63; 2×4×128×512 15.96 → 15.31; 1024×2048 27.31 → 26.65. bf16 DEST: 19.25 → 18.65, 10.03 → 9.35. Outputs bit-identical to the column-0 path on 18 cases: R1 / R3, 1 / 2 / 4 waves, ragged slice, bf16 / fp32 / bf8b, fp32 accumulation on / off, HEIGHT / WIDTH sharded. No exceptions |

Options inside `no_col0_expand`:
- **Wire form = compute form + in-DEST transpose + SFPU lane move:** graduated.
- **Row 0 as two 64 B pieces:** rejected unbuilt; it doubles the NoC reads and multicasts.
- **srcA unpack transpose:** rejected unbuilt; it truncates the fp32 partials to 19 bits, a precision cost.
- **No precision-trading option was graduated.**

### Graduated: `no_col0_expand`, one unqualified path, no predicate
- **Reader** (`kernels/distributed_rms_norm_reader.cpp`):
  - the root's publish writes column 0 straight into the 32 wire words at the page base (`col0_to_wire`);
  - port reads and the FWD_INV slot forward target the page base;
  - deleted: `rd_slot_expand` and the root's `tail_to_col0` loop, the member's `rd_mbr_expand` loop, and the
    `l1_acquire` that preceded them;
  - the member now only waits for every slot's bit.
- **Compute** (`kernels/distributed_rms_norm_compute.cpp`):
  - added `AddRow01` / `FinalizeRow01`, the raw sfpi bodies on vectors 0–1 (`VectorMode::RC_custom`);
  - added `Row01ToCol0` (`transpose_dest<DST_ACCUM_MODE, true>`, then the face 0 column 1 → face 2 column 0 move);
  - deleted `AddCol0` / `FinalizeCol0`.
- **R3-RS (WIDTH_SHARDED, several owners):** the same slot path. Its owner inv all-gather still uses
  `col0_to_page` / `tail_to_col0` on the finalized inv, which is now column-form from `Row01ToCol0`. No carve-out:
  measured correct.
- **Host, port and writer:** unchanged.
- **Carve-outs:** none.

### Whole op, before → after (µs)
Before = 2 runs at Perf 1 head this round; after = 4 runs.

| case | before | after | |
|---|---|---|---|
| **640×1792 (focus)** | 18.91 / 18.94 | 18.80 / 18.85 / 19.45 / 18.20 (≈ 18.8) | ≈ −0.1, inside the ±0.5 noise. Same-session pooled (optimizer, n = 24): 19.21 → 18.83 |
| decode 32×1792 | 9.99 / 10.11 | 9.28 / 9.05 / 9.06 / 9.01 | **−1.0 (−10 %)** |
| R1-X 2048×1280 | 40.38 / 41.11 | 40.48 / 41.22 / 42.73 / 41.25 | flat |
| R1-X 4096×1024 | 56.01 / 56.32 | 57.03 / 56.47 / 55.22 / 54.33 | flat |
| 2×4×128×512 | 14.64 / 16.10 | 15.04 / 15.15 / 15.29 / 15.09 | flat |
| 1024×2048 | 27.73 / 27.11 | 27.59 / 26.75 / 27.14 / 27.02 | flat |

**Device 0 timeline, focus shape (optimizer, medians):**
- root `rd_gather` per wave: 1.52 / 1.54 → 1.04 / 1.03 µs;
- compute end: 14.24 → 13.30;
- span: 17.5 → 16.8.

The slowest device moves less. The kernel end is now bound by the DRAM write burst, which starts ≈ 0.7 µs earlier but
ends about the same time, and by the last peer's wave-0 publish.

**Sharded guard** (`test_perf_ring_sharded`, before → after, µs):

| case | before | after |
|---|---|---|
| HEIGHT 640 | 19.11 | 18.51 |
| WIDTH 640 | 25.15 | 24.71 |
| HEIGHT 32 | 15.72 | 15.54 |
| WIDTH 32 | 12.49 | 12.21 |
| HEIGHT 2048 | 22.38 | 22.20 |
| WIDTH 2048 | 61.80 | 60.93 |
| HEIGHT 4096 | 25.80 | 24.81 |
| WIDTH 4096 | 134.62 | 134.15 |

No regression.

### Correctness
- **Golden `test_op`**, all 36 shapes, in 11 batches: 0 failed (every batch green; e.g. 640×1792 alone 301 passed /
  60 strict xfail).
- **`test_op_loose`:** 3 / 3 passed.
- **`test_regression`:** 36 passed.
- **`test_translated` + `test_ring_mock`:** 302 passed, 1 failed. The failure is
  `test_ds_distributed_norm[random_weights-prefill-131072-0.98]` (L1: footprint(R=1) > budget). It is **pre-existing**
  and recorded in Perf 1 as failing on HEAD.
- **Unit directory** (main, sharded, layouts, verifier regressions, precision baseline): 202 passed, 6 skipped.

### Instrumentation
- The op keeps `DeviceZoneScopedN`; `perf_instrumentation.hpp` / `MaybeDeviceZoneScope` is absent from this tree.
- **New port zones:** `prt_send` (each item sent) and `prt_notify` (each landed-entry notify to the roots). They time
  the Fabric exchange per entry.
- **Retired with their stages:** `rd_slot_expand` and `rd_mbr_expand`. `rd_mbr_gather` now times the member's wait only.
- **Finalize:** the transpose and lane move run inside `cmp_finalize`.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| kernel_lib `eltwise_chain` SFPU ops (`AddBinary` SFPU / `Rsqrt` / `MulUnary` / `AddUnary`) | capability | No chain op can limit an SFPU op to a chosen subset of DEST vectors (here vectors 0–1 of a tile, one call at the tile base with `RC_custom`), or fuse ·1/W + ε + rsqrt into one pass. The stock ops walk all 4 faces (`VectorMode::RC`). | 19,210 (focus, column-0 path with the RISC-V expansion it required) | 18,830 (`AddRow01` / `FinalizeRow01` + `Row01ToCol0`) | `kernels/distributed_rms_norm_compute.cpp:79` (`add_row01_body`), `:93` (`finalize_row01_body`) |
| no chain op wraps `transpose_dest`; nothing moves SFPU vectors between faces | capability | Putting inv in column 0 needs a 32×32 in-DEST transpose plus a face 0 col 1 → face 2 col 0 lane move inside the same DEST window as the finalize. The chain has no DEST-transpose element and no inter-face SFPU move. The srcA unpack-transpose alternative truncates fp32 to 19 bits. | — (no helper form exists) | included in the row above | `kernels/distributed_rms_norm_compute.cpp:107` (`col1_to_face2_body`), `:146` (`Row01ToCol0`) |

The ns pair is protocol-level: the previous column-0 form plus the RISC-V expansion it needed, against the wire-form
path, focus shape, same session, n = 24. The kernel-head comment in `distributed_rms_norm_compute.cpp` carries the same
justification. The Perf 1 `McastFamily` bypass in the reader is unchanged.

### Artifacts
- `perf_experiments/no_col0_expand/`: `v_base` / `v_cand` kernels, diffs, run scripts, logs. Test:
  `tests/.../perf_experiments/no_col0_expand/test_no_col0_expand.py`.
- `perf_experiments/early_next_load/`: every `EARLY_MODE` variant, `v_graduate` reader, diff, tools, logs. Test:
  `tests/.../perf_experiments/early_next_load/test_early_next_load.py`.
- `perf_experiments/_tools/`: added `core_timeline.py` (per-core zone timeline) and `sharded_summary.py`;
  `zone_report.py` gained `--only RUN`.
- Committed: sources, diffs, run scripts, and the summary logs under 64 KB. The raw device-profile CSVs and pytest logs (≈ 13 GB) stay on
  disk uncommitted.

### Result
2 ideas measured: 1 graduated, 1 null.
- **Focus 640×1792:** flat on the slowest device (≈ 18.8 µs, −0.1 to −0.4). Device 0 is −0.7 µs, but the slowest device
  is bound by the DRAM write burst and peer launch skew.
- **Decode 32×1792:** −10 %.
- **Sharded guard:** 0.2–1.0 µs faster on all 8 cells.
- **No regression.**

**Next lever:** start the output-write burst earlier, or overlap it with the reads. That means cutting the per-wave
post-landing chain further (compute finalize → inv → scale ≈ 1.7 µs per wave) or adding waves, since the DRAM is idle
between the last x read and the first write.
