# mhc_post_ttnn (ttnn.bringup.mhc_post)

Source: ai-generated (tt_ops_code_gen eval run 1046, branch `2026_09_30_0115_run1_mhc_post` @ `84be38c18b7`, started from
`mstaletovic/mhc-codegen` @ `5066bfadf95`): the op folder `ttnn/ttnn/operations/mhc_post` and its unit suite
`tests/ttnn/unit_tests/operations/mhc_post`. Left out: the run's `perf_experiments/` scripts, `agent_logs/` and the
agents' scratch probes (all on the source branch). The op's own history is in `changelog.md` (design, refinements,
two perf-tournament rounds); its golden suite is `eval/golden_tests/mhc_post` on tt_ops_code_gen `mstaletovic/mhc-goldens`.

1. ai-generated perf optimized version (Python ProgramDescriptor host side), moved to ttnn/ttnn/bringup/mhc_post_ttnn (the folder is not
   named mhc_post: a package of the op's own name would shadow ttnn.bringup.mhc_post): imports `ttnn.operations.mhc_post` ->
   `ttnn.bringup.mhc_post_ttnn`; `kernel_lib/perf_instrumentation.hpp` (on the source branch, not
   in this tree) ships as `kernels/perf_instrumentation.hpp`, as rms_norm_ttnn's does. Kernels otherwise unchanged.
   Unit suite in `tests/unit/`: passes on this tree's kernel helper library.
   - Needed by: glm53_flash_d_p (mHC: attn / ffn residual, 4 streams, C 4096).
2. C++ host side: `ttnn.bringup.mhc_post` is now a C++ TTNN op (the Python op stays importable as
   `ttnn.bringup.mhc_post_ttnn.mhc_post`, the parity reference). Kernels untouched.
   - What: `device/mhc_post_ttnn_program_factory.cpp` is a line-for-line port of `create_program_descriptor()`
     (row-weighted work split, L1 block solve, read help, CBs, CT / RT args, NoC modes, semaphores); a
     ProgramDescriptor factory (build on a cache miss, `override_runtime_arguments` patches the five buffer-address
     slots on a hit); `MhcPostDeviceOperation` + `ttnn::prim::bringup::mhc_post_ttnn` (program hash = the four
     tensor specs + the compute-config fields the builder reads); `mhc_post_ttnn.cpp` keeps `validate()`'s checks,
     order and exception types (UnsupportedAxisValue, ValueError); `_mhc_post_ttnn_program_descriptor` for the test.
   - Why: a Python ProgramDescriptor op pays its host build on every call. At GLM-5.3's shape (T 5120, C 4096,
     n 4, bf16) the host time per call went 0.676 -> 0.019 ms; device-bound wall 1.195 -> 1.161 ms per call.
   - Tests: `tests/unit/test_mhc_post_cpp_parity.py` (15 program-descriptor parity cases incl. GLM's shape, 5
     bit-identical device outputs incl. a program-cache hit on moved buffers); the unit suite runs on the C++ op
     (the knob tests stay on the Python builder, the only one with the knobs): 110 passed.
   - Needed by: glm53_flash_d_p.
3. The eval golden suite ships in `tests/golden/` (from tt_ops_code_gen `mstaletovic/mhc-goldens`; imports pointed at
   this folder, dispatching the C++ op): passes.
4. Model cases (bringup-fork-tests, task O.1): `tests/cases.py` (1 glm53_flash_d_p call(s), captured with
   GLM_MHC_IMPL=fused on the s56320 rung), `tests/reference.py` (float64 torch semantics), `tests/test_mhc_post_ttnn.py`
   (random per-chip inputs on the 2x2 mesh, PCC + rel L2 per output; limits from the measured error with margin).
5. Opt-in `comb_transposed=False` (default True keeps the op's comb^T math, bit-identical program: no define added).
   - What: `X'_j = post_j F + sum_i comb[j*n + i] X_i` (comb applied as stored, row-major comb[j][i] for output j).
     Only the coefficient expansion changes: `kernels/mhc_post_coef_expand.hpp` reads raw column `j*n + (t-1)`
     under the kernel define `MHC_POST_COMB_DIRECT` (added to both DM kernels by the program factory when the option
     is off). `MhcPostParams::comb_transposed` is in the program hash; `mhc_post` / the prim / the program descriptor
     / the nanobind (`comb_transposed=True` kw-only, also on `_mhc_post_ttnn_program_descriptor`) carry it. The
     Python builder (parity reference) has no such option; the parity suite only builds the default.
   - Why: Xing4.0's mHC residual is `post * y + comb @ x` (`xing40_a4b_d_p/reference/xing_ref.py:hc_residual`), not
     GLM's comb^T. Replaces the 20-op addcmul chain (90 ms per 5120-token chunk each for attn / ffn residual).
   - Files: kernels/mhc_post_coef_expand.hpp, kernels/mhc_post_common.hpp (comment), device/*_types.hpp,
     device/mhc_post_ttnn_device_operation.{hpp,cpp}, device/mhc_post_ttnn_program_factory.{hpp,cpp},
     mhc_post_ttnn.{hpp,cpp}, mhc_post_ttnn_nanobind.cpp; tests: cases.py (xing40 case, comb_transposed False),
     reference.py (comb_transposed), test_mhc_post_ttnn.py (passes the option; checks the comb and comb^T
     references differ; the mesh fixture opens the case mesh matching the box's device count, since a 2x2 submesh of
     a 4x2 box fails the 2D fabric handshake).
   - Needed by: xing40_a4b_d_p P.2 (tt/residual.py, XING_RESIDUAL_MIX=fused, the default).

6. Alias removal: `tt::tt_metal::MathFidelity` / `UnpackToDestMode` (#59097 removed the global aliases; the fork did
   not build). Default behaviour unchanged.
   - Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
   - Files: `mhc_post_ttnn.cpp`, `device/mhc_post_ttnn_program_factory.cpp`

7. `mhc_post(..., stats_weight=W)`: the next hc step's partial row from the same pass (deepseek_v41_flash_d_p P.14,
   owner's fused-mHC comparison).
   - What: a keyword option `stats_weight` (default None: the op, its program and its one-tensor return are unchanged).
     Given W (..., n C, 32) fp32 (the chip's slice of hc_fn^T, leading dims 1), a separate program (`device/
     mhc_post_stats_*`, `kernels/mhc_post_stats_*`) returns (X', stats): X' bit-identical to the op without it (per
     data vector every input loaded once, each output acc = F post_k then fused multiply-adds X_j w_jk in j order:
     WeightedSum's arithmetic), and stats (..., T, 32) fp32 = X' @ W (FPU matmul_tiles, X' = in0 / srcB, W = in1 / srcA
     as ttnn.matmul(x, W); fp32 DEST; compute_kernel_config's fidelity) with column n (n + 2) replaced by sum_c X'^2
     (exact fp32 SFPU) = mhc_pre_xing_pack(X' @ W, X'). Work: column blocks of B tiles (planner: the busiest core's
     units x (B + a per-unit overhead), B <= 8), units block-major so a core keeps its W block in L1; per column one
     DEST window (the unit's coefficient tiles, 3 a half, built by the writer in the SFPU lane layout from the post /
     comb row tiles; F + X_0..X_3 by UnpackToDestFp32; outputs and the column's sum of squares written in place), the
     unit's partial (FPU matmul over its X' tiles, its sum-of-squares tile re-laid coefficient-major and summed) to a
     DRAM scratch page; the row's root core (semaphore count) sums the row's NB partials in block order. Needs fp32 F
     and X, n <= 4.
   - Why: deepseek_v41_flash_d_p's next hc step read the new streams twice (the hc_fn matmul and mhc_pre_xing_pack's
     sum of squares) right after the residual wrote them. 2x2 p150, [1024, 4 x 2560] fp32 a chip, HiFi2 (wall, 30
     synced calls): mhc_post 0.270 ms, with stats_weight 0.327 ms, vs mhc_post + ttnn.matmul + mhc_pre_xing_pack
     0.638 ms. Model (MHC_FUSED=on, 2x2, chunk 53248-55296): ffn_hc 3.2 -> 1.44 ms a chunk, attn_residual /
     ffn_residual 2.44 / 3.10 -> 2.94 / 3.27 ms.
   - Measured precision (6 cases, random inputs): stats vs the unfused device path, projection rel L2 <= 2.3e-6, sum
     of squares <= 1.2e-7; vs float64 the projection equals the unfused path's own TF32 / srcA error.
   - Limits: the stats program is DRAM-bound; its W blocks add ~13 % reads at B = 8 (a W multicast to the cores of a
     block would remove them). Block widths below 8 lose to the per-unit work (B = 4 0.352, 2 0.487, 1 0.909 ms).
   - Tests: `tests/unit/test_mhc_post_stats.py` (DS41 2x2 / 4x4 shapes, small, ragged blocks (Ct = 13), n = 2,
     comb as stored at HiFi4: X' bit-identical, the limits above, columns past n (n + 2) zero, negative controls (a
     0.1 % column error, a dropped column tile), a program-cache hit, refusals). Regression, option off (2026-10-20,
     2x2): unit + golden + model cases 382 passed (= the recorded unit 164, golden 208, model cases 2, + the 8 new).
   - Needed by: deepseek_v41_flash_d_p P.14 (tt/residual.py:TtHcResidual.fused, tt/engram_add.py, setting MHC_FUSED).
   - Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `sources.cmake`,
     `device/mhc_post_stats_device_operation.{hpp,cpp}`, `device/mhc_post_stats_program_factory.cpp`,
     `kernels/mhc_post_stats_{common.hpp,reader.cpp,writer.cpp,compute.cpp}`, `tests/unit/test_mhc_post_stats.py`.

8. `post_col` / `comb_col` (post / comb at column offsets, e.g. the packed mixes) and a bf16 F with `stats_weight`
   (deepseek_v41_flash_d_p P.19).
   - What: keyword options `post_col=0`, `comb_col=0`: post_k is post's column post_col + k, comb[i*n + j] comb's
     column comb_col + i*n + j; nonzero, n is the residual's stream count (residual last dim / C) and post / comb may
     be one wider tensor (post=comb=mixes [pre | post | comb], post_col=n, comb_col=2n). Both programs: the plain one
     (DM defines MHC_POST_POST_COL / MHC_POST_COMB_COL in the coefficient expander) and the stats one (the same
     defines in its writer's coefficient build); the hash gets the offsets only when nonzero. With `stats_weight` the
     input F may now be bfloat16: its own CB (c_7, bf16 pages, UnpackToDestFp32: bf16 -> fp32 DEST exact; reader and
     compute define MHC_POST_STATS_F16, two CT args appended; the F slots of c_0 stay unused); fp32 F builds the
     program it always did. Offsets 0 / fp32 F: every program, CT / RT arg, define and hash unchanged.
   - Why: deepseek_v41_flash_d_p sliced post and comb out of the [R, 24] mixes before every residual (2 slices, 3
     programs a chip each) and typecast the bf16 attention / MoE sums to fp32 for the stats pass (16 typecasts a
     chunk). 2x2 p150, [1024, 4 x 2560] fp32 streams, HiFi2 (30 synced calls): stats with fp32 F 0.328 ms, with bf16 F
     + offsets 0.314 ms (and no typecast 0.046 ms / no 2 slices); plain 0.269 / 0.268 ms. Model (GLUE_FOLD=on, plain
     profile, chunk 53248-55296): attn_residual 3.31 -> 2.52 ms, ffn_residual 3.29 -> 2.88 ms, moe_add 1.58 -> 1.22 ms
     a chunk.
   - Tests: `tests/unit/test_mhc_post_mixes.py` (DS41 2x2 / 4x4 shapes and a small ragged one, comb^T and as stored,
     bf16 and fp32 F: X' and stats bit-identical to the sliced / typecast calls; negative controls one column off;
     refusals). Regression, options off (2026-10-22, 2x2): unit 191 (= 172 recorded + 19 new) + golden 208 + model
     cases 2 = 401 passed.
   - Needed by: deepseek_v41_flash_d_p P.19 (tt/residual.py, tt/model.py moe_add, setting GLUE_FOLD).
   - Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_ttnn_device_operation*.{hpp,cpp}`,
     `device/mhc_post_ttnn_program_factory.{hpp,cpp}`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
     `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_coef_expand.hpp`,
     `kernels/mhc_post_stats_{reader,writer,compute}.cpp`, `tests/unit/test_mhc_post_mixes.py`.

9. bf16 residual streams with `stats_weight` (deepseek_v41_flash_d_p P.20).
   - What: with `stats_weight`, the residual X may be bfloat16 when F is bfloat16 (was refused: fp32 X only; the
     plain program always took bf16 X). X tiles in bf16 pages (CB_IN, UnpackToDestFp32: exact in the fp32 DEST), X'
     written bf16 (CB_OUT, the packer's one rounding of the fp32 mix) and kept bf16 for the projection (CB_XS: the
     matmul reads the bf16 X' the next hc step reads), the sum of squares from the fp32 mix in DEST (before its
     rounding: rel <= ~2e-5 from the bf16 X'^2 sum, inside the fp32 statistic's noise); srcA / the packer
     reconfigured where the CBs alternate fp32 / bf16. Define MHC_POST_STATS_X16 (reader, writer, compute), present
     only with bf16 X; fp32 X: every program, CT / RT arg, define and hash unchanged.
   - Why: DeepSeek's GPU code keeps the mHC residual streams in bf16 (hc_post(...).type_as(x)); the stats pass is
     DRAM-bound (DS41-80), so bf16 streams halve its reads and writes. Model (BF16_ACTIVATIONS=on, 2x2, chunk
     53248-55296, plain profile vs FIX.10's): attn_residual 2.50 -> 1.80 ms, ffn_residual 2.87 -> 2.44 ms a chunk.
   - Tests: `tests/unit/test_mhc_post_bf16_streams.py` (DS41 2x2 / 4x4 shapes and a small ragged one, sliced post /
     comb and the packed mixes: X' = the fp32-X call's X' rounded to nearest (ties away) bit for bit, the sum-of-squares
     column bit-identical to it, the plain program's X' equal, the projection vs float64 of bf16(X') @ W within 1.1 x
     the fp32-X call's own error (measured 4.7e-3 vs 7.3e-3 at HiFi2); control: the fp32 X' projection is farther;
     refusal of a bf16 X with an fp32 F). Regression, option off (2026-10-22, 2x2): unit + golden + model cases 408
     passed (= the recorded 401 + 7 new).
   - Needed by: deepseek_v41_flash_d_p P.20 (tt/residual.py, tt/engram_add.py, setting BF16_ACTIVATIONS).
   - Files: `mhc_post_ttnn.cpp`, `device/mhc_post_stats_program_factory.cpp`,
     `kernels/mhc_post_stats_{reader,writer,compute}.cpp`, `tests/unit/test_mhc_post_bf16_streams.py`.

5. Model cases for deepseek_v41_flash_d_p (O.1, 2026-10-23): 3 calls (bf16 streams; the engram add's stats call, the
   packed-mixes call, the packed-mixes stats call at HiFi2), run by test_mhc_post_ds41; every output within
   noise_limit at its formats (X' 1.66e-3 vs 3.3e-3; the projection with the FPU's HiFi2 operand bits,
   reference.fpu_operands: 4.06e-3 vs 8.2e-3; sum of squares 5.9e-8 vs 1.0e-7). No op change.
   - Needed by: deepseek_v41_flash_d_p O.1
   - Files: `tests/cases.py`, `tests/reference.py`, `tests/test_mhc_post_ttnn.py`.

### Rebase onto llk_helper_library 6b80daf43e3: kernel API port (2026-10-27)
- What: `kernels/perf_instrumentation.hpp` includes `api/debug/kernel_profiler.hpp` (#60172); the chain elements of
  `mhc_post_compute.cpp` (`CoefCopy` / `SublayerCopy` / `ResidualCopy` / `PackOutput`) take a `DataflowBuffer` for
  their CB ahead of the tile base (#60306). Each buffer is declared where its element is built; that frame is live
  until the chain runs at the bottom of the recursion (elements keep a pointer to it). No behaviour change.
- Why: `mhc_post_compute` and `mhc_post_dm` no longer compiled.
- Needed by: rebase 2026-10-27
- Files: `kernels/perf_instrumentation.hpp`, `kernels/mhc_post_compute.cpp`

### output_tensor / stats_output / stats_scratch: write a caller's output buffer (2026-10-30)
- What: `output_tensor=`, `stats_output=`, `stats_scratch=` (default None = the op allocates them, unchanged): the caller's buffers for X', the stats row and (with `stats_weight`) the partials scratch ([1, 1, R, 32 blocks] float32 TILE DRAM). Each must have exactly the spec the op would allocate (checked on every call) and must not be an input. Shared helper `device/mhc_post_given_output.hpp`. Kernels and programs unchanged; outputs byte-identical to the allocating call.
- Why: each call allocated its output (~35 us of host). The model passes load-time buffers at the call sites whose
  output is read and dropped inside the layer. Measured: mhc_post 137.4 -> 101.4 us of host a call (8x4, 82 calls a chunk).
- Tests: `tests/unit/test_mhc_post_output_buffers.py` 3 passed on the 8x4. Model-case regression `tests/test_mhc_post_ttnn.py`: 206 passed (5 first errored in the batch, passed on re-run). `tests/golden/` does not import here (no `eval` package, DS41 known issue).
- Needed by: deepseek_v41_flash_d_p P.16 (FORK_OUT_BUFFERS, 2026-10-30)
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_ttnn_device_operation{.hpp,.cpp,_types.hpp}`, `device/mhc_post_stats_device_operation.{hpp,cpp}`, `device/mhc_post_given_output.hpp`, `tests/unit/test_mhc_post_output_buffers.py`, `tests/test_mhc_post_ttnn.py` (mesh fixture: the whole box when no case was captured on a box of this size)

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.mhc_post` (plain and stats_weight programs): every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (mhc_post): 120 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/mhc_post_stats_device_operation.cpp, device/mhc_post_stats_device_operation.hpp, device/mhc_post_stats_program_factory.cpp, device/mhc_post_ttnn_device_operation.cpp, device/mhc_post_ttnn_device_operation.hpp, device/mhc_post_ttnn_device_operation_types.hpp, device/mhc_post_ttnn_program_factory.cpp, device/mhc_post_ttnn_program_factory.hpp, mhc_post_ttnn.cpp, mhc_post_ttnn.hpp, mhc_post_ttnn_nanobind.cpp

### collapse_pre: the next mHC step's hc and collapsed input from the same pass (deepseek_v41_flash_d_p P.34)
- What: opt-in `collapse_pre=P` (with `stats_weight`, bf16 F and X, n <= 4) plus the next step's scalars `hc_scale`,
  `hc_base`, `hc_norm_width`, `hc_norm_eps`, `hc_eps`, `hc_sinkhorn_iters` (and optional `hc_output` / `y_output`
  buffers): the stats program also writes y = sum_i P[:, i] X'_i (per column, one DEST window after the unit's mix:
  the pre-block tile + the bf16 X' tiles it just packed for the matmul, mhc_pre_xing's ymix, the packer's bf16
  rounding) and, on each row's root after the partials' sum, the step's hc (DeepSeek's coefficients + Sinkhorn on the
  summed row, the SFPU code of mhc_pre_xing_compute.cpp copied into kernels/mhc_post_hc_sfpu.hpp). Returns
  (X', stats, hc, y). Define MHC_POST_COLLAPSE (writer, compute), CBs 8 / 9 / 10 / 22 / 23; the collapse_pre / hc / y
  addresses (writer) and the scalars (compute) are common runtime args, patched per call (no per-core host cost,
  DS41-145; a slot appended after the writer's variable-length per-core lists cannot be found on a cache hit,
  DS41-210); the hash adds collapse_pre's spec only. Off (default): every program, CT / RT arg, define and hash
  unchanged.
- Why: the next hc step (mhc_pre_xing(stats, X', deepseek=True, collapse_pre=)) re-read the 4 x H streams a call
  (6.5 MB on the 8x4 own rows) and was one more program; its pre is known when the residual runs (DS41-3).
- Tests: `tests/unit/test_mhc_post_collapse.py` (8x4 Galaxy, FABRIC_2D, the 11 x 10 grid cap; 8 passed): X', stats,
  hc, y bit-identical to mhc_post(stats_weight=) + mhc_pre_xing on its outputs at [160, 5120] (8x4 own rows),
  [640, 1280], [64, 640] (ragged blocks), the pre as the packed mixes and as its own tensor; a cache hit with other
  scalars into given hc / y buffers; negative control (another pre, another y); refusals. Fork regression not run on
  the 8x4 (owner OV-25: F.1 runs every suite); the off path's program is byte-for-byte the old one.
- Needed by: deepseek_v41_flash_d_p P.34 (setting POST_COLLAPSE: tt/residual.py, tt/engram_add.py, tt/mhc.py).
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
  `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_stats_{writer,compute}.cpp`,
  `kernels/mhc_post_hc_sfpu.hpp` (new), `tests/unit/test_mhc_post_collapse.py` (new).

### norm_weight: the collapsed y comes out RMS-normalized (deepseek_v41_flash_d_p P.35)
- What: opt-in `norm_weight=w` ((..., 1, C) fp32 TILE DRAM, leading dims 1) and `norm_eps` with `collapse_pre`: y
  (and `y_output`) is bf16((y * rsqrt(sum_c y^2 / C + eps)) * w) of the bf16 collapse y instead of y. Define
  MHC_POST_NORM (all three kernels), CBs 11-15 / 24 / 25, semaphore 1; the weight's address is a reader common arg, eps
  and 1 / C compute common args after the hc scalars; the hash adds the weight's spec. Off (default): every program,
  arg and hash unchanged. Per program:
  - Unit loop: the unit's y tiles stay in cb_y (sized max units a core x B). One DEST window squares and sums them
    lane-wise (exact bf16 values in fp32), then folds each row's 32 columns to 8 (lane-wise adds of the 4 DEST vectors
    that share its lanes). The writer turns those 8 columns coefficient-major (256 words: the 1024-word transpose costs
    8.2 us on BRISC). slot_total8 sums them, and they land in column n (n + 2) + 1 of the unit's partial row. The
    reader lays out the unit's weight tile (slot 4 c + g; two 64-byte reads a column tile).
  - Roots: with a spare core per row (units < cores: 100 units on 110 cores at [160, 5120]) the rows' roots are those
    cores, so they own no units. Otherwise the roots stay where they are.
  - Root's sum window: also computes rsqrt(sum / C + eps) of that column (the hc's Newton rsqrt; row-major, no
    transpose) -> cb_nrs. The root's idle reader then sends the 32 values (128 B) into page j of cb_nr on the owner of
    each of the row's units, plus one semaphore increment each, while the writer transposes the row for the hc.
  - Owners: each owner's reader lays out its rstd tiles (slot = DEST vector row block).
  - Norm phase (after the hc rows): R and W tiles once a window, up to 6 y columns, (y rstd) w fp32 on the SFPU (one
    template instance a column, DS41-189), one bf16 rounding -> cb_yn -> y.
- Why: under OWN_ROWS the model's attn_norm / ffn_norm were separate rms_norm programs on the y this pass just wrote
  (80 a chunk, 0.70 + 0.70 ms device, ~5 ms host a chunk on the 8x4).
- Measured (8x4, [160, 5120] bf16 a chip, 110 cores, traced 50 calls): mhc_post collapse 134.4 us a call, with
  norm_weight 139.4 us; the rms_norm it replaces 18.7 us. Device profile of the first version: the per-unit and root
  word-copy transposes (cols_to_soa 8.2 us, soa_to_cols 5.6 us on BRISC, volatile loops) dominate the program's tail.
  They are not changed here: the collapse program without the norm keeps them, and is a candidate pick of its own.
- Precision: the statistic is an exact fp32 sum of the bf16 values (the rms_norm fork's is FPU-grade, README Limits),
  so the output is closer to float64 than the separate norm, and not bit-identical to it (one bf16 step on some
  elements).
- Tests: `tests/unit/test_mhc_post_norm.py` (8x4 Galaxy, FABRIC_2D, 11 x 10 cap; 6 passed):
  - X' and hc are bit-identical to the call without norm_weight; stats too, except column 25 (= sum y^2, rel < 1e-5
    vs float64).
  - y_n vs float64 of the formula on the op's own bf16 y: at most one bf16 step, 1.0-1.3e-5 of the elements one step
    off.
  - Shapes: [160, 5120] (spare-core roots), [1280, 5120] (8 units a core, unit roots), [640, 1280], [64, 640]
    (ragged).
  - A cache hit with another weight / eps into a given y buffer; a negative control; refusals.
  - Fork regression not run on the 8x4 (owner OV-25).
- Needed by: deepseek_v41_flash_d_p P.35 (setting NORM_IN_MHC: tt/mhc.py norm_kwargs, tt/model.py).
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
  `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_stats_{reader,writer,compute}.cpp`,
  `tests/unit/test_mhc_post_norm.py` (new).

### feed_output: the normalized y also goes into the TP peers' column-split buffer (deepseek_v41_flash_d_p P.36)
- What: opt-in `feed_output=X` ((..., G T, C / G) in the residual's dtype, TILE DRAM interleaved, a persistent buffer
  at the same address on every chip) with `feed_semaphores=[ready, arrival]` (two global semaphores' L1 addresses),
  `feed_num_links` (default 1) and `feed_cluster_axis` (default 1); needs `norm_weight`. X then also receives what
  `all_to_all_async_generic(y, in_dim=2, out_dim=3, cluster_axis=axis)` would write: chip p of a group gets column
  block p of every group chip's normalized y, chip q's rows at [q T, (q + 1) T).
  - One program for every chip (the stats program from the same `build_descriptor`, plus the ports). Everything that
    differs per chip comes from a chip-independent feed table in the common args (`kernels/mhc_post_feed_layout.hpp`:
    fabric chip id -> mesh position, position -> chip id, the side of each group pair, the two sides' eth directions,
    the port cores, each target's unit range, and each chip's fabric channel per port slot); each kernel finds its own
    chip from the fabric routing table in L1 (`routing_l1_info_t::my_device_id`, `kernels/mhc_post_feed_common.hpp`
    Table). A first version with one program per chip (create_descriptor with the mesh coordinate) cost 336 us of
    host a call (DS41-147); the table version 110 us.
  - Ports: the grid's last 2 L cores (row-wise, the same logical cores on every chip), port d L + l for side d (the
    fabric's first hop toward the target: east / north 0, west / south 1, from get_eth_forwarding_direction - on a
    torus ring the far chip goes round the wrap; DS41-224; the same pattern in every group, checked) and link l;
    the units split over the cores before them (the norm's spare-core roots skip the ports). Each port carries to
    every target on its side the units u in [u_lo(t), u_hi(t)) with (u - u_lo) mod L == l, their tiles of column
    block t (`kernels/mhc_post_feed_common.hpp`).
  - Writer (define MHC_POST_FEED, common arg 3 = X's address, the table from 4): in the norm phase a y_n tile of the
    chip's own block goes straight to X; the others go to `y_output` (the staging); after a unit's writes land, one
    increment (semaphore 2) on the port of each (target, link) pair of the unit (one word a unit after the root list).
  - Port reader (`kernels/mhc_post_feed_reader.cpp`, NCRISC): waits for its units, reads the staged tiles into packet
    slots (CB 26, 8 slots of up to P tiles).
  - Port sender (`kernels/mhc_post_feed_sender.cpp`, BRISC): the chip's channel for its slot from the table into
    runtime arg 2, then the connection (semaphores 3 / 4); ready fence (an atomic increment into each target's port
    toward this chip, wait for one from each), then fabric scatter writes of up to P = min(4, payload / tile) tiles
    into the target's X, the stream's last tile alone fused with an increment of the target's arrival semaphore on its
    port 0; port 0 waits for the (G - 1) L arrivals. Each wait re-arms its semaphore by what it saw before anything
    the peers' next call could answer is sent.
  - Hash: the off hash + X's spec, links, axis, mesh shape; the semaphore and buffer addresses are runtime args
    patched on a cache hit (`override_runtime_arguments`). Off (default): every program, CT / RT arg, define and
    hash unchanged.
- Why: under OWN_ROWS DeepSeek-V4.1's MoE head takes the column split, so every layer ran a TP
  all_to_all_async_generic of ffn_norm's output (40 a chunk, a launch and ~0.1 ms host each, plus TP arrival skew).
  With NORM_IN_MHC that output is written by this pass.
- Measured (8x4 Galaxy, FABRIC_2D_TORUS_XY, [160, 5120] a chip, 2 links, 110 cores incl. 4 ports; 50-100 calls,
  wall): host enqueue 101.8 us without the feed, 109.6 us with it (the all-to-all it replaces: 98.4 us); synced loop
  146.7 / 192.1 us (+ the all-to-all: 98.0 us).
- Tests: `tests/unit/test_mhc_post_feed.py` (8x4 Galaxy opened 8 x 4, FABRIC_2D and FABRIC_2D_TORUS_XY, the 11 x 10
  grid cap; 24 passed): X bit-identical to the column blocks of the op's own y and to the stock all-to-all of y; X',
  stats, hc bit-identical to the call without feed_output; [160, 5120], [32, 5120], [1280, 5120] (several units a
  core), [640, 1280] (unit blocks straddle the chips' blocks), [64, 640]; links 1 and 2; 5 repeated calls on two
  buffers with the same semaphores; refusals. Fork regression not run on the 8x4 (owner OV-25: F.1 runs every suite).
- Needed by: deepseek_v41_flash_d_p P.36 (setting FFN_FEED: tt/ffn_feed.py, tt/mhc.py, tt/model.py,
  tt/moe_block_main_adapter.py).
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
  `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_stats_writer.cpp`,
  `kernels/mhc_post_feed_{layout.hpp,common.hpp,reader.cpp,sender.cpp}` (new), `tests/unit/test_mhc_post_feed.py`
  (new).

### fast: the collapse program pipelined (deepseek_v41_flash_d_p P.83)
- What: opt-in `fast=True` (needs `collapse_pre`; default False: every program, CT / RT arg, define and hash as
  before). Define MHC_POST_FAST (the stats kernels and the feed reader), MHC_POST_WSHARE (reader, with one unit a core),
  semaphore 5 (W share), CB_RAW 6 pages pushed to the compute (compute CT arg 29, UnpackToDestFp32), feed CB 16 slots.
  - Reader: a unit's columns stream in with a NoC transaction id each, at most 2 columns in flight a core, each pushed
    when its id lands; the W block after the first columns. With one unit a core (units <= cores, 2 <= RT <= 32) the
    RT cores of a block each read the W tiles i with i mod RT == their row and forward them into the others' cb_w (same
    L1 address: the only reservation), then one increment a forwarded tile; W leaves DRAM once, not RT times
    (12.8 -> 2.6 MB a call at [160, 5120]).
  - Compute: the coefficient and pre-block tiles built on the SFPU from the raw post / comb / pre rows (mask the
    lane holding the value, vConstTileId & 0xE, then OR the 8-lane subvector rotated by 1, 2, 4: exact bit copies;
    the BRISC word build took 30-37 us a unit); per column: the mix, X' pushed to the writer and to cb_xs, then that
    column's collapse y; W waited for only at the matmul; the unit's sum-of-squares tile folded 32 -> 8 columns
    (fold_tile) and summed over 8 slots (slot_total8_any).
  - Writer: pushes the raw rows (no word build); X' written a column at a time; the folded tile's 256-word transpose
    (folded_to_soa8) instead of 1024 words; a root copies the hc's n (n + 2) + 1 columns (plain pointers).
  - Feed port reader: up to SLOTS - 4 = 12 packets' reads in flight (a transaction id each, mod 12), pushed in order;
    more than SLOTS - 4 ahead deadlocks against the sender, which holds up to 4 slots unpopped until a flush (the first
    version with 8 in flight of 8 slots hung the 8x4, reset by the safe runner).
- Why: the op at [160, 5120] a chip ran 137.6 (ffn) / 179.7 us (attn, with the feed) a call, ~25-30 % of DRAM
  bandwidth. Device zones (op probe): W read 36-57 us (100 cores x 128 KB fp32), the BRISC coefficient build 37.5 us in
  parallel, then the mix 19 + collapse 9 + matmul 4 us serially (one unit a core: nothing overlapped), the
  sum-of-squares transpose 8 us, the root tail ~25 us, the feed's port reads one packet a barrier (22-50 us).
- Precision: X', y (normalized or not) and feed_output bit-identical to fast=False; stats bit-identical except column
  n (n + 2) (sum x'^2, another fp32 summation order: max abs 0.094 on values ~2e5, rel <= 1e-5 checked), so the hc
  (its rsqrt) within 2.4e-7.
- Measured (8x4 Galaxy, FABRIC_2D_TORUS_XY, [160, 5120] bf16 a chip, 110-core cap, device profiler, slowest chip of 32,
  median of 4 calls): no feed 137.6 -> 110.8 us; with the feed (2 links) 179.7 -> 141.0 us. The DRAM phase now moves
  ~17 MB in ~45 us (~385 GB/s: reads + the X' writes); what is left: the slowest cores' DRAM service (their columns
  land ~20 us after the median core's) and the root tail (sum 5.5 us, the hc's two BRISC transposes 6.4 + 5.4 us and
  its Sinkhorn 5.5 us), and on the attn site the feed after the norm (~36 us).
- Tests: `tests/unit/test_mhc_post_fast.py` (fast vs off: [64, 640] ragged with / without the feed, [160, 5120] with /
  without the feed and separate post / comb / pre with comb as stored, [640, 1280] feed, [1280, 5120] several units a
  core; refusal), run on the model's resident mesh through models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_fast.py:
  7 / 7 cases pass. Fork regression not run on the 8x4 (owner OV-25: F.1 runs every suite); fast=False is the old
  program (its kernels' only change: device zones, free without the profiler).
- Device zones (MaybeDeviceZoneScope, permanent) in the stats reader / writer / compute and the feed kernels; the op
  probe models/demos/deepseek_v41_flash_d_p/tests/perf/test_mhc_post_probe.py prints them per call.
- Cores: unchanged (units + spare-core roots + ports inside max_grid: 109 of the 110 cap at [160, 5120] with the feed).
- Needed by: deepseek_v41_flash_d_p P.83 (setting MHC_POST_FAST, tt/mhc.py collapse_kwargs).
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
  `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_stats_{reader,writer,compute}.cpp`,
  `kernels/mhc_post_feed_{reader,sender}.cpp`, `tests/unit/test_mhc_post_fast.py` (new).

### v2: feed landing in the ports' L1, W segments + multicast, SFPU hc transposes (deepseek_v41_flash_d_p PB.8)
- What: opt-in `v2=True` (needs `fast`; default False: every program, CT / RT arg, define and hash as before). Owner
  2026-11-20: two ideas from a colleague's mhc_post op (git f1bdc1a4c93: feed L1 landing; slice-sharing placement +
  fused SFPU mix), built into this fork (not ported), plus the pending hc transposes step.
  - Feed landing (define MHC_POST_LAND, writer / port reader / sender; semaphore 6 over workers + ports): cb_feed on the
    ports is the landing: one slot of P tiles per packet of the busiest port (host max_port_packets over every chip,
    side and link) + one page of flag words. The port reader zeroes the flags, then writes cb_feed's address into its
    landing word (0 at every launch); a worker reads port 0's word once (bounded poll) before its first landing write,
    writes each other-chip y tile to (port, slot, offset) of its stream position (mhc_post_feed_common.hpp stream_pos:
    P a packet, the stream's last tile alone, as the sender walks it), fences, then sets its flag word (stream order of
    the port's units) with an inline write. The port reader pushes each packet once the flags of every unit up to the
    packet's last tile are set; the sender is unchanged but for its slot count (land + 1) and 8 headers (a flush every
    8). No y_output staging writes, no port DRAM re-reads. Fits when (land + 1) x P tiles <= min(1 MB, the workers'
    CB bytes); else the staging path.
  - W segments (host) + multicast (define MHC_POST_WMCAST, reader): with one unit a core, block b's RT units on a
    vertical segment of RT cores (column-major segments, none holding a port; the feed's ports move to the grid's last
    column, bottom up); each core's W share goes to the block's other RT - 1 cores in one NoC multicast a tile (reader
    args: the segment's first / last core after the peer list) when every segment is one NoC column. The forward
    semaphore protocol is unchanged.
  - hc transposes (define MHC_POST_HCT, with the norm; writer / compute): the root's compute turns the summed row
    coefficient-major in the root window (rows_to_soa, DEST 0 -> 2, packed to cb_hsoa) and the Sinkhorn's hc back to
    row-major (soa_to_rows, DEST 0 -> 1, columns 24.. zero) on the SFPU: per output vector, 8 source vectors masked to
    one lane of each 8-lane subvector, rotated (subvec_shflror1) and ORed in (Horner), exact bit moves; the column
    parity is a dst_reg base shift (one code copy for both). The writer writes cb_fin and the hc tile as they come (no
    cols_to_soa / soa_to_cols, 6.4 + 5.4 us of BRISC word copies on the 8x4).
  - Compute (define MHC_POST_V2): the bottom half's coefficient tiles are built 16 vectors on (q + 16, inside the CF
    tiles) and its mix runs the top half's code after 16 dst_reg++ (mix_top / mix_bottom): -3 KB of trisc1 text; v2 with
    the device profiler's zones fit the 70 688 B kernel-config buffer only with it (72 960 B before).
- Why: the op probe on the 8x4 ([160, 5120] a chip, P.83 fast): the ffn call's end was the root's hc path (BRISC
  transposes 6.4 + 5.4 us after the Sinkhorn), the attn call's the feed (port DRAM re-reads of the staged y 24-34 us
  after the last unit, then the sends).
- Precision: bit-identical to `fast` on every output (X', stats incl. column 24, hc, y without the feed, feed_output):
  the same values move, every sum keeps its order.
- Measured (8x4 Galaxy, FABRIC_2D_TORUS_XY, 110-core cap, device profiler, slowest chip, median of 4-6 calls; fast /
  v2, us): [160, 5120] no feed 111.1 / 102.6, feed 143-150 / 138.4; [256, 5120] no feed 161.8 / 158.2, feed 249.6 /
  240.6. The multicast W share did not move the units' phase measurably (column reads + mix bound it: slowest partial
  ~87 us); 8 sender headers alone: no change (the sends are link / packet bound: P = 2 bf16 tiles a 4352 B packet,
  ~30-35 us for the busy port's ~200 tiles); streaming the packets by flags: ~-2 us over the whole-wave push.
- Tests: `tests/unit/test_mhc_post_v2.py` (v2 vs fast, every output bit-identical, 3 repeated v2 calls: [64, 640]
  ragged with / without the feed, [160, 5120] with / without, [256, 5120] with / without, [640, 1280] feed, [1280, 5120]
  (landing too big: staging); refusal), run on the model's resident mesh through
  models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_v2.py: 8 / 8. Fork regression not run on the 8x4 (owner
  PACKED_RULES: no fork suites on the 8x4); v2=False is the P.83 program (its kernels change only inside v2 defines;
  the port reader's extra CT args are appended only with the landing).
- Cores: unchanged counts (units + spare-core roots + 2 L ports: 109 of the 110 cap at [160, 5120] with the feed); no
  matmul program config (rule 18 caps matmuls only).
- Needed by: deepseek_v41_flash_d_p PB.8 (setting MHC_POST_V2, tt/mhc.py collapse_kwargs).
- Files: `mhc_post_ttnn.{hpp,cpp}`, `mhc_post_ttnn_nanobind.cpp`, `device/mhc_post_stats_device_operation.{hpp,cpp}`,
  `device/mhc_post_stats_program_factory.cpp`, `kernels/mhc_post_stats_{reader,writer,compute}.cpp`,
  `kernels/mhc_post_feed_{common.hpp,reader.cpp,sender.cpp}`, `tests/unit/test_mhc_post_v2.py` (new).
