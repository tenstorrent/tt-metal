# mhc_pre_ttnn (ttnn.bringup.mhc_pre)

Source: ai-generated (tt_ops_code_gen eval run 1045, branch `2026_09_30_0115_run1_mhc_pre` @ `963102ebe2f`, started from
`mstaletovic/mhc-codegen` @ `5066bfadf95`): the op folder `ttnn/ttnn/operations/mhc_pre` and its unit suite
`tests/ttnn/unit_tests/operations/mhc_pre`. Left out: the run's `perf_experiments/` scripts, `agent_logs/` and the
agents' scratch probes (all on the source branch). The op's own history is in `changelog.md` (design, refinements,
two perf-tournament rounds); its golden suite is `eval/golden_tests/mhc_pre` on tt_ops_code_gen `mstaletovic/mhc-goldens`.

1. ai-generated perf optimized version (Python ProgramDescriptor host side), moved to ttnn/ttnn/bringup/mhc_pre_ttnn (the folder is not
   named mhc_pre: a package of the op's own name would shadow ttnn.bringup.mhc_pre): imports `ttnn.operations.mhc_pre` ->
   `ttnn.bringup.mhc_pre_ttnn`; `kernel_lib/perf_instrumentation.hpp` (on the source branch, not
   in this tree) ships as `kernels/perf_instrumentation.hpp`, as rms_norm_ttnn's does. Kernels otherwise unchanged.
   Unit suite in `tests/unit/`: passes on this tree's kernel helper library.
   - Needed by: glm53_flash_d_p (mHC: attn / ffn hc + collapse, 4 streams, C 4096).
2. C++ host side: `ttnn.bringup.mhc_pre` is now a C++ TTNN op (the Python op stays importable as
   `ttnn.bringup.mhc_pre_ttnn.mhc_pre`, the parity reference). Kernels untouched.
   - What: `device/mhc_pre_ttnn_program_factory.cpp` is a line-for-line port of `make_plan()` (group-width cost
     model, L1 fit, depth fallback, group growth) and `create_program_descriptor()` (CB table incl. aliases, W column
     all-gather Mcast1D, per-group Mcast2D, NoC-flip placement sets, CT / RT args, fp32 unpack modes,
     `MHC_PRE_KERNEL_DEFINES`); a ProgramDescriptor factory (build on a cache miss, `override_runtime_arguments`
     patches the address slots of every reader / writer set on a hit); `MhcPreDeviceOperation` (outputs y, post,
     comb) + `ttnn::prim::bringup::mhc_pre_ttnn` (program hash = the input specs, n, the scalars, the compute config
     fields and the defines env); `mhc_pre_ttnn.cpp` keeps `validate()`'s checks, order and exception types;
     `_mhc_pre_ttnn_program_descriptor` for the test. The scalars are Python-style doubles rounded to fp32 once.
   - Why: at GLM-5.3's shape (T 5120, n*C 16384, bf16 X and W) the Python op was host-bound: host 2.037 -> 0.043 ms
     per call, wall 2.067 -> 0.597 ms per call.
   - Tests: `tests/unit/test_mhc_pre_cpp_parity.py` (17 program-descriptor parity cases over every dtype pair, narrow
     and tall groups, decode, ragged T, n 1..4, ranks 2..4, GLM's shape; 6 bit-identical device outputs incl. a
     program-cache hit); the unit suite and the eval golden suite (`tests/golden/`, from tt_ops_code_gen
     `mstaletovic/mhc-goldens`, run on the C++ op) pass.
   - Needed by: glm53_flash_d_p.
3. Model cases (bringup-fork-tests, task O.1): `tests/cases.py` (10 glm53_flash_d_p call(s), captured with
   GLM_MHC_IMPL=fused on the s56320 rung), `tests/reference.py` (float64 torch semantics), `tests/test_mhc_pre_ttnn.py`
   (random per-chip inputs on the 2x2 mesh, PCC + rel L2 per output; limits from the measured error with margin).
4. Xing4.0 entries (xing40_a4b_d_p, task P.2b, owner pick "adopt both fused mHC ops"): `ttnn.bringup.mhc_pre_xing` and
   `ttnn.bringup.mhc_pre_xing_pack`. New entries with their own program; `ttnn.bringup.mhc_pre` (its kernels, host
   code and program hash) is unchanged, so nothing changes for glm53_flash_d_p.
   - What: for a caller that splits the projection over the mesh (xing40's 4x2 SP x TP: each chip holds half of every
     stream's columns and all_reduces the partial row over axis 1).
     `mhc_pre_xing_pack(mix, streams)`: the caller's partial projection row [.., T, 32] (x @ fn^T in columns 0..23) ->
     the same row with column 24 = sum x^2 of the local streams (one pass, exact fp32 SFPU multiply-adds).
     `mhc_pre_xing(input, streams=None, *, scale, base, norm_width, n, norm_eps, hc_eps, sinkhorn_iters, clamp_min,
     clamp_max, coefficients_given=False)`: from the all-reduced row, the Xing math (xing_ref.py:hc_weights): r =
     rsqrt(ss / norm_width + norm_eps); pre = sigmoid (no + eps); post = 2 sigmoid; comb logits clamped to
     [clamp_min, clamp_max], exp(L - rowmax), then sinkhorn_iters x (row / (sum + eps), column / (sum + eps)).
     Output hc [.., T, 24] = [pre | post | comb row-major] (the layout xing40's tt/residual.py slices for
     `mhc_post(comb_transposed=False)`). With streams: y = sum_i pre_i x_i [.., T, C] (the collapse; column-local).
     `coefficients_given=True` takes a finished hc and only collapses.
   - Files: `mhc_pre_xing.{hpp,cpp}`, `device/mhc_pre_xing_device_operation.{hpp,cpp}`,
     `device/mhc_pre_xing_program_factory.cpp`, `kernels/mhc_pre_xing_{common.hpp,reader.cpp,writer.cpp,compute.cpp}`,
     bindings in `mhc_pre_ttnn_nanobind.cpp`, `sources.cmake`. Kernels: the writer (BRISC) scatters the row tile
     into the coefficient-major layout (`mhc_layout::slot_index`, lane = token) and back; the compute runs the
     coefficients and a plain-SFPI Sinkhorn lane-wise in fp32 (accurate exp, Newton reciprocal / rsqrt, the helpers
     of mhc_pre_compute.cpp); the collapse multiplies UnpackToDestFp32 stream tiles by a "pre-block" tile (8 row
     blocks x n streams) on the SFPU at dst_full_sync. Work: token tile-rows (coefficients, pack) or y column tiles
     (collapse) split over the grid. The scalars are runtime args patched on every call (one program for all layers;
     the hash has only the mode, n and the tensor specs).
   - Why: xing40 attn_hc / ffn_hc were ~112 small programs per call (composed Sinkhorn), the collapses 18.
     Device time per call on the 4x2 box (5120-token chunk, 1280 rows per chip): hc 977 -> 353 us (matmul 211 +
     pack 106 + all_reduce 18 + mhc_pre_xing 21), collapse 548 -> 134 us. Profile (40 layers, warm chunk
     [51200, 56320)): attn_hc 39.2 -> 14.3 ms, ffn_hc 39.2 -> 14.3 ms, attn_collapse / ffn_collapse 22.1 -> 5.3 ms each.
   - Tests: `tests/test_mhc_pre_xing.py` + `tests/xing_cases.py` (6 cases on the 4x2 mesh: pack, coef x 3 incl. a
     clamp-saturating one and T 512, collapse, coef + collapse; float64 reference `tests/reference.py:
     mhc_pre_xing_coefficients` / `mhc_pre_xing_collapse`, measured rel L2 <= 1.1e-7, limit 1e-5) and a CPU check
     of that reference against xing_ref.hc_weights. Regression of the unchanged path: unit (test_mhc_pre,
     blocking, cpp_parity) 48 passed; precision baseline + golden 215 passed; model cases (test_mhc_pre_ttnn.py) 10
     passed, after its fixture was changed to open the whole box when the case mesh (2x2) does not match it (a 2x2
     submesh of the 4x2 box fails the FABRIC_2D handshake; as P.2 did for mhc_post).
   - Needed by: xing40_a4b_d_p (tt/mhc.py, tt/collapse.py; XING_HC_IMPL=composed keeps the op chain).

5. Port to the new multicast helpers (#57547; kernel helper library @ `73027b6e6ff`, 2026-10-01). The old
   `kernel_lib/host/mcast_host.hpp` / `kernel_lib/mcast_pipe.hpp` became `kernel_lib/mcast/{host,kernel}/` with a new
   wire (11-word CT block with a FAMILY / ABSENT tag; RT block with sender coords, rectangle records and a role word)
   and attach()-only host queries; host and kernels move together.
   - Host (C++ factory and Python builder): `device/mcast_wire.hpp` / `_McastWire` attach each family to a scratch
     kernel (adopting the program's SEM_W_READY / SEM_MCAST_READY by `sem_ids`) and read the blocks back, so the
     writer still decodes them at `MCAST_CT_BASE` 22 / `MCAST_RT_BASE` 18. W broadcast: `Mcast1D(PerColumn,
     Mcast1DRotatingSenderConfig())` (was `rotating_sender=True`); group combine: `Mcast2DFixedSenderConfig(root)`.
     One-core groups and no-W-broadcast plans get the one-word ABSENT block and no RT words (were 6-word inactive
     wires + 4 zero RT words).
   - Writer kernel: group pipes via `optional_sender` / `optional_receiver`. W all-gather: the share is sent through a
     NON-rotating `SenderPipe` built from this core's sender record of the same wire (`send_w_share`), and the receive
     waits once for `w_events` (receive(round) now waits for round + 1). The family's own rotating Counter sender
     counts its own send with a local non-atomic `+=` (`LOCAL_NONATOMIC` on BH DM) that races the other rows' NoC
     atomic increments on the same word; with it the unit suite hung deterministically (1x1x640x7168, W broadcast).
     The send is the old pipe's (payload + inc_multicast to the rest of the column, no self-count). Upstream issue,
     worth reporting to the helper owners.
   - Default behaviour unchanged (same transfers, same semaphores). Tests (2026-10-01): unit 115 passed / 1 skipped,
     golden 206 passed, model cases 97 passed (87 on the 4x2 box; the 10 glm53 2x2 cases on cards 0,1,4,5).
   - Needed by: rebase onto origin/malimpic/llk_helper_library_rebased_0110_2 (the fork no longer compiled).
   - Files: `device/mcast_wire.hpp` (new), `device/mhc_pre_ttnn_program_factory.cpp`, `mhc_pre_program_descriptor.py`,
     `kernels/mhc_pre_writer.cpp`.

6. Port to the unified multicast helper (`kernel_lib/mcast/host/mcast.hpp`; kernel helper library @ `bdc576fa652`,
   2026-10-12). Mcast1D / Mcast2D and their wire became one `Mcast` (receivers + group size + sender config) with a new
   wire; host and kernel move together.
   - Host (C++ factory and Python builder): the W broadcast (old rotating Mcast1D PerColumn) is one group per column,
     column-major, rotating sender; each group combine (old Mcast2D, fixed sender at the box corner) is one group over
     the box with sender index 0. The old `sem_ids` adoption has no equivalent: the channel is attached at the
     program's own id (SEM_W_READY / SEM_MCAST_READY), and `device/mcast_wire.hpp` / `_McastWire` check that the
     program owns it and add none. Also `tt::tt_metal::MathFidelity` / `UnpackToDestMode` (#59097 removed the global
     aliases).
   - Writer kernel: the new rotating Counter sender counts its own send through a NoC atomic, so the race that 5.
     worked around is gone; `send_w_share` (a non-rotating pipe built from the old wire's sender record) is deleted
     and the share goes out through the channel's own sender. The receive waits for one more event on a core that
     sent (its own count).
   - Why: the old helper and the aliases no longer exist, so the fork did not build.
   - Default behaviour: same transfers and semaphore ids; the wire format is the new helper's.
   - Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
   - Files: `device/mcast_wire.hpp`, `device/mhc_pre_ttnn_program_factory.cpp`, `device/mhc_pre_xing_program_factory.cpp`,
     `mhc_pre_ttnn.cpp`, `mhc_pre_program_descriptor.py`, `kernels/mhc_pre_writer.cpp`.

7. `mhc_pre_xing(deepseek=True)`: DeepSeek-V4's coefficient math (deepseek_v41_flash_d_p FIX.4, owner OV-12).
   - What: a keyword option `deepseek` (default False). On: the coefficients follow DeepSeek's
     `hc_split_sinkhorn_kernel` (checkpoint `inference/kernel.py`; CPU port `deepseek_v41_flash_d_p/reference/
     kernels.py:sinkhorn_split`): pre = sigmoid(z) + hc_eps; L not clamped (clamp_min / clamp_max ignored); comb =
     exp(L - rowmax) / rowsum + hc_eps, then m / (colsum + hc_eps), then (sinkhorn_iters - 1) x (rows, columns). Same
     lane-wise fp32 SFPU arithmetic as the Xing path (accurate exp, Newton reciprocal / rsqrt). The kernel code is
     behind the compute define `MHC_XING_DEEPSEEK`, added only when the option is on; off, the kernels, compile-time
     args, runtime args, CBs and the program hash are the ones they were. `deepseek` with `coefficients_given=True` is
     refused (no coefficient math to select).
   - Why: deepseek_v41_flash_d_p computed these mixes with ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn,
     whose FPU steps (SEL matmul, add_tiles, RB / CB sum matmuls, mul_tiles, copies without UnpackToDestFp32) read
     fp32 at TF32: the step's error was 3.8-33x its noise floor, a -3.3e-4 bias and a 1.5-3.5 % comb column. On the
     golden inputs of the 5 hc gates (2x2, 1024 rows per chip) the coefficients vs a CPU fp32 Sinkhorn on the same
     mixes: source kernel rel 5.9e-4..1.2e-3, this option 6.5e-8..7.6e-8. Time (wall, 20 synced calls): rsqrt chain +
     source kernel + concat 0.45 ms -> this op 0.031 ms; the whole mixes step 0.77 -> 0.41 ms.
   - Tests: `tests/xing_cases.py` 5 cases `deepseek_v41_flash_d_p-2x2-coef-ds-t1024-*` (the checkpoint's scale / base
     of layers 0-1 attn / ffn, incl. L0 attn's comb bases -25.5..13.4, and a strong-logit case; measured rel
     5.2e-8..1.0e-7, limit 1e-5; with the option off the same check fails: comb rel 3.6e-4..5.4e-4);
     `tests/reference.py:mhc_pre_xing_coefficients(deepseek=True)`; `test_reference_matches_deepseek_kernel` (CPU:
     the reference == kernels.py:sinkhorn_split element-wise <= 2e-5, and differs from the Xing math).
     Regression, option off (2026-10-16, 2x2 box): model cases 103 passed (86 xing40 cases on the box's 2x2, 10
     glm53, the 5 new, 2 CPU reference checks); unit + golden 321 passed, 1 skipped (= unit 115 + 1 skipped,
     golden 206: the recorded counts). Re-run on a fresh build of the committed tree (c50953498f6, 2026-10-16,
     FIX.4 overseer.1): model cases 103 passed, unit + golden 321 passed / 1 skipped (= baseline).
   - Needed by: deepseek_v41_flash_d_p FIX.4 (tt/mhc.py, setting HC_SINKHORN=sfpu_fp32).
   - Files: `kernels/mhc_pre_xing_compute.cpp`, `device/mhc_pre_xing_device_operation.{hpp,cpp}`,
     `device/mhc_pre_xing_program_factory.cpp`, `mhc_pre_xing.{hpp,cpp}`, `mhc_pre_ttnn_nanobind.cpp`,
     `tests/{reference.py,xing_cases.py,test_mhc_pre_xing.py}`.

8. `mhc_pre_xing(..., collapse_pre=)`: the coefficients AND a collapse with a given pre in one program
   (deepseek_v41_flash_d_p P.14, owner's fused-mHC comparison).
   - What: an optional tensor `collapse_pre` (default None). Given (with streams, coefficients_given=False), the
     program computes hc from the reduced row as before and collapses the streams with collapse_pre's columns 0..n-1
     (y = sum_i collapse_pre_i x_i) instead of the pre it just computed; hc is still returned. Single-pass mHC
     (DeepSeek-V4: a sublayer collapses with the PREVIOUS sublayer's pre) then needs one program per hc step, not a
     coefficient program plus a separate collapse. The reader reads the pre row tile per segment (cb 7) and the writer
     builds the pre-block tile from it; both behind the data-movement define `MHC_XING_GIVEN_PRE` (absent when off),
     the extra CT / RT args are appended only when on, and the program hash takes `given_pre` + the pre's spec only
     when on, so every existing program (xing40, glm53, deepseek coefficients) is the one it was. collapse_pre may be
     (..., T, n..32) fp32 TILE (only columns 0..n-1 are read). Refused without streams, with coefficients_given, or
     wider than one tile.
   - Why: deepseek_v41_flash_d_p's attn_collapse / ffn_collapse were 4 slices + multiply + 3 addcmul each (5.0 + 5.0 ms
     a chunk on 2x2); the coefficient program already walks the streams' columns for its own collapse.
     2x2 p150, [1024, 4 x 2560] fp32 a chip (wall, 30 synced calls): coef + collapse_pre 0.164 ms vs coef 0.028 +
     the multiply / addcmul collapse 0.765 ms. Model (MHC_FUSED=on, chunk 53248-55296): attn / ffn_collapse
     5.0 + 5.0 -> 0 ms, attn / ffn_hc 3.2 -> 4.35 ms each before the residual stats (mhc_post_ttnn change 7).
   - Tests: `tests/unit/test_xing_collapse_pre.py` (DS41 2x2 / 4x4 shapes and a small one: hc bit-identical to the
     plain coefficient call, y bit-identical to the coefficients_given collapse and within 1e-6 of float64 (measured
     4e-8), negative control (the computed pre), a 32-column pre, refusals). Regression, option off (2026-10-20, 2x2):
     unit + golden 332 passed / 1 skipped (= the recorded 321 / 1 + the 11 new tests of changes 8-9), model cases 103
     passed (= recorded).
   - Needed by: deepseek_v41_flash_d_p P.14 (tt/mhc.py:TtHcMixes.fused, setting MHC_FUSED).
   - Files: `mhc_pre_xing.{hpp,cpp}`, `mhc_pre_ttnn_nanobind.cpp`, `device/mhc_pre_xing_device_operation.{hpp,cpp}`,
     `device/mhc_pre_xing_program_factory.cpp`, `kernels/mhc_pre_xing_{reader,writer}.cpp`,
     `tests/unit/test_xing_collapse_pre.py`.

9. `mhc_pre_xing_pack(None, streams, n=, engram_keys=, engram_weight=)`: the engram gate's partial sums in one pass
   (deepseek_v41_flash_d_p P.14).
   - What: keyword options `engram_keys` / `engram_weight` (default None; `mix` becomes Optional and must be None in
     this mode). A separate program (`device/mhc_engram_pack_*`, `kernels/mhc_engram_pack_*`): from the streams h
     (..., T, n C), the keys k (..., T, >= n C; columns 0..n C-1) and the weight w (..., 32, n C: the weight row on all
     32 rows of a tile) it writes a new row (..., T, 32) fp32 = [sum_c h_i w_i k_i | sum_c h_i^2 | sum_c k_i^2 | 0 ..]
     (i < n) of the chip's columns: per unit (stream, column block, token tile-row) one DEST window accumulates the
     three lane-wise in fp32 on the SFPU (UnpackToDestFp32 copies; (h w) rounded, then fused multiply-adds), the writer
     re-lays them coefficient-major, the compute sums the 32 column slots in order, the unit's 3 totals go into a zero
     partial row in a DRAM scratch, and the row's root core (semaphore count) sums its n G partials in unit order.
     Units are stream / block-major so a core keeps its weight block in L1. The plain mix + sum x^2 mode, its program
     and hash are unchanged.
   - Why: deepseek_v41_flash_d_p's engram_add computed these 12 sums with a weight multiply, 12 slices, 12 multiplies
     and 12 ttnn.sum (exact SFPU) + a concat: 2.51 ms vs 0.424 ms for this op (2x2, [1024, 4 x 2560] h, [1024,
     5 x 2560] k, wall, 30 synced calls). The model's engram_add section (2 layers, with the mhc_post identity-comb add):
     7.0 -> 1.78 ms a chunk.
   - Tests: `tests/unit/test_engram_pack.py` (DS41 2x2 / 4x4, small, ragged Ct = 13, n = 2: every column within 1e-6
     of float64 sums of the fp32 inputs (measured <= 1.4e-7), the dot within 1e-6 of the op chain (2.2e-7), the other
     columns exactly 0, negative controls (a dropped column tile, the weight ignored), a program-cache hit on fresh
     buffers, refusals). Regression: as change 8.
   - Needed by: deepseek_v41_flash_d_p P.14 (tt/engram_add.py:TtEngramAdd._partials_fused, setting MHC_FUSED).
   - Files: `mhc_pre_xing.{hpp,cpp}`, `mhc_pre_ttnn_nanobind.cpp`, `sources.cmake`,
     `device/mhc_engram_pack_device_operation.{hpp,cpp}`, `device/mhc_engram_pack_program_factory.cpp`,
     `kernels/mhc_engram_pack_{common.hpp,reader.cpp,writer.cpp,compute.cpp}`, `tests/unit/test_engram_pack.py`.

10. `mhc_pre_xing(..., partials=P)`: the TP group's partial rows stacked on dim 1, summed in the op
    (deepseek_v41_flash_d_p P.19).
   - What: keyword option `partials=1` (1..16; coefficient mode only). P > 1: the input is (1, P, T, 32), the chips'
     partial mix rows gathered on dim 1 (one all_gather); the reader reads row tile r of each partial (page
     p RT + r), the writer re-lays each coefficient-major, the compute copies them to DEST 0 / 1 and adds in dim-1 order
     on the SFPU (((p0 + p1) + p2) + ..., one IEEE fp32 rounding per add: the sum all_gather + ttnn.add gives) before
     the coefficient math; hc / y are (1, 1, T, ...). Defines MHC_XING_PARTIALS / MHC_XING_ROW_TILES on the three
     kernels, CB_ROW / CB_SOA 2 P pages, a hash branch of its own; P = 1: every program and hash unchanged.
   - Why: deepseek_v41_flash_d_p's hc row exchange was typecast to bf16 + bf16 fabric reduce-scatter + all_gather +
     typecast back (4 calls, the reduce-scatter 0.3 ms of host) for a [1024, 32] row; now one fabric all_gather of the
     fp32 rows, and the sum is the exact fp32 one (the hc mixes the reference computes in fp32). Model (GLUE_FOLD=on):
     attn_hc + ffn_hc host 10.8 -> 3.7 ms a chunk, device 4.02 -> 4.21 ms.
   - Tests: `tests/unit/test_xing_partials.py` (P = 2 / 4, DS41 2x2 / 4x4 shapes and a small one, deepseek and Xing math,
     with / without streams and collapse_pre: hc and y bit-identical to the call on the ttnn.add-summed row; negative
     control (a dropped partial); refusals). Regression, option off (2026-10-22, 2x2): unit 139 passed / 1 skipped +
     golden 206 = 345 / 1 (= the recorded 332 / 1 + 13 new), model cases 103 passed (= recorded).
   - Needed by: deepseek_v41_flash_d_p P.19 (tt/mhc.py:TtHcMixes.fused, setting GLUE_FOLD).
   - Files: `mhc_pre_xing.{hpp,cpp}`, `mhc_pre_ttnn_nanobind.cpp`, `device/mhc_pre_xing_device_operation.{hpp,cpp}`,
     `device/mhc_pre_xing_program_factory.cpp`, `kernels/mhc_pre_xing_{reader,writer,compute}.cpp`,
     `tests/unit/test_xing_partials.py`.

11. bf16 streams in `mhc_pre_xing` (collapse modes) and `mhc_pre_xing_pack` (sum of squares; engram mode with bf16
    keys) (deepseek_v41_flash_d_p P.20).
   - What: the `streams` (and `engram_keys`) may be bfloat16 TILE (were refused: fp32 only). bf16 streams: CB_X (and
     the engram CB_IN) in bf16 pages, unpacked to the fp32 DEST exactly (UnpackToDestFp32: bf16 -> fp32 is exact), so
     every sum is the fp32 program's on the same values; the collapse y is written in the streams' dtype (bf16: the
     packer rounds the fp32 sum once, to nearest, ties away from zero); the pack row / engram row stay fp32. The
     compute kernels reconfigure srcA before each copy from a CB of the other format; defines MHC_XING_X16 (compute)
     / MHC_ENGRAM_X16 (reader, compute), present only with bf16 streams. engram_keys must share the streams' dtype.
     fp32 streams: every program, CT / RT arg, define and hash unchanged (the hash includes the streams' spec).
   - Why: DeepSeek's GPU code (inference/model.py) keeps the mHC residual streams in bf16 (hc_post .type_as(x),
     hc_pre .to(x.dtype)); deepseek_v41_flash_d_p held them in fp32 and paid double DRAM traffic in every stream pass
     (DS41-80). Model (BF16_ACTIVATIONS=on, 2x2, chunk 53248-55296, plain profile vs FIX.10's): attn_hc 2.65 ->
     1.75 ms, ffn_hc 1.57 -> 1.12, engram_add 1.79 -> 1.34 ms a chunk.
   - Tests: `tests/unit/test_xing_bf16_streams.py` (DS41 2x2 / 4x4 shapes and a small one: pack and engram rows
     bit-identical to the fp32 call on the upcast values; hc bit-identical and y = the fp32 call's y rounded to nearest
     (ties away) bit for bit, for the coefficient + collapse, collapse_pre and coefficients_given calls; sum x^2 vs
     float64 <= 4.2e-7; negative control (truncation instead of rounding); refusals). Regression, option off
     (2026-10-22, 2x2): unit + golden 355 passed / 1 skipped (= the recorded 345 / 1 + 10 new), model cases 103
     passed (= recorded). No source tests (a new op: no tests/source_baseline.json).
   - Needed by: deepseek_v41_flash_d_p P.20 (tt/mhc.py, tt/collapse.py, tt/engram_add.py, setting BF16_ACTIVATIONS).
   - Files: `mhc_pre_xing.cpp`, `device/mhc_pre_xing_device_operation.cpp`, `device/mhc_pre_xing_program_factory.cpp`,
     `device/mhc_engram_pack_program_factory.cpp`, `kernels/mhc_pre_xing_compute.cpp`,
     `kernels/mhc_engram_pack_{reader,compute}.cpp`, `tests/unit/test_xing_bf16_streams.py`.

12. Model cases for deepseek_v41_flash_d_p (O.1, 2026-10-23): 16 mhc_pre_xing calls (partials 2, collapse_pre, bf16
    streams, deepseek math; each layer's captured scale / base) and 2 mhc_pre_xing_pack calls (bf16 pack, engram mode),
    `tests/xing_cases.py`, run by test_mhc_pre_xing_ds41; every output within noise_limit at its formats (hc fp32
    math: reference.mhc_pre_xing_coefficients(dtype=); the sums with one fp32 running accumulator,
    reference.running_sum). Worst device / limit 0.73. No op change.
    - Needed by: deepseek_v41_flash_d_p O.1
    - Files: `tests/xing_cases.py`, `tests/cases.py` (engram mode -> op mhc_pre_xing_pack), `tests/reference.py`,
      `tests/test_mhc_pre_xing.py`.

13. Tests: `tests/unit/test_mhc_pre_perf_inproc.py` reads device time from the real-time profiler (2026-10-27). It
    read DEVICE KERNEL DURATION through `ttnn.ReadDeviceProfiler` and so skipped unless the run set
    TT_METAL_DEVICE_PROFILER=1, i.e. in every suite run. It now measures each call with
    `profile_realtime_program_merged` (on in every run), the same knob A/B over the same shapes, dtypes and env
    selections, printed and never asserted. What it adds over `test_mhc_pre_perf.py` (the C++ op, default knobs, one
    call, numbers only under `--profile`) and `test_mhc_pre_perf_sweep.py` (the same knob calls, numbers only under
    `--profile`): the per-knob device ns in the ordinary run. Measured 2026-10-27, BH p150, bf16 X / fp32 W, default
    knobs: 640x1792 43.9 us, 640x7168 124.9 us, 1280x4096 139.6 us (README: 43.0 / 123.8 / 137.8). No op change.
    - Needed by: fork test hygiene (owner, 2026-10-27: no skip that hides a dead test)
    - Files: `tests/unit/test_mhc_pre_perf_inproc.py`

### Rebase onto llk_helper_library 6b80daf43e3: kernel API port (2026-10-27)
- What: `kernels/perf_instrumentation.hpp` includes `api/debug/kernel_profiler.hpp` (#60172); the two eltwise chains
  of `mhc_pre_compute.cpp` (`sumsq_row`, the y-mix) pass the kernel's `DataflowBuffer` objects to their elements
  (#60306: `BinaryFpu<..>{a, b, base_a, base_b}`, `PackTile<..>{out}`), as upstream migrated its own chain users.
  Same LLK calls and CB sync, no behaviour change.
- Why: all three kernels no longer compiled (missing header; the chain's static_assert on unbound buffers).
- Needed by: rebase 2026-10-27
- Files: `kernels/perf_instrumentation.hpp`, `kernels/mhc_pre_compute.cpp`

### hc_output / y_output (mhc_pre_xing, mhc_pre_xing_pack): write a caller's output buffer (2026-10-30)
- What: `hc_output=`, `y_output=` on `mhc_pre_xing` / `mhc_pre_xing_pack` (default None = the op allocates them, unchanged): the caller's buffers for hc and y, each exactly the spec the op would allocate (checked on every call), none of them an input. Kernels and programs unchanged; outputs byte-identical to the allocating call.
- Why: each call allocated its output (~35 us of host). The model passes load-time buffers at the call sites whose
  output is read and dropped inside the layer. Measured: mhc_pre_xing 126.9 -> 84.4 us of host a call (8x4, 80 calls a chunk); mhc_pre_xing_pack 110 -> 105 us (3 calls, not given buffers by the model).
- Tests: `tests/unit/test_xing_output_buffers.py` 3 passed on the 8x4. Regression: unit 144 passed, model cases 71 passed; stopped by the owner before the end (F.1 runs it).
- Needed by: deepseek_v41_flash_d_p P.16 (FORK_OUT_BUFFERS, 2026-10-30)
- Files: `mhc_pre_xing.{hpp,cpp}`, `mhc_pre_ttnn_nanobind.cpp`, `device/mhc_pre_xing_device_operation.{hpp,cpp}`, `tests/unit/test_xing_output_buffers.py`

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.mhc_pre_xing` and `mhc_pre_xing_pack` (engram pack too): every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (mhc_pre_xing / mhc_pre_xing_pack): 120 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/mhc_engram_pack_device_operation.cpp, device/mhc_engram_pack_device_operation.hpp, device/mhc_engram_pack_program_factory.cpp, device/mhc_pre_xing_device_operation.cpp, device/mhc_pre_xing_device_operation.hpp, device/mhc_pre_xing_program_factory.cpp, mhc_pre_ttnn_nanobind.cpp, mhc_pre_xing.cpp, mhc_pre_xing.hpp
