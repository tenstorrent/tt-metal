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
