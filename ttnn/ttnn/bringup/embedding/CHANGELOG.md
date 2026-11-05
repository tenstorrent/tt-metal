# embedding (fork)

- Source: `ttnn/cpp/ttnn/operations/embedding`
- Source SHA: `222c8142326c9b4b5936dbf401c4e7eb14c216b2`
- Python: `ttnn.bringup.*` (was `ttnn.*`)
- Forked for: deepseek_v41_flash_d_p FIX.15
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py, finished by hand in FIX.15 where the tool stopped before registering): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_embedding`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`, the binding registered in `bringup_nanobind.cpp`.

## Changes

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.embedding`: an interleaved output's work split runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1). Default: the device's whole grid, the stock op's program.
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. The stock op takes no core grid.
- Measured: 8x4 Galaxy, one 5120-token chunk: 120 -> 110 cores a chip (deepseek_v41_flash_d_p tests/perf/test_core_audit.py); device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/embedding_device_operation.cpp, device/embedding_device_operation.hpp, device/embedding_device_operation_types.hpp, device/embedding_program_factory_common.cpp, device/embedding_program_factory_common.hpp, device/embeddings_fused_program_factory.cpp, device/embeddings_fused_program_factory.hpp, device/embeddings_rm_program_factory.cpp, device/embeddings_rm_program_factory.hpp, device/embeddings_tilized_indices_program_factory.cpp, device/embeddings_tilized_indices_program_factory.hpp, device/kernels/compute/tilize_chunked.cpp, device/kernels/dataflow/embedding_ind_tilized.cpp, device/kernels/dataflow/embeddings.cpp, device/kernels/dataflow/embeddings_common.hpp, device/kernels/dataflow/embeddings_common_metal2.hpp, device/kernels/dataflow/embeddings_rm_writer_chunked.cpp, device/kernels/dataflow/embeddings_tilize.cpp, embedding.cpp, embedding.hpp, embedding_nanobind.cpp, embedding_nanobind.hpp

### Upstream sync to source @ `107f838f379` (2026-11-05): full sync, round-up helper cleanups
- What: a 3-way merge per file (base = fork_op.py's mechanical copy of the source @ `6b80daf43e3`, the fork's last sync / fork point, no skipped commits; upstream = the same copy @ `107f838f379` (origin/llk_helper_library); ours = this fork) (the fork was taken @ `222c8142326`, a bring-up branch commit whose source equals `6b80daf43e3`'s). 3 files merged clean, no conflicts. Taken (2 commits): `15a3e2ce379` #60814 (`<tt-metalium/tt_align.hpp>` merged into `math.hpp`, include dropped / replaced), `c420b48f699` #60730 (`round_up_to_mul32(x)` -> `tt::round_up(x, 32u)`).
- Behaviour: none (same page sizes). The fork's max_grid merged clean.
- Files: device/embeddings_fused_program_factory.cpp, device/embeddings_rm_program_factory.cpp, device/embeddings_tilized_indices_program_factory.cpp
