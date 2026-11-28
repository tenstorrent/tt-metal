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

### server order: the ids reorder as the reader (deepseek_v41_flash_d_p NC.9)
- What: options `server_start`, `server_valid`, `clamp_max` (all three or none; default none = the op as before). Set: the input is the engine's uint32 ids in the server's row order of every mesh row ([SP, 1, W], every chip the same: after the model's SP all-gather), and the fused (TILE output) reader reads each natural 32-id unit of its mesh row where deepseek_v41_flash_d_p's IdsReorder kernel reads it, positions >= server_valid and ids > clamp_max read as clamp_max. One program a mesh row (new `EmbeddingsServerOrderProgramFactory`, a Metal 2.0 mesh-workload spec factory: the row index a common runtime arg); server_start / server_valid are common runtime args rewritten on every program-cache hit (`override_runtime_arguments`), kept out of the hash by `EmbeddingParams::attribute_names` (clamp_max is in it).
- Why: OWNER 2026-11-28 (NC.9): the IdsReorder kernel and its ids tensor were one more launch a chunk before the embedding.
- Measured: 8x4, op level (models/demos/deepseek_v41_flash_d_p/tests/native/test_native_fusions.py): bytes equal to IdsReorder + this op (8192 and 5120 chunks, two starts / valid counts each, ids past the vocabulary), 0 program-cache growth on new values; host 163-165 -> 133-134 us a call (the SP all-gather included), device 42-48 -> 34-36 us. Cores unchanged (max_grid).
- Limits: GENERIC embeddings, a uint32 ROW_MAJOR input, a TILE interleaved output; the mesh's rows are the SP axis (input dim 0 == mesh rows).
- Needed by: deepseek_v41_flash_d_p NC.9 (tt/page_copy.IdsReorder.embed, tt/model.embed).
- Files: embedding.cpp, embedding.hpp, embedding_nanobind.cpp, device/embedding_device_operation.cpp, device/embedding_device_operation.hpp, device/embedding_device_operation_types.hpp, device/embeddings_fused_program_factory.cpp, device/embeddings_fused_program_factory.hpp, device/kernels/dataflow/embeddings_tilize.cpp
- Fork regression: not run (owner, NATIVE_PORT_RULES.md: no fork regression suites on the 8x4; the fork carries no source tests yet, tests/source.yaml).

### Brought to model_bringup (2026-11-28, deepseek_v41_flash_d_p MB.QB)
- What: the server-order option above (NC.9), taken as is from dnijemcevic/model_bringup_batching (e19a95cdda7; the fork's files byte-identical to that branch).
- Why: OWNER 2026-11-28, the single-request fusions on model_bringup (NATIVE_FUSIONS, default on).
- Measured (8x4, 5120 chunk, models/demos/deepseek_v41_flash_d_p/tests/native/test_native_mb_qb.py, results/MB.QB_layer.md): bytes equal to IdsReorder + this op (two starts / valid counts, ids past the vocabulary), repeat 0, 0 program-cache growth; host 166.0 -> 136.4 us a call, device 42.4 -> 34.5 us. Cores unchanged (max_grid).
- Used by: deepseek_v41_flash_d_p tt/page_copy.IdsReorder.embed, tt/model.embed (model_bringup).
