# nlp_create_qkv_heads (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads`
- Source SHA: `222c8142326c9b4b5936dbf401c4e7eb14c216b2`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: deepseek_v41_flash_d_p FIX.15
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py, finished by hand in FIX.15 where the tool stopped before registering): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_nlp_create_qkv_heads`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`, the binding registered in `bringup_nanobind.cpp`.

## Changes

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.nlp_create_qkv_heads`, `nlp_create_q_heads_split`: the program factory's grid is the device grid cut to max_grid. Default: the device's whole grid, the stock op's program.
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. The stock op takes no core grid.
- Measured: 8x4 Galaxy, one 5120-token chunk: 120 -> 110 cores a chip (deepseek_v41_flash_d_p tests/perf/test_core_audit.py); device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp, device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp, device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp, device/nlp_create_qkv_heads_device_operation.cpp, device/nlp_create_qkv_heads_device_operation.hpp, device/nlp_create_qkv_heads_program_factory.cpp, nlp_create_qkv_heads.cpp, nlp_create_qkv_heads.hpp, nlp_create_qkv_heads_nanobind.cpp, nlp_create_qkv_heads_nanobind.hpp
