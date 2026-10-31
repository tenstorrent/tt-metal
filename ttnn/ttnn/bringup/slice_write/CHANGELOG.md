# slice_write (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/slice_write`
- Source SHA: `222c8142326c9b4b5936dbf401c4e7eb14c216b2`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: deepseek_v41_flash_d_p FIX.15
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py, finished by hand in FIX.15 where the tool stopped before registering): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_slice_write`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`, the binding registered in `bringup_nanobind.cpp`.

## Changes

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.slice_write`: the interleaved row-major work split (and the cache-hit runtime-argument override) runs on that rectangle. Default: the device's whole grid, the stock op's program.
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. The stock op takes no core grid.
- Measured: 8x4 Galaxy, one 5120-token chunk: 120 -> 110 cores a chip (deepseek_v41_flash_d_p tests/perf/test_core_audit.py); device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/kernels/dataflow/slice_write_reader_interleaved.cpp, device/kernels/dataflow/slice_write_writer_interleaved.cpp, device/kernels/dataflow/slice_write_writer_interleaved_strided.cpp, device/slice_write_device_operation.cpp, device/slice_write_device_operation.hpp, device/slice_write_device_operation_types.hpp, device/slice_write_rm_interleaved_program_factory.cpp, device/slice_write_rm_interleaved_program_factory.hpp, device/slice_write_rm_sharded_input_program_factory.cpp, device/slice_write_rm_sharded_input_program_factory.hpp, device/slice_write_tiled_sharded_input_program_factory.cpp, device/slice_write_tiled_sharded_input_program_factory.hpp, slice_write.cpp, slice_write.hpp, slice_write_nanobind.cpp, slice_write_nanobind.hpp
