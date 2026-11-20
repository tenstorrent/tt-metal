# high_bw_all_gather_ttnn (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather`
- Source SHA: `6db7958e210451b1b2530ba9ef2c1ff613e16107`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: deepseek_v41_flash_d_p PB.6L
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_high_bw_all_gather_ttnn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Row range: gathered_dim_start
- What: option `gathered_dim_start` (default None = the stock prefix, the stock program): each rank moves its local
  range [start, start + gathered_dim_size / ranks) to the same range of its fixed output slot. The chunk walk adds
  the range's first page to every page it visits (`ChipShardGeometry::first_valid_page`, read from a 9th geometry
  common arg), behind the kernel define `HBAG_FIRST_PAGE`, absent when the option is off; the host adds the arg and
  the define only then and hashes the option's presence (its value is a runtime arg). The pages of a fabric chunk
  stay in one DRAM bank (start + p + k x banks). Validation: needs gathered_dim_size, an interleaved input, no
  prefix tensor, the range within the local rows. Also dropped the copied `ttnn::experimental::bringup` using-alias
  (it named the source's namespace and did not build).
- Why: a 1M history item takes ~2.8 ms on the gather column, longer than any compute phase it hides beside (~1.8 ms);
  in pieces of ~1 ms it hides (deepseek_v41_flash_d_p PB.6L).
- Needed by: deepseek_v41_flash_d_p PB.6L
- Files: high_bw_all_gather.hpp/.cpp, high_bw_all_gather_nanobind.cpp, device/high_bw_all_gather_device_operation.cpp,
  device/high_bw_all_gather_device_operation.hpp, device/high_bw_all_gather_device_operation_types.hpp,
  device/high_bw_all_gather_fabric_factory.cpp, device/kernels/fabric/fabric_all_gather_chunk_walk.hpp,
  device/kernels/fabric/fabric_all_gather_common.hpp, tests/unit/test_range_gather.py
