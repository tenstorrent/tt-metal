// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "grid_sample_utils.hpp"

namespace ttnn::operations::grid_sample {
namespace nearest {

tt::tt_metal::operation::ProgramWithCallbacks grid_sample_nearest_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& padding_mode,
    bool align_corners,
    bool use_precomputed_grid,
    bool batch_output_channels) {
    const bool is_sharded = grid_tensor.is_sharded();
    tt::tt_metal::Program program{};

    // Data formats and device
    const auto [input_cb_data_format, grid_cb_data_format, output_cb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));
    tt::tt_metal::IDevice* const device = output_tensor.device();

    // Shape and dimensions
    const auto& [input_shape, grid_shape, output_shape] =
        std::tie(input_tensor.padded_shape(), grid_tensor.padded_shape(), output_tensor.padded_shape());
    const uint32_t input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = utils::get_grid_batching_factor(grid_tensor, use_precomputed_grid, "nearest");
    const bool enable_split_reader = utils::should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid);
    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        all_cores = grid_shard_spec.grid;
        num_cores = grid_shard_spec.num_cores();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    } else {
        const uint32_t grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, grid_nsticks);

        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;

        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    uint32_t cb_idx = tt::CBIndex::c_0;

    // Create CBs
    const uint32_t grid_stick_size = is_sharded ? grid_shape[-1] * grid_tensor.element_size()
                                                : utils::get_aligned_stick_size(grid_shape, grid_tensor);
    const auto [grid_cb_index, grid_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        grid_stick_size,
        is_sharded ? grid_nsticks_per_core : 1,
        grid_cb_data_format,
        is_sharded ? grid_tensor.buffer() : nullptr);

    const uint32_t out_ntiles_c = 1;
    const uint32_t output_cb_page_size = (float)output_shape[-1];
    const uint32_t output_cb_pages =
        is_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        output_cb_page_size,
        output_cb_pages,
        output_cb_data_format,
        is_sharded ? output_tensor.buffer() : nullptr);

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = utils::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t grid_stick_size_arg = is_sharded ? grid_shape[-1] * grid_tensor.element_size()
                                                    : utils::get_aligned_stick_size(grid_shape, grid_tensor);

    // Writer kernel (nearest mode uses writer for computation when sharded)
    tt::tt_metal::KernelHandle writer_kernel_id = 0, writer1_kernel_id = 0;
    if (!is_sharded) {
        // Writer compile-time arguments for interleaved output
        std::vector<uint32_t> writer_compile_time_args = {
            output_cb_index,                                             // ct_arg[0]: output_cb_index
            utils::get_aligned_stick_size(output_shape, output_tensor),  // ct_arg[1]: output_stick_size
            out_ntiles_c                                                 // ct_arg[2]: out_ntiles_c
        };
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    } else {
        // Writer for nearest mode with sharded grid - performs the computation
        std::vector<uint32_t> writer_compile_time_args = {
            grid_cb_index,                               // ct_arg[0]: grid_cb_index
            output_cb_index,                             // ct_arg[1]: output_cb_index
            input_stick_size,                            // ct_arg[2]: input_stick_size
            grid_stick_size_arg,                         // ct_arg[3]: grid_stick_size
            input_height,                                // ct_arg[4]: input_height
            input_width,                                 // ct_arg[5]: input_width
            grid_batching_factor,                        // ct_arg[6]: grid_batching_factor
            static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[7]: grid_dtype
            grid_hw,                                     // ct_arg[8]: grid_hw
            use_precomputed_grid ? 1U : 0U,              // ct_arg[9]: use_precomputed_grid
            align_corners ? 1U : 0U,                     // ct_arg[10]: align_corners
            enable_split_reader ? 1U : 0U,               // ct_arg[11]: split_reader
            0U,                                          // ct_arg[12]: reader_id (will be set per core)
            grid_nsticks_per_core                        // ct_arg[13]: grid_nsticks_per_core
        };

        // Add tensor accessor args for input tensor
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(writer_compile_time_args);

        auto create_writer_config = [&](const std::vector<uint32_t>& args, auto processor, auto noc) {
            return tt::tt_metal::DataMovementConfig{.processor = processor, .noc = noc, .compile_args = args};
        };

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp",
            all_cores,
            create_writer_config(
                writer_compile_time_args,
                tt::tt_metal::DataMovementProcessor::RISCV_0,
                tt::tt_metal::NOC::RISCV_0_default));

        if (enable_split_reader) {
            auto writer1_compile_time_args = writer_compile_time_args;
            writer1_compile_time_args[12] = 1;  // ct_arg[12]: reader_id = 1

            writer1_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                "writer_grid_sample_nearest_sharded.cpp",
                all_cores,
                create_writer_config(
                    writer1_compile_time_args,
                    tt::tt_metal::DataMovementProcessor::RISCV_1,
                    tt::tt_metal::NOC::RISCV_1_default));
        }
    }

    // Set runtime arguments
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for nearest mode writer (which acts as reader/processor)
            std::vector<uint32_t> runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                i * grid_nsticks_per_core          // rt_arg[1]: grid_stick_offset
            };

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            if (enable_split_reader) {
                tt::tt_metal::SetRuntimeArgs(program, writer1_kernel_id, core, runtime_args);
            }
        }
    } else {
        uint32_t grid_processed = 0;
        uint32_t output_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            const uint32_t output_sticks = batch_output_channels ? grid_sticks : grid_sticks * grid_batching_factor;

            // Runtime arguments for interleaved writer
            std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
                output_sticks,                      // rt_arg[1]: output_sticks
                output_processed                    // rt_arg[2]: output_processed
            };

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

            grid_processed += grid_sticks;
            output_processed += output_sticks;
        }
    }

    // Runtime callback
    return {
        std::move(program),
        [=](const void*,
            tt::tt_metal::Program& prog,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& [input_tensor, grid_tensor] = std::tie(input_tensors[0], input_tensors[1]);
            const auto& output_tensor = output_tensors[0];

            if (is_sharded) {
                tt::tt_metal::UpdateDynamicCircularBufferAddress(prog, grid_cb_handle, *grid_tensor.buffer());
                tt::tt_metal::UpdateDynamicCircularBufferAddress(prog, output_cb_handle, *output_tensor.buffer());
                for (uint32_t i = 0; i < num_cores; i++) {
                    const CoreCoord& core = logical_cores[i];
                    tt::tt_metal::GetRuntimeArgs(prog, writer_kernel_id, core)[0] = input_tensor.buffer()->address();
                    if (enable_split_reader) {
                        tt::tt_metal::GetRuntimeArgs(prog, writer1_kernel_id, core)[0] =
                            input_tensor.buffer()->address();
                    }
                }
            } else {
                for (uint32_t i = 0; i < num_cores; i++) {
                    const CoreCoord& core = logical_cores[i];
                    tt::tt_metal::GetRuntimeArgs(prog, writer_kernel_id, core)[0] =
                        output_tensor.buffer()->address();  // rt_arg[0]: output_buffer_address
                }
            }
        }};
}

}  // namespace nearest
}  // namespace ttnn::operations::grid_sample
