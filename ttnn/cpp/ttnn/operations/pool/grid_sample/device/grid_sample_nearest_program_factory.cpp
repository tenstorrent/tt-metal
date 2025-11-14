// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Nearest neighbor grid sampling program factory - INTERLEAVED MODE ONLY
// Uses unified reader-writer kernel with optional split reader support

#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::grid_sample {

namespace {
constexpr uint32_t BUFFERING_FACTOR_NEAREST = 2;

static uint32_t get_grid_batching_factor_nearest(const Tensor& grid_tensor, bool use_precomputed_grid) {
    constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
    constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;
    return grid_tensor.logical_shape()[-1] /
           (use_precomputed_grid ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT : STANDARD_GRID_ELEMENTS_PER_POINT);
}

static uint32_t get_aligned_stick_size_nearest(const ttnn::Shape& shape, const Tensor& tensor) {
    const uint32_t stick_nbytes = shape[-1] * tensor.element_size();
    const uint32_t alignment = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                   ? tt::tt_metal::hal::get_dram_alignment()
                                   : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(stick_nbytes, alignment);
}
}  // anonymous namespace

tt::tt_metal::operation::ProgramWithCallbacks grid_sample_nearest_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels) {
    // Only interleaved mode supported
    TT_FATAL(!grid_tensor.is_sharded(), "Sharded mode not yet implemented for nearest neighbor grid sampling");
    TT_FATAL(!input_tensor.is_sharded(), "Sharded input not supported for nearest neighbor grid sampling");
    TT_FATAL(!output_tensor.is_sharded(), "Sharded output not supported for nearest neighbor grid sampling");

    tt::tt_metal::Program program{};

    // Get device and compute grid
    tt::tt_metal::IDevice* const device = output_tensor.device();
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    // Shapes
    const auto& input_shape = input_tensor.padded_shape();
    const auto& grid_shape = grid_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();

    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1];
    const uint32_t grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor_nearest(grid_tensor, use_precomputed_grid);

    // Calculate total work (number of grid sticks to process)
    const uint32_t total_grid_sticks = grid_tensor.physical_volume() / grid_shape[-1];

    // Distribute work across cores
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_grid_sticks);

    auto logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Data formats
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto grid_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype());

    // Stick sizes
    const uint32_t input_stick_size = get_aligned_stick_size_nearest(input_shape, input_tensor);
    const uint32_t grid_stick_size = get_aligned_stick_size_nearest(grid_shape, grid_tensor);
    const uint32_t output_stick_size = get_aligned_stick_size_nearest(output_shape, output_tensor);

    // Enable split reader for nearest neighbor
    // Both RISCV_0 and RISCV_1 process alternating grid points in parallel
    // Each reader has its own work CB and grid CB to avoid race conditions
    const bool enable_split_reader = true;

    // Create circular buffers
    uint32_t cb_idx = tt::CBIndex::c_0;

    // Work CB 0: Temporary storage for RISCV_0 sampled data
    const auto [work_cb_index_0, work_cb_handle_0] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        output_stick_size,
        grid_batching_factor * BUFFERING_FACTOR_NEAREST,
        input_cb_data_format);

    // Work CB 1: Temporary storage for RISCV_1 sampled data (only if split reader enabled)
    uint32_t work_cb_index_1 = work_cb_index_0;  // Default to same as CB0 if not used
    if (enable_split_reader) {
        const auto [cb_idx_1, cb_handle_1] = tt::tt_metal::create_cb(
            cb_idx++,
            program,
            all_cores,
            output_stick_size,
            grid_batching_factor * BUFFERING_FACTOR_NEAREST,
            input_cb_data_format);
        work_cb_index_1 = cb_idx_1;
    }

    // Grid CB 0: Temporary storage for one grid stick read from DRAM (RISCV_0)
    const auto [grid_cb_index_0, grid_cb_handle_0] =
        tt::tt_metal::create_cb(cb_idx++, program, all_cores, grid_stick_size, 1, grid_cb_data_format);

    // Grid CB 1: Temporary storage for one grid stick read from DRAM (RISCV_1, only if split reader enabled)
    uint32_t grid_cb_index_1 = grid_cb_index_0;  // Default to same as CB0 if not used
    if (enable_split_reader) {
        const auto [cb_idx_g1, cb_handle_g1] =
            tt::tt_metal::create_cb(cb_idx++, program, all_cores, grid_stick_size, 1, grid_cb_data_format);
        grid_cb_index_1 = cb_idx_g1;
    }

    // Compile-time arguments for the unified kernel (reader 0)
    std::vector<uint32_t> reader0_compile_time_args = {
        work_cb_index_0,                             // ct_arg[0]: work_cb_index
        grid_cb_index_0,                             // ct_arg[1]: grid_cb_index
        input_stick_size,                            // ct_arg[2]: input_stick_size
        grid_stick_size,                             // ct_arg[3]: grid_stick_size
        output_stick_size,                           // ct_arg[4]: output_stick_size
        input_height,                                // ct_arg[5]: input_height
        input_width,                                 // ct_arg[6]: input_width
        grid_batching_factor,                        // ct_arg[7]: grid_batching_factor
        static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[8]: grid_dtype
        grid_hw,                                     // ct_arg[9]: grid_hw
        use_precomputed_grid ? 1U : 0U,              // ct_arg[10]: use_precomputed_grid
        enable_split_reader ? 1U : 0U,               // ct_arg[11]: enable_split_reader
        0U                                           // ct_arg[12]: reader_id (0 for RISCV_0)
    };

    // Append TensorAccessor args to reader 0
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader0_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(reader0_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(reader0_compile_time_args);

    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
        "reader_writer_grid_sample_nearest_interleaved.cpp";

    // Create RISCV_0 kernel (reader 0)
    auto reader0_kernel_id = tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader0_compile_time_args});

    // Create RISCV_1 kernel (reader 1) if split reader enabled
    tt::tt_metal::KernelHandle reader1_kernel_id = 0;
    if (enable_split_reader) {
        // Compile-time arguments for reader 1 (RISCV_1)
        std::vector<uint32_t> reader1_compile_time_args = {
            work_cb_index_1,                             // ct_arg[0]: work_cb_index_1
            grid_cb_index_1,                             // ct_arg[1]: grid_cb_index_1 (separate)
            input_stick_size,                            // ct_arg[2]: input_stick_size
            grid_stick_size,                             // ct_arg[3]: grid_stick_size
            output_stick_size,                           // ct_arg[4]: output_stick_size
            input_height,                                // ct_arg[5]: input_height
            input_width,                                 // ct_arg[6]: input_width
            grid_batching_factor,                        // ct_arg[7]: grid_batching_factor
            static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[8]: grid_dtype
            grid_hw,                                     // ct_arg[9]: grid_hw
            use_precomputed_grid ? 1U : 0U,              // ct_arg[10]: use_precomputed_grid
            1U,                                          // ct_arg[11]: enable_split_reader
            1U                                           // ct_arg[12]: reader_id (1 for RISCV_1)
        };

        // Append TensorAccessor args to reader 1
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader1_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(reader1_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(reader1_compile_time_args);

        reader1_kernel_id = tt::tt_metal::CreateKernel(
            program,
            kernel_path,
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = reader1_compile_time_args});
    }

    // Set runtime arguments for each core
    uint32_t grid_processed = 0;
    uint32_t output_processed = 0;

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];

        // Determine how many grid sticks this core processes
        const uint32_t grid_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        // Calculate output sticks for this core
        const uint32_t output_sticks = batch_output_channels ? grid_sticks : grid_sticks * grid_batching_factor;

        std::vector<uint32_t> runtime_args = {
            input_tensor.buffer()->address(),   // rt_arg[0]: input_addr
            output_tensor.buffer()->address(),  // rt_arg[1]: output_addr
            grid_tensor.buffer()->address(),    // rt_arg[2]: grid_addr
            grid_sticks,                        // rt_arg[3]: grid_sticks
            grid_processed,                     // rt_arg[4]: grid_start_id
            output_processed                    // rt_arg[5]: output_start_id
        };

        // Set runtime args for both readers
        tt::tt_metal::SetRuntimeArgs(program, reader0_kernel_id, core, runtime_args);
        if (enable_split_reader) {
            tt::tt_metal::SetRuntimeArgs(program, reader1_kernel_id, core, runtime_args);
        }

        grid_processed += grid_sticks;
        output_processed += output_sticks;
    }

    // Runtime callback for address updates
    return {
        std::move(program),
        [=](const void*,
            tt::tt_metal::Program& prog,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& grid = input_tensors[1];
            const auto& output = output_tensors[0];

            // Update buffer addresses in runtime args for both readers
            for (uint32_t i = 0; i < num_cores; i++) {
                const CoreCoord& core = logical_cores[i];

                auto& runtime_args0 = tt::tt_metal::GetRuntimeArgs(prog, reader0_kernel_id, core);
                runtime_args0[0] = input.buffer()->address();
                runtime_args0[1] = output.buffer()->address();
                runtime_args0[2] = grid.buffer()->address();

                if (enable_split_reader) {
                    auto& runtime_args1 = tt::tt_metal::GetRuntimeArgs(prog, reader1_kernel_id, core);
                    runtime_args1[0] = input.buffer()->address();
                    runtime_args1[1] = output.buffer()->address();
                    runtime_args1[2] = grid.buffer()->address();
                }
            }
        }};
}

}  // namespace ttnn::operations::grid_sample
