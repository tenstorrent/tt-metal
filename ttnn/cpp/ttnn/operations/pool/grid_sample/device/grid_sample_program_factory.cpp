// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::grid_sample {

tt::tt_metal::operation::ProgramWithCallbacks grid_sample_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& mode,
    const std::string& padding_mode,
    bool use_precomputed_grid) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat grid_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    tt::tt_metal::IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& grid_shape = grid_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();

    const uint32_t batch_size = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];

    const uint32_t output_height = grid_shape[1];
    const uint32_t output_width = grid_shape[2];

    // Calculate stick sizes (full rows in last dimension)
    const uint32_t grid_buffer_alignment = grid_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                               ? tt::tt_metal::hal::get_dram_alignment()
                                               : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                              ? tt::tt_metal::hal::get_dram_alignment()
                                              : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                              ? tt::tt_metal::hal::get_dram_alignment()
                                              : tt::tt_metal::hal::get_l1_alignment();

    const uint32_t grid_stick_nbytes = grid_shape[-1] * grid_tensor.element_size();
    const uint32_t aligned_grid_stick_nbytes = tt::round_up(grid_stick_nbytes, grid_buffer_alignment);
    const uint32_t input_stick_nbytes = input_shape[-1] * input_tensor.element_size();
    const uint32_t aligned_input_stick_nbytes = tt::round_up(input_stick_nbytes, src_buffer_alignment);
    const uint32_t output_stick_nbytes = output_shape[-1] * output_tensor.element_size();
    const uint32_t aligned_output_stick_nbytes = tt::round_up(output_stick_nbytes, dst_buffer_alignment);

    // Calculate number of output sticks (total rows to process)
    uint32_t output_nsticks = grid_tensor.physical_volume() / grid_shape[-1];

    // Get device grid for multicore
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t total_cores = num_cores_x * num_cores_y;

    // Work distribution - distribute output sticks across cores
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, output_nsticks);

    uint32_t next_cb_index = tt::CBIndex::c_0;

    const uint32_t buffering_factor = 2;

    // CB0: Grid data buffer (holds grid coordinates for current output position)
    const uint32_t grid_cb_num_pages = buffering_factor;
    const auto [grid_cb_index, grid_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_grid_stick_nbytes, grid_cb_num_pages, grid_cb_data_format);

    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[-1] / tt::constants::TILE_WIDTH);
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * input_tensor.element_size();

    // CB1: Input data buffer (holds 4 input sticks for bilinear interpolation)
    uint32_t input_cb_num_pages = buffering_factor;  // 4 corner sticks for bilinear
    const auto [input_cb_index, input_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, input_cb_page_size, input_cb_num_pages, input_cb_data_format);

    // CB2: Scalar buffer (holds 4 bilinear interpolation weights)
    const uint32_t scalar_cb_num_pages = buffering_factor;
    const uint32_t scalar_cb_page_size = tile_size(grid_cb_data_format);  // 4 scalars, aligned
    const auto [scalar_cb_index, scalar_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, scalar_cb_page_size, scalar_cb_num_pages, grid_cb_data_format);

    // CB3: Output buffer
    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)input_shape[-1] / tt::constants::FACE_WIDTH);

    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const uint32_t output_cb_num_pages = out_ntiles_c * buffering_factor;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, output_cb_page_size, output_cb_num_pages, output_cb_data_format);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,              // input CB index
        (std::uint32_t)grid_cb_index,               // grid CB index
        (std::uint32_t)scalar_cb_index,             // scalar CB index
        (std::uint32_t)aligned_input_stick_nbytes,  // input stick size
        (std::uint32_t)aligned_grid_stick_nbytes,   // grid stick size
        (std::uint32_t)input_height,
        (std::uint32_t)input_width,
        (std::uint32_t)(output_height * output_width),  // output_hw_size at index 13
    };

    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(reader_compile_time_args);

    const uint32_t MAX_TILES_PER_REDUCTION = 8;

    // Grid sample parameters for pool compute kernel adaptation
    const uint32_t reduction_size = 4;                    // Always 4 for bilinear interpolation
    const bool split_reader = false;                      // No split reader for grid sample
    const uint32_t channels_per_shard = input_shape[-1];  // All channels in one "shard"
    const uint32_t in_nblocks_c = (uint32_t)std::ceil(
        (float)in_ntiles_c / MAX_TILES_PER_REDUCTION);    // For now 1, will add wide reduction support later
    const uint32_t max_rows_for_reduction = 4;            // 4 corner values
    const bool one_scalar_per_core = false;               // Scalars change during computation
    const uint32_t dummy_cb_id = 32;                      // Unused CB for split reader

    // Create compute kernels for different work loads
    // We'll create kernels for cores with same work amount together
    tt::tt_metal::KernelHandle compute_kernel_group_1 = 0;
    tt::tt_metal::KernelHandle compute_kernel_group_2 = 0;

    // Kernel for core group 1
    if (core_group_1.num_cores() > 0) {
        std::vector<uint32_t> compute_compile_time_args_1 = {
            in_ntiles_c,                  // 0: Input tiles per channel
            reduction_size,               // 1: Reduction size (4 for bilinear)
            split_reader,                 // 2: Split reader flag
            num_sticks_per_core_group_1,  // 3: Work per core for group 1
            channels_per_shard,           // 4: Channels per shard
            in_nblocks_c,                 // 5: Channel blocks
            max_rows_for_reduction,       // 6: Max rows
            input_cb_index,               // 7: Input CB
            dummy_cb_id,                  // 8: Input CB 1 (unused)
            scalar_cb_index,              // 9: Scalar CB
            dummy_cb_id,                  // 10: Scalar CB 1 (unused)
            output_cb_index,              // 11: Output CB
            one_scalar_per_core,          // 12: Scalar mode
            in_ntiles_c                   // 13: Tiles per channel (for CB space reservation)
        };

        compute_kernel_group_1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
            core_group_1,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,  // Use bfloat16
                .math_approx_mode = false,
                .compile_args = compute_compile_time_args_1,
                .defines = get_defines(pool::Pool2DType::AVG_POOL2D)});
    }

    // Kernel for core group 2 (if it exists)
    if (core_group_2.num_cores() > 0) {
        std::vector<uint32_t> compute_compile_time_args_2 = {
            in_ntiles_c,                  // 0: Input tiles per channel
            reduction_size,               // 1: Reduction size (4 for bilinear)
            split_reader,                 // 2: Split reader flag
            num_sticks_per_core_group_2,  // 3: Work per core for group 2
            channels_per_shard,           // 4: Channels per shard
            in_nblocks_c,                 // 5: Channel blocks
            max_rows_for_reduction,       // 6: Max rows
            input_cb_index,               // 7: Input CB
            dummy_cb_id,                  // 8: Input CB 1 (unused)
            scalar_cb_index,              // 9: Scalar CB
            dummy_cb_id,                  // 10: Scalar CB 1 (unused)
            output_cb_index,              // 11: Output CB
            one_scalar_per_core,          // 12: Scalar mode
            in_ntiles_c                   // 13: Tiles per channel (for CB space reservation)
        };

        compute_kernel_group_2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,  // Use bfloat16
                .math_approx_mode = false,
                .compile_args = compute_compile_time_args_2,
                .defines = get_defines(pool::Pool2DType::AVG_POOL2D)  // Use avg pool defines
            });
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,              // output CB index
        (std::uint32_t)aligned_output_stick_nbytes,  // output stick size
        (std::uint32_t)out_ntiles_c                  // number of tiles per channel (for ntiles_c pages per stick)
    };

    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp";
    std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_rt_arguments{
        input_tensor.buffer()->address(),
        grid_tensor.buffer()->address(),
        0,  // set in loop, num of sticks per core
        0   // set in loop, start_stick_id
    };

    std::vector<uint32_t> writer_rt_arguments{
        output_tensor.buffer()->address(),
        0,  // set in loop, num of sticks per core
        0   // set in loop, start_stick_id
    };

    for (uint32_t i = 0, sticks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t sticks_per_core = 0;
        tt::tt_metal::KernelHandle compute_kernel_for_core = 0;

        if (core_group_1.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[2] = sticks_per_core;   // num sticks for this core
        reader_rt_arguments[3] = sticks_processed;  // start stick id

        // Compute runtime arguments - pool kernel expects no runtime args (all compile-time)

        writer_rt_arguments[1] = sticks_per_core;   // num sticks for this core
        writer_rt_arguments[2] = sticks_processed;  // start stick id

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_arguments);

        sticks_processed += sticks_per_core;
    }

    // Runtime arguments callback following upsample pattern
    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& input_tensor = input_tensors.at(0);
        const auto& grid_tensor = input_tensors.at(1);
        const auto& output_tensor = output_tensors.at(0);

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_tensor.buffer()->address();
                runtime_args[1] = grid_tensor.buffer()->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_tensor.buffer()->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::grid_sample
