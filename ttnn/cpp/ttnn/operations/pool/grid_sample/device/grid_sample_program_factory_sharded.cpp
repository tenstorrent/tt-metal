// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "grid_sample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::grid_sample {

tt::tt_metal::operation::ProgramWithCallbacks grid_sample_program_factory_sharded(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& mode,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat grid_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    tt::tt_metal::IDevice* const device = output_tensor.device();

    const ttnn::Shape& input_shape = input_tensor.padded_shape();
    const ttnn::Shape& grid_shape = grid_tensor.padded_shape();
    const ttnn::Shape& output_shape = output_tensor.padded_shape();

    const uint32_t batch_size = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];

    const uint32_t output_height = grid_shape[1];
    const uint32_t output_width = grid_shape[2];

    // Calculate the number of data points batched into a single grid row
    const uint32_t grid_last_dim = grid_tensor.logical_shape()[-1];
    const uint32_t num_of_elements_per_grid_point =
        use_precomputed_grid ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT : STANDARD_GRID_ELEMENTS_PER_POINT;
    const uint32_t grid_batching_factor = grid_last_dim / num_of_elements_per_grid_point;

    // Get shard specs
    const tt::tt_metal::ShardSpec grid_shard_spec = grid_tensor.shard_spec().value();
    const tt::tt_metal::CoreRangeSet all_cores = grid_shard_spec.grid;
    const uint32_t ncores = grid_shard_spec.num_cores();

    // Calculate sticks per core for grid tensor
    const uint32_t grid_nsticks_per_core = grid_shard_spec.shape[0];

    // Calculate output sticks per core based on batch_output_channels mode
    const uint32_t output_nsticks_per_core =
        batch_output_channels ? grid_nsticks_per_core :    // batch_output_channels=true: 1:1 ratio
            grid_nsticks_per_core * grid_batching_factor;  // batch_output_channels=false: 1:K ratio

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t buffering_factor = 2;  // Data is already in shards

    // CB0: Grid data buffer - local sharded grid data
    // This CB points directly to the grid tensor's L1 buffer
    const uint32_t grid_stick_nbytes = grid_shape[-1] * grid_tensor.element_size();
    const uint32_t grid_cb_pagesize = grid_stick_nbytes;
    const uint32_t grid_cb_npages = grid_nsticks_per_core;

    const auto [grid_cb_index, grid_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        grid_cb_pagesize,
        grid_cb_npages,
        grid_cb_data_format,
        grid_tensor.buffer());

    // CB1: Input data buffer - for remote NOC reads from input tensor shards
    // This is NOT tied to a buffer since we'll do remote reads
    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[-1] / tt::constants::TILE_WIDTH);
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * input_tensor.element_size();
    const uint32_t input_cb_num_pages = buffering_factor;

    const auto [input_cb_index, input_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, input_cb_page_size, input_cb_num_pages, input_cb_data_format);

    // CB2: Scalar buffer (holds 4 bilinear interpolation weights)
    const uint32_t scalar_cb_num_pages = buffering_factor;
    const uint32_t scalar_cb_page_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    const auto [scalar_cb_index, scalar_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, scalar_cb_page_size, scalar_cb_num_pages, input_cb_data_format);

    // CB3: Output buffer - local sharded output (following pool pattern)
    // This CB points directly to the output tensor's L1 buffer
    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[-1] / tt::constants::FACE_WIDTH);
    const uint32_t output_cb_pagesize = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const uint32_t output_cb_npages = output_nsticks_per_core * out_ntiles_c;

    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        output_cb_pagesize,
        output_cb_npages,
        output_cb_data_format,
        output_tensor.buffer());

    const uint32_t input_stick_nbytes = input_shape[-1] * input_tensor.element_size();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,                // 0: input CB index
        (std::uint32_t)grid_cb_index,                 // 1: grid CB index
        (std::uint32_t)scalar_cb_index,               // 2: scalar CB index
        (std::uint32_t)input_stick_nbytes,            // 3: input stick size in bytes
        (std::uint32_t)grid_stick_nbytes,             // 4: grid stick size
        (std::uint32_t)input_height,                  // 5: input height
        (std::uint32_t)input_width,                   // 6: input width
        (std::uint32_t)grid_nsticks_per_core,         // 7: grid sticks per core
        (std::uint32_t)grid_batching_factor,          // 8: number of grids per spatial position
        (std::uint32_t)use_precomputed_grid ? 1 : 0,  // 9: precomputed grid flag
    };

    // Add input tensor accessor args for remote NOC reads
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    // Create reader defines for precomputed grid
    std::map<std::string, std::string> reader_defines;
    if (use_precomputed_grid) {
        reader_defines["USE_PRECOMPUTED_GRID"] = "1";
    }

    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp";

    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    // Create compute kernel (adapted from interleaved version)
    const uint32_t MAX_TILES_PER_REDUCTION = 8;

    // Grid sample parameters for pool compute kernel adaptation
    const uint32_t reduction_size = 4;                    // Always 4 for bilinear interpolation
    const bool split_reader = false;                      // No split reader for grid sample
    const uint32_t channels_per_shard = input_shape[-1];  // All channels in one "shard"
    const uint32_t in_nblocks_c = (uint32_t)std::ceil(
        (float)in_ntiles_c / MAX_TILES_PER_REDUCTION);  // For now 1, will add wide reduction support later
    const uint32_t max_rows_for_reduction = 16;         // 4 corner values
    const bool one_scalar_per_core = false;             // Scalars change during computation
    const uint32_t dummy_cb_id = 32;                    // Unused CB for split reader

    std::vector<uint32_t> compute_compile_time_args = {
        in_ntiles_c,                                     // 0: Input tiles per channel
        reduction_size,                                  // 1: Reduction size (4 for bilinear)
        split_reader,                                    // 2: Split reader flag
        grid_batching_factor * output_nsticks_per_core,  // 3: Total grid interpolations per core
        channels_per_shard,                              // 4: Channels per shard
        in_nblocks_c,                                    // 5: Channel blocks
        max_rows_for_reduction,                          // 6: Max rows
        input_cb_index,                                  // 7: Input CB
        dummy_cb_id,                                     // 8: Input CB 1 (unused)
        dummy_cb_id,                                     // 9: Index Input CB (unused)
        dummy_cb_id,                                     // 10: Index Input CB 1 (unused)
        scalar_cb_index,                                 // 11: Scalar CB
        dummy_cb_id,                                     // 12: Scalar CB 1 (unused)
        dummy_cb_id,                                     // 13: Tile Temp CB (unused)
        dummy_cb_id,                                     // 14: Index Tile Temp CB (unused)
        output_cb_index,                                 // 15: Output CB
        dummy_cb_id,                                     // 16: Index Output CB (unused)
        one_scalar_per_core,                             // 17: Scalar mode
        false,                                           // 18: Return Indices (unused)
    };

    const tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,  // Use bfloat16
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
            .defines = get_defines(pool::Pool2DType::AVG_POOL2D)  // Use avg pool defines
        });

    // Runtime arguments - only input buffer address needs updating
    const std::vector<uint32_t> reader_rt_arguments{
        input_tensor.buffer()->address(),  // 0: input buffer address
    };

    const std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, ncores, true);

    for (uint32_t i = 0; i < ncores; i++) {
        const CoreCoord& core = logical_cores[i];
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_arguments);
    }

    // Runtime arguments callback for sharded tensors
    const auto override_runtime_args_callback =
        [reader_kernel_id, grid_cb_handle, output_cb_handle, logical_cores, ncores](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const Tensor& input_tensor = input_tensors.at(0);
            const Tensor& grid_tensor = input_tensors.at(1);
            const Tensor& output_tensor = output_tensors.at(0);

            // Update sharded buffer addresses for grid and output CBs
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, grid_cb_handle, *grid_tensor.buffer());
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, output_cb_handle, *output_tensor.buffer());

            // Update runtime args for input buffer address
            for (uint32_t i = 0; i < ncores; i++) {
                const CoreCoord& core = logical_cores[i];
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_tensor.buffer()->address();  // Update input buffer address
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::grid_sample
