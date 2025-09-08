// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "grid_sample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"

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
    bool extend_channels) {
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

    // Calculate output sticks per core based on extend_channels mode
    const uint32_t output_nsticks_per_core =
        extend_channels ? grid_nsticks_per_core :          // extend_channels=true: 1:1 ratio
            grid_nsticks_per_core * grid_batching_factor;  // extend_channels=false: 1:K ratio

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t buffering_factor = 1;  // Data is already in shards

    // CB0: Grid data buffer - local sharded grid data
    // This CB points directly to the grid tensor's L1 buffer
    const uint32_t grid_stick_nbytes = grid_shape[-1] * grid_tensor.element_size();
    const uint32_t aligned_grid_stick_nbytes = tt::round_up(grid_stick_nbytes, grid_tensor.buffer()->alignment());
    const uint32_t grid_cb_pagesize = aligned_grid_stick_nbytes;
    const uint32_t grid_cb_npages = grid_nsticks_per_core * buffering_factor;

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
    const uint32_t input_cb_num_pages = 4 * buffering_factor;  // 4 corner points for bilinear

    const auto [input_cb_index, input_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, input_cb_page_size, input_cb_num_pages, input_cb_data_format);

    // CB2: Scalar buffer (holds 4 bilinear interpolation weights)
    const uint32_t scalar_cb_num_pages = buffering_factor;
    const uint32_t scalar_cb_page_size = tt::tt_metal::detail::TileSize(grid_cb_data_format);
    const auto [scalar_cb_index, scalar_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, scalar_cb_page_size, scalar_cb_num_pages, grid_cb_data_format);

    // CB3: Output buffer - local sharded output
    // This CB points directly to the output tensor's L1 buffer
    const uint32_t output_stick_nbytes = output_shape[-1] * output_tensor.element_size();
    const uint32_t aligned_output_stick_nbytes = tt::round_up(output_stick_nbytes, output_tensor.buffer()->alignment());
    const uint32_t output_cb_pagesize = aligned_output_stick_nbytes;
    const uint32_t output_cb_npages = output_nsticks_per_core * buffering_factor;

    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        output_cb_pagesize,
        output_cb_npages,
        output_cb_data_format,
        output_tensor.buffer());

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,                // 0: input CB index
        (std::uint32_t)grid_cb_index,                 // 1: grid CB index
        (std::uint32_t)scalar_cb_index,               // 2: scalar CB index
        (std::uint32_t)output_cb_index,               // 3: output CB index
        (std::uint32_t)aligned_grid_stick_nbytes,     // 4: grid stick size
        (std::uint32_t)input_height,                  // 5: input height
        (std::uint32_t)input_width,                   // 6: input width
        (std::uint32_t)grid_nsticks_per_core,         // 7: grid sticks per core
        (std::uint32_t)output_nsticks_per_core,       // 8: output sticks per core
        (std::uint32_t)grid_batching_factor,          // 9: number of grids per spatial position
        (std::uint32_t)extend_channels ? 1 : 0,       // 10: extend_channels flag
        (std::uint32_t)use_precomputed_grid ? 1 : 0,  // 11: precomputed grid flag
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

    // Runtime arguments - only input buffer address needs updating
    const std::vector<uint32_t> reader_rt_arguments{
        input_tensor.buffer()->address(),  // 0: input buffer address
    };

    const std::vector<CoreCoord> logical_cores = tt::tt_metal::corerange_to_cores(all_cores, ncores, true);

    for (uint32_t i = 0; i < ncores; i++) {
        const CoreCoord& core = logical_cores[i];
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_arguments);
    }

    // Runtime arguments callback for sharded tensors
    const tt::tt_metal::operation::OverrideRuntimeArgumentsCallback override_runtime_args_callback =
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
                std::vector<uint32_t>& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_tensor.buffer()->address();  // Update input buffer address
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::grid_sample
