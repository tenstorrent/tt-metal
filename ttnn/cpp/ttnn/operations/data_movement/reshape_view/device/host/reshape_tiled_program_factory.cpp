// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"
#include "reshape_program_factory.hpp"

namespace ttnn::operations::data_movement::reshape {

tt::tt_metal::operation::ProgramWithCallbacks reshape_tiled_program_factory(
    const Tensor& input_tensor, const Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_shape = input_tensor.get_logical_shape();
    auto output_shape = output_tensor.get_logical_shape();

    TT_ASSERT(input_shape.size() == 3 && output_shape.size() == 3, "Kernel designed for rank 3 tensors");

    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    TT_ASSERT(input_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // set up CB for mapping metadata
    constexpr auto reader_cb_len = 2;  // shrugs

    // 4x uint32_t
    auto mapping_dataformat = tt::tt_metal::datatype_to_dataformat_converter(
        tt::tt_metal::convert_to_data_type<ReshapeMapping::value_type>());

    constexpr auto mapping_page_size = sizeof(ReshapeMapping);
    constexpr auto mapping_cb_idx = tt::CBIndex::c_0;

    const tt::tt_metal::CircularBufferConfig cb_mapping_config =
        tt::tt_metal::CircularBufferConfig(mapping_page_size * reader_cb_len, {{mapping_cb_idx, mapping_dataformat}})
            .set_page_size(mapping_cb_idx, mapping_page_size);
    const auto cb_mapping = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_mapping_config);

    // set up CB for input tiles
    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto input_tile_size_bytes = tt::tile_size(input_cb_data_format);
    constexpr auto input_cb_idx = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            input_tile_size_bytes * reader_cb_len, {{input_cb_idx, input_cb_data_format}})
            .set_page_size(input_cb_idx, input_tile_size_bytes);
    auto cb_input = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_input_config);

    const auto tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    const auto tile_hw = (tile_shape[0] * tile_shape[1]);
    const auto num_output_tiles = output_tensor.volume() / tile_hw;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    std::vector<uint32_t> reader_compile_time_args = {
        tile_shape[0],
        tile_shape[1],
        input_shape[0],
        input_shape[1],
        input_shape[2],
        output_shape[0],
        output_shape[1],
        output_shape[2],
        input_is_dram,
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    uint32_t page_idx_start = 0, page_idx_end = 0;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_2;
        }
        page_idx_end += increment;

        const std::vector<uint32_t> reader_runtime_args = {input_buffer->address(), page_idx_start, page_idx_end};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        page_idx_start += increment;
    }

    return {.program = std::move(program) /*TODO RT override callback*/};
}
};  // namespace ttnn::operations::data_movement::reshape
