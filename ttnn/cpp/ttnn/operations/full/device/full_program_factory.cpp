// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt;
using namespace tt::constants;

union datatype {
    uint32_t u32;
    float f32;
} u;
namespace ttnn::operations::full {
FullOperation::ProgramFactory::cached_program_t FullOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto shape = operation_attributes.shape;
    auto dtype = operation_attributes.dtype;

    auto fill_value = operation_attributes.fill_value;

    auto grid = tensor_args.any.device()->compute_with_storage_grid_size();
    auto num_tiles = output.volume() / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, num_tiles);

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(data_format);

    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        u.f32 = std::get<float>(fill_value);
    }

    // Create program
    Program program = Program();

    // Create circular buffer
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_intermed0, 1},
        });

    // Create kernels
    std::map<string, string> reader_defines;
    switch (dtype) {
        case DataType::BFLOAT16: reader_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: reader_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: reader_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    auto reader_id = tt::operations::primary::CreateReadKernel(
        program, "ttnn/cpp/ttnn/operations/full/device/kernels/reader_full.cpp", all_cores, {}, reader_defines);

    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    auto writer_id = tt::operations::primary::CreateWriteKernel(
        program, "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp", all_cores, {dst_is_dram});

    // Set runtime arguments
    uint32_t num_cores_y = grid.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core))
            num_tiles_per_core = num_tiles_per_core_group_1;
        else if (core_group_2.core_coord_in_core_ranges(core))
            num_tiles_per_core = num_tiles_per_core_group_2;
        else
            TT_FATAL(false, "Core not in specified core ranges");
        SetRuntimeArgs(program, reader_id, core, {u.u32});
        std::vector<uint32_t> writer_args = {output.buffer()->address(), num_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, writer_id, core, writer_args);
        tile_offset += num_tiles_per_core;
    }
    return {std::move(program), {reader_id, writer_id, num_cores, num_cores_y}};
}

void FullOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.core_h;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::full
