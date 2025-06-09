// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "cpp/ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::constants;

// After the full modification and if there are no issues in the overall tests, it will be added to `bfloat16.hpp` and
// applied globally.
uint32_t get_bfloat16_rounded(const float val) {
    uint32_t float_bits = *reinterpret_cast<const uint32_t*>(&val);

    // upper 16 bits
    uint16_t bfloat16_bits = float_bits >> 16;

    // check Guard, Round, Sticky bits from lower 16 bits
    uint32_t lower_bits = float_bits & 0xFFFF;
    uint32_t guard_bit = (lower_bits >> 15) & 1;
    uint32_t round_bit = (lower_bits >> 14) & 1;
    uint32_t sticky_bit = (lower_bits & 0x3FFF) != 0;

    // Tie-to-even rounding rule
    if (guard_bit && (round_bit || sticky_bit || (bfloat16_bits & 1))) {
        bfloat16_bits += 1;
    }

    return static_cast<uint32_t>(bfloat16_bits) << 16;
}

union datatype {
    uint32_t u32;
    float f32;
} u;
namespace ttnn::operations::full {
FullOperation::ProgramFactory::cached_program_t FullOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto dtype = operation_attributes.dtype;
    auto fill_value = operation_attributes.fill_value;

    auto grid = tensor_args.any.device()->compute_with_storage_grid_size();
    auto num_tiles = output.physical_volume() / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(data_format);

    // Create program
    Program program = Program();

    // Create circular buffer
    auto cb_index = tt::CBIndex::c_24;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {cb_index, 1},
        });

    // Create kernels
    std::map<string, string> reader_defines;
    switch (dtype) {
        case DataType::BFLOAT16: reader_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: reader_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: reader_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        auto float_fill_value = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = get_bfloat16_rounded(float_fill_value);
        } else {
            u.f32 = float_fill_value;
        }
    }

    auto writer_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp",
        all_cores,
        {
            (uint32_t)cb_index,
        },
        reader_defines);

    // Set runtime arguments
    uint32_t num_cores_y = grid.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        std::vector<uint32_t> writer_args = {output.buffer()->address(), u.u32, num_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, writer_id, core, writer_args);
        tile_offset += num_tiles_per_core;
    }
    return {std::move(program), {writer_id, num_cores, num_cores_y}};
}

void FullOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
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
