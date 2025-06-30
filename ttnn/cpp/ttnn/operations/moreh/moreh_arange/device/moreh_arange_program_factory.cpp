// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_arange {
MorehArangeOperation::ProgramFactory::cached_program_t MorehArangeOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto dtype = output.dtype();
    auto W = output.padded_shape()[-1];
    auto Wt = tt::div_up(W, tt::constants::TILE_WIDTH);

    auto start = operation_attributes.start;
    auto step = operation_attributes.step;

    auto grid = tensor_args.any.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, Wt);

    // Create program
    Program program = Program();

    // Create circular buffer
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::datatype_to_dataformat_converter(dtype),
        {
            {tt::CBIndex::c_16, 1},
        });

    // Create write kernel
    std::map<std::string, std::string> writer_defines;
    switch (dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: writer_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    auto kernel_id = CreateWriteKernel(
        program,
        operation_attributes.untilize_out
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange_rm.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    // Set runtime arguments
    uint32_t core_h = grid.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }
        std::vector<uint32_t> writer_args = {
            output.buffer()->address(),
            tile_offset,
            num_tiles_per_core,
            *reinterpret_cast<uint32_t*>(&start),
            *reinterpret_cast<uint32_t*>(&step),
            output.element_size()};
        SetRuntimeArgs(program, kernel_id, core, writer_args);
        tile_offset += num_tiles_per_core;
    }
    return {std::move(program), {kernel_id, num_cores, core_h}};
}

void MorehArangeOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& kernel_id = cached_program.shared_variables.kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;
    auto src_dram_buffer_address = output.buffer()->address();

    for (uint32_t icore = 0; icore < num_cores; ++icore) {
        CoreCoord core = {icore / core_h, icore % core_h};
        auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
        runtime_args[0] = src_dram_buffer_address;
    }
}
}  // namespace ttnn::operations::moreh::moreh_arange
