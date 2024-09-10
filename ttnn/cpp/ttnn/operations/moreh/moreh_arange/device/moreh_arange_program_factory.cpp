// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_arange {
MorehArangeOperation::ProgramFactory::cached_program_t MorehArangeOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto W = output_tensor.get_legacy_shape()[-1];
    auto Wt = tt::div_up(W, tt::constants::TILE_WIDTH);

    auto start = operation_attributes.start;
    auto step = operation_attributes.step;

    auto grid = tensor_args.any.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, Wt);

    // Create program
    Program program = Program();

    // Create circular buffer
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_out0, 1},
        });

    // Create write kernel
    std::map<string, string> writer_defines;
    if (output_tensor.get_dtype() == DataType::BFLOAT16) {
        writer_defines["OUTPUT_DTYPE_BFLOAT16"] = 1;
    }
    if (output_tensor.get_dtype() == DataType::INT32) {
        writer_defines["OUTPUT_DTYPE_INT32"] = 1;
    }
    if (output_tensor.get_dtype() == DataType::FLOAT32) {
        writer_defines["OUTPUT_DTYPE_FLOAT32"] = 1;
    }
    bool dst_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    auto kernel_id = tt::operations::primary::CreateWriteKernel(
        program,
        operation_attributes.untilize_out
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange_rm.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    // Set RuntimeArgs
    uint32_t core_h = grid.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }
        vector<uint32_t> writer_args = {
            output_tensor.buffer()->address(),
            tile_offset,
            num_tiles_per_core,
            *reinterpret_cast<uint32_t*>(&start),
            *reinterpret_cast<uint32_t*>(&step),
            output_tensor.element_size()};
        SetRuntimeArgs(program, kernel_id, core, writer_args);
        tile_offset += num_tiles_per_core;
    }
    return {std::move(program), {kernel_id, num_cores, core_h}};
}

void MorehArangeOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& kernel_id = cached_program.shared_variables.kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;
    auto src_dram_buffer = output_tensor.buffer();

    for (uint32_t icore = 0; icore < num_cores; icore++) {
        CoreCoord core = {icore / core_h, icore % core_h};
        auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
        runtime_args[0] = src_dram_buffer->address();
    }
}
}  // namespace ttnn::operations::moreh::moreh_arange
