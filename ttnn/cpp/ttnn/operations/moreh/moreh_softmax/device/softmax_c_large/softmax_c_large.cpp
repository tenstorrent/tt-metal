// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

MorehSoftmaxOperation::MorehSoftmaxCLargeFactory::cached_program_t
MorehSoftmaxOperation::MorehSoftmaxCLargeFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    log_info(tt::LogTest, "Large tensor algorithm selected");
    const auto& input = tensor_args.input;
    const auto dim = operation_attributes.dim;
    const auto op = operation_attributes.op;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input.padded_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    uint32_t num_tiles = input.physical_volume() / shape[dim] / H / W * Ht * Wt;

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_tiles);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 2},                         // input
            {tt::CBIndex::c_16, 2},                        // output
            {tt::CBIndex::c_24, 1, intermed_data_format},  // exp(x)
            {tt::CBIndex::c_25, 1, intermed_data_format},  // recips
            {tt::CBIndex::c_26, 2, intermed_data_format},  // add
            {tt::CBIndex::c_27, 1},                        // max
            {tt::CBIndex::c_28, 1, intermed_data_format},  // tmp
        });

    // create read/wrtie kernel
    bool src_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_c_large.cpp",
        all_cores,
        {src_is_dram},
        reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_c_large.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    auto outer_stride = Ht * Wt;
    for (int i = dim; i < shape.rank() - 2; i++) {
        outer_stride *= shape[i];
    }
    auto dim_size = shape[dim];
    auto inner_size = outer_stride / dim_size;

    std::map<std::string, std::string> compute_defines;
    if (op == MorehSoftmaxOp::SOFTMAX || op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["SOFTMAX"] = "1";
    } else {
        compute_defines["SOFTMIN"] = "1";
    }
    if (op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["LOG"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, dim_size}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, dim_size}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            input.buffer()->address(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size};

        std::vector<uint32_t> writer_args = {
            output.buffer()->address(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

void MorehSoftmaxOperation::MorehSoftmaxCLargeFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.input.buffer()->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_softmax
