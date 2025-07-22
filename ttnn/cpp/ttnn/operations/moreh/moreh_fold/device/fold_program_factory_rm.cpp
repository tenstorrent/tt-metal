// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "fold_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_fold {

MorehFoldOperation::ProgramFactory::cached_program_t MorehFoldOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& input = tensor_args.input;

    auto output_size = operation_attributes.output_size;
    auto kernel_size = operation_attributes.kernel_size;
    auto dilation = operation_attributes.dilation;
    auto padding = operation_attributes.padding;
    auto stride = operation_attributes.stride;
    auto output_shape = output.logical_shape();
    auto output_shape_rank = output.logical_shape().rank();

    std::vector<uint32_t> ls;
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t l = (((output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1);
        ls.push_back(l);
    }
    uint32_t N = output_shape_rank == 4 ? output_shape[0] : 1;
    uint32_t C = output_shape_rank == 4 ? output_shape[1] : output_shape[0];
    uint32_t H = output_shape_rank == 4 ? output_shape[2] : output_shape[1];
    uint32_t W = output_shape_rank == 4 ? output_shape[3] : output_shape[2];
    uint32_t kernel_size_h = kernel_size[0];
    uint32_t kernel_size_w = kernel_size[1];
    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t padding_h = padding[0];
    uint32_t padding_w = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];
    uint32_t LH = ls[0];
    uint32_t LW = ls[1];

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};
    IDevice* device = input.device();

    uint32_t num_units = output.logical_volume() / output.logical_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_units);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    uint32_t unit_size = input.element_size();
    uint32_t input_cb_page_size = unit_size * input.logical_shape()[-1];
    uint32_t output_cb_page_size = unit_size * output.logical_shape()[-1];

    uint32_t aligned_input_cb_page_size = round_up_to_mul32(input_cb_page_size);
    uint32_t aligned_output_cb_page_size = round_up_to_mul32(output_cb_page_size);

    uint32_t input_cb_index = tt::CBIndex::c_0;    // input
    uint32_t output_cb_index = tt::CBIndex::c_16;  // ouput

    CircularBufferConfig input_cb_config =
        CircularBufferConfig(aligned_input_cb_page_size * 2, {{input_cb_index, data_format}})
            .set_page_size(input_cb_index, aligned_input_cb_page_size);
    auto input_cb = CreateCircularBuffer(program, all_cores, input_cb_config);

    CircularBufferConfig output_cb_config =
        CircularBufferConfig(aligned_output_cb_page_size * 2, {{output_cb_index, data_format}})
            .set_page_size(output_cb_index, aligned_output_cb_page_size);
    auto output_cb = CreateCircularBuffer(program, all_cores, output_cb_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    switch (input.dtype()) {
        case DataType::BFLOAT16: reader_defines["DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: reader_defines["DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    bool input_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool output_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(input_is_dram),
        static_cast<uint32_t>(input_cb_index),
        static_cast<uint32_t>(output_cb_index),
    };

    std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(output_is_dram),
        static_cast<uint32_t>(output_cb_index),
    };

    const auto reader_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/reader_fold_rm.cpp";
    const auto writer_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/writer_fold_rm.cpp";

    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    uint32_t start_id = 0;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    uint32_t g1_numcores = core_group_1.num_cores();
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        std::vector<uint32_t> reader_args = {
            input.buffer()->address(),
            N,
            C,
            H,
            W,
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            LH,
            LW,
            aligned_input_cb_page_size,
            aligned_output_cb_page_size,
            start_id,
            num_units_per_core};

        std::vector<uint32_t> writer_args = {
            output.buffer()->address(), aligned_output_cb_page_size, start_id, num_units_per_core};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);
        start_id += num_units_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void MorehFoldOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;
    auto input_buffer_address = tensor_args.input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();

    for (const auto& core : cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = input_buffer_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer_address;
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_fold
