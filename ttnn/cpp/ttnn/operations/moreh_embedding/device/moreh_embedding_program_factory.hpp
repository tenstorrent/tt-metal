// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_log.h"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

using namespace tt;

namespace ttnn::operations::moreh_embedding::detail {

operation::ProgramWithCallbacks moreh_embeddings_copy(
    const Tensor &input, const Tensor &weight, Tensor &output, const CoreRange core_range) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *input_buffer = input.buffer();
    tt_metal::Buffer *weight_buffer = weight.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weight_is_dram = weight.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t embedding_dim = weight.get_legacy_shape().without_padding()[-1];

    uint32_t H = output.get_legacy_shape().without_padding()[-3];
    uint32_t W = output.get_legacy_shape().without_padding()[-2];
    uint32_t units_to_divide = output.volume() / (output.get_legacy_shape()[-2] * output.get_legacy_shape()[-1]) *
                               div_up(W, TILE_HEIGHT) * div_up(embedding_dim, TILE_HEIGHT);

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(weight.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1, tt::DataFormat::Int32},  // input
            {CB::c_in1, 1},                         // weight
            {CB::c_out0, 1},                        // output
        });

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(weight)),
    };

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh_embedding/device/kernels/dataflow/reader_moreh_embeddings.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);

    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh_embedding/device/kernels/dataflow/writer_moreh_embeddings.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto input_addr = input.buffer()->address();
    const auto weight_addr = weight.buffer()->address();
    const auto output_addr = output.buffer()->address();

    // Set Runtime Args
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        vector<uint32_t> reader_args = {input_addr, weight_addr, units_per_core, tile_offset, H, W, embedding_dim};

        vector<uint32_t> writer_args = {
            output_addr,
            units_per_core,
            tile_offset,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += units_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, core_h](
                                                const void* operation,
                                                Program& program,
                                                const std::vector<Tensor>& input_tensors,
                                                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                const std::vector<Tensor>& output_tensors
                                              ) {

        const auto& input = input_tensors.at(0);
        const auto& weight = input_tensors.at(1);
        const auto& output = output_tensors.at(0);

        auto input_dram_buffer = input.buffer();
        auto weight_dram_buffer = weight.buffer();
        auto output_dram_buffer = output.buffer();

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_dram_buffer->address();
                runtime_args[1] = weight_dram_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_dram_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks moreh_embeddings_(
    const Tensor &intput,
    const Tensor &weight,
    const std::optional<float> max_norm,
    const float norm_type,
    Tensor &output,
    const CoreRange core_range,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // do weight norm
    if (max_norm.has_value()) {
        auto max_norm_val = max_norm.value();
    }

    return moreh_embeddings_copy(intput, weight, output, core_range);
}
}  // namespace ttnn::operations::moreh_embedding::detail
