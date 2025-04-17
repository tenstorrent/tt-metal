// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bits/stdint-uintn.h>
#include <math.h>
#include <cstdint>

#include "deinterleave_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::experimental::deinterleave {

DeinterleaveLocalOperation::ProgramFactoryLocal::cached_program_t
DeinterleaveLocalOperation::ProgramFactoryLocal::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    // TT_FATAL(operation_attributes.to_batch == false, "Deinterleave: bad configuration.");
    Program program;

    const auto& input = tensor_args.input;

    tt::log_warning(tt::LogOp, "Input buffer address {:#x}", input.buffer()->address());
    uint32_t i = 0;
    for (const auto& output : outputs) {
        tt::log_warning(tt::LogOp, "Output buffer[{}] address {:#x}", i, output->buffer()->address());
        i++;
    }

    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tensor.get_logical_shape()[-1] * tensor.element_size();
    };

    // uint32_t num_units = input.volume() / input.get_logical_shape()[-1];

    tt::tt_metal::CoreRangeSet worker_grid = input.memory_config().shard_spec.value().grid;
    // auto num_units_per_core = input.memory_config().shard_spec.value().shape[0];

    // uint32_t src_cb_id = CBIndex::c_0;
    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    uint32_t src_total_size = input.get_logical_shape()[0] * aligned_input_unit_size;

    // tt::tt_metal::CircularBufferConfig src_cb_config =
    //     tt::tt_metal::CircularBufferConfig(src_total_size, {{src_cb_id, input_data_format}})
    //         .set_page_size(src_cb_id, aligned_input_unit_size)
    //         .set_globally_allocated_address(*input.buffer());
    // auto src_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, src_cb_config);

    // uint32_t dst_cb_id = CBIndex::c_1;
    // auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    // uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    // uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
    // uint32_t dst_total_size = output.get_logical_shape()[0] * aligned_output_unit_size;

    // tt::tt_metal::CircularBufferConfig dst_cb_config =
    //     tt::tt_metal::CircularBufferConfig(dst_total_size, {{dst_cb_id, output_data_format}})
    //         .set_page_size(dst_cb_id, aligned_output_unit_size)
    //         .set_globally_allocated_address(*output.buffer());
    // auto dst_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, dst_cb_config);

    std::vector<uint32_t> dm0_compile_time_args, dm1_compile_time_args;

    // TT_FATAL(input_unit_size == output_unit_size, "Deinterleave: input and output unit size must be equal");

    auto per_core_width = operation_attributes.input_width;
    auto per_core_height = input.memory_config().shard_spec.value().shape[0] / operation_attributes.input_width;
    log_info(
        tt::LogOp,
        "DeinterleaveLocalOperation::ProgramFactoryLocal::create; stride_hw: {}; per core height {} per_core_width {}",
        operation_attributes.stride_hw,
        per_core_height,
        per_core_width);
    auto stick_size_bytes = aligned_input_unit_size;
    dm0_compile_time_args = {
        (uint32_t)input.buffer()->address(),
        (uint32_t)per_core_width,
        (uint32_t)per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
    };

    dm1_compile_time_args = {
        (uint32_t)input.buffer()->address() + per_core_width * stick_size_bytes,
        (uint32_t)per_core_width,
        (uint32_t)per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
    };

    auto dm0_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_local_kernel_rm.cpp",
        worker_grid,
        ReaderDataMovementConfig(dm0_compile_time_args, {}));

    auto dm1_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_local_kernel_rm.cpp",
        worker_grid,
        WriterDataMovementConfig(dm1_compile_time_args, {}));

    // 2x2 => 4/2 = 2 args each
    // 4x4 => 16/2 = 8 args each
    // 8x8 => 64/2 = 32 args each
    std::vector<uint32_t> dm0_runtime_args, dm1_runtime_args;
    for (uint32_t h = 0; h < operation_attributes.stride_hw[0]; h++) {
        for (uint32_t w = 0; w < operation_attributes.stride_hw[1]; w++) {
            auto out_idx = h * operation_attributes.stride_hw[1] + w;
            if (h % 2 == 0) {
                // dm0, processes even lines
                dm0_runtime_args.push_back(outputs[out_idx]->buffer()->address());
            } else {
                // dm1, processes odd lines
                dm1_runtime_args.push_back(outputs[out_idx]->buffer()->address());
            }
        }
    }

    // std::reverse(dm0_runtime_args.begin(), dm0_runtime_args.end());
    // std::reverse(dm1_runtime_args.begin(), dm1_runtime_args.end());

    SetCommonRuntimeArgs(program, dm0_kernel_id, dm0_runtime_args);

    SetCommonRuntimeArgs(program, dm1_kernel_id, dm1_runtime_args);

    TT_FATAL(worker_grid.size() == 1, "Deinterleave: shard spec CoreRangeSet must have single range");
    tt::tt_metal::CoreCoord device_grid =
        worker_grid.bounding_box().grid_size();  // input.device()->logical_grid_size();

    uint32_t num_of_shards = worker_grid.num_cores();
    auto cores = corerange_to_cores(worker_grid, std::nullopt, true);

    uint32_t out_batches = operation_attributes.stride_hw[0] * operation_attributes.stride_hw[1];
    // assuming a single core reads only one stick type from src from the interleaved data, thus fail if cores_in_batch
    // > num_of_shards
    TT_FATAL(
        out_batches <= num_of_shards, "Deinterleave: out_batches {} > num_of_shards {}", out_batches, num_of_shards);

    return {std::move(program), {dm0_kernel_id, dm1_kernel_id, worker_grid}};
}

void DeinterleaveLocalOperation::ProgramFactoryLocal::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& read_kernel_id = cached_program.shared_variables.read_kernel_id;
    const auto& write_kernel_id = cached_program.shared_variables.write_kernel_id;

    // auto input_buffer_address = tensor_args.input.buffer()->address();
    // auto output_buffer_address = output.buffer()->address();

    TT_FATAL(false, "to resolve overriding runtime args");
    // std::vector<std::vector<uint32_t>>& reader_args = GetRuntimeArgs(program, read_kernel_id);
    // reader_args[0] = input_buffer_address;
    // std::vector<std::vector<uint32_t>>& writer_args = GetRuntimeArgs(program, write_kernel_id);
    // writer_args[0] = output_buffer_address;
}
}  // namespace ttnn::operations::experimental::deinterleave
