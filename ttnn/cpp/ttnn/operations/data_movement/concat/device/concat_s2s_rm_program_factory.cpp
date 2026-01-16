// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_rm_program_factory.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::operations::data_movement::concat::program {

template <typename T>
static std::pair<std::vector<T>, std::vector<T>> split(std::vector<T> input, std::size_t index) {
    if (index > input.size()) {
        throw std::out_of_range{"split index out of range"};
    }
    std::vector<T> second{std::make_move_iterator(input.begin() + index), std::make_move_iterator(input.end())};
    input.erase(input.begin() + index, input.end());
    return {std::move(input), std::move(second)};
}

static CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    std::transform(cores.begin(), cores.end(), std::back_inserter(core_ranges), [](const CoreCoord& core) {
        return CoreRange(core);
    });
    return CoreRangeSet(core_ranges);
}

ConcatS2SRMProgramFactory::cached_program_t ConcatS2SRMProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;
    const unsigned int groups = operation_attributes.groups;
    TT_FATAL(dim == 3, "Sharded concat RM only supports dim=3");
    TT_FATAL(groups == 1 || dim == 3, "Sharded concat RM only supports groups > 1 when dim=3");

    TT_FATAL(
        input_tensors.size() == 2 && input_tensors[0].padded_shape()[-1] % groups == 0 &&
            input_tensors[1].padded_shape()[-1] % groups == 0,
        "Input channels must both be evenly divisible by groups");

    Program program = CreateProgram();

    const uint32_t num_output_rows = output.padded_shape()[-2];
    const uint32_t num_input_tensors = input_tensors.size();

    std::vector<CBHandle> cb_input(num_input_tensors);

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const CoreRangeSet all_cores = input_tensors[0].shard_spec().value().grid;

    std::vector<uint32_t> cb_ids(num_input_tensors);

    // input CBs
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        constexpr uint32_t num_input_num_units_per_shard_width = 1;
        const ShardSpec shard_spec = input_tensors[input_id].shard_spec().value();
        const uint32_t num_input_num_units_per_shard_height = shard_spec.shape[0];
        const uint32_t num_input_units = num_input_num_units_per_shard_height * num_input_num_units_per_shard_width;
        const uint32_t input_unit_size = shard_spec.shape[1] * input_tensors[input_id].element_size();
        const uint32_t input_page_size = round_up_to_mul32(input_unit_size);
        CircularBufferConfig input_cb_config =
            CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_input[input_id] = CreateCircularBuffer(program, all_cores, input_cb_config);
        cb_ids[input_id] = input_id;
    }

    // output CB
    constexpr uint32_t cb_dst_id = 16;
    const uint32_t num_output_units = output.shard_spec().value().shape[0];
    const uint32_t output_unit_size = output.shard_spec().value().shape[1] * output.element_size();
    const uint32_t output_page_size = round_up_to_mul32(output_unit_size);
    CircularBufferConfig output_cb_config =
        CircularBufferConfig(num_output_units * output_page_size, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    CBHandle cb_output = CreateCircularBuffer(program, all_cores, output_cb_config);

    const ShardSpec output_shard_spec = output.shard_spec().value();
    const uint32_t output_stick_size = output_shard_spec.shape[1] * output.element_size();

    const ShardSpec input_0_shard_spec = input_tensors[0].shard_spec().value();
    const ShardSpec input_1_shard_spec = input_tensors[1].shard_spec().value();
    const uint32_t input_0_stick_size = input_0_shard_spec.shape[1] * input_tensors[0].element_size();
    const uint32_t input_1_stick_size = input_1_shard_spec.shape[1] * input_tensors[1].element_size();
    const uint32_t input_0_stride = output_stick_size - input_0_stick_size;
    const uint32_t input_1_stride = output_stick_size - input_1_stick_size;
    const uint32_t num_output_rows_per_core = input_tensors[0].shard_spec().value().shape[0];
    const uint32_t num_pages_per_risc = tt::div_up(num_output_rows_per_core, 2);

    const uint32_t num_output_rows_per_core_last = num_output_rows % num_output_rows_per_core;
    const uint32_t num_pages_per_risc_last = tt::div_up(num_output_rows_per_core_last, 2);

    std::vector<uint32_t> compile_time_args_0 = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core * num_input_tensors,
        0,
        num_pages_per_risc,
        0,
        0,
        0,
        groups};

    std::vector<uint32_t> compile_time_args_1 = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core * num_input_tensors,
        num_pages_per_risc,
        num_output_rows_per_core,
        num_pages_per_risc * output_stick_size,
        num_pages_per_risc * input_0_stick_size,
        num_pages_per_risc * input_1_stick_size,
        groups};

    std::vector<uint32_t> compile_time_args_0_last = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core_last * num_input_tensors,
        0,
        num_pages_per_risc_last,
        0,
        0,
        0,
        groups};

    std::vector<uint32_t> compile_time_args_1_last = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core_last * num_input_tensors,
        num_pages_per_risc_last,
        num_output_rows_per_core_last,
        num_pages_per_risc_last * output_stick_size,
        num_pages_per_risc_last * input_0_stick_size,
        num_pages_per_risc_last * input_1_stick_size,
        groups};

    if (num_output_rows_per_core_last > 0) {
        const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
        const auto [first, last] = split(cores, cores.size() - 1);
        const CoreRangeSet first_cores = cores_to_corerangeset(first);
        const CoreRangeSet last_cores = cores_to_corerangeset(last);
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            first_cores,
            ReaderDataMovementConfig(compile_time_args_0));
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            first_cores,
            WriterDataMovementConfig(compile_time_args_1));

        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            last_cores,
            ReaderDataMovementConfig(compile_time_args_0_last));
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            last_cores,
            WriterDataMovementConfig(compile_time_args_1_last));
    } else {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            all_cores,
            ReaderDataMovementConfig(compile_time_args_0));
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            all_cores,
            WriterDataMovementConfig(compile_time_args_1));
    }

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = cb_input,
         .cb_output = cb_output,
         .all_cores = all_cores}};
}

void ConcatS2SRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    for (uint32_t input_id = 0; input_id < shared_vars.num_input_tensors; input_id++) {
        UpdateDynamicCircularBufferAddress(
            program, shared_vars.cb_inputs[input_id], *tensor_args.input_tensors[input_id].buffer());
    }
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *tensor_return_value.buffer());
}

}  // namespace ttnn::operations::data_movement::concat::program
