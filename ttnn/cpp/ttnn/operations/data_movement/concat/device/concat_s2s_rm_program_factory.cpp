// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_rm_program_factory.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace ttnn::prim {

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

tt::tt_metal::ProgramDescriptor ConcatS2SRMProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t groups = static_cast<uint32_t>(operation_attributes.groups);
    ProgramDescriptor desc;

    const uint32_t num_output_rows = output.padded_shape()[-2];
    const uint32_t num_input_tensors = input_tensors.size();
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const CoreRangeSet all_cores = input_tensors[0].shard_spec().value().grid;

    std::vector<uint32_t> cb_ids(num_input_tensors);
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        constexpr uint32_t num_input_num_units_per_shard_width = 1;
        const ShardSpec shard_spec = input_tensors[input_id].shard_spec().value();
        const uint32_t num_input_num_units_per_shard_height = shard_spec.shape[0];
        const uint32_t num_input_units = num_input_num_units_per_shard_height * num_input_num_units_per_shard_width;
        const uint32_t input_unit_size = shard_spec.shape[1] * input_tensors[input_id].element_size();
        const uint32_t input_page_size = round_up_to_mul32(input_unit_size);

        desc.cbs.push_back(CBDescriptor{
            .total_size = num_input_units * input_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_id),
                .data_format = cb_data_format,
                .page_size = input_page_size,
            }}},
            .buffer = input_tensors[input_id].buffer(),
        });
        cb_ids[input_id] = input_id;
    }

    constexpr uint32_t cb_dst_id = 16;
    const uint32_t num_output_units = output.shard_spec().value().shape[0];
    const uint32_t output_unit_size = output.shard_spec().value().shape[1] * output.element_size();
    const uint32_t output_page_size = round_up_to_mul32(output_unit_size);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_units * output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_dst_id),
            .data_format = cb_data_format,
            .page_size = output_page_size,
        }}},
        .buffer = output.buffer(),
    });

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

    KernelDescriptor::CompileTimeArgs compile_time_args_0 = {
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
        groups,
        cb_ids[0],
        cb_ids[1]};
    KernelDescriptor::CompileTimeArgs compile_time_args_1 = {
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
        groups,
        cb_ids[0],
        cb_ids[1]};

    KernelDescriptor::CompileTimeArgs compile_time_args_0_last = {
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
        groups,
        cb_ids[0],
        cb_ids[1]};
    KernelDescriptor::CompileTimeArgs compile_time_args_1_last = {
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
        groups,
        cb_ids[0],
        cb_ids[1]};

    auto append_reader_writer_pair = [&](const CoreRangeSet& core_ranges,
                                         const KernelDescriptor::CompileTimeArgs& reader_cta,
                                         const KernelDescriptor::CompileTimeArgs& writer_cta) {
        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = core_ranges;
        reader_desc.compile_time_args = reader_cta;
        reader_desc.config = ReaderConfigDescriptor{};
        desc.kernels.push_back(std::move(reader_desc));

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = core_ranges;
        writer_desc.compile_time_args = writer_cta;
        writer_desc.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(writer_desc));
    };

    if (num_output_rows_per_core_last > 0) {
        const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
        const auto [first, last] = split(cores, cores.size() - 1);
        const CoreRangeSet first_cores = cores_to_corerangeset(first);
        const CoreRangeSet last_cores = cores_to_corerangeset(last);
        append_reader_writer_pair(first_cores, compile_time_args_0, compile_time_args_1);
        append_reader_writer_pair(last_cores, compile_time_args_0_last, compile_time_args_1_last);
    } else {
        append_reader_writer_pair(all_cores, compile_time_args_0, compile_time_args_1);
    }

    return desc;
}

}  // namespace ttnn::prim
