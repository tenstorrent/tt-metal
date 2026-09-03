// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2i_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ConcatS2IProgramFactory::create_descriptor(
    const ConcatParams& /*operation_attributes*/, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    ProgramDescriptor desc;

    const uint32_t num_output_rows = output.padded_shape()[-1];
    const uint32_t num_input_tensors = input_tensors.size();
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const CoreRangeSet all_cores = input_tensors[0].shard_spec().value().grid;

    const uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
    // input CBs
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        constexpr uint32_t input_num_units_per_shard_width = 1;
        const ShardSpec& shard_spec = input_tensors[input_id].shard_spec().value();
        const uint32_t num_input_units = shard_spec.shape[0] * input_num_units_per_shard_width;
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
    }

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {num_input_tensors};
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {num_input_tensors, input_unit_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2i_width.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_s2i_width.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    const bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    const auto input_cores = input_tensors[0].shard_spec().value().grid;
    const uint32_t num_output_rows_per_core = tt::div_up(num_output_rows, input_cores.num_cores());

    uint32_t core_id = 0;
    for (const CoreCoord& core : cores) {
        const ShardSpec& input_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t curr_num_output_rows = (input_cores.contains(core)) ? num_output_rows_per_core : 0;

        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(num_input_tensors * 2);
        KernelDescriptor::RTArgList writer_runtime_args;
        writer_runtime_args.reserve(5 + num_input_tensors);
        writer_runtime_args.push_back(output.buffer());
        writer_runtime_args.push_back(core_id);
        writer_runtime_args.push_back(curr_num_output_rows);
        writer_runtime_args.push_back(num_input_tensors * input_shard_spec.shape[0]);
        writer_runtime_args.push_back(input_shard_spec.shape[0]);
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
        }
        reader_desc.runtime_args.emplace_back(core, std::move(reader_runtime_args));
        writer_desc.emplace_runtime_args(core, writer_runtime_args);
        core_id++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
