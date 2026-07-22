// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_width_only_program_factory.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ProgramDescriptor PadRmShardedWidthOnlyProgramFactory::create_descriptor(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    TT_ASSERT(
        output.shard_spec().has_value() and output.shard_spec()->shape[1] == output_padded_shape[-1],
        "ttnn.pad: pad_rm_sharded_width_only expects sharded output parameter with shard width equal to the width of "
        "the requested output tensor. Ensure pad_impl is calling this program factory correctly.");

    uint32_t W = input_tensor.logical_shape()[-1];
    uint32_t W_padded = output_padded_shape[3];

    auto unpadded_stick_bytes = W * input_tensor.element_size();
    auto padded_stick_bytes = W_padded * input_tensor.element_size();

    IDevice* device = input_tensor.device();

    // input shard spec
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height_unpadded = input_shard_spec.shape[0];

    // output shard spec
    auto shard_spec_padded = output.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    const auto& ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores_padded = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    Buffer* input_buffer = input_tensor.buffer();
    Buffer* output_buffer = output.buffer();

    ProgramDescriptor desc;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_shard_cb_index = tt::CBIndex::c_0;
    {
        // Sharded input CB — globally allocated to the input buffer; framework
        // patches the CB address on cache hits via cb.buffer.
        CBDescriptor cb_input;
        cb_input.total_size = shard_height_unpadded * unpadded_stick_bytes;
        cb_input.core_ranges = total_cores;
        cb_input.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_shard_cb_index),
            .data_format = input_cb_data_format,
            .page_size = unpadded_stick_bytes,
        });
        cb_input.buffer = input_buffer;
        desc.cbs.push_back(std::move(cb_input));
    }

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_shard_cb_index = tt::CBIndex::c_16;
    {
        // Sharded output CB — globally allocated to the output buffer.
        CBDescriptor cb_output;
        cb_output.total_size = shard_height_padded * padded_stick_bytes;
        cb_output.core_ranges = total_cores;
        cb_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_shard_cb_index),
            .data_format = output_cb_data_format,
            .page_size = padded_stick_bytes,
        });
        cb_output.buffer = output_buffer;
        desc.cbs.push_back(std::move(cb_output));
    }

    // construct const buffer with the pad_value
    tt::DataFormat pad_val_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t pad_val_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = padded_stick_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(pad_val_cb_index),
            .data_format = pad_val_cb_data_format,
            .page_size = padded_stick_bytes,
        }}},
    });

    // W front-pad offset: input_tensor_start is [N, C, H, W];
    uint32_t W_padding_front_bytes = input_tensor_start[3] * input_tensor.element_size();

    uint32_t padding_value_as_u32;
    if (input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        uint16_t bfloat_pad_value_bits = std::bit_cast<uint16_t>(bfloat16(pad_value));
        padding_value_as_u32 = *reinterpret_cast<uint32_t*>(&bfloat_pad_value_bits);
    } else if (input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32) {
        padding_value_as_u32 = *reinterpret_cast<const uint32_t*>(&pad_value);
    } else if (input_tensor.dtype() == tt::tt_metal::DataType::UINT16) {
        padding_value_as_u32 = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else if (
        input_tensor.dtype() == tt::tt_metal::DataType::INT32 ||
        input_tensor.dtype() == tt::tt_metal::DataType::UINT32) {
        padding_value_as_u32 = static_cast<uint32_t>(pad_value);  // for INT32 and UINT32
    } else {
        TT_THROW("ttnn.pad: unsupported data type for pad_rm_sharded_stickwise");
    }

    auto l1_alignment_bytes = hal::get_l1_alignment();
    uint32_t padded_stick_step = tt::round_up(
        padded_stick_bytes, l1_alignment_bytes);  // round padded_stick bytes to a multiple of l1_alignment_bytes
    uint32_t unpadded_stick_step = tt::round_up(
        unpadded_stick_bytes,
        l1_alignment_bytes);  // round unpadded_stick bytes to a multiple of l1_alignment_bytes

    std::vector<uint32_t> reader_ct_args = {
        unpadded_stick_bytes,
        padded_stick_bytes,
        shard_height_unpadded,
        shard_height_padded,
        W_padding_front_bytes,
        input_shard_cb_index,
        output_shard_cb_index,
        unpadded_stick_step,
        padded_stick_step};

    std::vector<uint32_t> writer_ct_args = {
        padded_stick_bytes,
        shard_height_padded,
        padding_value_as_u32,
        output.element_size(),
        output_shard_cb_index,
        pad_val_cb_index,
        padded_stick_step};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded_stickwise.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_padded;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded_stickwise.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores_padded;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Sharded readers/writers don't consume per-core runtime args (legacy code
    // called SetRuntimeArgs(..., {}) with empty arg lists).  CB addresses are
    // patched via cb.buffer on cache hits.  Mirror the legacy behavior by
    // emitting an empty rtarg list per active core so the kernel reserves slots.
    for (const auto& core : ordered_cores_with_data) {
        reader_desc.emplace_runtime_args(core, KernelDescriptor::RTArgList{});
        writer_desc.emplace_runtime_args(core, KernelDescriptor::RTArgList{});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
