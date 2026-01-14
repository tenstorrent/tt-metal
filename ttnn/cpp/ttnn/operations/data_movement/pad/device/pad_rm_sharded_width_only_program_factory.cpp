// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_width_only_program_factory.hpp"

#include <tt-metalium/hal.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::data_movement::pad::program {
PadRmShardedWidthOnlyProgramFactory::cached_program_t PadRmShardedWidthOnlyProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;
    Program program{};

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

    auto& all_cores_padded = shard_spec_padded.grid;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_shard_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_shard_cb_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_unpadded * unpadded_stick_bytes, {{input_shard_cb_index, input_cb_data_format}})
            .set_page_size(input_shard_cb_index, unpadded_stick_bytes)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto input_shard_cb = tt::tt_metal::CreateCircularBuffer(program, total_cores, input_shard_cb_config);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_shard_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig output_shard_cb_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_padded * padded_stick_bytes, {{output_shard_cb_index, output_cb_data_format}})
            .set_page_size(output_shard_cb_index, padded_stick_bytes)
            .set_globally_allocated_address(*output.buffer());
    auto output_shard_cb = tt::tt_metal::CreateCircularBuffer(program, total_cores, output_shard_cb_config);

    // construct const buffer with the pad_value
    tt::DataFormat pad_val_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t pad_val_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_pad_val_config =
        tt::tt_metal::CircularBufferConfig(padded_stick_bytes, {{pad_val_cb_index, pad_val_cb_data_format}})
            .set_page_size(pad_val_cb_index, padded_stick_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_pad_val_config);

    uint32_t W_padding_front_bytes = input_tensor_start[-3] * input_tensor.element_size();

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

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded_stickwise.cpp",
        all_cores_padded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded_stickwise.cpp",
        all_cores_padded,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores_padded, {});
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores_padded, {});

    return cached_program_t{std::move(program), {input_shard_cb, output_shard_cb}};
}

void PadRmShardedWidthOnlyProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.input_shard_cb, *input_buffer);
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.output_shard_cb, *output_buffer);
}

}  // namespace ttnn::operations::data_movement::pad::program
