// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_width_only_program_factory.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ttnn::device_operation::ProgramArtifacts PadRmShardedWidthOnlyProgramFactory::create_program_spec(
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

    // input shard spec
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height_unpadded = input_shard_spec.shape[0];

    // output shard spec
    auto shard_spec_padded = output.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    const auto& ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores_padded = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    TT_ASSERT(input_tensor.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat pad_val_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

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

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_rm_sharded_width_only";

    // Borrowed-memory DFBs: in0 views the input shard's L1, out0 views the output shard's L1
    // (supplied at runtime via the input/output tensor args).  pad is a fake CB the writer fills
    // with the pad value and reads by base pointer (self-loop).
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = unpadded_stick_bytes,
            .num_entries = shard_height_unpadded,
            .data_format_metadata = input_cb_data_format,
            .borrowed_from = m2::TensorParamName{"src"}},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out0"},
            .entry_size = padded_stick_bytes,
            .num_entries = shard_height_padded,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = m2::TensorParamName{"dst"}},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad"},
            .entry_size = padded_stick_bytes,
            .num_entries = 1,
            .data_format_metadata = pad_val_cb_data_format},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_dims_rm_sharded_stickwise.cpp"},
        // in0 read by base pointer only (self-loop); out0 consumed (sticks the writer produced).
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out0"},
                 .accessor_name = "out0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"unpadded_stick_bytes", unpadded_stick_bytes},
             {"padded_stick_bytes", padded_stick_bytes},
             {"unpadded_shard_height", shard_height_unpadded},
             {"padded_shard_height", shard_height_padded},
             {"W_front_pad_bytes", W_padding_front_bytes},
             {"unpadded_stick_step", unpadded_stick_step},
             {"padded_stick_step", padded_stick_step}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_dims_rm_sharded_stickwise.cpp"},
        // out0 produced by the writer; pad is a writer self-loop.
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out0"},
                 .accessor_name = "out0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"padded_stick_bytes", padded_stick_bytes},
             {"padded_shard_height", shard_height_padded},
             {"padding_value_as_u32", padding_value_as_u32},
             {"padding_value_num_bytes", static_cast<uint32_t>(output.element_size())}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};

    // out0 (borrowed) is produced by the writer and consumed by the reader; in0/pad self-loops.
    // All endpoints share a WorkUnitSpec — both kernels run on all_cores_padded.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "sharded",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores_padded},
    };

    // ---- ProgramRunArgs (mutable) ----
    // The sharded readers/writers consume only CTAs; no per-core runtime args (legacy emitted empty
    // arg lists).  The borrowed DFBs pull their backing L1 address from the src/dst tensor args.
    m2::ProgramRunArgs run;
    run.tensor_args = {
        {m2::TensorParamName{"src"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
