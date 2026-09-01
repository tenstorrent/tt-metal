// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_width_only_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName SH_W_READER{"reader"};
const KernelSpecName SH_W_WRITER{"writer"};
const DFBSpecName SH_W_IN_SHARD{"in_shard"};
const DFBSpecName SH_W_OUT_SHARD{"out_shard"};
const DFBSpecName SH_W_PAD{"pad"};
const TensorParamName SH_W_INPUT{"input"};
const TensorParamName SH_W_OUTPUT{"output"};
}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmShardedWidthOnlyProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
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

    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat pad_val_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // Input shard DFB — borrows the input buffer's L1 memory; the framework re-points it from the
    // input TensorArgument on every dispatch. The reader only takes its base pointer (a raw peek,
    // no FIFO ops), so the reader is its sole toucher and binds both endpoints (self-loop).
    // The entry count is clamped to the sticks the tensor actually holds: an input whose shard
    // spec is taller than the whole tensor would otherwise fail spec validation against the
    // borrowed tensor's packed size. The count is inert on device — the reader never runs FIFO
    // ops on this DFB — so the clamp only affects validation.
    const uint32_t num_input_sticks = static_cast<uint32_t>(input_tensor.logical_shape().volume() / W);
    DataflowBufferSpec in_shard_dfb{
        .unique_id = SH_W_IN_SHARD,
        .entry_size = unpadded_stick_bytes,
        .num_entries = std::min(shard_height_unpadded, num_input_sticks),
        .data_format_metadata = input_dfb_data_format,
        .borrowed_from = SH_W_INPUT,
    };

    // Output shard DFB — borrows the output buffer's L1 memory. A real FIFO here: the writer
    // produces padded sticks, the reader consumes them to overwrite the data region.
    DataflowBufferSpec out_shard_dfb{
        .unique_id = SH_W_OUT_SHARD,
        .entry_size = padded_stick_bytes,
        .num_entries = shard_height_padded,
        .data_format_metadata = output_dfb_data_format,
        .borrowed_from = SH_W_OUTPUT,
    };

    // Const buffer holding one stick of the pad value. Writer-only, no FIFO ops — self-loop.
    DataflowBufferSpec pad_dfb{
        .unique_id = SH_W_PAD,
        .entry_size = padded_stick_bytes,
        .num_entries = 1,
        .data_format_metadata = pad_val_dfb_data_format,
    };

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

    KernelSpec reader{
        .unique_id = SH_W_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "reader_pad_dims_rm_sharded_stickwise.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SH_W_IN_SHARD,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_W_IN_SHARD,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_W_OUT_SHARD,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"unpadded_stick_bytes", unpadded_stick_bytes},
                {"unpadded_shard_height", shard_height_unpadded},
                {"W_front_pad_bytes", W_padding_front_bytes},
                {"unpadded_stick_step", unpadded_stick_step},
                {"padded_stick_step", padded_stick_step},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = SH_W_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "writer_pad_dims_rm_sharded_stickwise.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SH_W_OUT_SHARD,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_W_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_W_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"padded_stick_bytes", padded_stick_bytes},
                {"padded_shard_height", shard_height_padded},
                {"padding_value_as_u32", padding_value_as_u32},
                {"padding_value_num_bytes", static_cast<uint32_t>(output.element_size())},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Neither sharded kernel takes runtime args: every per-core value is pinned by the hashed
    // shapes and shard specs, and the two shard base addresses ride the borrowed DFBs.

    ProgramSpec spec{
        .name = "pad_rm_sharded_width_only",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in_shard_dfb), std::move(out_shard_dfb), std::move(pad_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SH_W_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = SH_W_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {SH_W_READER, SH_W_WRITER},
                    .target_nodes = all_cores_padded,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {SH_W_INPUT, TensorArgument{input_mesh_tensor}},
        {SH_W_OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
