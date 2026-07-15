// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_width_only_program_factory.hpp"

#include <filesystem>
#include <vector>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ttnn::device_operation::ProgramArtifacts PadRmShardedWidthOnlyProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_INPUT_SHARD{"cb_input_shard"};    // legacy c_0: input shard (borrowed; reader local view)
    const DFBSpecName CB_OUTPUT_SHARD{"cb_output_shard"};  // legacy c_16: output shard (borrowed; real DFB)
    const DFBSpecName CB_PAD{"cb_pad"};                    // legacy c_1: pad-value scratchpad (writer self-loop)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "reader_pad_dims_rm_sharded_stickwise.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "writer_pad_dims_rm_sharded_stickwise.cpp";

    const auto& input_tensor = tensor_args.input;
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

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

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
    uint32_t padded_stick_step = tt::round_up(padded_stick_bytes, l1_alignment_bytes);
    uint32_t unpadded_stick_step = tt::round_up(unpadded_stick_bytes, l1_alignment_bytes);

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs.
    //   cb_input_shard  (borrowed input):  fake CB — reader reads the resident input shard by base
    //                                       pointer (local tensor view); bound as a self-loop.
    //   cb_output_shard (borrowed output): a real DFB — the writer produces padded sticks, the reader
    //                                       consumes them and patches in the real data after the front pad.
    //   cb_pad          (fresh L1):         writer self-loop pad-value scratchpad.
    // (cb_input_shard / cb_pad self-loops are validator-satisfying fake-CB workarounds; see report.)
    // ------------------------------------------------------------------------
    DataflowBufferSpec cb_input_spec{
        .unique_id = CB_INPUT_SHARD,
        .entry_size = static_cast<uint32_t>(unpadded_stick_bytes),
        .num_entries = shard_height_unpadded,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    };
    DataflowBufferSpec cb_output_spec{
        .unique_id = CB_OUTPUT_SHARD,
        .entry_size = static_cast<uint32_t>(padded_stick_bytes),
        .num_entries = shard_height_padded,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    };
    DataflowBufferSpec cb_pad_spec{
        .unique_id = CB_PAD,
        .entry_size = static_cast<uint32_t>(padded_stick_bytes),
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------------
    // Kernels. Both consume only compile-time args (no per-core runtime args).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings =
            {ProducerOf(CB_INPUT_SHARD, "cb_input_shard"),
             ConsumerOf(CB_INPUT_SHARD, "cb_input_shard"),
             ConsumerOf(CB_OUTPUT_SHARD, "cb_output_shard")},
        .compile_time_args =
            {{"unpadded_stick_bytes", static_cast<uint32_t>(unpadded_stick_bytes)},
             {"unpadded_shard_height", shard_height_unpadded},
             {"W_front_pad_bytes", W_padding_front_bytes},
             {"unpadded_stick_step", unpadded_stick_step},
             {"padded_stick_step", padded_stick_step}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings =
            {ProducerOf(CB_OUTPUT_SHARD, "cb_output_shard"),
             ProducerOf(CB_PAD, "cb_padding_value"),
             ConsumerOf(CB_PAD, "cb_padding_value")},
        .compile_time_args =
            {{"padded_stick_bytes", static_cast<uint32_t>(padded_stick_bytes)},
             {"padded_shard_height", shard_height_padded},
             {"padding_value_as_u32", padding_value_as_u32},
             {"padding_value_num_bytes", static_cast<uint32_t>(output.element_size())}},
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
    };

    // Both kernels take no named runtime args (data flows via CBs/compile-time args), so
    // runtime_arg_values stays empty; the kernels still run on their nodes per the work unit's
    // target_nodes.
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    WorkUnitSpec wu{
        .name = "pad_rm_sharded_width_only",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = all_cores_padded,
    };

    ProgramSpec spec{
        .name = "pad_rm_sharded_width_only",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_input_spec, cb_output_spec, cb_pad_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
