// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ttnn::device_operation::ProgramArtifacts PadTileCoreProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: input stream (reader produces, writer consumes)
    const DFBSpecName CB_PAD{"cb_pad"};  // legacy c_1: pad-value scratchpad (writer self-loop)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
        "reader_unary_interleaved_start_id_metal2.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp";

    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& output_shape = output_padded_shape;

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs (legacy CBs c_0 / c_1, both normal/non-borrowed).
    // ------------------------------------------------------------------------
    const uint32_t num_input_tiles = 2;
    const uint32_t num_pad_tiles = 1;
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec cb_pad_spec{
        .unique_id = CB_PAD,
        .entry_size = single_tile_size,
        .num_entries = num_pad_tiles,
        .data_format_metadata = cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case-1 page access on both kernels).
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------------
    // Pad value packed exactly as the legacy factory did (preserve dtype handling).
    // ------------------------------------------------------------------------
    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    const uint32_t num_unpadded_Xt = a.padded_shape()[3] / TILE_WIDTH;
    const uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    const uint32_t num_unpadded_Yt = a.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    const uint32_t num_unpadded_Z = a.padded_shape()[1];
    const uint32_t num_total_Z = output_shape[1];
    const uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    const uint32_t num_unpadded_W = a.padded_shape()[0];
    const uint32_t num_total_W = output_shape[0];
    const uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    const uint32_t num_unpadded_tiles = a.physical_volume() / TILE_HW;

    // ------------------------------------------------------------------------
    // Reader: streams the unpadded input tiles linearly into cb_in0 (c_0).
    // (Buffer-address RTA dropped; the input is reached via its TensorBinding.)
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ------------------------------------------------------------------------
    // Writer: consumes cb_in0 (c_0), fills/streams the pad-value scratchpad cb_pad (c_1,
    // PRODUCER+CONSUMER self-loop), and writes the padded output page-by-page (Case-1).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = CB_IN0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = CB_PAD, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CB_PAD, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "num_padded_Wt",
                  "num_unpadded_Z",
                  "num_padded_Zt",
                  "num_unpadded_Yt",
                  "num_padded_Yt",
                  "num_unpadded_Xt",
                  "num_padded_Xt",
                  "pad_value"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ------------------------------------------------------------------------
    // Single core (0,0): one work unit, one instance per kernel.
    // ------------------------------------------------------------------------
    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};
    const NodeCoord node{0, 0};

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    reader_run.runtime_arg_values.push_back({node, {{"num_pages", num_unpadded_tiles}, {"start_id", uint32_t{0}}}});

    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    writer_run.runtime_arg_values.push_back(
        {node,
         {{"num_unpadded_W", num_unpadded_W},
          {"num_padded_Wt", num_padded_Wt},
          {"num_unpadded_Z", num_unpadded_Z},
          {"num_padded_Zt", num_padded_Zt},
          {"num_unpadded_Yt", num_unpadded_Yt},
          {"num_padded_Yt", num_padded_Yt},
          {"num_unpadded_Xt", num_unpadded_Xt},
          {"num_padded_Xt", num_padded_Xt},
          {"pad_value", packed_pad_value}}});

    WorkUnitSpec wu{
        .name = "pad_tile_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = core_ranges,
    };

    ProgramSpec spec{
        .name = "pad_tile_single_core",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec, cb_pad_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(a.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
