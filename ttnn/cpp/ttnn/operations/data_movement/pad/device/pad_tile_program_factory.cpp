// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName TILE_SC_READER{"reader"};
const KernelSpecName TILE_SC_WRITER{"writer"};
const DFBSpecName TILE_SC_IN0{"in0"};
const DFBSpecName TILE_SC_PAD{"pad"};
const TensorParamName TILE_SC_INPUT{"input"};
const TensorParamName TILE_SC_OUTPUT{"output"};
}  // namespace

ttnn::device_operation::ProgramArtifacts PadTileCoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const NodeCoord node{0, 0};

    // This should allocate a DRAM buffer on the device

    const auto& output_shape = output_padded_shape;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "dfb_data_format: {}", dfb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    const uint32_t num_input_tiles = 2;
    DataflowBufferSpec in0_dfb{
        .unique_id = TILE_SC_IN0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    // Pad buffer: the writer reserves an entry and fills it with the pad value, then NoC-writes
    // that entry out repeatedly. Nothing ever pushes or drains it, so the writer is its only
    // toucher and binds both endpoints (self-loop).
    const uint32_t num_pad_tiles = 1;
    DataflowBufferSpec pad_dfb{
        .unique_id = TILE_SC_PAD,
        .entry_size = single_tile_size,
        .num_entries = num_pad_tiles,
        .data_format_metadata = dfb_data_format,
    };

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    uint32_t num_unpadded_Xt = a.padded_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.padded_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.padded_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.padded_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.physical_volume() / TILE_HW;

    // Tilized reader: the Metal 2.0 fork of eltwise/unary's shared reader, which lives beside the
    // legacy original. Its binding names (dfb::in, tensor::src) and named args (num_pages,
    // start_id) are the fork's interface, so this spec conforms to them rather than renaming.
    KernelSpec reader{
        .unique_id = TILE_SC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_SC_IN0,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_SC_INPUT,
                    .accessor_name = "src",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = TILE_SC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "writer_unary_pad_dims_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_SC_IN0,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = TILE_SC_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TILE_SC_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_SC_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_unpadded_W",
                     "num_padded_Wt",
                     "num_unpadded_Z",
                     "num_padded_Zt",
                     "num_unpadded_Yt",
                     "num_padded_Yt",
                     "num_unpadded_Xt",
                     "num_padded_Xt",
                     "pad_value"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ProgramSpec spec{
        .name = "pad_tile_single_core",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in0_dfb), std::move(pad_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = TILE_SC_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = TILE_SC_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {TILE_SC_READER, TILE_SC_WRITER},
                    .target_nodes = node,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = TILE_SC_READER,
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"num_pages", num_unpadded_tiles}, {"start_id", std::uint32_t{0}}}),
        },
        KernelRunArgs{
            .kernel = TILE_SC_WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"num_unpadded_W", num_unpadded_W},
                 {"num_padded_Wt", num_padded_Wt},
                 {"num_unpadded_Z", num_unpadded_Z},
                 {"num_padded_Zt", num_padded_Zt},
                 {"num_unpadded_Yt", num_unpadded_Yt},
                 {"num_padded_Yt", num_padded_Yt},
                 {"num_unpadded_Xt", num_unpadded_Xt},
                 {"num_padded_Xt", num_padded_Xt},
                 {"pad_value", packed_pad_value}}),
        },
    };
    run_args.tensor_args = {
        {TILE_SC_INPUT, TensorArgument{input_mesh_tensor}},
        {TILE_SC_OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
