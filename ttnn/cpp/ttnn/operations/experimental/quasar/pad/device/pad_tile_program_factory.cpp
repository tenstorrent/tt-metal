// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
// Metal 2.0 named resource ids for this factory.
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const DFBSpecName SRC0{"src0"};  // c_0: reader produces, writer consumes (real FIFO)
const DFBSpecName PAD{"pad"};    // c_1: writer-only scratch (fake CB -> self-loop)
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
const std::string WU{"wu"};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts PadTileCoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids below
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const auto& output_shape = output_padded_shape;

    const MeshTensor& input_mesh_tensor = a.mesh_tensor();
    const MeshTensor& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    const NodeCoord core{0, 0};

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

    // -------- DataflowBuffers (was: CBDescriptors) --------
    const uint32_t num_input_tiles = 2;
    const uint32_t num_pad_tiles = 1;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = SRC0,
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = PAD,
            .entry_size = single_tile_size,
            .num_entries = num_pad_tiles,
            .data_format_metadata = cb_data_format,
        },
    };

    // -------- TensorParameters (was: TensorAccessorArgs + buffer-address RTAs) --------
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
    };

    // -------- Reader KernelSpec (reads input tiles into SRC0) --------
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0,
                    .accessor_name = "src0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // -------- Writer KernelSpec (consumes SRC0, fills PAD scratch, writes output) --------
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
                                        "writer_unary_pad_dims_interleaved.cpp"),
        // PAD is a fake/scratch CB the writer only fills and reads by base pointer.
        // Self-loop it (PRODUCER + CONSUMER on the writer) so the validator's
        // producer-and-consumer rule is satisfied; see METAL2_PORT_REPORT.md.
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0,
                    .accessor_name = "src0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args = {{"pad_value", packed_pad_value}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "num_padded_Wt",
                  "num_unpadded_Z",
                  "num_padded_Zt",
                  "num_unpadded_Yt",
                  "num_padded_Yt",
                  "num_unpadded_Xt",
                  "num_padded_Xt"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ProgramSpec spec{
        .name = "pad_tile_single_core",
        .kernels = {reader, writer},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{.name = WU, .kernels = {READER, WRITER}, .target_nodes = core}},
    };

    // -------- ProgramRunArgs --------
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {{.node = core, .args = {{"num_pages", num_unpadded_tiles}, {"start_id", 0u}}}},
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{.node = core,
                  .args =
                      {{"num_unpadded_W", num_unpadded_W},
                       {"num_padded_Wt", num_padded_Wt},
                       {"num_unpadded_Z", num_unpadded_Z},
                       {"num_padded_Zt", num_padded_Zt},
                       {"num_unpadded_Yt", num_unpadded_Yt},
                       {"num_padded_Yt", num_padded_Yt},
                       {"num_unpadded_Xt", num_unpadded_Xt},
                       {"num_padded_Xt", num_padded_Xt}}}},
        },
    };
    run_args.tensor_args.emplace(INPUT, input_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
