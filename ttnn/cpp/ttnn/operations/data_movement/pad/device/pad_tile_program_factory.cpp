// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/host_api.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ttnn::device_operation::ProgramArtifacts PadTileCoreProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};

    const auto& output_shape = output_padded_shape;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    const uint32_t num_input_tiles = 2;
    const uint32_t num_pad_tiles = 1;  // src1 / "pad": scratch L1 for the pad-value tile

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

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_tile";

    // "in0": input tiles, produced by the (tilized) reader and consumed by the writer.
    // "pad" (legacy src1 CB): output-dtype L1 scratch the writer fills with the pad value once
    // and reuses as the NoC source for every padded tile; it has no FIFO peer (the writer is its
    // only user), so it is bound as a self-loop (producer + consumer on the writer).
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad"},
            .entry_size = single_tile_size,
            .num_entries = num_pad_tiles,
            .data_format_metadata = cb_data_format},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"out"}, .spec = output.tensor_spec()},
    };

    // Tilized reader (eltwise/unary reader, forked to *_m2 for the named-binding form).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_unary_pad_dims_interleaved.cpp"},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             // "pad" self-loop: the writer both reserves (producer) and consumes (consumer) the
             // pad-scratch DFB; it has no other endpoint.
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"}},
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
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};

    // Local DFBs (in0, pad) require their producer and consumer KernelSpecs to share the same
    // WorkUnitSpec. Both kernels run on the single core {0,0}, so a single WorkUnitSpec hosts both.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "single_core",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = core_ranges},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    reader_run.runtime_arg_values.push_back({CoreCoord{0, 0}, {{"num_pages", num_unpadded_tiles}, {"start_id", 0u}}});

    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};
    writer_run.runtime_arg_values.push_back(
        {CoreCoord{0, 0},
         {{"num_unpadded_W", num_unpadded_W},
          {"num_padded_Wt", num_padded_Wt},
          {"num_unpadded_Z", num_unpadded_Z},
          {"num_padded_Zt", num_padded_Zt},
          {"num_unpadded_Yt", num_unpadded_Yt},
          {"num_padded_Yt", num_padded_Yt},
          {"num_unpadded_Xt", num_unpadded_Xt},
          {"num_padded_Xt", num_padded_Xt},
          {"pad_value", packed_pad_value}}});

    run.kernel_run_args = {reader_run, writer_run};
    const auto& src_mesh = a.mesh_tensor();
    const auto& out_mesh = output.mesh_tensor();
    run.tensor_args = {
        {m2::TensorParamName{"src"}, src_mesh},
        {m2::TensorParamName{"out"}, out_mesh},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
