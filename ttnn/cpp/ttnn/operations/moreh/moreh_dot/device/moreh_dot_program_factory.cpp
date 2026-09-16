// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_dot {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/reader_moreh_dot.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/writer_moreh_dot.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp";

ttnn::device_operation::ProgramArtifacts MorehDotOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    const auto& input_a_mesh = input_a.mesh_tensor();
    const auto& input_b_mesh = input_b.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_a.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    uint32_t num_tiles = input_a.physical_volume() / tt::constants::TILE_HW;
    const auto& a_shape_wo_padding = input_a.logical_shape();
    uint32_t pad_h = a_shape_wo_padding[2] % tt::constants::TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % tt::constants::TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (tt::constants::TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (tt::constants::TILE_WIDTH) : (pad_w);

    IDevice* device = input_a.device();

    const uint32_t in0_t = 2;   // a
    const uint32_t in1_t = 2;   // b
    const uint32_t in2_t = 1;   // scaler
    const uint32_t out0_t = 2;  // out
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    const NodeCoord node = {0, 0};

    // ----- Resource names -----
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0{"in0"};        // legacy c_0
    const DFBSpecName IN1{"in1"};        // legacy c_1
    const DFBSpecName SCALER{"scaler"};  // legacy c_2
    const DFBSpecName OUT{"out"};        // legacy c_16
    const DFBSpecName IM0{"im0"};        // legacy c_24
    const DFBSpecName IM1{"im1"};        // legacy c_25

    const TensorParamName INPUT_A{"input_a"};
    const TensorParamName INPUT_B{"input_b"};
    const TensorParamName OUTPUT{"output"};

    // ----- Dataflow buffers (1:1 with legacy CBs; all bound to compute → data format required) -----
    DataflowBufferSpec dfb_in0{
        .unique_id = IN0, .entry_size = cb_tile_size, .num_entries = in0_t, .data_format_metadata = cb_data_format};
    DataflowBufferSpec dfb_in1{
        .unique_id = IN1, .entry_size = cb_tile_size, .num_entries = in1_t, .data_format_metadata = cb_data_format};
    DataflowBufferSpec dfb_scaler{
        .unique_id = SCALER, .entry_size = cb_tile_size, .num_entries = in2_t, .data_format_metadata = cb_data_format};
    DataflowBufferSpec dfb_out{
        .unique_id = OUT, .entry_size = cb_tile_size, .num_entries = out0_t, .data_format_metadata = cb_data_format};
    DataflowBufferSpec dfb_im0{
        .unique_id = IM0, .entry_size = cb_tile_size, .num_entries = im0_t, .data_format_metadata = cb_data_format};
    DataflowBufferSpec dfb_im1{
        .unique_id = IM1, .entry_size = cb_tile_size, .num_entries = im1_t, .data_format_metadata = cb_data_format};

    // ----- Reader kernel -----
    KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_A, .accessor_name = "src0"},
                TensorBinding{.tensor_parameter_name = INPUT_B, .accessor_name = "src1"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id", "mask_h", "mask_w"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };

    // ----- Writer kernel -----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    };

    // ----- Compute kernel -----
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = COMPUTE_KERNEL_PATH,
        .compiler_options =
            {.defines = {{"REDUCE_OP", "PoolType::SUM"}, {"REDUCE_DIM", "ReduceDim::REDUCE_ROW"}},
             .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                // im0 / im1 are compute-internal intermediates (self-loop: produced and consumed here).
                DFBBinding{.dfb_spec_name = IM0, .accessor_name = "im0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IM0, .accessor_name = "im0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IM1, .accessor_name = "im1", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IM1, .accessor_name = "im1", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
        // Style A: op resolves a TTNN ComputeKernelConfig; translate it to the Gen1 hardware config.
        .hw_config = to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
    };

    // ----- ProgramSpec -----
    ProgramSpec spec{
        .name = "moreh_dot",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {dfb_in0, dfb_in1, dfb_scaler, dfb_out, dfb_im0, dfb_im1},
        .tensor_parameters =
            {
                {.unique_id = INPUT_A, .spec = input_a_mesh.tensor_spec()},
                {.unique_id = INPUT_B, .spec = input_b_mesh.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output_mesh.tensor_spec()},
            },
        .work_units =
            {
                {.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node},
            },
    };

    // ----- ProgramRunArgs -----
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        {.kernel = READER,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
             node, {{"num_tiles", num_tiles}, {"start_id", 0u}, {"mask_h", mask_h}, {"mask_w", mask_w}})},
        {.kernel = WRITER,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"num_tiles", 1u}, {"start_id", 0u}})},
        {.kernel = COMPUTE,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"per_core_block_cnt", num_tiles}})},
    };
    run_args.tensor_args = {
        {INPUT_A, TensorArgument{input_a_mesh}},
        {INPUT_B, TensorArgument{input_b_mesh}},
        {OUTPUT, TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::moreh::moreh_dot
