// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RotateHalfDeviceOperation::RotateHalfProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& input,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::constants;

    const CoreCoord core({0, 0});
    const CoreRangeSet core_range(CoreRange(core, core));

    Tensor& output = tensor_return_value;

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    tt::DataFormat scalar_dfb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_dfb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;
    const uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    const uint32_t half_row_size = input.padded_shape()[-1] / TILE_WIDTH / 2;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    // Used for half of tensor that is multiplied
    const DFBSpecName SRC_MUL{"src_mul"};
    // Used for bcast scalar
    const DFBSpecName SRC_SCALAR{"src_scalar"};
    // Used for half of tensor that is not multiplied: the reader fills it and the writer drains it directly
    const DFBSpecName NO_MUL{"no_mul"};
    const DFBSpecName OUTPUT_MUL{"output_mul"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const uint32_t num_input_tiles = 2;
    DataflowBufferSpec src_mul_dfb{
        .unique_id = SRC_MUL,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    const uint32_t num_scalar_tiles = 1;
    DataflowBufferSpec src_scalar_dfb{
        .unique_id = SRC_SCALAR,
        .entry_size = scalar_single_tile_size,
        .num_entries = num_scalar_tiles,
        .data_format_metadata = dfb_data_format,
    };

    DataflowBufferSpec no_mul_dfb{
        .unique_id = NO_MUL,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    const uint32_t num_output_tiles = 2;
    DataflowBufferSpec output_mul_dfb{
        .unique_id = OUTPUT_MUL,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dfb_data_format,
    };

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
            "reader_rotate_half_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = NO_MUL,
                    .accessor_name = "in_no_mul",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SRC_MUL,
                    .accessor_name = "in_mul",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SRC_SCALAR,
                    .accessor_name = "in_scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"scalar_value", static_cast<uint32_t>(bfloat16_scalar)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "half_row_size", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
            "writer_rotate_half_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = NO_MUL,
                    .accessor_name = "out_no_mul",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_MUL,
                    .accessor_name = "out_mul",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "half_row_width", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Scalar-broadcast multiply (by -1) of the multiplied half. The fork's BCAST_OP / BCAST_LLKOP / BCAST_DIM
    // vocabulary for Mul + HW comes from the bcast op's own define table; BCAST_SCALAR makes it wait on the
    // single scalar tile once and never pop it.
    KernelSpec::CompilerOptions::Defines compute_defines(bcast_op_utils::get_defines(BcastOpDim::HW, BcastOpMath::MUL));
    compute_defines["BCAST_SCALAR"] = "1";

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw_metal2.cpp",
        // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC_MUL,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SRC_SCALAR,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_MUL,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"B", "Ht", "Wt"}},
        // The legacy compute descriptor left every field at its default; so does this one.
        .hw_config = ComputeHardwareConfig{},
    };

    KernelRunArgs reader_run_args{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core, {{"num_rows", num_rows}, {"half_row_size", half_row_size}, {"start_id", 0u}}),
    };
    KernelRunArgs writer_run_args{
        .kernel = WRITER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core, {{"num_rows", num_rows}, {"half_row_width", half_row_size}, {"start_id", 0u}}),
    };
    KernelRunArgs compute_run_args{
        .kernel = COMPUTE,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core,
            {
                {"B", 1u},
                {"Ht", 1u},
                {"Wt", num_tiles / 2},
            }),
    };

    ProgramSpec spec{
        .name = "rotate_half",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {src_mul_dfb, src_scalar_dfb, no_mul_dfb, output_mul_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "rotate_half",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = core_range,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
