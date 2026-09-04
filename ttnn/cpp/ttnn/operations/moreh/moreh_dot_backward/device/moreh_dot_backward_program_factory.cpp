// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "moreh_dot_backward_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static const std::filesystem::path READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/reader_moreh_dot_backward.cpp";
static const std::filesystem::path WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/writer_moreh_dot_backward.cpp";
static const std::filesystem::path COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp";

ttnn::device_operation::ProgramArtifacts MorehDotBackwardOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const MeshTensor& output_grad = tensor_args.output_grad.mesh_tensor();
    const MeshTensor& input = tensor_args.input.mesh_tensor();
    const MeshTensor& other = tensor_args.other.mesh_tensor();

    const auto& input_grad = tensor_return_value.at(0);
    const auto& other_grad = tensor_return_value.at(1);
    const bool has_input_grad = input_grad.has_value();
    const bool has_other_grad = other_grad.has_value();

    const NodeCoord node = {0, 0};

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    // Each DFB mirrors the legacy CB: double-buffered (num_entries = 2), one tile per entry.
    const uint32_t num_entries = 2;

    // ---- Resource names ----
    const TensorParamName OUTPUT_GRAD{"output_grad"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OTHER{"other"};
    const TensorParamName INPUT_GRAD{"input_grad"};
    const TensorParamName OTHER_GRAD{"other_grad"};

    const DFBSpecName IN0{"in0"};    // legacy c_0 (output_grad scalar)
    const DFBSpecName IN1{"in1"};    // legacy c_1 (input)
    const DFBSpecName IN2{"in2"};    // legacy c_2 (other)
    const DFBSpecName OUT0{"out0"};  // legacy c_16 (input_grad)
    const DFBSpecName OUT1{"out1"};  // legacy c_17 (other_grad)

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- Dataflow buffers (all compute-bound → data_format_metadata required) ----
    auto make_dfb = [&](const DFBSpecName& id) {
        return DataflowBufferSpec{
            .unique_id = id,
            .entry_size = cb_tile_size,
            .num_entries = num_entries,
            .data_format_metadata = cb_data_format,
        };
    };

    // ---- Tensor parameters ----
    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_GRAD, .spec = output_grad.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OTHER, .spec = other.tensor_spec()});
    if (has_input_grad) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = INPUT_GRAD, .spec = input_grad.value().mesh_tensor().tensor_spec()});
    }
    if (has_other_grad) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = OTHER_GRAD, .spec = other_grad.value().mesh_tensor().tensor_spec()});
    }

    // ---- Reader kernel ----
    KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_GRAD, .accessor_name = "s0"},
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "s1"},
                TensorBinding{.tensor_parameter_name = OTHER, .accessor_name = "s2"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"has_input_grad", "has_other_grad", "num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(output_grad.device().arch()),
    };

    // ---- Writer kernel ----
    // input_grad / other_grad are optional outputs: bind conditionally, and move the
    // selecting condition from a runtime arg to a compile-time define (HAS_INPUT_GRAD /
    // HAS_OTHER_GRAD). The writer #ifdef-gates the tensor::s0 / tensor::s1 accessors and
    // their write blocks on those defines.
    Group<TensorBinding> writer_tensor_bindings;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (has_input_grad) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT_GRAD, .accessor_name = "s0"});
        writer_defines.emplace("HAS_INPUT_GRAD", "1");
    }
    if (has_other_grad) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OTHER_GRAD, .accessor_name = "s1"});
        writer_defines.emplace("HAS_OTHER_GRAD", "1");
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .compiler_options = {.defines = std::move(writer_defines)},
        // OUT0 / OUT1 are program-local DFBs, bound 1P+1C unconditionally; whether they
        // are exercised is gated at runtime (compute) / at compile time (writer, via the
        // HAS_*_GRAD defines above).
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT1, .accessor_name = "out1", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = std::move(writer_tensor_bindings),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(output_grad.device().arch()),
    };

    // ---- Compute kernel ----
    // Legacy ComputeConfigDescriptor{} (all defaults) → ComputeGen1Config{} (defaults match).
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = COMPUTE_KERNEL_PATH,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = OUT1, .accessor_name = "out1", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"has_input_grad", "has_other_grad", "per_core_block_cnt"}},
        .hw_config = ComputeHardwareConfig{ComputeGen1Config{}},
    };

    // ---- Assemble the spec ----
    ProgramSpec spec;
    spec.name = "moreh_dot_backward";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = {make_dfb(IN0), make_dfb(IN1), make_dfb(IN2), make_dfb(OUT0), make_dfb(OUT1)};
    spec.tensor_parameters = std::move(tensor_parameters);
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}};

    // ---- Run args ----
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"has_input_grad", static_cast<uint32_t>(has_input_grad)},
                 {"has_other_grad", static_cast<uint32_t>(has_other_grad)},
                 {"num_tiles", num_tiles},
                 {"start_id", 0u}}),
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}, {"start_id", 0u}}),
        },
        KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"has_input_grad", static_cast<uint32_t>(has_input_grad)},
                 {"has_other_grad", static_cast<uint32_t>(has_other_grad)},
                 {"per_core_block_cnt", num_tiles}}),
        },
    };

    run_args.tensor_args.insert({OUTPUT_GRAD, output_grad});
    run_args.tensor_args.insert({INPUT, input});
    run_args.tensor_args.insert({OTHER, other});
    if (has_input_grad) {
        run_args.tensor_args.insert({INPUT_GRAD, input_grad.value().mesh_tensor()});
    }
    if (has_other_grad) {
        run_args.tensor_args.insert({OTHER_GRAD, other_grad.value().mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_dot_backward
