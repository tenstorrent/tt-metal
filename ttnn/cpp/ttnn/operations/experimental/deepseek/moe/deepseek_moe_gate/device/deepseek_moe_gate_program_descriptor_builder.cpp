// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_gate_program_descriptor_builder.hpp"

#include <bit>

#include <tt_stl/assert.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

namespace {

constexpr const char* kDeepseekMoeGateKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/kernels/deepseek_moe_gate_kernel.cpp";

// The one unified kernel source, built once per RISC.
const tt::tt_metal::experimental::KernelSpecName kReaderKernel{"reader"};
const tt::tt_metal::experimental::KernelSpecName kWriterKernel{"writer"};
const tt::tt_metal::experimental::KernelSpecName kComputeKernel{"compute"};

// Tensor-backed dataflow buffers, one per io tensor.
const tt::tt_metal::experimental::DFBSpecName kInputDfb{"input"};
const tt::tt_metal::experimental::DFBSpecName kBiasDfb{"bias"};
const tt::tt_metal::experimental::DFBSpecName kOutputDfb{"output"};
const tt::tt_metal::experimental::DFBSpecName kInputIndicesDfb{"input_indices"};
const tt::tt_metal::experimental::DFBSpecName kOutputIndicesDfb{"output_indices"};

uint32_t float_bits_u32(float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32 bits to pack into a compile-time arg");
    return std::bit_cast<uint32_t>(value);
}

// A dataflow buffer on the tensor's own sharded L1 memory: the whole per-core shard, split into tile-sized
// entries (or a single entry when the tile size does not divide the shard).
tt::tt_metal::experimental::DataflowBufferSpec make_tensor_backed_dfb(
    const tt::tt_metal::experimental::DFBSpecName& dfb_name,
    const tt::tt_metal::experimental::TensorParamName& tensor_name,
    const ttnn::Tensor& tensor) {
    const auto& spec = tensor.tensor_spec();
    const auto& tile = spec.tile();
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(spec.data_type());
    uint32_t tile_size = tile.get_tile_size(data_format);
    uint32_t total_size = tensor.buffer()->aligned_size_per_bank();
    uint32_t entry_size = (total_size % tile_size == 0) ? tile_size : total_size;
    return tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_name,
        .entry_size = entry_size,
        .num_entries = total_size / entry_size,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile,
        .borrowed_from = tensor_name,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts build_moe_gate_program_artifacts(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs) {
    using namespace tt::tt_metal::experimental;
    using tt::tt_metal::CoreRangeSet;
    using tt::tt_metal::DataType;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& input_indices_tensor = tensor_args.input_indices_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& output_indices_tensor = tensor_args.output_indices_tensor;

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input_tensor must be BFLOAT16");
    TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "bias_tensor must be BFLOAT16");
    TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16, "output_tensor must be BFLOAT16");

    TT_FATAL(input_tensor.is_sharded(), "input_tensor must be sharded");
    TT_FATAL(bias_tensor.is_sharded(), "bias_tensor must be sharded");
    TT_FATAL(input_indices_tensor.is_sharded(), "input_indices_tensor must be sharded");
    TT_FATAL(output_tensor.is_sharded(), "output_tensor must be sharded");
    TT_FATAL(output_indices_tensor.is_sharded(), "output_indices_tensor must be sharded");

    const auto& input_shard = input_tensor.shard_spec().value();
    const auto& bias_shard = bias_tensor.memory_config().shard_spec().value();
    const auto& in_indices_shard = input_indices_tensor.memory_config().shard_spec().value();
    const auto& output_shard = output_tensor.shard_spec().value();
    const auto& out_indices_shard = output_indices_tensor.memory_config().shard_spec().value();

    CoreRangeSet all_cores = input_shard.grid;

    TT_FATAL(input_shard.shape == bias_shard.shape, "Input and bias shard shapes must match");
    TT_FATAL(input_shard.orientation == bias_shard.orientation, "Input and bias shard orientations must match");
    TT_FATAL(bias_shard.grid.contains(all_cores), "Bias shard grid must contain input shard grid");

    TT_FATAL(input_shard.shape == in_indices_shard.shape, "Input and input-indices shard shapes must match");
    TT_FATAL(
        input_shard.orientation == in_indices_shard.orientation,
        "Input and input-indices shard orientations must match");
    TT_FATAL(in_indices_shard.grid.contains(all_cores), "Input-indices shard grid must contain input shard grid");

    TT_FATAL(output_shard.grid == out_indices_shard.grid, "Output and output-indices shard grids must match");
    TT_FATAL(output_shard.grid.contains(all_cores), "Output shard grid must contain input compute grid");

    const auto& in_tile = input_tensor.tensor_spec().tile();
    const auto& out_tile = output_tensor.tensor_spec().tile();
    TT_FATAL(in_tile == bias_tensor.tensor_spec().tile(), "Input and bias tiles must match");
    TT_FATAL(in_tile == input_indices_tensor.tensor_spec().tile(), "Input and input-indices tiles must match");
    TT_FATAL(out_tile == output_indices_tensor.tensor_spec().tile(), "Output tiles must match");

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& bias_mesh_tensor = bias_tensor.mesh_tensor();
    const auto& input_indices_mesh_tensor = input_indices_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();
    const auto& output_indices_mesh_tensor = output_indices_tensor.mesh_tensor();

    ProgramSpec spec{.name = "deepseek_moe_gate"};

    spec.tensor_parameters = {
        TensorParameter{.unique_id = kInputTensor, .spec = input_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = kBiasTensor, .spec = bias_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = kOutputTensor, .spec = output_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = kInputIndicesTensor, .spec = input_indices_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = kOutputIndicesTensor, .spec = output_indices_mesh_tensor.tensor_spec()},
    };

    // Declaration order is load-bearing: a dataflow buffer takes the lowest device slot not used by the
    // buffers declared before it, so this order puts input, bias, output, input_indices and output_indices on
    // slots 0-4. The compute kernel's hardware startup (deepseek_compute_kernel_init) configures unpack and
    // pack from slot 0 directly, which must be the input buffer.
    spec.dataflow_buffers = {
        make_tensor_backed_dfb(kInputDfb, kInputTensor, input_tensor),
        make_tensor_backed_dfb(kBiasDfb, kBiasTensor, bias_tensor),
        make_tensor_backed_dfb(kOutputDfb, kOutputTensor, output_tensor),
        make_tensor_backed_dfb(kInputIndicesDfb, kInputIndicesTensor, input_indices_tensor),
        make_tensor_backed_dfb(kOutputIndicesDfb, kOutputIndicesTensor, output_indices_tensor),
    };

    // NCRISC: makes the resident input, bias and input-index shards available to compute.
    spec.kernels.push_back(KernelSpec{
        .unique_id = kReaderKernel,
        .source = kDeepseekMoeGateKernelPath,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kInputDfb,
                    .accessor_name = "moe_gate_input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = kBiasDfb,
                    .accessor_name = "moe_gate_bias",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = kInputIndicesDfb,
                    .accessor_name = "moe_gate_input_indices",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = {{"moe_gate_is_active_core", 1}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // BRISC: waits for the scores and indices compute packs into the resident output shards.
    spec.kernels.push_back(KernelSpec{
        .unique_id = kWriterKernel,
        .source = kDeepseekMoeGateKernelPath,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kOutputDfb,
                    .accessor_name = "moe_gate_output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = kOutputIndicesDfb,
                    .accessor_name = "moe_gate_output_indices",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args = {{"moe_gate_is_active_core", 1}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    // TRISC: the gate itself.
    spec.kernels.push_back(KernelSpec{
        .unique_id = kComputeKernel,
        .source = kDeepseekMoeGateKernelPath,
        // A compute kernel built with no explicit level ran at O3 before Metal 2.0; a KernelSpec defaults to O2,
        // so the level is stated here.
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kInputDfb,
                    .accessor_name = "moe_gate_input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = kBiasDfb,
                    .accessor_name = "moe_gate_bias",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = kInputIndicesDfb,
                    .accessor_name = "moe_gate_input_indices",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = kOutputDfb,
                    .accessor_name = "moe_gate_output",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = kOutputIndicesDfb,
                    .accessor_name = "moe_gate_output_indices",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"moe_gate_eps", float_bits_u32(operation_attrs.eps)},
                {"moe_gate_scaling_factor", float_bits_u32(operation_attrs.scaling_factor)},
                {"moe_gate_enable_sigmoid", operation_attrs.enable_sigmoid ? 1u : 0u},
                {"moe_gate_is_active_core", 1},
            },
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = tt::tt_metal::Precision::Precise,
                .enable_32_bit_dest = false,
                .double_buffer_dest = true,
            },
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {kReaderKernel, kWriterKernel, kComputeKernel},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(kInputTensor, input_mesh_tensor);
    run_args.tensor_args.emplace(kBiasTensor, bias_mesh_tensor);
    run_args.tensor_args.emplace(kOutputTensor, output_mesh_tensor);
    run_args.tensor_args.emplace(kInputIndicesTensor, input_indices_mesh_tensor);
    run_args.tensor_args.emplace(kOutputIndicesTensor, output_indices_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate
