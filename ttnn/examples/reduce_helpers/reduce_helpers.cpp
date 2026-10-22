// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How a program factory uses the reduce helpers: column_mean_factory() with a single plan and
// row_mean_of_two_inputs_factory() with a sequence plan.
//
// Who does what
//   Factory         Owns the program: cores, tensors, the order in which the reader delivers input tiles,
//                   DFB IDs and the capacities of the input, output and accumulator DFBs.
//   Planner         make_reduce_plan() / make_reduce_sequence_plan(). Decides the algorithm, masking of partial
//                   tiles, scaling and accumulation, and sizes the auxiliary DFB.
//   Reader kernel   Pushes input tiles in the order the factory described to the planner.
//   Writer kernel   Fills the auxiliary DFB once from the planned recipe, then writes the outputs.
//   Compute kernel  Issues reduce<Call>() for each planned call. It makes no reduction decisions.
//
// Steps, marked in both factories
//   1. Factory   Fix the block each reduce call consumes and the order the reader delivers it in.
//   2. Factory   Describe the block and the compute kernel's hardware configuration to the planner.
//   3. Planner   Plan.
//   4. Factory   Declare the DFBs. Only the auxiliary DFB's size and format come from the plan.
//   5. Factory   Serialize the plan: call arguments for compute, the auxiliary recipe for the writer.
//   6. Factory   Declare the kernels, their DFB bindings and the work unit.
//   7. Kernels   kernels/dataflow/reader_*.cpp, kernels/dataflow/writer_auxiliary_and_output.cpp, kernels/compute/*.cpp

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/kernel_lib/host/reduce_host.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace reduce_host = ttnn::kernel_lib::host;
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

const ComputeHardwareConfig compute_hardware{.fpu_math_fidelity = MathFidelity::HiFi4};

DataflowBufferSpec tile_dfb(const char* name, tt::DataFormat format, uint32_t tiles) {
    return {
        .unique_id = DFBSpecName{name},
        .entry_size = tt::tile_size(format),
        .num_entries = tiles,
        .data_format_metadata = format};
}

DataflowBufferSpec auxiliary_dfb(const reduce_host::ReduceAuxiliaryCb& planned) {
    return {
        .unique_id = DFBSpecName{"auxiliary"},
        .entry_size = planned.page_size,
        .num_entries = planned.page_count,
        .data_format_metadata = planned.data_format};
}

KernelSpec::DFBBinding producer(const char* name) {
    return {.dfb_spec_name = DFBSpecName{name}, .accessor_name = name, .endpoint_type = DFBEndpointType::PRODUCER};
}

KernelSpec::DFBBinding consumer(const char* name) {
    return {.dfb_spec_name = DFBSpecName{name}, .accessor_name = name, .endpoint_type = DFBEndpointType::CONSUMER};
}

uint32_t tile_count(const ttnn::Tensor& tensor) {
    return tensor.padded_shape()[-2] / TILE_HEIGHT * (tensor.padded_shape()[-1] / TILE_WIDTH);
}

KernelSpec writer_kernel(uint32_t num_output_tiles, const std::vector<uint32_t>& auxiliary_args) {
    return KernelSpec{
        .unique_id = KernelSpecName{"writer"},
        .source =
            OVERRIDE_KERNEL_PREFIX "ttnn/examples/reduce_helpers/kernels/dataflow/writer_auxiliary_and_output.cpp",
        .dfb_bindings = {producer("auxiliary"), consumer("out")},
        .tensor_bindings = {{.tensor_parameter_name = TensorParamName{"output"}, .accessor_name = "dst"}},
        .compile_time_args = {{"num_output_tiles", num_output_tiles}},
        .hw_config = CreateWriterDataMovementConfig(),
        .advanced_options = {.compile_time_varargs = auxiliary_args},
    };
}

// mean(input, dim=H) for a [H, W] bfloat16 input on one core. H need not be a multiple of 32.
ttnn::device_operation::ProgramArtifacts column_mean_factory(const ttnn::Tensor& input, const ttnn::Tensor& output) {
    // 1. Factory: one call reduces the whole [Ht, Wt] block. The reader delivers strips of output_group columns,
    //    row by row within each strip, so compute holds output_group column results in DEST at once.
    constexpr uint32_t output_group = 4;
    const uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    TT_FATAL(Wt % output_group == 0, "The reader only delivers whole strips");
    const uint32_t input_dfb_tiles = 2 * output_group;

    // 2. Factory: describe the block and the compute kernel's hardware configuration. The planner rejects an
    //    output group larger than max_reduce_output_group() for this hardware configuration.
    auto block = reduce_host::ReduceBlockSpec::tiled(
        input.logical_shape()[-2], input.logical_shape()[-1], input.dtype(), output.dtype());
    block.input_cb_tiles = input_dfb_tiles;
    block.output_group = output_group;
    const auto hardware = reduce_host::make_reduce_hardware_config(input.device()->arch(), compute_hardware);

    // 3. Planner. The scalar is absent, so AVG divides by the logical height.
    const reduce_host::ReducePlan plan = reduce_host::make_reduce_plan(
        block, ReduceOpMath::AVG, ReduceOpDim::H, std::nullopt, ReduceFp32Mode::Fast, hardware);

    ProgramSpec spec{.name = "reduce_helpers_column_mean"};

    // 4. Factory: DFBs. Reduce pushes one output tile at a time.
    const tt::DataFormat input_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_format = datatype_to_dataformat_converter(output.dtype());
    spec.dataflow_buffers = {
        tile_dfb("input", input_format, input_dfb_tiles),
        auxiliary_dfb(plan.auxiliary_cb),
        tile_dfb("out", output_format, 2),
    };

    // 5. Factory: serialize. The plan names DFBs by logical IDs; the compute kernel binds them, in this order,
    //    to dfb::input, dfb::auxiliary and dfb::out.
    enum LogicalDfb : uint32_t { Input, Auxiliary, Output };
    const auto compute_args =
        reduce_host::ReduceCallArgs(plan, {.input_cb_id = Input, .auxiliary_cb_id = Auxiliary, .output_cb_id = Output})
            .get_compile_time_args();
    const auto writer_args = reduce_host::ReduceAuxiliaryArgs(plan, Auxiliary).get_compile_time_args();

    // 6. Factory: kernels and the work unit.
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"input"}, .spec = input.tensor_spec()},
        {.unique_id = TensorParamName{"output"}, .spec = output.tensor_spec()},
    };
    spec.kernels = {
        KernelSpec{
            .unique_id = KernelSpecName{"reader"},
            .source = OVERRIDE_KERNEL_PREFIX "ttnn/examples/reduce_helpers/kernels/dataflow/reader_column_strips.cpp",
            .dfb_bindings = {producer("input")},
            .tensor_bindings = {{.tensor_parameter_name = TensorParamName{"input"}, .accessor_name = "src"}},
            .compile_time_args = {{"Ht", Ht}, {"Wt", Wt}, {"output_group", output_group}},
            .hw_config = CreateReaderDataMovementConfig(),
        },
        writer_kernel(Wt, writer_args),
        KernelSpec{
            .unique_id = KernelSpecName{"compute"},
            .source = OVERRIDE_KERNEL_PREFIX "ttnn/examples/reduce_helpers/kernels/compute/column_mean.cpp",
            .dfb_bindings = {consumer("input"), consumer("auxiliary"), producer("out")},
            .hw_config = compute_hardware,
            .advanced_options = {.compile_time_varargs = compute_args},
        },
    };
    spec.work_units = {WorkUnitSpec{
        .name = "core",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}, KernelSpecName{"compute"}},
        .target_nodes = NodeCoord{0, 0},
    }};

    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(TensorParamName{"input"}, input.mesh_tensor());
    run_args.tensor_args.emplace(TensorParamName{"output"}, output.mesh_tensor());
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

// mean(concat([a, b], dim=W), dim=W) for bfloat16 inputs of equal height on one core, without concatenating:
// one reduce call per input; the first call's partial results wait in an accumulator DFB for the second.
ttnn::device_operation::ProgramArtifacts row_mean_of_two_inputs_factory(
    const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& output) {
    // 1. Factory: one call per input, each over its whole block, a before b. The reader streams a, then b,
    //    row-major. The accumulator holds the partial result of one call: one tile per tile row.
    constexpr uint32_t input_dfb_tiles = 2;
    const uint32_t Ht = a.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t accumulator_tiles = Ht;

    // 2. Factory: describe each call's block, the DFBs the sequence uses and the compute kernel's hardware
    //    configuration.
    auto describe = [&](const ttnn::Tensor& input) {
        auto block = reduce_host::ReduceBlockSpec::tiled(
            input.logical_shape()[-2], input.logical_shape()[-1], input.dtype(), output.dtype());
        block.input_cb_tiles = input_dfb_tiles;
        return reduce_host::ReduceCallConfig{
            .block = block,
            .reduce_math = ReduceOpMath::AVG,
            .reduce_dim = ReduceOpDim::W,
            .fp32_mode = ReduceFp32Mode::Fast,
        };
    };
    enum LogicalDfb : uint32_t { InputA, InputB, Auxiliary, Accumulator, Output };
    const reduce_host::ReduceSequenceCbIds sequence_dfbs{
        .auxiliary_cb_id = Auxiliary,
        .accumulator_cb_id = Accumulator,
        .output_cb_id = Output,
        .accumulator_cb_tiles = accumulator_tiles,
    };
    const auto hardware = reduce_host::make_reduce_hardware_config(a.device()->arch(), compute_hardware);

    // 3. Planner. AVG divides by the combined logical width of both calls.
    const reduce_host::ReduceSequencePlan plan =
        reduce_host::make_reduce_sequence_plan({{InputA, describe(a)}, {InputB, describe(b)}}, sequence_dfbs, hardware);

    ProgramSpec spec{.name = "reduce_helpers_row_mean_of_two_inputs"};

    // 4. Factory: DFBs.
    const tt::DataFormat input_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat output_format = datatype_to_dataformat_converter(output.dtype());
    spec.dataflow_buffers = {
        tile_dfb("in_a", input_format, input_dfb_tiles),
        tile_dfb("in_b", input_format, input_dfb_tiles),
        auxiliary_dfb(plan.auxiliary.cb),
        tile_dfb("accumulator", output_format, accumulator_tiles),
        tile_dfb("out", output_format, 2),
    };

    // 5. Factory: serialize. The compute kernel binds the logical IDs, in LogicalDfb order, to
    //    dfb::in_a, dfb::in_b, dfb::auxiliary, dfb::accumulator and dfb::out.
    const auto compute_args = plan.get_compile_time_args();
    const auto writer_args = plan.get_auxiliary_compile_time_args();

    // 6. Factory: kernels and the work unit. Compute both produces and consumes the accumulator.
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"a"}, .spec = a.tensor_spec()},
        {.unique_id = TensorParamName{"b"}, .spec = b.tensor_spec()},
        {.unique_id = TensorParamName{"output"}, .spec = output.tensor_spec()},
    };
    spec.kernels = {
        KernelSpec{
            .unique_id = KernelSpecName{"reader"},
            .source = OVERRIDE_KERNEL_PREFIX "ttnn/examples/reduce_helpers/kernels/dataflow/reader_two_inputs.cpp",
            .dfb_bindings = {producer("in_a"), producer("in_b")},
            .tensor_bindings =
                {{.tensor_parameter_name = TensorParamName{"a"}, .accessor_name = "a"},
                 {.tensor_parameter_name = TensorParamName{"b"}, .accessor_name = "b"}},
            .compile_time_args = {{"a_tiles", tile_count(a)}, {"b_tiles", tile_count(b)}},
            .hw_config = CreateReaderDataMovementConfig(),
        },
        writer_kernel(Ht, writer_args),
        KernelSpec{
            .unique_id = KernelSpecName{"compute"},
            .source = OVERRIDE_KERNEL_PREFIX "ttnn/examples/reduce_helpers/kernels/compute/row_mean_of_two_inputs.cpp",
            .dfb_bindings =
                {consumer("in_a"),
                 consumer("in_b"),
                 consumer("auxiliary"),
                 producer("accumulator"),
                 consumer("accumulator"),
                 producer("out")},
            .hw_config = compute_hardware,
            .advanced_options = {.compile_time_varargs = compute_args},
        },
    };
    spec.work_units = {WorkUnitSpec{
        .name = "core",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}, KernelSpecName{"compute"}},
        .target_nodes = NodeCoord{0, 0},
    }};

    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(TensorParamName{"a"}, a.mesh_tensor());
    run_args.tensor_args.emplace(TensorParamName{"b"}, b.mesh_tensor());
    run_args.tensor_args.emplace(TensorParamName{"output"}, output.mesh_tensor());
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Everything below runs the factories and checks their results; it is not part of the pattern.

TensorSpec tile_spec(uint32_t height, uint32_t width) {
    return TensorSpec(
        ttnn::Shape({height, width}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM)));
}

std::vector<bfloat16> random_values(uint32_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<bfloat16> values(count);
    for (auto& value : values) {
        value = bfloat16(distribution(rng));
    }
    return values;
}

void run(distributed::MeshDevice& device, const ttnn::device_operation::ProgramArtifacts& artifacts) {
    auto workload = MakeMeshWorkloadFromSpec(device, artifacts.spec);
    SetProgramRunArgs(workload.get_programs().begin()->second, artifacts.run_params);
    distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
}

bool matches(const std::vector<bfloat16>& actual, const std::vector<float>& expected, const char* name) {
    bool pass = actual.size() == expected.size();
    for (size_t i = 0; pass && i < actual.size(); ++i) {
        if (std::abs(static_cast<float>(actual[i]) - expected[i]) > 0.02F) {
            log_error(tt::LogAlways, "{}[{}] = {}, expected {}", name, i, static_cast<float>(actual[i]), expected[i]);
            pass = false;
        }
    }
    log_info(tt::LogAlways, "{}: {}", name, pass ? "passed" : "failed");
    return pass;
}

bool check_column_mean(distributed::MeshDevice& device, std::mt19937& rng) {
    constexpr uint32_t H = 100;
    constexpr uint32_t W = 256;
    const auto values = random_values(H * W, rng);
    const auto input = ttnn::Tensor::from_vector(values, tile_spec(H, W), &device);
    const auto output = ttnn::create_device_tensor(tile_spec(1, W), &device);

    run(device, column_mean_factory(input, output));

    std::vector<float> expected(W, 0.0F);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            expected[w] += static_cast<float>(values[h * W + w]) / H;
        }
    }
    return matches(output.to_vector<bfloat16>(), expected, "column_mean");
}

bool check_row_mean_of_two_inputs(distributed::MeshDevice& device, std::mt19937& rng) {
    constexpr uint32_t H = 64;
    constexpr uint32_t Wa = 96;
    constexpr uint32_t Wb = 40;
    const auto a_values = random_values(H * Wa, rng);
    const auto b_values = random_values(H * Wb, rng);
    const auto a = ttnn::Tensor::from_vector(a_values, tile_spec(H, Wa), &device);
    const auto b = ttnn::Tensor::from_vector(b_values, tile_spec(H, Wb), &device);
    const auto output = ttnn::create_device_tensor(tile_spec(H, 1), &device);

    run(device, row_mean_of_two_inputs_factory(a, b, output));

    std::vector<float> expected(H, 0.0F);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < Wa; ++w) {
            expected[h] += static_cast<float>(a_values[h * Wa + w]) / (Wa + Wb);
        }
        for (uint32_t w = 0; w < Wb; ++w) {
            expected[h] += static_cast<float>(b_values[h * Wb + w]) / (Wa + Wb);
        }
    }
    return matches(output.to_vector<bfloat16>(), expected, "row_mean_of_two_inputs");
}

}  // namespace

int main() {
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    std::mt19937 rng(42);
    bool pass = check_column_mean(*device, rng);
    pass &= check_row_mean_of_two_inputs(*device, rng);
    pass &= device->close();
    TT_FATAL(pass, "reduce_helpers example failed");
    return 0;
}
