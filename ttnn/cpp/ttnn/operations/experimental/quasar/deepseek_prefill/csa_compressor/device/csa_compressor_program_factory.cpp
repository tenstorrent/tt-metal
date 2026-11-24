// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor_device_operation.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

namespace ttnn::experimental::prim::qsr {
namespace {

using namespace tt::tt_metal;

constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint16_t);
constexpr uint32_t kStateScratchTiles = 7;
constexpr uint32_t kReaderScratchTiles = 4;

constexpr auto kStateKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/device/kernels/"
    "csa_state_update.cpp";
constexpr auto kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/device/kernels/"
    "reader_csa_compressor.cpp";
constexpr auto kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/device/kernels/"
    "compute_csa_compressor.cpp";
constexpr auto kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/device/kernels/"
    "writer_csa_compressor.cpp";

// Program-scope resource names, CSA-prefixed so the unity build cannot collide them with other ops' identically named
// constants. "Ca" / "Cb" are the two halves of the CSA projection.
const KernelSpecName CSA_STATE{"state"};
const KernelSpecName CSA_READER{"reader"};
const KernelSpecName CSA_COMPUTE{"compute"};
const KernelSpecName CSA_WRITER{"writer"};
const DFBSpecName CSA_CANDIDATE_KV{"candidate_kv"};
const DFBSpecName CSA_CANDIDATE_SCORE{"candidate_score"};
const DFBSpecName CSA_POOLED{"pooled"};
const DFBSpecName CSA_MAX_VALUES{"max_values"};
const DFBSpecName CSA_CA_BIAS{"ca_bias"};
const DFBSpecName CSA_CB_BIAS{"cb_bias"};
const ScratchpadSpecName CSA_READER_SCRATCH{"reader_scratch"};
const ScratchpadSpecName CSA_STATE_SCRATCH{"state_scratch"};
const TensorParamName CSA_KV{"kv"};
const TensorParamName CSA_GATE{"gate"};
const TensorParamName CSA_BIAS{"bias"};
const TensorParamName CSA_BASE_KV{"base_kv"};
const TensorParamName CSA_BASE_SCORE{"base_score"};
const TensorParamName CSA_PREDECESSOR_KV{"predecessor_kv"};
const TensorParamName CSA_PREDECESSOR_SCORE{"predecessor_score"};
const TensorParamName CSA_OUTPUT{"output"};
const TensorParamName CSA_OUTPUT_KV{"output_kv"};
const TensorParamName CSA_OUTPUT_SCORE{"output_score"};

std::pair<uint32_t, uint32_t> local_runtime(
    const CsaRuntimeParams& params, uint32_t local_seq, const MeshCoordinate& coord) {
    const uint32_t rank = coord[params.cluster_axis];
    const uint32_t local_global_start = rank * local_seq;
    const uint32_t local_valid = params.seq_len_actual > local_global_start
                                     ? std::min(local_seq, params.seq_len_actual - local_global_start)
                                     : 0;
    return {local_valid, params.first_token_position + local_global_start};
}

// kv/gate pack the Ca and Cb halves side by side, so the head dimension is half the projection width.
uint32_t factory_head_dim_of(const Tensor& kv) { return kv.logical_shape()[-1] / 2; }

// The cores a kernel runs on: the range for the spec, and the row-major list the per-core args follow. The cache
// hit re-derives them from the same shapes, since its args go to the same cores.
struct CsaCoreSplit {
    CoreRangeSet range;
    std::vector<CoreCoord> cores;
};

uint32_t csa_state_tiles(const Tensor& kv) { return 2 * factory_head_dim_of(kv) / tt::constants::TILE_WIDTH; }

// The 64-row slab is two tile rows of state_width_tiles each, and those tiles are independent:
// each one patches its own eight live rows from its own feature columns.
CsaCoreSplit csa_state_preparation_cores(const Tensor& kv) {
    const auto grid = kv.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t num_cores = std::min(csa_state_tiles(kv), grid_cores);
    CoreRangeSet range = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    std::vector<CoreCoord> cores = corerange_to_cores(range, num_cores, /*row_wise=*/true);
    return {std::move(range), std::move(cores)};
}

uint32_t csa_compression_output_tiles(const Tensor& kv) {
    const uint32_t local_seq = kv.logical_shape()[-2];
    const uint32_t output_height_tiles = (local_seq / 4 + 31) / 32;
    return output_height_tiles * (factory_head_dim_of(kv) / tt::constants::TILE_WIDTH);
}

// Output tiles are independent -- the softmax runs per feature column and windows never interact --
// so they split across the grid with no cross-core traffic. The state kernel needs cores of its
// own because it is a RISCV_0 data movement kernel, which the reader already occupies, so the two
// blocks are carved row-major with the state block starting where the compression block ends.
// Returns {compression cores, state cores}.
std::pair<CsaCoreSplit, CsaCoreSplit> csa_compression_cores(const Tensor& kv) {
    const auto grid = kv.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t state_tiles = csa_state_tiles(kv);
    const uint32_t state_core_count = std::min(state_tiles, grid_cores / 2);
    const uint32_t num_cores = std::min(csa_compression_output_tiles(kv), grid_cores - state_core_count);
    CoreRangeSet compression_range = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    std::vector<CoreCoord> cores = corerange_to_cores(compression_range, num_cores, /*row_wise=*/true);
    const CoreCoord state_start = grid_to_cores(grid_cores, grid.x, grid.y, /*row_wise=*/true)[num_cores];
    CoreRangeSet state_range = num_cores_to_corerangeset(state_start, state_core_count, grid, /*row_wise=*/true);
    std::vector<CoreCoord> state_cores = corerange_to_cores(state_range, state_core_count, /*row_wise=*/true);
    return {{std::move(compression_range), std::move(cores)}, {std::move(state_range), std::move(state_cores)}};
}

DataflowBufferSpec csa_tile_dfb(const DFBSpecName& name, uint32_t num_tiles) {
    return DataflowBufferSpec{
        .unique_id = name,
        .entry_size = kTileBytes,
        .num_entries = num_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
}

// The state kernel, shared by both programs. Its base state is the base state in state preparation and the
// predecessor state in compression, so the bound parameters are passed in.
KernelSpec csa_state_kernel_spec(
    const Tensor& kv,
    const TensorParamName& base_kv,
    const TensorParamName& base_score,
    const TensorParamName& output_kv,
    const TensorParamName& output_score) {
    const uint32_t head_dim = factory_head_dim_of(kv);
    return KernelSpec{
        .unique_id = CSA_STATE,
        .source = std::filesystem::path(kStateKernel),
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = CSA_STATE_SCRATCH, .accessor_name = "scratch"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = CSA_KV, .accessor_name = "kv"},
                TensorBinding{.tensor_parameter_name = CSA_GATE, .accessor_name = "gate"},
                TensorBinding{.tensor_parameter_name = CSA_BIAS, .accessor_name = "bias"},
                TensorBinding{.tensor_parameter_name = base_kv, .accessor_name = "base_kv"},
                TensorBinding{.tensor_parameter_name = base_score, .accessor_name = "base_score"},
                TensorBinding{.tensor_parameter_name = output_kv, .accessor_name = "output_kv"},
                TensorBinding{.tensor_parameter_name = output_score, .accessor_name = "output_score"},
            },
        .compile_time_args =
            {
                {"input_width_tiles", 2 * head_dim / tt::constants::TILE_WIDTH},
                {"state_width_tiles", head_dim / tt::constants::TILE_WIDTH},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"local_valid", "absolute_start", "state_tiles", "first_state_tile"},
            },
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };
}

KernelRunArgs csa_state_kernel_run_args(
    const CsaCoreSplit& state, uint32_t state_tiles, uint32_t local_valid, uint32_t absolute_start) {
    KernelRunArgs run_args{.kernel = CSA_STATE};
    const uint32_t num_cores = state.cores.size();
    const uint32_t base_tiles = state_tiles / num_cores;
    const uint32_t extra_tiles = state_tiles % num_cores;
    uint32_t first_tile = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t core_tiles = base_tiles + (i < extra_tiles ? 1 : 0);
        AddRuntimeArgsForNode(
            run_args.runtime_arg_values,
            state.cores[i],
            {
                {"local_valid", local_valid},
                {"absolute_start", absolute_start},
                {"state_tiles", core_tiles},
                {"first_state_tile", first_tile},
            });
        first_tile += core_tiles;
    }
    return run_args;
}

// Every value a cache hit has to refresh besides the tensor bindings -- the chunk position -- is the same on all
// of a kernel's cores, so the hit sets those args on each of its cores and leaves the shape-derived tile split
// from the miss in place.
KernelRunArgs csa_chunk_position_run_args(
    const KernelSpecName& kernel,
    const std::vector<CoreCoord>& cores,
    const char* valid_arg,
    uint32_t valid,
    uint32_t absolute_start) {
    KernelRunArgs run_args{.kernel = kernel};
    for (const auto& core : cores) {
        AddRuntimeArgsForNode(
            run_args.runtime_arg_values, core, {{valid_arg, valid}, {"absolute_start", absolute_start}});
    }
    return run_args;
}

ttnn::device_operation::ProgramArtifacts create_csa_state_preparation_program(
    const CsaRuntimeParams& params,
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const MeshCoordinate& coord) {
    const auto [local_valid, absolute_start] = local_runtime(params, args.kv.logical_shape()[-2], coord);
    const CsaCoreSplit state = csa_state_preparation_cores(args.kv);

    ProgramSpec spec{
        .name = "csa_state_preparation",
        .kernels = {csa_state_kernel_spec(args.kv, CSA_BASE_KV, CSA_BASE_SCORE, CSA_OUTPUT_KV, CSA_OUTPUT_SCORE)},
        .scratchpads =
            {
                ScratchpadSpec{.unique_id = CSA_STATE_SCRATCH, .size_per_node = kStateScratchTiles * kTileBytes},
            },
        .tensor_parameters =
            {
                TensorParameter{.unique_id = CSA_KV, .spec = args.kv.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_GATE, .spec = args.gate.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_BIAS, .spec = args.position_bias.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_BASE_KV, .spec = args.base_kv_state.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_BASE_SCORE, .spec = args.base_score_state.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_OUTPUT_KV, .spec = outputs[0].mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_OUTPUT_SCORE, .spec = outputs[1].mesh_tensor().tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "state",
                    .kernels = {CSA_STATE},
                    .target_nodes = state.range,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(
        csa_state_kernel_run_args(state, csa_state_tiles(args.kv), local_valid, absolute_start));
    run_args.tensor_args.emplace(CSA_KV, args.kv.mesh_tensor());
    run_args.tensor_args.emplace(CSA_GATE, args.gate.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BIAS, args.position_bias.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BASE_KV, args.base_kv_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BASE_SCORE, args.base_score_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_KV, outputs[0].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_SCORE, outputs[1].mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ttnn::device_operation::ProgramArtifacts create_csa_compression_program(
    const CsaRuntimeParams& params,
    const CsaCompressionInputs& args,
    std::array<Tensor, 3>& outputs,
    const MeshCoordinate& coord) {
    const auto grid = args.kv.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    TT_FATAL(grid_cores > 1, "CSA compression requires at least two worker cores");
    const uint32_t local_seq = args.kv.logical_shape()[-2];
    const auto [local_valid, absolute_start] = local_runtime(params, local_seq, coord);
    const uint32_t head_dim = factory_head_dim_of(args.kv);
    const uint32_t state_width_tiles = head_dim / tt::constants::TILE_WIDTH;
    const uint32_t input_width_tiles = 2 * state_width_tiles;
    const uint32_t output_tiles = csa_compression_output_tiles(args.kv);
    const uint32_t state_tiles = 2 * state_width_tiles;
    const auto [compression, state] = csa_compression_cores(args.kv);
    const uint32_t num_cores = compression.cores.size();

    KernelSpec reader_spec{
        .unique_id = CSA_READER,
        .source = std::filesystem::path(kReaderKernel),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = CSA_CANDIDATE_KV,
                    .accessor_name = "candidate_kv",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CANDIDATE_SCORE,
                    .accessor_name = "candidate_score",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CA_BIAS,
                    .accessor_name = "ca_bias",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CB_BIAS,
                    .accessor_name = "cb_bias",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = CSA_READER_SCRATCH, .accessor_name = "scratch"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = CSA_KV, .accessor_name = "kv"},
                TensorBinding{.tensor_parameter_name = CSA_GATE, .accessor_name = "gate"},
                TensorBinding{.tensor_parameter_name = CSA_BIAS, .accessor_name = "bias"},
                TensorBinding{.tensor_parameter_name = CSA_PREDECESSOR_KV, .accessor_name = "predecessor_kv"},
                TensorBinding{.tensor_parameter_name = CSA_PREDECESSOR_SCORE, .accessor_name = "predecessor_score"},
            },
        .compile_time_args =
            {
                {"input_width_tiles", input_width_tiles},
                {"output_width_tiles", state_width_tiles},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"output_tiles", "complete_windows", "absolute_start", "first_output_tile"},
            },
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    KernelSpec compute_spec{
        .unique_id = CSA_COMPUTE,
        .source = std::filesystem::path(kComputeKernel),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = CSA_CANDIDATE_KV,
                    .accessor_name = "candidate_kv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CANDIDATE_SCORE,
                    .accessor_name = "candidate_score",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CA_BIAS,
                    .accessor_name = "ca_bias",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_CB_BIAS,
                    .accessor_name = "cb_bias",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_POOLED,
                    .accessor_name = "pooled",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The running max: compute is its only toucher, so it is a self-loop.
                DFBBinding{
                    .dfb_spec_name = CSA_MAX_VALUES,
                    .accessor_name = "max_values",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = CSA_MAX_VALUES,
                    .accessor_name = "max_values",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"output_tiles"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = MathFidelity::HiFi4,
                .enable_32_bit_dest = true,
            },
    };

    KernelSpec writer_spec{
        .unique_id = CSA_WRITER,
        .source = std::filesystem::path(kWriterKernel),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = CSA_POOLED,
                    .accessor_name = "pooled",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = CSA_OUTPUT, .accessor_name = "output"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"output_tiles", "first_output_tile"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    KernelRunArgs reader_run_args{.kernel = CSA_READER};
    KernelRunArgs compute_run_args{.kernel = CSA_COMPUTE};
    KernelRunArgs writer_run_args{.kernel = CSA_WRITER};
    const uint32_t base_tiles = output_tiles / num_cores;
    const uint32_t extra_tiles = output_tiles % num_cores;
    uint32_t first_tile = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = compression.cores[i];
        const uint32_t core_tiles = base_tiles + (i < extra_tiles ? 1 : 0);

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"output_tiles", core_tiles},
                {"complete_windows", local_valid / 4},
                {"absolute_start", absolute_start},
                {"first_output_tile", first_tile},
            });
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"output_tiles", core_tiles}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"output_tiles", core_tiles},
                {"first_output_tile", first_tile},
            });

        first_tile += core_tiles;
    }

    ProgramSpec spec{
        .name = "csa_compression",
        .kernels =
            {
                std::move(reader_spec),
                std::move(compute_spec),
                std::move(writer_spec),
                csa_state_kernel_spec(
                    args.kv, CSA_PREDECESSOR_KV, CSA_PREDECESSOR_SCORE, CSA_OUTPUT_KV, CSA_OUTPUT_SCORE),
            },
        .dataflow_buffers =
            {
                csa_tile_dfb(CSA_CANDIDATE_KV, 8),
                csa_tile_dfb(CSA_CANDIDATE_SCORE, 8),
                csa_tile_dfb(CSA_POOLED, 1),
                csa_tile_dfb(CSA_MAX_VALUES, 1),
                csa_tile_dfb(CSA_CA_BIAS, 1),
                csa_tile_dfb(CSA_CB_BIAS, 1),
            },
        // The reader's and the state kernel's scratch: each filled and read by that one kernel, on its own cores.
        .scratchpads =
            {
                ScratchpadSpec{.unique_id = CSA_READER_SCRATCH, .size_per_node = kReaderScratchTiles * kTileBytes},
                ScratchpadSpec{.unique_id = CSA_STATE_SCRATCH, .size_per_node = kStateScratchTiles * kTileBytes},
            },
        .tensor_parameters =
            {
                TensorParameter{.unique_id = CSA_KV, .spec = args.kv.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_GATE, .spec = args.gate.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_BIAS, .spec = args.position_bias.mesh_tensor().tensor_spec()},
                TensorParameter{
                    .unique_id = CSA_PREDECESSOR_KV,
                    .spec = args.predecessor_kv_state.mesh_tensor().tensor_spec(),
                },
                TensorParameter{
                    .unique_id = CSA_PREDECESSOR_SCORE,
                    .spec = args.predecessor_score_state.mesh_tensor().tensor_spec(),
                },
                TensorParameter{.unique_id = CSA_OUTPUT, .spec = outputs[0].mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_OUTPUT_KV, .spec = outputs[1].mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = CSA_OUTPUT_SCORE, .spec = outputs[2].mesh_tensor().tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "compression",
                    .kernels = {CSA_READER, CSA_COMPUTE, CSA_WRITER},
                    .target_nodes = compression.range,
                },
                WorkUnitSpec{
                    .name = "state",
                    .kernels = {CSA_STATE},
                    .target_nodes = state.range,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(4);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(csa_state_kernel_run_args(state, state_tiles, local_valid, absolute_start));
    run_args.tensor_args.emplace(CSA_KV, args.kv.mesh_tensor());
    run_args.tensor_args.emplace(CSA_GATE, args.gate.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BIAS, args.position_bias.mesh_tensor());
    run_args.tensor_args.emplace(CSA_PREDECESSOR_KV, args.predecessor_kv_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_PREDECESSOR_SCORE, args.predecessor_score_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT, outputs[0].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_KV, outputs[1].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_SCORE, outputs[2].mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts CsaStatePreparationProgramFactory::create_mesh_workload_artifacts(
    const CsaRuntimeParams& params,
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        auto program = create_csa_state_preparation_program(params, args, outputs, coord);
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(program.spec),
            .run_params = std::move(program.run_params),
        });
    }
    return artifacts;
}

ProgramRunArgs CsaStatePreparationProgramFactory::override_runtime_arguments(
    const CsaRuntimeParams& params,
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const ttnn::MeshCoordinateRange& range) {
    const auto [local_valid, absolute_start] = local_runtime(params, args.kv.logical_shape()[-2], range.start_coord());

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(csa_chunk_position_run_args(
        CSA_STATE, csa_state_preparation_cores(args.kv).cores, "local_valid", local_valid, absolute_start));
    run_args.tensor_args.emplace(CSA_KV, args.kv.mesh_tensor());
    run_args.tensor_args.emplace(CSA_GATE, args.gate.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BIAS, args.position_bias.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BASE_KV, args.base_kv_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BASE_SCORE, args.base_score_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_KV, outputs[0].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_SCORE, outputs[1].mesh_tensor());
    return run_args;
}

ttnn::device_operation::MeshWorkloadArtifacts CsaCompressionProgramFactory::create_mesh_workload_artifacts(
    const CsaRuntimeParams& params,
    const CsaCompressionInputs& args,
    std::array<Tensor, 3>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        auto program = create_csa_compression_program(params, args, outputs, coord);
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(program.spec),
            .run_params = std::move(program.run_params),
        });
    }
    return artifacts;
}

ProgramRunArgs CsaCompressionProgramFactory::override_runtime_arguments(
    const CsaRuntimeParams& params,
    const CsaCompressionInputs& args,
    std::array<Tensor, 3>& outputs,
    const ttnn::MeshCoordinateRange& range) {
    const auto [local_valid, absolute_start] = local_runtime(params, args.kv.logical_shape()[-2], range.start_coord());
    const auto [compression, state] = csa_compression_cores(args.kv);

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(csa_chunk_position_run_args(
        CSA_READER, compression.cores, "complete_windows", local_valid / 4, absolute_start));
    run_args.kernel_run_args.push_back(
        csa_chunk_position_run_args(CSA_STATE, state.cores, "local_valid", local_valid, absolute_start));
    run_args.tensor_args.emplace(CSA_KV, args.kv.mesh_tensor());
    run_args.tensor_args.emplace(CSA_GATE, args.gate.mesh_tensor());
    run_args.tensor_args.emplace(CSA_BIAS, args.position_bias.mesh_tensor());
    run_args.tensor_args.emplace(CSA_PREDECESSOR_KV, args.predecessor_kv_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_PREDECESSOR_SCORE, args.predecessor_score_state.mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT, outputs[0].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_KV, outputs[1].mesh_tensor());
    run_args.tensor_args.emplace(CSA_OUTPUT_SCORE, outputs[2].mesh_tensor());
    return run_args;
}

}  // namespace ttnn::experimental::prim::qsr
