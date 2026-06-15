// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <ctime>
#include <limits>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp";

// Work split + per-device seed offset, shared so the miss-build and the hit-refresh
// (create_program_spec re-run from the adapter) derive the identical core list and seed offset.
struct RandWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
    uint32_t device_seed_offset = 0;
};

RandWorkSplit compute_rand_work_split(
    const RandDeviceOperation::operation_attributes_t& attrs,
    RandDeviceOperation::tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    const ttnn::MeshCoordinate mesh_coordinate =
        mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
    uint32_t device_seed_offset = 0;
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        const auto& mesh_shape = attrs.device->shape();
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += mesh_coordinate[i] * shard_stride;
                shard_stride *= mesh_shape[i];
            }
        }
        device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(cores.size());
    }
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores),
        device_seed_offset};
}

// Per-core seed; shared so the miss-build and the hit-refresh produce identical values.
uint32_t rand_seed_for_core(
    const RandDeviceOperation::operation_attributes_t& attrs, int i, uint32_t device_seed_offset) {
    return attrs.seed != 0 ? attrs.seed + i + device_seed_offset : get_random_seed();
}

}  // namespace

// Metal 2.0 program factory. Produces the immutable ProgramSpec plus its mutable
// ProgramRunArgs. seed/from/to live in ProgramRunArgs (NOT the spec), so they are
// re-applied on every dispatch (cache miss and cache hit) — the structural replacement
// for the old custom-hash-excludes-seed + get_dynamic_runtime_args() patching dance.
ttnn::device_operation::ProgramArtifacts RandDeviceOperation::RandProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto ws = compute_rand_work_split(operation_attributes, output, /*mesh_dispatch_coordinate=*/std::nullopt);

    const DataType output_dtype = output.dtype();
    const auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t intermed_num_tiles = 2;
    constexpr uint32_t in_out_num_tiles = 1;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "rand";

    // "intermed" (legacy CB c_24): fp32 staging, produced by compute, consumed by writer.
    // "dst" (legacy CB c_0): output-dtype L1 scratch the writer uses to pack bf16 before the
    // NoC write; it has no FIFO peer (the writer is its only user), so it is bound as a
    // self-loop (producer + consumer on the writer) to satisfy the producer/consumer invariant.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"intermed"},
            .entry_size = intermed_tile_size,
            .num_entries = intermed_num_tiles,
            .data_format_metadata = tt::DataFormat::Float32,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"dst"},
            .entry_size = dtype_tile_size,
            .num_entries = in_out_num_tiles,
            .data_format_metadata = out_data_format,
        },
    };

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"intermed"},
                    .accessor_name = "intermed",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"seed", "start_id", "num_tiles"},
                .common_runtime_arg_names = {"from", "to"},
            },
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                // if fp32_dest_acc_en is false a precision error may occur which makes the
                // generated number fall out of range [from, to)
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .math_approx_mode = true,
            },
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"intermed"},
                    .accessor_name = "intermed",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"dst"},
                    .accessor_name = "dst",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"dst"},
                    .accessor_name = "dst",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{
                    .tensor_parameter_name = m2::TensorParamName{"output"},
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    // The writer kernel only implements float32 and bfloat16 output paths. Gate the path via a
    // compile-time define, and fail fast on anything else so we never build a hangable program.
    switch (output_dtype) {
        case DataType::BFLOAT16: writer.compiler_options.defines = {{"OUTPUT_DTYPE_BFLOAT16", "1"}}; break;
        case DataType::FLOAT32: writer.compiler_options.defines = {{"OUTPUT_DTYPE_FLOAT32", "1"}}; break;
        default: TT_THROW("RandDeviceOperation: unsupported output dtype for writer kernel");
    }

    spec.kernels = {compute, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "rand",
            .kernels = {m2::KernelSpecName{"compute"}, m2::KernelSpecName{"writer"}},
            .target_nodes = ws.all_cores,
        },
    };

    // ---- ProgramRunArgs (mutable; seed/from/to refreshed every dispatch) ----
    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    m2::ProgramRunArgs run;
    m2::KernelRunArgs compute_run{.kernel = m2::KernelSpecName{"compute"}};
    compute_run.common_runtime_arg_values = {{"from", from_bits}, {"to", to_bits}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t tile_offset = 0;
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const auto& core = ws.cores[i];
        uint32_t units_per_core;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);

        compute_run.runtime_arg_values.push_back(
            {core, {{"seed", seed}, {"start_id", tile_offset}, {"num_tiles", units_per_core}}});
        writer_run.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", units_per_core}}});

        tile_offset += units_per_core;
    }
    run.kernel_run_args = {compute_run, writer_run};
    run.tensor_args = {{m2::TensorParamName{"output"}, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::operations::rand
