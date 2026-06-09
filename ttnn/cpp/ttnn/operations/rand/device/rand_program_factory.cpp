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

// Spec-level names — referenced from both the ProgramSpec and the ProgramRunArgs, and by the
// kernels via the dfb:: / ta:: / args:: accessor namespaces.
constexpr const char* COMPUTE_KERNEL = "compute";
constexpr const char* WRITER_KERNEL = "writer";
constexpr const char* RAND_DFB = "rand_tiles";    // compute (producer) -> writer (consumer), fp32 intermediate
constexpr const char* RAND_OUT_DFB = "rand_out";  // writer-owned output-dtype scratch (the descriptor dst CB)
constexpr const char* OUTPUT_TENSOR = "output";

// One rand tile in flight is produced and consumed at a time; double-buffer the DFB.
constexpr uint32_t RAND_DFB_NUM_ENTRIES = 2;

// Work split, shared by all three tiers so the immutable spec (work_units), the static args
// (start_id/num_tiles), and the dynamic args (per-core seed) all derive the identical core list.
// This is the Metal 2.0 analog of the descriptor-era compute_rand_work_split helper.
struct RandWorkSplit {
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

// Work split derived ONLY from ImmutableInfo (grid + padded volume). Shared by the tier-1 spec build
// and the tier-2 static args so both produce the identical core list — and, by taking only
// ImmutableInfo, neither can see seed/from/to. (Tensor::physical_volume() == padded_shape().volume().)
RandWorkSplit rand_work_split_grid(CoreCoord grid, uint32_t units_to_divide) {
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2, std::move(cores)};
}

RandWorkSplit rand_work_split(const RandDeviceOperation::immutable_info_t& info) {
    return rand_work_split_grid(info.grid, info.output_spec.padded_shape().volume() / constants::TILE_HW);
}

// Per-device seed offset for sharded meshes. This depends on the dispatch coordinate, so it is a
// DYNAMIC, per-coordinate value — it belongs in the dynamic tier, not the immutable work split.
uint32_t rand_device_seed_offset(
    const RandDeviceOperation::operation_attributes_t& attrs,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    size_t num_cores) {
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (shard_mask.empty()) {
        return 0;
    }
    const ttnn::MeshCoordinate mesh_coordinate =
        mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
    const auto& mesh_shape = attrs.device->shape();
    size_t shard_linear_idx = 0;
    size_t shard_stride = 1;
    for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
        if (shard_mask[i]) {
            shard_linear_idx += mesh_coordinate[i] * shard_stride;
            shard_stride *= mesh_shape[i];
        }
    }
    return static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(num_cores);
}

uint32_t units_for_core(const RandWorkSplit& ws, const CoreCoord& core) {
    if (ws.core_group_1.contains(core)) {
        return ws.units_per_core_group_1;
    }
    if (ws.core_group_2.contains(core)) {
        return ws.units_per_core_group_2;
    }
    TT_THROW("Core not in specified core ranges");
}

}  // namespace

// ============================================================================
// ImmutableInfo extraction — the cache key AND the sole input to the spec/static tiers
// ============================================================================
RandDeviceOperation::immutable_info_t RandDeviceOperation::extract_immutable_info(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // rand has no input tensor: the output layout is fully determined by the attributes (same
    // construction as compute_output_specs), and the compute grid by the target device.
    TensorSpec output_spec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
    return immutable_info_t{
        .output_spec = std::move(output_spec), .grid = operation_attributes.device->compute_with_storage_grid_size()};
}

// ============================================================================
// Tier 1 — immutable blueprint (cache miss only). Derives ONLY from ImmutableInfo.
// ============================================================================
m2::ProgramSpec RandDeviceOperation::create_program_spec(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& /*output*/) {
    const immutable_info_t info = extract_immutable_info(operation_attributes, tensor_args);
    const auto ws = rand_work_split(info);

    // Two DFBs mirror the descriptor-era two CBs: an fp32 intermediate the compute generator fills
    // (fp32_dest_acc_en keeps values in [from,to)), and an output-dtype scratch the writer narrows
    // into before the NoC write. fp32 output is written straight from the intermediate; bf16 output
    // is truncated (high 16 bits) into the scratch. Keeping the intermediate fp32 + truncating —
    // rather than packing to bf16 in compute — preserves the strict [from,to) bound; a hardware pack
    // to bf16 rounds and can lift the top of the range up to `to`.
    const auto fp32_format = tt::DataFormat::Float32;
    const auto out_data_format = datatype_to_dataformat_converter(info.output_spec.data_type());
    m2::DataflowBufferSpec rand_dfb{
        .unique_id = RAND_DFB,
        .entry_size = tile_size(fp32_format),
        .num_entries = RAND_DFB_NUM_ENTRIES,
        .data_format_metadata = fp32_format,
    };
    m2::DataflowBufferSpec rand_out_dfb{
        .unique_id = RAND_OUT_DFB,
        .entry_size = tile_size(out_data_format),
        .num_entries = 1,  // fixed scratch slot, reserved once and reused (as the descriptor dst CB was)
        .data_format_metadata = out_data_format,
    };

    // Compute kernel: PRODUCER of the rand DFB. Generates one rand tile per work unit in fp32 dest
    // (fp32_dest_acc_en keeps the generator in range), packing to the DFB's output format.
    m2::KernelSpec compute_kernel{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(RAND_DFB, RAND_DFB)},
        .runtime_arg_schema =
            {
                // All per-node, as in the descriptor-era kernel: seed/from/to are dynamic
                // (hash-excluded, re-applied each dispatch); start_id/num_tiles are the static work split.
                .runtime_arg_names = {"seed", "from", "to", "start_id", "num_tiles"},
            },
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,  // if false, generated numbers can fall out of [from,to)
                .dst_full_sync_en = false,
                .math_approx_mode = true,
            },
    };

    // Writer kernel: CONSUMER of the fp32 intermediate DFB and PRODUCER of the output-dtype scratch
    // DFB. fp32 output is written straight from the intermediate; bf16 output is narrowed into the
    // scratch. The path is selected by an OUTPUT_DTYPE_* define, as in the descriptor era; only
    // fp32/bf16 are supported, so reject anything else — fail fast rather than emit a hanging program.
    m2::KernelSpec::CompilerOptions writer_opts;
    switch (info.output_spec.data_type()) {
        case DataType::BFLOAT16: writer_opts.defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_opts.defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: TT_THROW("RandDeviceOperation: unsupported output dtype for writer kernel");
    }
    m2::KernelSpec writer_kernel{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .compiler_options = std::move(writer_opts),
        // Writer owns the output scratch: bound as both producer and consumer so the DFB has both
        // endpoints (Metal 2.0 requires a consumer). The kernel uses it as a fixed scratch slot.
        .dfb_bindings =
            {m2::ConsumerOf(RAND_DFB, RAND_DFB),
             m2::ProducerOf(RAND_OUT_DFB, RAND_OUT_DFB),
             m2::ConsumerOf(RAND_OUT_DFB, RAND_OUT_DFB)},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = OUTPUT_TENSOR}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    m2::ProgramSpec spec;
    spec.name = "rand";
    spec.kernels = {std::move(compute_kernel), std::move(writer_kernel)};
    spec.dataflow_buffers = {std::move(rand_dfb), std::move(rand_out_dfb)};
    spec.tensor_parameters = {m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = info.output_spec}};
    // Both kernels run on every work core; per-core differences live in the runtime args, not here.
    spec.work_units = {m2::WorkUnitSpec{
        .name = "rand_work",
        .kernels = {COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = ws.all_cores,
    }};
    return spec;
}

// ============================================================================
// Tier 2 — static run-args: work-split scalars, fixed for a cache entry
// ============================================================================
m2::ProgramRunArgs RandDeviceOperation::create_static_args(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& /*output*/) {
    const auto ws = rand_work_split(extract_immutable_info(operation_attributes, tensor_args));

    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel_spec_name = COMPUTE_KERNEL};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel_spec_name = WRITER_KERNEL};

    uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        const uint32_t units_per_core = units_for_core(ws, core);
        // start_id / num_tiles are pure functions of the work split (and therefore of the spec
        // identity), so they are STATIC: baked here on the miss and never re-applied on a hit.
        compute_args.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", units_per_core}}});
        writer_args.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", units_per_core}}});
        tile_offset += units_per_core;
    }

    m2::ProgramRunArgs params;
    params.kernel_run_args = {std::move(compute_args), std::move(writer_args)};
    return params;
}

// ============================================================================
// Tier 3 — dynamic run-args: seed/from/to + output address, refreshed every dispatch
// ============================================================================
m2::ProgramRunArgs RandDeviceOperation::create_dynamic_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Per-core seed/from/to, exactly as the descriptor era: derive the full work split (for the core
    // list + index), give each core a distinct seed (base + core index, plus the per-device offset on
    // a sharded mesh), and repeat from/to per core. These are excluded from the program hash and
    // re-applied on every dispatch.
    const auto ws = rand_work_split(extract_immutable_info(operation_attributes, tensor_args));
    const uint32_t device_seed_offset =
        rand_device_seed_offset(operation_attributes, mesh_dispatch_coordinate, ws.cores.size());

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel_spec_name = COMPUTE_KERNEL};
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const uint32_t seed = operation_attributes.seed != 0
                                  ? operation_attributes.seed + static_cast<uint32_t>(i) + device_seed_offset
                                  : get_random_seed();
        compute_args.runtime_arg_values.push_back(
            {ws.cores[i], {{"seed", seed}, {"from", from_bits}, {"to", to_bits}}});
    }

    m2::ProgramRunArgs params;
    params.kernel_run_args = {std::move(compute_args)};
    // The output address changes every call (a fresh output tensor), so the tensor arg is dynamic.
    // The framework matches "output" to the output MeshTensor and re-patches its address on hits.
    params.tensor_args.push_back(m2::ProgramRunArgs::TensorArgument{
        .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())});
    return params;
}

}  // namespace ttnn::operations::rand
