// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <bit>
#include <cstdint>
#include <ctime>
#include <limits>
#include <memory>
#include <random>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<std::int32_t>::max());

auto get_random_seed() -> std::uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp";

// Work split + per-device seed offset, shared by create_descriptor (cache miss) and
// override_runtime_arguments (cache hit) so both derive the identical core list and seed offset.
struct RandWorkSplit {
    std::uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    std::uint32_t units_per_core_group_1 = 0;
    std::uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
    std::uint32_t device_seed_offset = 0;
};

RandWorkSplit compute_rand_work_split(
    const RandDeviceOperation::operation_attributes_t& attrs,
    RandDeviceOperation::tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    auto grid = output.device()->compute_with_storage_grid_size();
    std::uint32_t units_to_divide = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    const ttnn::MeshCoordinate physical_mesh_coordinate =
        mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
    ttnn::MeshCoordinate distribution_coordinate = physical_mesh_coordinate;
    const tt::tt_metal::distributed::MeshShape* distribution_shape = std::addressof(attrs.device->shape());
    if (attrs.tensor_topology.has_value()) {
        auto tensor_coord = attrs.tensor_topology->get_tensor_coord(physical_mesh_coordinate);
        TT_FATAL(
            tensor_coord.has_value(),
            "Rand: physical mesh coordinate {} is not present in the tensor topology",
            physical_mesh_coordinate);
        distribution_coordinate = std::move(*tensor_coord);
        distribution_shape = std::addressof(attrs.tensor_topology->distribution_shape());
    }

    std::uint32_t device_seed_offset = 0;
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += distribution_coordinate[i] * shard_stride;
                shard_stride *= (*distribution_shape)[i];
            }
        }
        device_seed_offset = static_cast<std::uint32_t>(shard_linear_idx) * static_cast<std::uint32_t>(cores.size());
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

// Per-core seed; shared so the miss-build and the hit-patch produce identical values.
std::uint32_t rand_seed_for_core(
    const RandDeviceOperation::operation_attributes_t& attrs, int i, std::uint32_t device_seed_offset) {
    return attrs.seed != 0 ? attrs.seed + i + device_seed_offset : get_random_seed();
}

// Per-core work assignment. Single-sourced so the cache-miss build (create_descriptor) and the
// cache-hit patch (override_runtime_arguments) can never drift on core-group selection or tile_offset
// accumulation — each derives its runtime args from the same layout.
struct RandCoreWork {
    CoreCoord core;
    std::uint32_t units_per_core;
    std::uint32_t tile_offset;
};
std::vector<RandCoreWork> rand_core_layout(const RandWorkSplit& ws) {
    std::vector<RandCoreWork> layout;
    layout.reserve(ws.cores.size());
    std::uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        std::uint32_t units_per_core;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        layout.push_back({core, units_per_core, tile_offset});
        tile_offset += units_per_core;
    }
    return layout;
}

}  // namespace

ProgramDescriptor RandDeviceOperation::RandProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.restricted_mesh_coords.has_value() &&
        (!mesh_dispatch_coordinate.has_value() ||
         std::ranges::find(*operation_attributes.restricted_mesh_coords, *mesh_dispatch_coordinate) ==
             operation_attributes.restricted_mesh_coords->end())) {
        return {};
    }

    const auto ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const auto& all_cores = ws.all_cores;
    const auto num_cores_total = ws.cores.size();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const std::uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr std::uint32_t output_num_tiles = 2;

    constexpr std::uint32_t output_cb_id = CBIndex::c_24;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = output_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_id,
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    writer_ct_args.reserve(8);
    writer_ct_args.push_back(output_cb_id);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_total);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {output_cb_id};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        // Retain generated values in FP32 until packing so reduced destination
        // precision cannot cross the validated inclusive bounds.
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .math_approx_mode = true,
    };
    compute_desc.runtime_args.reserve(num_cores_total);

    const std::uint32_t lower_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.lower_bound);
    const std::uint32_t upper_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.upper_bound);

    const auto layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const std::uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);

        // seed/range bounds are DYNAMIC (omitted from the cache key / attribute_names): baked here for the
        // cache-miss build, and re-applied on every cache hit via override_runtime_arguments().
        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{seed, lower_bound_bits, upper_bound_bits, tile_offset, units_per_core});

        // Register the output address as a Buffer* binding so rand takes the fast cache-hit path
        // (real program caching) with the address correctly re-patched each dispatch.
        writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void RandDeviceOperation::RandProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Re-derive every per-dispatch arg on each cache hit from the same builder create_descriptor uses:
    // compute's seed/bounds and the writer's output address. override replaces resolve_bindings, so
    // the address is ours to re-apply too. Push order in create_descriptor: writer 0, compute 1.
    constexpr std::uint32_t writer_kernel_idx = 0;
    constexpr std::uint32_t compute_kernel_idx = 1;

    const auto ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const std::uint32_t lower_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.lower_bound);
    const std::uint32_t upper_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.upper_bound);
    const std::uint32_t out_addr = output.buffer()->address();

    const auto layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const std::uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);

        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_idx, core);
        compute_args[0] = seed;
        compute_args[1] = lower_bound_bits;
        compute_args[2] = upper_bound_bits;
        compute_args[3] = tile_offset;
        compute_args[4] = units_per_core;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx, core);
        writer_args[0] = out_addr;
        writer_args[1] = tile_offset;
        writer_args[2] = units_per_core;
    }
}

}  // namespace ttnn::operations::rand
