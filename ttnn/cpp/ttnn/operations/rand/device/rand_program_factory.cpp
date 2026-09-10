// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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

std::uint32_t get_random_seed() {
    static std::mutex mutex;
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_int_distribution<std::uint32_t> distribution(1, std::numeric_limits<std::uint32_t>::max());
    std::lock_guard<std::mutex> lock(mutex);
    return distribution(rng);
}

// Same finalizer as compute_kernel_lib::rand_mix32 in the kernels.
constexpr std::uint32_t mix32(std::uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}
constexpr std::uint32_t stream_key(std::uint32_t seed, std::uint32_t device_index) {
    return mix32(mix32(seed) ^ (device_index * 0x9E3779B9u + 0x7F4A7C15u));
}
constexpr std::uint32_t core_key(std::uint32_t stream, std::uint32_t core_index) {
    return mix32(stream ^ (core_index * 0x85EBCA6Bu + 0x165667B1u));
}

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp";

constexpr std::uint32_t output_cb_id = CBIndex::c_24;
constexpr std::uint32_t state_cb_id = CBIndex::c_25;
constexpr std::uint32_t state_page_bytes = 128;  // one uint32 row per core

// Work split + device index in the distribution, shared by create_descriptor (cache miss) and
// override_runtime_arguments (cache hit) so both derive the identical core list and keys.
struct RandWorkSplit {
    std::uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    std::uint32_t units_per_core_group_1 = 0;
    std::uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
    std::uint32_t device_index = 0;
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

    // Linear index over the sharded mesh dims only: replicas share an index and hence a stream.
    std::uint32_t device_index = 0;
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
        device_index = static_cast<std::uint32_t>(shard_linear_idx);
    }
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores),
        device_index};
}

// Per-dispatch keys. seed == 0 draws fresh host randomness on every dispatch (cache hits included).
struct RandKeys {
    std::uint32_t stream;           // Threefry key0; the same on every core so output is independent of the core grid
    std::uint32_t random_per_core;  // seed == 0: each core draws its own LFSR seed
};
RandKeys rand_keys(const RandDeviceOperation::operation_attributes_t& attrs, std::uint32_t device_index) {
    if (attrs.seed != 0) {
        return {stream_key(attrs.seed, device_index), 0};
    }
    return {get_random_seed(), 1};
}
std::uint32_t rand_key_for_core(const RandDeviceOperation::operation_attributes_t& attrs, const RandKeys& keys, int i) {
    if (attrs.generator == RandGenerator::THREEFRY) {
        return keys.stream;
    }
    return keys.random_per_core ? get_random_seed() : core_key(keys.stream, static_cast<std::uint32_t>(i));
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
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.restricted_mesh_coords.has_value() &&
        (!mesh_dispatch_coordinate.has_value() ||
         std::ranges::find(*operation_attributes.restricted_mesh_coords, *mesh_dispatch_coordinate) ==
             operation_attributes.restricted_mesh_coords->end())) {
        return {};
    }

    const RandWorkSplit ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const CoreRangeSet& all_cores = ws.all_cores;
    const size_t num_cores_total = ws.cores.size();
    const bool has_state = tensor_args.state.has_value();

    DataType output_dtype = output.dtype();
    tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_dtype);
    const std::uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr std::uint32_t output_num_tiles = 2;

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
    if (has_state) {
        // Page 0 carries the epoch to compute; page 1 stages the incremented epoch for the write-back.
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * state_page_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = state_cb_id,
                .data_format = tt::DataFormat::UInt32,
                .page_size = state_page_bytes,
            }}},
        });
    }

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    writer_ct_args.reserve(16);
    writer_ct_args.push_back(output_cb_id);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    writer_ct_args.push_back(has_state ? 1 : 0);
    writer_ct_args.push_back(state_cb_id);
    if (has_state) {
        TensorAccessorArgs(*tensor_args.state->buffer()).append_to(writer_ct_args);
    }

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
    compute_desc.compile_time_args = {
        output_cb_id, static_cast<std::uint32_t>(operation_attributes.generator), has_state ? 1u : 0u, state_cb_id};
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

    const RandKeys keys = rand_keys(operation_attributes, ws.device_index);
    const std::vector<RandCoreWork> layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const std::uint32_t key = rand_key_for_core(operation_attributes, keys, i);

        // key/range bounds are DYNAMIC (omitted from the cache key / attribute_names): baked here for the
        // cache-miss build, and re-applied on every cache hit via override_runtime_arguments().
        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                key, lower_bound_bits, upper_bound_bits, tile_offset, units_per_core, 0u});

        // Register the output address as a Buffer* binding so rand takes the fast cache-hit path
        // (real program caching) with the address correctly re-patched each dispatch.
        if (has_state) {
            writer_desc.emplace_runtime_args(
                core,
                {output.buffer(),
                 tile_offset,
                 units_per_core,
                 tensor_args.state->buffer(),
                 static_cast<std::uint32_t>(i)});
        } else {
            writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});
        }
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void RandDeviceOperation::RandProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Re-derive every per-dispatch arg on each cache hit from the same builder create_descriptor uses:
    // compute's key/bounds and the writer's output and state addresses. override replaces resolve_bindings, so
    // the addresses are ours to re-apply too. Push order in create_descriptor: writer 0, compute 1.
    constexpr std::uint32_t writer_kernel_idx = 0;
    constexpr std::uint32_t compute_kernel_idx = 1;

    const RandWorkSplit ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const std::uint32_t lower_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.lower_bound);
    const std::uint32_t upper_bound_bits = std::bit_cast<std::uint32_t>(operation_attributes.upper_bound);
    const std::uint32_t out_addr = output.buffer()->address();

    const RandKeys keys = rand_keys(operation_attributes, ws.device_index);
    const std::vector<RandCoreWork> layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const std::uint32_t key = rand_key_for_core(operation_attributes, keys, i);

        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_idx, core);
        compute_args[0] = key;
        compute_args[1] = lower_bound_bits;
        compute_args[2] = upper_bound_bits;
        compute_args[3] = tile_offset;
        compute_args[4] = units_per_core;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx, core);
        writer_args[0] = out_addr;
        writer_args[1] = tile_offset;
        writer_args[2] = units_per_core;
        if (tensor_args.state.has_value()) {
            writer_args[3] = tensor_args.state->buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::rand
