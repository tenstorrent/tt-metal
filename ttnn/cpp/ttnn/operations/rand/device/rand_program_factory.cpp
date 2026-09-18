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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/threefry_key.hpp"

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
    // Every core of the grid, participating or not. With a state tensor bound the writer runs on all of
    // them so each epoch page advances once per call: lockstep pages make the epoch a property of the
    // call rather than of the core, which is what keeps Threefry's output independent of the work split.
    CoreRangeSet grid_cores;
    std::vector<CoreCoord> all_grid_cores;
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
    // grid_to_cores walks the grid in a fixed order, so `cores` is a prefix of `all_grid_cores` and a
    // core's index -- its state page id -- is the same in both.
    const std::uint32_t num_grid_cores = grid.x * grid.y;
    CoreRangeSet grid_cores(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    auto all_grid_cores = grid_to_cores(num_grid_cores, grid.x, grid.y);

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
        device_index,
        std::move(grid_cores),
        std::move(all_grid_cores)};
}

// Per-dispatch keys, derived by evaluating Threefry-2x32 at (shard, core). seed == 0 draws fresh host
// randomness on every dispatch (cache hits included).
struct RandKeys {
    std::uint32_t seed;  // 0 means unseeded: keys come from the host RNG, not the PRF
    std::uint32_t device_index;
    std::uint32_t stream;  // Threefry key0; core-independent, so its output ignores the core grid
};
RandKeys rand_keys(const RandDeviceOperation::operation_attributes_t& attrs, std::uint32_t device_index) {
    if (attrs.seed == 0) {
        // Only Threefry reads `stream`; the LFSR path draws per core below, so do not burn a draw for it.
        const std::uint32_t stream = attrs.generator == RandGenerator::THREEFRY ? get_random_seed() : 0u;
        return {0u, device_index, stream};
    }
    return {attrs.seed, device_index, compute_kernel_lib::rand_stream_key(attrs.seed, device_index)};
}
std::uint32_t rand_key_for_core(const RandDeviceOperation::operation_attributes_t& attrs, const RandKeys& keys, int i) {
    if (attrs.generator == RandGenerator::THREEFRY) {
        return keys.stream;
    }
    if (keys.seed == 0) {
        return get_random_seed();
    }
    return compute_kernel_lib::rand_core_key(keys.seed, keys.device_index, static_cast<std::uint32_t>(i));
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
            .core_ranges = ws.grid_cores,
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
    // Idle cores carry no tiles but still bump their epoch page, so the writer spans the grid whenever a
    // state tensor is bound. Compute stays on the participating cores only.
    writer_desc.core_ranges = has_state ? ws.grid_cores : all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(has_state ? ws.all_grid_cores.size() : num_cores_total);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        output_cb_id,
        static_cast<std::uint32_t>(operation_attributes.generator),
        has_state ? 1u : 0u,
        state_cb_id,
        1u /*position salt*/};
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

    // Zero-tile writers: the output loop does not run, so these only read their epoch page and store it
    // back incremented.
    if (has_state) {
        for (std::size_t i = layout.size(); i < ws.all_grid_cores.size(); ++i) {
            writer_desc.emplace_runtime_args(
                ws.all_grid_cores[i],
                {output.buffer(), 0u, 0u, tensor_args.state->buffer(), static_cast<std::uint32_t>(i)});
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

    // The zero-tile writers on the idle cores hold the same two buffer addresses and must be re-patched too.
    if (tensor_args.state.has_value()) {
        const std::uint32_t state_addr = tensor_args.state->buffer()->address();
        for (std::size_t i = layout.size(); i < ws.all_grid_cores.size(); ++i) {
            auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx, ws.all_grid_cores[i]);
            writer_args[0] = out_addr;
            writer_args[1] = 0;
            writer_args[2] = 0;
            writer_args[3] = state_addr;
        }
    }
}

}  // namespace ttnn::operations::rand
