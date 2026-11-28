// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fabric_reduce_scatter.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/fabric_reduce_scatter_device_operation.hpp"
#include "device/fabric_reduce_scatter_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn {

namespace {

namespace frs = operations::fabric_reduce_scatter;

// The message is formatted only when the check fails: a passing call builds no string.
template <typename... Args>
void refuse_assert(bool ok, fmt::format_string<Args...> format, Args&&... args) {
    if (!ok) {
        throw frs::AssertionRefusal(fmt::format(format, std::forward<Args>(args)...));
    }
}

frs::Scheme parse_scheme(const std::string& scheme) {
    if (scheme == "ring") {
        return frs::Scheme::RING;
    }
    if (scheme == "dual_cycles") {
        return frs::Scheme::DUAL_CYCLES;
    }
    throw std::invalid_argument(fmt::format("fabric_all_gather: unknown scheme '{}'", scheme));
}

// The Python compares with "auto"; every other placement string falls back to any free core.
frs::Placement parse_placement(const std::string& placement) {
    return placement == "auto" ? frs::Placement::AUTO : frs::Placement::SIMPLE;
}

void check_dram_interleaved(const Tensor& t) {
    const auto& mc = t.memory_config();
    if (mc.buffer_type() != BufferType::DRAM || mc.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw std::invalid_argument("fabric_all_gather: input must be DRAM interleaved");
    }
    if (t.layout() == Layout::TILE) {
        const auto& logical = t.logical_shape();
        const auto& padded = t.padded_shape();
        if (logical[-1] != padded[-1] || logical[-2] != padded[-2]) {
            throw std::invalid_argument(
                fmt::format("fabric_all_gather: a TILE input must be tile-aligned (got {})", logical));
        }
    }
}

// The generic op's descriptor LRU held at most this many entries.
constexpr size_t kMaxDescriptorEntries = 512;

std::vector<FabricReduceScatterDescriptorEntry>& descriptor_entries() {
    static std::vector<FabricReduceScatterDescriptorEntry> entries;
    return entries;
}

void record_descriptor_entry(const FabricReduceScatterDescriptorEntry& entry) {
    auto& entries = descriptor_entries();
    if (std::find(entries.begin(), entries.end(), entry) != entries.end()) {
        return;
    }
    entries.push_back(entry);
    if (entries.size() > kMaxDescriptorEntries) {
        entries.erase(entries.begin());
    }
}

// The scratch of one (mesh, group size, block, dtype, dim), allocated on its first call and kept, as the Python route's
// _SCRATCH_CACHE did (NC.2): a call then allocates nothing but its output. Safe to share between calls: every chunk a
// neighbour writes into a chip's scratch is read by that chip's own cores before its program ends, and the ready fence
// keeps a neighbour's next call from writing before this chip has started it.
const Tensor& cached_scratch(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const frs::FabricReduceScatterParams& params,
    const Tensor& input) {
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, DataType, uint32_t>;
    static ttsl::Indestructible<std::map<Key, Tensor>> cache;
    const auto& shape = input.logical_shape();
    const Key key{
        static_cast<uint32_t>(mesh_device->id()), params.group_size, shape[2], shape[3], input.dtype(), params.dim};
    auto it = cache.get().find(key);
    if (it == cache.get().end()) {
        it = cache.get()
                 .emplace(
                     key,
                     create_device_tensor(
                         frs::FabricReduceScatterDeviceOperation::scratch_spec(params, input), mesh_device))
                 .first;
    }
    return it->second;
}

}  // namespace

Tensor fabric_reduce_scatter(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    tt::tt_fabric::Topology topology,
    uint32_t num_links,
    const std::string& scheme,
    const std::optional<Tensor>& output,
    const std::string& placement,
    int32_t dim,
    bool exact_fp32_add,
    bool cache_descriptors,
    bool program_hash,
    bool prepared,
    const std::optional<Tensor>& scratch_tensor) {
    refuse_assert(!prepared || program_hash, "prepared needs program_hash=True");
    refuse_assert(!program_hash || cache_descriptors, "program_hash needs cache_descriptors=True");

    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "fabric_reduce_scatter: input must be a device tensor");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "fabric_reduce_scatter: needs a 2D mesh, got {}", mesh_shape);

    frs::FabricReduceScatterParams params{
        .cluster_axis = cluster_axis,
        .topology = topology,
        .num_links = num_links,
        .scheme = parse_scheme(scheme),
        .placement = parse_placement(placement),
        .mesh_rows = mesh_shape[0],
        .mesh_cols = mesh_shape[1],
    };
    // Per-mesh state (plan and counters) first, as the Python's prepare(): before any tensor of this call.
    const frs::MeshState& state = frs::get_mesh_state(*mesh_device, frs::plan_inputs(params));
    const uint32_t G = state.plan->group_size;
    params.group_size = G;

    check_dram_interleaved(input_tensor);
    const auto& shape = input_tensor.logical_shape();
    refuse_assert(
        input_tensor.layout() == Layout::TILE && shape.rank() == 4 && shape[0] == 1 && shape[1] == 1, "{}", shape);
    refuse_assert(dim == 2 || dim == 3 || dim == -1 || dim == -2, "dim {}: blocks of rows (2) or of columns (3)", dim);
    params.dim = static_cast<uint32_t>(((dim % 4) + 4) % 4);
    params.exact_fp32_add = exact_fp32_add;
    if (exact_fp32_add) {
        refuse_assert(
            input_tensor.dtype() == DataType::FLOAT32,
            "exact_fp32_add needs FLOAT32 partials, got {}",
            input_tensor.dtype());
    }
    const uint32_t tile_h = tt::constants::TILE_HEIGHT;
    const uint32_t tile_w = tt::constants::TILE_WIDTH;
    uint32_t sb = 0;
    uint32_t w = 0;
    if (params.dim == 2) {
        refuse_assert(shape[2] % (tile_h * G) == 0, "rows {} must split into {} tile-aligned blocks", shape[2], G);
        sb = shape[2] / G;
        w = shape[3];
    } else {
        refuse_assert(shape[2] % tile_h == 0, "rows {} must be tile-aligned", shape[2]);
        refuse_assert(shape[3] % (tile_w * G) == 0, "columns {} must split into {} tile-aligned blocks", shape[3], G);
        sb = shape[2];
        w = shape[3] / G;
    }
    if (output.has_value()) {
        const auto& os = output->logical_shape();
        refuse_assert(
            os == Shape({1, 1, sb, w}) && output->dtype() == input_tensor.dtype(), "({}, [1, 1, {}, {}])", os, sb, w);
    }

    // Scratch: the caller's, or the cached one of this configuration (cached_scratch).
    std::optional<Tensor> scratch = scratch_tensor;
    if (!scratch.has_value()) {
        scratch = cached_scratch(mesh_device, params, input_tensor);
    }
    Tensor result = ttnn::prim::fabric_reduce_scatter(input_tensor, output, scratch, params);
    if (program_hash) {
        record_descriptor_entry(FabricReduceScatterDescriptorEntry{
            .prepared = prepared,
            .input_address = input_tensor.buffer()->address(),  // smuggled-rta-ok: inspection record, not an arg
            .scratch_address = scratch->buffer()->address(),    // smuggled-rta-ok: inspection record, not an arg
            .output_address = result.buffer()->address(),       // smuggled-rta-ok: inspection record, not an arg
            .num_programs = static_cast<uint32_t>(mesh_shape.mesh_size()),
        });
    }
    return result;
}

const std::vector<FabricReduceScatterDescriptorEntry>& fabric_reduce_scatter_descriptor_entries() {
    return descriptor_entries();
}

Tensor fabric_reduce_scatter_scratch_tensor(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    tt::tt_fabric::Topology topology,
    uint32_t num_links,
    const std::string& scheme,
    const std::string& placement,
    int32_t dim) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "fabric_reduce_scatter_scratch_tensor: input must be a device tensor");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "fabric_reduce_scatter_scratch_tensor: needs a 2D mesh, got {}", mesh_shape);
    frs::FabricReduceScatterParams params{
        .cluster_axis = cluster_axis,
        .topology = topology,
        .num_links = num_links,
        .scheme = parse_scheme(scheme),
        .placement = parse_placement(placement),
        .mesh_rows = mesh_shape[0],
        .mesh_cols = mesh_shape[1],
    };
    const frs::MeshState& state = frs::get_mesh_state(*mesh_device, frs::plan_inputs(params));
    params.group_size = state.plan->group_size;
    params.dim = static_cast<uint32_t>(((dim % 4) + 4) % 4);
    return create_device_tensor(
        frs::FabricReduceScatterDeviceOperation::scratch_spec(params, input_tensor), mesh_device);
}

void fabric_reduce_scatter_prepare(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    std::optional<uint32_t> cluster_axis,
    tt::tt_fabric::Topology topology,
    uint32_t num_links,
    const std::string& scheme,
    const std::string& placement) {
    const auto mesh_shape = mesh_device.shape();
    TT_FATAL(mesh_shape.dims() == 2, "fabric_reduce_scatter prepare: needs a 2D mesh, got {}", mesh_shape);
    frs::FabricReduceScatterParams params{
        .cluster_axis = cluster_axis,
        .topology = topology,
        .num_links = num_links,
        .scheme = parse_scheme(scheme),
        .placement = parse_placement(placement),
    };
    frs::get_mesh_state(mesh_device, frs::plan_inputs(params));
}

}  // namespace ttnn
