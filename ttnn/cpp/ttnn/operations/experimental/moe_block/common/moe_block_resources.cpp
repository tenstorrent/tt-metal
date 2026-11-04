// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_resources.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace resources_detail {

bool torus_x_of(tt::tt_fabric::FabricConfig cfg) {
    return cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X ||
           cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
}
bool torus_y_of(tt::tt_fabric::FabricConfig cfg) {
    return cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y ||
           cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
}

// The planners route a torus tie (a destination half a ring away) by the source's parity (mesh_routing.hpp route); the
// fabric's routing tables must send it the same way, or a planned link carries rows it was not sized for (a fabric
// older than main's parity split does not). Checked on each wrapped even ring, from a source of each parity, by
// comparing the fabric's first hop toward the tie with its first hop toward the neighbour the rule picks.
void check_fabric_ties(const tt::tt_metal::distributed::MeshDevice& mesh_device, const MeshTopology& topo) {
    const auto node = [&](ChipCoord c) {
        return mesh_device.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(c.r, c.c));
    };
    for (const bool x_axis : {false, true}) {
        const uint32_t n = x_axis ? topo.C : topo.R;
        if (!(x_axis ? topo.wrap_x : topo.wrap_y) || n % 2 != 0) {
            continue;
        }
        for (uint32_t pos = 0; pos < 2; ++pos) {
            const ChipCoord src = x_axis ? ChipCoord{0, pos} : ChipCoord{pos, 0};
            const ChipCoord dst = x_axis ? ChipCoord{0, pos + n / 2} : ChipCoord{pos + n / 2, 0};
            const auto want = first_hop(src, dst, topo);
            TT_FATAL(want.has_value(), "moe_block: no first hop for a tie");
            const auto peer = topo.neighbor(src, *want);
            TT_FATAL(peer.has_value(), "moe_block: no neighbour {} of a tie's source", dir_name(*want));
            const auto fabric_dir = tt::tt_fabric::get_eth_forwarding_direction(node(src), node(dst));
            const auto rule_dir = tt::tt_fabric::get_eth_forwarding_direction(node(src), node(*peer));
            if (!fabric_dir || !rule_dir) {
                continue;  // no fabric route to compare against
            }
            TT_FATAL(
                *fabric_dir == *rule_dir,
                "moe_block: the fabric routes a torus tie from chip ({}, {}) to ({}, {}) the other way than {} (the "
                "source-parity tie rule the planners use): the fabric's routing tables predate main's parity split",
                src.r,
                src.c,
                dst.r,
                dst.c,
                dir_name(*want));
        }
    }
}

constexpr uint32_t WORDS_ALIGN = 16;  // a words row (persistent_words rounds a core's words up to 16 B)

// The spec of `words` uint32 a core (rounded up to 16 B) HEIGHT_SHARDED over `cores` in `buffer_type`.
tt::tt_metal::TensorSpec words_spec(const CoreRangeSet& cores, uint32_t words, tt::tt_metal::BufferType buffer_type) {
    const uint32_t w = (words + 3) / 4 * 4;  // 16 B rows
    const MemoryConfig mc{
        TensorMemoryLayout::HEIGHT_SHARDED,
        buffer_type,
        tt::tt_metal::ShardSpec(cores, {1, w}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({cores.num_cores(), w}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mc));
}

// A tensor of `spec` over the mesh whose buffer sits at `address` without owning it (a slot of the words slab, which
// owns the memory): its buffer's address() is `address`, so it binds and patches like an allocated tensor.
Tensor view_at(tt::tt_metal::distributed::MeshDevice& mesh, const tt::tt_metal::TensorSpec& spec, uint32_t address) {
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local{
        .page_size = spec.compute_page_size_bytes(),
        .buffer_type = spec.memory_config().buffer_type(),
        .sharding_args = spec.compute_buffer_sharding_args(),
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated{.size = spec.compute_packed_buffer_size_bytes()};
    auto buffer = tt::tt_metal::distributed::MeshBuffer::create(replicated, local, &mesh, address);
    TT_FATAL(buffer->address() == address, "moe_block: a words slot's view is not at its address");
    return Tensor(tt::tt_metal::mesh_tensor_from_buffer_with_topology(
        std::move(*buffer),
        spec,
        tt::tt_metal::TensorTopology::create_fully_replicated_tensor_topology(mesh.shape())));
}

// The lowest offset in [0, capacity) where `bytes` are free on every core of `cores` (`taken`: each core's occupied
// [offset, end) ranges), or nullopt. The lowest feasible offset is 0 or the end of a range on one of these cores (any
// other feasible offset can slide down to one), so only those are tried. Overlap is checked per core: sets that share
// no core may share bytes, and three pairwise-intersecting sets never get one offset.
std::optional<uint32_t> first_fit(
    const std::map<tt::tt_metal::CoreCoord, std::vector<std::pair<uint32_t, uint32_t>>>& taken,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t bytes,
    uint32_t capacity) {
    std::vector<uint32_t> candidates = {0};
    for (const auto& c : cores) {
        if (auto it = taken.find(c); it != taken.end()) {
            for (const auto& [b, e] : it->second) {
                candidates.push_back(e);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (const uint32_t off : candidates) {
        if (uint64_t(off) + bytes > capacity) {
            break;
        }
        const bool free = std::all_of(cores.begin(), cores.end(), [&](const auto& c) {
            auto it = taken.find(c);
            return it == taken.end() || std::none_of(it->second.begin(), it->second.end(), [&](const auto& r) {
                       return off < r.second && r.first < off + bytes;
                   });
        });
        if (free) {
            return off;
        }
    }
    return std::nullopt;
}

}  // namespace resources_detail

std::shared_ptr<MoEBlockResources> MoEBlockResources::create(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MoEDtypes& dtypes) {
    return std::shared_ptr<MoEBlockResources>(new MoEBlockResources(mesh_device, dims, config, dtypes));
}

MoEBlockResources::MoEBlockResources(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MoEDtypes& dtypes) :
    mesh_device_(&mesh_device), dims_(dims), config_(config), dtypes_(dtypes) {
    dims_.check();
    config_.check();
    fabric_config_ = tt::tt_fabric::GetFabricConfig();
    max_payload_bytes_ = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    plan_ = moe_block::mesh_plan(
        mesh_device,
        config_.num_links,
        resources_detail::torus_x_of(fabric_config_),
        resources_detail::torus_y_of(fabric_config_));
    TT_FATAL(
        dims_.num_experts % plan_.topo.num_chips() == 0,
        "moe_block: {} experts don't divide over {} chips",
        dims_.num_experts,
        plan_.topo.num_chips());
    resources_detail::check_fabric_ties(mesh_device, plan_.topo);
    luts_ = moe_block::port_luts(plan_);
    plan_key_ = ttsl::hash::hash_objects_with_default_seed(
        config_,
        dims_,
        dtypes_,
        plan_.topo.R,
        plan_.topo.C,
        plan_.topo.wrap_x,
        plan_.topo.wrap_y,
        static_cast<uint32_t>(fabric_config_),
        max_payload_bytes_);
}

MoEBlockResources::~MoEBlockResources() { release(); }

tt::tt_metal::distributed::MeshDevice& MoEBlockResources::mesh_device() const {
    check_alive();
    return *mesh_device_;
}

tt::tt_metal::BufferType MoEBlockResources::persistent_buffer_type() const {
    return config_.persist_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
}

const PortCores& MoEBlockResources::ports(const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed) {
    check_alive();
    std::vector<std::pair<size_t, size_t>> cells;
    if (allowed) {
        for (const auto& c : *allowed) {
            cells.emplace_back(c.y, c.x);
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }
    auto it = ports_.find(cells);
    if (it == ports_.end()) {
        it = ports_.emplace(std::move(cells), place_ports(*mesh_device_, plan_, allowed)).first;
    }
    return it->second;
}

const Tensor& MoEBlockResources::persistent_tensor(const std::string& name, const tt::tt_metal::TensorSpec& spec) {
    check_alive();
    if (auto it = tensors_.find(name); it != tensors_.end()) {
        TT_FATAL(tensor_specs_.at(name) == spec, "moe_block: persistent tensor '{}' requested with another spec", name);
        return it->second;
    }
    check_not_frozen(name);
    auto t = ttnn::zeros(
        spec.logical_shape(), spec.data_type(), spec.layout(), std::ref(*mesh_device_), spec.memory_config());
    tensor_specs_.emplace(name, spec);
    return tensors_.emplace(name, std::move(t)).first->second;
}

bool MoEBlockResources::packs_words() const { return config_.persist_l1_small && config_.words_slab_bytes > 0; }

std::optional<uint32_t> MoEBlockResources::place_words(
    const std::vector<tt::tt_metal::CoreCoord>& cores, uint32_t bytes) {
    if (!words_slab_) {
        // (one shard a worker core, of the configured bytes - at most the largest free L1_SMALL block a bank now)
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        const CoreRangeSet all(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        const auto stats = mesh_device_->allocator()->get_statistics(tt::tt_metal::BufferType::L1_SMALL);
        const uint32_t room = static_cast<uint32_t>(
            std::min<size_t>(stats.largest_free_block_bytes, config_.words_slab_bytes) / resources_detail::WORDS_ALIGN *
            resources_detail::WORDS_ALIGN);
        if (room == 0) {
            return std::nullopt;
        }
        if (room < config_.words_slab_bytes) {
            log_warning(
                tt::LogOp,
                "moe_block: L1_SMALL holds {} B a bank for the persistent words slab, the config {} B: word sets "
                "that do not fit get their own L1_SMALL tensors (open the mesh with more L1_SMALL)",
                room,
                config_.words_slab_bytes);
        }
        words_slab_capacity_ = room;
        const auto spec = resources_detail::words_spec(all, room / 4, tt::tt_metal::BufferType::L1_SMALL);
        words_slab_ = ttnn::zeros(
            spec.logical_shape(), spec.data_type(), spec.layout(), std::ref(*mesh_device_), spec.memory_config());
        log_info(
            tt::LogOp,
            "moe_block: persistent words slab {} B a core at L1_SMALL {} (config {} B)",
            room,
            words_slab_->buffer()->address(),
            config_.words_slab_bytes);
    }
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    if (std::any_of(cores.begin(), cores.end(), [&](const auto& c) { return c.x >= grid.x || c.y >= grid.y; })) {
        return std::nullopt;  // (a core the slab does not cover)
    }
    const auto off = resources_detail::first_fit(words_slab_taken_, cores, bytes, words_slab_capacity_);
    if (off) {
        for (const auto& c : cores) {
            words_slab_taken_[c].emplace_back(*off, *off + bytes);
        }
        words_slab_used_ = std::max(words_slab_used_, *off + bytes);
    }
    return off;
}

const Tensor& MoEBlockResources::persistent_words(const std::string& name, const CoreRangeSet& cores, uint32_t words) {
    check_alive();
    const auto spec = resources_detail::words_spec(cores, words, persistent_buffer_type());
    const uint32_t bytes = spec.logical_shape()[-1] * 4;
    if (auto it = tensors_.find(name); it != tensors_.end()) {
        TT_FATAL(tensor_specs_.at(name) == spec, "moe_block: persistent words '{}' requested with another spec", name);
        return it->second;
    }
    check_not_frozen(name);
    std::optional<uint32_t> off;
    if (packs_words()) {
        off = place_words(corerange_to_cores(cores), bytes);
        if (!off) {
            log_warning(
                tt::LogOp,
                "moe_block: persistent words '{}' ({} B on {} cores) do not fit the {} B slab ({} B used): own L1_SMALL "
                "tensor",
                name,
                bytes,
                cores.num_cores(),
                words_slab_capacity_,
                words_slab_used_);
        }
    }
    if (!off) {
        return persistent_tensor(name, spec);
    }
    // (a view at the slot; zeroed on its own cores only - the slab's other slots are live)
    const uint32_t address = static_cast<uint32_t>(words_slab_->buffer()->address()) + *off;
    Tensor view = resources_detail::view_at(*mesh_device_, spec, address);
    const Tensor zeros = ttnn::zeros(spec.logical_shape(), spec.data_type(), spec.layout());
    ttnn::copy_to_device(zeros, view);
    TT_FATAL(view.buffer()->address() == address, "moe_block: persistent words '{}' moved off their slot", name);
    tensor_specs_.emplace(name, spec);
    return tensors_.emplace(name, std::move(view)).first->second;
}

void MoEBlockResources::release() {
    if (mesh_device_ == nullptr) {
        return;
    }
    plans_.clear();
    for (auto& [name, t] : tensors_) {
        t.deallocate(true);  // (a packed words slot: a view - marks it released, frees nothing)
    }
    tensors_.clear();
    tensor_specs_.clear();
    if (words_slab_) {
        words_slab_->deallocate(true);  // (after every view on it)
        words_slab_.reset();
    }
    words_slab_taken_.clear();
    words_slab_capacity_ = 0;
    words_slab_used_ = 0;
    ports_.clear();
    mesh_device_ = nullptr;
}

void MoEBlockResources::check_not_frozen(const std::string& name) const {
    TT_FATAL(!frozen_, "moe_block: resources are frozen; '{}' was not created before freeze()", name);
}

void MoEBlockResources::check_alive() const { TT_FATAL(mesh_device_ != nullptr, "moe_block: resources were released"); }

}  // namespace ttnn::operations::experimental::moe_block
