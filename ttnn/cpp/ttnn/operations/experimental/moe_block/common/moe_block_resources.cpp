// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_resources.hpp"

#include <algorithm>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/creation/creation.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace resources_detail {

constexpr uint32_t SEM_BYTES = 16;  // a semaphore a 16 B word (the NoC's write granule)
constexpr uint32_t SEM_SLOTS = 48;  // packed pool slots (persist_l1.SEM_SLOTS)

bool torus_x_of(tt::tt_fabric::FabricConfig cfg) {
    return cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X || cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
}
bool torus_y_of(tt::tt_fabric::FabricConfig cfg) {
    return cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y || cfg == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
}

uint8_t dir_mask(const std::vector<Dir>& dirs) {
    uint8_t m = 0;
    for (Dir d : dirs) {
        m |= static_cast<uint8_t>(1u << static_cast<uint32_t>(d));
    }
    return m;
}
constexpr uint8_t ALL_DIRS = 0xF;

// The plan with the ports of the directions outside `mask` removed.
MeshPlan restrict_dirs(const MeshPlan& plan, uint8_t mask) {
    MeshPlan out = plan;
    for (auto& [chip, ports] : out.chips) {
        for (Dir d : kDirs) {
            if ((mask & (1u << static_cast<uint32_t>(d))) == 0) {
                ports[static_cast<size_t>(d)].reset();
            }
        }
    }
    return out;
}

// The planners route a torus tie (a destination half a ring away) by `tie`; the fabric's routing tables must send it
// the same way, or a planned link carries rows it was not sized for. Checked on each wrapped even ring, from a source
// of each parity (the rules differ by parity), by comparing the fabric's first hop toward the tie with its first hop
// toward the neighbour the rule picks.
void check_tie_rule(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const MeshTopology& topo, FabricTieRule tie) {
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
            const auto want = first_hop(src, dst, topo, tie);
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
                "moe_block: config.tie_rule = {} disagrees with the fabric: a tie from chip ({}, {}) to ({}, {}) goes "
                "{} by the rule but the other way on the fabric. Main's routing tables need FabricTieRule::NorthEast; "
                "SourceParity needs the fabric's tie-break branch.",
                tie == FabricTieRule::NorthEast ? "NorthEast" : "SourceParity",
                src.r,
                src.c,
                dst.r,
                dst.c,
                dir_name(*want));
        }
    }
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
    dims_.validate();
    config_.validate();
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
    resources_detail::check_tie_rule(mesh_device, plan_.topo, config_.tie_rule);
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

const PortCores& MoEBlockResources::ports(
    std::optional<PortPlacement> mode, const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed) {
    return ports(std::vector<Dir>(kDirs.begin(), kDirs.end()), mode, allowed);
}

const PortCores& MoEBlockResources::ports(
    const std::vector<Dir>& dirs,
    std::optional<PortPlacement> mode,
    const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed) {
    check_alive();
    const uint8_t mask = resources_detail::dir_mask(dirs);
    const auto m = mode.value_or(config_.combine_tuning.place);
    std::vector<std::pair<size_t, size_t>> cells;
    if (allowed) {
        for (const auto& c : *allowed) {
            cells.emplace_back(c.y, c.x);
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }
    auto key = std::make_tuple(mask, m, std::move(cells));
    auto it = ports_.find(key);
    if (it == ports_.end()) {
        const auto placed = mask == resources_detail::ALL_DIRS
                                ? place_ports(*mesh_device_, plan_, m, allowed)
                                : place_ports(*mesh_device_, resources_detail::restrict_dirs(plan_, mask), m, allowed);
        it = ports_.emplace(std::move(key), placed).first;
    }
    return it->second;
}

const Tensor& MoEBlockResources::persistent_tensor(
    const std::string& name, const tt::tt_metal::TensorSpec& spec) {
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

const Tensor& MoEBlockResources::persistent_words(const std::string& name, const CoreRangeSet& cores, uint32_t words) {
    const uint32_t w = (words + 3) / 4 * 4;  // 16 B rows
    const uint32_t n = cores.num_cores();
    const MemoryConfig mc{
        TensorMemoryLayout::HEIGHT_SHARDED,
        persistent_buffer_type(),
        tt::tt_metal::ShardSpec(cores, {1, w}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({n, w}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mc));
    return persistent_tensor(name, spec);
}

const GlobalSemaphore& MoEBlockResources::global_semaphore(
    const std::string& name, const CoreRangeSet& cores, uint32_t initial_value) {
    check_alive();
    if (auto it = semaphores_.find(name); it != semaphores_.end()) {
        return it->second;
    }
    check_not_frozen(name);
    auto sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device_, cores, initial_value, persistent_buffer_type());
    return semaphores_.emplace(name, std::move(sem)).first->second;
}

uint32_t MoEBlockResources::semaphore_address(const std::string& name) {
    check_alive();
    if (!config_.persist_l1_small) {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        const CoreRangeSet all(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        return static_cast<uint32_t>(
            ttnn::global_semaphore::get_global_semaphore_address(global_semaphore("sem/" + name, all, 0)));
    }
    if (auto it = packed_slots_.find(name); it != packed_slots_.end()) {
        return static_cast<uint32_t>(packed_pool_->buffer()->address()) + resources_detail::SEM_BYTES * it->second;
    }
    check_not_frozen(name);
    if (!packed_pool_) {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        const CoreRangeSet all(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        packed_pool_ =
            persistent_words("sem/packed_pool", all, resources_detail::SEM_SLOTS * resources_detail::SEM_BYTES / 4);
    }
    const uint32_t slot = packed_slots_.size();
    TT_FATAL(
        slot < resources_detail::SEM_SLOTS, "moe_block: all {} packed semaphores taken", resources_detail::SEM_SLOTS);
    packed_slots_.emplace(name, slot);
    return static_cast<uint32_t>(packed_pool_->buffer()->address()) + resources_detail::SEM_BYTES * slot;
}

void MoEBlockResources::attach(const std::string& name, std::shared_ptr<void> owner) {
    check_alive();
    TT_FATAL(!attachments_.contains(name), "moe_block: attachment '{}' already exists", name);
    attachments_.emplace(name, std::move(owner));
}

bool MoEBlockResources::has(const std::string& name) const {
    return tensors_.contains(name) || semaphores_.contains(name) || packed_slots_.contains(name) ||
           plans_.contains(name) || attachments_.contains(name);
}

std::vector<std::string> MoEBlockResources::names() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : tensors_) {
        out.push_back(k);
    }
    for (const auto& [k, v] : semaphores_) {
        out.push_back(k);
    }
    for (const auto& [k, v] : packed_slots_) {
        out.push_back(k);
    }
    for (const auto& [k, v] : plans_) {
        out.push_back(k);
    }
    for (const auto& [k, v] : attachments_) {
        out.push_back(k);
    }
    std::sort(out.begin(), out.end());
    return out;
}

void MoEBlockResources::release() {
    if (mesh_device_ == nullptr) {
        return;
    }
    attachments_.clear();  // (first: e.g. DRISC senders stop before their buffers go)
    plans_.clear();
    packed_slots_.clear();
    packed_pool_.reset();
    semaphores_.clear();
    for (auto& [name, t] : tensors_) {
        t.deallocate(true);
    }
    tensors_.clear();
    tensor_specs_.clear();
    ports_.clear();
    mesh_device_ = nullptr;
}

void MoEBlockResources::check_not_frozen(const std::string& name) const {
    TT_FATAL(!frozen_, "moe_block: resources are frozen; '{}' was not created before freeze()", name);
}

void MoEBlockResources::check_alive() const {
    TT_FATAL(mesh_device_ != nullptr, "moe_block: resources were released");
}

}  // namespace ttnn::operations::experimental::moe_block
