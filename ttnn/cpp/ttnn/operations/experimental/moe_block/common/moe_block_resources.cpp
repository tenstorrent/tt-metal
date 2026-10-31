// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_resources.hpp"

#include <algorithm>

#include <tt-metalium/experimental/fabric/fabric.hpp>
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

}  // namespace resources_detail
using namespace resources_detail;

std::shared_ptr<MoEBlockResources> MoEBlockResources::create(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const MoEModelDims& dims, const MoEBlockConfig& config) {
    return std::shared_ptr<MoEBlockResources>(new MoEBlockResources(mesh_device, dims, config));
}

MoEBlockResources::MoEBlockResources(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const MoEModelDims& dims, const MoEBlockConfig& config) :
    mesh_device_(&mesh_device), dims_(dims), config_(config) {
    dims_.validate();
    config_.validate();
    fabric_config_ = tt::tt_fabric::GetFabricConfig();
    max_payload_bytes_ = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    plan_ = moe_block::mesh_plan(mesh_device, config_.num_links, torus_x_of(fabric_config_), torus_y_of(fabric_config_));
    TT_FATAL(
        dims_.num_experts % plan_.topo.num_chips() == 0,
        "moe_block: {} experts don't divide over {} chips",
        dims_.num_experts,
        plan_.topo.num_chips());
    luts_ = moe_block::port_luts(plan_);
    plan_key_ = ttsl::hash::hash_objects_with_default_seed(
        config_,
        dims_,
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
    check_alive();
    const auto m = mode.value_or(config_.combine_tuning.place);
    std::vector<std::pair<size_t, size_t>> cells;
    if (allowed) {
        for (const auto& c : *allowed) {
            cells.emplace_back(c.y, c.x);
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }
    const auto key = std::make_pair(m, cells);
    auto it = ports_.find(key);
    if (it == ports_.end()) {
        it = ports_.emplace(key, place_ports(*mesh_device_, plan_, m, allowed)).first;
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
        return static_cast<uint32_t>(packed_pool_->buffer()->address()) + SEM_BYTES * it->second;
    }
    check_not_frozen(name);
    if (!packed_pool_) {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        const CoreRangeSet all(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        packed_pool_ = persistent_words("sem/packed_pool", all, SEM_SLOTS * SEM_BYTES / 4);
    }
    const uint32_t slot = packed_slots_.size();
    TT_FATAL(slot < SEM_SLOTS, "moe_block: all {} packed semaphores taken", SEM_SLOTS);
    packed_slots_.emplace(name, slot);
    return static_cast<uint32_t>(packed_pool_->buffer()->address()) + SEM_BYTES * slot;
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
