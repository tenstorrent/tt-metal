// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MoEBlockResources: everything the MoE block's ops share across calls and across ops - the persistent L1 / DRAM
// words, the global semaphores, the port placement, the plans - built once per (mesh, model, config).
//
// Ownership and lifetime
//  * Created by the caller (Python MoEBlock, a test) with MoEBlockResources::create(mesh, dims, config), after the
//    mesh is open and the fabric configured, and BEFORE any per-call allocation (persistent L1 then sits at the top
//    of the allocator and never fragments a bank under a per-call buffer). Held by std::shared_ptr; Python holds it.
//  * The resources keep a reference to the mesh device; the caller must drop them (or call release()) before the
//    mesh is closed. release() frees every buffer, semaphore and attachment (e.g. persistent DRISC senders).
//  * Ops receive `MoEBlockResources&` in their host entry point (the ttnn-facing function), get-or-create what they
//    need there, and pass it to the device operation as ordinary tensor args (persistent tensors) and
//    hash-excluded attributes (GlobalSemaphores / addresses), patched in override_runtime_arguments. The device op's
//    attributes never hold the resources object. Ops fold plan_key() into compute_program_hash.
//  * Trace: every slot must exist before trace capture. Run one forward (or call each op's ensure function), then
//    freeze(): a get-or-create of a missing slot then throws instead of allocating inside a capture. Persistent
//    counters are cumulative across calls, so replays need no host-side reset.
//
// Per-op slots: each op names its slots "<op>/<name>" and creates them from its own directory (an
// `ensure_<op>_resources(MoEBlockResources&, ...)` in <op>/<op>.cpp) through the generic API below; nothing
// op-specific is declared here. Expected needs (from cpp_op_conversion_plan.md section 2):
//   TODO(fused_head): mbox (port mailboxes + flags + fwd slots + zone words), ewords, router slot_words, arr_sems,
//     XGather3 3 + 2L semaphores, RouterStream 2, RoutingStream semaphores; DRAM outputs y/se, rows, xbuf, route,
//     latent; routing plan tensors counts / regions / token_index / table / plan, slots, fslots; head plan.
//   TODO(sparse_dispatch_2d): 4 port semaphores (on the port cells), manifest sizing; placement via ports().
//   TODO(flat_routed_expert): gidx, done; the persistent DRISC senders as an attachment (RAII stop); flat plan.
//   TODO(routed_experts_combine): bar (every core), relay_words, dispatch semaphores (4), hcache (DRAM headers),
//     lut (DRAM per-chip accumulator LUT), out; the combine layout / cells plan.
//   TODO(fused_tail): ready semaphore, words, port_epoch (TailEngine); zrw, z (TailA2A); tail schedule.

#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "mesh_routing.hpp"
#include "moe_block_config.hpp"
#include "port_placement.hpp"

namespace ttnn::operations::experimental::moe_block {

class MoEBlockResources {
public:
    static std::shared_ptr<MoEBlockResources> create(
        tt::tt_metal::distributed::MeshDevice& mesh_device, const MoEModelDims& dims, const MoEBlockConfig& config);

    MoEBlockResources(const MoEBlockResources&) = delete;
    MoEBlockResources& operator=(const MoEBlockResources&) = delete;
    ~MoEBlockResources();

    tt::tt_metal::distributed::MeshDevice& mesh_device() const;
    const MoEModelDims& dims() const { return dims_; }
    const MoEBlockConfig& config() const { return config_; }
    const MeshTopology& topology() const { return plan_.topo; }
    const MeshPlan& mesh_plan() const { return plan_; }
    const PortLuts& port_luts() const { return luts_; }
    tt::tt_fabric::FabricConfig fabric_config() const { return fabric_config_; }
    size_t fabric_max_payload_bytes() const { return max_payload_bytes_; }

    // L1, or L1_SMALL with config.persist_l1_small.
    tt::tt_metal::BufferType persistent_buffer_type() const;

    // Structural key of everything the resources were built from (config, dims, mesh shape, wrap, fabric config, max
    // payload). Ops hash it instead of re-hashing the config; it holds no address.
    ttsl::hash::hash_t plan_key() const { return plan_key_; }

    // Port cores (sparse_all_gather_2d.place_ports), memoized per (mode, allowed cells). Default mode:
    // config.combine_tuning.place.
    const PortCores& ports(
        std::optional<PortPlacement> mode = std::nullopt,
        const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed = std::nullopt);

    // ---- persistent device memory (get-or-create by name; zero-filled at creation; same address for the lifetime)
    // A tensor of exactly this spec (DRAM or L1; replicated over the mesh). A second request under the same name must
    // pass the same spec.
    const Tensor& persistent_tensor(const std::string& name, const tt::tt_metal::TensorSpec& spec);
    // uint32 words per core: [len(cores), words] HEIGHT_SHARDED over `cores` (row-major), one shard of `words`
    // (rounded up to 16 B) per core, in persistent_buffer_type().
    const Tensor& persistent_words(const std::string& name, const CoreRangeSet& cores, uint32_t words);
    // A global semaphore on `cores` (persistent_buffer_type()), created at initial_value.
    const GlobalSemaphore& global_semaphore(const std::string& name, const CoreRangeSet& cores, uint32_t initial_value = 0);
    // A 16 B semaphore word on every worker core: a GlobalSemaphore address, or with persist_l1_small a slot of one
    // packed L1_SMALL pool (persist_l1.PackedSemaphore; zero-initialised). Same address on every core and chip.
    uint32_t semaphore_address(const std::string& name);

    // ---- host-side plans and owned objects
    // Get-or-create a host plan of type T (computed by `make` on first use).
    template <typename T, typename F>
    const T& plan(const std::string& name, F&& make) {
        auto it = plans_.find(name);
        if (it == plans_.end()) {
            check_not_frozen(name);
            it = plans_.emplace(name, std::make_shared<const T>(std::forward<F>(make)())).first;
        }
        return *std::any_cast<std::shared_ptr<const T>>(it->second);
    }
    // Keep `owner` alive until release() (e.g. persistent DRISC senders: their destructor stops them).
    void attach(const std::string& name, std::shared_ptr<void> owner);

    bool has(const std::string& name) const;
    std::vector<std::string> names() const;

    // After freeze(), creating a missing slot throws (trace capture / steady state).
    void freeze() { frozen_ = true; }
    bool frozen() const { return frozen_; }
    // Frees every buffer, semaphore and attachment; the object is unusable afterwards.
    void release();

private:
    MoEBlockResources(
        tt::tt_metal::distributed::MeshDevice& mesh_device, const MoEModelDims& dims, const MoEBlockConfig& config);
    void check_not_frozen(const std::string& name) const;
    void check_alive() const;

    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    MoEModelDims dims_;
    MoEBlockConfig config_;
    tt::tt_fabric::FabricConfig fabric_config_;
    size_t max_payload_bytes_ = 0;
    MeshPlan plan_;
    PortLuts luts_;
    ttsl::hash::hash_t plan_key_ = 0;
    bool frozen_ = false;

    std::map<std::pair<PortPlacement, std::vector<std::pair<size_t, size_t>>>, PortCores> ports_;
    std::map<std::string, Tensor> tensors_;
    std::map<std::string, tt::tt_metal::TensorSpec> tensor_specs_;
    std::map<std::string, GlobalSemaphore> semaphores_;
    std::optional<Tensor> packed_pool_;
    std::map<std::string, uint32_t> packed_slots_;
    std::map<std::string, std::any> plans_;
    std::map<std::string, std::shared_ptr<void>> attachments_;
};

}  // namespace ttnn::operations::experimental::moe_block
