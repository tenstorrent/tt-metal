// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MoEBlockResources: everything the MoE block's ops share across calls and across ops - the persistent L1 / DRAM
// words and tensors, the port placement, the plans - built once per (mesh, model, config).
//
// Ownership and lifetime
//  * Created by the caller (Python MoEBlock, a test) with MoEBlockResources::create(mesh, dims, config, dtypes), after
//    the mesh is open and the fabric configured, and BEFORE any per-call allocation (persistent L1 then sits at the
//    top of the allocator and never fragments a bank under a per-call buffer). Held by std::shared_ptr; Python holds
//    it.
//  * The resources keep a reference to the mesh device; the caller must drop them (or call release()) before the
//    mesh is closed. release() frees every buffer.
//  * Ops receive `MoEBlockResources&` in their host entry point (the ttnn-facing function), get-or-create what they
//    need there, and pass the persistent tensors to the device operation as ordinary tensor args. The device op's
//    attributes never hold the resources object. Ops fold plan_key() into compute_program_hash.
//  * Trace: every slot must exist before trace capture. Run one forward, then freeze(): a get-or-create of a missing
//    slot then throws instead of allocating inside a capture. Persistent counters are cumulative across calls, so
//    replays need no host-side reset.
//
// Sync words: persistent_words() - a HEIGHT_SHARDED uint32 tensor over the cores that use it, passed to the prim as a
// tensor arg, so its address binds like any buffer (BufferBinding) and a program-cache hit patches it with no op code.
//
// Per-op slots: each op names its slots "<op>/<name>" and creates them from its own directory through the generic
// API below; nothing op-specific is declared here.

#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "mesh_routing.hpp"
#include "moe_block_config.hpp"
#include "port_placement.hpp"

namespace ttnn::operations::experimental::moe_block {

class MoEBlockResources {
public:
    // Throws if the fabric routes a torus tie other than by the source's parity (mesh_routing.hpp route).
    static std::shared_ptr<MoEBlockResources> create(
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        const MoEModelDims& dims,
        const MoEBlockConfig& config,
        const MoEDtypes& dtypes = MoEDtypes{});

    MoEBlockResources(const MoEBlockResources&) = delete;
    MoEBlockResources& operator=(const MoEBlockResources&) = delete;
    ~MoEBlockResources();

    tt::tt_metal::distributed::MeshDevice& mesh_device() const;
    const MoEModelDims& dims() const { return dims_; }
    const MoEBlockConfig& config() const { return config_; }
    const MoEDtypes& dtypes() const { return dtypes_; }
    const MeshTopology& topology() const { return plan_.topo; }
    const MeshPlan& mesh_plan() const { return plan_; }
    const PortLuts& port_luts() const { return luts_; }
    tt::tt_fabric::FabricConfig fabric_config() const { return fabric_config_; }
    size_t fabric_max_payload_bytes() const { return max_payload_bytes_; }

    // L1, or L1_SMALL with config.persist_l1_small.
    tt::tt_metal::BufferType persistent_buffer_type() const;

    // Structural key of everything the resources were built from (config, dims, dtypes, mesh shape, wrap, fabric
    // config, max payload). Ops hash it instead of re-hashing the config; it holds no address.
    ttsl::hash::hash_t plan_key() const { return plan_key_; }

    // Port cores (place_ports), memoized per allowed cells.
    const PortCores& ports(const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed = std::nullopt);

    // ---- persistent device memory (get-or-create by name; zero-filled at creation; same address for the lifetime)
    // A tensor of exactly this spec (DRAM or L1; replicated over the mesh). A second request under the same name must
    // pass the same spec.
    const Tensor& persistent_tensor(const std::string& name, const tt::tt_metal::TensorSpec& spec);
    // uint32 words per core (the standard for sync words, see above): [len(cores), words] HEIGHT_SHARDED over `cores`
    // (row-major), one shard of `words` (rounded up to 16 B) per core, in persistent_buffer_type().
    // Packed (persist_l1_small and config.words_slab_bytes > 0): the tensor is a non-owning view at a slot of one
    // L1_SMALL slab (all worker cores, words_slab_bytes a core) - its buffer's address IS the slot's address, so it binds
    // and patches like any tensor. A slot is placed first-fit at the lowest offset free on every core of `cores` (word
    // sets on disjoint core sets share bytes; the L1_SMALL allocator would reserve each set's bytes on every core),
    // zeroed on its own cores at creation, and never moves or is reused until release(). `cores` must be every core
    // that touches the words on any chip (the union over the chips). A set the slab cannot hold gets its own tensor.
    const Tensor& persistent_words(const std::string& name, const CoreRangeSet& cores, uint32_t words);

    // ---- host-side plans and owned objects
    // Get-or-create a host plan of type T (computed by `make` on first use).
    template <typename T, typename F>
    const T& plan(const std::string& name, F&& make) {
        return *plan_ptr<T>(name, std::forward<F>(make));
    }
    // The same plan as a shared pointer, for an op that keeps it past the resources' release() (e.g. in a cached
    // program's shared variables).
    template <typename T, typename F>
    std::shared_ptr<const T> plan_ptr(const std::string& name, F&& make) {
        auto it = plans_.find(name);
        if (it == plans_.end()) {
            check_not_frozen(name);
            it = plans_.emplace(name, std::make_shared<const T>(std::forward<F>(make)())).first;
        }
        return std::any_cast<std::shared_ptr<const T>>(it->second);
    }

    // After freeze(), creating a missing slot throws (trace capture / steady state).
    void freeze() { frozen_ = true; }
    bool frozen() const { return frozen_; }
    // Frees every buffer and plan; the object is unusable afterwards.
    void release();

private:
    MoEBlockResources(
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        const MoEModelDims& dims,
        const MoEBlockConfig& config,
        const MoEDtypes& dtypes);
    void check_not_frozen(const std::string& name) const;
    void check_alive() const;
    bool packs_words() const;
    // A free offset in the slab for `bytes` on every core of `cores` (creating the slab at first use), nullopt if none.
    std::optional<uint32_t> place_words(const std::vector<tt::tt_metal::CoreCoord>& cores, uint32_t bytes);

    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    MoEModelDims dims_;
    MoEBlockConfig config_;
    MoEDtypes dtypes_;
    tt::tt_fabric::FabricConfig fabric_config_;
    size_t max_payload_bytes_ = 0;
    MeshPlan plan_;
    PortLuts luts_;
    ttsl::hash::hash_t plan_key_ = 0;
    bool frozen_ = false;

    // allowed cells -> placement
    std::map<std::vector<std::pair<size_t, size_t>>, PortCores> ports_;
    std::map<std::string, Tensor> tensors_;
    std::map<std::string, tt::tt_metal::TensorSpec> tensor_specs_;
    std::map<std::string, std::any> plans_;
    // persistent_words: the slab and each core's occupied [offset, end) ranges in it
    std::optional<Tensor> words_slab_;
    uint32_t words_slab_capacity_ = 0;
    uint32_t words_slab_used_ = 0;
    std::map<tt::tt_metal::CoreCoord, std::vector<std::pair<uint32_t, uint32_t>>> words_slab_taken_;
};

}  // namespace ttnn::operations::experimental::moe_block
