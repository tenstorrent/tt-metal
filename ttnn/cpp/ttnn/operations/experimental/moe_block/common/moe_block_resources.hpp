// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MoEBlockResources: the state the MoE block's ops share, kept for as long as the block exists.
//
// Why this exists
// ---------------
// The block's ops (head, dispatch, experts + combine, tail) need things that outlive one call and that several ops
// share: small "persistent words" in L1 that kernels use to synchronize across calls, the cores that talk to each
// fabric link ("ports"), the fabric's route to every chip, and plans (layouts, sizes) that are expensive to compute.
// Creating these once per block, instead of per call or per op, keeps calls fast and keeps the ops consistent with
// each other. One object holds them all, built once per (mesh, model, config).
//
// Ownership and lifetime
//  * The caller (the Python MoEBlock, or a test) creates it with MoEBlockResources::create(mesh, dims, config, dtypes)
//    after the mesh is open and the fabric is configured, and before any per-call buffer is allocated. Its persistent
//    L1 then sits at the top of the allocator, where it never splits the free space that per-call buffers use. It is
//    held by a std::shared_ptr; Python holds it.
//  * It keeps a reference to the mesh device. The caller must drop it (or call release(), which frees every buffer)
//    before closing the mesh.
//  * Ops receive `MoEBlockResources&` in their host entry point, get or create what they need there, and pass the
//    persistent tensors to the device operation as ordinary tensor arguments. A device operation never stores the
//    resources object itself.
//  * Program caching: ops add plan_key() to their program hash. The key is unique to this object (it includes a
//    serial number, plus the config, dims, mesh and fabric). Plans place ports and words wherever this object found
//    free L1, so a program built for one block must never be replayed for another block, even one with the same config.
//    (Replaying one did happen when blocks of the same config were built and released in turn, and it hung the device.)
//  * Trace: everything must exist before trace capture, so run one forward first. Persistent counters keep counting
//    across calls, so replays need no reset from the host.
//
// Sync words: persistent_words() is a HEIGHT_SHARDED uint32 tensor over the cores that use it. It is passed to the op
// as a tensor argument, so its address is bound like any buffer and a program-cache hit updates it automatically.
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
    // Asks the fabric its first hop between every pair of chips (first_hops(): the ops' ports follow it).
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
    // the fabric's first hop from every chip to every other (mesh_routing.hpp FirstHops)
    const FirstHops& first_hops() const { return first_hops_; }
    tt::tt_fabric::FabricConfig fabric_config() const { return fabric_config_; }
    size_t fabric_max_payload_bytes() const { return max_payload_bytes_; }

    // Structural key of everything the resources were built from (config, dims, dtypes, mesh shape, wrap, fabric
    // config, max payload). Ops hash it instead of re-hashing the config; it holds no address.
    ttsl::hash::hash_t plan_key() const { return plan_key_; }

    // Port cores (place_ports), memoized per allowed cells (nullopt: the whole grid - a placement of its own; an empty
    // `allowed` is not the whole grid and fails as too few cells).
    const PortCores& ports(const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed = std::nullopt);

    // ---- persistent device memory (get-or-create by name; zero-filled at creation; same address for the lifetime)
    // A tensor of exactly this spec (DRAM or L1; replicated over the mesh). A second request under the same name must
    // pass the same spec.
    const Tensor& persistent_tensor(const std::string& name, const tt::tt_metal::TensorSpec& spec);
    // uint32 words per core (the standard for sync words, see above): [len(cores), words] HEIGHT_SHARDED over `cores`
    // (row-major), one shard of `words` (rounded up to 16 B) per core, in persistent_buffer_type().
    // Packed (persist_l1_small and config.words_slab_bytes > 0): the tensor is a non-owning view at a slot of one
    // L1_SMALL slab (all worker cores, words_slab_bytes a core) - its buffer's address IS the slot's address, so it
    // binds and patches like any tensor. A slot is placed first-fit at the lowest offset free on every core of `cores`
    // (word sets on disjoint core sets share bytes; the L1_SMALL allocator would reserve each set's bytes on every
    // core), zeroed on its own cores at creation, and never moves or is reused until release(). `cores` must be every
    // core that touches the words on any chip (the union over the chips). A set the slab cannot hold gets its own
    // tensor.
    const Tensor& persistent_words(const std::string& name, const CoreRangeSet& cores, uint32_t words);

    // ---- host-side plans and owned objects
    // Get-or-create a host plan of type T (computed by `make` on first use).
    template <typename T, typename F>
    const T& plan(const std::string& name, F&& make) {
        return *plan_ptr<T>(name, std::forward<F>(make));
    }
    // A plan made earlier under `name`, or nullptr (an op reading what another op of the block planned).
    template <typename T>
    const T* find_plan(const std::string& name) const {
        auto it = plans_.find(name);
        return it == plans_.end() ? nullptr : std::any_cast<std::shared_ptr<const T>>(it->second).get();
    }
    // The same plan as a shared pointer, for an op that keeps it past the resources' release() (e.g. in a cached
    // program's shared variables).
    template <typename T, typename F>
    std::shared_ptr<const T> plan_ptr(const std::string& name, F&& make) {
        auto it = plans_.find(name);
        if (it == plans_.end()) {
            it = plans_.emplace(name, std::make_shared<const T>(std::forward<F>(make)())).first;
        }
        return std::any_cast<std::shared_ptr<const T>>(it->second);
    }

    // MoEBlockConfig.chunk_splits = b: the ops plan for one piece from here on. Drops every plan and persistent tensor
    // made so far (MoEBlock picking its splits by planning): only before any op call or weight preparation.
    void set_chunk_splits(uint32_t b);

    // Frees every buffer and plan; the object is unusable afterwards.
    void release();
    // The words slab's bytes a core: its capacity, and the end of the furthest word set placed in it so far.
    uint32_t words_slab_capacity() const { return words_slab_capacity_; }
    uint32_t words_slab_used() const { return words_slab_used_; }

private:
    MoEBlockResources(
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        const MoEModelDims& dims,
        const MoEBlockConfig& config,
        const MoEDtypes& dtypes);
    void drop_state();  // every plan, persistent tensor and port placement (release, set_chunk_splits)
    ttsl::hash::hash_t make_plan_key() const;
    // TT_FATAL after release() or once the mesh is closed (MeshDevice::is_initialized). (A mesh object already
    // destroyed cannot be detected: the caller must release() first, see above.)
    void check_alive() const;
    bool packs_words() const;
    // L1, or L1_SMALL with config.persist_l1_small.
    tt::tt_metal::BufferType persistent_buffer_type() const;
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
    FirstHops first_hops_;
    ttsl::hash::hash_t plan_key_ = 0;

    // allowed cells (nullopt: the whole grid) -> placement
    std::map<std::optional<std::vector<std::pair<size_t, size_t>>>, PortCores> ports_;
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
