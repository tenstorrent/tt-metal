// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Reduce-scatter over the fabric along one mesh axis (or the whole mesh): input [1, 1, G * Sb, W] TILE, DRAM
// interleaved, a partial per chip; every chip gets its block of the group's sum, [1, 1, Sb, W] (dim 2), or its
// column block [1, 1, S, Wb] (dim 3). cache_descriptors / program_hash / prepared select nothing in the native op
// (its program cache always applies); only their Python preconditions are kept. scratch_tensor: caller-held
// scratch from fabric_reduce_scatter_scratch_tensor; allocated per call when absent.
Tensor fabric_reduce_scatter(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis = 0u,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    uint32_t num_links = 1,
    const std::string& scheme = "ring",
    const std::optional<Tensor>& output = std::nullopt,
    const std::string& placement = "auto",
    int32_t dim = 2,
    bool exact_fp32_add = false,
    bool cache_descriptors = false,
    bool program_hash = false,
    bool prepared = false,
    const std::optional<Tensor>& scratch_tensor = std::nullopt);

// The scratch tensor a fabric_reduce_scatter call with these arguments uses ([1, 1, (G + 1) Sb, W]).
Tensor fabric_reduce_scatter_scratch_tensor(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis = 0u,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    uint32_t num_links = 1,
    const std::string& scheme = "ring",
    const std::string& placement = "auto",
    int32_t dim = 2);

// The Python op's prepare(): the plan and the counters of one group configuration, built once per mesh (the first
// build creates the global semaphores and synchronizes the mesh). Call it at load time so that no call pays for it.
void fabric_reduce_scatter_prepare(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    std::optional<uint32_t> cluster_axis = 0u,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    uint32_t num_links = 1,
    const std::string& scheme = "ring",
    const std::string& placement = "auto");

// One program_hash call's record (the generic op's _DESC_CACHE entry), for inspection only.
struct FabricReduceScatterDescriptorEntry {
    bool prepared = false;
    uint64_t input_address = 0;
    uint64_t scratch_address = 0;
    uint64_t output_address = 0;
    uint32_t num_programs = 0;

    bool operator==(const FabricReduceScatterDescriptorEntry&) const = default;
};

// The program_hash calls' records, oldest first, one per distinct entry.
const std::vector<FabricReduceScatterDescriptorEntry>& fabric_reduce_scatter_descriptor_entries();

}  // namespace ttnn
