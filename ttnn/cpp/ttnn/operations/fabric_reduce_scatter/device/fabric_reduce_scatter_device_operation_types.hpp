// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

// scheme "ring" (one ring or line per group) or "dual_cycles" (two edge-disjoint Hamiltonian cycles over a
// torus with both sides >= 3, cluster_axis None, Ring).
enum class Scheme : uint8_t { RING, DUAL_CYCLES };

// placement "auto" (ports next to their link's Ethernet core) or "simple" (any free core; what "auto" turns
// into under TT_METAL_EMULE_MODE).
enum class Placement : uint8_t { AUTO, SIMPLE };

// What the caller chose and the entry resolved; every field is structural and hashed by
// FabricReduceScatterDeviceOperation::compute_program_hash. The plan, semaphores and fabric routes are per-mesh state
// looked up from these (fabric_reduce_scatter_planner.hpp, fabric_reduce_scatter_program_factory.hpp), never
// attributes.
struct FabricReduceScatterParams {
    std::optional<uint32_t> cluster_axis;  // 0, 1, or none (the whole mesh)
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    uint32_t num_links = 1;
    Scheme scheme = Scheme::RING;
    Placement placement = Placement::AUTO;
    uint32_t dim = 2;  // normalized: 2 = row blocks, 3 = column blocks
    bool exact_fp32_add = false;
    // The mesh shape the plan was built for (MeshDevice::reshape keeps the id) and the group size.
    uint32_t mesh_rows = 0;
    uint32_t mesh_cols = 0;
    uint32_t group_size = 0;
};

struct FabricReduceScatterInputs {
    const Tensor& input;
    // Preallocated output ([1, 1, Sb, W], input's dtype, TILE, DRAM interleaved).
    const std::optional<Tensor>& output;
    // Caller-held scratch ([1, 1, (G + 1) Sb, W], input's dtype, TILE, DRAM interleaved); allocated per call
    // by the entry when absent.
    const std::optional<Tensor>& scratch;
};

// The Python op's `assert`s (refusals the tests expect as AssertionError: "exact_fp32_add needs FLOAT32",
// "columns ... must split", "program_hash needs cache_descriptors", "prepared needs program_hash"). The
// binding raises it as Python's AssertionError. Plan failures (no direct link, too few links, too few cores,
// unknown scheme) are the Python's ValueError: throw std::invalid_argument.
struct AssertionRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::fabric_reduce_scatter
