// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "matmul_reduce_scatter_planner.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// Everything the public entry resolved before launch. Every field is structural, so the default program key
// (every attribute + the tensor specs) is exact: the plan is solved from shapes, dtypes, the group and the
// device's fixed grid / L1 bank size, so equal attributes mean an equal program.
struct MatmulReduceScatterParams {
    uint32_t cluster_axis = 0;
    int32_t scatter_dim = -2;  // canonical (-1 or -2)
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    uint32_t group_size = 0;                 // devices along cluster_axis (G)
    ScheduleKind kind = ScheduleKind::Line;  // Ring with G >= 3 is a ring; a 2-device ring is the line
    uint32_t num_links = 0;                  // resolved L (None: every usable link)
    // the compute config fields the kernels use (fp32 DEST and full sync are always on)
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    bool math_approx_mode = false;
    tt::tt_metal::MemoryConfig output_memory_config;
    ChunkPlan plan;  // solved by the entry (it sizes the staging tensor before launch)
};

struct MatmulReduceScatterInputs {
    const Tensor& input_tensor;  // A (..., M, K), DRAM interleaved, tiled
    const Tensor& weight;        // W (K, N), DRAM interleaved, tiled
    // Per-mesh L1 staging (cb_finished + arrival zones, height-sharded over the full grid; same address on every
    // core and chip). Reallocated when a call's plan needs another size, so its address is refreshed on a hit.
    const Tensor& staging;
    const std::optional<Tensor>& output;  // preallocated output (in place refused)
};

// Refusals, named after their Python counterparts. The binding raises UnsupportedAxisValue as
// ttnn.operations._op_contract.UnsupportedAxisValue and NotImplementedRefusal as NotImplementedError (the
// planner's L1 misses and the sharded output config). ValueError cases throw std::invalid_argument. EXCLUSIONS is
// empty, so there is no ExcludedCell.
struct UnsupportedAxisValue : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct NotImplementedRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::matmul_reduce_scatter
