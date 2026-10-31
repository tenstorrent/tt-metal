// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <tt_stl/reflection.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// The mesh wiring and port cores one cached program is built for (from MoEBlockResources; computed once, shared).
struct SparseDispatch2dPorts {
    MeshPlan plan;
    PortCores cores;
};

struct SparseDispatch2dParams {
    // Structural, hashed. plan_key: MoEBlockResources::plan_key() (config, dims, mesh shape, wrap, fabric config, max
    // payload): with `placement` it determines `ports`.
    ttsl::hash::hash_t plan_key = 0;
    PortPlacement placement = PortPlacement::Noc1;
    uint32_t group = 8;            // rows a reader -> sender hand-off (and fabric headers a sender)
    uint32_t manifest_batch = 16;  // manifest entries a stream sends at once
    uint32_t pack = 1;             // rows a fabric packet (1 or 2)
    uint32_t spin_limit = 0;       // debug: a reader gives up after this many idle polls and reports (0: never)
    // Derived from the hashed fields above; not hashed itself.
    std::shared_ptr<const SparseDispatch2dPorts> ports;

    // (reflection / logging: the structural fields; `ports` follows from them)
    static constexpr auto attribute_names =
        std::forward_as_tuple("plan_key", "placement", "group", "manifest_batch", "pack", "spin_limit");
    auto attribute_values() const {
        return std::forward_as_tuple(plan_key, placement, group, manifest_batch, pack, spin_limit);
    }
};

struct SparseDispatch2dInputs {
    Tensor xbuf;   // [1, 1, N * S, W] ROW_MAJOR, DRAM interleaved; own rows in place; written in place
    Tensor route;  // [1, 1, 1, RW * S] uint32 ROW_MAJOR, DRAM interleaved: the own rows' route words
    // The port cores' semaphore words (MoEBlockResources persistent words "sparse_dispatch_2d/sems"): per port core
    // the three ring counters and the fence's ready word, 16 B apart. Persistent and zero between calls.
    Tensor sems;
    std::optional<Tensor> debug;  // 64 B a port (port index (r * C + c) * 8 + dir * 2 + link) with spin_limit
};

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
