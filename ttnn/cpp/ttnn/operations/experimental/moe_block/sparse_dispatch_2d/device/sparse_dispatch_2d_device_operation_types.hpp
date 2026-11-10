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
    // payload): it determines `ports`.
    ttsl::hash::hash_t plan_key = 0;
    uint32_t group = 8;            // rows a reader -> sender hand-off (and fabric headers a sender)
    uint32_t manifest_batch = 16;  // manifest entries a stream sends at once
    uint32_t pack = 1;             // rows a fabric packet (1 or 2)
    // Derived from the hashed fields above; not hashed itself.
    std::shared_ptr<const SparseDispatch2dPorts> ports;

    // (reflection / logging: the structural fields; `ports` follows from them)
    static constexpr auto attribute_names = std::forward_as_tuple("plan_key", "group", "manifest_batch", "pack");
    auto attribute_values() const { return std::forward_as_tuple(plan_key, group, manifest_batch, pack); }
};

struct SparseDispatch2dInputs {
    Tensor xbuf;   // [1, 1, N * S, W] ROW_MAJOR, DRAM interleaved; own rows in place; written in place
    Tensor route;  // [1, 1, 1, RW * S] uint32 ROW_MAJOR, DRAM interleaved: the own rows' route words
    // The port cores' semaphore words (MoEBlockResources persistent words "sparse_dispatch_2d/sems"): per port core
    // the three ring counters and the fence's ready word, 16 B apart. Persistent and zero between calls.
    Tensor sems;
    // The block's L1 arena of the call (height-sharded over the worker grid, one address on every core and chip): the
    // dispatch's CBs are its first CbSizes::arena_bytes() a port core - manifest first, then rows, meta, plan, stage,
    // 64 B aligned. Absent: static CBs (a standalone call).
    std::optional<Tensor> arena;
};

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
