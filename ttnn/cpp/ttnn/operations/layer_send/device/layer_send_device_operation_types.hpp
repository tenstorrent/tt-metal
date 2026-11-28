// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::layer_send {

// The per-layer fabric sends of deepseek_v41_flash_d_p (NC.3), one program kind each (their io and per-call values:
// layer_send.hpp).
enum class SendKind : uint8_t {
    WINDOW = 0,        // tt/window_send.WindowSend.__call__
    WINDOW_FUSED = 1,  // tt/window_send.WindowSend.fused
    SNAP_STATIC = 2,   // tt/snapshot_send.SnapshotSend.__call__ (one program a block pattern)
    SNAP_DYN = 3,      // tt/snapshot_send.SnapshotSend._call_dyn
    SNAP_MULTI = 4,    // tt/snapshot_send.SnapshotSend.multi
    GLUED = 5,         // tt/packed_glued.GluedSend.__call__
};

// One fabric connection of a sender core: the chip it opens it to (its first hop), the link, and for the window /
// glued sends the chip its packets are routed to (dst).
struct Hop {
    uint32_t dir = 0;  // snapshot: index in ("Y+", "Y-", "X+", "X-"); window / glued: 0
    uint32_t peer_r = 0, peer_c = 0, link = 0;
    uint32_t dst_r = 0, dst_c = 0;
};

// Everything structural about one Python send instance, made once (load time) by the model from its routes
// (ttnn.bringup.combine_rf sparse_combine_l1.route / sparse_all_gather_2d.mesh_plan) and kept; its hash is computed
// once. ``hops[q]``: chip q's (chip-linear) connections: window / glued the one to q + 1 (empty: no send), snapshot
// every fabric neighbour in mesh_plan's order. ``consts`` per kind (layer_send.cpp make_plan). ``sems``: the global
// semaphore addresses the Python instance created. ``pattern`` (SNAP_STATIC): the 4 block codes (0xFF zero, 0xFE
// local, else the owner's chip-linear index).
struct SendPlan {
    SendKind kind = SendKind::WINDOW;
    uint32_t rows = 0, cols = 0;
    uint32_t wrap_x = 0, wrap_y = 0;
    std::vector<std::vector<Hop>> hops;
    std::vector<uint32_t> consts;
    std::vector<uint32_t> sems;
    std::vector<uint32_t> pattern;
    uint64_t hash = 0;
};

struct LayerSendParams {
    std::shared_ptr<const SendPlan> plan;
    // Per-call values (never in the program key: written into the cached programs' runtime args on every call).
    std::vector<uint32_t> values;

    static constexpr auto attribute_names = std::forward_as_tuple("kind", "plan_hash");
    auto attribute_values() const {
        return std::make_tuple(static_cast<uint32_t>(plan->kind), static_cast<uint64_t>(plan->hash));
    }
};

struct LayerSendInputs {
    std::vector<Tensor> io;
};

}  // namespace ttnn::operations::layer_send
