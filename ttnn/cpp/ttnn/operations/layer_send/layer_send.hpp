// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "device/layer_send_device_operation_types.hpp"

namespace ttnn {

// The plan of one send instance (made once, at load time): kind (operations::layer_send::SendKind), the mesh shape,
// the torus wraps, per chip (chip-linear) its fabric hops (dir, peer row, peer col, link, dst row, dst col), the
// kind's constants, the global semaphore addresses and (static snapshot) the block pattern.
std::shared_ptr<const operations::layer_send::SendPlan> layer_send_plan(
    uint32_t kind,
    uint32_t rows,
    uint32_t cols,
    bool wrap_x,
    bool wrap_y,
    const std::vector<std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>>& hops,
    const std::vector<uint32_t>& consts,
    const std::vector<uint32_t>& sems,
    const std::vector<uint32_t>& pattern);

// One call of a per-layer fabric send of deepseek_v41_flash_d_p (in place; returns io[0]):
//   WINDOW        io [region, tail]                                    values []
//   WINDOW_FUSED  io [region, tail, rows, ring]                        values (src row, ring row, rows) x pieces
//   SNAP_STATIC   io [ring, slab]                                      values [page0, ring row x 4]
//   SNAP_DYN      io [ring, slab]                                      values [page0, ring rows (4 x 8 bits), codes]
//   SNAP_MULTI    io [ring_0 .. ring_{n-1}, slab_0 .. slab_{n-1}]      values (page0, ring rows, codes) x n
//   GLUED         io [region, rows, tail_0 .. tail_5, ring_0 .. ring_5] values (active, tail, page, rot) x 6,
//                                                                       (own row, ring row, rows) x pieces x 6
Tensor layer_send(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const operations::layer_send::SendPlan>& plan,
    const std::vector<uint32_t>& values);

}  // namespace ttnn
