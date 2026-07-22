// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Attributes (non-tensor config) for the Flash KDA recurrent state update.
struct FlashKdaParams {
    std::uint32_t num_items;  // N — number of independent per-core state-update items (one core each)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Input tensors for the Flash KDA recurrent state update (single step, no chunk loop).
//
// All tensors are float32, TILE_LAYOUT, DRAM. Shape conventions (N is the batched-item axis,
// one item per core):
//   S_prev  : [N, Dk, Dv]  previous recurrent state
//   g       : [N, Dk, 1]   per-key-dim-row decay (column layout: one value per row, replicated
//                          across all Dv columns of that row)
//   k       : [N, 1, Dk]   key vector (row layout)
//   v       : [N, 1, Dv]   value vector (row layout)
//   beta    : [N, 1, 1]    per-item scalar gate
//   q       : [N, 1, Dk]   query vector (row layout)
struct FlashKdaInputs {
    Tensor S_prev;
    Tensor g;
    Tensor k;
    Tensor v;
    Tensor beta;
    Tensor q;
};

}  // namespace ttnn::prim
