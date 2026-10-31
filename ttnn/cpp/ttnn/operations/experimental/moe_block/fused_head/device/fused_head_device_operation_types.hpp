// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "head_program.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

struct FusedHeadParams {
    // The head's structure: the planned layout, the model, the switches its kernels see.
    HeadProgramConfig config;
    // MoEBlockResources::plan_key() and the resources' serial (fused_head.cpp): the persistent words' addresses are
    // baked into the programs (several at an offset inside a word tensor), so one cached program serves one resources
    // object - the serial stands for them in the hash instead of their addresses.
    ttsl::hash::hash_t plan_key = 0;
    uint64_t resources_serial = 0;
    // Hash-excluded: the host model (made from config on this mesh) and the persistent words' addresses.
    std::shared_ptr<const HeadModel> model;
    HeadWordAddresses words;

    static constexpr auto attribute_names = std::forward_as_tuple("plan_key", "resources_serial");
    auto attribute_values() const { return std::forward_as_tuple(plan_key, resources_serial); }
};

struct FusedHeadInputs {
    Tensor x;                      // [1, 1, rows, H / TP] bf16 TILE, DRAM: this chip's shard of the residual
    std::optional<Tensor> x_full;  // the DRAM fallback: x gathered over TP [1, 1, rows, H]
    Tensor w;                      // the shared expert's weights (prepare_fused_head_weights)
    Tensor w_router;               // W_r^T [E, H / TP] bf16 TILE
    Tensor router_bias;            // [E, 32] fp32 TILE
    Tensor router_index;           // [E, 32] uint16 TILE
    // The resources' persistent DRAM tensors the head writes (outputs) or exchanges through.
    Tensor y;                                          // the shared expert's reduced output (returned)
    Tensor rows;                                       // the owners' joint routing rows
    Tensor xbuf;                                       // the dispatch buffer
    Tensor route;                                      // the owned tokens' route words
    std::optional<Tensor> latent;                      // the owners' latent rows (a latent model)
    Tensor slots;                                      // the routing exchange's slots
    Tensor fslots;                                     // its forwarding slots
    Tensor counts, regions, token_index, table, plan;  // the combine plan the routing planner writes
};

}  // namespace ttnn::operations::experimental::moe_block::fused_head
