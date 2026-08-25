// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::experimental::prim::detail {

struct MoEComputeL1Plan {
    uint32_t a2a_tiles_per_step;
    uint32_t a2a_buffer_slots;
    uint32_t a2a_tiles;
    uint32_t weight_tiles_per_block;
    uint32_t weight_pipeline_slots;
    uint32_t matmul_static_bytes;
};

constexpr uint32_t moe_compute_l1_deficit(
    const MoEComputeL1Plan& plan, uint32_t circular_buffer_base, uint32_t lowest_l1_tensor_address) {
    const uint32_t required_l1_tensor_base = circular_buffer_base + plan.matmul_static_bytes;
    return required_l1_tensor_base > lowest_l1_tensor_address ? required_l1_tensor_base - lowest_l1_tensor_address : 0;
}

constexpr MoEComputeL1Plan plan_moe_compute_l1(
    uint32_t intermediate_tiles, uint32_t ring_cores, uint32_t weight_tile_bytes, bool has_bias) {
    constexpr uint32_t bf16_tile_bytes = 2048;
    constexpr uint32_t fast_weight_tiles_per_block = 28;
    constexpr uint32_t fast_weight_slots = 3;
    constexpr uint32_t compact_weight_tiles_per_block = 4;
    constexpr uint32_t compact_weight_slots = 1;
    // Leave at least half of worker L1 for the sharded input/output tensors,
    // semaphores, kernel binaries, and non-matmul CBs.  The decision is based
    // solely on the public primitive geometry and hardware storage classes.
    constexpr uint32_t fast_a2a_budget_bytes = 512 * 1024;

    const uint32_t per_core_tiles = ((intermediate_tiles + ring_cores - 1) / ring_cores + 1) & ~1u;
    // The ring consumes the local shard before forwarding it.  The final step
    // consumes the last remote shard but does not need to send it back to its
    // origin, so slot zero can be reused after ring_cores - 1 hops.  Keeping a
    // twelfth slot for that redundant final send wastes one full activation
    // shard on every matmul core.
    const uint32_t a2a_buffer_slots = ring_cores - 1;
    const uint32_t a2a_tiles = per_core_tiles * a2a_buffer_slots;
    const bool compact = a2a_tiles * bf16_tile_bytes > fast_a2a_budget_bytes;
    const uint32_t weight_tiles = compact ? compact_weight_tiles_per_block : fast_weight_tiles_per_block;
    const uint32_t slots = compact ? compact_weight_slots : fast_weight_slots;
    const uint32_t bookkeeping_bytes = 16 + 16 + 32 + (has_bias ? bf16_tile_bytes : 0);
    return {
        per_core_tiles,
        a2a_buffer_slots,
        a2a_tiles,
        weight_tiles,
        slots,
        a2a_tiles * bf16_tile_bytes + weight_tiles * slots * weight_tile_bytes + bookkeeping_bytes};
}

static_assert(plan_moe_compute_l1(14336 / 32, 12, 576, false).a2a_tiles_per_step == 38);
static_assert(plan_moe_compute_l1(14336 / 32, 12, 576, false).a2a_buffer_slots == 11);
static_assert(plan_moe_compute_l1(14336 / 32, 12, 576, false).a2a_tiles == 418);
static_assert(plan_moe_compute_l1(14336 / 32, 12, 576, false).weight_pipeline_slots == 1);
static_assert(plan_moe_compute_l1(2048 / 32, 12, 576, false).weight_pipeline_slots == 3);

}  // namespace ttnn::experimental::prim::detail
