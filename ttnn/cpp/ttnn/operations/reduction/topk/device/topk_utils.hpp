// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::prim {
uint32_t largest_power_of_two(uint32_t x);

// Structural multi-core eligibility (shape/K requirements only, no grid or L1
// feasibility): width gate (>= multi_core_min_width, or the Ht-aware relaxation
// for inputs with <= multi_core_low_ht_max_tile_rows tile rows), reduced width
// below the 16-bit bitonic index limit, power-of-two width, and K <=
// multi_core_max_k. This is the single source of truth shared by
// select_program_factory, validate_on_program_cache_miss, and the composite
// router in topk.cpp (the router passes a pinned num_tile_rows to disable the
// Ht-aware relaxation — the composite measured faster on that cell); the
// cost/grid check (verify_multi_core_cost) comes on top.
bool topk_multicore_structurally_eligible(uint32_t reduced_width, uint32_t num_tile_rows, uint32_t k);

struct TopKCoreConfig {
    uint16_t num_cores = 0;
    uint16_t split_size = 0;
    uint16_t rem = 0;
    uint16_t final_input_size = 0;
    uint16_t selected_x = 0;
    uint16_t selected_y = 0;
};

std::optional<TopKCoreConfig> find_topk_core_config(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size,
    uint32_t tile_width = 32);

bool verify_multi_core_cost(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size,
    uint32_t tile_width = 32);

bool verify_single_core_cost(const ttnn::Tensor& input_tensor, uint32_t k, bool uint16_output);

// True when the op must use 32-bit indices: the padded reduced dim does not fit in 16 bits, or the
// input is fp32, which sorts with fp32 dest accumulation and loads indices as INT32.
bool is_uint32_index_required(const ttnn::Tensor& input_tensor, int8_t dim);
}  // namespace ttnn::prim
