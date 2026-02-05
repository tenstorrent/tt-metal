// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>

//=============================================================================
// MoE Ring All-to-All Configuration
// Two arrangements supported:
// 1. Boundary-optimized
// 2. Evenly distributed
//=============================================================================

namespace matmul_wo_ring {

constexpr uint32_t NUM_CORES = 12;

constexpr uint32_t NUM_W_TILES_W = 28;
constexpr uint32_t NUM_W_TILES_H = 512;

constexpr uint32_t W_TXNS_PER_BLOCK = 2;
constexpr uint32_t W_TILES_PER_TXN = 7;

constexpr uint32_t N_TILES_PER_ITER = 7;

//-----------------------------------------------------------------------------
// Precomputed lookup tables (generated at compile time)
//-----------------------------------------------------------------------------
constexpr uint32_t K_TILES_PER_CORE_A[NUM_CORES] = {
    44,
    44,
    42,
    42,
    42,
    42,
    42,
    42,
    44,
    44,
    42,
    42,
};

constexpr uint32_t K_TILES_PER_CORE_B[NUM_CORES] = {
    44,
    42,
    42,
    44,
    42,
    42,
    44,
    42,
    42,
    44,
    42,
    42,
};

constexpr uint32_t MAX_K_TILES_PER_CORE = 44;
}  // namespace matmul_wo_ring
