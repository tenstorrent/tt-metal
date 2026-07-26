// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Addressing shared by the paged remote-read implementations.
// Changing transport must not change the physical page selected for a token.
namespace sparse_sdpa::paged {

struct TokenLocation {
    uint32_t logical_bundle;
    uint32_t owner;
    uint32_t local_row;
};

template <uint32_t bundle_tokens, uint32_t chunk_local, uint32_t sp>
constexpr TokenLocation locate(uint32_t logical_token) {
    static_assert(sp > 0);
    static_assert(bundle_tokens == chunk_local * sp);
    const uint32_t in_bundle = logical_token % bundle_tokens;
    return {
        .logical_bundle = logical_token / bundle_tokens,
        .owner = in_bundle / chunk_local,
        .local_row = in_bundle % chunk_local,
    };
}

constexpr uint32_t pool_page(
    uint32_t physical_bundle, uint32_t layer, uint32_t num_layers, uint32_t chunk_local, uint32_t local_row) {
    return (physical_bundle * num_layers + layer) * chunk_local + local_row;
}

constexpr uint32_t owner_flat(
    uint32_t owner, uint32_t sp_axis, uint32_t my_mesh_row, uint32_t my_mesh_col, uint32_t mesh_cols) {
    return sp_axis == 0 ? owner * mesh_cols + my_mesh_col : my_mesh_row * mesh_cols + owner;
}

}  // namespace sparse_sdpa::paged
