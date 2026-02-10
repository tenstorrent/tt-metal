// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

// Compile-time args: role, num_tiles, CBs
constexpr uint32_t device_role = get_compile_time_arg_val(0);
constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
constexpr uint32_t local_cb = get_compile_time_arg_val(2);
constexpr uint32_t received_cb_r1 = get_compile_time_arg_val(3);
constexpr uint32_t received_cb_r2 = get_compile_time_arg_val(4);
constexpr uint32_t received_cb_r3 = get_compile_time_arg_val(5);
constexpr uint32_t output_cb = get_compile_time_arg_val(6);
constexpr uint32_t scratch_cb = get_compile_time_arg_val(7);

// Load tiles from CB to dest register
FORCE_INLINE void load_tiles_to_dest(uint32_t cb, uint32_t n_tiles) {
    copy_tile_to_dst_init_short(local_cb);
    cb_wait_front(cb, n_tiles);
    acquire_dst();
    for (uint32_t i = 0; i < n_tiles; i++) {
        copy_tile(cb, i, i);
    }
    cb_pop_front(cb, n_tiles);
}

// Accumulate tiles from CB into dest register
FORCE_INLINE void accumulate_tiles_to_dest(uint32_t cb, uint32_t n_tiles) {
    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb_r1);
    cb_wait_front(cb, n_tiles);
    for (uint32_t i = 0; i < n_tiles; i++) {
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, i, i);
    }
    cb_pop_front(cb, n_tiles);
}

// Pack tiles from dest register to CB
FORCE_INLINE void pack_tiles_from_dest(uint32_t cb, uint32_t n_tiles) {
    cb_reserve_back(cb, n_tiles);
    for (uint32_t i = 0; i < n_tiles; i++) {
        pack_tile(i, cb, i);
    }
    release_dst();
    cb_push_back(cb, n_tiles);
}

void kernel_main() {
    if constexpr (device_role == MESH_LEAF) {
        return;
    }

    // Big init for binary operations
    binary_op_init_common(local_cb, received_cb_r1, scratch_cb);

    // Load local tiles to dest
    load_tiles_to_dest(local_cb, num_tiles);

    // Switch to binary_dest_reuse mode and accumulate
    accumulate_tiles_to_dest(received_cb_r1, num_tiles);

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        accumulate_tiles_to_dest(received_cb_r2, num_tiles);
    }

    if constexpr (device_role == MESH_ROOT1) {
        accumulate_tiles_to_dest(received_cb_r3, num_tiles);
    }

    // Pack final result once
    pack_tiles_from_dest(scratch_cb, num_tiles);
}
