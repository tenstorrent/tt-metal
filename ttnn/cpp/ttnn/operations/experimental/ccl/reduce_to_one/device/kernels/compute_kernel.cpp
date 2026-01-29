// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "api/debug/dprint.h"

namespace NAMESPACE {

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

constexpr uint32_t local_cb = get_compile_time_arg_val(0);        // Input tensor
constexpr uint32_t received_cb_r1 = get_compile_time_arg_val(1);  // Round 1: LEAF → ROOT*
constexpr uint32_t received_cb_r2 = get_compile_time_arg_val(2);  // Round 2: ROOT3 → ROOT2/ROOT1
constexpr uint32_t received_cb_r3 = get_compile_time_arg_val(3);  // Round 3: ROOT2 → ROOT1
constexpr uint32_t output_cb = get_compile_time_arg_val(4);       // Final output tensor (ROOT1 only)
constexpr uint32_t scratch_cb = get_compile_time_arg_val(5);      // Scratch for intermediate results
constexpr uint32_t num_tiles = get_compile_time_arg_val(6);
constexpr uint32_t device_role = get_compile_time_arg_val(7);
constexpr uint32_t scratch_cb2 = get_compile_time_arg_val(8);  // Second scratch buffer (stable addr)

// Helper to perform one reduction step: in1_cb + in2_cb → out_cb
template <uint32_t in1_cb, uint32_t in2_cb, uint32_t out_cb>
FORCE_INLINE void reduce_step() {
    cb_reserve_back(out_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_wait_front(in2_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in1_cb, in2_cb, i, i, 0);
        pack_tile(0, out_cb, i);
        release_dst();
    }

    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in2_cb, num_tiles);
    cb_push_back(out_cb, num_tiles);
}

void MAIN {
    // LEAF devices don't do compute - early exit
    if constexpr (device_role == MESH_LEAF) {
        return;
    }

    // Staged reduction using scratch_cb and scratch_cb2 (stable addresses for trace mode)
    // Each round uses a different received CB to prevent data overwrites:
    // ROOT3 (1 stage): local + received_r1 → scratch_cb2
    // ROOT2 (2 stages): local + received_r1 → scratch_cb, scratch_cb + received_r2 → scratch_cb2
    // ROOT1 (3 stages): scratch_cb → scratch_cb2 → output_cb (write to output_cb only once)

    if constexpr (device_role == MESH_ROOT3) {
        // ROOT3: 1 stage, direct to scratch_cb2 (receives from LEAF via received_cb_r1)
        binary_op_init_common(local_cb, received_cb_r1, scratch_cb2);
        add_tiles_init(local_cb, received_cb_r1);
        reduce_step<local_cb, received_cb_r1, scratch_cb2>();
    }

    if constexpr (device_role == MESH_ROOT2) {
        // ROOT2: 2 stages via scratch
        // Stage 1: local + received_r1 (from LEAF) → scratch
        binary_op_init_common(local_cb, received_cb_r1, scratch_cb);
        add_tiles_init(local_cb, received_cb_r1);
        reduce_step<local_cb, received_cb_r1, scratch_cb>();
        // Stage 2: scratch + received_r2 (from ROOT3) → scratch_cb2
        add_tiles_init(scratch_cb, received_cb_r2);
        reduce_step<scratch_cb, received_cb_r2, scratch_cb2>();
    }

    if constexpr (device_role == MESH_ROOT1) {
        // ROOT1: 3 stages - write to scratch_cb, writer will NOC copy to output
        // Stage 1: local + received_r1 (from LEAF) → scratch_cb2
        binary_op_init_common(local_cb, received_cb_r1, scratch_cb2);
        add_tiles_init(local_cb, received_cb_r1);
        reduce_step<local_cb, received_cb_r1, scratch_cb2>();
        // Stage 2: scratch_cb2 + received_r2 (from ROOT3) → scratch_cb
        add_tiles_init(scratch_cb2, received_cb_r2);
        reduce_step<scratch_cb2, received_cb_r2, scratch_cb>();
        // Stage 3: scratch_cb + received_r3 (from ROOT2) → scratch_cb2 (writer will NOC copy to output)
        add_tiles_init(scratch_cb, received_cb_r3);
        reduce_step<scratch_cb, received_cb_r3, scratch_cb2>();
    }
}

}  // namespace NAMESPACE
