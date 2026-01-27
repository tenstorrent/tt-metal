// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

constexpr uint32_t local_cb = get_compile_time_arg_val(0);
constexpr uint32_t received_cb = get_compile_time_arg_val(1);
constexpr uint32_t output_cb = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
constexpr uint32_t device_role = get_compile_time_arg_val(4);

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

    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in2_cb, num_tiles);
}

void MAIN {
    // LEAF devices don't do compute - early exit
    if constexpr (device_role == MESH_LEAF) {
        return;
    }

    // Staged reduction using 32x32 tiles for compute optimization
    // Stage 1 (all ROOTs): local + received → output
    // Stage 2 (ROOT2, ROOT1): output + received → local (reuse local_cb)
    // Stage 3 (ROOT1 only): local + received → output
    //
    // Final result: ROOT3 → output_cb, ROOT2 → local_cb, ROOT1 → output_cb

    binary_op_init_common(local_cb, received_cb, output_cb);
    add_tiles_init(local_cb, received_cb);

    // Stage 1: local + received → output (all ROOT devices)
    reduce_step<local_cb, received_cb, output_cb>();

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        // Stage 2: output + received → local
        add_tiles_init(output_cb, received_cb);
        reduce_step<output_cb, received_cb, local_cb>();
    }

    if constexpr (device_role == MESH_ROOT1) {
        // Stage 3: local + received → output
        add_tiles_init(local_cb, received_cb);
        reduce_step<local_cb, received_cb, output_cb>();
    }
}

}  // namespace NAMESPACE
