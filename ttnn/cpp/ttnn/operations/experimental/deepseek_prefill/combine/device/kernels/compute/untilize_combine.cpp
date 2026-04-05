// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is a COMPUTE kernel. Do NOT include "api/dataflow/dataflow_api.h" here.
// Signaling uses CB push/pop (not semaphores — those require NOC access unavailable on TRISC).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/dprint.h"

// Compile-time args:
//   0: cb_untilize_out_id - CB for reader->compute messages (sentinel-terminated)
//   1: cb_compute_ack_id  - CB for compute->reader ack signal
//   2: cb_in_id           - CB for untilize input (c_0, data already present)
//   3: cb_untilized_id    - CB for untilize output (c_19)
//   4: hidden_size        - hidden dimension (e.g., 7168)
void kernel_main() {
    constexpr uint32_t cb_untilize_out_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_compute_ack_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_untilized_id = get_compile_time_arg_val(3);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(4);

    // 7168 / 32 = 224 tiles per tile-row, 8 tiles per DEST section = 28 blocks
    constexpr uint32_t block_ct_dim = 8;
    constexpr uint32_t full_ct_dim = hidden_size / 32;           // 7168 / 32
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;  // 28

    // Initialize pack_untilize with block_ct_dim=8, full_ct_dim=224
    ckernel::pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilized_id);

    while (true) {
        // Wait for reader signal (sentinel check)
        cb_wait_front(cb_untilize_out_id, 1);
        uint32_t val = read_tile_value(cb_untilize_out_id, 0, 0);
        cb_pop_front(cb_untilize_out_id, 1);

        if (val == 0xFFFFFFFF) {
            // exit
            break;
        }

        // Untilize 224 tiles (one tile-row) from cb_in_id -> cb_untilized_id
        // 28 blocks of 8 tiles each, output is 32 rows × 7168 datums row-major
        for (uint32_t block = 0; block < num_blocks; block++) {
            MATH((llk_math_wait_for_dest_available()));

            // UNPACK 8 tiles into SRCA, MATH copies each to DEST slot 0..7
            for (uint32_t c = 0; c < block_ct_dim; c++) {
                UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
                    cb_in_id, block * block_ct_dim + c)));
                MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(c)));
            }

            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

            // PACK untilizes all 8 tiles at correct column offset within the 7168-wide row
            PACK((llk_packer_wait_for_math_done()));
            PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(1, cb_untilized_id, FACE_R_DIM, 4, block)));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        }

        // Signal reader that we processed this page via ack CB
        cb_reserve_back(cb_compute_ack_id, 1);
        cb_push_back(cb_compute_ack_id, 1);
    }

    ckernel::pack_untilize_uninit(cb_untilized_id);
}
