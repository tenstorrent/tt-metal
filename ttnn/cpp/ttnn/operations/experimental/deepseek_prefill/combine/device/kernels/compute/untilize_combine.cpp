// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "ckernel.h"
#include "ckernel_defs.h"

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

// Compile-time args:
//   0: cb_signal_id    - CB for reader->compute signaling (c_17)
//   1: cb_untilize_id  - CB for compute untilized output (c_18)
//   2: cb_in_id        - CB for untilize input tile data (c_0)
//   3: hidden_size     - hidden dimension (e.g., 7168)
//   4: read_batch_size - number of rows per untilize batch (e.g., 32)
void kernel_main() {
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(3);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(4);

    constexpr uint32_t block_ct_dim = 8;
    constexpr uint32_t full_ct_dim = hidden_size / 32;
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

    compute_kernel_hw_startup(cb_in_id, cb_untilize_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilize_id);

    while (true) {
        cb_reserve_back(cb_untilize_id, read_batch_size);
        cb_wait_front(cb_signal_id, 1);
        uint32_t val = read_tile_value(cb_signal_id, 0, 0);
        cb_pop_front(cb_signal_id, 1);
        if (val == ROUTE_INFO_SENTINEL) {
            break;
        }
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_wait_front(cb_in_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
            cb_pop_front(cb_in_id, block_ct_dim);
        }

        cb_push_back(cb_untilize_id, read_batch_size);
    }
    pack_untilize_uninit(cb_untilize_id);
}
