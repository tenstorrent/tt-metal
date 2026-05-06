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
#include "api/debug/dprint.h"
#include "internal/circular_buffer_interface.h"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

// Push a single uint32 scalar value to cb_id from a compute kernel (pack TRISC only).
// Dataflow kernels use get_write_ptr(cb_id), but compute kernels have to go through the
// local CB interface and apply cb_addr_shift.
inline void push_scalar_signal(uint32_t cb_id, uint32_t value) {
#ifdef TRISC_PACK
    cb_reserve_back(cb_id, 1);
    auto wr_ptr = (get_local_cb_interface(cb_id).fifo_wr_ptr) << cb_addr_shift;
    auto* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wr_ptr);
    slot[0] = value;
    cb_push_back(cb_id, 1);
#endif
}

// Compile-time args:
//   0: cb_signal_id       - CB for reader_untilize -> compute signaling (c_3)
//   1: cb_untilize_id     - CB for compute's untilized output (c_2)
//   2: cb_in_id           - CB for untilize input tile data (c_0)
//   3: read_batch_size    - number of rows per untilize batch (e.g., 32)
//   4: full_ct_dim        - hidden_size / 32 (e.g., 224 for hidden_size=7168)
//   5: block_ct_dim       - largest divisor of full_ct_dim <= 8
//   6: cb_stop_signal_id  - CB for compute -> zero_init_writer per-batch + stop signal (c_8)
void kernel_main() {
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(3);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(4);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stop_signal_id = get_compile_time_arg_val(6);
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

        // Tell zero_init_writer that a batch has been pushed onto cb_untilize_id and is ready to send.
        push_scalar_signal(cb_stop_signal_id, 0x00000000);
    }
    pack_untilize_uninit(cb_untilize_id);

    // Tell zero_init_writer to exit its send loop — all batches have been untilized.
    push_scalar_signal(cb_stop_signal_id, ROUTE_INFO_SENTINEL);
}
