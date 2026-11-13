// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Batched (up to 8 DST faces at once) reduction across ring_size slots in the active
// ping-pong half of cb_in0, summing slots in chip order.
void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t group_pages = get_compile_time_arg_val(4);

    CircularBuffer cb_meta(cb_meta_id);
    cb_meta.wait_front(1);
    const uint32_t half = ckernel::read_tile_value(cb_meta_id, 0, 0);
    cb_meta.pop_front(1);

    const uint32_t slot_pages = num_blocks * group_pages;
    const uint32_t slot_base = half * slot_pages;
    const uint32_t wait_pages = (half + 1u) * slot_pages;

    CircularBuffer cb_in(cb_in0);
    CircularBuffer cb_out(cb_out0);
    cb_in.wait_front(wait_pages);
    cb_out.reserve_back(group_pages);

    compute_kernel_hw_startup(cb_in0, cb_in0, cb_out0);

    constexpr uint32_t kMaxDst = 8;
    for (uint32_t base = 0; base < group_pages; base += kMaxDst) {
        const uint32_t count = (group_pages - base < kMaxDst) ? (group_pages - base) : kMaxDst;
        copy_init(cb_in0);
        tile_regs_acquire();
        for (uint32_t t = 0; t < count; ++t) {
            copy_tile(cb_in0, slot_base + base + t, t);
        }
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0);
        for (uint32_t block = 1; block < num_blocks; ++block) {
            const uint32_t b_off = slot_base + block * group_pages + base;
            for (uint32_t t = 0; t < count; ++t) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0, b_off + t, t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < count; ++t) {
            pack_tile<true>(t, cb_out0, base + t);
        }
        tile_regs_release();
    }

    cb_out.push_back(group_pages);
    cb_in.pop_front(wait_pages);
}
