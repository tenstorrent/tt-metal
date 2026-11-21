// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/tensor_accessor.h"

namespace {

constexpr uint32_t kRows = 16;
constexpr uint32_t kNegInfKey = 0x80u;
constexpr uint32_t kSlotBytes = kRows * 2 * sizeof(uint32_t);

FORCE_INLINE void merge_slots(uint32_t half_base, uint32_t first, uint32_t count, uint32_t stride, uint32_t dst_addr) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t r = 0; r < kRows; ++r) {
        uint32_t best = kNegInfKey;
        uint32_t bi = 0;
        for (uint32_t j = 0; j < count; ++j) {
            volatile tt_l1_ptr uint32_t* s =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(half_base + (first + j * stride) * kSlotBytes);
            const uint32_t k = s[2 * r];
            if (k > best) {
                best = k;
                bi = s[2 * r + 1];
            }
        }
        dst[2 * r] = best;
        dst[2 * r + 1] = bi;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    const uint32_t members = get_arg_val<uint32_t>(4);
    const uint32_t wait = get_arg_val<uint32_t>(5);
    const uint32_t lead_x = get_arg_val<uint32_t>(6);
    const uint32_t lead_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t half = get_compile_time_arg_val(0);
    constexpr uint32_t num_workers = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr uint32_t num_groups = get_compile_time_arg_val(3);
    constexpr uint32_t merge_cb = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t reduce_x = get_compile_time_arg_val(6);
    constexpr uint32_t reduce_y = get_compile_time_arg_val(7);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_in = get_compile_time_arg_val(9);
    constexpr uint32_t cb_keys = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();

    const uint32_t merge_base = CircularBuffer(merge_cb).get_write_ptr();
    const uint32_t half_base = merge_base + half * num_workers * kSlotBytes;
    const uint32_t my_slot = half_base + core_id * kSlotBytes;
    volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_slot);

    if constexpr (half == 0) {
        CircularBuffer in(cb_in);
        in.reserve_back(tiles);
        in.push_back(tiles);
    }
    CircularBuffer(cb_keys).wait_front(1);
    volatile tt_l1_ptr uint32_t* kp =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CircularBuffer(cb_keys).get_read_ptr());
    for (uint32_t r = 0; r < kRows; ++r) {
        uint32_t bm = kNegInfKey;
        uint32_t bc = 0;
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t k = kp[(2 * half + (c >> 4)) * 256 + r * 16 + (c & 15)];
            const uint32_t mono = (k - 0x00800000u) >> 14;
            const uint32_t col = (0x3FFFu - (k & 0x3FFFu)) * 32 + c;
            if (mono > bm || (mono == bm && mono != kNegInfKey && col < bc)) {
                bm = mono;
                bc = col;
            }
        }
        slot[2 * r] = bm;
        slot[2 * r + 1] = bm != kNegInfKey ? col0 + bc : 0u;
    }

    const uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    if (members == 0) {
        noc_async_write(my_slot, get_noc_addr(lead_x, lead_y, my_slot), kSlotBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lead_x, lead_y, sem_addr), 1);
        noc_async_atomic_barrier();
        return;
    }
    if (wait > 0) {
        noc_semaphore_wait_min(sem, wait);
    }
    noc_semaphore_set(sem, 0);
    merge_slots(half_base, core_id, members, 1, my_slot);
    if (core_id != 0) {
        noc_async_write(my_slot, get_noc_addr(reduce_x, reduce_y, my_slot), kSlotBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(reduce_x, reduce_y, sem_addr), 1);
        noc_async_atomic_barrier();
        return;
    }

    const uint32_t out_l1 = ((merge_base + 2 * num_workers * kSlotBytes + 63) & ~63u) + half * 64;
    const uint32_t tmp_l1 = out_l1 + 2 * 64 - half * 64 + half * kSlotBytes;
    merge_slots(half_base, 0, num_groups, group, tmp_l1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    volatile tt_l1_ptr uint32_t* tmp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tmp_l1);
    for (uint32_t r = 0; r < kRows; ++r) {
        out[r] = tmp[2 * r + 1];
    }
    const auto dst = TensorAccessor(out_args, out_addr, out_page_bytes);
    noc_async_write(out_l1, dst.get_noc_addr(0, half * kRows * sizeof(uint32_t)), kRows * sizeof(uint32_t));
    noc_async_write_barrier();
}
