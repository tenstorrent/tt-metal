// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port reader (NCRISC, NoC0) — TRANSPORT, adapted from examples/fabric_all_gather's reader.
//
// Items, head-major: for every entry e (one (batch, kv head) K + V slice), for every source k this port sends
// (k = 0: this device's own slice, from the K/V inputs; k > 0: a relay, from this device's landing slot of that
// source, read only once the arrival counter says upstream's item (e, k - 1) has landed). Each item is this port's
// bank share of the entry's K pages then V pages, in chunks; chunks go to the sender through cb_port_chunks in
// batches of up to `group` chunks (a batch also ends at every item end, so a relay never waits on an unpushed tail).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ring_joint_sdpa_chunk_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr auto k_args = TensorAccessorArgs<5>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto lk_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto lv_args = TensorAccessorArgs<lk_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    uint32_t a = 0;
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lk_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t kpe = get_arg_val<uint32_t>(a++);  // K pages per entry
    const uint32_t vpe = get_arg_val<uint32_t>(a++);  // V pages per entry
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t item_chunks = get_arg_val<uint32_t>(a++);  // chunks of one item on this port (host-counted)
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t num_entries = get_arg_val<uint32_t>(a++);
    const uint32_t num_sources = get_arg_val<uint32_t>(a++);
    const uint32_t up_sources = get_arg_val<uint32_t>(a++);  // sources upstream sends on this port's stream
    const uint32_t slots_idx = a;                            // my landing slot of source k (k > 0)
    if (item_chunks == 0 || num_sources == 0) {
        return;
    }

    const auto k_in = TensorAccessor(k_args, k_addr, page_bytes);
    const auto v_in = TensorAccessor(v_args, v_addr, page_bytes);
    const auto lk = TensorAccessor(lk_args, lk_addr, page_bytes);
    const auto lv = TensorAccessor(lv_args, lv_addr, page_bytes);
    const uint32_t k_slot_pages = num_entries * kpe;
    const uint32_t v_slot_pages = num_entries * vpe;
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    auto chunk = [&](uint64_t src, uint32_t n) {
        if (batch == 0) {
            need = group - pos % group;
            cb_reserve_back(cb, run_pages * need);
            wptr = get_write_ptr(cb);
        }
        noc_async_read(src, wptr + batch * chunk_bytes, n * page_bytes);
        --chunks_left;
        if (++batch == need || chunks_left == 0) {
            noc_async_read_barrier();
            cb_push_back(cb, run_pages * batch);
            pos += batch;
            batch = 0;
        }
    };

    for (uint32_t e = 0; e < num_entries; ++e) {
        for (uint32_t k = 0; k < num_sources; ++k) {
            if (k == 0) {
                for_each_chunk(kpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                    chunk(k_in.get_noc_addr(e * kpe + page), n);
                });
                for_each_chunk(vpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                    chunk(v_in.get_noc_addr(e * vpe + page), n);
                });
            } else {
                // upstream increments my counter once per item; its item (e, k - 1) has index e * up_sources + k - 1
                noc_semaphore_wait_min(arrived, e * up_sources + k);
                const uint32_t slot = get_arg_val<uint32_t>(slots_idx + k);
                for_each_chunk(kpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                    chunk(lk.get_noc_addr(slot * k_slot_pages + e * kpe + page), n);
                });
                for_each_chunk(vpe, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n) {
                    chunk(lv.get_noc_addr(slot * v_slot_pages + e * vpe + page), n);
                });
            }
            if (batch > 0) {  // item end: push the tail
                noc_async_read_barrier();
                cb_push_back(cb, run_pages * batch);
                pos += batch;
                batch = 0;
            }
        }
    }
}
