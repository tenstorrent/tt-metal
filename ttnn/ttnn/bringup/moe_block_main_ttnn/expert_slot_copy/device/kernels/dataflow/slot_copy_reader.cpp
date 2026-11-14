// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert_slot_copy reader (RISCV_1, NoC0): per job, its run table into CB_TAB (shared with the writer), then this
// part's chunks from the source chip's DRAM into CB_DATA (ESC_DEPTH pages of ESC_CHUNK bytes), up to ESC_BATCH chunks
// a read barrier (a batch never wraps the CB's end: the reader is its only producer and tracks the write page).
// Compile-time args: 0 CB_DATA, 1 CB_TAB, 2 table bytes (a page of runs), then the TensorAccessorArgs of runs.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "slot_copy_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_data = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(1);
    constexpr uint32_t tab_bytes = get_compile_time_arg_val(2);
    constexpr auto runs_args = TensorAccessorArgs<3>();

    const uint32_t src[2] = {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1)};
    const auto runs = TensorAccessor(runs_args, get_arg_val<uint32_t>(4), tab_bytes);
    const uint32_t n_jobs = get_arg_val<uint32_t>(7);
    uint32_t wpos = 0;  // CB_DATA's write page after the pending chunks

    for (uint32_t j = 0; j < n_jobs; ++j) {
        const esc::Job jb = esc::job(j);
        cb_reserve_back(cb_tab, 1);
        const uint32_t tab_l1 = get_write_ptr(cb_tab);
        noc_async_read_page(jb.table, runs, tab_l1);
        noc_async_read_barrier();
        cb_push_back(cb_tab, 1);
        const volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(tab_l1);
        uint32_t pending = 0;
        esc::for_my_chunks(
            tab, jb.part, jb.parts, [&](uint32_t w, uint32_t sb, uint32_t so, uint32_t, uint32_t, uint32_t len) {
                cb_reserve_back(cb_data, pending + 1);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(sb, src[w] + so),
                    get_write_ptr(cb_data) + pending * ESC_CHUNK,
                    len);
                ++pending;
                wpos = wpos + 1 == ESC_DEPTH ? 0 : wpos + 1;
                if (pending == ESC_BATCH || wpos == 0) {
                    noc_async_read_barrier();
                    cb_push_back(cb_data, pending);
                    pending = 0;
                }
            });
        if (pending) {
            noc_async_read_barrier();
            cb_push_back(cb_data, pending);
        }
    }
}
