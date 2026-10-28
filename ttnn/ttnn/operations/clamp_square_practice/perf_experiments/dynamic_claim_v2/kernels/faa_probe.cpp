// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// dynamic_claim_v2 DIAGNOSTIC (not a production kernel): claims-only fetch-and-add stress probe.
// Every core repeatedly does NOC_AT_INS_INCR_GET(+1) on one shared counter (semaphore 0 on the counter
// core) and logs each returned value, until it sees a value >= total. Host checks that the multiset of
// returned values < total is exactly {0..total-1}.
// Defines: RET_MODE 0 = read the return word right after noc_async_atomic_barrier()
//                   1 = poll until the return word changes (round-1 protocol)
//          TRAFFIC  0 = none; 1 = each claim is followed by reading 8 tiles of DRAM into L1
//          OWN_RET  0 = NoC returns to semaphore 1; 1 = returns to a word in the log CB
// CT: [cb_log, cb_traffic, page_bytes, accessor(src)..., accessor(log)...]
// RT: [counter_x, counter_y, total, src_addr, src_pages, log_addr, core_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef RET_MODE
#define RET_MODE 1
#endif
#ifndef TRAFFIC
#define TRAFFIC 0
#endif

constexpr uint32_t LOG_PAGE = 2048;
constexpr uint32_t LOG_MAX = LOG_PAGE / 4 - 1;

void kernel_main() {
    constexpr uint32_t cb_log = get_compile_time_arg_val(0);
    constexpr uint32_t cb_traffic = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto log_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t counter_x = get_arg_val<uint32_t>(0);
    const uint32_t counter_y = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);
    const uint32_t src_addr = get_arg_val<uint32_t>(3);
    const uint32_t src_pages = get_arg_val<uint32_t>(4);
    const uint32_t log_addr = get_arg_val<uint32_t>(5);
    const uint32_t core_id = get_arg_val<uint32_t>(6);

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    const auto logacc = TensorAccessor(log_args, log_addr, LOG_PAGE);

    volatile tt_l1_ptr uint32_t* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_log));
    const uint32_t ret_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr);
    const uint64_t counter_noc = get_noc_addr(counter_x, counter_y, get_semaphore(0));

    uint32_t n = 0;
    uint32_t prev = 0xFFFFFFFFu;  // semaphore 1 initial value
    uint32_t tpage = core_id * 8;
    for (;;) {
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index, write_at_cmd_buf, counter_noc, NOC_UNICAST_WRITE_VC, 1, 31, false, false, ret_addr);
        noc_async_atomic_barrier();
        uint32_t v;
#if RET_MODE == 0
        invalidate_l1_cache();
        v = *ret;
#else
        uint32_t spins = 0;
        do {
            invalidate_l1_cache();
            v = *ret;
        } while (v == prev && ++spins < 1000000);
#endif
        prev = v;
        if (n < LOG_MAX) {
            log[1 + n] = v;
        }
        ++n;
        if (v >= total) {
            break;
        }
#if TRAFFIC
        uint32_t l1 = get_write_ptr(cb_traffic);
        for (uint32_t i = 0; i < 8; ++i) {
            noc_async_read_page(tpage % src_pages, src, l1);
            l1 += page_bytes;
            ++tpage;
        }
        noc_async_read_barrier();
#endif
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_RET_ADDR_LO, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    log[0] = n;
    // land the RISC stores before the NoC reads the page (read-back fence)
    do {
        invalidate_l1_cache();
    } while (log[0] != n);
    noc_async_write(get_write_ptr(cb_log), logacc.get_noc_addr(core_id), LOG_PAGE);
    noc_async_write_barrier();
}
