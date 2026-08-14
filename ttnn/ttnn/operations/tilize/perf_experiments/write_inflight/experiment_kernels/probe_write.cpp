// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// write_inflight DIAGNOSTIC probe — the WRITE STAGE ALONE, nothing upstream.
//
// *** THIS PROBE IS DELIBERATELY NOT CORRECTNESS-CHECKED. ***
// It writes whatever bytes happen to be sitting in an L1 scratch window into the
// destination buffer. Its ONLY purpose is to establish the achievable
// L1 -> interleaved-DRAM write bandwidth roofline for the traffic pattern the
// tilize writer actually issues (N cores x whole TILE pages). Every candidate
// arm of the bake-off IS correctness-gated; this file is the ceiling instrument.
//
// No reader, no compute, no CB handshake: one kernel per core issuing
// `num_pages` writes of `page_bytes` each, so the number it produces is a pure
// fabric+issue number with no producer to be starved by.
//
// Destination page stream (matches the op's writer, tilize_writer.cpp):
//   page(p) = start_page + (p / run_len) * run_stride + (p % run_len)
// i.e. runs of `run_len` CONSECUTIVE tile pages (one block of WT_CHUNK tiles),
// the next run `run_stride` pages later (the next tile-row, stride = WT).
// run_len = num_pages, run_stride = 0 gives one fully contiguous run.
//
// Axes:
//   * `ppb`  — pages per write barrier: the IN-FLIGHT WINDOW.
//   * `mode` — how the writes are issued (plain / trid double-issue / VC spread
//              / both NoCs).
//   * page_bytes, core count, page stream shape — from the host.

#include "api/dataflow/dataflow_api.h"

#define M_PLAIN 0
#define M_TRID 1
#define M_VC 2
#define M_DUALNOC 3

void kernel_main() {
    constexpr uint32_t cb_scratch = 0;

    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t ppb = get_compile_time_arg_val(1);  // pages per barrier group
    constexpr uint32_t window_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mode = get_compile_time_arg_val(3);
    constexpr uint32_t ahead = get_compile_time_arg_val(4);  // M_TRID: groups left outstanding
    constexpr uint32_t run_len = get_compile_time_arg_val(5);
    constexpr uint32_t run_stride = get_compile_time_arg_val(6);
    // 1 = every write is sourced from the SAME L1 address (the padded plan's
    // whole-pad tiles, which all come from one pre-stamped scratch tile).
    constexpr uint32_t fixed_src = get_compile_time_arg_val(7);
    // Issue each destination PAGE as `split` sub-transactions of page_bytes/split
    // at consecutive offsets. 1 = one whole page per transaction (master.md B5).
    // This holds the destination BANK STREAM fixed and varies only the
    // transaction size, which the G axis (different page sizes -> different bank
    // maps) cannot separate.
    constexpr uint32_t split = get_compile_time_arg_val(8);
    constexpr auto dst_args = TensorAccessorArgs<9>();
    constexpr uint32_t sub_bytes = page_bytes / split;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    if (num_pages == 0) {
        return;
    }
    const auto acc = TensorAccessor(dst_args, dst_addr);
    const uint32_t base = get_write_ptr(cb_scratch);

    // p -> destination page id, the op's own (run_len, run_stride) stream.
    auto page_of = [&](uint32_t p) -> uint32_t {
        if constexpr (run_stride == 0) {
            return start_page + p;
        } else {
            const uint32_t run = p / run_len;
            return start_page + run * run_stride + (p - run * run_len);
        }
    };

    uint32_t l1 = base;
    auto bump = [&]() {
        if constexpr (fixed_src) {
            return;  // one pre-stamped source tile for every page
        } else {
            l1 += page_bytes;
            if (l1 - base + page_bytes > window_bytes) {
                l1 = base;
            }
        }
    };

    if constexpr (mode == M_TRID) {
        // Rotating transaction ids: `ahead` groups stay outstanding across the
        // barrier, so the write NoC is never drained at a group boundary. trids
        // 1..ahead+1 (0 means "untagged", so it is never used as a slot).
        constexpr uint32_t n_trid = ahead + 1;
        static_assert(n_trid <= 15, "trid ring must fit NOC_MAX_TRANSACTION_ID");
        uint32_t issue_slot = 0, wait_slot = 0, outstanding = 0;
        for (uint32_t p = 0; p < num_pages;) {
            noc_async_write_set_trid(issue_slot + 1);
            const uint32_t end = (p + ppb <= num_pages) ? p + ppb : num_pages;
            for (; p < end; ++p) {
                noc_async_write(l1, acc.get_noc_addr(page_of(p)), page_bytes);
                bump();
            }
            issue_slot = (issue_slot + 1 == n_trid) ? 0 : issue_slot + 1;
            ++outstanding;
            if (outstanding > ahead) {
                noc_async_write_barrier_with_trid(wait_slot + 1);
                wait_slot = (wait_slot + 1 == n_trid) ? 0 : wait_slot + 1;
                --outstanding;
            }
        }
        while (outstanding) {
            noc_async_write_barrier_with_trid(wait_slot + 1);
            wait_slot = (wait_slot + 1 == n_trid) ? 0 : wait_slot + 1;
            --outstanding;
        }
        // MANDATORY: brisck.cc asserts the packet tags are cleared after
        // kernel_main returns; a left-behind trid halts the core in firmware.
        noc_async_write_set_trid(0);
        return;
    }

    if constexpr (mode == M_DUALNOC) {
        // Both NoCs from the writer RISC. get_noc_addr() takes the noc index
        // because NOC1's coordinate space is mirrored on Wormhole.
        uint32_t in_group = 0;
        for (uint32_t p = 0; p < num_pages; ++p) {
            const uint8_t n = (uint8_t)(p & 1);
            noc_async_write(l1, acc.get_noc_addr(page_of(p), 0, n), page_bytes, n);
            bump();
            if (++in_group == ppb) {
                noc_async_write_barrier(0);
                noc_async_write_barrier(1);
                in_group = 0;
            }
        }
        noc_async_write_barrier(0);
        noc_async_write_barrier(1);
        return;
    }

    // M_PLAIN / M_VC
    uint32_t in_group = 0;
    for (uint32_t p = 0; p < num_pages; ++p) {
        if constexpr (mode == M_VC) {
            // Unicast VCs are 0-3 on Wormhole (4/5 are the multicast VCs); the
            // default for every write in the op is NOC_UNICAST_WRITE_VC == 1.
            noc_async_write(l1, acc.get_noc_addr(page_of(p)), page_bytes, noc_index, p & 3u);
        } else if constexpr (split == 1) {
            noc_async_write(l1, acc.get_noc_addr(page_of(p)), page_bytes);
        } else {
            const uint32_t pg = page_of(p);
            for (uint32_t s = 0; s < split; ++s) {
                noc_async_write(l1 + s * sub_bytes, acc.get_noc_addr(pg, s * sub_bytes), sub_bytes);
            }
        }
        bump();
        if (++in_group == ppb) {
            noc_async_write_barrier();
            in_group = 0;
        }
    }
    noc_async_write_barrier();
}
