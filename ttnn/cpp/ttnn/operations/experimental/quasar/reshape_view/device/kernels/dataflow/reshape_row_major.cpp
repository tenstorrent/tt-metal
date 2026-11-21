// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar reshape, ROW_MAJOR layout: T DM harts of one kernel, no DataflowBuffer.
//
// A row-major reshape keeps the flat byte stream and only moves the page boundaries: source page s holds
// stream bytes [s * src_page_bytes, (s + 1) * src_page_bytes) and destination page d holds
// [d * dst_page_bytes, (d + 1) * dst_page_bytes). The stream is cut into windows (host: whole destination
// pages when several fit in a window, else window_bytes pieces of one destination page). For each of its
// windows a hart
//   1. gathers the window from the source pages that overlap it, one NOC read per overlap, straight into
//      its private stage (the Quasar NOC reads and writes at byte granularity, so a read starts at any
//      byte of a source page and lands at any byte of the stage),
//   2. waits for those reads, and
//   3. scatters the window to the destination pages it covers, one NOC write per overlap.
// No RISC ever touches the payload.
//
// Hart t of T owns windows t, t + T, ... of this node's range and tags its transfers with transaction id
// t + 1, so its barrier waits on its own transfers only. Each hart ping-pongs between two stage halves:
// the writes of window k drain while the reads of window k + 1 fill the other half, and the one barrier
// per window covers both.
//
// Compile-time args: src_page_bytes, dst_page_bytes, window_bytes, windows_per_page, pages_per_window,
//                    num_dst_pages
// Runtime args:      window_start, num_windows

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t src_page_bytes = get_arg(args::src_page_bytes);
    constexpr uint32_t dst_page_bytes = get_arg(args::dst_page_bytes);
    constexpr uint32_t window_bytes = get_arg(args::window_bytes);
    // > 1: a destination page is wider than a window and is moved in window_bytes pieces.
    constexpr uint32_t windows_per_page = get_arg(args::windows_per_page);
    // Whole destination pages per window when windows_per_page == 1.
    constexpr uint32_t pages_per_window = get_arg(args::pages_per_window);
    constexpr uint32_t num_dst_pages = get_arg(args::num_dst_pages);

    const uint32_t window_start = get_arg(args::window_start);
    const uint32_t num_windows = get_arg(args::num_windows);

    const uint32_t t = get_my_thread_id();
    const uint32_t T = get_num_threads();

    Noc noc;
    const auto src = TensorAccessor(tensor::input);
    const auto dst = TensorAccessor(tensor::output);
    // One Scratchpad per node: hart t owns bytes [t * 2 * window_bytes, (t + 1) * 2 * window_bytes).
    Scratchpad<uint8_t> stage(scratch::stage);
    const uint32_t stage_base = t * 2 * window_bytes;
    const NocOptVals txn{.trid = t + 1};

    uint32_t half = 0;
    for (uint32_t w = t; w < num_windows; w += T) {
        const uint32_t window = window_start + w;
        uint64_t first;  // stream offset of the window
        uint32_t len;
        if constexpr (windows_per_page > 1) {
            const uint32_t in_page = (window % windows_per_page) * window_bytes;
            first = static_cast<uint64_t>(window / windows_per_page) * dst_page_bytes + in_page;
            len = dst_page_bytes - in_page < window_bytes ? dst_page_bytes - in_page : window_bytes;
        } else {
            const uint32_t page = window * pages_per_window;
            const uint32_t pages = num_dst_pages - page < pages_per_window ? num_dst_pages - page : pages_per_window;
            first = static_cast<uint64_t>(page) * dst_page_bytes;
            len = pages * dst_page_bytes;
        }
        const uint32_t base = stage_base + half * window_bytes;

        for (uint32_t done = 0; done < len;) {
            const uint64_t at = first + done;
            const uint32_t offset = at % src_page_bytes;
            const uint32_t n = src_page_bytes - offset < len - done ? src_page_bytes - offset : len - done;
            noc.async_read<NocOptions::TXN_ID>(
                src,
                stage,
                n,
                {.page_id = static_cast<uint32_t>(at / src_page_bytes), .offset_bytes = offset},
                {.offset_bytes = base + done},
                txn);
            done += n;
        }
        // Also covers the previous window's writes, whose stage half the next window reads into.
        noc.async_read_barrier<NocOptions::TXN_ID>(txn);

        for (uint32_t done = 0; done < len;) {
            const uint64_t at = first + done;
            const uint32_t offset = at % dst_page_bytes;
            const uint32_t n = dst_page_bytes - offset < len - done ? dst_page_bytes - offset : len - done;
            noc.async_write<NocOptions::TXN_ID>(
                stage,
                dst,
                n,
                {.offset_bytes = base + done},
                {.page_id = static_cast<uint32_t>(at / dst_page_bytes), .offset_bytes = offset},
                txn);
            done += n;
        }
        half ^= 1;
    }
    noc.async_write_barrier<NocOptions::TXN_ID>(txn);
}
