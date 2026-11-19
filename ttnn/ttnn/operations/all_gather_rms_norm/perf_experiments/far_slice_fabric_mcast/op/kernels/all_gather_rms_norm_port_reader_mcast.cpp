// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader, FABRIC MULTICAST variant (perf experiment far_slice_fabric_mcast).
// One per (direction, link) per chip. Two jobs, no relays:
//  - stream this chip's own slice (this link's columns, every send entry) from the members' landing slots into the
//    sender's chunk CB, in the sender's order (step t: item t of every entry);
//  - announce landed rows of the incoming stream to the members. The stream has one source per upstream chip that
//    reaches this chip (1-hop unicast and/or multicast from p-1, multicast from p-2, ...), each counting its items on
//    its own arrival counter SEM_ARR_BASE + source position (an mcast packet increments the same address on every
//    chip it lands on). Row-block xb is complete once every source's counter covers its items through step xb; then
//    sem_landed[stream] is bumped for every member the stream touches, for each row of the block (row-ascending).
// Notifications are polled between batches (non-blocking) so the own-slice streaming never waits on arrivals.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"
#include "all_gather_rms_norm_port_walk.hpp"

using namespace agrms_port;

void kernel_main() {
#if defined(AGRMS_ABLATE_EMPTY) || defined(AGRMS_ABLATE_LOCAL)
    return;
#endif
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr uint32_t R_t = get_compile_time_arg_val(3);
    constexpr uint32_t n_groups = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t w_pitch = get_compile_time_arg_val(6);
    constexpr uint32_t xfer_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Wt_lp = get_compile_time_arg_val(8);
    constexpr uint32_t rows_per_group = get_compile_time_arg_val(9);
    constexpr uint32_t sem_local_loaded_id = get_compile_time_arg_val(10);
    constexpr bool own_from_input = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t cb = agrms_cb::cb_port_chunks;
    constexpr uint32_t MAX_SRC = 8;

    uint32_t a = 0;
    const uint32_t local_loaded_expected = get_arg_val<uint32_t>(a++);
    const uint32_t landed_sem_id = get_arg_val<uint32_t>(a++);
    const uint32_t E_out = get_arg_val<uint32_t>(a++);
    const uint32_t out_idx = a;
    a += E_out;
    const uint32_t segidx_idx = a;
    a += E_out;
    const uint32_t n_segs = get_arg_val<uint32_t>(a++);
    const uint32_t segs_idx = a;
    a += n_segs;
    // incoming sources: n_src, then per source: arrival semaphore id, entry count, packed entries
    const uint32_t n_src = get_arg_val<uint32_t>(a++);
    uint32_t src_sem[MAX_SRC], src_ne[MAX_SRC], src_ent[MAX_SRC];
    for (uint32_t s = 0; s < n_src; ++s) {
        src_sem[s] = get_arg_val<uint32_t>(a++);
        src_ne[s] = get_arg_val<uint32_t>(a++);
        src_ent[s] = a;
        a += src_ne[s];
    }
    const uint32_t cnt_idx = a;
    a += k;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);

    const Geom g{
        page_bytes,
        run_pages,
        R_t,
        n_groups,
        k,
        w_pitch,
        xfer_rows,
        Wt_lp,
        get_write_ptr(agrms_cb::cb_x_landing),
        rows_per_group,
        x_addr};
    const uint32_t n_xb = g.n_xb();
    const uint32_t landed_addr = get_semaphore(landed_sem_id);

    auto out_entry = [&](uint32_t e) {
        return unpack_entry(get_arg_val<uint32_t>(out_idx + e), get_arg_val<uint32_t>(segidx_idx + e), segs_idx);
    };

    // items of source s through step xb (inclusive)
    auto src_through = [&](uint32_t s, uint32_t xb) {
        uint32_t n = 0;
        for (uint32_t e = 0; e < src_ne[s]; ++e) {
            n += g.items_before(unpack_entry(get_arg_val<uint32_t>(src_ent[s] + e)), xb + 1);
        }
        return n;
    };
    auto block_landed = [&](uint32_t xb) {
        for (uint32_t s = 0; s < n_src; ++s) {
            volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(src_sem[s]));
            if (*c < src_through(s, xb)) {
                return false;
            }
        }
        return true;
    };

    auto notify_rows = [&](uint32_t xb) {
        const uint32_t r0 = xb * xfer_rows;
        const uint32_t rows = g.item_rows(xb);
        uint32_t grp, li;
        g.place(r0, grp, li);
        for (uint32_t r = r0; r < r0 + rows; ++r, g.advance(grp, li)) {
            for (uint32_t m = 0; m < k; ++m) {
                const uint32_t c = get_arg_val<uint32_t>(cnt_idx + m);
                if (c != 0) {
                    const uint32_t xy = get_common_arg_val<uint32_t>(grp * k + m);
                    noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), c);
                }
            }
        }
    };
    uint32_t next_notify = n_src > 0 ? 0 : n_xb;
    auto poll_notify = [&]() {
        while (next_notify < n_xb && block_landed(next_notify)) {
            notify_rows(next_notify++);
        }
    };

    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    auto stream_item = [&](const Entry& e, uint32_t xb, bool from_input) {
        for_each_chunk_of_item(
            g,
            e,
            xb,
            [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
                if (batch == 0) {
                    need = group - pos % group;
                    while (!cb_pages_reservable_at_back(cb, run_pages * need)) {
                        poll_notify();
                    }
                    cb_reserve_back(cb, run_pages * need);
                    wptr = get_write_ptr(cb);
                }
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
#ifndef AGRMS_ABLATE_NOREAD
                noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, addr), wptr + batch * chunk_bytes, bytes);
#endif
                if (++batch == need || last) {
                    noc_async_read_barrier();
                    cb_push_back(cb, run_pages * batch);
                    pos += batch;
                    batch = 0;
                }
            },
            from_input);
    };

    bool has_out = false;
    for (uint32_t e = 0; e < E_out; ++e) {
        has_out |= !out_entry(e).empty();
    }
    if (has_out) {
        {
            AGRMS_ZONE("AG_LOCAL_WAIT");
            volatile tt_l1_ptr uint32_t* ll =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local_loaded_id));
            while (*ll < local_loaded_expected) {
                poll_notify();
            }
        }
        AGRMS_ZONE("AG_STREAM_OWN");
        for (uint32_t t = 0; t < n_xb; ++t) {
            for (uint32_t e = 0; e < E_out; ++e) {
                const Entry en = out_entry(e);
                if (!en.empty()) {
                    stream_item(en, t, own_from_input);
                }
            }
            poll_notify();
        }
    }
    {
        AGRMS_ZONE("AG_FINAL_ARRIVAL");
        while (next_notify < n_xb) {
            poll_notify();
        }
    }
    noc_async_atomic_barrier();
}
