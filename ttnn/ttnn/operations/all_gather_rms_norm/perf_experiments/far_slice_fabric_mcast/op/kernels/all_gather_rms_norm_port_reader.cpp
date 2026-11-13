// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader (NCRISC / NoC0), one per (direction, link) per chip.
// Adapted from the fabric_all_gather example's reader (batch / barrier / push discipline, arrival waits), with the
// sources moved from DRAM into the members' L1: the chip's own slice is read from the members' cb_x_landing (loaded
// there by the member readers, one DRAM read for both directions), relayed slices from the same landing slots after
// the upstream port's arrival counter covers them.
//
// Item order (identical in the sender and, on the upstream chip, the producer of my arrival counter): step t sends
// own item t, then relay entry e's item t - e * relay_lag. A relay item (e, xb) is upstream item (e - 1, xb), sent
// by the upstream at its step t - relay_lag, so one wait per step ("upstream complete through step t - lag") covers
// every relay of the step.
// Landing notifications (TRANSPORT -> member): row-block xb is complete on this stream once the upstream's last
// non-empty entry sent item xb; the reader then bumps sem_landed[stream] by 1 for every member of every row in it
// that the stream touches at all (host flag; row-filtered entries make the per-row entry count vary, a flag does not),
// row-ascending, so a member's local row i is complete on this stream iff sem_landed >= i + 1.

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
    constexpr uint32_t group = get_compile_time_arg_val(2);  // chunk slots per batch (the CB holds 2 x group)
    constexpr uint32_t R_t = get_compile_time_arg_val(3);
    constexpr uint32_t n_groups = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t w_pitch = get_compile_time_arg_val(6);
    constexpr uint32_t xfer_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Wt_lp = get_compile_time_arg_val(8);
    constexpr uint32_t relay_lag = get_compile_time_arg_val(9);
    constexpr uint32_t sem_arrival_id = get_compile_time_arg_val(10);
    constexpr uint32_t sem_local_loaded_id = get_compile_time_arg_val(11);
    constexpr uint32_t rows_per_group = get_compile_time_arg_val(12);   // 0: round-robin rows (see Geom::place)
    constexpr bool own_from_input = get_compile_time_arg_val(13) != 0;  // native HEIGHT: own slice in the input shard
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t cb = agrms_cb::cb_port_chunks;

    uint32_t a = 0;
    const uint32_t local_loaded_expected = get_arg_val<uint32_t>(a++);
    const uint32_t landed_sem_id = get_arg_val<uint32_t>(a++);
    const uint32_t E_out = get_arg_val<uint32_t>(a++);
    const uint32_t out_idx = a;
    a += E_out;
    const uint32_t segidx_idx = a;  // per send entry: (first segment << 16) | segment count
    a += E_out;
    const uint32_t n_segs = get_arg_val<uint32_t>(a++);
    const uint32_t segs_idx = a;  // member segments of the send entries (see Entry)
    a += n_segs;
    const uint32_t E_in = get_arg_val<uint32_t>(a++);
    const uint32_t in_idx = a;
    a += E_in;
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
    volatile tt_l1_ptr uint32_t* arrived =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_arrival_id));
    const uint32_t landed_addr = get_semaphore(landed_sem_id);

    auto out_entry = [&](uint32_t e) {
        return unpack_entry(get_arg_val<uint32_t>(out_idx + e), get_arg_val<uint32_t>(segidx_idx + e), segs_idx);
    };
    auto in_entry = [&](uint32_t e) { return unpack_entry(get_arg_val<uint32_t>(in_idx + e)); };

    bool has_out = false;
    for (uint32_t e = 0; e < E_out; ++e) {
        has_out |= !out_entry(e).empty();
    }
    // incoming: items of the upstream's entries; the last non-empty one closes a row-block
    int32_t last_in = -1;
    uint32_t total_in = 0;
    for (uint32_t e = 0; e < E_in; ++e) {
        const Entry en = in_entry(e);
        if (!en.empty()) {
            last_in = static_cast<int32_t>(e);
            total_in += g.items_before(en, n_xb);
        }
    }
    const bool has_in = last_in >= 0;

    // upstream items sent through upstream step s (item (e, xb) goes out at step xb + e * lag)
    auto upstream_through = [&](int32_t s) {
        uint32_t n = 0;
        for (uint32_t e = 0; e < E_in; ++e) {
            const Entry en = in_entry(e);
            const int32_t c = s - static_cast<int32_t>(e * relay_lag) + 1;
            if (c > 0) {
                n += g.items_before(en, static_cast<uint32_t>(c) < n_xb ? static_cast<uint32_t>(c) : n_xb);
            }
        }
        return n;
    };

    uint32_t next_notify = 0;
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
    // upstream complete through step s_done: notify every row-block whose last incoming item is in it
    auto notify_until = [&](int32_t s_done) {
        while (next_notify < n_xb &&
               static_cast<int32_t>(next_notify + static_cast<uint32_t>(last_in) * relay_lag) <= s_done) {
            notify_rows(next_notify++);
        }
    };

    // Batches end at the next multiple of `group` chunks in the CB (2 x group chunks: a reservation never wraps) or
    // at the end of an item (a relay waits on its upstream; an unpushed tail would hold every downstream back).
    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    auto stream_item = [&](const Entry& e, uint32_t xb, bool from_input) {
        for_each_chunk_of_item(
            g,
            e,
            xb,
            [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
                if (batch == 0) {
                    need = group - pos % group;
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

    if (has_out) {
        // the own slice is in the members' landing slots once every local member has loaded it
        {
            AGRMS_ZONE("AG_LOCAL_WAIT");
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local_loaded_id)),
                local_loaded_expected);
        }
        const uint32_t steps = n_xb + (E_out - 1) * relay_lag;
        const Entry own = out_entry(0);
        for (uint32_t t = 0; t < steps; ++t) {
            if (t < n_xb && !own.empty()) {
                stream_item(own, t, own_from_input);
            }
            if (has_in && t >= relay_lag) {
                const int32_t s = static_cast<int32_t>(t - relay_lag);
                AGRMS_ZONE("AG_ARRIVAL_WAIT");
                noc_semaphore_wait_min(arrived, upstream_through(s));
                notify_until(s);
            }
            for (uint32_t e = 1; e < E_out; ++e) {
                if (t < e * relay_lag) {
                    break;
                }
                const uint32_t xb = t - e * relay_lag;
                if (xb < n_xb) {
                    const Entry en = out_entry(e);
                    if (!en.empty()) {
                        stream_item(en, xb, false);
                    }
                }
            }
        }
    }
    if (has_in) {
        {
            AGRMS_ZONE("AG_FINAL_ARRIVAL");  // the last relayed items from upstream (wait only)
            noc_semaphore_wait_min(arrived, total_in);
        }
        AGRMS_ZONE("AG_FINAL_NOTIFY");
        notify_until(0x7FFFFFFF);
    }
    noc_async_atomic_barrier();
}
