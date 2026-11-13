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

#ifdef AGRMS_NOTIFY_STATEFUL
// landing_notify_fanout (perf experiment), raw stateful NoC atomic increment. Bypasses noc_semaphore_inc
// (dataflow_api.h): every call re-programs the whole AT command buffer (ret addr, target lo / mid / coordinate,
// NOC_CTRL, AT_LEN_BE, AT_DATA, CMD_CTRL: ~8 uncached NoC register writes, ~100 ns per increment measured as
// AG_FINAL_NOTIFY / 16 increments) although only the target core changes between the landing notifications of one
// burst. Here the burst-invariant registers (local semaphore address, ctrl, opcode, increment) are written once per
// burst; each increment writes the target coordinate and issues. Same transaction (non-posted INCR_GET, wrap 31,
// unicast write VC, counted in noc_nonposted_atomics_acked so noc_async_atomic_barrier still covers it). The kernel's
// AT command buffer is used by nothing else in between (the reader's NoC reads use the read command buffer).
FORCE_INLINE void notify_inc_set_state(uint32_t l1_addr, uint32_t incr) {
    const uint64_t addr = get_noc_addr(l1_addr);
    while (!noc_cmd_buf_ready(noc_index, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        write_at_cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED | NOC_CMD_AT);
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        write_at_cmd_buf,
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(31) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_AT_DATA, incr);
}
// xy: packed virtual (x | y << 16), as in the common args
FORCE_INLINE void notify_inc_with_state(uint32_t xy) {
    const uint64_t addr = get_noc_addr(xy & 0xFFFF, xy >> 16, 0);
    while (!noc_cmd_buf_ready(noc_index, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        write_at_cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_atomics_acked[noc_index] += 1;
}
// coord: the AT command buffer's target-coordinate register value, precomputed
FORCE_INLINE void notify_inc_coord(uint32_t coord) {
    while (!noc_cmd_buf_ready(noc_index, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_atomics_acked[noc_index] += 1;
}
FORCE_INLINE uint32_t notify_coord_of(uint32_t xy) {
    return (uint32_t)(get_noc_addr(xy & 0xFFFF, xy >> 16, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}
#endif
#if defined(AGRMS_NOTIFY_TABLE) && !(defined(AGRMS_NOTIFY_STATEFUL) && defined(AGRMS_NOTIFY_HOIST))
#error "AGRMS_NOTIFY_TABLE needs AGRMS_NOTIFY_STATEFUL and AGRMS_NOTIFY_HOIST"
#endif

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
    // Perf 1 (pre-send critical path): own items xb < dram_items are read straight from the input tensor, so the first
    // packets do not wait for the members' own-slice load (member launch skew ~1.1 us); the later own items come from
    // the members' landing slots after sem_local_loaded, as before. 2 = the port CB's two batches (every item is
    // >= item_min_chunks chunks). Native HEIGHT reads the own slice in place from its resident shard: nothing to wait.
    constexpr uint32_t dram_items = own_from_input ? 0 : get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
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
#ifdef AGRMS_NOTIFY_MCAST
    const uint32_t n_mc_rects = get_arg_val<uint32_t>(a++);
    const uint32_t mc_idx = a;  // per rectangle: (x0 | y0 << 16), (x1 | y1 << 16), core count (virtual coordinates)
#endif

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
#ifdef AGRMS_NOTIFY_HOIST
    // the members this stream touches (flag 1), once: the per-row loop then visits only them
    uint32_t notified[k];
    uint32_t n_notified = 0;
    for (uint32_t m = 0; m < k; ++m) {
        if (get_arg_val<uint32_t>(cnt_idx + m) != 0) {
            notified[n_notified++] = m;
        }
    }
#endif
#ifdef AGRMS_NOTIFY_TABLE
    // target coordinates of every (group, notified member), in local memory (n_groups * k words at most), memoized
    // per group on its first notification (a group's later rows -- the critical last row-blocks once n_groups < R_t --
    // then cost one coordinate write + issue per member). Cleared just before the first notification.
    constexpr uint32_t NO_COORD = 0xFFFFFFFFu;
    uint32_t coords[n_groups * k];
    bool coords_cleared = false;
    auto clear_coords = [&]() {
        if (!coords_cleared) {
            for (uint32_t i = 0; i < n_groups * n_notified; i += n_notified) {
                coords[i] = NO_COORD;
            }
            coords_cleared = true;
        }
    };
#endif
#ifdef AGRMS_NOTIFY_WRITE
    // landing_notify_fanout: absolute-value notification. Each (member, stream) landed semaphore has exactly one
    // writer (this port), and its value after local row li landed is li + 1 (the flags are 0/1), so a plain 4-byte NoC
    // write of li + 1 replaces the atomic increment: writes pipeline (the command buffer frees at injection, acks are
    // collected by the final write barrier), while each non-posted atomic holds the AT command buffer ~100 ns.
    // Same source, destination and static VC -> the writes to one semaphore arrive in issue order (monotonic).
    // Source words: 16 B-aligned slots (L1 alignment of the destination semaphore) in this port core's otherwise
    // unused cb_x_landing (allocated on the port cores only so its base address matches the members').
    const uint32_t val_tbl = get_write_ptr(agrms_cb::cb_x_landing);
#endif
    auto notify_row = [&](uint32_t grp, uint32_t li) {
#ifdef AGRMS_NOTIFY_TABLE
        uint32_t* cr = coords + grp * n_notified;
        if (n_notified != 0 && cr[0] == NO_COORD) {
            for (uint32_t j = 0; j < n_notified; ++j) {
                cr[j] = notify_coord_of(get_common_arg_val<uint32_t>(grp * k + notified[j]));
            }
        }
        for (uint32_t j = 0; j < n_notified; ++j) {
            notify_inc_coord(cr[j]);
        }
        return;
#endif
#ifdef AGRMS_NOTIFY_WRITE
        const uint32_t src = val_tbl + li * 16;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src) = li + 1;  // same value every time: no in-flight hazard
#endif
#ifdef AGRMS_NOTIFY_HOIST
        for (uint32_t j = 0; j < n_notified; ++j) {
            const uint32_t m = notified[j];
            {
                constexpr uint32_t c = 1;
#else
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t c = get_arg_val<uint32_t>(cnt_idx + m);
            if (c != 0) {
#endif
                const uint32_t xy = get_common_arg_val<uint32_t>(grp * k + m);
#ifdef AGRMS_NOTIFY_WRITE
#ifndef AGRMS_NOTIFY_REPEAT
#define AGRMS_NOTIFY_REPEAT 1
#endif
                for (uint32_t rep = 0; rep < AGRMS_NOTIFY_REPEAT; ++rep) {  // probe only: cost per write
                    noc_async_write_one_packet(src, get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), 4);
                }
#elif defined(AGRMS_NOTIFY_STATEFUL)
                notify_inc_with_state(xy);  // c is a 0/1 flag: the burst state increments by 1
#else
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), c);
#endif
            }
        }
    };
    auto notify_rows = [&](uint32_t xb) {
#ifdef AGRMS_NOTIFY_MCAST
        // one multicast +1 per rectangle of the active members: sem_landed[stream] counts this stream's row-blocks
        for (uint32_t i = 0; i < n_mc_rects; ++i) {
            const uint32_t lo = get_arg_val<uint32_t>(mc_idx + 3 * i);
            const uint32_t hi = get_arg_val<uint32_t>(mc_idx + 3 * i + 1);
            const uint32_t n = get_arg_val<uint32_t>(mc_idx + 3 * i + 2);
            const uint64_t mc = noc_index == 0
                                    ? get_noc_multicast_addr(lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16, landed_addr)
                                    : get_noc_multicast_addr(hi & 0xFFFF, hi >> 16, lo & 0xFFFF, lo >> 16, landed_addr);
            noc_semaphore_inc_multicast(mc, 1, n);
        }
        return;
#endif
        const uint32_t r0 = xb * xfer_rows;
        const uint32_t rows = g.item_rows(xb);
        uint32_t grp, li;
#ifdef AGRMS_NOTIFY_REVERSE
        for (uint32_t r = r0 + rows; r-- > r0;) {
            g.place(r, grp, li);
            notify_row(grp, li);
        }
#elif defined(AGRMS_NOTIFY_PLACE)
        for (uint32_t r = r0; r < r0 + rows; ++r) {  // probe: forward order, placement recomputed per row
            g.place(r, grp, li);
            notify_row(grp, li);
        }
#else
    g.place(r0, grp, li);
    for (uint32_t r = r0; r < r0 + rows; ++r, g.advance(grp, li)) {
        notify_row(grp, li);
    }
#endif
    };
    // upstream complete through step s_done: notify every row-block whose last incoming item is in it
    auto notify_until = [&](int32_t s_done) {
#ifdef AGRMS_NOTIFY_STATEFUL
        bool armed = false;
#endif
        while (next_notify < n_xb &&
               static_cast<int32_t>(next_notify + static_cast<uint32_t>(last_in) * relay_lag) <= s_done) {
#ifdef AGRMS_NOTIFY_STATEFUL
            if (!armed) {
#ifdef AGRMS_NOTIFY_TABLE
                clear_coords();
#endif
                notify_inc_set_state(landed_addr, 1);
                armed = true;
            }
#endif
            notify_rows(next_notify++);
        }
    };
#ifdef AGRMS_NOTIFY_EARLY
    // non-blocking: notify every row-block whose last incoming item has already landed (polled between items)
    auto try_notify = [&]() {
        while (has_in && next_notify < n_xb) {
            const int32_t s = static_cast<int32_t>(next_notify + static_cast<uint32_t>(last_in) * relay_lag);
            invalidate_l1_cache();
            if (*arrived < upstream_through(s)) {
                break;
            }
            AGRMS_ZONE("AG_EARLY_NOTIFY");
            notify_until(s);
        }
    };
#endif

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

    // own item from the input tensor (same chunks, same order as the sender's walk over the landing slots)
    const auto x_acc = TensorAccessor(x_args, x_addr, page_bytes);
    auto stream_item_input = [&](const Entry& e, uint32_t xb) {
        for_each_chunk_of_item_src(g, e, xb, [&](uint32_t r, uint32_t col, uint32_t bytes, bool last) {
            if (batch == 0) {
                need = group - pos % group;
                cb_reserve_back(cb, run_pages * need);
                wptr = get_write_ptr(cb);
            }
            uint32_t dst = wptr + batch * chunk_bytes;
            const uint32_t page0 = r * Wt_lp + col;
            for (uint32_t b = 0; b < bytes; b += page_bytes, dst += page_bytes) {
#ifndef AGRMS_ABLATE_NOREAD
                noc_async_read(x_acc.get_noc_addr(page0 + b / page_bytes), dst, page_bytes);
#endif
            }
            if (++batch == need || last) {
                noc_async_read_barrier();
                cb_push_back(cb, run_pages * batch);
                pos += batch;
                batch = 0;
            }
        });
    };

    if (has_out) {
        // the own slice is in the members' landing slots once every local member has loaded it
        bool local_ready = false;
        auto await_local = [&]() {
            if (!local_ready) {
                AGRMS_ZONE("AG_LOCAL_WAIT");
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local_loaded_id)),
                    local_loaded_expected);
                local_ready = true;
            }
        };
        const uint32_t steps = n_xb + (E_out - 1) * relay_lag;
        const Entry own = out_entry(0);
        for (uint32_t t = 0; t < steps; ++t) {
#ifdef AGRMS_NOTIFY_EARLY
            try_notify();
#endif
            if (t < n_xb && !own.empty()) {
                if (t < dram_items) {
                    AGRMS_ZONE("AG_INPUT_ITEM");
                    stream_item_input(own, t);
                } else {
                    await_local();
                    stream_item(own, t, own_from_input);
                }
            }
            if (has_in && t >= relay_lag) {
                const int32_t s = static_cast<int32_t>(t - relay_lag);
                {
                    AGRMS_ZONE("AG_ARRIVAL_WAIT");  // wait only (the notify fan-out has its own zone)
                    noc_semaphore_wait_min(arrived, upstream_through(s));
                }
#ifndef AGRMS_NOTIFY_LATE
                AGRMS_ZONE("AG_STEP_NOTIFY");
                notify_until(s);
#endif
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
#ifdef AGRMS_NOTIFY_EARLY
                        try_notify();
#endif
                    }
                }
            }
#ifdef AGRMS_NOTIFY_LATE
            // the step's relays are pushed to the sender first (downstream chips' critical transport), then the
            // landing notifications of the blocks the same arrival wait completed
            if (has_in && t >= relay_lag) {
                AGRMS_ZONE("AG_STEP_NOTIFY");
                notify_until(static_cast<int32_t>(t - relay_lag));
            }
#endif
        }
    }
    if (has_in) {
#ifdef AGRMS_NOTIFY_SPLIT_FINAL
        // each remaining row-block as soon as its own last item has landed (not after the whole stream)
        while (next_notify < n_xb) {
            const int32_t s = static_cast<int32_t>(next_notify + static_cast<uint32_t>(last_in) * relay_lag);
            {
                AGRMS_ZONE("AG_FINAL_ARRIVAL");
                noc_semaphore_wait_min(arrived, upstream_through(s));
            }
            AGRMS_ZONE("AG_FINAL_NOTIFY");
            notify_until(s);
        }
#endif
        {
            AGRMS_ZONE("AG_FINAL_ARRIVAL");  // the last relayed items from upstream (wait only)
            noc_semaphore_wait_min(arrived, total_in);
        }
        AGRMS_ZONE("AG_FINAL_NOTIFY");
        notify_until(0x7FFFFFFF);
    }
    noc_async_atomic_barrier();
#ifdef AGRMS_NOTIFY_WRITE
    noc_async_write_barrier();
#endif
}
