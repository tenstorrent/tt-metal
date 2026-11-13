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
#ifdef AGRMS_PSR_IDLE_NOTIFY
    // port_stream_rate: the landing notifications are drained one tile-row at a time while the reader would otherwise
    // wait (CB space, read barrier, upstream arrival), so the fan-out never delays a read the sender is waiting for.
    // Rows go out in ascending order as before; notify_allowed = the upstream step known complete.
    int32_t notify_allowed = -1;
    uint32_t nn_r = 0;
    auto notify_one_row = [&]() -> bool {
        if (!has_in || nn_r >= R_t ||
            static_cast<int32_t>(nn_r / xfer_rows + static_cast<uint32_t>(last_in) * relay_lag) > notify_allowed) {
            return false;
        }
        uint32_t grp, li;
        g.place(nn_r, grp, li);
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t c = get_arg_val<uint32_t>(cnt_idx + m);
            if (c != 0) {
                const uint32_t xy = get_common_arg_val<uint32_t>(grp * k + m);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), c);
            }
        }
        ++nn_r;
        return true;
    };
    auto reserve_idle = [&](uint32_t pages) {
        while (!cb_pages_reservable_at_back(cb, pages)) {
            notify_one_row();
        }
        cb_reserve_back(cb, pages);
    };
    auto barrier_idle = [&]() {
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            notify_one_row();
        }
        invalidate_l1_cache();
    };
    auto arrival_idle = [&](uint32_t target) {
        while (true) {
            invalidate_l1_cache();
            if (*arrived >= target) {
                break;
            }
            notify_one_row();
        }
    };
#define AGRMS_PSR_RESERVE(pages) reserve_idle(pages)
#define AGRMS_PSR_BARRIER() barrier_idle()
#else
#define AGRMS_PSR_RESERVE(pages) cb_reserve_back(cb, pages)
#define AGRMS_PSR_BARRIER() noc_async_read_barrier()
#endif

    // Batches end at the next multiple of `group` chunks in the CB (2 x group chunks: a reservation never wraps) or
    // at the end of an item (a relay waits on its upstream; an unpushed tail would hold every downstream back).
    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    // port_stream_rate (AGRMS_PSR_RELAY_OVERLAP): an item's last batch may be left in flight (`pend` chunks read, not
    // yet barriered / pushed) so the landing notifications go out while its reads complete; finish_pending() closes it
    // before anything else touches the CB.
    uint32_t pend = 0;
    auto finish_pending = [&]() {
        if (pend) {
            noc_async_read_barrier();
            cb_push_back(cb, run_pages * pend);
            pos += pend;
            pend = 0;
        }
    };
    auto stream_item = [&](const Entry& e, uint32_t xb, bool from_input, bool defer = false) {
        finish_pending();
        for_each_chunk_of_item(
            g,
            e,
            xb,
            [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
                if (batch == 0) {
                    need = group - pos % group;
                    {
                        AGRMS_PZONE("AG_R_RESV");
                        AGRMS_PSR_RESERVE(run_pages * need);
                    }
                    wptr = get_write_ptr(cb);
                }
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
#ifdef AGRMS_PSR_CHUNK_PUSH
                noc_async_read_set_trid(batch + 1);  // one transaction id per chunk slot of the batch
#endif
#ifndef AGRMS_ABLATE_NOREAD
                noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, addr), wptr + batch * chunk_bytes, bytes);
#endif
#ifdef AGRMS_PSR_CHUNK_PUSH
                // port_stream_rate: the batch's reads are all in flight; each chunk is pushed as soon as ITS read lands
                if (++batch == need || last) {
                    noc_async_read_set_trid(0);
                    AGRMS_PZONE("AG_R_BAR");
                    for (uint32_t i = 0; i < batch; ++i) {
                        noc_async_read_barrier_with_trid(i + 1);
                        cb_push_back(cb, run_pages);
                    }
                    pos += batch;
                    batch = 0;
                }
                return;
#endif
                if (last && defer) {
                    pend = batch + 1;
                    batch = 0;
                } else if (++batch == need || last) {
                    {
                        AGRMS_PZONE("AG_R_BAR");
                        AGRMS_PSR_BARRIER();
                    }
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
        finish_pending();
        for_each_chunk_of_item_src(g, e, xb, [&](uint32_t r, uint32_t col, uint32_t bytes, bool last) {
            if (batch == 0) {
                need = group - pos % group;
                AGRMS_PSR_RESERVE(run_pages * need);
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
                AGRMS_PSR_BARRIER();
                cb_push_back(cb, run_pages * batch);
                pos += batch;
                batch = 0;
            }
        });
    };

#ifdef AGRMS_PSR_DESC
    // port_stream_rate: chunk descriptors for the sender, one per cb_port_chunks slot (2 x group), in the port core's
    // own cb_x_landing (allocated on the ports for its address only: nothing lands on a port core). Slot i of the
    // ring belongs to CB chunk slot i, so the CB's own flow control orders the writes against the sender's reads.
    // Word 0: destination L1 address (landing slot on the downstream member), word 1: member NoC xy, word 2: bytes |
    // last-chunk-of-item << 31. The sender then walks nothing.
    volatile tt_l1_ptr uint32_t* desc =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(agrms_cb::cb_x_landing));
    auto put_desc = [&](uint32_t slot, uint32_t land, uint32_t xy, uint32_t bytes, bool last) {
        volatile tt_l1_ptr uint32_t* d = desc + 3 * slot;
        d[0] = land;
        d[1] = xy;
        d[2] = bytes | (last ? 0x80000000u : 0u);
    };
    auto stream_item_desc = [&](const Entry& e, uint32_t xb, bool from_input, bool from_tensor) {
        for_each_chunk_of_item_full(
            g, e, xb, [&](uint32_t w, uint32_t land, uint32_t in, uint32_t r, uint32_t col, uint32_t bytes, bool last) {
                if (batch == 0) {
                    need = group - pos % group;
                    {
                        AGRMS_PZONE("AG_R_RESV");
                        AGRMS_PSR_RESERVE(run_pages * need);
                    }
                    wptr = get_write_ptr(cb);
                }
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
                uint32_t dst = wptr + batch * chunk_bytes;
#ifndef AGRMS_ABLATE_NOREAD
                if (from_tensor) {
                    const uint32_t page0 = r * Wt_lp + col;
                    for (uint32_t b = 0; b < bytes; b += page_bytes, dst += page_bytes) {
                        noc_async_read(x_acc.get_noc_addr(page0 + b / page_bytes), dst, page_bytes);
                    }
                } else {
                    noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, from_input ? in : land), dst, bytes);
                }
#endif
                const uint32_t slot = (pos + batch) % (2 * group);
                put_desc(slot, land, xy, bytes, last);
                if (++batch == need || last) {
                    {
                        AGRMS_PZONE("AG_R_BAR");
                        AGRMS_PSR_BARRIER();
                    }
                    cb_push_back(cb, run_pages * batch);
                    pos += batch;
                    batch = 0;
                }
            });
    };
#ifndef AGRMS_PSR_DESC_HYBRID  // hybrid: the input items keep the source-only walk; the sender walks those itself
#define stream_item_input(en, xb) stream_item_desc(en, xb, false, true)
#endif
#define stream_item(en, xb, fi, ...) stream_item_desc(en, xb, fi, false)
#endif

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
#ifdef AGRMS_PSR_OWN_AHEAD
        // port_stream_rate: own item t + 1 is read before step t's landing notifications (they wait one own item)
        auto read_own = [&](uint32_t t) {
            if (t < n_xb && !own.empty()) {
                if (t < dram_items) {
                    AGRMS_ZONE("AG_INPUT_ITEM");
                    stream_item_input(own, t);
                } else {
                    await_local();
                    stream_item(own, t, own_from_input);
                }
            }
        };
        read_own(0);
#endif
        for (uint32_t t = 0; t < steps; ++t) {
#ifdef AGRMS_PSR_OWN_AHEAD
            if (false) {
#else
            if (t < n_xb && !own.empty()) {
#endif
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
#ifdef AGRMS_PSR_IDLE_NOTIFY
                    arrival_idle(upstream_through(s));
                    notify_allowed = s;
#else
                    noc_semaphore_wait_min(arrived, upstream_through(s));
#endif
                }
#if !defined(AGRMS_PSR_RELAY_FIRST) && !defined(AGRMS_PSR_RELAY_OVERLAP) && !defined(AGRMS_PSR_IDLE_NOTIFY)
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
#ifdef AGRMS_PSR_RELAY_OVERLAP
                        stream_item(en, xb, false, true);
#else
                        stream_item(en, xb, false);
#endif
                    }
                }
            }
#ifdef AGRMS_PSR_RELAY_OVERLAP
            // port_stream_rate: the relay reads of this step are issued; the notifications go out while they complete
            if (has_in && t >= relay_lag) {
                AGRMS_ZONE("AG_STEP_NOTIFY");
                notify_until(static_cast<int32_t>(t - relay_lag));
            }
            finish_pending();
#endif
#ifdef AGRMS_PSR_OWN_AHEAD
            read_own(t + 1);
#endif
#ifdef AGRMS_PSR_RELAY_FIRST
            // port_stream_rate: the relay items of this step go to the sender before the members' landing
            // notifications (the fan-out no longer delays the next packets)
            if (has_in && t >= relay_lag) {
                AGRMS_ZONE("AG_STEP_NOTIFY");
                notify_until(static_cast<int32_t>(t - relay_lag));
            }
#endif
        }
    }
    finish_pending();
    if (has_in) {
#ifdef AGRMS_PSR_IDLE_NOTIFY
        {
            AGRMS_ZONE("AG_FINAL_ARRIVAL");  // the last relayed items from upstream (earlier rows notified meanwhile)
            arrival_idle(total_in);
        }
        AGRMS_ZONE("AG_FINAL_NOTIFY");
        notify_allowed = 0x7FFFFFFF;
        while (notify_one_row()) {
        }
#else
        {
            AGRMS_ZONE("AG_FINAL_ARRIVAL");  // the last relayed items from upstream (wait only)
            noc_semaphore_wait_min(arrived, total_in);
        }
        AGRMS_ZONE("AG_FINAL_NOTIFY");
        notify_until(0x7FFFFFFF);
#endif
    }
    noc_async_atomic_barrier();
}
