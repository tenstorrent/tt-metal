// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
// (K3's latent all-to-all) the reader's walk: chunks into chip b's same core (multi-hop fabric unicast: a header
// routed to b a unit), LATZ slot of my chip, its rows; a core's last chunk with the fused increment of its counter
template <typename C, typename H, typename L>
FORCE_INLINE void la_send(uint32_t a0, C& conn, H* hdrs, L&& lpoll, uint32_t& rpos, uint32_t m) {
    const uint32_t nu = get_arg_val<uint32_t>(a0), u0 = a0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) {
        return;
    }
    const uint32_t c0 = u0 + 2 * nu + 1, t0a = c0 + 2 * nc;
    const uint32_t latz = get_arg_val<uint32_t>(t0a + 2) + m * LA_CHL, lctr = get_arg_val<uint32_t>(t0a + 3),
                   LW = get_arg_val<uint32_t>(t0a + 4);
    {  // (end-to-end fence: every destination chip started this call - its LATZ is free; my neighbour's ready only
       // fences the first hop)
        const uint32_t fence = get_arg_val<uint32_t>(t0a + 5), need = get_arg_val<uint32_t>(t0a + 6);
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence) < need) {
            invalidate_l1_cache();
            lpoll();
        }
    }
    uint32_t u = 0, lim = GROUP - rpos % GROUP;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, u * RP);
        rpos += u;
        u = 0;
        lim = GROUP - rpos % GROUP;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = get_arg_val<uint32_t>(u0 + 2 * un), b = w0 & 0xFF, slot = (w0 >> 8) & 0xFF;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[GROUP + 1 + un];
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i),
                           rw = (get_arg_val<uint32_t>(c0 + 2 * i + 1) >> (4 * b)) & 0xF;
            const uint32_t r0 = rw & 3, cnt = (rw >> 2) * LW;
            if (cnt == 0) {
                continue;
            }
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t dst0 = latz + (slot * MT + r0) * LW * 2048;
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                lpoll();
                while (!cb_pages_available_at_front(0, RP * (u + 1))) {
                    lpoll();
                }
                cb_wait_front(0, RP * (u + 1));
                const uint32_t src = get_read_ptr(0) + u * RP * 2048;
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 2048);
                noc_async_writes_flushed();  // (this unit's one header: its previous packet has left)
                if (t0 + RP >= cnt) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(x, y, lctr), 1, true}, n * 2048);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 2048);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, n * 2048, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) {
                    pop_batch();
                }
            }
        }
        if (u > 0) {
            pop_batch();
        }
    }
}
