// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
// (down + reduce-scatter) my units' chunks, the reader's walk: one fabric packet each into the downstream chip's same
// core, its zone slot of this direction (slot by the core's half); a core's last chunk carries the fused increment of
// its arrival counter. Pops at the reader's batch boundaries. args at r0: as the reader's, then the zone base, the
// arrival counter's address, P (zone slot tiles; 0: compact zones - a core's slot at the sum of its tiles of the
// blocks in the unit's prefix mask for the core's half, w0 bits 20 + 4 half)
template <typename C, typename H, typename L>
FORCE_INLINE void rs_send(uint32_t r0, C& conn, H* hdrs, L&& lpoll, uint32_t& rpos, uint32_t m) {
    const uint32_t nu = get_arg_val<uint32_t>(r0), u0 = r0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) {
        return;
    }
    const uint32_t c0 = u0 + 2 * nu + 1, en0 = c0 + 2 * nc;
    const uint32_t zb = get_arg_val<uint32_t>(en0 + G + 2), actr = get_arg_val<uint32_t>(en0 + G + 3);
    const uint32_t P = get_arg_val<uint32_t>(en0 + G + 4);
    // (chunk m into the downstream's zones once its cores read chunk m - 1's: its port's credits; with the router's
    // buffers over the zones (a "router done" count, en0 + G + 11) chunk 0's too - its router phase over: credits m +
    // 1)
    const uint32_t z0 = (get_arg_val<uint32_t>(en0 + G + 11) >> 16) & 1;
    if (m + z0 > 0) {
        volatile tt_l1_ptr uint32_t* zc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(en0 + G + 6));
        while (*zc < m + z0) {
            invalidate_l1_cache();
            lpoll();
        }
    }
    uint32_t u = 0, lim = GROUP - rpos % GROUP, h = 0;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, u * RP);
        rpos += u;
        u = 0;
        lim = GROUP - rpos % GROUP;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = get_arg_val<uint32_t>(u0 + 2 * un), w1 = get_arg_val<uint32_t>(u0 + 2 * un + 1);
        const uint32_t b = (w0 >> 8) & 0xFF, mask = (w0 >> 16) & 3;
        const uint32_t np_ = (w0 >> 28) & 1 ? RS_NPASS : 1;  // (pieces: the reader's walk - pass-major over my cores)
        for (uint32_t ip = 0; ip < np_; ++ip) {
            for (uint32_t i = 0; i < nc; ++i) {
                const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i), cnts = get_arg_val<uint32_t>(c0 + 2 * i + 1);
                uint32_t cnt = cnt_of(cnts, b);
                const uint32_t hf = (cw >> 16) & 1;
                if (!((mask >> hf) & 1) || cnt == 0) {
                    continue;
                }
                uint32_t po = 0;  // (the piece's tile offset in the sub-block)
                if (np_ > 1) {
                    const uint32_t pn = cw >> 27, nr = cnt / pn, bw = pn / RS_NPASS, rm = pn % RS_NPASS;
                    po = nr * (ip * bw + (ip < rm ? ip : rm));
                    cnt = nr * (bw + (ip < rm ? 1 : 0));
                }
                const bool last = ip + 1 == np_;  // (the core's last piece carries its arrival increment)
                const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
                uint32_t zo;
                if (P) {
                    zo = (hf ? (w1 >> 16) : (w1 & 0xFFFF)) * P;
                } else {
                    const uint32_t pm = (w0 >> (20 + 4 * hf)) & 0xF;
                    zo = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        if ((pm >> j) & 1) {
                            zo += (cnts >> (8 * j)) & 0xFF;
                        }
                    }
                }
                const uint32_t dst0 = zb + (zo + po) * 2048;
                const uint64_t ctr = get_noc_addr(x, y, actr);
                for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                    lpoll();  // (the logits / routing go between my packets: ahead of my next one)
                    while (!cb_pages_available_at_front(0, RP * (u + 1))) {
                        lpoll();
                    }
                    cb_wait_front(0, RP * (u + 1));
                    const uint32_t src = get_read_ptr(0) + u * RP * 2048;
                    const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                    const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 2048);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    if (last && t0 + RP >= cnt) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, n * 2048);
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
        }
        if (u > 0) {
            pop_batch();
        }
    }
}
