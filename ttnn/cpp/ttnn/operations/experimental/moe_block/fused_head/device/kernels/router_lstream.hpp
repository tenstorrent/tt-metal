// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router's L stream (LStream).
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
struct LStream {
    uint32_t r0, nu, nc, buf, word, zone, actr, et, rpl, need, ns, sfree, u, i, sent;
    void init(uint32_t a) {
        r0 = a;
        nu = get_arg_val<uint32_t>(a);
        nc = get_arg_val<uint32_t>(a + 1 + 2 * nu);
        const uint32_t t = a + 2 + 2 * nu + nc;
        buf = get_arg_val<uint32_t>(t);
        word = get_arg_val<uint32_t>(t + 1);
        zone = get_arg_val<uint32_t>(t + 2);
        actr = get_arg_val<uint32_t>(t + 3);
        et = get_arg_val<uint32_t>(t + 4);
        rpl = get_arg_val<uint32_t>(t + 5);
        need = get_arg_val<uint32_t>(t + 6);  // (a sub-block's word when in: 1 hand-off, or et engine tiles)
        ns = get_arg_val<uint32_t>(t + 7);    // (sub-block slots a unit: core i in slot i % ns, then core i + ns told)
        sfree = get_arg_val<uint32_t>(t + 8);
        u = 0;
        i = 0;
        sent = 0;
    }
    bool done() const { return u >= nu; }
    static uint32_t len(uint32_t a) {
        const uint32_t nu = get_arg_val<uint32_t>(a);
        return 2 + 2 * nu + get_arg_val<uint32_t>(a + 1 + 2 * nu) + 9;
    }
    // send what is ready (unit u: an owner chip, multi-hop; core i of my link: its partial of that owner's rows);
    // blocking: until everything is sent
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking) {
        while (u < nu) {
            if (i >= nc) {
                ++u;
                i = 0;
                continue;
            }
            const uint32_t w0 = get_arg_val<uint32_t>(r0 + 1 + 2 * u), mesh = get_arg_val<uint32_t>(r0 + 2 + 2 * u);
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * (u * nc + i));
            invalidate_l1_cache();
            if (*wd < need) {
                if (!blocking) {
                    return;
                }
                noc_semaphore_wait_min(wd, need);
            }
            const uint32_t cw = get_arg_val<uint32_t>(r0 + 2 + 2 * nu + i), x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t src0 = buf + ((u * ns + i % ns) * et) * 4096;
            const uint32_t dst0 = zone + (w0 >> 16) * et * 4096;
            for (uint32_t t0 = 0; t0 < et; t0 += rpl) {
                const uint32_t n = et - t0 < rpl ? et - t0 : rpl;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 4096);
                noc_async_writes_flushed();  // (the header's previous packet has left)
                (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(w0 & 0xFFFF), static_cast<uint16_t>(mesh));
                if (t0 + rpl >= et) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(x, y, actr + 16 * (w0 >> 16)), 1, true},
                        n * 4096);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 4096);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src0 + t0 * 4096, n * 4096, reinterpret_cast<uint32_t>(hdr));
            }
            if (i + ns < nc) {  // (the slot's next core may write it once the sub-block left)
                noc_async_writes_flushed();
                const uint32_t cn = get_arg_val<uint32_t>(r0 + 2 + 2 * nu + i + ns);
                noc_semaphore_inc(get_noc_addr(cn & 0xFF, (cn >> 8) & 0xFF, sfree + 16 * u), 1);
            }
            ++i;
        }
    }
    void rearm() {
        for (uint32_t c = 0; c < nu * nc; ++c) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * c) = 0;
        }
    }
};
