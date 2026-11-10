// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router's L stream (LStream).
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
struct LStream {
    uint32_t r0, nu, nc, buf, word, zone, actr, et, rpl, need, ns, sfree, u, i, sent;
    uint32_t rit, c0, rbi;  // (router iterations: my "rit" word, router core 0's NoC, the router cores an iteration)
    void init(uint32_t a) {
        r0 = a;
        nu = mb_args::arg(a);
        nc = mb_args::arg(a + 1 + 2 * nu);
        const uint32_t t = a + 2 + 2 * nu + nc;
        buf = mb_args::arg(t);
        word = mb_args::arg(t + 1);
        zone = mb_args::arg(t + 2);
        actr = mb_args::arg(t + 3);
        et = mb_args::arg(t + 4);
        rpl = mb_args::arg(t + 5);
        need = mb_args::arg(t + 6);  // (a sub-block's word when in: 1 hand-off, or et engine tiles)
        ns = mb_args::arg(t + 7);    // (sub-block slots a unit: core i in slot i % ns, then core i + ns told)
        sfree = mb_args::arg(t + 8);
        rit = mb_args::arg(t + 9);
        c0 = mb_args::arg(t + 10);
        rbi = mb_args::arg(t + 11);
        u = 0;
        i = 0;
        sent = 0;
    }
    bool done() const { return u >= nu; }
    static uint32_t len(uint32_t a) {
        const uint32_t nu = mb_args::arg(a);
        return 2 + 2 * nu + mb_args::arg(a + 1 + 2 * nu) + 12;
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
            const uint32_t w0 = mb_args::arg(r0 + 1 + 2 * u), mesh = mb_args::arg(r0 + 2 + 2 * u);
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * (u * nc + i));
            invalidate_l1_cache();
            if (*wd < need) {
                if (!blocking) {
                    return;
                }
                noc_semaphore_wait_min(wd, need);
            }
            const uint32_t cw = mb_args::arg(r0 + 2 + 2 * nu + i), x = cw & 0xFF, y = (cw >> 8) & 0xFF;
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
                const uint32_t cn = mb_args::arg(r0 + 2 + 2 * nu + i + ns);
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
    bool active() const { return nu > 0 && nc > 0; }
    // the next router iteration's sub-blocks (its slots and words reused: the matmul cores write them only once router
    // core 0 told them go, past this iteration's count of mine)
    void next_iteration() {
        rearm();
        u = 0;
        i = 0;
    }
};
