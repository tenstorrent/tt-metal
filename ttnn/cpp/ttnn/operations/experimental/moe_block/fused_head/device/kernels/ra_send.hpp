// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing exchange's send (RA_SEND).
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#ifndef RA_NF_
#define RA_NF_ 16  // (a forwarding port's slots: their flags, then the landed count - RoutingStream.NF)
#endif
// one packet's way on: to the next hop (kind 0: the destination's DRAM slot page par N + src, its arrival counter on
// planner core 0; kind 1: the next forwarding port's DRAM slot page (fbase: the forward slots tensor, slot | page <<
// 8), its flag on that port), chunks of MAXP, the last with the increment. (A forward slot in DRAM, not the port's L1:
// a packet can land before that chip's head runs - in L1 the program still running there overwrote it, and the
// forwarder then sent a garbage-sized packet past its L1's end into the next hop's L1)
template <typename C, typename H>
FORCE_INLINE void ra_hop(
    C& conn,
    H hdr,
    uint32_t src_l1,
    uint32_t bytes,
    uint32_t a,
    uint32_t par,
    uint32_t srcid,
    uint32_t PCAP,
    uint32_t sl,
    uint32_t c0,
    uint32_t c1,
    uint32_t px,
    uint32_t py,
    uint32_t mesh,
    uint32_t fbase,
    uint32_t flags,
    uint32_t RA_N,
    uint32_t MAXP) {
    const uint32_t chip = get_arg_val<uint32_t>(a), kind = get_arg_val<uint32_t>(a + 1);
    const uint32_t nx = get_arg_val<uint32_t>(a + 2), ny = get_arg_val<uint32_t>(a + 3),
                   slot = get_arg_val<uint32_t>(a + 4);
    uint64_t dst, inc;
    if (kind == 0) {
        const InterleavedAddrGen<true> slg = {.bank_base_address = sl, .page_size = PCAP};
        dst = get_noc_addr(par * RA_N + srcid, slg);
        inc = get_noc_addr(px, py, par ? c1 : c0);
    } else {
        const InterleavedAddrGen<true> fg = {.bank_base_address = fbase, .page_size = PCAP};
        dst = get_noc_addr(slot >> 8, fg);
        inc = get_noc_addr(nx, ny, flags + 16 * (slot & 0xFF));
    }
    for (uint32_t off = 0; off < bytes; off += MAXP) {
        const uint32_t sz = bytes - off < MAXP ? bytes - off : MAXP;
        noc_async_writes_flushed();  // (the header's previous packet has left)
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(chip), static_cast<uint16_t>(mesh));
        if (off + sz >= bytes) {
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst + off, inc, 1, true}, sz);
        } else {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst + off}, sz);
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src_l1 + off, sz, reinterpret_cast<uint32_t>(hdr));
    }
    if (kind == 1) {  // (+1 on the next forwarder's landed count, behind the packet: its poll reads one word)
        noc_async_writes_flushed();
        hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(nx, ny, flags + 16 * RA_NF_), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    }
}
FORCE_INLINE uint32_t ra_bytes(uint32_t pb, uint32_t RA_M, uint32_t RA_HBP) {
    const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
    uint32_t n = 0;
    for (uint32_t q = 0; q < RA_M; ++q) {
        n += nq[q];
    }
    return (RA_HBP + 4 * n + 63) / 64 * 64;
}
struct RASend {
    uint32_t r0, nd, nr, self_, B, PCAP, sl, c0, c1, px, py, calls, ready, mesh, fbase, flags, done;
    void init(uint32_t a) {
        r0 = a;
        nd = get_arg_val<uint32_t>(a);
        if (!nd) {
            done = 1;
            return;
        }
        nr = get_arg_val<uint32_t>(a + 1);
        self_ = get_arg_val<uint32_t>(a + 2);
        B = get_arg_val<uint32_t>(a + 3);
        PCAP = get_arg_val<uint32_t>(a + 4);
        sl = get_arg_val<uint32_t>(a + 5);
        c0 = get_arg_val<uint32_t>(a + 6);
        c1 = get_arg_val<uint32_t>(a + 7);
        px = get_arg_val<uint32_t>(a + 8);
        py = get_arg_val<uint32_t>(a + 9);
        calls = get_arg_val<uint32_t>(a + 10);
        ready = get_arg_val<uint32_t>(a + 11);
        mesh = get_arg_val<uint32_t>(a + 12);
        fbase = get_arg_val<uint32_t>(a + 13);
        flags = get_arg_val<uint32_t>(a + 14);
        done = 0;
    }
    static uint32_t len(uint32_t a) {
        const uint32_t n = get_arg_val<uint32_t>(a);
        return n ? 15 + 5 * n : 1;
    }
    template <typename C, typename H>
    void poll(
        C& conn,
        H hdr,
        bool blocking,
        uint32_t RA_N,
        uint32_t RA_M,
        uint32_t RA_EPC,
        uint32_t RA_ME,
        uint32_t RA_HBP,
        uint32_t RA_BB,
        uint32_t RA_BBP,
        uint32_t MAXP) {
        if (done) {
            return;
        }
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        invalidate_l1_cache();
        if (*rd < RA_M) {
            if (!blocking) {
                return;
            }
            noc_semaphore_wait_min(rd, RA_M);
        }
        DeviceZoneScopedN("SE-RASEND");
        noc_semaphore_inc(get_noc_addr(ready), 0u - RA_M);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* cl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls);
        const uint32_t par = cl[0] & 1;  // (every port core and planner core counts its calls: they stay in step)
        cl[0] = cl[0] + 1;
        for (uint32_t j = 0; j < nd; ++j) {
            const uint32_t pb = B + j * PCAP;
            const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
            // (the per-expert totals: the planners' - the destination's owner instance wrote them into the header)
            reinterpret_cast<uint32_t*>(pb)[14] = par;  // (for the forwarders: the call's parity, the source)
            reinterpret_cast<uint32_t*>(pb)[15] = RA_ME;
            (void)nq;
            const uint32_t bytes = ra_bytes(pb, RA_M, RA_HBP);
            if (j < nr) {
                ra_hop(
                    conn,
                    hdr,
                    pb,
                    bytes,
                    r0 + 15 + 5 * j,
                    par,
                    RA_ME,
                    PCAP,
                    sl,
                    c0,
                    c1,
                    px,
                    py,
                    mesh,
                    fbase,
                    flags,
                    RA_N,
                    MAXP);
            } else {  // (SELF: my own chip's packet)
                const InterleavedAddrGen<true> slg = {.bank_base_address = sl, .page_size = PCAP};
                noc_async_write(pb, get_noc_addr(par * RA_N + RA_ME, slg), bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, par ? c1 : c0), 1);
            }
        }
        noc_async_writes_flushed();
        done = 1;
    }
};
// a forwarding port's slots: each packet (of another chip, on its way) landed in its DRAM slot page with its flag set -
// read into the scratch slot (L1, this program's) and sent on to its next hop (any order), the flag re-armed. args at
// r0: n, scratch base, flags, PCAP, slot tensor, c0, c1, planner core 0 (x, y), mesh, the forward slots tensor, this
// port's first page, then per slot (next chip, kind, x, y, slot | page << 8)
struct RAFwd {
    uint32_t r0, n, fb, fl, PCAP, sl, c0, c1, px, py, mesh, fd, pg0, left, lastc;
    uint32_t sent[RA_NF_];
    void init(uint32_t a) {
        r0 = a;
        n = get_arg_val<uint32_t>(a);
        left = n;
        lastc = 0;
        if (!n) {
            return;
        }
        fb = get_arg_val<uint32_t>(a + 1);
        fl = get_arg_val<uint32_t>(a + 2);
        PCAP = get_arg_val<uint32_t>(a + 3);
        sl = get_arg_val<uint32_t>(a + 4);
        c0 = get_arg_val<uint32_t>(a + 5);
        c1 = get_arg_val<uint32_t>(a + 6);
        px = get_arg_val<uint32_t>(a + 7);
        py = get_arg_val<uint32_t>(a + 8);
        mesh = get_arg_val<uint32_t>(a + 9);
        fd = get_arg_val<uint32_t>(a + 10);
        pg0 = get_arg_val<uint32_t>(a + 11);
        for (uint32_t k = 0; k < RA_NF_; ++k) {
            sent[k] = 0;
        }
    }
    static uint32_t len(uint32_t a) {
        const uint32_t n = get_arg_val<uint32_t>(a);
        return n ? 12 + 5 * n : 1;
    }
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking, uint32_t RA_N, uint32_t RA_M, uint32_t RA_HBP, uint32_t MAXP) {
        // (the landed count - one increment a packet, behind its data and flag - says when a scan of the flags can find
        // something: a poll between the other streams' packets reads one word)
        volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl + 16 * RA_NF_);
        while (left) {
            bool any = false;
            invalidate_l1_cache();
            const uint32_t c = *ctr;
            if (c == lastc) {
                if (!blocking) {
                    return;
                }
                continue;
            }
            lastc = c;
            for (uint32_t k = 0; k < n; ++k) {
                volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl + 16 * k);
                if (sent[k] || *f == 0) {
                    continue;
                }
                DeviceZoneScopedN("SE-RAFWD");
                const uint32_t pb = fb + k * PCAP;
                {
                    const InterleavedAddrGen<true> fg = {.bank_base_address = fd, .page_size = PCAP};
                    noc_async_read(get_noc_addr(pg0 + k, fg), pb, PCAP);
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                }
                const uint32_t par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb)[14];
                const uint32_t src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb)[15];
                uint32_t bytes = ra_bytes(pb, RA_M, RA_HBP);
                if (bytes > PCAP) {  // (debug: a slot whose counts say more than a slot holds - not this call's
                                     // packet: recorded in the flag's spare words (bytes, how often), sent clamped)
                    f[1] = bytes;
                    f[2] = f[2] + 1;
                    f[3] = src;
                    bytes = PCAP;
                }
                ra_hop(
                    conn,
                    hdr,
                    pb,
                    bytes,
                    r0 + 12 + 5 * k,
                    par,
                    src,
                    PCAP,
                    sl,
                    c0,
                    c1,
                    px,
                    py,
                    mesh,
                    fd,
                    fl,
                    RA_N,
                    MAXP);
                // (re-arm: one packet a slot a call - by an atomic decrement, ordered with the previous hop's remote
                // increments: a plain store could wipe a next call's increment landing between the read and the store)
                noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(f)), 0u - 1u);
                sent[k] = 1;
                --left;
                any = true;
            }
            if (!blocking && !any) {
                return;
            }
        }
        if (blocking &&
            n) {  // (re-arm once every packet's count has landed: by subtracting - a next call's packets may
                  // already land here (another row a call ahead: its count must survive, or the next call's
                  // poll never scans its flag - the forward lost, the destination's planner waiting forever)
            while (*ctr < n) {
                invalidate_l1_cache();
            }
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ctr)), 0u - n);
            noc_async_atomic_barrier();
            lastc = 0;
        }
        noc_async_writes_flushed();
    }
};
