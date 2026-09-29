// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) reader: the initial state S [K,V] once, then per chunk the seven prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv from DRAM. All fp32.
//
// Four compile variants (host selects via defines; no define = the plain reader):
//   GDN_MCAST_SENDER   — this core is its head's v-block-0. It reads the six SHARED V-independent
//                        tensors (kd, q_decay, intra, k_dec_t, dl, t_inv) from DRAM once per chunk
//                        and multicasts them into the sibling v-block cores' CBs (identical CB
//                        base addresses on every scan core — CBs are declared on one CoreRangeSet).
//   GDN_MCAST_RECEIVER — a sibling v-block core. Reads only its private V-sliced tensors (v_beta,
//                        s0) from DRAM; the shared block arrives over the NoC. Handshake:
//                        reserve CB space -> ready.up(sender) -> wait(valid) -> push.
//   GDN_FUSED_RECEIVER — chunk_gdn_fused consumer: zero DRAM intermediates. Reads only its V-slice
//                        of s0; ALL seven per-chunk tensors — v_beta included (Option U: the
//                        producer computes and sends it, keeping the compute kernels byte-identical)
//                        — arrive over the NoC from the producers' writers. This core is receiver
//                        (h, vb) of NV per head: it carries V columns [vb*Vt, +Vt) (Vt = the slice
//                        width, CT arg 2). Handshake per chunk: reserve the 7 CBs -> reset VALID ->
//                        atomically increment credit[h] on the producer that owns the chunk
//                        (c % NP of this head) -> wait VALID -> push. The producer sends only once
//                        all NV receivers have credited. A one-time init barrier (SEM_INIT) orders
//                        the producers' zeroing of their credit words before any credit.
// The handshake follows the production matmul in0 mcast idiom (reader_bmm_tile_layout_in0_
// sender_padding.cpp / _receiver.cpp): ready counts receivers that RESERVED space (so the sender
// can never overwrite unconsumed data), the data mcasts and the valid-flag mcast share one NOC /
// static VC with linked=true chaining (data-before-flag), and an async_writes_flushed() sits
// between data and flag ON EVERY ARCH: on Blackhole it orders flag-after-data (NoC latency >
// L1<->RISCV latency); everywhere it also proves the data mcasts have read their L1 source slots
// before the pushes let compute pop them and the next chunk's DRAM reads reuse them (the CBs are
// single-buffered — without the flush that reuse is a silent data race; the flag mcast runs on a
// different cmd buf and orders nothing). Everything runs on the reader's NOC_0, so the multicast
// rectangle coords arrive UNSWAPPED (top-left -> bottom-right).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#if defined(GDN_MCAST_SENDER) || defined(GDN_MCAST_RECEIVER) || defined(GDN_FUSED_RECEIVER)
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"
// Semaphore ids (SEM_READY/SEM_VALID) arrive as the two trailing compile-time args after the
// accessor chain — read in kernel_main below, so factory and kernel cannot drift.
#endif

// The seven per-chunk input CBs sit at PREP'S OUTPUT indices (v_beta=14, dl=22; the rest were
// already aligned) so the fused program declares one hand-off CB set on the producer/receiver
// union. Must match chunk_gdn_scan.cpp (compute) and both program factories.
constexpr uint32_t cb_dl = 22, cb_S = 8, cb_Tinv = 13;
constexpr uint32_t cb_vbeta = 14, cb_kd = 18, cb_qdecay = 19, cb_intra = 20, cb_kdec_t = 24;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    (void)has_s0;

#if defined(GDN_FUSED_RECEIVER)
    // Fused receivers touch DRAM only for s0; the accessor chain is a single block.
    constexpr auto s0_a = TensorAccessorArgs<5>();
#elif defined(GDN_MCAST_RECEIVER)
    // Receivers only access their private V-sliced tensors; the accessor chain has two blocks.
    constexpr auto vb_a = TensorAccessorArgs<5>();
    constexpr auto s0_a = TensorAccessorArgs<vb_a.next_compile_time_args_offset()>();
#else
    constexpr auto vb_a = TensorAccessorArgs<5>();
    constexpr auto kd_a = TensorAccessorArgs<vb_a.next_compile_time_args_offset()>();
    constexpr auto qd_a = TensorAccessorArgs<kd_a.next_compile_time_args_offset()>();
    constexpr auto it_a = TensorAccessorArgs<qd_a.next_compile_time_args_offset()>();
    constexpr auto kc_a = TensorAccessorArgs<it_a.next_compile_time_args_offset()>();
    constexpr auto dl_a = TensorAccessorArgs<kc_a.next_compile_time_args_offset()>();
    constexpr auto ti_a = TensorAccessorArgs<dl_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<ti_a.next_compile_time_args_offset()>();
#endif

#if defined(GDN_MCAST_SENDER) || defined(GDN_MCAST_RECEIVER) || defined(GDN_FUSED_RECEIVER)
    // Handshake semaphore ids: the two trailing compile-time args the factory appends AFTER the
    // TensorAccessorArgs chain (unconditionally, on every variant, so the offsets stay uniform;
    // the plain reader has no semaphores and simply doesn't read them). s0_a is the LAST accessor
    // on the sender's 8-accessor chain, the mcast receiver's 2-accessor chain, and the fused
    // receiver's 1-accessor chain alike, so the same offset expression is correct everywhere.
    // SEM_READY: receivers -> sender: "my CB space for this chunk is reserved"
    // SEM_VALID: sender -> receivers: "this chunk's shared data is in your CBs"
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(s0_a.next_compile_time_args_offset());
    constexpr uint32_t SEM_VALID = get_compile_time_arg_val(s0_a.next_compile_time_args_offset() + 1);
#if defined(GDN_FUSED_RECEIVER)
    // Fused-receiver extras (trailing CT args after the semaphore ids): the init-barrier semaphore,
    // and the union-declared CB + byte offset holding the producers' per-head credit words.
    constexpr uint32_t SEM_INIT = get_compile_time_arg_val(s0_a.next_compile_time_args_offset() + 2);
    constexpr uint32_t CB_CREDIT = get_compile_time_arg_val(s0_a.next_compile_time_args_offset() + 3);
    constexpr uint32_t CREDIT_OFF = get_compile_time_arg_val(s0_a.next_compile_time_args_offset() + 4);
    constexpr uint32_t NBUF = get_compile_time_arg_val(s0_a.next_compile_time_args_offset() + 5);  // hand-off slots
    (void)SEM_READY;  // superseded by the credit words on this variant
#endif
#endif

    // This core handles head h, V-block vb (columns [vb*Vt, vb*Vt+Vt) of the full V dimension).
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
#if defined(GDN_FUSED_RECEIVER)
    const uint32_t s0_addr = get_arg_val<uint32_t>(3);
    // NP producers share this head's prep work round-robin (producer p owns chunks c = p, p+NP,
    // ...); the per-chunk credit goes to producer (c % NP) — that rotation IS the in-order delivery
    // mechanism: a producer sends chunk c only once every receiver of the head has reserved chunk
    // c's slots, so at most one hand-off per receiver is in flight and VALIDs cannot interleave.
    // N_INIT = init-barrier increments to expect before the first credit (NP in the per-head form).
    // The producers' virtual worker coords follow as NP (x, y) pairs from arg 6.
    const uint32_t NP = get_arg_val<uint32_t>(4);
    const uint32_t N_INIT = get_arg_val<uint32_t>(5);
#elif defined(GDN_MCAST_RECEIVER)
    const uint32_t vb_addr = get_arg_val<uint32_t>(3);
    const uint32_t s0_addr = get_arg_val<uint32_t>(4);
    const uint32_t sender_x = get_arg_val<uint32_t>(5);  // virtual worker coords of the sender
    const uint32_t sender_y = get_arg_val<uint32_t>(6);
#else
    const uint32_t vb_addr = get_arg_val<uint32_t>(3);
    const uint32_t kd_addr = get_arg_val<uint32_t>(4);
    const uint32_t qd_addr = get_arg_val<uint32_t>(5);
    const uint32_t it_addr = get_arg_val<uint32_t>(6);
    const uint32_t kc_addr = get_arg_val<uint32_t>(7);
    const uint32_t dl_addr = get_arg_val<uint32_t>(8);
    const uint32_t ti_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);
#endif
#if defined(GDN_MCAST_SENDER)
    // Receiver rectangle (virtual worker coords, NOC_0 orientation: top-left -> bottom-right).
    const uint32_t rcv_x0 = get_arg_val<uint32_t>(11);
    const uint32_t rcv_y0 = get_arg_val<uint32_t>(12);
    const uint32_t rcv_x1 = get_arg_val<uint32_t>(13);
    const uint32_t rcv_y1 = get_arg_val<uint32_t>(14);
    const uint32_t num_dests = get_arg_val<uint32_t>(15);  // NV-1; excludes the sender
#endif

    const uint32_t tb = get_tile_size(cb_vbeta);  // all inputs fp32 -> same tile size
#if !defined(GDN_FUSED_RECEIVER)
    const auto vb_acc = TensorAccessor(vb_a, vb_addr, tb);
#endif
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);
#if !defined(GDN_MCAST_RECEIVER) && !defined(GDN_FUSED_RECEIVER)
    const auto kd_acc = TensorAccessor(kd_a, kd_addr, tb);
    const auto qd_acc = TensorAccessor(qd_a, qd_addr, tb);
    const auto it_acc = TensorAccessor(it_a, it_addr, tb);
    const auto kc_acc = TensorAccessor(kc_a, kc_addr, tb);
    const auto dl_acc = TensorAccessor(dl_a, dl_addr, tb);
    const auto ti_acc = TensorAccessor(ti_a, ti_addr, tb);
#endif

    // V-independent tile counts (full reads). cv/kv are per-row Vt and handled by read_vslice.
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;

    Noc noc;

#if !defined(GDN_MCAST_RECEIVER) && !defined(GDN_MCAST_SENDER) && !defined(GDN_FUSED_RECEIVER)
    // Full (V-independent) read: n contiguous tiles from `base` into the CB.
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };
#endif

    // V-slice read: R row-groups of Vt tiles each, laid out in DRAM with row stride Vt_full and
    // this core's column offset vb*Vt. Packs contiguously ([R, Vt]) into the CB. `row_base` is the
    // first-tile index of the tensor's [R, Vt_full] block for this (head[, chunk]).
    auto read_vslice = [&](const auto& acc, uint32_t cb_id, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(R * Vt);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t src = row_base + r * Vt_full + vb * Vt;
            const uint32_t dstt = r * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(acc, cb, tb, {.page_id = src + vt}, {.offset_bytes = (dstt + vt) * tb});
            }
        }
        noc.async_read_barrier();
        cb.push_back(R * Vt);
    };

    // initial state S [K, V] (once) — host always provides it (zeros if none). V-sliced
    // (degenerates to the full state on fused receivers: vb = 0, Vt = Vt_full).
    read_vslice(s0_acc, cb_S, h * Kt * Vt_full, Kt);

#if defined(GDN_MCAST_SENDER)
    Semaphore<> ready(SEM_READY);
    Semaphore<> valid(SEM_VALID);
    // set_multicast sources its 4-byte value from the sender's LOCAL copy of `valid` — and it
    // reads that L1 word asynchronously, when the NIU processes the command. Preset it to VALID
    // once; any future write to this word must be preceded by noc.async_writes_flushed() or
    // async_write_barrier(), or an in-flight set_multicast can pick up the new value (see the
    // teardown, where the barrier deliberately comes BEFORE the reset).
    valid.set(VALID);

    // Stage a shared group: reserve + issue DRAM reads into this core's own CB slot, WITHOUT
    // pushing (the write pointer must still address the slot when the mcast reads it). Returns
    // the slot's L1 address — identical on every sibling core (same CB config, same push/pop
    // history: exactly n tiles per chunk on both sides).
    auto stage_group = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        return cb.get_write_ptr();
    };

    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_vslice(vb_acc, cb_vbeta, hc * Ct * Vt_full, Ct);  // private v_beta slice (unchanged)

        // Stage all six shared groups, then one barrier for the lot.
        const uint32_t p_kd = stage_group(kd_acc, cb_kd, hc * ck, ck);
        const uint32_t p_qd = stage_group(qd_acc, cb_qdecay, hc * ck, ck);
        const uint32_t p_it = stage_group(it_acc, cb_intra, hc * cc, cc);
        const uint32_t p_kc = stage_group(kc_acc, cb_kdec_t, hc * kc, kc);
        const uint32_t p_dl = stage_group(dl_acc, cb_dl, hc * 1, 1);
        const uint32_t p_ti = stage_group(ti_acc, cb_Tinv, hc * cc, cc);
        noc.async_read_barrier();

        // All receivers have reserved this chunk's CB space (their slots are writable).
        ready.wait(num_dests);
        ready.set(0);

        // Data mcasts, linked so they chain with the flag mcast on one static VC (data-before-
        // flag ordering). num_dests excludes the sender — no local copy is performed.
        MulticastEndpoint mcast_dst;
        auto mcast_group = [&](uint32_t addr, uint32_t n) {
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(addr),
                mcast_dst,
                n * tb,
                num_dests,
                {},
                {.noc_x_start = rcv_x0, .noc_y_start = rcv_y0, .noc_x_end = rcv_x1, .noc_y_end = rcv_y1, .addr = addr},
                /*linked=*/true);
        };
        mcast_group(p_kd, ck);
        mcast_group(p_qd, ck);
        mcast_group(p_it, cc);
        mcast_group(p_kc, kc);
        mcast_group(p_dl, 1);
        mcast_group(p_ti, cc);
        // Flush on EVERY arch, for two reasons. Blackhole: NoC latency exceeds L1<->RISCV latency,
        // so without it a receiver could see VALID with the data still in flight. All arches: the
        // flush proves every data mcast has read its L1 source slot; only then may the pushes
        // below let compute pop the slots and the next chunk's stage_group reuse them (nbuf=1 —
        // the flag mcast runs on a different cmd buf and provides no such ordering).
        noc.async_writes_flushed();
        valid.set_multicast(noc, rcv_x0, rcv_y0, rcv_x1, rcv_y1, num_dests);  // unlinked: ends chain

        // Advance the sender's own CBs only now (the mcasts addressed the pre-push slots).
        CircularBuffer(cb_kd).push_back(ck);
        CircularBuffer(cb_qdecay).push_back(ck);
        CircularBuffer(cb_intra).push_back(cc);
        CircularBuffer(cb_kdec_t).push_back(kc);
        CircularBuffer(cb_dl).push_back(1);
        CircularBuffer(cb_Tinv).push_back(cc);
    }

    // Barrier BEFORE resetting the local valid word: set_multicast reads its 4-byte payload from
    // that word asynchronously, so resetting first could multicast INVALID for the final chunk
    // and deadlock every receiver at valid.wait(VALID). The barrier waits until all nonposted
    // writes (data + flag mcasts) are acked; only then is the local reset safe.
    noc.async_write_barrier();
    valid.set(INVALID);  // restore the semaphore's initial value

#elif defined(GDN_MCAST_RECEIVER)
    Semaphore<> ready(SEM_READY);
    Semaphore<> valid(SEM_VALID);

    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_vslice(vb_acc, cb_vbeta, hc * Ct * Vt_full, Ct);  // private v_beta slice (unchanged)

        // Reserve this chunk's space in every shared CB FIRST — the ready inc is the sender's
        // proof that these slots are writable (compute has popped the previous chunk).
        CircularBuffer(cb_kd).reserve_back(ck);
        CircularBuffer(cb_qdecay).reserve_back(ck);
        CircularBuffer(cb_intra).reserve_back(cc);
        CircularBuffer(cb_kdec_t).reserve_back(kc);
        CircularBuffer(cb_dl).reserve_back(1);
        CircularBuffer(cb_Tinv).reserve_back(cc);

        // Reset our valid flag BEFORE signalling ready: a fast sender may mcast VALID immediately
        // after the inc, and a late reset would overwrite it (lost wakeup -> deadlock).
        valid.set(INVALID);
        ready.up(noc, sender_x, sender_y, 1);
        valid.wait(VALID);

        // The shared bytes are in our CBs; make them visible to compute.
        CircularBuffer(cb_kd).push_back(ck);
        CircularBuffer(cb_qdecay).push_back(ck);
        CircularBuffer(cb_intra).push_back(cc);
        CircularBuffer(cb_kdec_t).push_back(kc);
        CircularBuffer(cb_dl).push_back(1);
        CircularBuffer(cb_Tinv).push_back(cc);
    }

    valid.set(INVALID);  // local store: restore the initial value (the last wait left it VALID)
    // Drain the ready atomics: no non-posted inc may be in flight at kernel exit.
    noc.async_atomic_barrier();

#elif defined(GDN_FUSED_RECEIVER)
    Semaphore<> init(SEM_INIT);

    // v_beta arrives as THIS receiver's V-slice: Vt here is the slice width (Vtl), so cv = Ct*Vtl
    // tiles per chunk. The CB itself is the producer-sized ring (cv_full*nbuf tiles, union
    // declaration); the producer derives the matching slot from the global chunk index.
    constexpr uint32_t cv = Ct * Vt;

    // Pipelined hand-off: D = NBUF-1 chunks are credited ahead of the one being
    // waited for, each in its own slot with its own VALID flag (semaphore id SEM_VALID + slot) and its
    // own credit word credit[h][slot] on the owning producer. The round trip credit -> VALID is thus
    // hidden behind D receiver steps instead of sitting on the critical path.
    constexpr uint32_t D = (NBUF > 1) ? NBUF - 1 : 1;

    // The producers' credit words sit in the last tile of the union-declared u/mask CB — the same L1
    // address on every core of the program — so this receiver can name head h's words on any
    // producer without being told an address. Word (h, slot) is at CREDIT_OFF + 4*(h*NBUF + slot).
    const uint32_t credit_base = CircularBuffer(CB_CREDIT).get_read_ptr() + CREDIT_OFF + 4 * h * NBUF;

    // Init barrier: dispatch re-initializes only Semaphore objects per launch, so the producers
    // zero their credit words themselves and then bump `init` here; crediting earlier could land
    // an increment on a word about to be zeroed (a hang at credit == NV).
    init.wait(N_INIT);

    // Issue the hand-off of chunk c: reserve, mark its slot INVALID, credit its owner.
    // reserve_back does not remember earlier unpushed reservations, so ask for D chunks' worth: that
    // holds iff compute has popped chunk c - NBUF, i.e. iff slot (c % NBUF) is free — exactly the
    // condition the credit promises the producer. (The v_beta ring is NV*NBUF chunks deep for this
    // slice and therefore free a fortiori.)
    auto issue = [&](uint32_t c) {
        {
            DeviceZoneScopedN("rx_reserve");
            CircularBuffer(cb_vbeta).reserve_back(D * cv);
            CircularBuffer(cb_kd).reserve_back(D * ck);
            CircularBuffer(cb_qdecay).reserve_back(D * ck);
            CircularBuffer(cb_intra).reserve_back(D * cc);
            CircularBuffer(cb_kdec_t).reserve_back(D * kc);
            CircularBuffer(cb_dl).reserve_back(D * 1);
            CircularBuffer(cb_Tinv).reserve_back(D * cc);
        }
        const uint32_t slot = c % NBUF;
        // Reset the slot's flag BEFORE crediting: a fast producer may set VALID immediately after
        // the credit lands, and a late reset would overwrite it (lost wakeup -> deadlock).
        Semaphore<>(SEM_VALID + slot).set(INVALID);
        // Credit the owner of chunk c by incrementing ITS copy of credit[h][slot]. INVARIANT: a slot
        // is credited only after it was reserved, i.e. after compute popped the chunk
        // that last used it — which the producer's VALID for that chunk preceded, which its reset of
        // this very word preceded. Hence the word counts exactly one chunk at a time for any NP.
        const uint32_t pi = c % NP;
        const uint64_t dst = get_noc_addr(
            get_arg_val<uint32_t>(6 + 2 * pi),
            get_arg_val<uint32_t>(7 + 2 * pi),
            credit_base + 4 * slot,
            noc.get_noc_id());
        noc_semaphore_inc(dst, 1, noc.get_noc_id());
    };

    uint32_t next = 0;
    for (; next < D && next < NC; next++) {
        issue(next);
    }
    for (uint32_t c = 0; c < NC; c++) {
        {
            DeviceZoneScopedN("rx_wait_valid");
            Semaphore<>(SEM_VALID + (c % NBUF)).wait(VALID);
        }

        // The chunk's seven blocks are in our CBs; make them visible to compute.
        CircularBuffer(cb_vbeta).push_back(cv);
        CircularBuffer(cb_kd).push_back(ck);
        CircularBuffer(cb_qdecay).push_back(ck);
        CircularBuffer(cb_intra).push_back(cc);
        CircularBuffer(cb_kdec_t).push_back(kc);
        CircularBuffer(cb_dl).push_back(1);
        CircularBuffer(cb_Tinv).push_back(cc);

        if (next < NC) {
            issue(next);
            next++;
        }
    }

    for (uint32_t s = 0; s < NBUF; s++) {
        Semaphore<>(SEM_VALID + s).set(INVALID);  // local store: restore the initial values
    }
    // Drain the credit atomics: no non-posted inc may be in flight at kernel exit.
    noc.async_atomic_barrier();

#else
    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_vslice(vb_acc, cb_vbeta, hc * Ct * Vt_full, Ct);  // v_beta [C, V] slice
        read_into(kd_acc, cb_kd, hc * ck, ck);                 // V-independent: full read
        read_into(qd_acc, cb_qdecay, hc * ck, ck);
        read_into(it_acc, cb_intra, hc * cc, cc);
        read_into(kc_acc, cb_kdec_t, hc * kc, kc);
        read_into(dl_acc, cb_dl, hc * 1, 1);
        read_into(ti_acc, cb_Tinv, hc * cc, cc);
    }
#endif
}
