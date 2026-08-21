// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused-program (chunk_gdn_fused) PRODUCER writer: replaces the prep writer's DRAM drain with a
// direct NoC hand-off. Per chunk it waits for the seven compute-pushed intermediates
//   v_beta [C,V], t_inv [C,C], kd [C,K], intra [C,C], q_decay [C,K], k_dec_t [K,C], dl [1 tile]
// and writes them into the paired RECEIVER (scan) core's CBs under the shipped ready/valid
// handshake of reader_chunk_gdn_scan.cpp — the receiver (GDN_FUSED_RECEIVER) cannot tell this
// computing sender from the phased DRAM-reading one. v_beta ships whole (Option U for F1: the
// producer computes and sends it, keeping both compute kernels byte-identical to the phased
// path; Option R — scan-side recompute from v and beta — is deferred to F2).
//
// Addressing (F3a): the seven hand-off CBs are declared on the UNION of producer+receiver cores
// (identical base addresses both sides), so this core's CB base == the receiver's CB base. The
// receiver reserves/pushes every CB exactly once per GLOBAL chunk c, so its reserved slot for
// chunk c is base + (c % NBUF)*slot_bytes — and that is where each mcast must land. At NP=1 that
// equals this core's read_ptr (the F2 lockstep lemma), but at NP>1 producer p only pushes/pops
// its OWNED chunks c = p, p+NP, ..., so its local slot index (owned-chunk count % NBUF) diverges
// from the receiver's (c % NBUF): sourcing stays read_ptr (correct local data), the DESTINATION
// is computed explicitly from the global c. The destination is a 1x1 rectangle (one receiver),
// which is orientation-neutral: start == end, so this writer's NOC (whichever the WriterConfig
// assigns) needs no coordinate swap.
//
// Ordering at NP>1 (P1): the receiver drives in-order delivery by crediting SEM_READY on
// producer (c % NP) only when chunk c's slots are reserved — each producer's ready.wait(1)
// therefore fires exactly for its own next owned chunk, and at most one hand-off is in flight
// per receiver at any time (VALID mcasts cannot interleave across producers).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"

// CB indices (prep compute's output slots == the scan side's hand-off slots after the F1 renumber;
// must match chunk_gdn_prep.cpp, chunk_gdn_scan.cpp and the fused program factory).
constexpr uint32_t cb_Tinv = 13, cb_vbeta = 14, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_kdec_t = 24, cb_dl = 22;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    // Handshake semaphore ids straight from the factory's SemaphoreDescriptors (no accessor
    // chain on this kernel — nothing touches DRAM).
    // SEM_READY: receiver -> producer: "my CB space for this chunk is reserved"
    // SEM_VALID: producer -> receiver: "this chunk's seven blocks are in your CBs"
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_VALID = get_compile_time_arg_val(4);
    // Hand-off CB depth (slots per CB) — the factory passes the same value it sizes the CBs with,
    // so the explicit slot arithmetic below can never drift from the allocation.
    constexpr uint32_t NBUF = get_compile_time_arg_val(5);

    const uint32_t NC = get_arg_val<uint32_t>(0);     // GLOBAL chunk count of this head
    const uint32_t NP = get_arg_val<uint32_t>(1);     // producers for this head
    const uint32_t p = get_arg_val<uint32_t>(2);      // this producer's index: owns c = p, p+NP, ...
    const uint32_t rcv_x = get_arg_val<uint32_t>(3);  // virtual worker coords of the receiver
    const uint32_t rcv_y = get_arg_val<uint32_t>(4);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kc = Kt * Ct;

    const uint32_t tb = get_tile_size(cb_vbeta);  // all hand-off CBs are fp32 -> same tile size

    Noc noc;
    Semaphore<> ready(SEM_READY);
    Semaphore<> valid(SEM_VALID);
    // set_multicast sources its 4-byte value from this core's LOCAL copy of `valid` — read
    // asynchronously, when the NIU processes the command. Preset it to VALID once; any later
    // write to this word must be preceded by a flush/barrier (see the teardown, where the
    // barrier deliberately comes BEFORE the reset).
    valid.set(VALID);

    // Hand-off CB base addresses, captured BEFORE any pop (read_ptr starts at the CB base, and
    // the union declaration makes each base identical on the receiver). The explicit destination
    // for global chunk c is base + (c % NBUF)*slot_bytes — the slot the receiver reserved.
    const uint32_t base_vbeta = CircularBuffer(cb_vbeta).get_read_ptr();
    const uint32_t base_Tinv = CircularBuffer(cb_Tinv).get_read_ptr();
    const uint32_t base_kd = CircularBuffer(cb_kd).get_read_ptr();
    const uint32_t base_intra = CircularBuffer(cb_intra).get_read_ptr();
    const uint32_t base_qdecay = CircularBuffer(cb_qdecay).get_read_ptr();
    const uint32_t base_kdec_t = CircularBuffer(cb_kdec_t).get_read_ptr();
    const uint32_t base_dl = CircularBuffer(cb_dl).get_read_ptr();

    // One linked 1x1-rect mcast per hand-off CB, sourced from the CB's front slot (this core's
    // local data for the chunk) into the receiver's explicitly computed slot. linked=true chains
    // the data mcasts with the valid-flag mcast on one static VC (data-before-flag ordering).
    MulticastEndpoint mcast_dst;
    auto send = [&](uint32_t cb_id, uint32_t n, uint32_t dst_addr) {
        const uint32_t addr = CircularBuffer(cb_id).get_read_ptr();
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(addr),
            mcast_dst,
            n * tb,
            1,
            {},
            {.noc_x_start = rcv_x, .noc_y_start = rcv_y, .noc_x_end = rcv_x, .noc_y_end = rcv_y, .addr = dst_addr},
            /*linked=*/true);
    };

    for (uint32_t c = p; c < NC; c += NP) {
        const uint32_t slot = c % NBUF;  // the receiver's reserved slot for GLOBAL chunk c
        // Wait for the chunk's outputs in the phased prep writer's drain order (roughly
        // compute's push order), so producer-side backpressure matches that writer exactly.
        CircularBuffer(cb_vbeta).wait_front(cv);
        CircularBuffer(cb_Tinv).wait_front(cc);
        CircularBuffer(cb_kd).wait_front(ck);
        CircularBuffer(cb_intra).wait_front(cc);
        CircularBuffer(cb_qdecay).wait_front(ck);
        CircularBuffer(cb_kdec_t).wait_front(kc);
        CircularBuffer(cb_dl).wait_front(1);

        // The receiver has reserved chunk c's CB space (its slots are writable) and credited
        // THIS producer (it targets producer c % NP == p — that's how we own this credit).
        ready.wait(1);
        ready.set(0);

        send(cb_vbeta, cv, base_vbeta + slot * cv * tb);
        send(cb_Tinv, cc, base_Tinv + slot * cc * tb);
        send(cb_kd, ck, base_kd + slot * ck * tb);
        send(cb_intra, cc, base_intra + slot * cc * tb);
        send(cb_qdecay, ck, base_qdecay + slot * ck * tb);
        send(cb_kdec_t, kc, base_kdec_t + slot * kc * tb);
        send(cb_dl, 1, base_dl + slot * 1 * tb);
        // Flush on EVERY arch, for two reasons. Blackhole: NoC latency exceeds L1<->RISCV
        // latency, so without it the receiver could see VALID with data still in flight. All
        // arches: the flush proves every data mcast has read its L1 source slot; only then may
        // the pops below let compute reuse the slots for this producer's NEXT owned chunk (the
        // flag mcast runs on a different cmd buf and provides no such ordering).
        noc.async_writes_flushed();
        valid.set_multicast(noc, rcv_x, rcv_y, rcv_x, rcv_y, 1);  // unlinked: ends the chain

        // Free the slots for compute's next chunk only now (the mcasts sourced them).
        CircularBuffer(cb_vbeta).pop_front(cv);
        CircularBuffer(cb_Tinv).pop_front(cc);
        CircularBuffer(cb_kd).pop_front(ck);
        CircularBuffer(cb_intra).pop_front(cc);
        CircularBuffer(cb_qdecay).pop_front(ck);
        CircularBuffer(cb_kdec_t).pop_front(kc);
        CircularBuffer(cb_dl).pop_front(1);
    }

    // Barrier BEFORE resetting the local valid word: set_multicast reads its payload from that
    // word asynchronously, so resetting first could multicast INVALID for the final chunk and
    // deadlock the receiver at valid.wait(VALID). Only after all nonposted writes are acked is
    // the local reset safe (payload-source rule).
    noc.async_write_barrier();
    valid.set(INVALID);  // restore the semaphore's initial value
}
