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
// Addressing: the seven hand-off CBs are declared on the UNION of producer+receiver cores
// (identical base addresses both sides) with nbuf=1 and an equal per-chunk push/pop cadence, so
// this core's read_ptr == the receiver's reserved slot == the CB base (the shipped lockstep
// lemma) — each mcast writes source address -> same destination address. The destination is a
// 1x1 rectangle (one receiver), which is orientation-neutral: start == end, so this writer's NOC
// (whichever the WriterConfig assigns) needs no coordinate swap.

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

    const uint32_t NC = get_arg_val<uint32_t>(0);
    const uint32_t rcv_x = get_arg_val<uint32_t>(1);  // virtual worker coords of the receiver
    const uint32_t rcv_y = get_arg_val<uint32_t>(2);

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

    // One linked 1x1-rect mcast per hand-off CB, sourced from the CB's front slot (== the
    // receiver's reserved slot, lockstep lemma). linked=true chains the data mcasts with the
    // valid-flag mcast on one static VC (data-before-flag ordering).
    MulticastEndpoint mcast_dst;
    auto send = [&](uint32_t cb_id, uint32_t n) {
        const uint32_t addr = CircularBuffer(cb_id).get_read_ptr();
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(addr),
            mcast_dst,
            n * tb,
            1,
            {},
            {.noc_x_start = rcv_x, .noc_y_start = rcv_y, .noc_x_end = rcv_x, .noc_y_end = rcv_y, .addr = addr},
            /*linked=*/true);
    };

    for (uint32_t c = 0; c < NC; c++) {
        // Wait for the chunk's outputs in the phased prep writer's drain order (roughly
        // compute's push order), so producer-side backpressure matches that writer exactly.
        CircularBuffer(cb_vbeta).wait_front(cv);
        CircularBuffer(cb_Tinv).wait_front(cc);
        CircularBuffer(cb_kd).wait_front(ck);
        CircularBuffer(cb_intra).wait_front(cc);
        CircularBuffer(cb_qdecay).wait_front(ck);
        CircularBuffer(cb_kdec_t).wait_front(kc);
        CircularBuffer(cb_dl).wait_front(1);

        // The receiver has reserved this chunk's CB space (its slots are writable).
        ready.wait(1);
        ready.set(0);

        send(cb_vbeta, cv);
        send(cb_Tinv, cc);
        send(cb_kd, ck);
        send(cb_intra, cc);
        send(cb_qdecay, ck);
        send(cb_kdec_t, kc);
        send(cb_dl, 1);
        // Flush on EVERY arch, for two reasons. Blackhole: NoC latency exceeds L1<->RISCV
        // latency, so without it the receiver could see VALID with data still in flight. All
        // arches: the flush proves every data mcast has read its L1 source slot; only then may
        // the pops below let compute overwrite the slots with chunk c+1 (nbuf=1 — the flag
        // mcast runs on a different cmd buf and provides no such ordering).
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
