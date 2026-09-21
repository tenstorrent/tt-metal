// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused-program (chunk_gdn_fused) PRODUCER writer: replaces the prep writer's DRAM drain with a
// direct NoC hand-off into the NV RECEIVER cores of ONE head (per-head producer form; the pooled
// form generalizes the item walk and the owner function, not this protocol). Per item (chunk c):
//   1. wait for the seven compute-pushed intermediates
//        v_beta [C,V], t_inv [C,C], kd [C,K], intra [C,C], q_decay [C,K], k_dec_t [K,C], dl [1 tile]
//   2. wait credit[h] == NV      — every receiver of head h has reserved chunk c's slots
//      credit[h] <- 0
//   3. v_beta: NV V-slice writes (1x1-rectangle multicasts, unlinked), slice v -> receiver v's ring slot
//   4. the six V-independent tensors: LINKED multicasts to the head's 1xNV row rectangle (num_dests NV)
//   5. noc.async_write_barrier()  — waits for ACKS. A flush only proves departure, and the unicast
//      slices are not part of the linked chain, so only the barrier orders every byte before VALID.
//   6. VALID multicast to the rectangle (unlinked: ends the chain)
//   7. pop the seven CBs (the writes have completed, so compute may reuse the slots)
//
// Addressing: the seven hand-off CBs are declared on the UNION of producer and receiver cores, so this
// core's CB base == every receiver's CB base. Six shared CBs: the receiver reserves/pushes n tiles per
// chunk, so its slot for chunk c is base + (c % NBUF)*n*tile. v_beta: the CB is producer-sized
// (cv*NBUF tiles) but a receiver reserves only Ct*Vtl per chunk, so its ring has NV*NBUF slots and its
// slot for chunk c is base + ((c*Ct*Vtl) mod (cv*NBUF))*tile; row r of receiver v's slice is source
// tiles [r*Vt + v*Vtl, +Vtl) of this core's front slot.
//
// Credit words: BH words at CB_CREDIT's base + CREDIT_OFF — the last tile of the
// union-declared u/mask CB, hence the same L1 address on every core. Dispatch re-initializes only
// Semaphore objects per launch, so this kernel zeroes the words itself and then bumps the SEM_INIT
// semaphore on each of its receivers; a receiver credits nothing before all its producers have done so.
//
// Multicast rectangle: given by the host already ORDERED for this kernel's NoC (NOC_1 wants
// bottom-right -> top-left; coordinates themselves are virtual and never flipped on Blackhole).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"

// CB indices (prep compute's output slots == the scan side's hand-off slots;
// must match chunk_gdn_prep.cpp, chunk_gdn_scan.cpp and the fused program factory).
constexpr uint32_t cb_Tinv = 13, cb_vbeta = 14, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_kdec_t = 24, cb_dl = 22;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);          // FULL V in tiles: this core's v_beta width
    constexpr uint32_t SEM_VALID = get_compile_time_arg_val(3);   // producer -> receivers: "chunk c landed"
    constexpr uint32_t SEM_INIT = get_compile_time_arg_val(4);    // producer -> receivers: "credit words zeroed"
    constexpr uint32_t NBUF = get_compile_time_arg_val(5);        // hand-off CB depth (slots), from the factory
    constexpr uint32_t NV = get_compile_time_arg_val(6);          // receivers per head
    constexpr uint32_t Vtl = get_compile_time_arg_val(7);         // per-receiver V-slice width (tiles)
    constexpr uint32_t CB_CREDIT = get_compile_time_arg_val(8);   // union-declared CB holding the credit words
    constexpr uint32_t CREDIT_OFF = get_compile_time_arg_val(9);  // byte offset of credit[0] in that CB
    constexpr bool UNICAST = get_compile_time_arg_val(10) != 0;   // A/B: NV unicast writes instead of multicasts
    static_assert(Vtl * NV == Vt, "NV receivers must tile the full V width");

    const uint32_t NC = get_arg_val<uint32_t>(0);   // GLOBAL chunk count of this head
    const uint32_t NP = get_arg_val<uint32_t>(1);   // producers for this head
    const uint32_t p = get_arg_val<uint32_t>(2);    // this producer's index within the head: owns c = p, p+NP, ...
    const uint32_t h = get_arg_val<uint32_t>(3);    // head index (selects the credit word)
    const uint32_t BH = get_arg_val<uint32_t>(4);   // number of credit words to zero
    const uint32_t mx0 = get_arg_val<uint32_t>(5);  // multicast rectangle, ordered for this kernel's NoC
    const uint32_t my0 = get_arg_val<uint32_t>(6);
    const uint32_t mx1 = get_arg_val<uint32_t>(7);
    const uint32_t my1 = get_arg_val<uint32_t>(8);
    // Receiver v's virtual worker coords: args 9 + 2v, 10 + 2v (v_beta slice targets, init barrier).
    auto rcv_x = [](uint32_t v) { return get_arg_val<uint32_t>(9 + 2 * v); };
    auto rcv_y = [](uint32_t v) { return get_arg_val<uint32_t>(10 + 2 * v); };

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kc = Kt * Ct;
    constexpr uint32_t cvl = Ct * Vtl;       // a receiver's v_beta tiles per chunk
    constexpr uint32_t VB_RING = cv * NBUF;  // a receiver's v_beta ring, in tiles

    const uint32_t tb = get_tile_size(cb_vbeta);  // all hand-off CBs are fp32 -> same tile size

    Noc noc;
    Semaphore<> valid(SEM_VALID);
    Semaphore<> init(SEM_INIT);
    // set_multicast sources its 4-byte value from this core's LOCAL copy of `valid` — read
    // asynchronously, when the NIU processes the command. Preset it to VALID once; any later
    // write to this word must be preceded by a barrier (see the teardown).
    valid.set(VALID);

    // Credit words: zero them, then tell every receiver of this head (init barrier).
    volatile tt_l1_ptr uint32_t* credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CircularBuffer(CB_CREDIT).get_read_ptr() + CREDIT_OFF);
    for (uint32_t i = 0; i < BH; i++) {
        noc_semaphore_set(credit + i, 0);
    }
    for (uint32_t v = 0; v < NV; v++) {
        init.up(noc, rcv_x(v), rcv_y(v), 1);
    }

    // Hand-off CB base addresses, captured BEFORE any pop (read_ptr starts at the CB base, and
    // the union declaration makes each base identical on every receiver).
    const uint32_t base_vbeta = CircularBuffer(cb_vbeta).get_read_ptr();
    const uint32_t base_Tinv = CircularBuffer(cb_Tinv).get_read_ptr();
    const uint32_t base_kd = CircularBuffer(cb_kd).get_read_ptr();
    const uint32_t base_intra = CircularBuffer(cb_intra).get_read_ptr();
    const uint32_t base_qdecay = CircularBuffer(cb_qdecay).get_read_ptr();
    const uint32_t base_kdec_t = CircularBuffer(cb_kdec_t).get_read_ptr();
    const uint32_t base_dl = CircularBuffer(cb_dl).get_read_ptr();

    MulticastEndpoint mcast_dst;
    // One LINKED multicast of a shared CB's front slot into the head's rectangle.
    UnicastEndpoint ucast_dst;
    auto send_unicast = [&](uint32_t src_addr, uint32_t v, uint32_t n, uint32_t dst_addr) {
        noc.async_write(
            CoreLocalMem<uint32_t>(src_addr),
            ucast_dst,
            n * tb,
            {},
            {.noc_x = rcv_x(v), .noc_y = rcv_y(v), .addr = dst_addr});
    };
    auto send_shared = [&](uint32_t cb_id, uint32_t n, uint32_t dst_addr) {
        const uint32_t addr = CircularBuffer(cb_id).get_read_ptr();
        if constexpr (UNICAST) {
            for (uint32_t v = 0; v < NV; v++) {
                send_unicast(addr, v, n, dst_addr);
            }
            return;
        }
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(addr),
            mcast_dst,
            n * tb,
            NV,
            {},
            {.noc_x_start = mx0, .noc_y_start = my0, .noc_x_end = mx1, .noc_y_end = my1, .addr = dst_addr},
            /*linked=*/true);
    };
    // One UNLINKED write of `n` tiles to a single receiver (1x1 rectangle: orientation-neutral).
    auto send_slice = [&](uint32_t src_addr, uint32_t v, uint32_t n, uint32_t dst_addr) {
        if constexpr (UNICAST) {
            send_unicast(src_addr, v, n, dst_addr);
            return;
        }
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(src_addr),
            mcast_dst,
            n * tb,
            1,
            {},
            {.noc_x_start = rcv_x(v),
             .noc_y_start = rcv_y(v),
             .noc_x_end = rcv_x(v),
             .noc_y_end = rcv_y(v),
             .addr = dst_addr},
            /*linked=*/false);
    };

    for (uint32_t c = p; c < NC; c += NP) {
        const uint32_t slot = c % NBUF;  // the receivers' reserved slot for GLOBAL chunk c (shared CBs)
        // Wait for the chunk's outputs in the phased prep writer's drain order (roughly
        // compute's push order), so producer-side backpressure matches that writer exactly.
        {
            DeviceZoneScopedN("tx_wait_cb");
            CircularBuffer(cb_vbeta).wait_front(cv);
            CircularBuffer(cb_Tinv).wait_front(cc);
            CircularBuffer(cb_kd).wait_front(ck);
            CircularBuffer(cb_intra).wait_front(cc);
            CircularBuffer(cb_qdecay).wait_front(ck);
            CircularBuffer(cb_kdec_t).wait_front(kc);
            CircularBuffer(cb_dl).wait_front(1);
        }

        // All NV receivers of head h have reserved chunk c's slots. Exactly NV — an over-credit
        // would be a protocol bug and shows up as a hang here rather than as corrupt output.
        {
            DeviceZoneScopedN("tx_wait_credit");
            noc_semaphore_wait(credit + h, NV);
        }
        noc_semaphore_set(credit + h, 0);

        {
            DeviceZoneScopedN("tx_issue");
            // v_beta slices: row r of receiver v's slice <- this slot's tiles [r*Vt + v*Vtl, +Vtl).
            const uint32_t vb_src = CircularBuffer(cb_vbeta).get_read_ptr();
            const uint32_t vb_dst = base_vbeta + ((c * cvl) % VB_RING) * tb;
            for (uint32_t v = 0; v < NV; v++) {
                for (uint32_t r = 0; r < Ct; r++) {
                    send_slice(vb_src + (r * Vt + v * Vtl) * tb, v, Vtl, vb_dst + r * Vtl * tb);
                }
            }
            // The six shared tensors, linked, into the rectangle.
            send_shared(cb_Tinv, cc, base_Tinv + slot * cc * tb);
            send_shared(cb_kd, ck, base_kd + slot * ck * tb);
            send_shared(cb_intra, cc, base_intra + slot * cc * tb);
            send_shared(cb_qdecay, ck, base_qdecay + slot * ck * tb);
            send_shared(cb_kdec_t, kc, base_kdec_t + slot * kc * tb);
            send_shared(cb_dl, 1, base_dl + slot * 1 * tb);
        }
        {
            // Every write above must have LANDED before the flag: the slices are unlinked unicasts
            // and could otherwise overtake the flag. The barrier waits for acks (a flush would not).
            DeviceZoneScopedN("tx_barrier");
            noc.async_write_barrier();
        }
        {
            DeviceZoneScopedN("tx_valid");
            valid.set_multicast(noc, mx0, my0, mx1, my1, NV);  // unlinked: ends the chain
        }

        // Free the slots for compute's next chunk only now (the writes have completed).
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
    // deadlock the receivers at valid.wait(VALID). Only after all nonposted writes are acked is
    // the local reset safe (payload-source rule).
    noc.async_write_barrier();
    // Drain the init.up() atomics: non-posted increments, counted apart from writes, and no NoC
    // transaction may be outstanding at kernel exit. Their acks return on this NoC while the credits
    // that prove the increments landed arrive on the other, so only the barrier makes it a guarantee.
    noc.async_atomic_barrier();
    valid.set(INVALID);  // restore the semaphore's initial value
}
