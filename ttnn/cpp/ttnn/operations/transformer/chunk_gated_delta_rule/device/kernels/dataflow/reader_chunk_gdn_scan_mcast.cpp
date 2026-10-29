// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) reader, leader-multicast variant (host: TT_GDN_SCAN_MCAST=1).
//
// The NV V-block cores of one value head consume byte-identical V-independent prep intermediates
// (kd, q_decay, intra, k_dec_t, dl, t_inv: 15 fp32 tiles per 32-token step at Ct=1, Kt=4). One leader
// core per head reads them once and multicasts each CB's step payload into the siblings' identical CB
// slots (matmul in0-mcast handshake: receivers reserve, signal ready, wait VALID). v_beta and the
// initial state are V-block specific and stay per core. CB slot addresses match across cores because
// every scan core has the same CB layout and pops each of these CBs exactly once per step.
//
// Pipelining (v2): the scan step is bound by the reader's serial chain (DRAM read latency -> wait for
// receivers -> 6 multicasts -> ack barrier), not by compute (probe: ~2 us math vs ~6 us dataflow per
// step). So (a) the reads for step c+1 (v_beta on every core, the shared tiles on the leader) are issued
// right after step c's multicast and only awaited at the top of step c+1, and (b) the six data
// multicasts are linked to the trailing VALID semaphore multicast (same NoC path => ordered), so the
// leader only needs async_writes_flushed() instead of a full write-ack barrier. This requires the
// shared CBs and v_beta to have 2 slots (host: nbuf 2 in mcast mode).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_dl = 11, cb_S = 8, cb_Tinv = 13;
constexpr uint32_t cb_vbeta = 17, cb_kd = 18, cb_qdecay = 19, cb_intra = 20, cb_kdec_t = 24;

#ifndef GDN_SCAN_SEM_SENDER
#error "reader_chunk_gdn_scan_mcast.cpp requires GDN_SCAN_SEM_SENDER/RECEIVER/VALID defines"
#endif
constexpr uint32_t sem_sender = GDN_SCAN_SEM_SENDER;      // on the leader: receivers-ready counter
constexpr uint32_t sem_receiver = GDN_SCAN_SEM_RECEIVER;  // on receivers: VALID when the step's tiles landed
constexpr uint32_t sem_valid = GDN_SCAN_SEM_VALID;        // constant VALID, relayed by the leader

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    (void)has_s0;

    constexpr auto vb_a = TensorAccessorArgs<5>();
    constexpr auto kd_a = TensorAccessorArgs<vb_a.next_compile_time_args_offset()>();
    constexpr auto qd_a = TensorAccessorArgs<kd_a.next_compile_time_args_offset()>();
    constexpr auto it_a = TensorAccessorArgs<qd_a.next_compile_time_args_offset()>();
    constexpr auto kc_a = TensorAccessorArgs<it_a.next_compile_time_args_offset()>();
    constexpr auto dl_a = TensorAccessorArgs<kc_a.next_compile_time_args_offset()>();
    constexpr auto ti_a = TensorAccessorArgs<dl_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<ti_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t vb_addr = get_arg_val<uint32_t>(3);
    const uint32_t kd_addr = get_arg_val<uint32_t>(4);
    const uint32_t qd_addr = get_arg_val<uint32_t>(5);
    const uint32_t it_addr = get_arg_val<uint32_t>(6);
    const uint32_t kc_addr = get_arg_val<uint32_t>(7);
    const uint32_t dl_addr = get_arg_val<uint32_t>(8);
    const uint32_t ti_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);
    // Multicast group (all NoC/virtual coords): leader flag, leader core, tight rectangle, receiver count.
    const uint32_t is_leader = get_arg_val<uint32_t>(11);
    const uint32_t leader_x = get_arg_val<uint32_t>(12);
    const uint32_t leader_y = get_arg_val<uint32_t>(13);
    const uint32_t rect_sx = get_arg_val<uint32_t>(14);
    const uint32_t rect_sy = get_arg_val<uint32_t>(15);
    const uint32_t rect_ex = get_arg_val<uint32_t>(16);
    const uint32_t rect_ey = get_arg_val<uint32_t>(17);
    const uint32_t num_dests = get_arg_val<uint32_t>(18);

    const uint32_t tb = get_tile_size(cb_vbeta);  // all inputs fp32 -> same tile size
    const auto vb_acc = TensorAccessor(vb_a, vb_addr, tb);
    const auto kd_acc = TensorAccessor(kd_a, kd_addr, tb);
    const auto qd_acc = TensorAccessor(qd_a, qd_addr, tb);
    const auto it_acc = TensorAccessor(it_a, it_addr, tb);
    const auto kc_acc = TensorAccessor(kc_a, kc_addr, tb);
    const auto dl_acc = TensorAccessor(dl_a, dl_addr, tb);
    const auto ti_acc = TensorAccessor(ti_a, ti_addr, tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;
    constexpr uint32_t cv = Ct * Vt;

    Noc noc;
    Semaphore<>(sem_valid).set(VALID);

    // Shared (V-independent) CBs in the order they are pushed; tile counts per step.
    constexpr uint32_t NSH = 6;
    constexpr uint32_t sh_cb[NSH] = {cb_kd, cb_qdecay, cb_intra, cb_kdec_t, cb_dl, cb_Tinv};
    constexpr uint32_t sh_n[NSH] = {ck, ck, cc, kc, 1, cc};

    // V-slice read into an already reserved CB slot (no barrier, no push): R row-groups of Vt tiles, DRAM
    // row stride Vt_full, this core's column offset vb*Vt; packed contiguously ([R, Vt]) in the CB.
    auto issue_vslice = [&](const auto& acc, uint32_t cb_id, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t src = row_base + r * Vt_full + vb * Vt;
            const uint32_t dstt = r * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(acc, cb, tb, {.page_id = src + vt}, {.offset_bytes = (dstt + vt) * tb});
            }
        }
    };
    // Issue the n tile reads of one shared tensor into its (already reserved) CB slot; no barrier.
    auto issue_reads = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
    };
    // Reserve this step's slot in every shared CB (same slot address on all group cores) and, on the
    // leader, issue the DRAM reads for it. `slot` receives the slot write pointers.
    auto prefetch_step = [&](uint32_t c, uint32_t* slot) {
        const uint32_t hc = h * NC + c;
        CircularBuffer(cb_vbeta).reserve_back(cv);
        issue_vslice(vb_acc, cb_vbeta, hc * Ct * Vt_full, Ct);  // v_beta [C, V] slice (per core)
        for (uint32_t i = 0; i < NSH; i++) {
            CircularBuffer cb(sh_cb[i]);
            cb.reserve_back(sh_n[i]);
            slot[i] = cb.get_write_ptr();
        }
        if (is_leader) {
            issue_reads(kd_acc, cb_kd, hc * ck, ck);
            issue_reads(qd_acc, cb_qdecay, hc * ck, ck);
            issue_reads(it_acc, cb_intra, hc * cc, cc);
            issue_reads(kc_acc, cb_kdec_t, hc * kc, kc);
            issue_reads(dl_acc, cb_dl, hc * 1, 1);
            issue_reads(ti_acc, cb_Tinv, hc * cc, cc);
        }
    };

    // initial state S [K, V] (once) — host always provides it (zeros if none). V-sliced, per core.
    CircularBuffer(cb_S).reserve_back(Kt * Vt);
    issue_vslice(s0_acc, cb_S, h * Kt * Vt_full, Kt);
    noc.async_read_barrier();
    CircularBuffer(cb_S).push_back(Kt * Vt);

    uint32_t slot[NSH];
    prefetch_step(0, slot);

    for (uint32_t c = 0; c < NC; c++) {
        // Step c's own reads (v_beta everywhere, the shared tiles on the leader) have landed.
        noc.async_read_barrier();
        CircularBuffer(cb_vbeta).push_back(cv);

        if (is_leader) {
            // All receivers have reserved their slots for this step.
            Semaphore<> sender_sem(sem_sender);
            sender_sem.wait(num_dests);
            sender_sem.set(0);
            // Six linked data multicasts followed by the VALID flag multicast on the same path: the flag
            // cannot overtake the data, so no write-ack barrier is needed before it (matmul in0-mcast
            // pattern). flushed() only waits for the packets to leave this core.
            for (uint32_t i = 0; i < NSH; i++) {
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(slot[i]),
                    MulticastEndpoint{},
                    sh_n[i] * tb,
                    num_dests,
                    {},
                    {.noc_x_start = rect_sx,
                     .noc_y_start = rect_sy,
                     .noc_x_end = rect_ex,
                     .noc_y_end = rect_ey,
                     .addr = slot[i]},
                    true /* linked: next packet on the same path */);
            }
            Semaphore<>(sem_valid).relay_multicast(
                noc, Semaphore<>(sem_receiver), rect_sx, rect_sy, rect_ex, rect_ey, num_dests, /*linked=*/false);
            noc.async_writes_flushed();
        } else {
            Semaphore<> receiver_sem(sem_receiver);
            receiver_sem.set(INVALID);
            Semaphore<>(sem_sender).up(noc, leader_x, leader_y, 1);
            receiver_sem.wait(VALID);
        }
        for (uint32_t i = 0; i < NSH; i++) {
            CircularBuffer(sh_cb[i]).push_back(sh_n[i]);
        }

        // Overlap the next step's DRAM latency with this step's compute (2-slot CBs; the reserve blocks
        // only until compute has popped step c-1).
        if (c + 1 < NC) {
            prefetch_step(c + 1, slot);
        }
    }
}
