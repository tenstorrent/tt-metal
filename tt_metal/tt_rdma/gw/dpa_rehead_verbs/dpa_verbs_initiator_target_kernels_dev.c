/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * TT-RDMA P1.4b: native event-driven RoCE->TT-RDMA re-head on the BF3 DPA.
 *
 * `target_thread_kernel` is the RC responder: a persistent DPA EU thread that, on each RC recv completion
 * (a RoCE WRITE_IMM landed its payload into the host landing buffer and consumed a recv WR -> CQE), re-heads
 * the landed payload as a TT-RDMA WRITE frame and gather-egresses it on the PF ETH SQ to the Blackhole,
 * then re-posts the recv. Arm does ZERO per-frame work; the RC-recv CQE hardware-triggers the loop (no
 * doorbell, no flexio_window -> dissolves the A3.3b doorbell serializer). Egress leg is the proven P1.4a
 * doca_verbs 2-SGE gather (`gw/dpa_rehead_verbs/`).
 *
 * Header handling: the host pre-builds a 46B TT frame-header TEMPLATE and replicates it across a TT_RING-slot
 * DPA-heap header ring (dmac=BH, smac=uplink, ethertype 0x1AF6, opcode/ver/plen/rkey/roff/imm fixed, seq=0).
 * The DPA patches ONLY the 4-byte seq field of the per-frame slot (frame offset 14 + 8 = 22). Stage-2 pipelines
 * the recv side via a deep pre-posted RQ but paces the ETH SQ 1:1 (reap one send CQE per posted frame) so the
 * SQ always drains; the per-slot ring keeps each in-flight frame's header independent.
 */

#include <doca_dpa_dev.h>
#include <doca_dpa_dev_verbs.h>
#include <doca_dpa_dev_buf.h>
#include <doca_dpa_dev_sync_event.h>
#include <dpaintrin.h>

#include "../common/dpa_verbs_initiator_target_common_defs.h"

/* TT-RDMA seq field lives at frame byte 22 (14B L2 + TT hdr offset 8). Little-endian store. */
#define TT_SEQ_OFFSET (14 + 8)

/* Stage-2 pipeline depth: recvs pre-posted before the requester starts (< RQ depth VERBS_SAMPLE_QUEUE_SIZE=64,
 * and < TT_RING). The loop re-posts one per completion, holding ~this many WRITE_IMMs in flight. */
#define TT_PREPOST (48)

static inline void tt_put32_le(uint8_t* p, uint32_t v) {
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

/* Post one RC receive WR (landing buffer SGE) and commit. */
static void tt_post_recv(
    doca_dpa_dev_verbs_qp_t qp, struct doca_dpa_dev_verbs_recv_wr* recv_wr, struct doca_dpa_dev_verbs_sge* sge) {
    doca_dpa_dev_verbs_recv_wr_set_sg_list(recv_wr, sge);
    doca_dpa_dev_verbs_recv_wr_set_sg_num_sge(recv_wr, 1);
    doca_dpa_dev_verbs_qp_post_recv_wr(qp, recv_wr);
    doca_dpa_dev_verbs_qp_commit_recv(qp);
}

/**
 * @brief Initiator thread kernel (unchanged stock; not used by the re-head target).
 */
__dpa_global__ void initiator_thread_kernel(uint64_t arg) {
    (void)arg;
    doca_dpa_dev_thread_finish();
}

/**
 * @brief Thread A (extended dpa_ctx): the RC responder / producer.
 *
 * P-GW1 single-thread, single-context, HW-EVENT-DRIVEN re-head. Activated by the SF RC recv completion (no
 * host round-trip = lowest latency). On each recv (a RoCE WRITE_IMM landed a payload into the landing buffer +
 * consumed a recv WR): re-post the recv, patch seq into the DPA-heap header, 2-SGE gather-egress
 * [hdr]+[landed payload] on the SF ETH SQ (reaches p0 via the HW-offloaded eswitch flow), wait the send CQE.
 * ETH SQ + RC QP are on the SAME (extended) ctx, so ONE thread drives both -- no cross-ctx, no Thread B.
 */
__dpa_global__ void target_thread_kernel(uint64_t arg) {
    struct dpa_thread_arg* ta = (struct dpa_thread_arg*)arg;
    doca_dpa_dev_completion_element_t ce;
    struct doca_dpa_dev_verbs_recv_wr recv_wr;
    struct doca_dpa_dev_verbs_sge rsge;      /* RC recv landing SGE (dummy for WRITE_IMM — data lands at the
                                              * requester's remote addr = ring slot, not this SGE) */
    uint8_t* hdr = (uint8_t*)ta->tt_hdr_buf; /* DPA-heap header RING base (seq-patchable per slot) */
    uint64_t seq = 0;
    uint64_t eth_posted = 0, eth_done = 0;

    if (ta->dpa_ctx_handle) {
        doca_dpa_dev_device_set(ta->dpa_ctx_handle);
    }

    rsge.addr = ta->local_dpa_buff_addr;
    rsge.lkey = ta->local_dpa_buff_addr_mmap_handle;
    rsge.length = ta->tt_plen;

    /* Stage-2 RING + 1:1 ETH pacing: TT_PREPOST recvs were pre-posted by the trigger RPC. Frame i lands at +
     * gathers from ring slot (i%TT_RING) (the requester WROTE base+slot*plen); seq is patched into header slot
     * (i%TT_RING). The recv side is pipelined via the deep RQ, but each posted ETH send is reaped 1:1 in the
     * same iteration so the ETH SQ always drains (the earlier no-wait async drain deadlocked at eth_done=1).
     * ETH throughput scaling past this 1-in-flight pacing is the Stage-3 N-EU concern. */
    while (1) {
        /* HW event: a RoCE WRITE_IMM completed on the SF RC QP. */
        while (!doca_dpa_dev_get_completion(ta->dpa_comp_handle, &ce));
        doca_dpa_dev_completion_ack(ta->dpa_comp_handle, 1);

        tt_post_recv(ta->dpa_verbs_qp_handle, &recv_wr, &rsge); /* re-arm one RQ slot */

        uint32_t slot = (uint32_t)(seq % TT_RING); /* LANDING ring slot (host MR gather — varies safely) */
        /* SINGLE header slot (DPA-heap offset 0). A DPA-heap gather at a non-zero offset never completes on this
         * silicon (proven: constant SGE0 egresses; any varying SGE0 — even 64B-aligned — stalls after frame 1;
         * the host-MR landing SGE1 varies fine). Safe to reuse one slot because blocking-drain-one keeps just
         * ONE eth send in flight: frame N's send completes before frame N+1 re-patches seq. */
        uint8_t* hslot = hdr;
        seq++;
        tt_put32_le(hslot + TT_SEQ_OFFSET, (uint32_t)seq); /* distinct seq per in-flight frame */

        /* Fix (send_wr-reuse): build a FRESH, zero-initialized WR + gather SGE list every iteration. The
         * async loop re-posts without waiting between posts, so a single reused struct risks being mutated
         * while a prior post still references it, and any field the setters don't touch would carry over
         * stale. A per-iteration struct removes that hazard. */
        struct doca_dpa_dev_verbs_send_wr send_wr = {0};
        struct doca_dpa_dev_verbs_sge gsge[2] = {0}; /* ETH gather: [DPA-heap hdr slot] + [landed payload slot] */
        gsge[0].addr = (uint64_t)hslot;
        gsge[0].length = ta->tt_frame_hdr;
        gsge[0].lkey = ta->tt_hdr_mkey;
        gsge[1].addr = ta->local_dpa_buff_addr + (uint64_t)slot * ta->tt_plen; /* landed payload slot */
        gsge[1].length = ta->tt_plen;
        gsge[1].lkey = ta->tt_pay_mkey;

        doca_dpa_dev_verbs_send_wr_set_sg_list(&send_wr, gsge);
        doca_dpa_dev_verbs_send_wr_set_sg_num_sge(&send_wr, 2);
        doca_dpa_dev_verbs_send_wr_set_opcode(&send_wr, DOCA_DPA_DEV_VERBS_SEND_WR_OPCODE_SEND);
        doca_dpa_dev_verbs_send_wr_set_send_flags(&send_wr, DOCA_DPA_DEV_VERBS_SEND_WR_FLAGS_SIGNALED);
        doca_dpa_dev_verbs_send_wr_set_fence_mode(&send_wr, DOCA_DPA_DEV_VERBS_SEND_WR_FM_NO_FENCE);
        doca_dpa_dev_verbs_eth_sq_post_send_wr(ta->eth_sq_handle, &send_wr);
        doca_dpa_dev_verbs_eth_sq_commit_send(ta->eth_sq_handle);
        eth_posted++;

        /* Fix (blocking-drain-one-per-iteration): reap EXACTLY ONE ETH send CQE this iteration, matching
         * the Stage-1 sync kernel's 1:1 post:reap pacing so the ETH SQ always drains. The prior async
         * non-blocking drain reaped ~1 CQE then eth_done stuck (the SQ stopped transmitting after ~2
         * frames -> backpressure deadlock). The recv side stays pipelined via the deep pre-posted RQ
         * (TT_PREPOST), so this is not the Stage-1 full serializer. */
        while (!doca_dpa_dev_get_completion(ta->eth_comp_handle, &ce));
        doca_dpa_dev_completion_type_t eth_ct = doca_dpa_dev_get_completion_type(ce);
        uint32_t eth_syn =
            (eth_ct == DOCA_DPA_DEV_COMP_SEND_ERR) ? doca_dpa_dev_completion_element_get_error_syndrome(ce) : 0;
        doca_dpa_dev_completion_ack(ta->eth_comp_handle, 1); /* read type/syndrome BEFORE ack recycles the CE */
        eth_done++;

        /* Diagnostic (free): confirm the ETH send actually SUCCEEDED. A SEND_ERR completion means a bad
         * WQE (gather addr/mkey/opcode), not a pacing/drain problem — surface it + its syndrome at once. */
        if (eth_ct == DOCA_DPA_DEV_COMP_SEND_ERR) {
            DOCA_DPA_DEV_LOG_INFO("ERR eth SEND_ERR seq=%lu syndrome=0x%x\n", seq, eth_syn);
        } else if (seq <= 3 || seq % 500 == 0) {
            DOCA_DPA_DEV_LOG_INFO(
                "DBG recv#%lu eth_posted=%lu eth_done=%lu type=%d\n", seq, eth_posted, eth_done, (int)eth_ct);
        }
    }
    doca_dpa_dev_thread_finish();
}

/**
 * @brief RPC: post the first RC receive (arms the RQ before the requester starts WRITE_IMM'ing).
 */
__dpa_rpc__ uint64_t target_trigger_first_iteration_rpc(
    doca_dpa_dev_t dpa_ctx_handle,
    doca_dpa_dev_verbs_qp_t dpa_verbs_qp_handle,
    uint64_t addr,
    uint32_t mkey,
    uint32_t length) {
    if (dpa_ctx_handle) {
        doca_dpa_dev_device_set(dpa_ctx_handle);
    }

    struct doca_dpa_dev_verbs_recv_wr recv_wr;
    struct doca_dpa_dev_verbs_sge sge;
    sge.addr = addr;
    sge.lkey = mkey;
    sge.length = length;

    /* Stage-2: pre-post TT_PREPOST recvs (N posts + ONE commit — the stock deep-RQ idiom) so multiple
     * WRITE_IMMs can land in parallel and the pipeline starts full. The kernel re-posts one per completion. */
    for (uint32_t i = 0; i < TT_PREPOST; i++) {
        doca_dpa_dev_verbs_recv_wr_set_sg_list(&recv_wr, &sge);
        doca_dpa_dev_verbs_recv_wr_set_sg_num_sge(&recv_wr, 1);
        doca_dpa_dev_verbs_qp_post_recv_wr(dpa_verbs_qp_handle, &recv_wr);
    }
    doca_dpa_dev_verbs_qp_commit_recv(dpa_verbs_qp_handle);
    return 0;
}

/**
 * @brief Thread B (pf_dpa_ctx): the egress engine.
 *
 * Busy-polls the DPA-heap `produced` counter (bumped by Thread A on each RC recv, or by the host in the
 * self-test). Per new frame: patch seq into the DPA-heap header (a DPA-local store -- allowed, unlike the
 * host header that faulted), 2-SGE gather-egress [hdr]+[landed payload] on the PF ETH SQ, drain the ETH CQE.
 * Runs on pf_dpa_ctx (the ONLY ctx that can drive the PF ETH SQ); kicked once via a notification completion.
 */
/* consumed count persists across Thread B activations (event-driven; reschedule re-runs it). */
static uint64_t g_tt_consumed = 0;

__dpa_global__ void tt_egress_thread_kernel(uint64_t arg) {
    struct dpa_thread_arg* ta = (struct dpa_thread_arg*)arg;
    volatile uint64_t* produced = (volatile uint64_t*)ta->tt_produced_addr;
    uint8_t* hdr = (uint8_t*)ta->tt_hdr_buf;
    struct doca_dpa_dev_verbs_sge sge[2];
    struct doca_dpa_dev_verbs_send_wr wr;
    doca_dpa_dev_completion_element_t ce;

    if (ta->dpa_ctx_handle) {
        doca_dpa_dev_device_set(ta->dpa_ctx_handle);
    }

    DOCA_DPA_DEV_LOG_INFO("Thread B activate: produced=%lu consumed=%lu\n", *produced, g_tt_consumed);

    sge[0].addr = ta->tt_hdr_buf;
    sge[0].length = ta->tt_frame_hdr;
    sge[0].lkey = ta->tt_hdr_mkey;
    sge[1].addr = ta->local_dpa_buff_addr;
    sge[1].length = ta->tt_plen;
    sge[1].lkey = ta->tt_pay_mkey;

    /* Drain every frame produced since the last activation, one gather-egress each. */
    while (g_tt_consumed < *produced) {
        g_tt_consumed++;
        tt_put32_le(hdr + TT_SEQ_OFFSET, (uint32_t)g_tt_consumed); /* DPA-heap store: OK */

        doca_dpa_dev_verbs_send_wr_set_sg_list(&wr, sge);
        doca_dpa_dev_verbs_send_wr_set_sg_num_sge(&wr, 2);
        doca_dpa_dev_verbs_send_wr_set_opcode(&wr, DOCA_DPA_DEV_VERBS_SEND_WR_OPCODE_SEND);
        doca_dpa_dev_verbs_send_wr_set_send_flags(&wr, DOCA_DPA_DEV_VERBS_SEND_WR_FLAGS_SIGNALED);
        doca_dpa_dev_verbs_send_wr_set_fence_mode(&wr, DOCA_DPA_DEV_VERBS_SEND_WR_FM_NO_FENCE);
        doca_dpa_dev_verbs_eth_sq_post_send_wr(ta->eth_sq_handle, &wr);
        doca_dpa_dev_verbs_eth_sq_commit_send(ta->eth_sq_handle);

        while (!doca_dpa_dev_get_completion(ta->eth_comp_handle, &ce));
        doca_dpa_dev_completion_ack(ta->eth_comp_handle, 1);
    }

    doca_dpa_dev_thread_reschedule();
}

/**
 * @brief RPC (pf_dpa_ctx): kick Thread B into its busy-poll loop via its notification completion.
 * If produced_addr != 0, also set produced=n DPA-side first (self-test: DPA-to-DPA flag write is coherent
 * with Thread B's DPA-side poll, unlike a host h2d into live-polled DPA memory).
 */
__dpa_rpc__ uint64_t tt_kick_egress_rpc(
    doca_dpa_dev_notification_completion_t notif_handle, doca_dpa_dev_uintptr_t produced_addr, uint64_t n) {
    if (produced_addr) {
        *(volatile uint64_t*)produced_addr = n;
    }
    doca_dpa_dev_thread_notify(notif_handle);
    return 0;
}

/**
 * @brief RPC: requester-free egress self-test.
 *
 * Egresses `count` TT-RDMA frames on the PF ETH SQ WITHOUT any RC recv -- proves the extended-ctx DPA drives
 * the PF ETH SQ + the cross-PD host-memory gather works, before wiring the RoCE requester. Patches seq into
 * the single header template per frame; batched signalling (last WR of each 64-batch) + drain 1 CQE.
 */
__dpa_rpc__ uint64_t tt_selftest_egress_rpc(
    doca_dpa_dev_t dpa_ctx_handle,
    doca_dpa_dev_verbs_eth_sq_t eth_sq_handle,
    doca_dpa_dev_completion_t eth_comp_handle,
    doca_dpa_dev_uintptr_t hdr_buf,
    uint32_t hdr_mkey,
    doca_dpa_dev_uintptr_t pay_buf,
    uint32_t pay_mkey,
    uint32_t count,
    uint32_t plen,
    uint32_t frame_hdr) {
    uint8_t* hdr = (uint8_t*)hdr_buf;
    struct doca_dpa_dev_verbs_sge sge[2];
    struct doca_dpa_dev_verbs_send_wr wr;
    doca_dpa_dev_completion_element_t ce;
    const uint32_t BATCH = 64;
    uint32_t posted = 0;

    if (dpa_ctx_handle) {
        doca_dpa_dev_device_set(dpa_ctx_handle);
    }

    sge[0].addr = hdr_buf;
    sge[0].length = frame_hdr;
    sge[0].lkey = hdr_mkey;
    sge[1].addr = pay_buf;
    sge[1].length = plen;
    sge[1].lkey = pay_mkey;

    while (posted < count) {
        uint32_t n = count - posted;
        if (n > BATCH) {
            n = BATCH;
        }
        for (uint32_t j = 0; j < n; j++) {
            enum doca_dpa_dev_verbs_send_wr_flags flags = (j == n - 1) ? DOCA_DPA_DEV_VERBS_SEND_WR_FLAGS_SIGNALED : 0;
            /* NOTE: no seq patch here -- hdr is HOST memory; the DPA can only gather it via mkey,
             * not store into it. The self-test only checks p0 egress, so a constant header is fine.
             * (The real target kernel patches seq into a DPA-HEAP header; see create_tt_rehead.) */
            (void)hdr;
            doca_dpa_dev_verbs_send_wr_set_sg_list(&wr, sge);
            doca_dpa_dev_verbs_send_wr_set_sg_num_sge(&wr, 2);
            doca_dpa_dev_verbs_send_wr_set_opcode(&wr, DOCA_DPA_DEV_VERBS_SEND_WR_OPCODE_SEND);
            doca_dpa_dev_verbs_send_wr_set_send_flags(&wr, flags);
            doca_dpa_dev_verbs_send_wr_set_fence_mode(&wr, DOCA_DPA_DEV_VERBS_SEND_WR_FM_NO_FENCE);
            doca_dpa_dev_verbs_eth_sq_post_send_wr(eth_sq_handle, &wr);
        }
        doca_dpa_dev_verbs_eth_sq_commit_send(eth_sq_handle);
        while (!doca_dpa_dev_get_completion(eth_comp_handle, &ce));
        doca_dpa_dev_completion_ack(eth_comp_handle, 1);
        posted += n;
    }
    return posted;
}

/**
 * @brief RPC: initiator first iteration (stock stub; unused by the re-head target).
 */
__dpa_rpc__ uint64_t initiator_trigger_first_iteration_rpc(
    doca_dpa_dev_t dpa_ctx_handle,
    doca_dpa_dev_verbs_qp_t dpa_verbs_qp_handle,
    uint64_t addr,
    uint32_t mkey,
    uint32_t length,
    doca_dpa_dev_uintptr_t remote_dpa_buff_addr,
    doca_dpa_dev_mmap_t remote_dpa_buff_addr_mmap_handle) {
    (void)dpa_ctx_handle;
    (void)dpa_verbs_qp_handle;
    (void)addr;
    (void)mkey;
    (void)length;
    (void)remote_dpa_buff_addr;
    (void)remote_dpa_buff_addr_mmap_handle;
    return 0;
}
