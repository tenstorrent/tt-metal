/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef DPA_VERBS_INITIATOR_TARGET_COMMON_DEFS_H_
#define DPA_VERBS_INITIATOR_TARGET_COMMON_DEFS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * Verbs Sample Buffer Start Value
 */
#define VERBS_SAMPLE_LOCAL_BUF_START_VALUE (0x9)
/**
 * Verbs Sample Buffer End Value
 */
#define VERBS_SAMPLE_LOCAL_BUF_END_VALUE (0xF)

/* Stage-2 async-ring pipelining: TT_RING landing+header slots. The requester WRITEs frame i to
 * base+(i%TT_RING)*plen; the re-head kernel gathers the matching slot without waiting per-frame. Must exceed
 * max in-flight (RC RQ depth + ETH_MAX_INFLIGHT) so a wrapping WRITE never overwrites an un-egressed slot. */
#define TT_RING (256)

/* Header-ring slot STRIDE (bytes). The DPA-heap header-ring gather (ETH SQ SGE0) STALLS for slot>=1 when the
 * slots are packed at the raw 46B header stride — an unaligned DPA-heap gather offset never completes (proven
 * on silicon: pinning SGE0 to slot 0 while the landing MR SGE1 varies by 256B egresses fine; varying SGE0 by
 * 46B stalls after frame 1). Spacing header slots on a 64B-aligned stride fixes it. The gather LENGTH stays
 * TT_FRAME_HDR (46) — only the slot spacing is padded. Must be >= TT_FRAME_HDR and 64B-aligned. */
#define TT_HDR_STRIDE (64)

#if __cplusplus
extern "C" {
#endif

/**
 * Sample's DPA Thread arguments struct
 */
struct dpa_thread_arg {
    doca_dpa_dev_t dpa_ctx_handle;
    doca_dpa_dev_completion_t dpa_comp_handle;
    doca_dpa_dev_verbs_qp_t dpa_verbs_qp_handle;
    doca_dpa_dev_sync_event_t comp_sync_event_handle;
    int comp_sync_event_val;
    volatile doca_dpa_dev_uintptr_t local_dpa_buff_addr;
    doca_dpa_dev_uintptr_t remote_dpa_buff_addr;
    doca_dpa_dev_mmap_t local_dpa_buff_addr_mmap_handle;
    doca_dpa_dev_mmap_t remote_dpa_buff_addr_mmap_handle;
    uint32_t local_dpa_buff_addr_length;
    int return_status;
    /* ---- TT-RDMA P1.4b re-head additions ---- */
    doca_dpa_dev_verbs_eth_sq_t eth_sq_handle; /* PF ETH SQ (egress to BH on p0) */
    doca_dpa_dev_completion_t eth_comp_handle; /* ETH SQ send completion (separate from the RC recv CQ) */
    doca_dpa_dev_uintptr_t tt_hdr_buf;         /* 46B TT frame-header template (host patches all but seq) */
    uint32_t tt_hdr_mkey;                      /* PF mkey over tt_hdr_buf (ETH SQ gather SGE 0) */
    uint32_t tt_pay_mkey;                      /* PF mkey over the landing buffer (ETH SQ gather SGE 1) */
    uint32_t tt_plen;                          /* payload bytes per frame */
    uint32_t tt_frame_hdr;                     /* frame header length (46) */
    /* two-thread coupling: monotonic "produced" counter in DPA-heap (Thread A writes, Thread B reads) */
    doca_dpa_dev_uintptr_t tt_produced_addr;
    /* Thread A notifies Thread B's notification completion after each bump (event-driven wake). */
    doca_dpa_dev_notification_completion_t tt_egress_notif_handle;
} __dpa_global__;

#if __cplusplus
}
#endif

#endif /* DPA_VERBS_INITIATOR_TARGET_COMMON_DEFS_H_ */
