/*
 * Copyright (c) 2024-2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

/*
 * TT-RDMA MODIFICATIONS (Tenstorrent, 2026) to the NVIDIA DOCA sample
 * `eth_txq_batch_send_ethernet_frames` — turns the one-shot demo into a line-rate TT-RDMA-v1 HW-TX
 * blaster (the BF3 gateway TX leg + BH-RX-ceiling finder). See docs/tt-rdma-v1/tt-rdma-gateway-sender.md.
 * Changes vs the stock sample:
 *   - REGULAR_PKT_SIZE 1500 -> 4126 (14 L2 + 32 TT hdr + 4080 jumbo payload); TASKS_IN_TASK_BATCH -> 64;
 *     TASK_BATCHES_NUM 1 -> 4 (pipeline depth).
 *   - Frame builder emits ethertype 0x1AF6 + a 32B little-endian tt_rdma_hdr_t (WRITE opcode 0x10,
 *     rkey 0x00CAFE42) + payload.
 *   - Send callback keeps bufs alive (re-submit pre-filled frames) and drops per-packet logging.
 *   - Run loop pipelined: keep TASK_BATCHES_NUM batches in flight, busy-poll (no per-batch drain).
 * Measured ~143 Gbps wire on a BlueField-3 uplink (mlx5_0). Build/run: deploy_doca_sender.sh.
 */

#include <endian.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_cpu_data_path.h>
#include <doca_error.h>
#include <doca_log.h>

#include "common.h"
#include "eth_common.h"
#include "eth_flow_common.h"

DOCA_LOG_REGISTER(ETH_TXQ_BATCH_SEND_ETHERNET_FRAMES);

#define SLEEP_IN_NANOS (10 * 1000)        /* sample the task batch every 10 microseconds  */
#define MAX_BURST_SIZE 256                /* Max burst size to set for eth_txq */
#define MAX_LIST_LENGTH 1                 /* Max number of elements in a doca_buf */
#define TASKS_IN_TASK_BATCH 64            /* Number of tasks associated with task batch (max 64) */
#define BUFS_NUM TASKS_IN_TASK_BATCH      /* Number of DOCA buffers */
#define TASK_BATCHES_NUM 4                /* Pipeline depth: keep this many batches in flight (4x64=256=MAX_BURST) */
#define REGULAR_PKT_SIZE 4126             /* 14 (L2) + 32 (TT-RDMA hdr) + 4080 (jumbo payload) */
#define SEND_TASK_BATCH_USER_DATA 0x43210 /* User data for send task batch */
#define UNKNOWN_ETHER_TYPE_IPV4 0x88b5    /* Unknown IPV4 type */
/* --- TT-RDMA blaster mods --- : re-submit the same pre-filled frames NUM_SEND_ITERS times (HW TX) */
#define TT_RDMA_ETHERTYPE 0x1AF6
#define TT_RDMA_WRITE_OPCODE 0x10
#define TT_RDMA_RKEY 0x00CAFE42u
#define NUM_SEND_ITERS 300000 /* 300k batches x 64 frames = ~19.2M jumbo frames */

struct eth_txq_batch_send_sample_objects {
    struct eth_core_resources core_resources;         /* A struct to hold ETH core resources */
    struct eth_flow_common_resources flow_resources;  /* A struct to hold flow resources */
    struct doca_eth_txq* eth_txq;                     /* DOCA ETH TXQ context */
    struct doca_buf* eth_frame_bufs[BUFS_NUM];        /* DOCA buffers array to contain regular ethernet frames */
    struct doca_task_batch* send_task_batch;          /* Send task batch */
    uint8_t src_mac_addr[DOCA_DEVINFO_MAC_ADDR_SIZE]; /* Device MAC address */
    uint32_t inflight_task_batches;                   /* In flight task batches */
};

/*
 * ETH TXQ send task batch common callback
 *
 * @task_batch [in]: Completed task batch
 * @tasks_num [in]: Task number associated with task batch
 * @ctx_user_data [in]: User provided data, used to store sample state
 * @task_batch_user_data [in]: User provided data, used for identifying the task batch
 * @task_user_data_array [in]: Array of user provided data, each used for identifying the each task behind task batch
 * @pkt_array [in]: Array of packets, each associated to one send task that's part of the send task batch
 * @status_array [in]: Array of status, each associated to one send task that's part of the send task batch (in
 * successful CB, all are DOCA_SUCCESS)
 */
static void task_batch_send_common_cb(
    struct doca_task_batch* task_batch,
    uint16_t tasks_num,
    union doca_data ctx_user_data,
    union doca_data task_batch_user_data,
    union doca_data* task_user_data_array,
    struct doca_buf** pkt_array,
    doca_error_t* status_array) {
    doca_error_t status;
    size_t packet_size;
    uint32_t* inflight_task_batches;

    /* TT-RDMA blaster: just mark the batch done. Do NOT free the bufs (they are re-submitted every
     * iteration with the same pre-filled frames) and do NOT log per-packet (300k batches would spam). */
    (void)status;
    (void)packet_size;
    (void)tasks_num;
    (void)task_batch_user_data;
    (void)task_user_data_array;
    (void)pkt_array;
    (void)status_array;
    inflight_task_batches = (uint32_t*)ctx_user_data.ptr;
    (*inflight_task_batches)--;

    doca_task_batch_free(task_batch);
}

/*
 * Destroy ETH TXQ context related resources
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to destroy its ETH TXQ context
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t destroy_eth_txq_ctx(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status;
    enum doca_ctx_states ctx_state;
    struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = SLEEP_IN_NANOS,
    };

    status = doca_ctx_stop(state->core_resources.core_objs.ctx);
    if (status == DOCA_ERROR_IN_PROGRESS) {
        while (state->inflight_task_batches != 0) {
            (void)doca_pe_progress(state->core_resources.core_objs.pe);
            nanosleep(&ts, &ts);
        }

        status = doca_ctx_get_state(state->core_resources.core_objs.ctx, &ctx_state);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed get status of context, err: %s", doca_error_get_name(status));
            return status;
        }

        status = ctx_state == DOCA_CTX_STATE_IDLE ? DOCA_SUCCESS : DOCA_ERROR_BAD_STATE;
    }

    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to stop DOCA context, err: %s", doca_error_get_name(status));
        return status;
    }

    status = doca_eth_txq_destroy(state->eth_txq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy DOCA ETH TXQ context, err: %s", doca_error_get_name(status));
        return status;
    }

    return DOCA_SUCCESS;
}

/*
 * Destroy DOCA buffers for the packets
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to destroy its packets DOCA buffers
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t destroy_eth_txq_packet_buffers(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status;

    for (uint32_t i = 0; i < BUFS_NUM; i++) {
        status = doca_buf_dec_refcount(state->eth_frame_bufs[i], NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy eth_frame_bufs[%u] buffer, err: %s", i, doca_error_get_name(status));
            return status;
        }
    }

    return DOCA_SUCCESS;
}

/*
 * Destroy ETH TXQ task batch
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to destroy its task batch
 */
static void destroy_eth_txq_task_batch(struct eth_txq_batch_send_sample_objects* state) {
    doca_task_batch_free(state->send_task_batch);
}

/*
 * Retrieve ETH TXQ task batch
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to retrieve its task batch
 */
static void retrieve_eth_txq_task_batch(struct eth_txq_batch_send_sample_objects* state) {
    struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = SLEEP_IN_NANOS,
    };

    while (state->inflight_task_batches != 0) {
        (void)doca_pe_progress(state->core_resources.core_objs.pe);
        nanosleep(&ts, &ts);
    }
}

/*
 * Submit ETH TXQ task batch
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to submit its task batch
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t submit_eth_txq_task_batch(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status;

    status = doca_task_batch_submit(state->send_task_batch);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to submit send task batch, err: %s", doca_error_get_name(status));
        return status;
    }

    state->inflight_task_batches++;

    return DOCA_SUCCESS;
}

/*
 * Create ETH TXQ task batch
 *
 * @state [in/out]: eth_txq_batch_send_sample_objects struct to create task batch with its ETH TXQ context
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t create_eth_txq_task_batch(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status;
    union doca_data task_batch_user_data;
    struct doca_buf** pkt_array;
    union doca_data* task_user_data_array;

    task_batch_user_data.u64 = SEND_TASK_BATCH_USER_DATA;
    status = doca_eth_txq_task_batch_send_allocate(
        state->eth_txq,
        TASKS_IN_TASK_BATCH,
        task_batch_user_data,
        &pkt_array,
        &task_user_data_array,
        &(state->send_task_batch));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate send task batch, err: %s", doca_error_get_name(status));
        return status;
    }

    for (uint32_t i = 0; i < TASKS_IN_TASK_BATCH; i++) {
        pkt_array[i] = state->eth_frame_bufs[i];
        task_user_data_array[i].u64 = i;
    }

    return DOCA_SUCCESS;
}

/*
 * Create DOCA buffers for the packets
 *
 * @dest_mac_addr [in]: Destination MAC address to set in ethernet header
 * @state [in/out]: eth_txq_batch_send_sample_objects struct to create its packet DOCA buffers
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t create_eth_txq_packet_buffers(
    uint8_t* dest_mac_addr, struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status, clean_status;
    struct ether_hdr* eth_hdr;
    uint8_t* payload;
    void* pkt_addr;
    uint32_t i;

    for (i = 0; i < BUFS_NUM; i++) {
        pkt_addr = (void*)(((uint8_t*)state->core_resources.mmap_addr) + (i * REGULAR_PKT_SIZE));
        status = doca_buf_inventory_buf_get_by_data(
            state->core_resources.core_objs.buf_inv,
            state->core_resources.core_objs.src_mmap,
            pkt_addr,
            REGULAR_PKT_SIZE,
            &(state->eth_frame_bufs[i]));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR(
                "Failed to create DOCA buffer for regular ethernet frame, err: %s", doca_error_get_name(status));
            break;
        }

        /* Create regular packet header + payload */
        eth_hdr = (struct ether_hdr*)pkt_addr;
        payload = (uint8_t*)(eth_hdr + 1);
        memcpy(&(eth_hdr->src_addr), state->src_mac_addr, DOCA_DEVINFO_MAC_ADDR_SIZE);
        memcpy(&(eth_hdr->dst_addr), dest_mac_addr, DOCA_DEVINFO_MAC_ADDR_SIZE);
        /* TT-RDMA-v1 WRITE frame: ethertype 0x1AF6, then a 32B little-endian header, then payload. */
        eth_hdr->ether_type = htobe16(TT_RDMA_ETHERTYPE);
        uint32_t tt_plen = REGULAR_PKT_SIZE - (uint32_t)sizeof(struct ether_hdr) - 32u;
        payload[0] = TT_RDMA_WRITE_OPCODE; /* opcode */
        payload[1] = 0x01;                 /* version_flags: ver=1 */
        payload[2] = 0;
        payload[3] = 0; /* tag */
        payload[4] = tt_plen & 0xff;
        payload[5] = (tt_plen >> 8) & 0xff;
        payload[6] = (tt_plen >> 16) & 0xff;
        payload[7] = (tt_plen >> 24) & 0xff; /* length */
        memset(&payload[8], 0, 4);           /* seq */
        payload[12] = TT_RDMA_RKEY & 0xff;
        payload[13] = (TT_RDMA_RKEY >> 8) & 0xff;
        payload[14] = (TT_RDMA_RKEY >> 16) & 0xff;
        payload[15] = (TT_RDMA_RKEY >> 24) & 0xff; /* rkey */
        memset(&payload[16], 0, 16);               /* remote_offset(8) + imm(4) + cksum(4) = 0 */
        payload[32] = 'T';
        payload[33] = 'T';
        payload[34] = 'W';
        payload[35] = 'R';
        memset(&payload[36], (int)i, tt_plen - 4); /* TT payload */
    }

    if (status != DOCA_SUCCESS) {
        for (uint32_t j = 0; j < i; j++) {
            clean_status = doca_buf_dec_refcount(state->eth_frame_bufs[j], NULL);
            if (clean_status != DOCA_SUCCESS) {
                return status;
            }
        }

        return status;
    }

    return DOCA_SUCCESS;
}

/*
 * Create ETH TXQ context related resources
 *
 * @state [in/out]: eth_txq_batch_send_sample_objects struct to create its ETH TXQ context
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
static doca_error_t create_eth_txq_ctx(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status, clean_status;
    union doca_data user_data;

    status = doca_eth_txq_create(state->core_resources.core_objs.dev, MAX_BURST_SIZE, &(state->eth_txq));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create ETH TXQ context, err: %s", doca_error_get_name(status));
        return status;
    }

    status = doca_eth_txq_set_type(state->eth_txq, DOCA_ETH_TXQ_TYPE_REGULAR);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set type, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    status = doca_eth_txq_task_batch_send_set_conf(
        state->eth_txq,
        DOCA_TASK_BATCH_MAX_TASKS_NUMBER_64,
        TASK_BATCHES_NUM,
        task_batch_send_common_cb,
        task_batch_send_common_cb);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to configure send task batch, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    state->core_resources.core_objs.ctx = doca_eth_txq_as_doca_ctx(state->eth_txq);
    if (state->core_resources.core_objs.ctx == NULL) {
        DOCA_LOG_ERR("Failed to retrieve DOCA ETH TXQ context as DOCA context, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    status = doca_pe_connect_ctx(state->core_resources.core_objs.pe, state->core_resources.core_objs.ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect PE, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    user_data.ptr = &(state->inflight_task_batches);
    status = doca_ctx_set_user_data(state->core_resources.core_objs.ctx, user_data);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set user data for DOCA context, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    status = doca_ctx_start(state->core_resources.core_objs.ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start DOCA context, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    status = doca_eth_txq_apply_queue_id(state->eth_txq, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to apply queue ID, err: %s", doca_error_get_name(status));
        goto destroy_eth_txq;
    }

    return DOCA_SUCCESS;
destroy_eth_txq:
    clean_status = doca_eth_txq_destroy(state->eth_txq);
    state->eth_txq = NULL;

    if (clean_status != DOCA_SUCCESS) {
        return clean_status;
    }

    return status;
}

/*
 * Clean sample resources
 *
 * @state [in]: eth_txq_batch_send_sample_objects struct to clean
 */
static void eth_txq_cleanup(struct eth_txq_batch_send_sample_objects* state) {
    doca_error_t status;

    if (state->flow_resources.df_port != NULL) {
        eth_flow_common_destroy_flow_port(&(state->flow_resources));
    }

    eth_flow_common_cleanup_flow();

    if (state->eth_txq != NULL) {
        status = destroy_eth_txq_ctx(state);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy eth_txq_ctx, err: %s", doca_error_get_name(status));
            return;
        }
    }

    if (state->core_resources.core_objs.dev != NULL) {
        status = destroy_eth_core_resources(&(state->core_resources));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy core_resources, err: %s", doca_error_get_name(status));
            return;
        }
    }
}

/*
 * Check if device supports needed capabilities
 *
 * @devinfo [in]: Device info for device to check
 * @return: DOCA_SUCCESS in case the device supports needed capabilities and DOCA_ERROR otherwise
 */
static doca_error_t check_device(struct doca_devinfo* devinfo) {
    doca_error_t status;
    uint32_t max_supported_burst_size;

    status = doca_eth_txq_cap_get_max_burst_size(devinfo, MAX_LIST_LENGTH, 0, &max_supported_burst_size);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get supported max burst size, err: %s", doca_error_get_name(status));
        return status;
    }

    if (max_supported_burst_size < MAX_BURST_SIZE) {
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    status = doca_eth_txq_cap_is_type_supported(devinfo, DOCA_ETH_TXQ_TYPE_REGULAR, DOCA_ETH_TXQ_DATA_PATH_TYPE_CPU);
    if (status != DOCA_SUCCESS && status != DOCA_ERROR_NOT_SUPPORTED) {
        DOCA_LOG_ERR("Failed to check supported type, err: %s", doca_error_get_name(status));
        return status;
    }

    return status;
}

/*
 * Run ETH TXQ batch send ethernet frames
 *
 * @ib_dev_name [in]: IB device name of a doca device
 * @dest_mac_addr [in]: destination MAC address to associate with the ethernet frames
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise
 */
doca_error_t eth_txq_batch_send_ethernet_frames(const char* ib_dev_name, uint8_t* dest_mac_addr) {
    doca_error_t status, clean_status;
    struct eth_txq_batch_send_sample_objects state;
    struct eth_core_config cfg = {
        .mmap_size = REGULAR_PKT_SIZE * BUFS_NUM,
        .inventory_num_elements = BUFS_NUM,
        .check_device = check_device,
        .ibdev_name = ib_dev_name};

    memset(&state, 0, sizeof(struct eth_txq_batch_send_sample_objects));
    status = allocate_eth_core_resources(&cfg, &(state.core_resources));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed allocate core resources, err: %s", doca_error_get_name(status));
        return status;
    }

    status = doca_devinfo_get_mac_addr(
        doca_dev_as_devinfo(state.core_resources.core_objs.dev), state.src_mac_addr, DOCA_DEVINFO_MAC_ADDR_SIZE);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get device MAC address, err: %s", doca_error_get_name(status));
        goto txq_cleanup;
    }

    status = create_eth_txq_ctx(&state);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create/start ETH TXQ context, err: %s", doca_error_get_name(status));
        goto txq_cleanup;
    }

    status = eth_flow_common_init_flow();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to init flow, err: %s", doca_error_get_name(status));
        goto txq_cleanup;
    }

    status = eth_flow_common_create_flow_port(state.core_resources.core_objs.dev, 0, &(state.flow_resources));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create flow port, err: %s", doca_error_get_name(status));
        goto txq_cleanup;
    }

    status = create_eth_txq_packet_buffers(dest_mac_addr, &state);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create packet buffers, err: %s", doca_error_get_name(status));
        goto txq_cleanup;
    }

    /* TT-RDMA blaster loop: the frames are pre-filled once (bufs kept alive by the callback), so each
     * iteration just re-allocates a task batch over the same bufs, submits (HW TX), and drains. */
    DOCA_LOG_INFO(
        "TT-RDMA blaster: sending %d batches x %d frames = %ld jumbo frames on HW TX...",
        NUM_SEND_ITERS,
        TASKS_IN_TASK_BATCH,
        (long)NUM_SEND_ITERS * TASKS_IN_TASK_BATCH);
    for (long iter = 0; iter < NUM_SEND_ITERS; iter++) {
        /* Throttle to TASK_BATCHES_NUM batches in flight, busy-polling (no sleep) so the HW TX
         * pipeline stays full. The same pre-filled bufs are re-used across the in-flight batches. */
        while (state.inflight_task_batches >= TASK_BATCHES_NUM) {
            (void)doca_pe_progress(state.core_resources.core_objs.pe);
        }

        status = create_eth_txq_task_batch(&state);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create task batch (iter %ld), err: %s", iter, doca_error_get_name(status));
            goto drain_and_free;
        }

        status = submit_eth_txq_task_batch(&state);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to submit task batch (iter %ld), err: %s", iter, doca_error_get_name(status));
            goto drain_and_free;
        }
    }
drain_and_free:
    while (state.inflight_task_batches != 0) {
        (void)doca_pe_progress(state.core_resources.core_objs.pe);
    }
    DOCA_LOG_INFO("TT-RDMA blaster: done");

    goto destroy_packet_buffers;

destroy_txq_task_batch:
    destroy_eth_txq_task_batch(&state);
destroy_packet_buffers:
    clean_status = destroy_eth_txq_packet_buffers(&state);
    if (clean_status != DOCA_SUCCESS) {
        return clean_status;
    }
txq_cleanup:
    eth_txq_cleanup(&state);

    return status;
}
