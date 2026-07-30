/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#include <sys/socket.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <doca_rdma_bridge.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>
#include <doca_dpa.h>
#include <doca_error.h>
#include <doca_dev.h>
#include <doca_umem.h>
#include <doca_uar.h>
#include <doca_sync_event.h>
#include <doca_flow.h>

#include "common.h"
#include "dpa_common.h"
#include "../common/dpa_verbs_initiator_target_common_defs.h"

DOCA_LOG_REGISTER(DPA_VERBS::SAMPLE);

/* ---- TT-RDMA P1.4b re-head constants ---- */
#define TT_ETH_SQ_QID 0
#define TT_FLOW_PORT_ID 0
#define TT_BH_DMAC "02:00:00:00:00:02" /* Blackhole RXQ2 (override: TT_DMAC) */
#define TT_ETHERTYPE 0x1af6
#define TT_OPCODE_WRITE 0x10
#define TT_VER 0x01
#define TT_RKEY_DEFAULT 0x00CAFE42
#define TT_FRAME_HDR 46 /* 14B L2 + 32B TT-RDMA header (ETH SQ gather SGE 0) */
#define TT_SQ_DEPTH 256 /* ETH SQ wr_num + eth completion depth */
#define TT_PLEN_DEFAULT 1024

static inline void tt_put32_le(uint8_t* p, uint32_t v) {
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

/**
 * DPA sample application
 */
extern struct doca_dpa_app* dpa_sample_app;

/**
 * Sync event mask
 */
#define SYNC_EVENT_MASK_FFS (0xFFFFFFFFFFFFFFFF)
/**
 * Max IP address length
 */
#define MAX_IP_ADDRESS_LEN (128)
/**
 * Sample's sync event wait threshold
 */
#define VERBS_SAMPLE_EVENT_WAIT_THRESHOLD (9)
/**
 * QP queue size
 */
#define VERBS_SAMPLE_QUEUE_SIZE                                                  \
    (64) /* deep RQ now works — the depth-4 lock was the DBR umem (see below). \
          * Kernel still posts 1 recv/iter (Stage-2 pipelines N-in-flight). */
/**
 * Socket Port used for communication
 */
#define VERBS_SAMPLE_SIN_PORT (5000)
/**
 * Sample's Hop limit
 */
#define VERBS_SAMPLE_HOP_LIMIT (255)
/**
 * Sample's DBR size
 */
#define VERBS_SAMPLE_DBR_SIZE                                              \
    (4096) /* ★ THE depth-4 fix: a 64B DBR umem faults the DPA recv-post \
            * (Fatal 0x2) at rq_wr>4; a full 4KB page is required for the  \
            * external-datapath QP DBR. This one line unlocks deep RQ. */
/**
 * Max send scatter\gather elements
 */
#define VERBS_SAMPLE_MAX_SEND_SEGS (1)
/**
 * Max receive scatter\gather element
 */
#define VERBS_SAMPLE_MAX_RECEIVE_SEGS (1)
/**
 * Log WQEBB size
 */
#define VERBS_SAMPLE_LOG_WQEBB_SIZE (6)
/**
 * WQEBB size
 */
#define VERBS_SAMPLE_WQEBB_SIZE (1U << VERBS_SAMPLE_LOG_WQEBB_SIZE)
/**
 * Cacheline size
 */
#define VERBS_SAMPLE_CACHELINE_SIZE (64)

/**
 * kernel/RPC declaration
 */
doca_dpa_func_t initiator_thread_kernel;
doca_dpa_func_t initiator_trigger_first_iteration_rpc;
doca_dpa_func_t target_thread_kernel;
doca_dpa_func_t target_trigger_first_iteration_rpc;
doca_dpa_func_t tt_selftest_egress_rpc;
doca_dpa_func_t tt_egress_thread_kernel;
doca_dpa_func_t tt_kick_egress_rpc;

/**
 * Verbs Sample's Configuration Struct
 */
struct verbs_config {
    char pf_device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /* PF DOCA device name */
    char sf_device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /* SF DOCA device name */
    char target_ip_addr[MAX_IP_ADDRESS_LEN];           /* Target ip address */
    bool is_target;                                    /* Sample is acting as initiator or target */
    uint32_t gid_index;                                /* GID index */
};

/**
 * Verbs Sample's Resources Struct
 */
struct verbs_resources {
    struct verbs_config* cfg;           /* Verbs sample configuration parameters */
    struct doca_dev* pf_dev;            /* PF DOCA device */
    struct doca_dev* dev;               /* DOCA device to use, when running from host it will be PF DOCA device
                                and when running from DPU, it will be SF DOCA device */
    struct doca_dpa* pf_dpa_ctx;        /* PF DOCA DPA context */
    struct doca_dpa* dpa_ctx;           /* DOCA DPA context to use, when running from host it will be pf_dpa_ctx
                                and when running from DPU, it will be an extended dpa_ctx */
    doca_dpa_dev_t dpa_ctx_handle;      /* DOCA DPA context handle */
    struct doca_sync_event* comp_event; /* DOCA completion sync event */
    doca_dpa_dev_sync_event_t comp_event_handle;    /* DOCA completion sync event handle */
    doca_dpa_dev_uintptr_t dpa_thread_arg_dev_ptr;  /* DOCA DPA thread arguments handle */
    struct doca_dpa_thread* dpa_thread;             /* DOCA DPA thread */
    struct doca_verbs_context* verbs_context;       /* DOCA Verbs Context */
    struct doca_verbs_qp* verbs_qp;                 /* DOCA Verbs Queue Pair */
    struct doca_verbs_pd* verbs_pd;                 /* DOCA Verbs Protection Domain */
    struct doca_verbs_ah_attr* verbs_ah_attr;       /* DOCA Verbs Address Handle attribute */
    struct doca_umem* dpa_qp_umem;                  /* DOCA DPA QP umem */
    doca_dpa_dev_uintptr_t dpa_qp_umem_dev_ptr;     /* DOCA DPA QP umem handle */
    struct doca_umem* dpa_qp_dbr_umem;              /* DOCA DPA QP dbr umem */
    doca_dpa_dev_uintptr_t dpa_qp_dbr_umem_dev_ptr; /* DOCA DPA QP dbr umem handle */
    struct doca_uar* dpa_uar;                       /* DOCA DPA uar */
    int conn_socket;                                /* Connection socket fd */
    uint32_t local_qp_number;                       /* Local QP number */
    uint32_t remote_qp_number;                      /* Remote QP number */
    struct doca_verbs_gid gid;                      /* local gid address */
    struct doca_verbs_gid remote_gid;               /* remote gid address */
    doca_dpa_dev_verbs_qp_t verbs_qp_handle;        /* DOCA Verbs QP handle */
    struct dpa_completion_obj dpa_completion_obj;   /* DOCA DPA completion object */
    doca_dpa_dev_uintptr_t local_dpa_buff_addr;     /* DOCA DPA local buffer address */
    doca_dpa_dev_uintptr_t remote_dpa_buff_addr;    /* DOCA DPA remote buffer address */
    struct doca_mmap_obj local_buff_mmap_obj;       /* DOCA DPA local buffer mmap object */
    doca_dpa_dev_mmap_t remote_dpa_mmap_handle;     /* DOCA DPA remote buffer mmap handle */
    /* ---- TT-RDMA P1.4b re-head (target only) ---- */
    struct doca_verbs_context* pf_verbs_context;  /* PF verbs context (for the ETH SQ) */
    struct doca_verbs_pd* pf_verbs_pd;            /* PF verbs PD */
    struct doca_flow_port* df_port;               /* PF DOCA Flow port (egress) */
    struct doca_verbs_eth_sq* eth_sq;             /* PF ETH SQ -> p0 -> BH */
    struct dpa_completion_obj eth_completion_obj; /* ETH SQ send completion (unattached) */
    doca_dpa_dev_verbs_eth_sq_t eth_sq_handle;    /* ETH SQ DPA handle */
    void* tt_land_buf;                            /* host landing buffer (RoCE WRITE_IMM target) */
    struct ibv_mr* tt_land_sf_mr;                 /* landing buf reg on SF PD (recv + advertised rkey) */
    struct ibv_mr* tt_land_pf_mr;                 /* landing buf reg on PF PD (ETH gather source) */
    void* tt_hdr_buf;                             /* 46B TT frame-header template (host, for the selftest) */
    struct ibv_mr* tt_hdr_mr;                     /* header template reg on PF PD */
    uint32_t tt_plen;                             /* payload bytes per frame (TT_PLEN) */
    uint32_t tt_land_lkey;                        /* SF lkey for the recv SGE */
    /* ---- two-thread egress engine (Thread B on pf_dpa_ctx) ---- */
    doca_dpa_dev_uintptr_t tt_produced_addr;             /* DPA-heap `produced` counter (A writes, B reads) */
    doca_dpa_dev_uintptr_t tt_hdr_dpa_addr;              /* DPA-heap header (B patches seq -> DPA-local store) */
    struct doca_mmap_obj tt_hdr_dpa_mmap;                /* mmap over the DPA-heap header (ETH gather SGE0 mkey) */
    struct doca_dpa_thread* egress_thread;               /* Thread B */
    struct dpa_notification_completion_obj egress_notif; /* kicks Thread B */
    doca_dpa_dev_uintptr_t egress_arg_dev_ptr;           /* Thread B's dpa_thread_arg */
};

/*
 * Setup client connection
 *
 * @server_ip [in]: server IP address
 * @client_sock_fd [out]: client socket file descriptor
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t connection_client_setup(const char* server_ip, int* client_sock_fd) {
    struct sockaddr_in socket_addr = {0};
    int client_fd;

    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd < 0) {
        DOCA_LOG_ERR("Failed to create socket");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(VERBS_SAMPLE_SIN_PORT);

    if (inet_pton(AF_INET, server_ip, &(socket_addr.sin_addr)) <= 0) {
        close(client_fd);
        DOCA_LOG_ERR("inet_pton error occurred");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (connect(client_fd, (struct sockaddr*)&socket_addr, sizeof(socket_addr)) < 0) {
        close(client_fd);
        DOCA_LOG_ERR("Unable to connect to server at %s", server_ip);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }
    DOCA_LOG_INFO("Client has successfully connected to the server");

    *client_sock_fd = client_fd;
    return DOCA_SUCCESS;
}

/*
 * Setup server connection
 *
 * @server_sock_fd [out]: server socket file descriptor
 * @conn_socket [out]: connection socket
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t connection_server_setup(int* server_sock_fd, int* conn_socket) {
    struct sockaddr_in socket_addr = {0}, client_addr = {0};
    int addrlen = sizeof(client_addr);
    int opt = 1;
    int server_fd = 0;
    int new_socket = 0;
    char client_ip[INET_ADDRSTRLEN];

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        DOCA_LOG_ERR("Failed to create socket %d", server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        DOCA_LOG_ERR("Failed to set socket options");
        close(server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        DOCA_LOG_ERR("Failed to set socket options");
        close(server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(VERBS_SAMPLE_SIN_PORT);
    socket_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, (struct sockaddr*)&socket_addr, sizeof(socket_addr)) < 0) {
        DOCA_LOG_ERR("Failed to bind port");
        close(server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (listen(server_fd, 1) < 0) {
        DOCA_LOG_ERR("Failed to listen");
        close(server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }
    DOCA_LOG_INFO("Server is listening for incoming connections");

    new_socket = accept(server_fd, (struct sockaddr*)&client_addr, (socklen_t*)&addrlen);
    if (new_socket < 0) {
        DOCA_LOG_ERR("Failed to accept connection %d", new_socket);
        close(server_fd);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
    DOCA_LOG_INFO("Server is connected to client at IP: %s and port: %i", client_ip, ntohs(socket_addr.sin_port));

    *(server_sock_fd) = server_fd;
    *(conn_socket) = new_socket;

    return DOCA_SUCCESS;
}

/*
 * Close client's oob connection
 *
 * @oob_sock_fd [in]: client's oob socket file descriptor
 */
static void oob_connection_client_close(int oob_sock_fd) {
    if (oob_sock_fd > 0) {
        close(oob_sock_fd);
    }
}

/*
 * Close server's oob connection
 *
 * @oob_sock_fd [in]: server's oob socket file descriptor
 * @oob_client_sock [in]: client's oob socket file descriptor
 */
static void oob_connection_server_close(int oob_sock_fd, int oob_client_sock) {
    if (oob_client_sock > 0) {
        close(oob_client_sock);
    }

    if (oob_sock_fd > 0) {
        close(oob_sock_fd);
    }
}

/*
 * Create verbs AH
 *
 * @verbs_context [in]: verbs context
 * @gid_index [in]: gid index
 * @addr_type [in]: address type
 * @verbs_ah_attr [out]: verbs AH attribute
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t create_verbs_ah_attr(
    struct doca_verbs_context* verbs_context,
    uint32_t gid_index,
    enum doca_verbs_addr_type addr_type,
    struct doca_verbs_ah_attr** verbs_ah_attr) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_ah_attr* new_ah_attr = NULL;

    status = doca_verbs_ah_attr_create(verbs_context, &new_ah_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs ah: %s", doca_error_get_descr(status));
        return status;
    }

    status = doca_verbs_ah_attr_set_addr_type(new_ah_attr, addr_type);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set address type: %s", doca_error_get_descr(status));
        goto destroy_verbs_ah;
    }

    status = doca_verbs_ah_attr_set_sgid_index(new_ah_attr, gid_index);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set sgid index: %s", doca_error_get_descr(status));
        goto destroy_verbs_ah;
    }

    status = doca_verbs_ah_attr_set_hop_limit(new_ah_attr, VERBS_SAMPLE_HOP_LIMIT);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set hop limit: %s", doca_error_get_descr(status));
        goto destroy_verbs_ah;
    }

    *verbs_ah_attr = new_ah_attr;

    return DOCA_SUCCESS;

destroy_verbs_ah:
    tmp_status = doca_verbs_ah_attr_destroy(new_ah_attr);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs AH: %s", doca_error_get_descr(tmp_status));
    }

    return status;
}

/*
 * Calculate QP external umem size
 *
 * @rq_size [in]: receive queue size
 * @sq_size [in]: send queue size
 * @return: umem size
 */
static uint32_t calc_qp_external_umem_size(uint32_t rq_size, uint32_t sq_size) {
    uint32_t rq_ring_size = 0;
    uint32_t sq_ring_size = 0;

    if (rq_size != 0) {
        rq_ring_size = (uint32_t)(common_utils_next_power_of_two(rq_size) * sizeof(struct mlx5_wqe_data_seg));
    }
    if (sq_size != 0) {
        sq_ring_size = (uint32_t)(common_utils_next_power_of_two(sq_size) * VERBS_SAMPLE_WQEBB_SIZE);
    }

    return common_utils_align_up_uint32(rq_ring_size + sq_ring_size, VERBS_SAMPLE_CACHELINE_SIZE);
}

/*
 * Create verbs QP
 *
 * @verbs_context [in]: verbs context
 * @dpa_ctx [in]: DPA context
 * @verbs_pd [in]: verbs pd
 * @dpa_completion [in]: DPA completion
 * @dpa_uar [in]: DPA UAR
 * @qp_rq_wr [in]: QP receive queue work request
 * @qp_sq_wr [in]: QP send queue work request
 * @dpa_umem_dev_ptr [in]: DPA umem pointer
 * @dpa_umem [in]: DPA umem
 * @dpa_dbr_umem_dev_ptr [in]: DPA dbr umem pointer
 * @dpa_dbr_umem [in]:  DPA dbr umem
 * @verbs_qp [out]: verbs QP
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t create_verbs_qp(
    struct doca_verbs_context* verbs_context,
    struct doca_dpa* dpa_ctx,
    struct doca_verbs_pd* verbs_pd,
    struct doca_dpa_completion* dpa_completion,
    struct doca_uar* dpa_uar,
    uint32_t qp_rq_wr,
    uint32_t qp_sq_wr,
    doca_dpa_dev_uintptr_t* dpa_umem_dev_ptr,
    struct doca_umem** dpa_umem,
    doca_dpa_dev_uintptr_t* dpa_dbr_umem_dev_ptr,
    struct doca_umem** dpa_dbr_umem,
    struct doca_verbs_qp** verbs_qp) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_qp_init_attr* verbs_qp_init_attr = NULL;
    struct doca_verbs_qp* new_qp = NULL;
    uint32_t external_umem_size = 0;

    status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs qp attributes: %s", doca_error_get_descr(status));
        return status;
    }

    status = doca_verbs_qp_init_attr_set_external_datapath_en(verbs_qp_init_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs external datapath en: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    external_umem_size = calc_qp_external_umem_size(qp_rq_wr, qp_sq_wr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to calc external umem size: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_mem_alloc(dpa_ctx, external_umem_size, dpa_umem_dev_ptr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to alloc dpa memory for external umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_umem_dpa_create(
        dpa_ctx,
        *dpa_umem_dev_ptr,
        external_umem_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ |
            DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        dpa_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create dpa umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_umem(verbs_qp_init_attr, *dpa_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs qp external umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_mem_alloc(dpa_ctx, VERBS_SAMPLE_DBR_SIZE, dpa_dbr_umem_dev_ptr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to alloc dpa memory for external dbr umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_umem_dpa_create(
        dpa_ctx,
        *dpa_dbr_umem_dev_ptr,
        VERBS_SAMPLE_DBR_SIZE,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ |
            DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        dpa_dbr_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create dpa dbr umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_dbr_umem(verbs_qp_init_attr, *dpa_dbr_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs qp external dbr umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_uar(verbs_qp_init_attr, dpa_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs qp external uar: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_pd(verbs_qp_init_attr, verbs_pd);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs PD: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_sq_wr(verbs_qp_init_attr, qp_sq_wr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set SQ size: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_rq_wr(verbs_qp_init_attr, qp_rq_wr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set RQ size: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set QP type: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_dpa_completion(verbs_qp_init_attr, dpa_completion);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs CQ number: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_dpa_completion(verbs_qp_init_attr, dpa_completion);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs CQ number: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_max_sges(verbs_qp_init_attr, VERBS_SAMPLE_MAX_SEND_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set send_max_sges: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_max_sges(verbs_qp_init_attr, VERBS_SAMPLE_MAX_RECEIVE_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set receive_max_sges: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_create(verbs_context, verbs_qp_init_attr, &new_qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs QP: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    *verbs_qp = new_qp;

    return DOCA_SUCCESS;

destroy_resources:
    if (verbs_qp_init_attr != NULL) {
        tmp_status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes: %s", doca_error_get_descr(tmp_status));
        }
    }

    if (*dpa_dbr_umem != NULL) {
        tmp_status = doca_umem_destroy(*dpa_dbr_umem);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy dpa dbr umem: %s", doca_error_get_descr(tmp_status));
        }
    }

    if (*dpa_dbr_umem_dev_ptr != 0) {
        tmp_status = doca_dpa_mem_free(dpa_ctx, *dpa_dbr_umem_dev_ptr);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy dpa memory of dbr umem: %s", doca_error_get_descr(tmp_status));
        }
    }

    if (*dpa_umem != NULL) {
        tmp_status = doca_umem_destroy(*dpa_umem);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy dpa umem: %s", doca_error_get_descr(tmp_status));
        }
    }

    if (*dpa_umem_dev_ptr != 0) {
        tmp_status = doca_dpa_mem_free(dpa_ctx, *dpa_umem_dev_ptr);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy dpa memory of umem: %s", doca_error_get_descr(tmp_status));
        }
    }

    if (new_qp != NULL) {
        tmp_status = doca_verbs_qp_destroy(new_qp);
        if (tmp_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs QP: %s", doca_error_get_descr(tmp_status));
        }
    }

    return status;
}

/*
 * Create completion sync event
 *
 * @resources [in]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t create_completion_sync_event(struct verbs_resources* resources) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;

    status = doca_sync_event_create(&(resources->comp_event));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca sync event: %s", doca_error_get_descr(status));
        return status;
    }

    status = doca_sync_event_add_publisher_location_dpa(resources->comp_event, resources->pf_dpa_ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set dpa as publisher for doca sync event: %s", doca_error_get_descr(status));
        goto destroy_comp_event;
    }

    status = doca_sync_event_add_subscriber_location_cpu(resources->comp_event, resources->pf_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set cpu as subscriber for doca sync event: %s", doca_error_get_descr(status));
        goto destroy_comp_event;
    }

    status = doca_sync_event_start(resources->comp_event);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start doca sync event: %s", doca_error_get_descr(status));
        goto destroy_comp_event;
    }

    status =
        doca_sync_event_get_dpa_handle(resources->comp_event, resources->pf_dpa_ctx, &(resources->comp_event_handle));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get doca sync event dpa handle: %s", doca_error_get_descr(status));
        goto destroy_comp_event;
    }

    return status;

destroy_comp_event:
    tmp_status = doca_sync_event_destroy(resources->comp_event);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca sync event: %s", doca_error_get_descr(tmp_status));
    }

    return status;
}

/*
 * Open verbs context, verbs pd and doca device from device name
 *
 * @device_name [in]: device name
 * @verbs_ctx [out]: verbs context
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t open_verbs_resources(
    char* device_name, struct doca_verbs_context** verbs_ctx, struct doca_verbs_pd** verbs_pd, struct doca_dev** dev) {
    struct doca_devinfo** devinfo_list = NULL;
    char ibdev_name[DOCA_DEVINFO_IBDEV_NAME_SIZE + 1] = {0};
    uint32_t nb_devs = 0;
    doca_error_t status = DOCA_SUCCESS;

    status = doca_devinfo_create_list(&devinfo_list, &nb_devs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create devinfo list: %s", doca_error_get_descr(status));
        return status;
    }

    /* Search for the requested device */
    for (uint32_t i = 0; i < nb_devs; i++) {
        status = doca_devinfo_get_ibdev_name(devinfo_list[i], ibdev_name, DOCA_DEVINFO_IBDEV_NAME_SIZE);
        if (status == DOCA_SUCCESS && (strcmp(ibdev_name, device_name) == 0)) {
            status = doca_verbs_context_create(devinfo_list[i], DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, verbs_ctx);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to create verbs context: %s", doca_error_get_descr(status));
                (void)doca_devinfo_destroy_list(devinfo_list);
                return status;
            }

            status = doca_verbs_pd_create(*verbs_ctx, verbs_pd);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to create verbs pd: %s", doca_error_get_descr(status));
                (void)doca_verbs_context_destroy(*verbs_ctx);
                (void)doca_devinfo_destroy_list(devinfo_list);
                return status;
            }

            status = doca_verbs_pd_as_doca_dev(*verbs_pd, dev);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to create doca dev: %s", doca_error_get_descr(status));
                (void)doca_verbs_pd_destroy(*verbs_pd);
                (void)doca_verbs_context_destroy(*verbs_ctx);
                (void)doca_devinfo_destroy_list(devinfo_list);
                return status;
            }

            break;
        }
    }

    status = doca_devinfo_destroy_list(devinfo_list);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy devinfo list: %s", doca_error_get_descr(status));
        if (*verbs_ctx != NULL) {
            (void)doca_dev_close(*dev);
            (void)doca_verbs_pd_destroy(*verbs_pd);
            (void)doca_verbs_context_destroy(*verbs_ctx);
        }
        return status;
    }

    if (*verbs_ctx == NULL) {
        DOCA_LOG_ERR("The requested device was not found");
        return DOCA_ERROR_NOT_FOUND;
    }

    return DOCA_SUCCESS;
}

#ifdef DOCA_ARCH_DPU
/*
 * Open doca device from device name (without verbs resources)
 *
 * @device_name [in]: device name
 * @dev [out]: doca device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t open_doca_device(const char* device_name, struct doca_dev** dev) {
    struct doca_devinfo** devinfo_list = NULL;
    char ibdev_name[DOCA_DEVINFO_IBDEV_NAME_SIZE + 1] = {0};
    uint32_t nb_devs = 0;
    doca_error_t status;

    status = doca_devinfo_create_list(&devinfo_list, &nb_devs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create devinfo list: %s", doca_error_get_descr(status));
        return status;
    }

    /* Search for the requested device */
    for (uint32_t i = 0; i < nb_devs; i++) {
        status = doca_devinfo_get_ibdev_name(devinfo_list[i], ibdev_name, DOCA_DEVINFO_IBDEV_NAME_SIZE);
        if (status == DOCA_SUCCESS && (strcmp(ibdev_name, device_name) == 0)) {
            status = doca_dev_open(devinfo_list[i], dev);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to open doca device: %s", doca_error_get_descr(status));
                (void)doca_devinfo_destroy_list(devinfo_list);
                return status;
            }
            break;
        }
    }

    status = doca_devinfo_destroy_list(devinfo_list);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy devinfo list: %s", doca_error_get_descr(status));
        if (*dev != NULL) {
            (void)doca_dev_close(*dev);
        }
        return status;
    }

    if (*dev == NULL) {
        DOCA_LOG_ERR("The requested device was not found: %s", device_name);
        return DOCA_ERROR_NOT_FOUND;
    }

    return DOCA_SUCCESS;
}
#endif /* DOCA_ARCH_DPU */

/*
 * Destroy local resources
 *
 * @resources [in]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t destroy_local_resources(struct verbs_resources* resources) {
    doca_error_t status = DOCA_SUCCESS;

    if (resources->comp_event) {
        status = doca_sync_event_destroy(resources->comp_event);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca sync event: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->verbs_qp) {
        if (resources->dpa_qp_dbr_umem != NULL) {
            status = doca_umem_destroy(resources->dpa_qp_dbr_umem);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to destroy dpa qp dbr umem: %s", doca_error_get_descr(status));
                return status;
            }
        }

        if (resources->dpa_qp_dbr_umem_dev_ptr != 0) {
            status = doca_dpa_mem_free(resources->dpa_ctx, resources->dpa_qp_dbr_umem_dev_ptr);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to destroy dpa memory of qp dbr: %s", doca_error_get_descr(status));
                return status;
            }
        }

        if (resources->dpa_qp_umem != NULL) {
            status = doca_umem_destroy(resources->dpa_qp_umem);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to destroy dpa qp umem: %s", doca_error_get_descr(status));
                return status;
            }
        }

        if (resources->dpa_qp_umem_dev_ptr != 0) {
            status = doca_dpa_mem_free(resources->dpa_ctx, resources->dpa_qp_umem_dev_ptr);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to destroy dpa memory of qp ring buffer: %s", doca_error_get_descr(status));
                return status;
            }
        }

        status = doca_verbs_qp_destroy(resources->verbs_qp);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs QP: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->verbs_ah_attr) {
        status = doca_verbs_ah_attr_destroy(resources->verbs_ah_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs AH: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->dpa_completion_obj.dpa_comp) {
        status = dpa_completion_obj_destroy(&resources->dpa_completion_obj);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca dpa completion obj %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->dpa_uar) {
        status = doca_uar_destroy(resources->dpa_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca uar: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->dpa_thread) {
        status = doca_dpa_thread_destroy(resources->dpa_thread);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca dpa thread: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->dpa_thread_arg_dev_ptr) {
        status = doca_dpa_mem_free(resources->dpa_ctx, resources->dpa_thread_arg_dev_ptr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca dpa thread argument memory: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->local_buff_mmap_obj.dpa_mmap_handle) {
        status = doca_mmap_obj_destroy(&resources->local_buff_mmap_obj);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca mmap object: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->local_dpa_buff_addr) {
        status = doca_dpa_mem_free(resources->dpa_ctx, resources->local_dpa_buff_addr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca dpa local buff memory: %s", doca_error_get_descr(status));
            return status;
        }
    }

#ifdef DOCA_ARCH_DPU
    if (resources->dpa_ctx) {
        status = doca_dpa_destroy(resources->dpa_ctx);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy extended doca dpa context: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->dev) {
        status = doca_dev_close(resources->dev);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to close sf device: %s", doca_error_get_descr(status));
            return status;
        }
    }
#endif

    if (resources->pf_dpa_ctx) {
        status = doca_dpa_destroy(resources->pf_dpa_ctx);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy pf doca dpa context: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->pf_dev) {
        status = doca_dev_close(resources->pf_dev);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to close pf device: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->verbs_pd) {
        status = doca_verbs_pd_destroy(resources->verbs_pd);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs PD: %s", doca_error_get_descr(status));
            return status;
        }
    }

    if (resources->verbs_context) {
        status = doca_verbs_context_destroy(resources->verbs_context);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to destroy doca verbs Context: %s", doca_error_get_descr(status));
            return status;
        }
    }

    return DOCA_SUCCESS;
}

/* ==================== TT-RDMA P1.4b re-head resources (target) ==================== */

/* Init DOCA Flow (vnf) once. */
static doca_error_t tt_init_flow(void) {
    doca_error_t result, tmp_result;
    struct doca_flow_cfg* flow_cfg;

    result = doca_flow_cfg_create(&flow_cfg);
    if (result != DOCA_SUCCESS) {
        return result;
    }
    result = doca_flow_cfg_set_pipe_queues(flow_cfg, 1);
    if (result == DOCA_SUCCESS) {
        result = doca_flow_cfg_set_mode_args(flow_cfg, "vnf");
    }
    if (result == DOCA_SUCCESS) {
        result = doca_flow_cfg_set_nr_counters(flow_cfg, (1 << 19));
    }
    if (result == DOCA_SUCCESS) {
        result = doca_flow_init(flow_cfg);
    }
    tmp_result = doca_flow_cfg_destroy(flow_cfg);
    if (tmp_result != DOCA_SUCCESS) {
        DOCA_ERROR_PROPAGATE(result, tmp_result);
    }
    return result;
}

/* Start a DOCA Flow port on the PF (vnf mode; egress needs no explicit rule -- see P1.4a). */
static doca_error_t tt_create_flow_port(struct doca_dev* dev, uint16_t port_id, struct doca_flow_port** df_port) {
    doca_error_t status, tmp;
    struct doca_flow_port_cfg* port_cfg;

    status = doca_flow_port_cfg_create(&port_cfg);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_flow_port_cfg_set_port_id(port_cfg, port_id);
    if (status == DOCA_SUCCESS) {
        status = doca_flow_port_cfg_set_dev(port_cfg, dev);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_flow_port_start(port_cfg, df_port);
    }
    tmp = doca_flow_port_cfg_destroy(port_cfg);
    if (tmp != DOCA_SUCCESS) {
        DOCA_ERROR_PROPAGATE(status, tmp);
    }
    return status;
}

/* Create the PF ETH SQ (max_sges=2 for the [hdr]+[payload] gather). */
static doca_error_t tt_create_eth_sq(
    struct doca_verbs_context* ctx,
    struct doca_verbs_pd* pd,
    struct doca_dpa* dpa_ctx,
    struct doca_dpa_completion* dpa_comp,
    uint32_t wr_num,
    uint16_t queue_id,
    struct doca_verbs_eth_sq** out) {
    doca_error_t status, tmp;
    struct doca_verbs_eth_sq_init_attr* attr = NULL;

    status = doca_verbs_eth_sq_init_attr_create(&attr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_verbs_eth_sq_init_attr_set_pd(attr, pd);
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_wr_num(attr, wr_num);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_max_sges(attr, 2);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_queue_id(attr, queue_id);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_dpa(attr, dpa_ctx);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_dpa_completion(attr, dpa_comp);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_ts_source_type(attr, DOCA_VERBS_TS_SOURCE_DEFAULT);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_init_attr_set_external_datapath_en(attr, 1);
    }
    if (status == DOCA_SUCCESS) {
        status = doca_verbs_eth_sq_create(ctx, attr, out);
    }
    tmp = doca_verbs_eth_sq_init_attr_destroy(attr);
    if (tmp != DOCA_SUCCESS && status == DOCA_SUCCESS) {
        status = tmp;
    }
    return status;
}

/*
 * Create the two-thread egress engine (Thread B) on pf_dpa_ctx: a DPA-heap `produced` flag, a DPA-heap TT
 * header (so the DPA can patch seq), a mmap over that header for the ETH gather mkey, and Thread B itself with
 * a notification completion to kick it. Requires the ETH SQ + landing buffer + host header template to exist.
 */
static doca_error_t create_tt_egress_engine(struct verbs_resources* r) {
    doca_error_t status;
    struct dpa_thread_arg arg = {0};

    /* DPA-heap produced counter (pf_dpa_ctx) */
    status = doca_dpa_mem_alloc(r->pf_dpa_ctx, sizeof(uint64_t), &r->tt_produced_addr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_dpa_memset(r->pf_dpa_ctx, r->tt_produced_addr, 0, sizeof(uint64_t));
    if (status != DOCA_SUCCESS) {
        return status;
    }

    /* DPA-heap header RING (pf_dpa_ctx): TT_RING slots, each pre-filled with the host template; the kernel
     * patches only the per-frame seq into slot (seq%TT_RING). mmap covers the whole ring for the ETH gather
     * SGE0 mkey. A ring (not one slot) is what lets the kernel pipeline without a header alias. */
    status = doca_dpa_mem_alloc(r->pf_dpa_ctx, (size_t)TT_RING * TT_HDR_STRIDE, &r->tt_hdr_dpa_addr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    for (uint32_t i = 0; i < TT_RING; i++) {
        status = doca_dpa_h2d_memcpy(
            r->pf_dpa_ctx, r->tt_hdr_dpa_addr + (uint64_t)i * TT_HDR_STRIDE, r->tt_hdr_buf, TT_FRAME_HDR);
        if (status != DOCA_SUCCESS) {
            return status;
        }
    }
    r->tt_hdr_dpa_mmap.mmap_type = MMAP_TYPE_DPA;
    r->tt_hdr_dpa_mmap.doca_dpa = r->pf_dpa_ctx;
    r->tt_hdr_dpa_mmap.doca_device = r->pf_dev;
    /* doca_mmap_obj_init always RDMA-exports -> needs RDMA perms (matches the stock local_buff_mmap_obj). */
    r->tt_hdr_dpa_mmap.permissions =
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ;
    r->tt_hdr_dpa_mmap.memrange_addr = (void*)r->tt_hdr_dpa_addr;
    r->tt_hdr_dpa_mmap.memrange_len = (size_t)TT_RING * TT_HDR_STRIDE;
    status = doca_mmap_obj_init(&r->tt_hdr_dpa_mmap);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to mmap DPA-heap header ring: %s", doca_error_get_descr(status));
        return status;
    }

    /* Thread B on pf_dpa_ctx */
    status = doca_dpa_mem_alloc(r->pf_dpa_ctx, sizeof(struct dpa_thread_arg), &r->egress_arg_dev_ptr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_dpa_thread_create(r->pf_dpa_ctx, &r->egress_thread);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_dpa_thread_set_func_arg(r->egress_thread, &tt_egress_thread_kernel, r->egress_arg_dev_ptr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_dpa_thread_start(r->egress_thread);
    if (status != DOCA_SUCCESS) {
        return status;
    }

    /* notification completion to kick Thread B into its busy-poll loop */
    r->egress_notif.doca_dpa = r->pf_dpa_ctx;
    r->egress_notif.thread = r->egress_thread;
    status = dpa_notification_completion_obj_init(&r->egress_notif);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to create egress notification: %s", doca_error_get_descr(status));
        return status;
    }
    status = doca_dpa_thread_run(r->egress_thread);
    if (status != DOCA_SUCCESS) {
        return status;
    }

    /* Fill Thread B's arg (runs natively on pf_dpa_ctx -> dpa_ctx_handle=0) */
    arg.dpa_ctx_handle = 0;
    arg.eth_sq_handle = r->eth_sq_handle;
    arg.eth_comp_handle = r->eth_completion_obj.handle;
    arg.tt_hdr_buf = r->tt_hdr_dpa_addr;
    arg.tt_hdr_mkey = r->tt_hdr_dpa_mmap.dpa_mmap_handle;
    arg.local_dpa_buff_addr = (doca_dpa_dev_uintptr_t)r->tt_land_buf;
    arg.tt_pay_mkey = r->tt_land_pf_mr->lkey;
    arg.tt_plen = r->tt_plen;
    arg.tt_frame_hdr = TT_FRAME_HDR;
    arg.tt_produced_addr = r->tt_produced_addr;
    status = doca_dpa_h2d_memcpy(r->pf_dpa_ctx, r->egress_arg_dev_ptr, &arg, sizeof(arg));
    if (status != DOCA_SUCCESS) {
        return status;
    }

    DOCA_LOG_INFO(
        "TT egress engine (Thread B) up: produced@0x%lx hdr_dpa@0x%lx hdr_mkey=0x%x",
        (uint64_t)r->tt_produced_addr,
        (uint64_t)r->tt_hdr_dpa_addr,
        r->tt_hdr_dpa_mmap.dpa_mmap_handle);
    return DOCA_SUCCESS;
}

/*
 * Create the TT-RDMA re-head resources on the PF: verbs ctx+PD, DOCA Flow port, ETH SQ (+ its send
 * completion), the host landing buffer (dual-registered: SF PD = RoCE WRITE_IMM target/advertised rkey,
 * PF PD = ETH SQ gather source), and the 46B TT frame-header template. Requires pf_dev + the extended
 * dpa_ctx + the SF verbs_pd to already exist.
 */
static doca_error_t create_tt_rehead_resources(struct verbs_resources* resources) {
    doca_error_t status;
    struct ibv_pd *sf_ibv_pd = NULL, *pf_ibv_pd = NULL;
    uint8_t dmac[6] = {0}, smac[6] = {0};
    const char* dmac_env = getenv("TT_DMAC");
    uint32_t rkey = getenv("TT_RKEY") ? (uint32_t)strtoul(getenv("TT_RKEY"), NULL, 0) : TT_RKEY_DEFAULT;
    uint8_t *f, *h;

    struct ibv_pd* pf_ibv_pd0 = NULL;

    resources->tt_plen = getenv("TT_PLEN") ? (uint32_t)strtoul(getenv("TT_PLEN"), NULL, 0) : TT_PLEN_DEFAULT;

    /* pf_verbs_context / pf_verbs_pd were created by open_verbs_resources(PF) in create_local_resources, and
     * pf_dpa_ctx was created from that verbs PD's dev -> the ETH SQ (verbs ctx/PD + pf_dpa_ctx) is fully
     * P1.4a-linked. The ETH SQ must be on the PF (mlx5_0) to egress p0 -> BH (an SF ETH SQ does NOT reach p0). */
    pf_ibv_pd0 = doca_verbs_bridge_verbs_pd_get_ibv_pd(resources->pf_verbs_pd);
    if (!pf_ibv_pd0) {
        DOCA_LOG_ERR("TT: failed to get PF ibv_pd");
        return DOCA_ERROR_DRIVER;
    }

    /* DOCA Flow on the PF (egress path to p0) */
    status = tt_init_flow();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to init DOCA Flow: %s", doca_error_get_descr(status));
        return status;
    }
    status = tt_create_flow_port(resources->pf_dev, TT_FLOW_PORT_ID, &resources->df_port);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to start PF flow port: %s", doca_error_get_descr(status));
        return status;
    }

    /* ETH SQ + its completion MUST be created on pf_dpa_ctx (creation on the extended ctx = DevX failure).
     * But the re-head THREAD runs on the extended dpa_ctx, so we fetch the ETH SQ + completion DPA handles
     * for the extended ctx (the extended ctx is built on pf_dpa_ctx, so it should reach these PF resources). */
    resources->eth_completion_obj.doca_dpa = resources->dpa_ctx; /* P-GW1: SF/extended ctx (single-context) */
    resources->eth_completion_obj.queue_size = TT_SQ_DEPTH;
    resources->eth_completion_obj.thread = NULL;
    status = dpa_completion_obj_init(&resources->eth_completion_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to create ETH SQ completion: %s", doca_error_get_descr(status));
        return status;
    }

    /* ETH SQ on the SF verbs ctx/PD + extended dpa_ctx (same ctx as the RC recv CQ) */
    status = tt_create_eth_sq(
        resources->verbs_context,
        resources->verbs_pd,
        resources->dpa_ctx,
        resources->eth_completion_obj.dpa_comp,
        TT_SQ_DEPTH,
        TT_ETH_SQ_QID,
        &resources->eth_sq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to create ETH SQ: %s", doca_error_get_descr(status));
        return status;
    }
    /* Handle for the extended ctx: single thread on this ctx drives BOTH the SF recv CQ and this SF ETH SQ. */
    status = doca_verbs_eth_sq_get_dpa_handle(resources->eth_sq, resources->dpa_ctx, &resources->eth_sq_handle);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to get ETH SQ DPA handle: %s", doca_error_get_descr(status));
        return status;
    }

    /* ibv PDs: SF (RC QP recv + advertised rkey) and PF (ETH SQ gather) */
    sf_ibv_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(resources->verbs_pd);
    pf_ibv_pd = pf_ibv_pd0;
    if (!sf_ibv_pd) {
        DOCA_LOG_ERR("TT: failed to get SF ibv_pd");
        return DOCA_ERROR_DRIVER;
    }

    /* Single-context (SF): landing buffer registered once on the SF PD -- serves the RC recv target +
     * advertised rkey AND the SF ETH SQ gather source (same PD as the ETH SQ). */
    (void)pf_ibv_pd;
    /* Stage-2 async-ring: TT_RING landing slots. The ring requester WRITEs frame i to
     * advertised_addr + (i%TT_RING)*plen, so the advertised MR must span the whole ring (a single-slot MR
     * would give the requester a remote-access error past slot 0). */
    resources->tt_land_buf = calloc(TT_RING, resources->tt_plen);
    if (!resources->tt_land_buf) {
        return DOCA_ERROR_NO_MEMORY;
    }
    resources->tt_land_sf_mr = ibv_reg_mr(
        sf_ibv_pd,
        resources->tt_land_buf,
        (size_t)TT_RING * resources->tt_plen,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!resources->tt_land_sf_mr) {
        DOCA_LOG_ERR("TT: failed to register landing buffer");
        return DOCA_ERROR_DRIVER;
    }
    resources->tt_land_pf_mr = resources->tt_land_sf_mr; /* single PD: gather source == recv MR */
    resources->tt_land_lkey = resources->tt_land_sf_mr->lkey;

    /* 46B TT frame-header template on the SF PD (ETH gather SGE 0) */
    resources->tt_hdr_buf = calloc(1, TT_FRAME_HDR);
    if (!resources->tt_hdr_buf) {
        return DOCA_ERROR_NO_MEMORY;
    }
    resources->tt_hdr_mr = ibv_reg_mr(sf_ibv_pd, resources->tt_hdr_buf, TT_FRAME_HDR, IBV_ACCESS_LOCAL_WRITE);
    if (!resources->tt_hdr_mr) {
        DOCA_LOG_ERR("TT: failed to register header template");
        return DOCA_ERROR_DRIVER;
    }

    sscanf(
        dmac_env ? dmac_env : TT_BH_DMAC,
        "%02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx",
        &dmac[0],
        &dmac[1],
        &dmac[2],
        &dmac[3],
        &dmac[4],
        &dmac[5]);
    (void)doca_devinfo_get_mac_addr(doca_dev_as_devinfo(resources->pf_dev), smac, DOCA_DEVINFO_MAC_ADDR_SIZE);
    f = (uint8_t*)resources->tt_hdr_buf;
    memcpy(f, dmac, 6);
    memcpy(f + 6, smac, 6);
    f[12] = (TT_ETHERTYPE >> 8) & 0xff;
    f[13] = TT_ETHERTYPE & 0xff;
    h = f + 14;
    h[0] = TT_OPCODE_WRITE;
    h[1] = TT_VER;
    h[2] = 0;
    h[3] = 0;
    tt_put32_le(h + 4, resources->tt_plen); /* length */
    tt_put32_le(h + 8, 0);                  /* seq (DPA patches) */
    tt_put32_le(h + 12, rkey);              /* rkey -> BH MR slot */
    tt_put32_le(h + 16, 0);
    tt_put32_le(h + 20, 0);
    tt_put32_le(h + 24, 0);
    tt_put32_le(h + 28, 0); /* CRC (pool ignores) */

    DOCA_LOG_INFO(
        "TT re-head resources: plen=%u sf_rkey=0x%x pf_lkey=0x%x hdr_lkey=0x%x land=%p dmac=%s",
        resources->tt_plen,
        resources->tt_land_sf_mr->rkey,
        resources->tt_land_pf_mr->lkey,
        resources->tt_hdr_mr->lkey,
        resources->tt_land_buf,
        dmac_env ? dmac_env : TT_BH_DMAC);

    /* DPA-heap header on the EXTENDED ctx: the single thread patches seq into it (DPA-local store; a host
     * header faulted, constraint #4). doca_mmap(MMAP_TYPE_DPA) gives the ETH gather SGE0 mkey. The host
     * header (tt_hdr_buf/tt_hdr_mr) above is the template source copied in here. */
    status = doca_dpa_mem_alloc(resources->dpa_ctx, TT_FRAME_HDR, &resources->tt_hdr_dpa_addr);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    status = doca_dpa_h2d_memcpy(resources->dpa_ctx, resources->tt_hdr_dpa_addr, resources->tt_hdr_buf, TT_FRAME_HDR);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    resources->tt_hdr_dpa_mmap.mmap_type = MMAP_TYPE_DPA;
    resources->tt_hdr_dpa_mmap.doca_dpa = resources->dpa_ctx;
    resources->tt_hdr_dpa_mmap.doca_device = resources->dev; /* SF */
    resources->tt_hdr_dpa_mmap.permissions =
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ;
    resources->tt_hdr_dpa_mmap.memrange_addr = (void*)resources->tt_hdr_dpa_addr;
    resources->tt_hdr_dpa_mmap.memrange_len = TT_FRAME_HDR;
    status = doca_mmap_obj_init(&resources->tt_hdr_dpa_mmap);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("TT: failed to mmap DPA-heap header: %s", doca_error_get_descr(status));
        return status;
    }
    DOCA_LOG_INFO(
        "TT single-thread re-head ready: hdr_dpa@0x%lx hdr_mkey=0x%x",
        (uint64_t)resources->tt_hdr_dpa_addr,
        resources->tt_hdr_dpa_mmap.dpa_mmap_handle);

    /* P-GW1 single-context: ONE thread (target_thread_kernel) does recv->re-head->egress on the SF ETH SQ.
     * The two-thread engine (create_tt_egress_engine) is superseded. */
    (void)create_tt_egress_engine; /* retained for reference */
    return DOCA_SUCCESS;
}

/*
 * Create local resources
 *
 * @cfg [in]: sample's verbs configuration
 * @resources [out]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t create_local_resources(struct verbs_config* cfg, struct verbs_resources* resources) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    union ibv_gid rgid;
    struct ibv_pd* pd;
    int ret = 0;
    resources->cfg = cfg;

#ifdef DOCA_ARCH_DPU
    /* On DPU: Open PF device WITH verbs -- the TT re-head ETH SQ needs a PF verbs ctx/PD, and creating
     * pf_dpa_ctx from this verbs PD's dev (doca_verbs_pd_as_doca_dev) gives the P1.4a linkage the ETH SQ
     * requires (open_doca_device's dev is not ETH-SQ-capable -> "Operation not supported"). */
    status = open_verbs_resources(
        cfg->pf_device_name, &resources->pf_verbs_context, &resources->pf_verbs_pd, &resources->pf_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open PF device with verbs: %s", doca_error_get_descr(status));
        return status;
    }

    /* On DPU: Open SF device with verbs for RDMA operations */
    status =
        open_verbs_resources(cfg->sf_device_name, &resources->verbs_context, &resources->verbs_pd, &resources->dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open SF device with verbs: %s", doca_error_get_descr(status));
        goto close_pf_dev;
    }
#else
    /* On Host: Open PF device with verbs (used for both DPA and RDMA) */
    status =
        open_verbs_resources(cfg->pf_device_name, &resources->verbs_context, &resources->verbs_pd, &resources->pf_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open PF device with verbs: %s", doca_error_get_descr(status));
        return status;
    }

    resources->dev = resources->pf_dev;
#endif

    status = doca_dpa_create(resources->pf_dev, &resources->pf_dpa_ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create pf doca dpa context: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_set_app(resources->pf_dpa_ctx, dpa_sample_app);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set pf doca dpa app: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_start(resources->pf_dpa_ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start pf doca dpa context: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

#ifdef DOCA_ARCH_DPU
    status = doca_dpa_device_extend(resources->pf_dpa_ctx, resources->dev, &resources->dpa_ctx);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to extend doca dpa context: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_get_dpa_handle(resources->dpa_ctx, &resources->dpa_ctx_handle);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get doca dpa context handle: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }
#else
    resources->dpa_ctx = resources->pf_dpa_ctx;
#endif

    status = doca_dpa_mem_alloc(resources->dpa_ctx, sizeof(uint64_t), &resources->local_dpa_buff_addr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_dpa_mem_alloc failed (%s)", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_memset(resources->dpa_ctx, resources->local_dpa_buff_addr, 0, sizeof(uint64_t));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_dpa_memset failed (%s)", doca_error_get_descr(status));
        goto destroy_resources;
    }

    resources->local_buff_mmap_obj.mmap_type = MMAP_TYPE_DPA;
    resources->local_buff_mmap_obj.doca_dpa = resources->dpa_ctx;
    resources->local_buff_mmap_obj.doca_device = resources->dev;
    resources->local_buff_mmap_obj.permissions =
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ;
    resources->local_buff_mmap_obj.memrange_addr = (void*)resources->local_dpa_buff_addr;
    resources->local_buff_mmap_obj.memrange_len = sizeof(uint64_t);
    status = doca_mmap_obj_init(&resources->local_buff_mmap_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_mmap_obj_init failed (%s)", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_dpa_mem_alloc(resources->dpa_ctx, sizeof(struct dpa_thread_arg), &resources->dpa_thread_arg_dev_ptr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to alloc doca dpa thread argument memory: %s", doca_error_get_descr(status));
        goto destroy_mmap_obj;
    }

    status = doca_dpa_thread_create(resources->dpa_ctx, &resources->dpa_thread);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca dpa thread: %s", doca_error_get_descr(status));
        goto destroy_mmap_obj;
    }

    if (cfg->is_target) {
        status = doca_dpa_thread_set_func_arg(
            resources->dpa_thread, &target_thread_kernel, resources->dpa_thread_arg_dev_ptr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set doca dpa thread func: %s", doca_error_get_descr(status));
            goto destroy_mmap_obj;
        }
    } else {
        status = doca_dpa_thread_set_func_arg(
            resources->dpa_thread, &initiator_thread_kernel, resources->dpa_thread_arg_dev_ptr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set doca dpa thread func: %s", doca_error_get_descr(status));
            goto destroy_mmap_obj;
        }
    }

    status = doca_dpa_thread_start(resources->dpa_thread);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start doca dpa thread: %s", doca_error_get_descr(status));
        goto destroy_mmap_obj;
    }

    status = doca_uar_dpa_create(resources->dpa_ctx, &resources->dpa_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca uar: %s", doca_error_get_descr(status));
        goto destroy_mmap_obj;
    }

    resources->dpa_completion_obj.doca_dpa = resources->dpa_ctx;
    resources->dpa_completion_obj.queue_size =
        VERBS_SAMPLE_QUEUE_SIZE; /* single depth knob (was a decoupled literal 4) */
    resources->dpa_completion_obj.thread = resources->dpa_thread;
    status = dpa_completion_obj_init(&resources->dpa_completion_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function dpa_completion_obj_init failed (%s)", doca_error_get_descr(status));
        goto destroy_mmap_obj;
    }

    status = doca_dpa_thread_run(resources->dpa_thread);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to run doca dpa thread: %s", doca_error_get_descr(status));
        goto destroy_completion_obj;
    }

    status = doca_rdma_bridge_get_dev_pd(resources->dev, &pd);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get ibv_pd: %s", doca_error_get_descr(status));
        goto destroy_completion_obj;
    }

    ret = ibv_query_gid(pd->context, 1, cfg->gid_index, &rgid);
    if (ret) {
        DOCA_LOG_ERR("Failed to query ibv gid attributes");
        status = DOCA_ERROR_DRIVER;
        goto destroy_completion_obj;
    }
    memcpy(resources->gid.raw, rgid.raw, DOCA_GID_BYTE_LENGTH);

    status = create_verbs_ah_attr(
        resources->verbs_context, cfg->gid_index, DOCA_VERBS_ADDR_TYPE_IPv4, &resources->verbs_ah_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs ah: %s", doca_error_get_descr(status));
        goto destroy_completion_obj;
    }

    status = create_verbs_qp(
        resources->verbs_context,
        resources->dpa_ctx,
        resources->verbs_pd,
        resources->dpa_completion_obj.dpa_comp,
        resources->dpa_uar,
        VERBS_SAMPLE_QUEUE_SIZE,
        VERBS_SAMPLE_QUEUE_SIZE,
        &resources->dpa_qp_umem_dev_ptr,
        &resources->dpa_qp_umem,
        &resources->dpa_qp_dbr_umem_dev_ptr,
        &resources->dpa_qp_dbr_umem,
        &resources->verbs_qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs qp: %s", doca_error_get_descr(status));
        goto destroy_completion_obj;
    }
    resources->local_qp_number = doca_verbs_qp_get_qpn(resources->verbs_qp);

    /* TT-RDMA re-head: the target (responder) adds the PF ETH SQ egress leg. */
    if (cfg->is_target) {
        status = create_tt_rehead_resources(resources);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create TT re-head resources: %s", doca_error_get_descr(status));
            goto destroy_completion_obj;
        }
    }

    status = create_completion_sync_event(resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create completion sync event: %s", doca_error_get_descr(status));
        goto destroy_completion_obj;
    }

    return DOCA_SUCCESS;

destroy_completion_obj:
    tmp_status = dpa_completion_obj_destroy(&resources->dpa_completion_obj);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca dpa completion obj %s", doca_error_get_descr(status));
    }

destroy_mmap_obj:
    tmp_status = doca_mmap_obj_destroy(&resources->local_buff_mmap_obj);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca mmap object: %s", doca_error_get_descr(status));
    }

destroy_resources:
    tmp_status = destroy_local_resources(resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy resources: %s", doca_error_get_descr(tmp_status));
    }

    return status;

#ifdef DOCA_ARCH_DPU
close_pf_dev:
    tmp_status = doca_dev_close(resources->pf_dev);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to close PF device: %s", doca_error_get_descr(tmp_status));
    }
    return status;
#endif
}

/*
 * Exchange local RDMA parameters with remote peer
 *
 * @resources [in]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t exchange_params_with_remote_peer(struct verbs_resources* resources) {
    /* TT re-head target advertises its HOST landing buffer (RoCE WRITE_IMM target) + SF rkey; the stock
     * initiator advertises its DPA-heap buffer + mmap handle. */
    uint64_t adv_addr =
        resources->cfg->is_target ? (uint64_t)resources->tt_land_buf : (uint64_t)resources->local_dpa_buff_addr;
    uint32_t adv_key =
        resources->cfg->is_target ? resources->tt_land_sf_mr->rkey : resources->local_buff_mmap_obj.dpa_mmap_handle;

    if (send(resources->conn_socket, &adv_addr, sizeof(uint64_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to send local buffer address");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_dpa_buff_addr, sizeof(uint64_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to receive remote buffer address ");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (send(resources->conn_socket, &adv_key, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to send local MKEY");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_dpa_mmap_handle, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to receive remote MKEY, err = %s", strerror(errno));
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (send(resources->conn_socket, &resources->local_qp_number, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to send local QP number");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_qp_number, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG_ERR("Failed to receive remote QP number, err = %s", strerror(errno));
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (send(resources->conn_socket, &resources->gid.raw, sizeof(resources->gid.raw), 0) < 0) {
        DOCA_LOG_ERR("Failed to send local GID address");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_gid.raw, sizeof(resources->gid.raw), 0) < 0) {
        DOCA_LOG_ERR("Failed to receive remote GID address, err = %s", strerror(errno));
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    return DOCA_SUCCESS;
}

/*
 * Connect local and remote QPs
 *
 * @resources [in]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t connect_verbs_qp(struct verbs_resources* resources) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_qp_attr* verbs_qp_attr = NULL;

    status = doca_verbs_ah_attr_set_gid(resources->verbs_ah_attr, resources->remote_gid);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set remote gid: %s", doca_error_get_descr(status));
        return status;
    }

    /* Create QP attributes */
    status = doca_verbs_qp_attr_create(&verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs QP attributes: %s", doca_error_get_descr(status));
        return status;
    }

    /* Set QP attributes for RST2INIT */
    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set next state: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set allow remote write: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set allow remote read: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set port number: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    /* Modify QP - RST2INIT */
    status = doca_verbs_qp_modify(
        resources->verbs_qp,
        verbs_qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to modify QP: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    /* Set QP attributes for INIT2RTR */
    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set next state: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set RQ PSN: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, resources->remote_qp_number);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set destination QP number: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set minimum RNR timer: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_MTU_SIZE_1K_BYTES);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set path MTU: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, resources->verbs_ah_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set address handle: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    /* Modify QP - INIT2RTR */
    status = doca_verbs_qp_modify(
        resources->verbs_qp,
        verbs_qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | DOCA_VERBS_QP_ATTR_DEST_QP_NUM |
            DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_AH_ATTR);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to modify QP: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    /* Set QP attributes for RTR2RTS */
    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set next state: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set SQ PSN: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    status = doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set ACK timeout: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set retry counter: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set RNR retry: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }
    /* Modify QP - RTR2RTS */
    status = doca_verbs_qp_modify(
        resources->verbs_qp,
        verbs_qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN | DOCA_VERBS_QP_ATTR_ACK_TIMEOUT |
            DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to modify QP: %s", doca_error_get_descr(status));
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes: %s", doca_error_get_descr(status));
        return status;
    }

    DOCA_LOG_INFO("QP has been successfully connected and ready to use");

    return DOCA_SUCCESS;

destroy_verbs_qp_attr:
    tmp_status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes: %s", doca_error_get_descr(tmp_status));
    }

    return status;
}

/*
 * Init data path attributes in DPA
 *
 * @resources [in]: verbs resources
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
static doca_error_t init_datapath_attr_in_dpa(struct verbs_resources* resources) {
    doca_error_t status = DOCA_SUCCESS;
    struct dpa_thread_arg arg = {0};

    /* Init qp on the dpa */
    status = doca_verbs_qp_get_dpa_handle(resources->verbs_qp, resources->dpa_ctx, &resources->verbs_qp_handle);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get dpa qp handle: %s", doca_error_get_descr(status));
        return status;
    }

    /* Copy start value to the initiator dpa buffer */
    if (!resources->cfg->is_target) {
        uint64_t local_buf_val = VERBS_SAMPLE_LOCAL_BUF_START_VALUE;
        status =
            doca_dpa_h2d_memcpy(resources->dpa_ctx, resources->local_dpa_buff_addr, &local_buf_val, sizeof(uint64_t));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR(
                "Failed to initialize dpa memory of qp external datapath attributes: %s", doca_error_get_descr(status));
            return status;
        }
    }

    /* init dpa thread arg on the dpa */
    arg.dpa_ctx_handle = resources->dpa_ctx_handle;
    arg.dpa_comp_handle = resources->dpa_completion_obj.handle;
    arg.dpa_verbs_qp_handle = resources->verbs_qp_handle;
    arg.comp_sync_event_handle = resources->comp_event_handle;
    arg.comp_sync_event_val = VERBS_SAMPLE_EVENT_WAIT_THRESHOLD + 1;
    if (resources->cfg->is_target) {
        /* TT re-head: recv lands into (and the ETH SQ gathers from) the host landing buffer. */
        arg.local_dpa_buff_addr = (doca_dpa_dev_uintptr_t)resources->tt_land_buf;
        arg.local_dpa_buff_addr_mmap_handle = resources->tt_land_lkey; /* SF lkey for the recv SGE */
        arg.local_dpa_buff_addr_length = resources->tt_plen;
        arg.eth_sq_handle = resources->eth_sq_handle;
        arg.eth_comp_handle = resources->eth_completion_obj.handle;
        arg.tt_hdr_buf = resources->tt_hdr_dpa_addr;                  /* DPA-heap header (seq-patchable) */
        arg.tt_hdr_mkey = resources->tt_hdr_dpa_mmap.dpa_mmap_handle; /* ETH gather SGE0 mkey */
        arg.tt_pay_mkey = resources->tt_land_pf_mr->lkey;             /* SF lkey (single PD) for the ETH gather SGE1 */
        arg.tt_plen = resources->tt_plen;
        arg.tt_frame_hdr = TT_FRAME_HDR;
        /* Thread A (this arg) bumps produced + notifies Thread B on each RC recv. */
        arg.tt_produced_addr = resources->tt_produced_addr;
        arg.tt_egress_notif_handle = resources->egress_notif.handle;
    } else {
        arg.local_dpa_buff_addr = resources->local_dpa_buff_addr;
        arg.local_dpa_buff_addr_mmap_handle = resources->local_buff_mmap_obj.dpa_mmap_handle;
        arg.local_dpa_buff_addr_length = resources->local_buff_mmap_obj.memrange_len;
    }
    arg.remote_dpa_buff_addr = resources->remote_dpa_buff_addr;
    arg.remote_dpa_buff_addr_mmap_handle = resources->remote_dpa_mmap_handle;
    status =
        doca_dpa_h2d_memcpy(resources->dpa_ctx, resources->dpa_thread_arg_dev_ptr, &arg, sizeof(struct dpa_thread_arg));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to update dpa thread argument: %s", doca_error_get_descr(status));
        return status;
    }

    return status;
}

/*
 * Target's Verbs sample
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_verbs_target(struct verbs_config* cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    int server_sock_fd = -1;
    uint64_t retval = 0;

    resources.conn_socket = -1;

    status = create_local_resources(cfg, &resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create local resources: %s", doca_error_get_descr(status));
        return status;
    }

    /* TT re-head requester-free self-test: egress TT_SELFTEST frames on the PF ETH SQ (no RoCE peer).
     * Proves the extended-ctx DPA drives the PF ETH SQ + the cross-PD host-mem gather works. */
    if (getenv("TT_SELFTEST")) {
        uint32_t n = (uint32_t)strtoul(getenv("TT_SELFTEST"), NULL, 0);
        DOCA_LOG_INFO("TT SELFTEST: egressing %u frames on the SF ETH SQ (plen=%u)...", n, resources.tt_plen);
        /* P-GW1: run on the extended ctx (where the SF ETH SQ lives). */
        status = doca_dpa_rpc(
            resources.dpa_ctx,
            &tt_selftest_egress_rpc,
            &retval,
            resources.dpa_ctx_handle,
            resources.eth_sq_handle,
            resources.eth_completion_obj.handle,
            (doca_dpa_dev_uintptr_t)resources.tt_hdr_buf,
            resources.tt_hdr_mr->lkey,
            (doca_dpa_dev_uintptr_t)resources.tt_land_buf,
            resources.tt_land_pf_mr->lkey,
            n,
            resources.tt_plen,
            (uint32_t)TT_FRAME_HDR);
        DOCA_LOG_INFO("TT SELFTEST done: rpc status=%s posted=%lu", doca_error_get_descr(status), retval);
        goto target_cleanup;
    }

    /* Two-thread self-test: kick Thread B, then host-bump `produced`=N -> Thread B egresses N frames on the
     * PF ETH SQ (no RoCE peer). Validates the DPA-heap-flag-driven egress engine + DPA-heap seq patch. */
    if (getenv("TT_SELFTEST2")) {
        uint64_t n = strtoull(getenv("TT_SELFTEST2"), NULL, 0);
        DOCA_LOG_INFO("TT SELFTEST2: kick Thread B + DPA-set produced=%lu (plen=%u)...", n, resources.tt_plen);
        status = doca_dpa_rpc(
            resources.pf_dpa_ctx,
            &tt_kick_egress_rpc,
            &retval,
            resources.egress_notif.handle,
            resources.tt_produced_addr,
            n);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("TT SELFTEST2: kick failed: %s", doca_error_get_descr(status));
            goto target_cleanup;
        }
        DOCA_LOG_INFO("TT SELFTEST2: kicked + produced set DPA-side; Thread B draining. Sleeping 3s...");
        sleep(3);
        {
            uint64_t consumed_seq = 0;
            (void)doca_dpa_d2h_memcpy(
                resources.pf_dpa_ctx, &consumed_seq, resources.tt_produced_addr, sizeof(consumed_seq));
            DOCA_LOG_INFO("TT SELFTEST2 done: produced readback=%lu (check p0 tx delta == %lu)", consumed_seq, n);
        }
        goto target_cleanup;
    }

    status = connection_server_setup(&server_sock_fd, &resources.conn_socket);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to setup OOB connection with remote peer: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    status = exchange_params_with_remote_peer(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to exchange params with remote peer: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    status = connect_verbs_qp(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect doca verbs QP: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    status = init_datapath_attr_in_dpa(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to init datapath attr in dpa: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    /* Start data path */
    status = doca_dpa_rpc(
        resources.dpa_ctx,
        &target_trigger_first_iteration_rpc,
        &retval,
        resources.dpa_ctx_handle,
        resources.verbs_qp_handle,
        (doca_dpa_dev_uintptr_t)resources.tt_land_buf,
        resources.tt_land_lkey,
        resources.tt_plen);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("RPC failed to post receive: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    DOCA_LOG_INFO("Waiting on completion sync event...");
    status = doca_sync_event_wait_gt(resources.comp_event, VERBS_SAMPLE_EVENT_WAIT_THRESHOLD, SYNC_EVENT_MASK_FFS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to wait on completion sync event: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }

    /* check return status*/
    struct dpa_thread_arg args;
    status =
        doca_dpa_d2h_memcpy(resources.dpa_ctx, &args, resources.dpa_thread_arg_dev_ptr, sizeof(struct dpa_thread_arg));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to copy dpa thread argument: %s", doca_error_get_descr(status));
        goto target_cleanup;
    }
    status = (args.return_status == 0) ? DOCA_SUCCESS : DOCA_ERROR_BAD_STATE;

target_cleanup:
    tmp_status = destroy_local_resources(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy local resources: %s", doca_error_get_descr(tmp_status));
        DOCA_ERROR_PROPAGATE(status, tmp_status);
    }

    oob_connection_server_close(server_sock_fd, resources.conn_socket);

    return status;
}

/*
 * Initiator's Verbs sample
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_verbs_initiator(struct verbs_config* cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    uint64_t retval = 0;

    resources.conn_socket = -1;

    status = create_local_resources(cfg, &resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs resources: %s", doca_error_get_descr(status));
        return status;
    }

    status = connection_client_setup(cfg->target_ip_addr, &resources.conn_socket);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to setup OOB connection with remote peer: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

    status = exchange_params_with_remote_peer(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to exchange params with remote peer: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

    status = connect_verbs_qp(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect doca verbs QP: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

    status = init_datapath_attr_in_dpa(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to init datapath attr in dpa: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

    /* sleep to make sure target did post receive */
    sleep(2);

    /* start data path */
    status = doca_dpa_rpc(
        resources.dpa_ctx,
        &initiator_trigger_first_iteration_rpc,
        &retval,
        resources.dpa_ctx_handle,
        resources.verbs_qp_handle,
        resources.local_dpa_buff_addr,
        resources.local_buff_mmap_obj.dpa_mmap_handle,
        resources.local_buff_mmap_obj.memrange_len,
        resources.remote_dpa_buff_addr,
        resources.remote_dpa_mmap_handle);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("RPC failed to trigger first iteration: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

    DOCA_LOG_INFO("Waiting on completion sync event...");
    status = doca_sync_event_wait_gt(resources.comp_event, VERBS_SAMPLE_EVENT_WAIT_THRESHOLD, SYNC_EVENT_MASK_FFS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to wait on completion sync event: %s", doca_error_get_descr(status));
        goto initiator_cleanup;
    }

initiator_cleanup:
    tmp_status = destroy_local_resources(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy local resources: %s", doca_error_get_descr(tmp_status));
        DOCA_ERROR_PROPAGATE(status, tmp_status);
    }

    oob_connection_client_close(resources.conn_socket);

    return status;
}
