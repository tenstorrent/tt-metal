// SPDX-License-Identifier: BSD-3-Clause
//
// TT-RDMA P1.5: matched RoCEv2 WRITE_IMM requester for the DPA re-head target
// (doca_dpa/dpa_verbs_initiator_target, TT re-head build).
//
// Plain libibverbs (no DPA, no root) x86-host generator that speaks the sample's trivial socket OOB and posts
// N RDMA-WRITE-WITH-IMM to the target's advertised landing buffer. The WRITE lands the payload; the IMM
// consumes a recv WR on the target -> a recv CQE that wakes Thread A (the DPA RC responder), which bumps the
// DPA-heap `produced` counter + notifies Thread B (the pf_dpa_ctx egress engine) -> re-head -> p0 -> BH.
//
// OOB (mirrors exchange_params_with_remote_peer, per-field send-then-recv):
//   u64 buff_addr, u32 mkey(rkey), u32 qpn, 16B gid.raw.
// QP params (mirror the target): RoCEv2 IPv4, path_mtu 1K, rq_psn/sq_psn 0, ack_timeout 14, retry 7.
//
//   gcc -O2 -o tt_p15_requester tt_p15_requester.c -libverbs
//   ./tt_p15_requester -s 10.99.0.1 -d mlx5_0 -g 3 -n 1000 -l 1024   # host mlx5_0 gid idx 3 = 10.99.0.10 v2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmp_dbl(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

#define OOB_PORT 5000
#define TT_RING 256 /* MUST match the target's TT_RING (async ring landing slots; common_defs.h TT_RING) */

static int oob_connect(const char* ip) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(OOB_PORT);
    if (inet_pton(AF_INET, ip, &a.sin_addr) <= 0) {
        fprintf(stderr, "bad ip\n");
        close(fd);
        return -1;
    }
    if (connect(fd, (struct sockaddr*)&a, sizeof(a)) < 0) {
        perror("connect");
        close(fd);
        return -1;
    }
    return fd;
}

static int xchg(int fd, void* snd, void* rcv, size_t n) {
    if (send(fd, snd, n, 0) != (ssize_t)n) {
        perror("send");
        return -1;
    }
    if (recv(fd, rcv, n, MSG_WAITALL) != (ssize_t)n) {
        perror("recv");
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    const char *server = "10.99.0.1", *dev_name = "mlx5_0";
    int gid_index = 3, count = 1000, plen = 1024, burst = 64, lat_iters = 0, c;

    while ((c = getopt(argc, argv, "s:d:g:n:l:b:L:")) != -1) {
        switch (c) {
            case 's': server = optarg; break;
            case 'd': dev_name = optarg; break;
            case 'g': gid_index = atoi(optarg); break;
            case 'n': count = atoi(optarg); break;
            case 'l': plen = atoi(optarg); break;
            case 'b': burst = atoi(optarg); break;
            case 'L': lat_iters = atoi(optarg); break; /* latency mode: N single-op WRITE_IMM RTTs */
            default:
                fprintf(stderr, "usage: %s -s <sf_ip> -d <dev> -g <gid_idx> -n <count> -l <plen>\n", argv[0]);
                return 1;
        }
    }

    /* Open the RoCE device */
    struct ibv_device** dev_list = ibv_get_device_list(NULL);
    struct ibv_device* dev = NULL;
    for (int i = 0; dev_list && dev_list[i]; i++) {
        if (!strcmp(ibv_get_device_name(dev_list[i]), dev_name)) {
            dev = dev_list[i];
            break;
        }
    }
    if (!dev) {
        fprintf(stderr, "device %s not found\n", dev_name);
        return 1;
    }
    struct ibv_context* ctx = ibv_open_device(dev);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    struct ibv_cq* cq = ibv_create_cq(ctx, burst + 8, NULL, NULL, 0);

    /* Source buffer (the payload that lands at the target + gets re-headed to the BH) */
    char* buf = aligned_alloc(4096, plen);
    for (int i = 0; i < plen; i++) {
        buf[i] = (char)i;
    }
    struct ibv_mr* mr =
        ibv_reg_mr(pd, buf, plen, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr) {
        perror("ibv_reg_mr");
        return 1;
    }

    /* RC QP */
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = burst + 8;
    qia.cap.max_recv_wr = 1;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    struct ibv_qp* qp = ibv_create_qp(pd, &qia);
    if (!qp) {
        perror("ibv_create_qp");
        return 1;
    }

    /* Local GID */
    union ibv_gid lgid;
    if (ibv_query_gid(ctx, 1, gid_index, &lgid)) {
        fprintf(stderr, "query_gid failed\n");
        return 1;
    }

    /* OOB exchange (mirror the target: send-then-recv per field) */
    int fd = oob_connect(server);
    if (fd < 0) {
        return 1;
    }
    uint64_t l_addr = (uint64_t)(uintptr_t)buf, r_addr = 0;
    uint32_t l_rkey = mr->rkey, r_rkey = 0;
    uint32_t l_qpn = qp->qp_num, r_qpn = 0;
    union ibv_gid rgid;
    if (xchg(fd, &l_addr, &r_addr, sizeof(uint64_t))) {
        return 1;
    }
    if (xchg(fd, &l_rkey, &r_rkey, sizeof(uint32_t))) {
        return 1;
    }
    if (xchg(fd, &l_qpn, &r_qpn, sizeof(uint32_t))) {
        return 1;
    }
    if (xchg(fd, lgid.raw, rgid.raw, 16)) {
        return 1;
    }
    close(fd);
    printf("OOB done: remote addr=0x%lx rkey=0x%x qpn=%u\n", r_addr, r_rkey, r_qpn);

    /* RST -> INIT */
    struct ibv_qp_attr attr = {0};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        perror("modify INIT");
        return 1;
    }

    /* INIT -> RTR (RoCEv2, path_mtu 1K, rq_psn 0 == target sq_psn 0) */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = r_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    memcpy(attr.ah_attr.grh.dgid.raw, rgid.raw, 16);
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(
            qp,
            &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER)) {
        perror("modify RTR");
        return 1;
    }

    /* RTR -> RTS (sq_psn 0 == target rq_psn 0, ack_timeout 14, retry 7) */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(
            qp,
            &attr,
            IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_MAX_QP_RD_ATOMIC)) {
        perror("modify RTS");
        return 1;
    }
    struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)buf, .length = plen, .lkey = mr->lkey};

    /* Latency mode: one signaled WRITE_IMM at a time, measure post->completion RTT (host<->SF RoCE
     * round-trip incl. HW terminate + ACK; the re-head->p0 happens after and adds ~1-2us on-chip). */
    if (lat_iters > 0) {
        double* lat = calloc(lat_iters, sizeof(double));
        for (int i = 0; i < lat_iters; i++) {
            struct ibv_send_wr wr = {0}, *bad = NULL;
            struct ibv_wc wc;
            wr.wr_id = i;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.imm_data = htonl((uint32_t)(i + 1));
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = r_addr;
            wr.wr.rdma.rkey = r_rkey;
            double t0 = now_us();
            if (ibv_post_send(qp, &wr, &bad)) {
                perror("post_send");
                return 1;
            }
            while (ibv_poll_cq(cq, 1, &wc) == 0);
            lat[i] = now_us() - t0;
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC err %s\n", ibv_wc_status_str(wc.status));
                return 1;
            }
        }
        qsort(lat, lat_iters, sizeof(double), cmp_dbl);
        double sum = 0;
        for (int i = 0; i < lat_iters; i++) {
            sum += lat[i];
        }
        printf(
            "LATENCY (%d x WRITE_IMM %dB, post->completion RTT): min=%.2f p50=%.2f p99=%.2f max=%.2f mean=%.2f us\n",
            lat_iters,
            plen,
            lat[0],
            lat[lat_iters / 2],
            lat[(int)(lat_iters * 0.99)],
            lat[lat_iters - 1],
            sum / lat_iters);
        free(lat);
        goto cleanup;
    }

    printf("QP connected (RTS). Posting %d WRITE_IMM of %dB to 0x%lx...\n", count, plen, r_addr);

    /* Post count WRITE_WITH_IMM; signal last-of-burst + poll to pace. All target the SAME remote addr
     * (roff 0) -- the target re-heads each landed frame; the IMM (seq) drives Thread A per frame. */
    double bw_t0 = now_us();
    int posted = 0, completed = 0;
    while (posted < count) {
        int n = count - posted;
        if (n > burst) {
            n = burst;
        }
        for (int j = 0; j < n; j++) {
            struct ibv_send_wr wr = {0}, *bad = NULL;
            wr.wr_id = posted + j;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.imm_data = htonl((uint32_t)(posted + j + 1));
            wr.send_flags = (j == n - 1) ? IBV_SEND_SIGNALED : 0;
            /* async ring: frame i -> landing slot (i % TT_RING); the re-head gathers the matching slot. */
            wr.wr.rdma.remote_addr = r_addr + (uint64_t)((posted + j) % TT_RING) * plen;
            wr.wr.rdma.rkey = r_rkey;
            if (ibv_post_send(qp, &wr, &bad)) {
                perror("post_send");
                return 1;
            }
        }
        /* wait for the burst's signaled completion */
        struct ibv_wc wc;
        int got = 0;
        while (got < 1) {
            int ne = ibv_poll_cq(cq, 1, &wc);
            if (ne < 0) {
                fprintf(stderr, "poll_cq err\n");
                return 1;
            }
            if (ne == 0) {
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC error: %s (%d) wr_id=%lu\n", ibv_wc_status_str(wc.status), wc.status, wc.wr_id);
                return 1;
            }
            got += ne;
        }
        completed += n;
        posted += n;
    }
    double bw_secs = (now_us() - bw_t0) / 1e6;
    double frame_bytes = plen; /* payload bytes acked (RoCE-level; wire frame = 46B TT hdr + plen after re-head) */
    printf(
        "DONE: posted=%d completed=%d WRITE_IMM ok in %.4f s -> %.2f Gbps, %.3f Mpps (payload %dB; "
        "re-head adds 46B/frame on p0)\n",
        posted,
        completed,
        bw_secs,
        (count * frame_bytes * 8.0) / (bw_secs * 1e9),
        (count / bw_secs) / 1e6,
        plen);

cleanup:
    ibv_destroy_qp(qp);
    ibv_dereg_mr(mr);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);
    return 0;
}
