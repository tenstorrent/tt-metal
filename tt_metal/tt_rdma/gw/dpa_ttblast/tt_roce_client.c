// SPDX-License-Identifier: Apache-2.0
//
// A3.3b-3 matching RoCEv2 requester for the merged DPA gateway's RDMA-CM responder.
// Runs on the x86 HOST. Connects to the SF GID via RDMA CM, reads the landing MR {rkey,addr,len} the
// responder advertises in the ESTABLISHED private_data, then issues <count> RDMA_WRITE_WITH_IMM to it.
// Each WRITE_IMM produces a responder recv completion -> the responder bumps the doorbell -> the DPA re-heads
// the landed payload to the Blackhole. (This is our own minimal peer; generic apps like ib_write_bw connect
// the same way via RDMA CM but layer their own MR-exchange protocol on top -- see gw/A3_rehead_plan.md.)
//
// usage: tt_roce_client <server_ip> <port> <count> <plen>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <rdma/rdma_cma.h>

struct pdata {
    uint32_t rkey;
    uint64_t addr;
    uint32_t len;
} __attribute__((__packed__));

static struct rdma_cm_event* wait_ev(struct rdma_event_channel* ec, enum rdma_cm_event_type want) {
    struct rdma_cm_event* ev;
    if (rdma_get_cm_event(ec, &ev)) {
        return NULL;
    }
    if (ev->event != want) {
        fprintf(stderr, "cm: expected %s got %s\n", rdma_event_str(want), rdma_event_str(ev->event));
        return NULL;
    }
    return ev;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <server_ip> <port> <count> <plen>\n", argv[0]);
        return 1;
    }
    const char* ip = argv[1];
    uint16_t port = (uint16_t)strtoul(argv[2], NULL, 0);
    uint64_t count = strtoull(argv[3], NULL, 0);
    uint32_t plen = (uint32_t)strtoul(argv[4], NULL, 0);
    struct rdma_event_channel* ec;
    struct rdma_cm_id* id;
    struct rdma_cm_event* ev;
    struct ibv_qp_init_attr qia = {0};
    struct rdma_conn_param cp = {0};
    struct sockaddr_in dst = {0};
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_mr* mr;
    struct pdata adv;
    char* buf;
    uint64_t i, done = 0;
    const uint32_t BATCH = 64;

    dst.sin_family = AF_INET;
    dst.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &dst.sin_addr) != 1) {
        fprintf(stderr, "bad ip %s\n", ip);
        return 1;
    }
    ec = rdma_create_event_channel();
    if (!ec || rdma_create_id(ec, &id, NULL, RDMA_PS_TCP)) {
        fprintf(stderr, "cm: channel/id\n");
        return 1;
    }
    if (rdma_resolve_addr(id, NULL, (struct sockaddr*)&dst, 2000) || !(ev = wait_ev(ec, RDMA_CM_EVENT_ADDR_RESOLVED))) {
        return 1;
    }
    rdma_ack_cm_event(ev);
    if (rdma_resolve_route(id, 2000) || !(ev = wait_ev(ec, RDMA_CM_EVENT_ROUTE_RESOLVED))) {
        return 1;
    }
    rdma_ack_cm_event(ev);

    pd = ibv_alloc_pd(id->verbs);
    cq = ibv_create_cq(id->verbs, 256, NULL, NULL, 0);
    buf = malloc(plen ? plen : 256);
    memcpy(buf, "A33b-ROCE-CM-REQ-", 17);
    mr = ibv_reg_mr(pd, buf, plen ? plen : 256, IBV_ACCESS_LOCAL_WRITE);
    if (!pd || !cq || !buf || !mr) {
        fprintf(stderr, "client: pd/cq/buf/mr\n");
        return 1;
    }
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 4;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    if (rdma_create_qp(id, pd, &qia)) {
        fprintf(stderr, "client: create_qp\n");
        return 1;
    }
    cp.initiator_depth = 4;
    cp.responder_resources = 4;
    cp.retry_count = 7;
    cp.rnr_retry_count = 7;
    if (rdma_connect(id, &cp) || !(ev = wait_ev(ec, RDMA_CM_EVENT_ESTABLISHED))) {
        return 1;
    }
    if (!ev->param.conn.private_data || ev->param.conn.private_data_len < sizeof(adv)) {
        fprintf(stderr, "client: no landing MR advertised in private_data\n");
        return 1;
    }
    memcpy(&adv, ev->param.conn.private_data, sizeof(adv));
    rdma_ack_cm_event(ev);
    printf(
        "client: connected; landing rkey=0x%x addr=0x%lx len=%u -> %lu WRITE_IMM x %uB\n",
        adv.rkey,
        (unsigned long)adv.addr,
        adv.len,
        (unsigned long)count,
        plen);

    for (i = 0; i < count; i++) {
        struct ibv_sge sge = {0};
        struct ibv_send_wr wr = {0}, *bad;
        int signaled = ((i % BATCH) == (BATCH - 1)) || (i == count - 1);

        sge.addr = (uint64_t)(uintptr_t)buf;
        sge.length = plen < adv.len ? plen : adv.len;
        sge.lkey = mr->lkey;
        wr.wr_id = i;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.imm_data = htonl((uint32_t)i);
        wr.wr.rdma.remote_addr = adv.addr; /* one landing slot (b-3 basic); ring + roff later */
        wr.wr.rdma.rkey = adv.rkey;
        wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
        if (ibv_post_send(id->qp, &wr, &bad)) {
            fprintf(stderr, "client: post_send %lu failed\n", (unsigned long)i);
            return 1;
        }
        if (signaled) {
            struct ibv_wc wc;
            int n;
            do {
                n = ibv_poll_cq(cq, 1, &wc);
            } while (n == 0);
            if (n < 0 || wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "client: send wc status %s\n", ibv_wc_status_str(wc.status));
                return 1;
            }
            done = i + 1;
        }
    }
    printf("client: done, %lu WRITE_IMM acked\n", (unsigned long)done);
    rdma_disconnect(id);
    return 0;
}
