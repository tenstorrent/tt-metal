/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TT-RDMA gateway BRIDGE (Architecture B, Phase B1) — Tenstorrent, 2026.
 *
 * A modification of the NVIDIA DOCA sample `rdma_write_immediate_responder`. The stock sample terminates a
 * RoCEv2 RC QP, receives one RDMA WRITE_WITH_IMM, prints the written string, and stops. This bridge turns it
 * into the inbound leg of the DPU gateway: on every WRITE_IMM completion it FORWARDS the payload landed in
 * the responder mmap to the Blackhole over the TT rail as a native TT-RDMA-v1 WRITE frame (ethertype 0x1AF6
 * + 32B tt_rdma_hdr), then re-posts the receive task so it runs continuously.
 *
 * So: ConnectX HW terminates full-spec RoCEv2 (PSN/ICRC/ACK, all in silicon); this bridge does only the lean
 * TT-RDMA re-origination; the BH drainer pool lands it (unchanged, validated 200G lossless). B1 uses a raw
 * AF_PACKET socket on the uplink (p0) for egress — simple + proven from tt_rdma_bf3_send; B3 swaps this for
 * the DOCA Eth-Tx datapath (doca_ttblast) / DPA for line rate.
 *
 * Config via env (keeps the stock argp untouched):
 *   TTBRIDGE_IFACE   egress netdev toward the BH   (default p0)
 *   TTBRIDGE_DMAC    BH RXQ2 dest MAC              (default 02:00:00:00:00:02)
 *   TTBRIDGE_RKEY    TT-RDMA rkey (-> MR slot)     (default 0x00CAFE42)
 *   TTBRIDGE_PLEN    bytes to forward per WRITE    (default 256)
 *   TTBRIDGE_MAX     stop after N forwards         (default 1)
 * Build/run: deploy_rdma_bridge.sh (vendors this + builds against the stock DOCA rdma sample sources).
 */

#include <arpa/inet.h>
#include <errno.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <doca_ctx.h>
#include <doca_error.h>
#include <doca_log.h>

/* B3 HW-TX egress (DOCA Eth-TX on the uplink) -- the same datapath doca_ttblast proved at ~143-198G. */
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_cpu_data_path.h>
#include <doca_mmap.h>
#include <doca_pe.h>

#include "rdma_common.h"
#include "eth_common.h"
#include "eth_flow_common.h"

#define MAX_BUFF_SIZE (4096) /* responder mmap size — big enough for a jumbo forward */

DOCA_LOG_REGISTER(TT_RDMA_BRIDGE::SAMPLE);

/* ---- TT-RDMA egress: shared config ---- */
static uint32_t g_rkey = 0x00CAFE42u;
static unsigned g_plen = 256u;
static unsigned long g_max_fwd = 1u;
static unsigned long g_fwded = 0u;
static uint32_t g_seq = 0u;
/* TTBRIDGE_BURST: re-emit the same landed payload N times per WRITE_IMM (default 1 = real 1:1 bridging).
 * A validation-only knob: one RoCE WRITE then produces a DENSE gateway-originated TT-RDMA stream so the
 * BH drainer pool's dense-stream ring accounting can validate + land gateway frames end-to-end. */
static unsigned long g_burst = 1u;
static uint8_t g_dmac[6]; /* BH RXQ2 dest MAC (shared by both egress paths) */

/* Egress backend: 0 = raw AF_PACKET (B1, ~13G ceiling), 1 = DOCA Eth-TX HW-TX (B3, line-rate). */
static int g_egress_doca = 1;

/* ---- B1 raw AF_PACKET egress state ---- */
static int g_fd = -1;
static struct sockaddr_ll g_sa;
static unsigned char g_frame[4200];

/* ---- B3 DOCA Eth-TX egress state ----
 * A DOCA Eth-TX (REGULAR, CPU data path) on the uplink device, driven by its OWN progress engine (the bridge
 * main loop progresses both PEs). A pool of persistent doca_bufs over a TX mmap holds pre-built frames; each
 * forward pops a free slot, patches the dynamic TT-RDMA header + copies the RoCE payload, and submits one
 * doca_eth_txq_task_send. The send completion recycles the slot. Frame size is fixed (14 + 32 + g_plen), so
 * the buf data-length never changes. Single-threaded (submit + completion + backpressure all on the main
 * loop thread) -> no locking. Modeled on the proven doca_ttblast (tt-rdma-gateway-sender.md). */
#define TT_TX_POOL 512      /* TX buf pool depth = max frames in flight */
#define TT_TX_MAX_BURST 256 /* doca_eth_txq max burst */
struct tt_doca_tx {
    struct eth_core_resources core;        /* dev + mmap + buf_inv + pe (allocate_eth_core_resources) */
    struct eth_flow_common_resources flow; /* DOCA Flow port on the uplink (required for HW TX) */
    struct doca_eth_txq* txq;
    struct doca_buf* bufs[TT_TX_POOL];
    uint32_t free_stack[TT_TX_POOL]; /* indices of free slots */
    uint32_t free_top;               /* number of free slots on the stack */
    uint32_t inflight;               /* submitted-but-not-completed sends */
    uint32_t frame_size;             /* 14 (L2) + 32 (TT hdr) + g_plen */
    uint8_t smac[6];                 /* uplink device MAC */
    int started;                     /* ctx started (for cleanup gating) */
};
static struct tt_doca_tx g_tx;
#define TT_TX_SLEEP_NANOS (10 * 1000)

static void put_u16(unsigned char* p, uint16_t v) {
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
}
static void put_u32(unsigned char* p, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        p[i] = (v >> (8 * i)) & 0xff;
    }
}
static void put_u64(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (v >> (8 * i)) & 0xff;
    }
}

/* CRC-32 (reflected 0xEDB88320) — matches tt_rdma_crc32 / the BH ETH-CTRL ROCE_ICRC poly; the RX kernel
 * drops frames whose header_cksum mismatches. */
static uint32_t tt_crc32(const unsigned char* p, unsigned n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (unsigned i = 0; i < n; i++) {
        crc ^= p[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

/* Build the 32B TT-RDMA-v1 WRITE header at `h` (identical wire layout to doca_ttblast + tt_rdma_bf3_send). */
static void tt_build_hdr(unsigned char* h, unsigned plen, uint32_t roff, uint32_t imm) {
    h[0] = 0x10; /* WRITE opcode */
    h[1] = 0x01; /* version_flags: ver=1, no IMM */
    put_u16(h + 2, 0);
    put_u32(h + 4, plen);             /* length */
    put_u32(h + 8, ++g_seq);          /* seq (BH does not order-check the pool path) */
    put_u32(h + 12, g_rkey);          /* rkey -> MR slot */
    put_u64(h + 16, roff);            /* remote offset within the MR */
    put_u32(h + 24, imm);             /* carry the RoCE imm through for traceability */
    put_u32(h + 28, tt_crc32(h, 28)); /* header_cksum over [0..27] */
}

/* ============================ B1 raw AF_PACKET egress ============================ */

/* Open the raw egress socket toward the BH and pre-build the L2 header. Returns 0 on success. */
static int egress_raw_init(const char* iface) {
    g_fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (g_fd < 0) {
        DOCA_LOG_ERR("egress raw socket: %s", strerror(errno));
        return -1;
    }
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(g_fd, SIOCGIFINDEX, &ifr) < 0) {
        DOCA_LOG_ERR("SIOCGIFINDEX %s: %s", iface, strerror(errno));
        return -1;
    }
    int ifindex = ifr.ifr_ifindex;
    unsigned char smac[6] = {0};
    if (ioctl(g_fd, SIOCGIFHWADDR, &ifr) == 0) {
        memcpy(smac, ifr.ifr_hwaddr.sa_data, 6);
    }

    memset(&g_sa, 0, sizeof(g_sa));
    g_sa.sll_family = AF_PACKET;
    g_sa.sll_ifindex = ifindex;
    g_sa.sll_halen = 6;
    memset(g_frame, 0, sizeof(g_frame));
    for (int i = 0; i < 6; i++) {
        g_frame[i] = g_dmac[i];
        g_sa.sll_addr[i] = g_dmac[i];
    }
    memcpy(g_frame + 6, smac, 6);
    g_frame[12] = 0x1a;
    g_frame[13] = 0xf6;
    DOCA_LOG_INFO(
        "TT-RDMA egress ready (RAW AF_PACKET): iface=%s ifindex=%d rkey=0x%08x plen=%u max=%lu burst=%lu",
        iface,
        ifindex,
        g_rkey,
        g_plen,
        g_max_fwd,
        g_burst);
    return 0;
}

static void egress_raw_send(const unsigned char* payload, unsigned plen, uint32_t roff, uint32_t imm) {
    if (plen > 4080u) {
        plen = 4080u;
    }
    tt_build_hdr(g_frame + 14, plen, roff, imm);
    memcpy(g_frame + 14 + 32, payload, plen);
    unsigned flen = 14u + 32u + plen;
    if (sendto(g_fd, g_frame, flen, 0, (struct sockaddr*)&g_sa, sizeof(g_sa)) < 0) {
        DOCA_LOG_ERR("egress sendto: %s", strerror(errno));
    }
}

/* ============================ B3 DOCA Eth-TX HW-TX egress ============================ */

/* Device must support the REGULAR CPU TXQ + our burst size. */
static doca_error_t tt_tx_check_device(struct doca_devinfo* devinfo) {
    doca_error_t status;
    uint32_t max_burst;

    status = doca_eth_txq_cap_get_max_burst_size(devinfo, 1 /*MAX_LIST_LENGTH*/, 0, &max_burst);
    if (status != DOCA_SUCCESS) {
        return status;
    }
    if (max_burst < TT_TX_MAX_BURST) {
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    return doca_eth_txq_cap_is_type_supported(devinfo, DOCA_ETH_TXQ_TYPE_REGULAR, DOCA_ETH_TXQ_DATA_PATH_TYPE_CPU);
}

/* Send completion: recycle the slot; both success + error callbacks land here. */
static void tt_tx_send_cb(
    struct doca_eth_txq_task_send* task_send, union doca_data task_user_data, union doca_data ctx_user_data) {
    struct tt_doca_tx* tx = (struct tt_doca_tx*)ctx_user_data.ptr;
    uint32_t slot = (uint32_t)task_user_data.u64;

    tx->free_stack[tx->free_top++] = slot;
    tx->inflight--;
    doca_task_free(doca_eth_txq_task_send_as_doca_task(task_send));
}

static void tt_tx_err_cb(
    struct doca_eth_txq_task_send* task_send, union doca_data task_user_data, union doca_data ctx_user_data) {
    doca_error_t st = doca_task_get_status(doca_eth_txq_task_send_as_doca_task(task_send));
    DOCA_LOG_ERR("HW-TX send task failed: %s", doca_error_get_name(st));
    tt_tx_send_cb(task_send, task_user_data, ctx_user_data); /* still recycle the slot */
}

/* Bring up the DOCA Eth-TX on `txdev`: core resources (dev+mmap+buf_inv+pe), REGULAR txq, DOCA Flow port,
 * and a pool of pre-built frame bufs. Returns 0 on success. */
static int egress_doca_init(const char* txdev) {
    doca_error_t st;
    union doca_data ud;

    g_tx.frame_size = 14u + 32u + g_plen;
    struct eth_core_config cfg = {
        .mmap_size = g_tx.frame_size * TT_TX_POOL,
        .inventory_num_elements = TT_TX_POOL,
        .check_device = tt_tx_check_device,
        .ibdev_name = txdev,
    };

    st = allocate_eth_core_resources(&cfg, &g_tx.core);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: allocate_eth_core_resources(%s): %s", txdev, doca_error_get_name(st));
        return -1;
    }
    st = doca_devinfo_get_mac_addr(doca_dev_as_devinfo(g_tx.core.core_objs.dev), g_tx.smac, DOCA_DEVINFO_MAC_ADDR_SIZE);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: get MAC: %s", doca_error_get_name(st));
        return -1;
    }

    st = doca_eth_txq_create(g_tx.core.core_objs.dev, TT_TX_MAX_BURST, &g_tx.txq);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: eth_txq_create: %s", doca_error_get_name(st));
        return -1;
    }
    st = doca_eth_txq_set_type(g_tx.txq, DOCA_ETH_TXQ_TYPE_REGULAR);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: set_type: %s", doca_error_get_name(st));
        return -1;
    }
    st = doca_eth_txq_task_send_set_conf(g_tx.txq, tt_tx_send_cb, tt_tx_err_cb, TT_TX_POOL);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: task_send_set_conf: %s", doca_error_get_name(st));
        return -1;
    }
    g_tx.core.core_objs.ctx = doca_eth_txq_as_doca_ctx(g_tx.txq);
    st = doca_pe_connect_ctx(g_tx.core.core_objs.pe, g_tx.core.core_objs.ctx);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: pe_connect_ctx: %s", doca_error_get_name(st));
        return -1;
    }
    ud.ptr = &g_tx;
    st = doca_ctx_set_user_data(g_tx.core.core_objs.ctx, ud);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: set_user_data: %s", doca_error_get_name(st));
        return -1;
    }
    st = doca_ctx_start(g_tx.core.core_objs.ctx);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: ctx_start: %s", doca_error_get_name(st));
        return -1;
    }
    g_tx.started = 1;
    st = doca_eth_txq_apply_queue_id(g_tx.txq, 0);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: apply_queue_id: %s", doca_error_get_name(st));
        return -1;
    }

    /* DOCA Flow port on the uplink -- required for HW TX in switchdev (same as doca_ttblast). */
    st = eth_flow_common_init_flow();
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: flow init: %s", doca_error_get_name(st));
        return -1;
    }
    st = eth_flow_common_create_flow_port(g_tx.core.core_objs.dev, 0, &g_tx.flow);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: create_flow_port: %s", doca_error_get_name(st));
        return -1;
    }

    /* Build the TX buf pool: one persistent doca_buf per fixed-size slot, L2 + static TT hdr pre-filled. */
    for (uint32_t i = 0; i < TT_TX_POOL; i++) {
        uint8_t* base = (uint8_t*)g_tx.core.mmap_addr + (size_t)i * g_tx.frame_size;
        st = doca_buf_inventory_buf_get_by_data(
            g_tx.core.core_objs.buf_inv, g_tx.core.core_objs.src_mmap, base, g_tx.frame_size, &g_tx.bufs[i]);
        if (st != DOCA_SUCCESS) {
            DOCA_LOG_ERR("HW-TX: buf_get[%u]: %s", i, doca_error_get_name(st));
            return -1;
        }
        struct ether_hdr* eh = (struct ether_hdr*)base;
        memcpy(eh->dst_addr, g_dmac, 6);
        memcpy(eh->src_addr, g_tx.smac, 6);
        eh->ether_type = htobe16(0x1AF6);
        g_tx.free_stack[i] = i;
    }
    g_tx.free_top = TT_TX_POOL;
    DOCA_LOG_INFO(
        "TT-RDMA egress ready (DOCA HW-TX): dev=%s pool=%u frame=%uB rkey=0x%08x plen=%u max=%lu burst=%lu",
        txdev,
        TT_TX_POOL,
        g_tx.frame_size,
        g_rkey,
        g_plen,
        g_max_fwd,
        g_burst);
    return 0;
}

static void egress_doca_send(const unsigned char* payload, unsigned plen, uint32_t roff, uint32_t imm) {
    doca_error_t st;
    struct doca_eth_txq_task_send* task;
    union doca_data ud;
    uint32_t slot;
    uint8_t* base;

    if (plen > g_plen) {
        plen = g_plen; /* pool frame size is fixed at 14+32+g_plen */
    }

    /* Backpressure: if every slot is in flight, progress the TX PE until one frees. This stalls the RoCE
     * receive callback -> the RoCE flow naturally back-pressures. */
    while (g_tx.free_top == 0) {
        (void)doca_pe_progress(g_tx.core.core_objs.pe);
    }

    slot = g_tx.free_stack[--g_tx.free_top];
    base = (uint8_t*)g_tx.core.mmap_addr + (size_t)slot * g_tx.frame_size;
    tt_build_hdr(base + 14, plen, roff, imm); /* L2 + static hdr already built at init */
    memcpy(base + 14 + 32, payload, plen);

    ud.u64 = slot;
    st = doca_eth_txq_task_send_allocate_init(g_tx.txq, g_tx.bufs[slot], ud, &task);
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: allocate_init: %s", doca_error_get_name(st));
        g_tx.free_stack[g_tx.free_top++] = slot;
        return;
    }
    st = doca_task_submit(doca_eth_txq_task_send_as_doca_task(task));
    if (st != DOCA_SUCCESS) {
        DOCA_LOG_ERR("HW-TX: task_submit: %s", doca_error_get_name(st));
        doca_task_free(doca_eth_txq_task_send_as_doca_task(task));
        g_tx.free_stack[g_tx.free_top++] = slot;
        return;
    }
    g_tx.inflight++;
}

/* ============================ egress dispatch ============================ */

static int egress_init(void) {
    const char* iface = getenv("TTBRIDGE_IFACE");
    const char* txdev = getenv("TTBRIDGE_TXDEV");
    const char* dmac_s = getenv("TTBRIDGE_DMAC");
    const char* rkey_s = getenv("TTBRIDGE_RKEY");
    const char* plen_s = getenv("TTBRIDGE_PLEN");
    const char* max_s = getenv("TTBRIDGE_MAX");
    const char* burst_s = getenv("TTBRIDGE_BURST");
    const char* egr_s = getenv("TTBRIDGE_EGRESS");

    if (!iface) {
        iface = "p0";
    }
    if (!txdev) {
        txdev = "mlx5_0";
    }
    if (!dmac_s) {
        dmac_s = "02:00:00:00:00:02";
    }
    if (rkey_s) {
        g_rkey = (uint32_t)strtoul(rkey_s, NULL, 0);
    }
    if (plen_s) {
        g_plen = (unsigned)strtoul(plen_s, NULL, 0);
    }
    if (max_s) {
        g_max_fwd = strtoul(max_s, NULL, 0);
    }
    if (burst_s) {
        g_burst = strtoul(burst_s, NULL, 0);
        if (g_burst < 1u) {
            g_burst = 1u;
        }
    }
    if (g_plen > 4080u) {
        g_plen = 4080u;
    }
    /* Default = DOCA HW-TX (B3); TTBRIDGE_EGRESS=raw selects the B1 AF_PACKET fallback. */
    g_egress_doca = !(egr_s && strcmp(egr_s, "raw") == 0);

    unsigned dm[6] = {0};
    if (sscanf(dmac_s, "%x:%x:%x:%x:%x:%x", &dm[0], &dm[1], &dm[2], &dm[3], &dm[4], &dm[5]) != 6) {
        DOCA_LOG_ERR("TTBRIDGE_DMAC parse failed: %s", dmac_s);
        return -1;
    }
    for (int i = 0; i < 6; i++) {
        g_dmac[i] = (uint8_t)dm[i];
    }

    return g_egress_doca ? egress_doca_init(txdev) : egress_raw_init(iface);
}

/* Build + send one TT-RDMA WRITE frame from `payload` (plen bytes) at remote offset `roff`. */
static void egress_send_ttrdma(const unsigned char* payload, unsigned plen, uint32_t roff, uint32_t imm) {
    if (g_egress_doca) {
        egress_doca_send(payload, plen, roff, imm);
    } else {
        egress_raw_send(payload, plen, roff, imm);
    }
}

/* Progress the egress backend (DOCA TX completions). No-op for the raw path. Called each main-loop tick. */
static void egress_progress(void) {
    if (g_egress_doca && g_tx.core.core_objs.pe != NULL) {
        (void)doca_pe_progress(g_tx.core.core_objs.pe);
    }
}

/* Drain any in-flight HW-TX sends before shutdown. */
static void egress_drain(void) {
    struct timespec ts = {.tv_sec = 0, .tv_nsec = TT_TX_SLEEP_NANOS};
    if (!g_egress_doca) {
        return;
    }
    while (g_tx.inflight != 0) {
        if (doca_pe_progress(g_tx.core.core_objs.pe) == 0) {
            nanosleep(&ts, &ts);
        }
    }
}

/* Tear down the egress backend. */
static void egress_cleanup(void) {
    if (!g_egress_doca) {
        if (g_fd >= 0) {
            close(g_fd);
        }
        g_fd = -1;
        return;
    }
    egress_drain();
    for (uint32_t i = 0; i < TT_TX_POOL; i++) {
        if (g_tx.bufs[i] != NULL) {
            (void)doca_buf_dec_refcount(g_tx.bufs[i], NULL);
        }
    }
    if (g_tx.flow.df_port != NULL) {
        (void)eth_flow_common_destroy_flow_port(&g_tx.flow);
    }
    eth_flow_common_cleanup_flow();
    if (g_tx.started) {
        (void)doca_ctx_stop(g_tx.core.core_objs.ctx);
        g_tx.started = 0;
    }
    if (g_tx.txq != NULL) {
        (void)doca_eth_txq_destroy(g_tx.txq);
        g_tx.txq = NULL;
    }
    if (g_tx.core.core_objs.dev != NULL) {
        (void)destroy_eth_core_resources(&g_tx.core);
    }
}

/*
 * Write the connection details and the mmap details for the requester to read,
 * and read the connection details of the requester.
 */
static doca_error_t write_read_connection(struct rdma_config* cfg, struct rdma_resources* resources) {
    doca_error_t result;

    result = write_file(
        cfg->local_connection_desc_path, (char*)resources->rdma_conn_descriptor, resources->rdma_conn_descriptor_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to write the RDMA connection details: %s", doca_error_get_descr(result));
        return result;
    }
    result =
        write_file(cfg->remote_resource_desc_path, (char*)resources->mmap_descriptor, resources->mmap_descriptor_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to write the RDMA mmap details: %s", doca_error_get_descr(result));
        return result;
    }
    DOCA_LOG_INFO(
        "You can now copy %s and %s to the requester", cfg->local_connection_desc_path, cfg->remote_resource_desc_path);
    if (cfg->transport_type == DOCA_RDMA_TRANSPORT_TYPE_DC) {
        return result;
    }
    DOCA_LOG_INFO("Please copy %s from the requester and then press enter", cfg->remote_connection_desc_path);
    wait_for_enter();
    result = read_file(
        cfg->remote_connection_desc_path,
        (char**)&resources->remote_rdma_conn_descriptor,
        &resources->remote_rdma_conn_descriptor_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to read the remote RDMA connection details: %s", doca_error_get_descr(result));
    }
    return result;
}

static doca_error_t rdma_receive_prepare_and_submit_task(struct rdma_resources* resources);

/*
 * RDMA receive task completed callback — the BRIDGE hook. On each RoCEv2 WRITE_WITH_IMM, forward the payload
 * the requester landed in our mmap to the BH as a TT-RDMA WRITE, then re-post to keep bridging.
 */
static void rdma_receive_completed_callback(
    struct doca_rdma_task_receive* rdma_receive_task, union doca_data task_user_data, union doca_data ctx_user_data) {
    struct rdma_resources* resources = (struct rdma_resources*)ctx_user_data.ptr;
    doca_be32_t immediate_data;
    enum doca_rdma_opcode op_code;
    doca_error_t* first_encountered_error = (doca_error_t*)task_user_data.ptr;
    doca_error_t result = DOCA_SUCCESS;

    op_code = doca_rdma_task_receive_get_result_opcode(rdma_receive_task);
    if (op_code != DOCA_RDMA_OPCODE_RECV_WRITE_WITH_IMM) {
        result = DOCA_ERROR_UNEXPECTED;
        DOCA_LOG_ERR("Got incorrect opcode (want RECV_WRITE_WITH_IMM)");
        goto free_task;
    }
    immediate_data = doca_rdma_task_receive_get_result_immediate_data(rdma_receive_task);

    /* Forward: the requester's RDMA WRITE landed the payload directly in resources->mmap_memrange. Re-head it
     * as a TT-RDMA WRITE toward the BH (offset 0 for B1 byte-exact correctness). */
    for (unsigned long b = 0; b < g_burst; b++) {
        egress_send_ttrdma((const unsigned char*)resources->mmap_memrange, g_plen, 0u, (uint32_t)immediate_data);
    }
    g_fwded++;
    if ((g_fwded & 0x3FFu) == 0u || g_fwded <= 4u) {
        DOCA_LOG_INFO(
            "bridged WRITE_IMM #%lu -> TT-RDMA (imm=0x%x, %u bytes)", g_fwded, (unsigned)immediate_data, g_plen);
    }

free_task:
    doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
    DOCA_ERROR_PROPAGATE(*first_encountered_error, result);

    /* Keep bridging: re-post a receive task until we hit the forward cap (then let the ctx drain to idle). */
    if (result == DOCA_SUCCESS && g_fwded < g_max_fwd) {
        (void)rdma_receive_prepare_and_submit_task(resources);
    }

    resources->num_remaining_tasks--;
    if (resources->num_remaining_tasks == 0) {
        if (resources->cfg->use_rdma_cm == true) {
            (void)rdma_cm_disconnect(resources);
        }
        (void)doca_ctx_stop(resources->rdma_ctx);
    }
}

static void rdma_receive_error_callback(
    struct doca_rdma_task_receive* rdma_receive_task, union doca_data task_user_data, union doca_data ctx_user_data) {
    struct rdma_resources* resources = (struct rdma_resources*)ctx_user_data.ptr;
    struct doca_task* task = doca_rdma_task_receive_as_task(rdma_receive_task);
    doca_error_t* first_encountered_error = (doca_error_t*)task_user_data.ptr;
    doca_error_t result = doca_task_get_status(task);

    DOCA_ERROR_PROPAGATE(*first_encountered_error, result);
    DOCA_LOG_ERR("RDMA receive task failed: %s", doca_error_get_descr(result));
    doca_task_free(task);
    resources->num_remaining_tasks--;
    if (resources->num_remaining_tasks == 0) {
        if (resources->cfg->use_rdma_cm == true) {
            (void)rdma_cm_disconnect(resources);
        }
        (void)doca_ctx_stop(resources->rdma_ctx);
    }
}

static doca_error_t rdma_write_immediate_export_and_connect(struct rdma_resources* resources) {
    doca_error_t result;

    if (resources->cfg->use_rdma_cm == true) {
        return rdma_cm_connect(resources);
    }

    result = doca_rdma_export(
        resources->rdma,
        &(resources->rdma_conn_descriptor),
        &(resources->rdma_conn_descriptor_size),
        &(resources->connections[0]));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to export RDMA: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_mmap_export_rdma(
        resources->mmap,
        resources->doca_device,
        (const void**)&(resources->mmap_descriptor),
        &(resources->mmap_descriptor_size));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to export DOCA mmap for RDMA: %s", doca_error_get_descr(result));
        return result;
    }
    result = write_read_connection(resources->cfg, resources);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to write/read connection details: %s", doca_error_get_descr(result));
        return result;
    }
    if (resources->cfg->transport_type == DOCA_RDMA_TRANSPORT_TYPE_DC) {
        return result;
    }
    result = doca_rdma_connect(
        resources->rdma,
        resources->remote_rdma_conn_descriptor,
        resources->remote_rdma_conn_descriptor_size,
        resources->connections[0]);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to connect RDMA: %s", doca_error_get_descr(result));
    }
    return result;
}

static doca_error_t rdma_receive_prepare_and_submit_task(struct rdma_resources* resources) {
    struct doca_rdma_task_receive* rdma_receive_task = NULL;
    union doca_data task_user_data = {0};
    doca_error_t result;

    task_user_data.ptr = &(resources->first_encountered_error);
    /* NULL dst buffer: the receive task only surfaces the IMM completion; the data itself is written by the
     * requester straight into our exported mmap (resources->mmap_memrange). */
    result = doca_rdma_task_receive_allocate_init(resources->rdma, NULL, task_user_data, &rdma_receive_task);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate RDMA receive task: %s", doca_error_get_descr(result));
        return result;
    }
    resources->num_remaining_tasks++;
    result = doca_task_submit(doca_rdma_task_receive_as_task(rdma_receive_task));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to submit RDMA receive task: %s", doca_error_get_descr(result));
        doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
    }
    return result;
}

static void rdma_write_imm_responder_state_change_callback(
    const union doca_data user_data,
    struct doca_ctx* ctx,
    enum doca_ctx_states prev_state,
    enum doca_ctx_states next_state) {
    struct rdma_resources* resources = (struct rdma_resources*)user_data.ptr;
    struct rdma_config* cfg = resources->cfg;
    doca_error_t result = DOCA_SUCCESS;
    (void)prev_state;
    (void)ctx;

    switch (next_state) {
        case DOCA_CTX_STATE_STARTING: DOCA_LOG_INFO("RDMA context entered starting state"); break;
        case DOCA_CTX_STATE_RUNNING:
            DOCA_LOG_INFO("RDMA context is running");
            result = rdma_write_immediate_export_and_connect(resources);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("export_and_connect failed: %s", doca_error_get_descr(result));
                break;
            }
            if (cfg->use_rdma_cm == true) {
                break;
            }
            result = rdma_receive_prepare_and_submit_task(resources);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("prepare_and_submit_task failed: %s", doca_error_get_descr(result));
            }
            break;
        case DOCA_CTX_STATE_STOPPING: DOCA_LOG_INFO("RDMA context stopping; inflight tasks flushed"); break;
        case DOCA_CTX_STATE_IDLE:
            DOCA_LOG_INFO("RDMA context stopped");
            resources->run_pe_progress = false;
            break;
        default: break;
    }
    if (result != DOCA_SUCCESS) {
        DOCA_ERROR_PROPAGATE(resources->first_encountered_error, result);
        (void)doca_ctx_stop(ctx);
    }
}

/*
 * Bridge entry — same name/signature as the stock sample so the stock *_main.c drives it.
 */
doca_error_t rdma_write_immediate_responder(struct rdma_config* cfg) {
    struct rdma_resources resources = {0};
    union doca_data ctx_user_data = {0};
    const uint32_t mmap_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE;
    const uint32_t rdma_permissions = DOCA_ACCESS_FLAG_RDMA_WRITE;
    struct timespec ts = {.tv_sec = 0, .tv_nsec = SLEEP_IN_NANOS};
    doca_error_t result, tmp_result;

    if (egress_init() != 0) {
        return DOCA_ERROR_INITIALIZATION;
    }

    result = allocate_rdma_resources(
        cfg, mmap_permissions, rdma_permissions, doca_rdma_cap_task_receive_is_supported, &resources);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate RDMA Resources: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_rdma_task_receive_set_conf(
        resources.rdma, rdma_receive_completed_callback, rdma_receive_error_callback, NUM_RDMA_TASKS);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to set RDMA receive task conf: %s", doca_error_get_descr(result));
        goto destroy_resources;
    }
    result = doca_ctx_set_state_changed_cb(resources.rdma_ctx, rdma_write_imm_responder_state_change_callback);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to set state change cb: %s", doca_error_get_descr(result));
        goto destroy_resources;
    }
    ctx_user_data.ptr = &(resources);
    result = doca_ctx_set_user_data(resources.rdma_ctx, ctx_user_data);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set context user data: %s", doca_error_get_descr(result));
        goto destroy_resources;
    }
    if (cfg->use_rdma_cm == true) {
        resources.is_requester = false;
        resources.require_remote_mmap = true;
        resources.task_fn = rdma_receive_prepare_and_submit_task;
        result = config_rdma_cm_callback_and_negotiation_task(&resources, true, false);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to config RDMA CM: %s", doca_error_get_descr(result));
            goto destroy_resources;
        }
    }
    result = doca_ctx_start(resources.rdma_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start RDMA context: %s", doca_error_get_descr(result));
        goto destroy_resources;
    }
    while (resources.run_pe_progress) {
        int busy = doca_pe_progress(resources.pe);
        egress_progress(); /* drain HW-TX send completions (recycle bufs) */
        if (busy == 0) {
            nanosleep(&ts, &ts);
        }
    }
    result = resources.first_encountered_error;
    DOCA_LOG_INFO("Bridge done: forwarded %lu WRITE_IMM -> TT-RDMA", g_fwded);

destroy_resources:
    egress_cleanup();
    if (resources.buf_inventory != NULL) {
        tmp_result = doca_buf_inventory_stop(resources.buf_inventory);
        if (tmp_result != DOCA_SUCCESS) {
            DOCA_ERROR_PROPAGATE(result, tmp_result);
        }
        tmp_result = doca_buf_inventory_destroy(resources.buf_inventory);
        if (tmp_result != DOCA_SUCCESS) {
            DOCA_ERROR_PROPAGATE(result, tmp_result);
        }
    }
    tmp_result = destroy_rdma_resources(&resources, cfg);
    if (tmp_result != DOCA_SUCCESS) {
        DOCA_ERROR_PROPAGATE(result, tmp_result);
    }
    return result;
}
