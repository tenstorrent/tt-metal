// SPDX-License-Identifier: Apache-2.0
//
// BF3-side raw-L2 sender for the TT-RDMA RX path. Modes:
//
//   Legacy PROBE (opcode arg omitted / 0): 64B frame, payload magic 0xB1F5C0DE + "TT-RDMA-RX-PROBE".
//   TT-RDMA frame (opcode > 0): [L2][32B tt_rdma_hdr][payload] for the on-core dispatch kernel.
//   BLAST (threads>1 or batch>1): N pthreads each sendmmsg()-ing `batch` frames/syscall -> saturate
//     the wire to find the BH RX ceiling (the single-sendto path is syscall-bound at ~9 Gbps).
//
// Single-purpose (fixed-shape frames) so it is safe to allowlist for passwordless sudo.
//
//   sudo tt_rdma_bf3_send <iface> [count] [dst_mac] [ethertype] [opcode] [payload_len] [rkey] [roff]
//                         [threads] [batch] [badcrc]
//   e.g. blast WRITE: sudo tt_rdma_bf3_send enp193s0f0np0 40000000 02:00:00:00:00:02 0x1af6 0x10 4080
//                     0x00CAFE42 0 8 64
#define _GNU_SOURCE
#include <errno.h>
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

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

// CRC-32 (poly 0x04C11DB7, reflected 0xEDB88320), matching tt_rdma_crc32 in tt_rdma_hdr_build.h
// (the chip/gateway oracle + the ETH-CTRL ROCE_ICRC hardware polynomial).
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

// Per-thread blast state: each thread owns a socket and sendmmsg()es `batch` copies of `frame`.
struct blast_arg {
    int ifindex;
    unsigned char frame[4200];
    unsigned frame_len;
    long count;  // in: target for this thread; out: actually sent
    int batch;
};

static void* blast_thread(void* a) {
    struct blast_arg* t = (struct blast_arg*)a;
    int fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (fd < 0) {
        t->count = 0;
        return NULL;
    }
    struct sockaddr_ll sa;
    memset(&sa, 0, sizeof(sa));
    sa.sll_family = AF_PACKET;
    sa.sll_ifindex = t->ifindex;
    sa.sll_halen = 6;
    memcpy(sa.sll_addr, t->frame, 6);

    const int B = t->batch;
    struct iovec* iov = (struct iovec*)calloc(B, sizeof(struct iovec));
    struct mmsghdr* msgs = (struct mmsghdr*)calloc(B, sizeof(struct mmsghdr));
    for (int i = 0; i < B; i++) {
        iov[i].iov_base = t->frame;  // all point at the same (read-only) frame -> B identical sends
        iov[i].iov_len = t->frame_len;
        msgs[i].msg_hdr.msg_iov = &iov[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
        msgs[i].msg_hdr.msg_name = &sa;
        msgs[i].msg_hdr.msg_namelen = sizeof(sa);
    }
    long sent = 0;
    while (sent < t->count) {
        int want = B;
        if (t->count - sent < (long)B) {
            want = (int)(t->count - sent);
        }
        int n = sendmmsg(fd, msgs, want, 0);
        if (n > 0) {
            sent += n;
        } else if (errno == ENOBUFS || errno == EAGAIN || errno == EINTR) {
            continue;  // TX ring full -> retry (this is the wire being the limit, good)
        } else {
            break;
        }
    }
    t->count = sent;
    free(iov);
    free(msgs);
    close(fd);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(
            stderr,
            "usage: %s <iface> [count] [dst_mac] [ethertype] [opcode] [payload_len] [rkey] [roff] "
            "[threads] [batch] [badcrc]\n",
            argv[0]);
        return 2;
    }
    const char* iface = argv[1];
    long count = (argc > 2) ? atol(argv[2]) : 1000;
    const char* dmac_s = (argc > 3) ? argv[3] : "02:00:00:00:00:02";  // unicast -> TT RXQ2 ("other")
    unsigned etht = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 0) : 0x1af6u;
    unsigned opcode = (argc > 5) ? (unsigned)strtoul(argv[5], NULL, 0) : 0u;  // 0 => legacy PROBE frame
    unsigned plen = (argc > 6) ? (unsigned)strtoul(argv[6], NULL, 0) : 32u;
    uint32_t rkey = (argc > 7) ? (uint32_t)strtoul(argv[7], NULL, 0) : 0u;
    uint64_t roff = (argc > 8) ? (uint64_t)strtoull(argv[8], NULL, 0) : 0u;
    int threads = (argc > 9) ? atoi(argv[9]) : 1;
    int batch = (argc > 10) ? atoi(argv[10]) : 1;
    if (threads < 1) {
        threads = 1;
    }
    if (batch < 1) {
        batch = 1;
    }
    // Test hook (arg[11] badcrc=1): corrupt the header CRC so the RX kernel's crc_check must drop the
    // frame. Positional (not env) so it survives the allowlisted `sudo` (which strips the environment).
    const int badcrc = (argc > 11) ? atoi(argv[11]) : 0;
    const uint32_t crc_xor = badcrc ? 0xFFFFFFFFu : 0u;
    // Perf hook (arg[12] readlat=N>0): with opcode 0x20 (READ_REQ), measure the READ round-trip latency
    // -- send a READ_REQ, wait for the BH's READ_RESP (0x21) on the wire, time it, N times -> percentiles.
    const int readlat = (argc > 12) ? atoi(argv[12]) : 0;
    // arg[13] imm: override imm_data (e.g. CONTROL 0xF0 sub-opcode: 1=REGISTER, 2=DEREGISTER). Sentinel
    // 0xFFFFFFFF => auto (magic for _IMM opcodes, else 0).
    const uint32_t imm_arg = (argc > 13) ? (uint32_t)strtoul(argv[13], NULL, 0) : 0xFFFFFFFFu;
    // arg[14] readresp=N>0: act as a READ RESPONDER for BH-initiator READ tests -- receive N READ_REQ
    // (0x20) frames from the BH, reply each with a READ_RESP (0x21) carrying the echoed tag/seq + a
    // 'READ' payload of the requested length. dst_mac (arg[3]) is the BH RXQ2 MAC (02:...:02).
    const int readresp = (argc > 14) ? atoi(argv[14]) : 0;
    // arg[15] arqsend=N>0: Phase 2.2a software-ARQ SENDER. Emit `count` reliable WRITE (0x10) frames with
    // monotonic seq 1..count (word2), landing at the BH MR, then collect the BH's cumulative ACKs (0x40)
    // and report the highest ack_seq reached. arg[16] gapseq=K>0: skip sending seq K once (simulate loss)
    // so the BH receiver stalls at K-1 and emits dup-ACKs (its ooo counter rises) -- exercises the
    // out-of-order path. With gapseq=0 the in-order run must drive the BH watermark to `count`.
    const int arqsend = (argc > 15) ? atoi(argv[15]) : 0;
    const uint32_t gapseq = (argc > 16) ? (uint32_t)strtoul(argv[16], NULL, 0) : 0u;
    const int arqpace = (argc > 17) ? atoi(argv[17]) : 0;  // inter-frame gap (µs) for the ARQ sender
    // arg[18] arqresp=N>0: Phase 2.2b ARQ RESPONDER for the BH-initiator retransmit test. Receive reliable
    // WRITE (0x10) frames from BH, track the in-order cumulative rx_last, and send a cumulative ACK (0x40)
    // back to BH's RXQ2 (dst_mac) per frame. arg[19] dropreq=K>0: drop the frame with seq K on its FIRST
    // arrival once (simulate loss) so the BH watermark stalls at K-1 and its RTO fires a Go-Back-N resend.
    const int arqresp = (argc > 18) ? atoi(argv[18]) : 0;
    const uint32_t dropreq = (argc > 19) ? (uint32_t)strtoul(argv[19], NULL, 0) : 0u;

    unsigned dm[6] = {0};
    if (sscanf(dmac_s, "%x:%x:%x:%x:%x:%x", &dm[0], &dm[1], &dm[2], &dm[3], &dm[4], &dm[5]) != 6) {
        fprintf(stderr, "bad dst mac '%s'\n", dmac_s);
        return 2;
    }

    int fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (fd < 0) {
        perror("socket (need root / CAP_NET_RAW)");
        return 1;
    }
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
        perror("SIOCGIFINDEX");
        return 1;
    }
    int ifindex = ifr.ifr_ifindex;
    unsigned char smac[6] = {0x02, 0, 0, 0, 0, 0x0b};
    if (ioctl(fd, SIOCGIFHWADDR, &ifr) == 0) {
        memcpy(smac, ifr.ifr_hwaddr.sa_data, 6);
    }

    unsigned char frame[4200];
    memset(frame, 0, sizeof(frame));
    for (int i = 0; i < 6; i++) {
        frame[i] = (unsigned char)dm[i];
    }
    memcpy(frame + 6, smac, 6);
    frame[12] = (unsigned char)(etht >> 8);
    frame[13] = (unsigned char)(etht & 0xff);

    unsigned frame_len;
    if (opcode == 0) {
        frame[14] = 0xde;
        frame[15] = 0xc0;
        frame[16] = 0xf5;
        frame[17] = 0xb1;
        memcpy(frame + 18, "TT-RDMA-RX-PROBE", 16);
        frame_len = 64;
    } else {
        if (plen > 4080u) {
            plen = 4080u;
        }
        unsigned char* h = frame + 14;
        const int is_imm = (opcode == 0x02u || opcode == 0x11u);  // SEND_IMM / WRITE_IMM
        h[0] = (unsigned char)opcode;
        h[1] = (unsigned char)(0x01u | (is_imm ? 0x10u : 0u));  // version_flags: ver=1, bit4=IMM_PRESENT
        put_u16(h + 2, 0);
        put_u32(h + 4, plen);
        put_u32(h + 8, 1);  // seq (fixed; the BH kernel does not order-check)
        put_u32(h + 12, rkey);
        put_u64(h + 16, roff);
        const uint32_t imm_val = (imm_arg != 0xFFFFFFFFu) ? imm_arg : (is_imm ? 0xC0DE1257u : 0u);
        put_u32(h + 24, imm_val);                    // imm_data (override via arg[13]; else magic for _IMM, else 0)
        put_u32(h + 28, tt_crc32(h, 28) ^ crc_xor);  // header_cksum over bytes [0..27] (BADCRC corrupts it)
        if (opcode == 0x20u || opcode == 0x40u || opcode == 0xF0u) {
            // READ_REQ / ACK / CONTROL: HEADER-ONLY on the wire (wire-protocol §1). The length field carries the
            // request size / ack_seq; no payload follows. Pad to 48 B post-L2 (32 hdr + 16 zero pad) so the
            // 62 B frame (+FCS=66) clears the 64 B Ethernet runt minimum — else the MAC pads it and the RX
            // kernel's header-only framing desyncs. Must match TT_RDMA_HDR_ONLY_BYTES (tt_rdma_wire.h).
            memset(frame + 14 + 32, 0, 16);
            frame_len = 14u + 48u;
        } else {
            unsigned char* pay = frame + 14 + 32;
            pay[0] = 'T';
            pay[1] = 'T';
            pay[2] = 'W';
            pay[3] = 'R';
            for (unsigned i = 4; i < plen; i++) {
                pay[i] = (unsigned char)(i & 0xff);
            }
            frame_len = 14u + 32u + plen;
        }
    }

    // ---- READ RESPONDER: reflect BH's READ_REQ (0x20) as a READ_RESP (0x21). For BH-initiator tests. ----
    if (readresp > 0) {
        struct sockaddr_ll sa;
        memset(&sa, 0, sizeof(sa));
        sa.sll_family = AF_PACKET;
        sa.sll_ifindex = ifindex;
        sa.sll_halen = 6;
        memcpy(sa.sll_addr, frame, 6);  // dst = BH RXQ2 (02:...:02)
        if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("bind");
        }
        struct packet_mreq pmr;
        memset(&pmr, 0, sizeof(pmr));
        pmr.mr_ifindex = ifindex;
        pmr.mr_type = PACKET_MR_PROMISC;  // BH's READ_REQ dst MAC is the initiator's, not ours
        setsockopt(fd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pmr, sizeof(pmr));
        struct timeval rto = {5, 0};  // give up after 5s idle
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof(rto));
        unsigned char rb[4200];
        int served = 0;
        while (served < readresp) {
            ssize_t r = recv(fd, rb, sizeof(rb), 0);
            if (r < 0) {
                break;  // idle timeout
            }
            if (r < 14 + 32 || rb[12] != 0x1a || rb[13] != 0xf6 || rb[14] != 0x20) {
                continue;  // not a READ_REQ
            }
            const unsigned char* rq = rb + 14;                // request header
            uint16_t tag = (uint16_t)(rq[2] | (rq[3] << 8));  // correlation tag
            uint32_t req_len = rq[4] | (rq[5] << 8) | (rq[6] << 16) | (rq[7] << 24);
            uint32_t rseq = rq[8] | (rq[9] << 8) | (rq[10] << 16) | (rq[11] << 24);
            if (req_len > 4080u) {
                req_len = 4080u;
            }
            unsigned char* h = frame + 14;  // build READ_RESP into the pre-addressed frame (dst=BH)
            h[0] = 0x21;                    // READ_RESP
            h[1] = 0x01;
            put_u16(h + 2, tag);      // echo tag for the initiator's correlation
            put_u32(h + 4, req_len);  // payload length
            put_u32(h + 8, rseq);     // echo seq
            put_u32(h + 12, 0);       // rkey unused
            put_u64(h + 16, 0);       // roff unused
            put_u32(h + 24, 0);       // imm
            put_u32(h + 28, tt_crc32(h, 28));
            unsigned char* pay = frame + 14 + 32;
            pay[0] = 'R';
            pay[1] = 'E';
            pay[2] = 'A';
            pay[3] = 'D';
            for (unsigned i = 4; i < req_len; i++) {
                pay[i] = (unsigned char)(i & 0xff);
            }
            sendto(fd, frame, 14u + 32u + req_len, 0, (struct sockaddr*)&sa, sizeof(sa));
            served++;
        }
        printf("readresp: served %d/%d READ_REQ -> READ_RESP\n", served, readresp);
        close(fd);
        return 0;
    }

    // ---- ARQ SENDER (Phase 2.2a): send `count` reliable WRITEs seq 1..count, collect cumulative ACKs. ----
    if (arqsend > 0) {
        struct sockaddr_ll sa;
        memset(&sa, 0, sizeof(sa));
        sa.sll_family = AF_PACKET;
        sa.sll_ifindex = ifindex;
        sa.sll_halen = 6;
        memcpy(sa.sll_addr, frame, 6);  // dst = BH RXQ2 (02:...:02)
        if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("bind");
        }
        struct packet_mreq pmr;
        memset(&pmr, 0, sizeof(pmr));
        pmr.mr_ifindex = ifindex;
        pmr.mr_type = PACKET_MR_PROMISC;  // BH's ACK dst MAC is the initiator's; grab it regardless
        setsockopt(fd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pmr, sizeof(pmr));
        // Build a WRITE (0x10) template landing at the MR (rkey/roff from args; payload TTWR...).
        unsigned char* h = frame + 14;
        h[0] = 0x10;
        h[1] = 0x01;
        put_u16(h + 2, 0);
        put_u32(h + 4, plen);
        put_u32(h + 12, rkey);
        put_u64(h + 16, roff);
        put_u32(h + 24, 0);
        unsigned char* pay = frame + 14 + 32;
        pay[0] = 'T';
        pay[1] = 'T';
        pay[2] = 'W';
        pay[3] = 'R';
        for (unsigned i = 4; i < plen; i++) {
            pay[i] = (unsigned char)(i & 0xff);
        }
        const uint32_t N = (uint32_t)count;
        uint32_t sent = 0;
        for (uint32_t seq = 1; seq <= N; seq++) {
            if (gapseq != 0u && seq == gapseq) {
                continue;  // simulate loss of this frame (permanent gap for the in-order receiver)
            }
            put_u32(h + 8, seq);               // reliable sequence number (word2)
            put_u32(h + 28, tt_crc32(h, 28));  // recompute header CRC for this seq
            sendto(fd, frame, 14u + 32u + plen, 0, (struct sockaddr*)&sa, sizeof(sa));
            sent++;
            if (arqpace > 0) {
                usleep((useconds_t)arqpace);  // inter-frame gap (µs) to keep the in-order test un-reordered
            }
        }
        // Collect cumulative ACKs (0x40) for up to ~2s idle; track the highest ack_seq the BH reached.
        struct timeval rto = {2, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof(rto));
        unsigned char rb[128];
        uint32_t max_ack = 0, n_ack = 0;
        for (;;) {
            ssize_t r = recv(fd, rb, sizeof(rb), 0);
            if (r < 0) {
                break;  // idle timeout -> done collecting
            }
            if (r < 14 + 32 || rb[12] != 0x1a || rb[13] != 0xf6 || rb[14] != 0x40) {
                continue;  // not an ACK
            }
            const unsigned char* a = rb + 14;
            uint32_t ack_seq = a[8] | (a[9] << 8) | (a[10] << 16) | (a[11] << 24);
            if (ack_seq > max_ack) {
                max_ack = ack_seq;
            }
            n_ack++;
            if (gapseq == 0u && max_ack >= N) {
                break;  // in-order run fully acknowledged
            }
        }
        printf(
            "arqsend: sent %u/%u (gapseq=%u) acks_rx=%u max_ack=%u %s\n",
            sent,
            N,
            gapseq,
            n_ack,
            max_ack,
            (gapseq == 0u ? (max_ack >= N ? "FULL-ACK" : "INCOMPLETE")
                          : (max_ack == gapseq - 1u ? "STALLED-AT-GAP" : "UNEXPECTED")));
        close(fd);
        return 0;
    }

    // ---- ARQ RESPONDER (Phase 2.2b): ACK BH's reliable WRITEs; drop one to force a Go-Back-N resend. ----
    if (arqresp > 0) {
        struct sockaddr_ll sa;
        memset(&sa, 0, sizeof(sa));
        sa.sll_family = AF_PACKET;
        sa.sll_ifindex = ifindex;
        sa.sll_halen = 6;
        memcpy(sa.sll_addr, frame, 6);  // ACK dst = BH RXQ2 (02:...:02)
        if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("bind");
        }
        struct packet_mreq pmr;
        memset(&pmr, 0, sizeof(pmr));
        pmr.mr_ifindex = ifindex;
        pmr.mr_type = PACKET_MR_PROMISC;
        setsockopt(fd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pmr, sizeof(pmr));
        struct timeval rto = {6, 0};  // give up after 6s idle
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof(rto));
        unsigned char rb[4200];
        uint32_t rx_last = 0, n_rx = 0, n_acktx = 0, dropped = 0;
        while ((int)rx_last < arqresp) {
            ssize_t r = recv(fd, rb, sizeof(rb), 0);
            if (r < 0) {
                break;  // idle timeout
            }
            if (r < 14 + 32 || rb[12] != 0x1a || rb[13] != 0xf6 || rb[14] != 0x10) {
                continue;  // not a reliable WRITE
            }
            const unsigned char* w = rb + 14;
            uint32_t seq = w[8] | (w[9] << 8) | (w[10] << 16) | (w[11] << 24);
            n_rx++;
            if (dropreq != 0u && seq == dropreq && dropped == 0u) {
                dropped = 1u;  // drop this frame once (simulate loss) -- do not ACK past it
                continue;
            }
            if (seq == rx_last + 1u) {
                rx_last = seq;  // in-order (BH resends the suffix in order after the gap)
            }
            // Send a cumulative ACK(rx_last) back to the BH initiator.
            unsigned char* h = frame + 14;
            h[0] = 0x40;  // ACK
            h[1] = 0x01;
            put_u16(h + 2, 0);
            put_u32(h + 4, 0);
            put_u32(h + 8, rx_last);  // cumulative ack_seq
            put_u32(h + 12, 0);
            put_u64(h + 16, 0);
            put_u32(h + 24, 0);
            put_u32(h + 28, tt_crc32(h, 28));
            memset(frame + 14 + 32, 0, 16);  // header-only: pad to 48B (runt-safe)
            sendto(fd, frame, 14u + 48u, 0, (struct sockaddr*)&sa, sizeof(sa));
            n_acktx++;
        }
        printf(
            "arqresp: rx=%u rx_last=%u/%d acks_tx=%u dropped=%u %s\n",
            n_rx,
            rx_last,
            arqresp,
            n_acktx,
            dropped,
            ((int)rx_last >= arqresp ? "ALL-ACKED" : "INCOMPLETE"));
        close(fd);
        return 0;
    }

    // ---- READ round-trip latency: send READ_REQ, time until READ_RESP (0x21) returns. ----
    if (readlat > 0) {
        struct sockaddr_ll sa;
        memset(&sa, 0, sizeof(sa));
        sa.sll_family = AF_PACKET;
        sa.sll_ifindex = ifindex;
        sa.sll_halen = 6;
        memcpy(sa.sll_addr, frame, 6);
        if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("bind");
        }
        // Promiscuous: the READ_RESP's dst MAC is the initiator's, not this host's, so the NIC would
        // otherwise filter it out (tcpdump sees it only because it enables promisc).
        struct packet_mreq pmr;
        memset(&pmr, 0, sizeof(pmr));
        pmr.mr_ifindex = ifindex;
        pmr.mr_type = PACKET_MR_PROMISC;
        setsockopt(fd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pmr, sizeof(pmr));
        struct timeval rto = {0, 50000};  // 50 ms recv timeout per sample
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof(rto));
        double* us = (double*)malloc((size_t)readlat * sizeof(double));
        int got = 0;
        unsigned char rb[4200];
        for (int i = 0; i < readlat; i++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (sendto(fd, frame, frame_len, 0, (struct sockaddr*)&sa, sizeof(sa)) != (ssize_t)frame_len) {
                continue;
            }
            int done = 0;
            for (;;) {
                ssize_t r = recv(fd, rb, sizeof(rb), 0);
                if (r < 0) {
                    break;  // 50ms timeout -> drop this sample
                }
                if (r >= 14 + 32 && rb[12] == 0x1a && rb[13] == 0xf6 && rb[14] == 0x21) {
                    done = 1;
                    break;  // READ_RESP (op 0x21)
                }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double e = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
                if (e > 50000.0) {
                    break;
                }
            }
            if (done) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                us[got++] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
            }
        }
        if (got == 0) {
            printf("readlat: 0/%d responses (is the BH in read-target mode?)\n", readlat);
            free(us);
            close(fd);
            return 1;
        }
        // sort for percentiles (simple insertion sort; N is small)
        for (int a = 1; a < got; a++) {
            double v = us[a];
            int b = a - 1;
            while (b >= 0 && us[b] > v) {
                us[b + 1] = us[b];
                b--;
            }
            us[b + 1] = v;
        }
        double sum = 0;
        for (int a = 0; a < got; a++) {
            sum += us[a];
        }
        printf(
            "readlat: n=%d  min=%.1f  p50=%.1f  avg=%.1f  p99=%.1f  max=%.1f  (us round-trip)\n",
            got,
            us[0],
            us[got / 2],
            sum / got,
            us[(int)(got * 0.99)],
            us[got - 1]);
        free(us);
        close(fd);
        return 0;
    }

    // ---- BLAST path: N threads x sendmmsg(batch) to saturate the wire. ----
    if (threads > 1 || batch > 1) {
        close(fd);
        pthread_t* th = (pthread_t*)calloc(threads, sizeof(pthread_t));
        struct blast_arg* ta = (struct blast_arg*)calloc(threads, sizeof(struct blast_arg));
        long per = count / threads;
        for (int i = 0; i < threads; i++) {
            ta[i].ifindex = ifindex;
            memcpy(ta[i].frame, frame, sizeof(frame));
            ta[i].frame_len = frame_len;
            ta[i].count = (i == threads - 1) ? (count - per * (threads - 1)) : per;
            ta[i].batch = batch;
            pthread_create(&th[i], NULL, blast_thread, &ta[i]);
        }
        long total = 0;
        for (int i = 0; i < threads; i++) {
            pthread_join(th[i], NULL);
            total += ta[i].count;
        }
        printf(
            "tt_rdma_bf3_send: BLAST sent %ld/%ld frames (%d threads x batch %d) frame_len=%u plen=%u\n",
            total,
            count,
            threads,
            batch,
            frame_len,
            plen);
        free(th);
        free(ta);
        return 0;
    }

    // ---- Legacy single-sendto path (M-1b probe / correctness tests). ----
    struct sockaddr_ll sa;
    memset(&sa, 0, sizeof(sa));
    sa.sll_family = AF_PACKET;
    sa.sll_ifindex = ifindex;
    sa.sll_halen = 6;
    memcpy(sa.sll_addr, frame, 6);

    long sent = 0, failed = 0;
    int first_errno = 0;
    for (long i = 0; i < count; i++) {
        if (opcode != 0) {
            put_u32(frame + 14 + 8, (uint32_t)(i + 1));                    // seq = 1..count
            put_u32(frame + 14 + 28, tt_crc32(frame + 14, 28) ^ crc_xor);  // re-stamp CRC after seq change
        }
        ssize_t r = sendto(fd, frame, frame_len, 0, (struct sockaddr*)&sa, sizeof(sa));
        if (r == (ssize_t)frame_len) {
            sent++;
        } else {
            failed++;
            if (first_errno == 0) {
                first_errno = errno;
            }
        }
    }
    printf(
        "tt_rdma_bf3_send: sent %ld/%ld (failed %ld, first errno=%d '%s') on %s frame_len=%u plen=%u\n",
        sent,
        count,
        failed,
        first_errno,
        first_errno ? strerror(first_errno) : "-",
        iface,
        frame_len,
        plen);
    close(fd);
    return 0;
}
