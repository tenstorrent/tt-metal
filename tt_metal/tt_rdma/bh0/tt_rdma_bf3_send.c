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
//                         [threads] [batch]
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
            "[threads] [batch]\n",
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
        h[0] = (unsigned char)opcode;
        h[1] = 0x01;
        put_u16(h + 2, 0);
        put_u32(h + 4, plen);
        put_u32(h + 8, 1);  // seq (fixed; the BH kernel does not order-check)
        put_u32(h + 12, rkey);
        put_u64(h + 16, roff);
        put_u32(h + 24, 0);
        put_u32(h + 28, 0);
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
            put_u32(frame + 14 + 8, (uint32_t)(i + 1));  // seq = 1..count
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
