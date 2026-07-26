// SPDX-License-Identifier: Apache-2.0
//
// BF3-side raw-L2 sender for the TT-RDMA RX path. Two modes:
//
//   Legacy PROBE (opcode arg omitted / 0): emits a 64B frame whose payload starts with magic
//   0xB1F5C0DE + "TT-RDMA-RX-PROBE" (M-1b reception byte-match).
//
//   TT-RDMA frame (opcode > 0): emits [L2][32B tt_rdma_hdr][payload] so the on-core dispatch kernel
//   (bh_rdma_rx_dispatch) can parse + dispatch by opcode. Header is little-endian per tt_rdma_wire.h:
//   opcode, version_flags=1, tag=0, length=payload_len, seq=i (incrementing), rkey, remote_offset,
//   imm=0, cksum=0 (Stage 1 kernel does not check CRC). Payload = "TTWR" + incrementing bytes.
//
// Single-purpose (fixed-shape frames) so it is safe to allowlist for passwordless sudo.
//
//   sudo tt_rdma_bf3_send <iface> [count] [dst_mac] [ethertype] [opcode] [payload_len] [rkey] [roff]
//   e.g. WRITE: sudo tt_rdma_bf3_send enp193s0f0np0 256 02:00:00:00:00:02 0x1af6 0x10 256 0x07000042 0
#include <errno.h>
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(
            stderr, "usage: %s <iface> [count] [dst_mac] [ethertype] [opcode] [payload_len] [rkey] [roff]\n", argv[0]);
        return 2;
    }
    const char* iface = argv[1];
    int count = (argc > 2) ? atoi(argv[2]) : 1000;
    const char* dmac_s = (argc > 3) ? argv[3] : "02:00:00:00:00:02";  // unicast -> TT RXQ2 ("other")
    unsigned etht = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 0) : 0x1af6u;
    unsigned opcode = (argc > 5) ? (unsigned)strtoul(argv[5], NULL, 0) : 0u;  // 0 => legacy PROBE frame
    unsigned plen = (argc > 6) ? (unsigned)strtoul(argv[6], NULL, 0) : 32u;
    uint32_t rkey = (argc > 7) ? (uint32_t)strtoul(argv[7], NULL, 0) : 0u;
    uint64_t roff = (argc > 8) ? (uint64_t)strtoull(argv[8], NULL, 0) : 0u;

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
    unsigned char* pay;  // payload start
    if (opcode == 0) {
        // legacy PROBE: magic + label at payload
        frame[14] = 0xde;
        frame[15] = 0xc0;
        frame[16] = 0xf5;
        frame[17] = 0xb1;
        memcpy(frame + 18, "TT-RDMA-RX-PROBE", 16);
        frame_len = 64;
        pay = NULL;
    } else {
        if (plen > 4080u) {
            plen = 4080u;
        }
        unsigned char* h = frame + 14;  // 32B TT-RDMA header
        h[0] = (unsigned char)opcode;   // opcode
        h[1] = 0x01;                    // version_flags: ver=1
        put_u16(h + 2, 0);              // tag
        put_u32(h + 4, plen);           // length (payload bytes)
        put_u32(h + 8, 0);              // seq (patched per-frame below)
        put_u32(h + 12, rkey);          // rkey
        put_u64(h + 16, roff);          // remote_offset
        put_u32(h + 24, 0);             // imm
        put_u32(h + 28, 0);             // header_cksum (Stage 1 kernel ignores)
        pay = frame + 14 + 32;
        pay[0] = 'T';
        pay[1] = 'T';
        pay[2] = 'W';
        pay[3] = 'R';
        for (unsigned i = 4; i < plen; i++) {
            pay[i] = (unsigned char)(i & 0xff);
        }
        frame_len = 14u + 32u + plen;
    }

    struct sockaddr_ll sa;
    memset(&sa, 0, sizeof(sa));
    sa.sll_family = AF_PACKET;
    sa.sll_ifindex = ifindex;
    sa.sll_halen = 6;
    memcpy(sa.sll_addr, frame, 6);

    int sent = 0, failed = 0, first_errno = 0;
    for (int i = 0; i < count; i++) {
        if (opcode != 0) {
            put_u32(frame + 14 + 8, (uint32_t)(i + 1));  // seq = 1..count
        }
        ssize_t r = sendto(fd, frame, frame_len, 0, (struct sockaddr*)&sa, sizeof(sa));
        if (r == (ssize_t)frame_len) {
            sent++;
        } else {
            failed++;
            if (first_errno == 0) {
                first_errno = errno;  // capture WHY jumbo is rejected (EMSGSIZE=90, EINVAL=22, ...)
            }
        }
    }
    printf(
        "tt_rdma_bf3_send: sent %d/%d (failed %d, first errno=%d '%s') on %s frame_len=%u plen=%u\n",
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
