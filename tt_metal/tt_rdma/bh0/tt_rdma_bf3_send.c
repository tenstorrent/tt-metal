// SPDX-License-Identifier: Apache-2.0
//
// BH.1 / M-1b BF3-side raw sender: emit N raw-L2 frames (default ethertype 0x1AF6) out a NIC netdev
// toward the TT eth rail, so the on-core RX kernel (bh_rdma_recv_probe) can catch them.
//
// Single-purpose (only builds + sends a fixed-shape frame) so it is safe to allowlist for passwordless
// sudo in nic-debug-sudoers.sh, instead of granting sudo to python/scapy.
//
//   sudo tt_rdma_bf3_send <iface> [count] [dst_mac aa:bb:..] [ethertype_hex]
//   e.g. sudo tt_rdma_bf3_send enp193s0f0np0 2000 02:00:00:00:00:02 0x1af6
//
// Payload begins with magic 0xB1F5C0DE (little-endian on the wire: de c0 f5 b1) so the receiver can
// byte-match: after L2 strip the TT RX buffer word[0] should read 0xB1F5C0DE.
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <iface> [count] [dst_mac] [ethertype_hex]\n", argv[0]);
        return 2;
    }
    const char* iface = argv[1];
    int count = (argc > 2) ? atoi(argv[2]) : 1000;
    const char* dmac_s = (argc > 3) ? argv[3] : "02:00:00:00:00:02";  // unicast -> TT RXQ2 ("other")
    unsigned etht = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 0) : 0x1af6u;

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

    // Build a 64-byte frame: [dst][src][ethertype][magic + label + pad]
    unsigned char frame[64];
    memset(frame, 0, sizeof(frame));
    for (int i = 0; i < 6; i++) {
        frame[i] = (unsigned char)dm[i];
    }
    memcpy(frame + 6, smac, 6);
    frame[12] = (unsigned char)(etht >> 8);
    frame[13] = (unsigned char)(etht & 0xff);
    // payload magic 0xB1F5C0DE little-endian, then an ASCII label
    frame[14] = 0xde;
    frame[15] = 0xc0;
    frame[16] = 0xf5;
    frame[17] = 0xb1;
    memcpy(frame + 18, "TT-RDMA-RX-PROBE", 16);

    struct sockaddr_ll sa;
    memset(&sa, 0, sizeof(sa));
    sa.sll_family = AF_PACKET;
    sa.sll_ifindex = ifindex;
    sa.sll_halen = 6;
    memcpy(sa.sll_addr, frame, 6);

    int sent = 0;
    for (int i = 0; i < count; i++) {
        if (sendto(fd, frame, sizeof(frame), 0, (struct sockaddr*)&sa, sizeof(sa)) == (ssize_t)sizeof(frame)) {
            sent++;
        }
    }
    printf("tt_rdma_bf3_send: sent %d/%d frames on %s dst=%s ethertype=0x%04x\n", sent, count, iface, dmac_s, etht);
    close(fd);
    return 0;
}
