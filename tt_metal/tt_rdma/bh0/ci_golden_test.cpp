// SPDX-License-Identifier: Apache-2.0
//
// HW-less CI unit test (Phase 0.4 of tt-rdma-production-plan.md): the TT-RDMA-v1 wire-header oracle.
// Builds the spec §7 golden headers with tt_rdma_build_hdr and asserts they are byte-for-byte the frozen
// golden vectors in tt_rdma_wire.h, plus the canonical CRC-32 vector. No hardware, no tt-metal, no DOCA
// — runs anywhere on any PR:
//
//   g++ -std=c++17 -I<repo-root> tt_metal/tt_rdma/bh0/ci_golden_test.cpp -o /tmp/ci_golden && /tmp/ci_golden
//
// If a golden vector drifts (wire format changed, CRC changed, struct packing changed), this fails —
// catching a wire-incompatibility before it ships to either the chip kernel or the DOCA gateway codec.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"

static int g_pass = 0, g_fail = 0;

static void check_hdr(const char* name, const tt_rdma_hdr_t* h, const uint8_t golden[32]) {
    if (std::memcmp(h, golden, 32) == 0) {
        std::printf("  PASS: %s header == golden\n", name);
        ++g_pass;
    } else {
        std::printf("  FAIL: %s header != golden\n    got:   ", name);
        const uint8_t* g = (const uint8_t*)h;
        for (int i = 0; i < 32; ++i) {
            std::printf("%02x ", g[i]);
        }
        std::printf("\n    want:  ");
        for (int i = 0; i < 32; ++i) {
            std::printf("%02x ", golden[i]);
        }
        std::printf("\n");
        ++g_fail;
    }
}

static void check_u32(const char* name, uint32_t got, uint32_t want) {
    if (got == want) {
        std::printf("  PASS: %s = 0x%08x\n", name, got);
        ++g_pass;
    } else {
        std::printf("  FAIL: %s = 0x%08x (want 0x%08x)\n", name, got, want);
        ++g_fail;
    }
}

int main() {
    std::printf("== TT-RDMA-v1 wire-header golden vectors (HW-less) ==\n");

    // Canonical CRC-32 (poly 0x04C11DB7, ETH-CTRL ICRC) check vector.
    check_u32("crc32(\"123456789\")", tt_rdma_crc32((const uint8_t*)"123456789", 9), 0xCBF43926u);

    // §7.1 SEND: 64 B payload, tag=0xCAFE, seq=0x1234.
    tt_rdma_hdr_t h;
    tt_rdma_build_hdr(&h, TT_OP_SEND, TT_RDMA_VERSION, 0xCAFE, 64, 0x1234, 0, 0, 0);
    check_hdr("SEND", &h, TT_GOLDEN_SEND_HDR);

    // §7.2 WRITE_IMM: 1408 B, tag=1, seq=0x12345678, rkey=0xDEADBEEF, remote_offset=0x10000, imm=0xAB.
    tt_rdma_build_hdr(
        &h, TT_OP_WRITE_IMM, TT_RDMA_VERSION | TT_VF_IMM_PRESENT, 0x0001, 1408, 0x12345678, 0xDEADBEEF, 0x10000, 0xAB);
    check_hdr("WRITE_IMM", &h, TT_GOLDEN_WRITE_IMM_HDR);

    // §7.5 ACK: header-only (length 0), cumulative ack=0x12345677, imm=1.
    tt_rdma_build_hdr(&h, TT_OP_ACK, TT_RDMA_VERSION, 0, 0, 0x12345677, 0, 0, 1);
    check_hdr("ACK", &h, TT_GOLDEN_ACK_HDR);

    std::printf("== CI golden: %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
