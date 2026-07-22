// SPDX-License-Identifier: Apache-2.0
//
// BH.0 heartbeat kernel — TT-RDMA Blackhole bring-up gate (see
// docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md §2.4 BH.0 + the BH.0 deep-dive).
//
// Runs on the SUBORDINATE erisc (RISC1 / DM1) of an active eth core. It is a
// stand-in for the eventual RX/TX drain loop and exists to prove ONE thing:
// while a persistent kernel is resident on RISC1, RISC0 (active_erisc) keeps
// yielding to the base FW (service_eth_msg via its go-loop idle-wait) so the
// trained link stays UP. If port_status leaves UP while this runs, the
// coexistence model is wrong and everything downstream is blocked.
//
// HARD CONTRACT (tt_rdma_l1_layout.h): this kernel only writes inside the RDMA
// SW-L1 region and MUST NOT write >= MEM_SYSENG_RESERVED_BASE (0x70000) — the
// base FW / boot_results (0x7CC00) / mailboxes (0x7D000) live there and writing
// it BRICKS THE LINK. It also makes NO base-FW/api-table calls: link
// maintenance is RISC0's job, not RISC1's.

#include <cstdint>

void kernel_main() {
    // arg0 = heartbeat L1 byte address. Host passes TT_RDMA_RCB_ADDR (from
    //        tt_rdma_l1_layout.h) — inside the RDMA region, clear of 0x70000+.
    // arg1 = spin iterations between beats (paces the writes so a host poll /
    //        tt-exalens brxy can watch the counter advance).
    const uint32_t hb_addr = get_arg_val<uint32_t>(0);
    const uint32_t spin = get_arg_val<uint32_t>(1);

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);

    uint32_t beat = 0;
    for (;;) {         // persistent — mirrors the eventual drain loop
        *hb = ++beat;  // the heartbeat the gate watches advance
        for (volatile uint32_t i = 0; i < spin; ++i) {
            // pace only — deliberately NO NoC ops, NO service_eth_msg, NO base-FW calls
        }
    }
    // For a first *bounded* smoke (clean program completion, no reset needed),
    // replace the `for (;;)` above with `for (uint32_t b = 0; b < num_beats; ++b)`
    // and take num_beats as arg2.
}
