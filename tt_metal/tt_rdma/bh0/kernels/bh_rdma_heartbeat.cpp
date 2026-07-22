// SPDX-License-Identifier: Apache-2.0
//
// BH.0 heartbeat kernel — TT-RDMA Blackhole bring-up gate (see
// docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md §2.4 BH.0 + the BH.0 deep-dive).
//
// Runs on the SUBORDINATE erisc (RISC1 / DM1) of an active eth core. It is a
// stand-in for the eventual RX/TX drain loop and exists to prove ONE thing:
// while a resident kernel runs on RISC1, RISC0 (active_erisc) keeps
// yielding to the base FW (service_eth_msg via its go-loop idle-wait) so the
// trained link stays UP. If port_status leaves UP while this runs, the
// coexistence model is wrong and everything downstream is blocked.
//
// STOP MODEL: the kernel always RETURNS so the RISC0 go-loop can reap it and
// RISC1 goes back to idle (no chip reset needed, host close_device() is clean).
// It ends on whichever comes first: a bounded beat count (gtest) OR a host-set
// graceful-stop flag in L1 (standalone tool). A never-returning busy-spin would
// pin RISC1 in an active power state until a reset — deliberately avoided here.
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
    // arg2 = num_beats. 0 = run until the host sets the stop flag (standalone soak tool). >0 =
    // BOUNDED (gtest: writes beat counts up to num_beats then returns, final hb == num_beats).
    const uint32_t num_beats = get_arg_val<uint32_t>(2);
    // arg3 = stop-flag L1 byte address (TT_RDMA_STOP_ADDR), or 0 to disable. The soak tool passes
    // it and writes non-zero to ask the kernel to finish; the gtest passes 0 (bounded only).
    const uint32_t stop_addr = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;  // clear on entry so a stale value from a prior boot can't stop us immediately
    }

    for (uint32_t beat = 1;; ++beat) {
        *hb = beat;  // heartbeat the gate watches advance (final value == num_beats when bounded)
        if (num_beats != 0 && beat >= num_beats) {
            break;  // bounded (gtest) — reached the target count
        }
        if (stop != nullptr && *stop != 0) {
            break;  // host asked us to stop (soak tool) — return so the go-loop reaps us -> RISC1 idles
        }
        for (volatile uint32_t i = 0; i < spin; ++i) {
            // pace only — deliberately NO NoC ops, NO service_eth_msg, NO base-FW calls
        }
    }
}
