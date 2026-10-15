// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// DRAM endpoint spread (Perf 2). A Blackhole DRAM channel has 3 NoC endpoints that all serve the same
// channel memory at the same bank-local address; dram_bank_to_noc_xy (what TensorAccessor uses) holds
// one per NoC, so every core's writes to a bank funnel through one NoC endpoint. The host passes all 3
// endpoints per bank (packed y << NOC_ADDR_NODE_ID_BITS | x, from device.dram_core_from_logical_core)
// as runtime args; the writer rotates its requests over them. Interleaved DRAM only: bank = page % banks.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kDramEndpointsPerBank = 3;

struct DramEndpointSpread {
    uint32_t xy[NUM_DRAM_BANKS * kDramEndpointsPerBank];
    uint32_t next;  // rotation counter, seeded per core so cores start on different endpoints

    void load(uint32_t first_arg, uint32_t seed) {
        for (uint32_t i = 0; i < NUM_DRAM_BANKS * kDramEndpointsPerBank; ++i) {
            xy[i] = get_arg_val<uint32_t>(first_arg + i);
        }
        next = seed % kDramEndpointsPerBank;
    }

    // Same bank-local address as `table_addr` (from the accessor), next endpoint of page's bank.
    uint64_t addr(uint64_t table_addr, uint32_t page) {
        const uint32_t port = next;
        next = next + 1 == kDramEndpointsPerBank ? 0 : next + 1;
        return ((uint64_t)xy[(page % NUM_DRAM_BANKS) * kDramEndpointsPerBank + port] << NOC_ADDR_COORD_SHIFT) |
               (table_addr & NOC_LOCAL_ADDR_MASK);
    }
};
