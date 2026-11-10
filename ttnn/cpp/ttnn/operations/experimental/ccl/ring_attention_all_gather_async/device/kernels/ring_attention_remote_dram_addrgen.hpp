// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace ring_attention_all_gather {

struct RemoteDramBankTable {
    static constexpr uint32_t kEntries = NUM_NOCS * NUM_DRAM_BANKS;
    static constexpr uint64_t kLocalAddrMask = (static_cast<uint64_t>(1) << NOC_ADDR_LOCAL_BITS) - 1;

    bool enabled = false;
    std::array<uint32_t, kEntries> packed_xy{};

    static RemoteDramBankTable from_args(uint32_t& arg_idx) {
        RemoteDramBankTable table;
        table.enabled = get_arg_val<uint32_t>(arg_idx++) != 0;
        for (uint32_t i = 0; i < kEntries; ++i) {
            table.packed_xy[i] = get_arg_val<uint32_t>(arg_idx++);
        }
        return table;
    }

    FORCE_INLINE uint64_t remap(uint64_t local_noc_addr, uint32_t page_id, uint8_t noc) const {
        if (!enabled) {
            return local_noc_addr;
        }
        const uint32_t bank = interleaved_addr_gen::get_bank_index<true>(
            page_id, interleaved_addr_gen::get_bank_offset_index<true>(page_id));
        const uint64_t xy = packed_xy[noc * NUM_DRAM_BANKS + bank];
        return (xy << NOC_ADDR_COORD_SHIFT) | (local_noc_addr & kLocalAddrMask);
    }
};

struct RemoteDramAddrGen {
    AbstractTensorAccessorWrapper local;
    const RemoteDramBankTable* table = nullptr;

    uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = noc_index) const {
        return table->remap(local.get_noc_addr(page_id, offset, noc), page_id, noc);
    }
};

}  // namespace ring_attention_all_gather
