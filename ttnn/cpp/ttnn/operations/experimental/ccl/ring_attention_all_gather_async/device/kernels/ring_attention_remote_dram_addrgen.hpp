// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Remote-chip DRAM addressing for the ring-attention fabric writers.
//
// A fabric write ships a full NOC address in the packet header, and the receiving chip issues the NOC
// write with that (x, y) verbatim. The writers build that address with THIS chip's L1
// dram_bank_to_noc_xy table. On Blackhole the TRANSLATED coordinate of a logical DRAM bank depends on
// WHICH bank the chip has harvested, so two chips with different harvested DRAM banks place the same
// logical bank at different coordinates: an address built here lands in the wrong bank (or at a
// coordinate that does not exist) over there.
//
// The host therefore appends the RECEIVING chip's table (one word per (noc, bank), packed exactly like
// the L1 table) after the kernel's other runtime args, preceded by a flag that is set only when that
// table differs from the local one. With the flag clear the kernel's addressing is untouched.
//
// Only interleaved DRAM destinations are remapped (the host enforces this when the flag is set). The
// per-bank DRAM address offsets (bank_to_dram_offset) are all zero on Blackhole, so the local part of
// the address is the same on both chips and only the (x, y) is rewritten.

#pragma once

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace ring_attention_all_gather {

struct RemoteDramBankTable {
    // KEEP IN SYNC with the host (make_remote_dram_bank_table_args in the all-gather program factory):
    // the host pushes hal.get_num_nocs() * allocator num_dram_banks words, which match these macros.
    static constexpr uint32_t kEntries = NUM_NOCS * NUM_DRAM_BANKS;
    static constexpr uint64_t kLocalAddrMask = (static_cast<uint64_t>(1) << NOC_ADDR_LOCAL_BITS) - 1;

    bool enabled = false;
    std::array<uint32_t, kEntries> packed_xy{};

    // Consumes [flag, kEntries words] from the runtime args at arg_idx. The words are always present
    // (zero when the flag is clear) so the layout is fixed.
    static RemoteDramBankTable from_args(uint32_t& arg_idx) {
        RemoteDramBankTable table;
        table.enabled = get_arg_val<uint32_t>(arg_idx++) != 0;
        for (uint32_t i = 0; i < kEntries; ++i) {
            table.packed_xy[i] = get_arg_val<uint32_t>(arg_idx++);
        }
        return table;
    }

    // Rewrites the (x, y) of a local interleaved-DRAM NOC address for `page_id` (built for `noc`) to the
    // receiving chip's placement of the same bank. Identity when the tables match.
    FORCE_INLINE uint64_t remap(uint64_t local_noc_addr, uint32_t page_id, uint8_t noc) const {
        if (!enabled) {
            return local_noc_addr;
        }
        // Same page -> bank mapping as InterleavedAddrGen::get_noc_addr.
        const uint32_t bank = interleaved_addr_gen::get_bank_index<true>(
            page_id, interleaved_addr_gen::get_bank_offset_index<true>(page_id));
        const uint64_t xy = packed_xy[noc * NUM_DRAM_BANKS + bank];
        return (xy << NOC_ADDR_COORD_SHIFT) | (local_noc_addr & kLocalAddrMask);
    }
};

// Drop-in replacement for AbstractTensorAccessorWrapper in the fabric write helpers
// (fabric_write_unidir / scatter_fabric_write_unidir only call get_noc_addr(id, offset, noc)).
struct RemoteDramAddrGen {
    AbstractTensorAccessorWrapper local;
    // Pointer rather than a copy: the write helpers take the addrgen by value per packet.
    const RemoteDramBankTable* table = nullptr;

    uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = noc_index) const {
        return table->remap(local.get_noc_addr(page_id, offset, noc), page_id, noc);
    }
};

}  // namespace ring_attention_all_gather
