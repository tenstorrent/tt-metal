// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Native sharded roll copy kernel — supports both L1-sharded and DRAM-sharded tensors.
//
// Roll is realised as a gather: the host computes, per destination core, a list of
// segment-copy descriptors. Each descriptor copies a contiguous run of `copy_size` bytes
// from a source shard (possibly on another core/bank) into the local output shard.
//
// L1 mode: segment offsets can be arbitrary (a within-row rotation produces unaligned runs),
// so every segment is staged through a scratch CB: we NOC-read an alignment-padded superset
// of the run into the scratch buffer, then copy the exact bytes out locally. The scratch CB
// is double-buffered so reads overlap copies.
//
// DRAM mode: source data is read from DRAM via get_noc_addr_from_bank_id; the result is
// staged in scratch L1. The destination is also DRAM, written back via NOC async write.
// DRAM transfers are tile-aligned so no sub-alignment padding is needed.

// Local copy of `copy_size` bytes. Prefers 32-bit word copies; falls back to bytes.
inline void local_copy(uint32_t src_addr, uint32_t dst_addr, uint32_t copy_size) {
    if (((src_addr | dst_addr) & 3u) == 0) {
        const uint32_t words = copy_size >> 2;
        volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
        volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
        for (uint32_t w = 0; w < words; w++) {
            d32[w] = s32[w];
        }
        volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_addr);
        volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_addr);
        for (uint32_t b = words << 2; b < copy_size; b++) {
            d8[b] = s8[b];
        }
    } else {
        volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_addr);
        volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_addr);
        for (uint32_t b = 0; b < copy_size; b++) {
            d8[b] = s8[b];
        }
    }
}

void kernel_main() {
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t scratch_half = get_compile_time_arg_val(3);
    constexpr uint32_t mode = get_compile_time_arg_val(4);             // 0=L1, 1=DRAM_TILE, 2=DRAM_RM
    constexpr uint32_t shard_size = get_compile_time_arg_val(5);       // DRAM_RM: full shard bytes
    constexpr uint32_t dram_src0_cb_id = get_compile_time_arg_val(6);  // DRAM_RM: staging slot 0
    constexpr uint32_t dram_src1_cb_id = get_compile_time_arg_val(7);  // DRAM_RM: staging slot 1
    constexpr uint32_t dram_dst_cb_id = get_compile_time_arg_val(8);   // DRAM_RM: dst assembly

    // Scratch CB is used by L1 mode (alignment padding) and DRAM TILE mode (staging).
    // Declared here so both branches can access it.
    CircularBuffer scratch_cb(scratch_cb_id);
    const uint32_t scratch_base = scratch_cb.get_write_ptr();

    Noc noc;

    uint32_t arg_idx = 0;

    if constexpr (mode == 2) {
        // DRAM RM mode: read full source shard(s) from DRAM into L1 staging, assemble the
        // rolled result in L1 via local element copies, then write full shard to DRAM dst.
        // All DRAM NOC transfers are shard-sized (validated DRAM-alignment aligned) — no
        // sub-alignment issue. The host computes the staging row pitch using the DRAM alignment
        // (64B on Blackhole, 32B on Wormhole) so the staged layout matches the in-DRAM layout.
        CircularBuffer src_cbs[2] = {CircularBuffer(dram_src0_cb_id), CircularBuffer(dram_src1_cb_id)};
        CircularBuffer dst_cb(dram_dst_cb_id);
        const uint32_t src_base[2] = {src_cbs[0].get_write_ptr(), src_cbs[1].get_write_ptr()};
        const uint32_t dst_base = dst_cb.get_write_ptr();

        AllocatorBank<AllocatorBankType::DRAM> dram_bank;

        const uint32_t dst_bank_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_bank_base = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t num_src = get_arg_val<uint32_t>(arg_idx++);

        for (uint32_t s = 0; s < num_src; s++) {
            const uint32_t bank_id = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t bank_addr = get_arg_val<uint32_t>(arg_idx++);
            noc.async_read(
                dram_bank,
                CoreLocalMem<uint32_t>(src_base[s]),
                shard_size,
                {.bank_id = bank_id, .addr = bank_addr},
                {});
        }
        noc.async_read_barrier();

        const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t t = 0; t < num_transfers; t++) {
            const uint32_t src_slot = get_arg_val<uint32_t>(arg_idx++);
            uint32_t src_off = get_arg_val<uint32_t>(arg_idx++);
            uint32_t dst_off = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t copy_sz = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t src_str = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t dst_str = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);
            for (uint32_t row = 0; row < num_rows; row++) {
                local_copy(src_base[src_slot] + src_off, dst_base + dst_off, copy_sz);
                src_off += src_str;
                dst_off += dst_str;
            }
        }

        noc.async_write(
            CoreLocalMem<uint32_t>(dst_base),
            dram_bank,
            shard_size,
            {},
            {.bank_id = dst_bank_id, .addr = dst_bank_base});
        noc.async_write_barrier();

    } else if constexpr (mode == 1) {
        // DRAM mode: source is a DRAM bank, destination is a DRAM bank.
        // Per-core header: dst_bank_id, dst_bank_base_addr.
        AllocatorBank<AllocatorBankType::DRAM> dram_bank;

        const uint32_t dst_bank_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_bank_base = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);

        for (uint32_t t = 0; t < num_transfers; t++) {
            const uint32_t src_bank_id = get_arg_val<uint32_t>(arg_idx++);
            uint32_t src_bank_addr = get_arg_val<uint32_t>(arg_idx++);  // bank_base + intra_shard_offset
            uint32_t dst_offset = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t copy_size = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t src_stride = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t dst_stride = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);

            for (uint32_t row = 0; row < num_rows; row++) {
                // Read tile-sized chunk from DRAM source into L1 scratch.
                noc.async_read(
                    dram_bank,
                    CoreLocalMem<uint32_t>(scratch_base),
                    copy_size,
                    {.bank_id = src_bank_id, .addr = src_bank_addr},
                    {});
                noc.async_read_barrier();

                // Write from L1 scratch to DRAM destination.
                noc.async_write(
                    CoreLocalMem<uint32_t>(scratch_base),
                    dram_bank,
                    copy_size,
                    {},
                    {.bank_id = dst_bank_id, .addr = dst_bank_base + dst_offset});
                noc.async_write_barrier();

                src_bank_addr += src_stride;
                dst_offset += dst_stride;
            }
        }
    } else {
        // L1 mode: source is an L1 shard on another core, destination is this core's L1.
        CircularBuffer output_cb(output_cb_id);
        const uint32_t dst_base = output_cb.get_write_ptr();
        const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);

        for (uint32_t t = 0; t < num_transfers; t++) {
            const uint32_t src_noc_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t src_noc_y = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t src_cb_id = get_arg_val<uint32_t>(arg_idx++);
            uint32_t src_l1_offset = get_arg_val<uint32_t>(arg_idx++);
            uint32_t dst_offset = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t copy_size = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t src_stride = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t dst_stride = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);

            CircularBuffer src_cb(src_cb_id);
            const uint32_t src_cb_base = src_cb.get_read_ptr();

            // Double-buffered pipeline: prefetch row+1 while copying row.
            auto issue_read = [&](uint32_t off, uint32_t half_base) -> uint32_t {
                const uint32_t pre_pad = off & (alignment - 1);
                uint32_t read_size = pre_pad + copy_size;
                read_size = (read_size + alignment - 1) & ~(alignment - 1);
                noc.async_read(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(half_base),
                    read_size,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = src_cb_base + (off - pre_pad)},
                    {});
                return pre_pad;
            };

            uint32_t buf = 0;
            uint32_t pre = issue_read(src_l1_offset, scratch_base);
            noc.async_read_barrier();
            for (uint32_t row = 0; row < num_rows; row++) {
                const uint32_t cur_half = scratch_base + buf * scratch_half;
                const bool has_next = (row + 1) < num_rows;
                const uint32_t next_buf = buf ^ 1u;
                uint32_t next_pre = 0;
                if (has_next) {
                    next_pre = issue_read(src_l1_offset + src_stride, scratch_base + next_buf * scratch_half);
                }
                local_copy(cur_half + pre, dst_base + dst_offset, copy_size);
                if (has_next) {
                    noc.async_read_barrier();
                }
                src_l1_offset += src_stride;
                dst_offset += dst_stride;
                pre = next_pre;
                buf = next_buf;
            }
        }
    }
}
