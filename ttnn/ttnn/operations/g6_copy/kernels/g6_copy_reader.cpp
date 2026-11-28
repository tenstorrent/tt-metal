// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// load_block: source DRAM -> cb_page_staging, one block of block_pages pages at a time.
// Raw dataflow by design: no kernel-lib helper reads multi-burst DRAM bank runs into a CB.
//
// read_trids (common RT arg) is the read-pipeline depth: how many blocks of reads stay on the wire.
//   1  — issue a block, drain-everything noc_async_read_barrier(), push (the Phase 0 / R2 loop).
//   T>1 — block k carries transaction id (k % T) + 1 (kernel-lib set_read_trid); once T blocks are
//         outstanding the OLDEST is retired with async_read_barrier_with_trid and pushed, so block
//         k+1..k+T-1 keep streaming across block k's hand-off. Block k and k-T share an id, so one
//         wait both retires the old block and frees the id.
//   The CB write pointer advances only on push, so it lags the in-flight blocks: a block lands at
//   write_ptr + in_flight slots, wrapped by hand inside the ring, and the reserve covers every slot
//   the reader holds (in_flight + 1 blocks) so no landing slot is still owned by the writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_copy_stage_zones.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_copy_page_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);
    const uint32_t read_trids = get_common_arg_val<uint32_t>(2);
    const uint32_t cb_depth_blocks = get_common_arg_val<uint32_t>(3);
    const uint32_t bank_major = get_common_arg_val<uint32_t>(4);
    const uint32_t tensor_pages = get_common_arg_val<uint32_t>(5);

    const uint32_t core_first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const uint32_t walk_rotation = get_arg_val<uint32_t>(2);

    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;

    if (read_trids <= 1) {
        uint32_t block_first = core_first_page;
        uint32_t pages_left = core_pages;
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t block_len = pages_left < block_pages ? pages_left : block_pages;
            // The nominal block_pages is reserved and pushed even on a short last block, so the ring
            // (capacity cb_depth_blocks * block_pages) never wraps inside a block.
            {
                G6C_STAGE_ZONE("G6C-RD-RESERVE");
                cb_reserve_back(cb_page_staging, block_pages);
            }
            {
                G6C_STAGE_ZONE("G6C-RD-ISSUE");
                g6_copy::walk_block<true>(
                    src,
                    block_first,
                    block_len,
                    walk_rotation,
                    bank_major,
                    tensor_pages,
                    get_write_ptr(cb_page_staging));
            }
            {
                G6C_STAGE_ZONE("G6C-RD-BARRIER");
                noc_async_read_barrier();
            }
            {
                G6C_STAGE_ZONE("G6C-RD-PUSH");
                cb_push_back(cb_page_staging, block_pages);
            }
            block_first += block_len;
            pages_left -= block_len;
        }
        return;
    }

    // Trid pipeline. Nothing has been pushed yet, so the write pointer is the ring's base.
    Noc noc;
    const uint32_t block_bytes = block_pages * src.get_aligned_page_size();
    const uint32_t ring_base = get_write_ptr(cb_page_staging);
    const uint32_t ring_bytes = cb_depth_blocks * block_bytes;

    uint32_t block_first = core_first_page;
    uint32_t pages_left = core_pages;
    for (uint32_t k = 0; k < num_blocks; ++k) {
        const uint32_t trid = (k % read_trids) + 1;
        if (k >= read_trids) {
            G6C_STAGE_ZONE("G6C-RD-RETIRE");
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);  // block k - read_trids
            cb_push_back(cb_page_staging, block_pages);
        }
        const uint32_t in_flight = k < read_trids - 1 ? k : read_trids - 1;
        {
            G6C_STAGE_ZONE("G6C-RD-RESERVE");
            cb_reserve_back(cb_page_staging, (in_flight + 1) * block_pages);
        }
        uint32_t slot = get_write_ptr(cb_page_staging) - ring_base + in_flight * block_bytes;
        if (slot >= ring_bytes) {
            slot -= ring_bytes;
        }
        const uint32_t block_len = pages_left < block_pages ? pages_left : block_pages;
        {
            G6C_STAGE_ZONE("G6C-RD-ISSUE");
            dataflow_kernel_lib::set_read_trid(noc, trid);
            g6_copy::walk_block<true>(
                src, block_first, block_len, walk_rotation, bank_major, tensor_pages, ring_base + slot);
        }
        block_first += block_len;
        pages_left -= block_len;
    }
    // Drain the outstanding blocks oldest first, in issue order.
    const uint32_t outstanding = num_blocks < read_trids ? num_blocks : read_trids;
    for (uint32_t k = num_blocks - outstanding; k < num_blocks; ++k) {
        G6C_STAGE_ZONE("G6C-RD-RETIRE");
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, (k % read_trids) + 1);
        cb_push_back(cb_page_staging, block_pages);
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);
}
