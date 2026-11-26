// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// staged writer (DRAM -> DRAM only, BRISC/NoC1): drains cb_staged_pages block by block to the target
// placement. Per block only the data must have left L1 before the slot is freed; the acks are waited
// once, before return (BB-03).
//
// Raw NoC calls, no kernel_lib helper: see g6_to_memory_config_move.hpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/ttnn/operations/g6_to_memory_config/kernels/g6_to_memory_config_move.hpp"

void kernel_main() {
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(0);
    constexpr uint32_t staged_block_pages = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    const auto dst = TensorAccessor(dst_args, dst_base);
    const uint32_t page_size = dst.get_aligned_page_size();

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = (num_pages + staged_block_pages - 1) / staged_block_pages;

    uint32_t page = start_page;
    uint32_t remaining = num_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_pages = remaining < staged_block_pages ? remaining : staged_block_pages;
        {
            DeviceZoneScopedN("G6-STAGED-WAIT");
            cb_wait_front(cb_staged_pages, staged_block_pages);
        }
        {
            DeviceZoneScopedN("G6-STAGED-WRITE");
            g6_to_memory_config::move_page_range<false>(
                dst, page, block_pages, get_read_ptr(cb_staged_pages), page_size);
        }
        {
            DeviceZoneScopedN("G6-STAGED-FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_staged_pages, staged_block_pages);
        page += block_pages;
        remaining -= block_pages;
    }

    DeviceZoneScopedN("G6-STAGED-DRAIN");
    noc_async_write_barrier();
}
