// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t dfb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_ctl_id = tt::CBIndex::c_1;     // control: this_core_blocks -> compute/writer
    constexpr uint32_t cb_region_id = tt::CBIndex::c_2;  // scratch for expert_region_offsets page
    constexpr uint32_t cb_counts_id = tt::CBIndex::c_3;  // scratch for total_counts_per_expert page
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_page_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t num_pages_in_row = get_compile_time_arg_val(1);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_compile_time_arg_val(2);

    constexpr auto src_tensor_args = TensorAccessorArgs<3>();
    constexpr uint32_t after_src = src_tensor_args.next_compile_time_args_offset();
    constexpr bool region_aware = get_compile_time_arg_val(after_src) != 0;
    constexpr uint32_t num_experts = get_compile_time_arg_val(after_src + 1);

    const auto s = TensorAccessor(src_tensor_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

    // Bound this core's block count by the filled prefix of the padded dispatch buffer. valid_blocks is the
    // GLOBAL-max fill across all experts: since region offsets restart per destination device, max_e(region[e]
    // + align32(count[e])) == the fullest device's fill; the tilize is a mesh op bounded by the slowest device,
    // so using that global max on every core matches an ideal per-device skip and needs no device index.
    uint32_t core_blocks = num_rows / tile_height;
    if constexpr (region_aware) {
        constexpr auto region_args = TensorAccessorArgs<after_src + 2>();
        constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
        const uint32_t region_addr = get_arg_val<uint32_t>(9);
        const uint32_t counts_addr = get_arg_val<uint32_t>(10);
        const auto region_acc = TensorAccessor(region_args, region_addr);
        const auto counts_acc = TensorAccessor(counts_args, counts_addr);

        // Both routing tensors are [1, num_experts] uint32 -> a single page each.
        const uint32_t bytes = num_experts * 4;
        DataflowBuffer dfb_region(cb_region_id);
        DataflowBuffer dfb_counts(cb_counts_id);
        dfb_region.reserve_back(1);
        dfb_counts.reserve_back(1);
        CoreLocalMem<uint32_t> region_dst(dfb_region.get_write_ptr());
        CoreLocalMem<uint32_t> counts_dst(dfb_counts.get_write_ptr());
        noc.async_read(region_acc, region_dst, bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(counts_acc, counts_dst, bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();

        volatile tt_l1_ptr uint32_t* region = (volatile tt_l1_ptr uint32_t*)dfb_region.get_write_ptr();
        volatile tt_l1_ptr uint32_t* counts = (volatile tt_l1_ptr uint32_t*)dfb_counts.get_write_ptr();
        uint32_t valid_rows = 0;
        for (uint32_t e = 0; e < num_experts; e++) {
            const uint32_t end = region[e] + ((counts[e] + tile_height - 1) / tile_height) * tile_height;
            if (end > valid_rows) {
                valid_rows = end;
            }
        }
        const uint32_t valid_blocks = valid_rows / tile_height;  // valid_rows is tile-aligned
        const uint32_t start_block = start_page_id / tile_height;
        uint32_t bound = (valid_blocks > start_block) ? (valid_blocks - start_block) : 0;
        if (bound > core_blocks) {
            bound = core_blocks;
        }
        core_blocks = bound;

        // Publish this core's block count to compute + writer.
        DataflowBuffer dfb_ctl(cb_ctl_id);
        dfb_ctl.reserve_back(1);
        volatile tt_l1_ptr uint32_t* ctl = (volatile tt_l1_ptr uint32_t*)dfb_ctl.get_write_ptr();
        ctl[0] = core_blocks;
        dfb_ctl.push_back(1);
    }

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        dfb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(s, dst, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
                page_id++;
                l1_write_addr += width_size;
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < core_blocks; i++) {
        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, page_id);
        }
        page_id += tile_height * num_pages_in_row;
    }
}
