// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }

    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
#ifdef FUSE_BIAS
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(2);
#endif
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t num_shard_to_write_back = get_arg_val<uint32_t>(5);
    const uint32_t reshard_tensor_start_offset = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* per_core_N_reshard_bytes = (tt_l1_ptr uint32_t*)(get_arg_addr(7));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(8));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(9));

    // COMPILE TIME ARGS
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(1);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    // WRITER
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_tensor_stride_w_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_reshard_tensor_stride_w_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t in3_num_pages = get_compile_time_arg_val(10);
    constexpr uint32_t dfb_id_in3 = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(dfb_id_in3);
    constexpr DataFormat bias_data_format = get_dataformat(dfb_id_in3);
#endif

    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("dfb_in1");
    constexpr uint32_t dfb_id_out = get_named_compile_time_arg_val("dfb_out");
    constexpr uint32_t dfb_id_out_reshard = get_named_compile_time_arg_val("dfb_out_reshard");
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the
    // in1 CB pages are sized to match, so the block size in L1 must use the padded page stride
    // (in1_num_pages * in1_page_size) rather than get_tile_size() (the unpadded tile size).
    constexpr uint32_t in1_block_size_bytes = in1_num_pages * in1_page_size;

    Noc noc;
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_out(dfb_id_out);
    DataflowBuffer dfb_out_reshard(dfb_id_out_reshard);
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in3(dfb_id_in3);
#endif

    //  READER
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;
    constexpr DataFormat in1_data_format = get_dataformat(dfb_id_in1);

    AllocatorBank<AllocatorBankType::DRAM> dram_bank;
    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        dram_bank, in1_page_size, {.bank_id = dram_bank_id, .addr = in1_tensor_addr}, NocOptVals{.vc = vc});

#ifdef ARCH_GRAYSKULL
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Operand 1
        dfb_in1.reserve_back(in1_block_num_tiles);
        l1_write_addr_in1 = dfb_in1.get_write_ptr();

        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                dram_bank,
                CoreLocalMem<uint32_t>(l1_write_addr_in1),
                in1_page_size,
                {.bank_id = dram_bank_id, .addr = in1_tensor_addr + l1_read_addr_in1},
                {},
                NocOptVals{.vc = vc});
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        noc.async_read_barrier();
        dfb_in1.push_back(in1_block_num_tiles);
    }
#else
    constexpr uint32_t total_num_blocks_in_buffer = 3;
    constexpr uint32_t total_num_trid = 4;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    // Reserve 2 blocks up front when num_blocks > 1: the loop's reserve_back at line 133
    // only fires after block 1's writes complete, so the initial reservation must cover
    // both block 0 and block 1 to avoid writing outside the reserved CB window.
    // When num_blocks == 1 the CB is single-buffered, so reserve only 1 block.
    constexpr uint32_t initial_reserved_blocks = (num_blocks > 1) ? 2 : 1;
    dfb_in1.reserve_back(in1_block_num_tiles * initial_reserved_blocks);
    uint32_t l1_write_addr_in1_offset = 0;
    uint32_t l1_write_addr_in1_start = dfb_in1.get_write_ptr();
    l1_write_addr_in1 = l1_write_addr_in1_start;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
                dram_bank,
                CoreLocalMem<uint32_t>(l1_write_addr_in1),
                in1_page_size,
                {.bank_id = dram_bank_id, .addr = in1_tensor_addr + l1_read_addr_in1},
                {},
                NocOptVals{.vc = vc, .trid = curr_block_trid});
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        if (num_free_blocks_in_buffer == 2) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(block_trid_to_wait)});
            dfb_in1.push_back(in1_block_num_tiles);
            // wait for next block trid
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
            // reserve for next block
            dfb_in1.reserve_back(in1_block_num_tiles * 2);
        } else {
            num_free_blocks_in_buffer -= 1;
        }

        if (curr_block_trid == total_num_blocks_in_buffer) {
            l1_write_addr_in1_offset = 0;
            curr_block_trid = 1;
        } else {
            l1_write_addr_in1_offset += in1_block_size_bytes;
            curr_block_trid += 1;
        }
        l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
    }
    // last block to wait
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(block_trid_to_wait)});
    dfb_in1.push_back(in1_block_num_tiles);
#endif

#ifdef FUSE_BIAS
    // Operand 1
    dfb_in3.reserve_back(in1_block_w);
    uint32_t l1_write_addr_in3 = dfb_in3.get_write_ptr();
    uint32_t l1_read_addr_in3 = 0;

    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        dram_bank, in3_page_size, {.bank_id = dram_bank_id, .addr = in3_tensor_addr}, NocOptVals{.vc = vc});

    for (uint32_t h = 0; h < in3_num_pages; ++h) {
        noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            dram_bank,
            CoreLocalMem<uint32_t>(l1_write_addr_in3),
            in3_page_size,
            {.bank_id = dram_bank_id, .addr = in3_tensor_addr + l1_read_addr_in3},
            {},
            NocOptVals{.vc = vc});
        l1_read_addr_in3 += in3_page_size;
        l1_write_addr_in3 += in3_page_size;
    }

    // Barrier! make sure the reads are done
    noc.async_read_barrier();
    dfb_in3.push_back(in1_block_w);
#endif

    // WRITER
    dfb_out.wait_front(out_block_num_tiles);

#ifndef SKIP_WRITE_BACK
    uint32_t index_offset = 0;
    uint32_t l1_read_addr_out_offset = 0;

    for (uint32_t i = 0; i < num_shard_to_write_back; ++i) {
        uint32_t l1_read_addr_out = dfb_out.get_read_ptr() + l1_read_addr_out_offset;
        uint32_t l1_write_addr_out_reshard = dfb_out_reshard.get_write_ptr();

        if (i == 0) {
            l1_write_addr_out_reshard += reshard_tensor_start_offset;
        }

        UnicastEndpoint dst_ep;
        uint32_t reshard_dest_local_addr = l1_write_addr_out_reshard;

        for (uint32_t h = 0; h < per_core_M; ++h) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr_out),
                dst_ep,
                per_core_N_reshard_bytes[index_offset],
                {},
                {.noc_x = in0_mcast_sender_noc_x[index_offset],
                 .noc_y = in0_mcast_sender_noc_y[index_offset],
                 .addr = reshard_dest_local_addr});
            l1_read_addr_out += out_tensor_stride_w_bytes;
            reshard_dest_local_addr += out_reshard_tensor_stride_w_bytes;
        }
        l1_read_addr_out_offset += per_core_N_reshard_bytes[index_offset];

        index_offset += 3;
    }
    noc.async_write_barrier();
#endif

    dfb_out.pop_front(out_block_num_tiles);

    // Restore NCRISC_RD_CMD_BUF NOC_CTRL to the firmware default (VC=1, set in
    // noc_init). set_async_read_state<NocOptions::CUSTOM_VC> writes a per-bank VC into NOC_CTRL
    // and this hardware register persists across kernel launches. Kernels that
    // follow (e.g. 1d-multicast matmul readers running in DM_DEDICATED_NOC mode)
    // rely on NOC_CTRL being at its initialized value and do not re-set it, so
    // they inherit the stale custom VC and suffer reduced DRAM bandwidth.
    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        dram_bank, in1_page_size, {.bank_id = dram_bank_id, .addr = in1_tensor_addr}, NocOptVals{.vc = 1});
}
