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
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

#ifdef ARCH_QUASAR
// Quasar emulator self-loopback workaround: a NoC transfer whose source and destination are both this
// core's L1 (a write-back whose storage core is this core) is done with the shared RISC copy helper, which also issues
// the fence that orders the uncached-alias stores ahead of the semaphore / credit that publishes them. The helper takes
// uncached-alias addresses (what the DFB getters hand out on Quasar DM); a LocalTensorAccessor base is the
// plain (cached) L1 address, so normalise both operands first.
#include "ttnn/operations/kernel_helper_functions/local_l1_copy.hpp"
#include "internal/local_mem/address_access.h"
FORCE_INLINE uint32_t l1_uncached(uint32_t addr) {
    return addr >= MEM_L1_UNCACHED_BASE ? addr : MEM_L1_UNCACHED_BASE + (addr - MEM_L1_BASE);
}
#endif

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }

    // in1 / bias are DRAM-sharded: read by bank (this reader's dram_bank_id) at offsets into each tensor.
    const auto in1_accessor = TensorAccessor(tensor::in1);
#ifdef FUSE_BIAS
    const auto bias_accessor = TensorAccessor(tensor::bias);
#endif
    const uint32_t dram_bank_id = get_arg(args::dram_bank_id);
    const uint32_t vc = get_arg(args::vc);
    const uint32_t dram_reader_index = get_arg(args::dram_reader_index);
    const uint32_t num_shard_to_write_back = get_arg(args::num_shard_to_write_back);
    const uint32_t reshard_tensor_start_offset = get_arg(args::reshard_tensor_start_offset);

    // COMPILE TIME ARGS
    constexpr auto in1_page_size = get_arg(args::in1_page_size);
    constexpr auto in1_num_pages = get_arg(args::in1_num_pages);
    // in1 block args
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr auto num_blocks = get_arg(args::num_blocks);
    // WRITER
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);
    constexpr auto out_tensor_stride_w_bytes = get_arg(args::out_tensor_stride_w_bytes);
    constexpr auto out_reshard_tensor_stride_w_bytes = get_arg(args::out_reshard_tensor_stride_w_bytes);
    constexpr auto per_core_M = get_arg(args::per_core_M);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_tile_bytes = get_tile_size(dfb::out);
    constexpr uint32_t out_row_tiles = out_tensor_stride_w_bytes / out_tile_bytes;
    constexpr auto workers_per_bank = get_arg(args::workers_per_bank);
    constexpr auto bank_row_stride_tiles = get_arg(args::bank_row_stride_tiles);
    constexpr auto reader_width_tiles = get_arg(args::reader_width_tiles);

#ifdef FUSE_BIAS
    constexpr auto in3_page_size = get_arg(args::in3_page_size);
    constexpr auto in3_num_pages = get_arg(args::in3_num_pages);
#endif

    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the
    // in1 buffer's entries are sized to match, so the block size in L1 must use the padded page
    // stride (in1_num_pages * in1_page_size) rather than get_tile_size() (the unpadded tile size).
    constexpr uint32_t in1_block_size_bytes = in1_num_pages * in1_page_size;
#ifdef SPLIT_DRAM_BANK
    static_assert(workers_per_bank > 1);
    constexpr uint32_t tiles_per_k_row = reader_width_tiles;
    constexpr uint32_t k_rows_per_block = in1_block_num_tiles / tiles_per_k_row;
    constexpr uint32_t in1_tile_size_bytes = in1_block_size_bytes / in1_block_num_tiles;
    const uint32_t shard_column_offset_tiles = dram_reader_index * tiles_per_k_row;
#endif

    const Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    // The output tensor's own L1 shard: only its base address is needed (no FIFO traffic).
    const LocalTensorAccessor<uint32_t> out_shard(tensor::out_shard);
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in3(dfb::bias);
#endif

    //  READER
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;

    const BankView dram_bank(in1_accessor);
    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        dram_bank, in1_page_size, {.bank_id = dram_bank_id, .offset_bytes = 0}, NocOptVals{.vc = vc});

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
                {.bank_id = dram_bank_id, .offset_bytes = l1_read_addr_in1},
                {},
                NocOptVals{.vc = vc});
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        noc.async_read_barrier();
        dfb_in1.push_back(in1_block_num_tiles);
    }
#else
    constexpr uint32_t total_num_blocks_in_buffer = get_arg(args::in1_buffer_blocks);
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    // Reserve 2 blocks up front when num_blocks > 1: the loop's reserve_back
    // only fires after block 1's writes complete, so the initial reservation must cover
    // both block 0 and block 1 to avoid writing outside the reserved buffer window.
    // When num_blocks == 1 the buffer is single-buffered, so reserve only 1 block.
    constexpr uint32_t initial_reserved_blocks = (num_blocks > 1) ? 2 : 1;
    dfb_in1.reserve_back(in1_block_num_tiles * initial_reserved_blocks);
    uint32_t l1_write_addr_in1_offset = 0;
    const uint32_t l1_write_addr_in1_start = dfb_in1.get_write_ptr();
    l1_write_addr_in1 = l1_write_addr_in1_start;
    for (uint32_t block = 0; block < num_blocks; ++block) {
#ifdef SPLIT_DRAM_BANK
        for (uint32_t row = 0; row < k_rows_per_block; ++row) {
            const uint32_t source_tile =
                (block * k_rows_per_block + row) * bank_row_stride_tiles + shard_column_offset_tiles;
            const uint32_t read_size = tiles_per_k_row * in1_tile_size_bytes;
            const uint32_t read_offset = source_tile * in1_tile_size_bytes;
            // A split reader's row may exceed one NoC burst (e.g. the LM
            // head). Use its real compile-time bound so the read helper
            // packetizes it and accounts for every response under this TRID.
            noc.async_read<NocOptions::TXN_ID, tiles_per_k_row * in1_tile_size_bytes>(
                dram_bank,
                CoreLocalMem<uint32_t>(l1_write_addr_in1),
                read_size,
                {.bank_id = dram_bank_id, .offset_bytes = read_offset},
                {},
                NocOptVals{.vc = vc, .trid = curr_block_trid});
            l1_write_addr_in1 += read_size;
        }
#else
        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
                dram_bank,
                CoreLocalMem<uint32_t>(l1_write_addr_in1),
                in1_page_size,
                {.bank_id = dram_bank_id, .offset_bytes = l1_read_addr_in1},
                {},
                NocOptVals{.vc = vc, .trid = curr_block_trid});
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }
#endif

        if (block > 0) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(block_trid_to_wait)});
            dfb_in1.push_back(in1_block_num_tiles);
            // wait for next block trid
            block_trid_to_wait = block_trid_to_wait == total_num_blocks_in_buffer ? 1 : (block_trid_to_wait + 1);
            // reserve for next block
            dfb_in1.reserve_back(in1_block_num_tiles * 2);
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
    // The next read's tag setup fences each earlier block's submission. The last
    // block has no successor, so explicitly fence it before sampling its count.
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(block_trid_to_wait)});
    dfb_in1.push_back(in1_block_num_tiles);
#endif

#ifdef FUSE_BIAS
    // Operand 1
    dfb_in3.reserve_back(in1_block_w);
    uint32_t l1_write_addr_in3 = dfb_in3.get_write_ptr();
    uint32_t l1_read_addr_in3 = 0;
#ifdef SPLIT_DRAM_BANK
    constexpr uint32_t in3_tile_size_bytes = (in3_num_pages * in3_page_size) / reader_width_tiles;
    l1_read_addr_in3 = dram_reader_index * reader_width_tiles * in3_tile_size_bytes;
#endif
    const BankView bias_bank(bias_accessor);
    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        bias_bank, in3_page_size, {.bank_id = dram_bank_id, .offset_bytes = 0}, NocOptVals{.vc = vc});

    for (uint32_t h = 0; h < in3_num_pages; ++h) {
        noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            bias_bank,
            CoreLocalMem<uint32_t>(l1_write_addr_in3),
            in3_page_size,
            {.bank_id = dram_bank_id, .offset_bytes = l1_read_addr_in3},
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
        // Raw address needed: the shard sits at the same L1 address on every core, so this core's base is the
        // remote write address on the storage core (and a local_l1_copy operand on Quasar).
        uint32_t l1_write_addr_out_reshard =
            static_cast<uint32_t>(local_mem::detail::AddressAccess::address(out_shard.local_mem()));

        if (i == 0) {
            l1_write_addr_out_reshard += reshard_tensor_start_offset;
        }

        // This shard's triple out of the write-back block.
        const uint32_t per_core_N_reshard_bytes = get_vararg(index_offset);
        const uint32_t in0_mcast_sender_noc_x = get_vararg(index_offset + 1);
        const uint32_t in0_mcast_sender_noc_y = get_vararg(index_offset + 2);

        const UnicastEndpoint dst_ep;
        uint32_t reshard_dest_local_addr = l1_write_addr_out_reshard;
#ifdef ARCH_QUASAR
        // The storage core for this shard is this core: copy with the RISC (see local_l1_copy).
        const bool reshard_is_local = noc.is_local_bank(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y);
#endif

        for (uint32_t h = 0; h < per_core_M; ++h) {
            // Compute packs subblocks, while tensor shards store complete tile rows.
            // A one-row or full-width subblock is already contiguous; otherwise split
            // each destination row at subblock boundaries without an L1 reorder copy.
            constexpr bool contiguous_rows = out_subblock_h == 1 || out_subblock_w == out_row_tiles;
            uint32_t written = 0;
            while (written < per_core_N_reshard_bytes) {
                uint32_t source = l1_read_addr_out;
                uint32_t bytes = per_core_N_reshard_bytes;
                if constexpr (!contiguous_rows) {
                    const uint32_t col = (l1_read_addr_out_offset + written) / out_tile_bytes;
                    const uint32_t tile = (h / out_subblock_h) * out_subblock_h * out_row_tiles +
                                          (col / out_subblock_w) * out_subblock_h * out_subblock_w +
                                          (h % out_subblock_h) * out_subblock_w + col % out_subblock_w;
                    source = dfb_out.get_read_ptr() + tile * out_tile_bytes;
                    const uint32_t subblock_bytes = (out_subblock_w - col % out_subblock_w) * out_tile_bytes;
                    const uint32_t remaining = per_core_N_reshard_bytes - written;
                    bytes = subblock_bytes < remaining ? subblock_bytes : remaining;
                }
#ifdef ARCH_QUASAR
                if (reshard_is_local) {
                    local_l1_copy(l1_uncached(reshard_dest_local_addr + written), l1_uncached(source), bytes);
                } else
#endif
                {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(source),
                        dst_ep,
                        bytes,
                        {},
                        {.noc_x = in0_mcast_sender_noc_x,
                         .noc_y = in0_mcast_sender_noc_y,
                         .addr = reshard_dest_local_addr + written});
                }
                written += bytes;
            }
            l1_read_addr_out += out_tensor_stride_w_bytes;
            reshard_dest_local_addr += out_reshard_tensor_stride_w_bytes;
        }
        l1_read_addr_out_offset += per_core_N_reshard_bytes;

        index_offset += 3;
    }
    noc.async_write_barrier();
#endif

    dfb_out.pop_front(out_block_num_tiles);

#ifdef ARCH_QUASAR
    // Quasar's command buffer has no NOC_CTRL (it's split into NOC_CTRL_LO/HI, and under NOC API v2/v3 reads use the
    // RoCC buffer), so restore the firmware-default VC (1) through the NOC API. Unlike the path below, this also
    // rewrites the read address/length state; the NOC API has no VC-only restore yet.
    noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
        dram_bank, in1_page_size, {.bank_id = dram_bank_id, .addr = in1_tensor_addr}, NocOptVals{.vc = 1});
#else
    // Restore only NCRISC read_cmd_buf NOC_CTRL VC to firmware default (VC=1, set in noc_init); a full
    // set_async_read_state would also clobber NOC_TARG_ADDR_* / NOC_AT_LEN_BE sticky regs that the next kernel's
    // async_read_*_with_state fast paths read.
    const uint32_t noc_rd_cmd_field =
        NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
#endif
}
