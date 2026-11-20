// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x_injector_reader: reads every active expert's token rows of x from DRAM into the tilizer's row-major CB
// Runs on: x helper relays (p.relays[NR..], one per gate/up rectangle), RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - reads x rows (dispatch buffer, or rows token_index[r] of x: flat_options::x_indexed) from DRAM over NoC 1, every
//    NR-th super-block from its offset; rows past an expert's token count are skipped
//  - writes 32-row x super-block-wide chunks into the row-major CB (x_row_major_cb) and its super-block count into CB 6
//  - syncs only through those CBs with x_injector_tilize_compute; the tiles go on via x_injector_helper_writer
// Note: x reads on NoC 1, the relay's sends (x_injector_helper_writer) on NoC 0: one NoC per RISC (dedicated-NoC
//       mode). NoC 1 is also the fabric routers' NoC that wrote the dispatch buffer.

// Flat expert x injector reader (an x helper relay's BRISC). Extracts every expert's tokens from the row-major bf16
// dispatch buffer (one row per page, interleaved) and stages them for tilizing: for each virtual expert (a sub-block of
// MT row tiles of expert e), each super-block of 32 K tiles (1024 columns) and each row tile m, one chunk of 32 rows x
// 2 KB goes into the row-major CB (32 pages of 2 KB = 32 rows of 1024 bf16). Rows at or past the expert's token count
// are not read (their tiles are padding: their outputs are never written back). Named CT args: x_row_major_cb (RM_CB),
// x_page_bytes (the token row's bytes / PPR), row_chunks_per_read_barrier (BATCH), token_index_arg (the runtime-arg
// index of the token index address, x_indexed), x_pages_per_row (PPR); flat_options: NUM_EXPERTS = local_experts, MT =
// row_tiles_per_sub_block, NSB = super_blocks_per_sub_block (super-blocks per row)
// x_indexed: flat row r (region space) reads x row token_index[r] (x = the all-gathered tokens rather than a dispatch
// buffer). Each sub-block's MT x 32 indices are read once (one small read + barrier) into this RISC's schedule scratch
// (free after load_schedule).
// RT: 0 dispatch buffer address, 1 STRIDE, 2 OFF (this relay takes super-blocks OFF, OFF + STRIDE, ... in stream
// order),
//     then per expert: region row offset, token count
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "flat_accessors.hpp"
#include "flat_options.hpp"
#include "expert_schedule.hpp"

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    constexpr uint32_t rm_cb = get_named_compile_time_arg_val("x_row_major_cb");
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("x_page_bytes");
    constexpr uint32_t num_e = flat_options::local_experts;
    constexpr uint32_t row_tiles = flat_options::row_tiles_per_sub_block;
    constexpr uint32_t super_blocks_per_sub_block = flat_options::super_blocks_per_sub_block;
    constexpr uint32_t batch = get_named_compile_time_arg_val("row_chunks_per_read_barrier");
    constexpr uint32_t super_block_tiles = flat_options::super_block_tiles;
    constexpr uint32_t segment_bytes =
        super_block_tiles * 64;  // one row of a super-block: super_block_tiles tiles x 32 bf16
    // x page = row_bytes (x_page_bytes) / pages per row (x_pages_per_row): a token row is PPR consecutive pages (PPR 8
    // at H 4096 spreads every row over the 8 DRAM banks); a segment is one read (page >= segment_bytes) or
    // segment_bytes / page reads
    constexpr uint32_t pages_per_row = get_named_compile_time_arg_val("x_pages_per_row");
    constexpr uint32_t pread = row_bytes < segment_bytes ? row_bytes : segment_bytes;
    const auto x_accessor = TensorAccessor(flat_acc::x, moe_block_args::arg(0), row_bytes);
    auto read_seg = [&](uint32_t row, uint32_t j, uint32_t dst) {
        const uint32_t b = j * segment_bytes;
        for (uint32_t q = 0; q < segment_bytes; q += pread) {
            noc_async_read(
                x_accessor.get_noc_addr(row * pages_per_row + (b + q) / row_bytes, (b + q) % row_bytes),
                dst + q,
                pread);
        }
    };
    // x_indexed: one page holds the whole [1, rows] index row: address it as bank 0's page 0 plus a byte offset
    uint64_t idx_base = 0;
    if constexpr (flat_options::x_indexed) {
        constexpr uint32_t token_index_arg = get_named_compile_time_arg_val("token_index_arg");
        idx_base = TensorAccessor(flat_acc::token_index, moe_block_args::arg(token_index_arg), 4).get_noc_addr(0);
    }
    // (MT x 32 words, <= 512 B: within this RISC's schedule scratch)
    const uint32_t idx_l1 = schedule_scratch(ScratchOwner::Brisc);
    uint32_t in_batch = 0, l1 = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(rm_cb, 32 * in_batch);
        in_batch = 0;
    };
    uint32_t stride = moe_block_args::arg(1), soff = moe_block_args::arg(2);
    uint32_t g = 0;  // super-block index in stream order
    // The schedule: the active experts' regions (RT 3.. are the expert_schedule.hpp args, CB 7 this RISC's scratch);
    // the tilizer gets this relay's super-block count in CB 6.
    ExpertSchedule schedule;
    load_schedule<num_e>(schedule, 3, schedule_scratch(ScratchOwner::Brisc));
    {
        const uint32_t total_super_blocks = schedule.total_sub_blocks * super_blocks_per_sub_block;
        cb_reserve_back(tt::CBIndex::c_6, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_6)) =
            total_super_blocks > soff ? (total_super_blocks - soff + stride - 1) / stride : 0;
        cb_push_back(tt::CBIndex::c_6, 1);
    }
    for (uint32_t a = 0; a < schedule.entries; ++a) {
        const uint32_t off = schedule.first_row[a], count = schedule.tokens[a];
        const uint32_t subs = (count + row_tiles * 32 - 1) / (row_tiles * 32);
        for (uint32_t s = 0; s < subs; ++s) {
            volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
            if constexpr (flat_options::x_indexed) {
                bool mine = false;  // does this relay take any of the sub-block's super-blocks
                for (uint32_t j = 0; j < super_blocks_per_sub_block && !mine; ++j) {
                    mine = (g + j) % stride == soff;
                }
                if (mine) {
                    const uint32_t rows =
                        count - s * row_tiles * 32 < row_tiles * 32 ? count - s * row_tiles * 32 : row_tiles * 32;
                    // whole 32-row groups: the region is padded to 32 rows, so the read stays inside it
                    noc_async_read(
                        idx_base + (off + s * row_tiles * 32) * 4,
                        reinterpret_cast<uint32_t>(idx),
                        (rows + 31) / 32 * 128);
                    noc_async_read_barrier();
                }
            }
            for (uint32_t j = 0; j < super_blocks_per_sub_block; ++j, ++g) {
                if (g % stride != soff) {
                    continue;
                }
                for (uint32_t m = 0; m < row_tiles; ++m) {
                    if (in_batch == 0) {
                        cb_reserve_back(rm_cb, 32 * batch);
                        l1 = get_write_ptr(rm_cb);
                    }
                    const uint32_t dst = l1 + in_batch * 32 * segment_bytes;
                    const uint32_t first_row = s * row_tiles * 32 + m * 32;
                    for (uint32_t r = 0; r < 32 && first_row + r < count; ++r) {
                        read_seg(
                            flat_options::x_indexed ? idx[m * 32 + r] : off + first_row + r,
                            j,
                            dst + r * segment_bytes);
                    }
                    if (++in_batch == batch) {
                        flush();
                    }
                }
            }
        }
    }
    if (in_batch) {
        flush();
    }
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
