// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

namespace dataflow_kernel_lib {

/**
 * Generate one index tile for the sorting ops (top-k, sort, moe, sampling).
 *
 * The tile holds, for every element of the corresponding width tile, that element's position
 * along the width: the tile at width position `wt` covers indices [wt*32, wt*32+31], repeated
 * down all 32 rows. The ops carry it alongside the values through the sort network so that the
 * original position of each value survives the sort.
 *
 * The 32x32 tile is four 16x16 faces stored contiguously: top-left, top-right, bottom-left,
 * bottom-right. The index depends only on the column, so all 16 rows of a face are identical
 * and the two lower faces repeat the two upper ones. Only 32 of the 1024 entries are distinct.
 *
 * Rather than store all 1024 entries with the RISC, seed two rows of each upper face and let the
 * NoC replicate them. Per tile that is 32 stores of a 32-bit word holding two indices, and 15 NoC
 * read commands, against 1024 16-bit stores. This is the same construction the DeepSeek gate
 * writers already use
 * (experimental/reduction/deepseek_grouped_gate, experimental/deepseek_prefill/moe_grouped_topk).
 *
 * T selects the index width, which must match the top-k LLK's dest mode: uint16_t for the
 * UInt16 index tile (LO16 dest, WH/BH), uint32_t for the Int32 index tile (INT32 dest, Quasar).
 *
 * @param dfb_id Dataflow buffer to write the generated index tile to
 * @param wt     Width tile position [0, Wt)
 */
template <typename T = uint16_t>
FORCE_INLINE void generate_index_tile(const uint32_t dfb_id, const uint32_t wt) {
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_faces = 2;
    constexpr uint32_t face_size = 16;

    DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(one_tile);

    const uint32_t tile_addr = dfb.get_write_ptr();
    const uint32_t w = wt << 5;  // wt * 32

    if constexpr (sizeof(T) == 2) {
        constexpr uint32_t face_line_bytes = face_size * sizeof(T);         // 32
        constexpr uint32_t face_bytes = face_size * face_size * sizeof(T);  // 512

        // How many rows of each upper face the RISC seeds before the NoC replicates them. Per tile
        // that is tile_faces * seed_rows * (face_size / 2) stores and
        // tile_faces * (face_size / seed_rows - 1) + 1 NoC read commands, so a second seed row
        // trades 16 more stores for 16 fewer reads: 32 stores and 15 reads here, against 16 and 31
        // at one row. Two is the measured optimum on `ttnn.sort [1,1,32,8192]`, the shape where
        // this cost is most exposed: 1 row 839 us, 2 rows 805 us, 4 rows 839 us, 8 rows 925 us.
        // The reads are asynchronous and overlap, so those four points are the evidence rather
        // than a per-instruction cost model.
        //
        // The optimum depends on what a store costs against a NoC read command, which is a property
        // of the machine. One to four rows sit inside a 4% band on the numbers above, so a part
        // with a different ratio pays at most that much until someone re-measures; eight rows is
        // where the trade clearly turns. Only this constant needs to change.
        constexpr uint32_t seed_rows = 2;
        constexpr uint32_t seed_words = seed_rows * face_size / 2;
        constexpr uint32_t seed_bytes = seed_rows * face_line_bytes;

        for (uint32_t j = 0; j < tile_faces; ++j) {
            // Seed rows of the face: two 16-bit indices per 32-bit word, every row identical.
            const uint32_t face_addr = tile_addr + j * face_bytes;
            volatile tt_l1_ptr uint32_t* seed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(face_addr);
            for (uint32_t r = 0; r < seed_rows; ++r) {
                uint32_t value = w + face_size * j;
                for (uint32_t m = 0; m < face_size / 2; ++m) {
                    seed[r * (face_size / 2) + m] = ((value + 1) << 16) | value;
                    value += 2;
                }
            }
            // A baby-RISCV store can retire before its write lands in L1, and the RISCV core and
            // the NoC are different L1 clients with no program-order guarantee between them. Read
            // the last word back — a blocking load — so the seed is in L1 before the NoC reads it.
            const uint32_t last_word = seed[seed_words - 1];
            asm volatile("" ::"r"(last_word) : "memory");

            // Replicate the seed down the remaining rows of the face.
            const uint64_t seed_noc_addr = get_noc_addr(face_addr);
            uint32_t dst_addr = face_addr + seed_bytes;
            for (uint32_t k = seed_rows; k < face_size; k += seed_rows) {
                noc_async_read(seed_noc_addr, dst_addr, seed_bytes);
                dst_addr += seed_bytes;
            }
        }
        noc_async_read_barrier();

        // The two lower faces are copies of the two upper ones.
        noc_async_read(get_noc_addr(tile_addr), tile_addr + tile_faces * face_bytes, tile_faces * face_bytes);
        noc_async_read_barrier();

        dfb.push_back(one_tile);
        return;
    }

    volatile tt_l1_ptr T* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(tile_addr);
    uint32_t count = 0;
    for (uint32_t i = 0; i < tile_faces; ++i) {
        for (uint32_t j = 0; j < tile_faces; ++j) {
            for (uint32_t k = 0; k < face_size; ++k) {
                for (uint32_t l = 0; l < face_size; ++l) {
                    ptr[count] = l + face_size * j + w;
                    count++;
                }
            }
        }
    }
    dfb.push_back(one_tile);
}

}  // namespace dataflow_kernel_lib
