// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute-core writer (BRISC, NoC1): startup_block (reduce scalers + key-tail mask tile) and store_out_block.
//
// Mask tile (only when logical_n % 32 != 0): column c of every row = 0 if c < valid_cols else -inf (bf16 0xFF80).
// Built without a whole-tile CPU loop: zero the tile over the NoC, write the -inf lanes of row 0 of faces 0/1,
// then replicate that row to the other 15 rows of each face and faces 0/1 to faces 2/3 with local NoC copies.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ring_joint_sdpa_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t Sq = get_compile_time_arg_val(4);
    constexpr uint32_t DVt = get_compile_time_arg_val(7);
    constexpr uint32_t NQ = get_compile_time_arg_val(8);  // unit row space (q_heads_per_unit heads stacked)
    constexpr uint32_t LQ = get_compile_time_arg_val(9);
    constexpr uint32_t C = get_compile_time_arg_val(12);
    constexpr bool causal = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t mask_cols = get_compile_time_arg_val(22);
    constexpr bool has_joint = LQ > 0;
    constexpr auto out_args = TensorAccessorArgs<23>();
    [[maybe_unused]] constexpr auto jout_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t jout_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_idx = get_arg_val<uint32_t>(2);

    // ---- startup_block ----
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    if constexpr (causal) {
        // tile 0: causal diagonal (row i sees cols j <= i; -inf above the diagonal); tile 1: all -inf (dead tile).
        // Zeroed over the NoC; only face 0's 120 upper-triangle lanes and one 16-lane -inf row are CPU-written,
        // everything else is replicated with local NoC copies.
        Noc noc;
        CircularBuffer mask_cb(cb_mask);
        const uint32_t tile_bytes = get_tile_size(cb_mask);
        mask_cb.reserve_back(2);
        const uint32_t base = get_write_ptr(cb_mask);
        noc.async_write_zeros(mask_cb, 2 * tile_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        constexpr uint32_t FACE_BYTES = 16 * 16 * 2;
        constexpr uint32_t ROW_BYTES = 16 * 2;
        volatile tt_l1_ptr uint16_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
        for (uint32_t i = 0; i < 16; ++i) {
            for (uint32_t j = i + 1; j < 16; ++j) {
                f0[i * 16 + j] = 0xFF80;
            }
        }
        const uint32_t f1 = base + FACE_BYTES;  // tile 0 face 1 (rows 0-15, cols 16-31): all -inf
        volatile tt_l1_ptr uint16_t* f1p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(f1);
        for (uint32_t j = 0; j < 16; ++j) {
            f1p[j] = 0xFF80;
        }
        for (uint32_t r = 1; r < 16; ++r) {
            noc_async_read(get_noc_addr(f1), f1 + r * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        noc_async_read(get_noc_addr(base), base + 3 * FACE_BYTES, FACE_BYTES);  // face 3 = face 0 (face 2 zero)
        for (uint32_t fc = 0; fc < 4; ++fc) {                                   // tile 1 = four -inf faces
            noc_async_read(get_noc_addr(f1), base + tile_bytes + fc * FACE_BYTES, FACE_BYTES);
        }
        noc_async_read_barrier();
        mask_cb.push_back(2);
    } else if constexpr (mask_cols != 0) {
        Noc noc;
        CircularBuffer mask_cb(cb_mask);
        const uint32_t tile_bytes = get_tile_size(cb_mask);
        mask_cb.reserve_back(1);
        const uint32_t base = get_write_ptr(cb_mask);
        noc.async_write_zeros(mask_cb, tile_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        // row 0 of face 0 (cols 0..15) and face 1 (cols 16..31): 16 bf16 each, 2 per word
        constexpr uint32_t FACE_BYTES = 16 * 16 * 2;
        constexpr uint32_t ROW_BYTES = 16 * 2;
        volatile tt_l1_ptr uint16_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
        volatile tt_l1_ptr uint16_t* f1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + FACE_BYTES);
        for (uint32_t c = mask_cols; c < 32; ++c) {
            if (c < 16) {
                f0[c] = 0xFF80;
            } else {
                f1[c - 16] = 0xFF80;
            }
        }
        for (uint32_t r = 1; r < 16; ++r) {
            noc_async_read(get_noc_addr(base), base + r * ROW_BYTES, ROW_BYTES);
            noc_async_read(get_noc_addr(base + FACE_BYTES), base + FACE_BYTES + r * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        noc_async_read(get_noc_addr(base), base + 2 * FACE_BYTES, 2 * FACE_BYTES);  // faces 2, 3 = faces 0, 1
        noc_async_read_barrier();
        mask_cb.push_back(1);
    }

    // ---- store_out_block ----
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t W = get_arg_val<uint32_t>(3);
    const uint32_t my_units = W + get_arg_val<uint32_t>(4);
    const uint32_t tail_rows_c = get_arg_val<uint32_t>(9);  // compute rows (credit) of the tail piece
    for (uint32_t i = 0; i < my_units; ++i) {
        const bool is_tail = i >= W;
        const AttnUnit unit = is_tail ? tail_piece(4) : decode_unit<Sq, NQ, LQ>(core_idx + i * C);
        const uint32_t credit = (is_tail ? tail_rows_c : Sq) * DVt;
        {
            MaybeDeviceZoneScope("writer_out_wait");
            cb_wait_front(cb_out, credit);
        }
        MaybeDeviceZoneScope("writer_out_write");
        uint32_t rptr = get_read_ptr(cb_out);
        if (!unit.joint) {
            const uint32_t page = (unit.bh * NQ + unit.row0) * DVt;
            for (uint32_t t = 0; t < unit.rows * DVt; ++t) {
                noc_async_write_tile(page + t, out_acc, rptr);
                rptr += tile_bytes;
            }
        } else {
            if constexpr (has_joint) {
                const auto jout_acc = TensorAccessor(jout_args, jout_addr, tile_bytes);
                const uint32_t page = (unit.bh * LQ + unit.row0) * DVt;
                for (uint32_t t = 0; t < unit.rows * DVt; ++t) {
                    noc_async_write_tile(page + t, jout_acc, rptr);
                    rptr += tile_bytes;
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, credit);
    }
}
