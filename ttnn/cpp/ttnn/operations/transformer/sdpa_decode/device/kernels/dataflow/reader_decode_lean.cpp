// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Length-balanced decode reader: walks this core's slice of the (batch, kv head, k chunk) work list
// (see lean_schedule.hpp) and streams Q, then K/V chunk by chunk, for every piece of it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/lean_schedule.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr bool is_q_sharded = get_compile_time_arg_val(7);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(8);
    constexpr bool is_paged_attention = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(10);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(11);
    constexpr uint32_t Bkv = get_compile_time_arg_val(12);
    constexpr bool is_cur_pos_tensor_sharded = get_compile_time_arg_val(25);
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(24);
    constexpr uint32_t q_page_size_bytes = get_compile_time_arg_val(27);

    constexpr auto q_args = TensorAccessorArgs<36>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg_idx++);
    // Physical coordinates of the core holding each user's Q shard (used when Q is height sharded).
    tt_l1_ptr uint32_t* q_shard_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += B;
    tt_l1_ptr uint32_t* q_shard_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));

    // cur_pos: reader fills c_8 for the writer and copies the stick into c_15 for compute.
    CircularBuffer cb_writer(cb_writer_cur_pos);
    cb_writer.reserve_back(1);
    const uint32_t index_cb_wr_ptr = cb_writer.get_write_ptr();
    if constexpr (!is_cur_pos_tensor_sharded) {
        const auto addrg = TensorAccessor(pos_args, pos_addr);
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
        noc.async_read_barrier();
    }
    CircularBuffer cb_compute(cb_compute_cur_pos);
    cb_compute.reserve_back(1);
    {
        const uint8_t noc_id = noc.get_noc_id();
        UnicastEndpoint pos_src;
        noc.async_read(
            pos_src,
            CoreLocalMem<uint32_t>(cb_compute.get_write_ptr()),
            index_stick_size_B,
            {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = index_cb_wr_ptr},
            {});
        noc.async_read_barrier();
    }
    cb_writer.push_back(1);
    cb_compute.push_back(1);
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);

    constexpr uint32_t k_chunk_size = Sk_chunk_t * tt::constants::TILE_HEIGHT;
    auto get_n = [&](uint32_t b) -> uint32_t { return lean::chunks_for_pos(pos[b], k_chunk_size); };
    const lean::Schedule sched = lean::make_schedule<B, num_kv_heads>(num_cores, get_n);

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    const auto mask_reader = TensorAccessor(mask_args, 0);

    // Page table rows are re-read into one fixed L1 slot whenever the batch changes.
    volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
    uint32_t page_table_batch = UINT32_MAX;
    if constexpr (is_paged_attention) {
        CircularBuffer(cb_id_page_table).reserve_back(1);
    }

    // Q stays resident while consecutive pieces belong to the same user (compute pops it on a user
    // change); the Q CB holds two users so the next one prefetches during the current piece.
    uint32_t q_batch = UINT32_MAX;
    lean::PieceIter<B, num_kv_heads> it(core_idx, sched);
    lean::Piece p;
    while (it.next(get_n, p)) {
        if (p.b != q_batch) {
            read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles, is_q_sharded, false, false, barrier_threshold>(
                false,
                false,
                q_addr,
                q_shard_xs[p.b],
                q_shard_ys[p.b],
                q_chunk_size_bytes,
                q_args,
                q_page_size_bytes,
                p.b * q_chunk_tiles);
            q_batch = p.b;
        }

        if constexpr (is_paged_attention) {
            if (p.b != page_table_batch) {
                page_table_ptr_u32 = read_page_table_for_batch(
                    noc, cb_id_page_table, p.b, page_table_args, page_table_addr, page_table_page_size);
                page_table_batch = p.b;
            }
            uint32_t barrier_count = 0;
            for (uint32_t k_chunk = p.j0; k_chunk < p.j1; ++k_chunk) {
                const uint32_t row = k_chunk * Sk_chunk_t;
                read_k<cb_k_in, DHt, num_kv_heads, block_size_t, k_tile_bytes, barrier_threshold, false, false, 0>(
                    k_chunk_tiles, p.h, Sk_chunk_t, row, k_reader, nullptr, page_table_ptr_u32, barrier_count);
                read_v<cb_v_in, vDHt, num_kv_heads, block_size_t, v_tile_bytes, barrier_threshold, false, false, 0>(
                    v_chunk_tiles, p.h, Sk_chunk_t, row, v_reader, nullptr, page_table_ptr_u32, barrier_count);
            }
        } else {
            const uint32_t kb = (p.b % Bkv) * num_kv_heads * St;
            const uint32_t k_start = (kb + p.h * St) * DHt + p.j0 * Sk_chunk_t * DHt;
            const uint32_t v_start = (kb + p.h * St) * vDHt + p.j0 * Sk_chunk_t * vDHt;
            read_kv_mask_chunks<
                DHt,
                vDHt,
                barrier_threshold,
                mask_tile_bytes,
                PNHt,
                false,
                cb_k_in,
                cb_v_in,
                cb_mask_in,
                false>(
                p.j0,
                p.j1,
                k_start,
                v_start,
                0,
                Sk_chunk_t,
                k_chunk_tiles,
                v_chunk_tiles,
                mask_chunk_tiles,
                k_reader,
                v_reader,
                mask_reader,
                k_tile_bytes,
                v_tile_bytes,
                St);
        }
    }
}
