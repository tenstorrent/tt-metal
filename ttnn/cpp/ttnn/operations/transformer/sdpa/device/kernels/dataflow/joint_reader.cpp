// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t valid_Lt = get_compile_time_arg_val(7);
    constexpr uint32_t padded_Nqt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nkt = get_compile_time_arg_val(9);
    constexpr uint32_t padded_Lqt = get_compile_time_arg_val(10);
    constexpr uint32_t padded_Lkt = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);

    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(13);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(14);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(15);
    constexpr auto q_args = TensorAccessorArgs<16>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto joint_q_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
    constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);
    const bool participates = get_arg_val<uint32_t>(argidx++) == 1;
    const bool is_injector = get_arg_val<uint32_t>(argidx++) == 1;
    const bool is_sink = get_arg_val<uint32_t>(argidx++) == 1;
    const uint32_t prev_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
    if (participates) {
        Semaphore<>(valid_semaphore_id).set(VALID);
    }

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t kv_chunk_tiles = Sk_chunk_t * DHt;
    CircularBuffer cb_k(cb_k_in);
    CircularBuffer cb_v(cb_v_in);

    // Chained K or V chunk: receive it from the previous core, or read it and hand it on to the next core
    // at the same CB address. The receiver signals readiness on the sender's semaphore first, so the write
    // lands in a slot the receiver has reserved.
    auto chained_read = [&](CircularBuffer& cb,
                            uint32_t cb_id,
                            const auto& generator,
                            const Slice& slice,
                            uint32_t end_tile,
                            uint32_t tile_bytes,
                            bool transpose,
                            bool should_receive,
                            bool should_forward) {
        uint32_t addr = 0;
        if (should_receive) {
            cb.reserve_back(kv_chunk_tiles);
            addr = cb.get_write_ptr();
            Semaphore<> receiver_sem(receiver_semaphore_id);
            receiver_sem.set(INVALID);
            Semaphore<>(sender_semaphore_id).up(noc, prev_x, prev_y, 1);
            receiver_sem.wait(VALID);
            cb.push_back(kv_chunk_tiles);
        } else if (should_forward) {
            cb.reserve_back(kv_chunk_tiles);
            addr = cb.get_write_ptr();
            fetch_block(generator, slice, end_tile, cb_id, addr, tile_bytes, transpose);
        } else {
            read_block(generator, slice, end_tile, cb_id, tile_bytes, transpose);
        }
        if (should_forward) {
            Semaphore<> sender_sem(sender_semaphore_id);
            sender_sem.wait(1);
            sender_sem.set(0);
            noc.async_write(
                CoreLocalMem<uint32_t>(addr),
                UnicastEndpoint{},
                kv_chunk_tiles * tile_bytes,
                {},
                {.noc_x = next_x, .noc_y = next_y, .addr = addr});
            noc.async_writes_flushed();
            if (!should_receive) {
                cb.push_back(kv_chunk_tiles);
            }
            Semaphore<>(valid_semaphore_id).relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_x, next_y);
        }
    };

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr);

    const auto input_tile_logical = TensorTileShape(B, NH, valid_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, valid_Lt, DHt);
    const auto cat_q_generator =
        CatAddrGenerator(q_reader, input_tile_logical, padded_Nqt, joint_q_reader, joint_tile_logical, padded_Lqt);
    const auto cat_k_generator =
        CatAddrGenerator(k_reader, input_tile_logical, padded_Nkt, joint_k_reader, joint_tile_logical, padded_Lkt);
    const auto cat_v_generator =
        CatAddrGenerator(v_reader, input_tile_logical, padded_Nkt, joint_v_reader, joint_tile_logical, padded_Lkt);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                const auto q_row_start_tile = q_chunk * Sq_chunk_t;
                const auto q_row_end_tile = q_row_start_tile + Sq_chunk_t;
                const auto q_slice = Slice(nb, nq, q_row_start_tile, q_row_end_tile, 0, DHt);

                read_block(
                    cat_q_generator, q_slice, q_row_end_tile, cb_q_in, q_tile_bytes, false /*transpose*/
                );

                const bool should_forward = participates && !is_sink && (q_chunk - local_q_start) < next_core_q_chunks;
                const bool should_receive = participates && !is_injector;
                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                    const auto kv_row_start_tile = k_chunk * Sk_chunk_t;
                    const auto kv_row_end_tile = kv_row_start_tile + Sk_chunk_t;
                    const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_end_tile, 0, DHt);

                    chained_read(
                        cb_k,
                        cb_k_in,
                        cat_k_generator,
                        kv_slice,
                        kv_row_end_tile,
                        k_tile_bytes,
                        true,
                        should_receive,
                        should_forward);
                    chained_read(
                        cb_v,
                        cb_v_in,
                        cat_v_generator,
                        kv_slice,
                        kv_row_end_tile,
                        v_tile_bytes,
                        false,
                        should_receive,
                        should_forward);
                }
            }
        }
    }
}
