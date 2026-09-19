// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
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

    constexpr auto q_args = TensorAccessorArgs<13>();
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

    // KV store-and-forward chain (unicast): see joint_sdpa_program_factory.cpp.
    constexpr uint32_t chain_args_offset = joint_v_args.next_compile_time_args_offset();
    constexpr bool kv_chain_enabled = get_compile_time_arg_val(chain_args_offset + 0) == 1;
    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(chain_args_offset + 1);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(chain_args_offset + 2);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(chain_args_offset + 3);
    const bool is_chain_participant = kv_chain_enabled && get_arg_val<uint32_t>(argidx++) != 0;
    const bool is_injector = get_arg_val<uint32_t>(argidx++) != 0;
    const bool is_sink = get_arg_val<uint32_t>(argidx++) != 0;
    const uint32_t prev_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
    if (is_chain_participant) {
        Semaphore<>(valid_semaphore_id).set(VALID);
    }

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

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

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    CircularBuffer cb_k(cb_k_in);
    CircularBuffer cb_v(cb_v_in);

    // One K or V chunk: receive it from the previous chain core, or read it from DRAM through the
    // concatenated generator; then forward it to the next core if this core is a sender for this
    // Q iteration. Chain cores share identical CB layouts and push histories, so the local write
    // pointer is also the receiver's slot address. Mirrors the plain SDPA reader's unicast protocol.
    auto fetch_kv_chunk = [&](CircularBuffer& cb,
                              const auto& generator,
                              const Slice& slice,
                              uint32_t end_tile,
                              uint32_t cb_id,
                              uint32_t tile_bytes,
                              bool transpose,
                              bool should_receive,
                              bool should_forward) {
        cb.reserve_back(k_chunk_tiles);
        const uint32_t chunk_addr = cb.get_write_ptr();
        if (should_receive) {
            Semaphore<> receiver_sem(receiver_semaphore_id);
            receiver_sem.set(INVALID);
            Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
            receiver_sem.wait(VALID);
            cb.push_back(k_chunk_tiles);
        } else {
            fetch_block(generator, slice, end_tile, cb_id, chunk_addr, tile_bytes, transpose, 0);
            if (!should_forward) {
                cb.push_back(k_chunk_tiles);
            }
        }
        if (should_forward) {
            Semaphore<> sender_sem(sender_semaphore_id);
            sender_sem.wait(1);
            sender_sem.set(0);
            noc.async_write(
                CoreLocalMem<uint32_t>(chunk_addr),
                UnicastEndpoint{},
                k_chunk_tiles * tile_bytes,
                {},
                {.noc_x = next_physical_x, .noc_y = next_physical_y, .addr = chunk_addr});
            noc.async_writes_flushed();
            if (!should_receive) {
                cb.push_back(k_chunk_tiles);
            }
            Semaphore<>(valid_semaphore_id)
                .relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_physical_x, next_physical_y);
        }
    };

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            uint32_t q_iter = 0;
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk, ++q_iter) {
                const auto q_row_start_tile = q_chunk * Sq_chunk_t;
                const auto q_row_end_tile = q_row_start_tile + Sq_chunk_t;
                const auto q_slice = Slice(nb, nq, q_row_start_tile, q_row_end_tile, 0, DHt);

                read_block(
                    cat_q_generator, q_slice, q_row_end_tile, cb_q_in, q_tile_bytes, false /*transpose*/
                );

                const bool should_forward = is_chain_participant && !is_sink && (q_iter < next_core_q_chunks);
                const bool should_receive = is_chain_participant && !is_injector;
                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                    const auto kv_row_start_tile = k_chunk * Sk_chunk_t;
                    const auto kv_row_end_tile = kv_row_start_tile + Sk_chunk_t;
                    const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_end_tile, 0, DHt);

                    fetch_kv_chunk(
                        cb_k,
                        cat_k_generator,
                        kv_slice,
                        kv_row_end_tile,
                        cb_k_in,
                        k_tile_bytes,
                        true,
                        should_receive,
                        should_forward);
                    fetch_kv_chunk(
                        cb_v,
                        cat_v_generator,
                        kv_slice,
                        kv_row_end_tile,
                        cb_v_in,
                        v_tile_bytes,
                        false,
                        should_receive,
                        should_forward);
                }
            }
        }
    }
}
