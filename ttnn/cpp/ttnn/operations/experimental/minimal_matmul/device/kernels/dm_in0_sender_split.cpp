// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);
    constexpr uint32_t in3_tile_size = get_compile_time_arg_val(21);

    // Load input/output addresses and range parameters
    // Output addresses are at the end (N_chunks of them)
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in3_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    // N output addresses are at the end - save the RT arg index for make_tensor_accessor_tuple_uniform_page_size
    const uint32_t out_addr_rt_arg_idx = argidx;

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<22>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);

    // Create tuple of TensorAccessorArgs for all output tensors
    // This uses template metaprogramming to create N accessor args at compile time
    constexpr uint32_t tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, tensor_args_cta_offset>();

    // Create tuple of TensorAccessors with concrete types (required for noc_async_write_tile)
    // Note: Using our custom helper that takes actual page_size value, not CTA index
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

#ifdef FUSE_BIAS
    // Compute the CTA offset for bias accessor dynamically - it comes after all N_chunks output tensor accessors
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr, in2_tile_size);
#endif

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    // Output shapes for each chunk - each has N_tiles_per_chunk width
    // N_tiles_per_chunk is already guaranteed to be tile-aligned by validation
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

    volatile tt_l1_ptr uint32_t* in0_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_semaphore_addr);
    *(in0_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        reuse_block = false;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = n_forward ? N_start_tile + n_block_iter * N_block_tiles
                                        : N_start_tile + (N_blocks_per_core - 1 - n_block_iter) * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        write_block_sync_granular_split_variadic<M_block_tiles, N_block_tiles>(
                            outputs_tuple,
                            out0_shape,
                            N_tiles_per_chunk,
                            cb_id_out,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end);
                    }
                }

                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                }

                cb_push_back(cb_id_in0, in0_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    uint64_t in0_unicast_data_addr = get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                    noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                }
            }

#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_async_read_tile(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

            k_forward = !k_forward;
            reuse_block = true;

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;

            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    write_block_sync_granular_split_variadic<M_block_tiles, N_block_tiles>(
                        outputs_tuple,
                        out0_shape,
                        N_tiles_per_chunk,
                        cb_id_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile,
                        n_tile_end);
                }
            }
        }
        n_forward = !n_forward;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
