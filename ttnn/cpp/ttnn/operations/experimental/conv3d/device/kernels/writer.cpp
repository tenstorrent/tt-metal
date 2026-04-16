// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_matmul_result_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_matmul_interm_tiled = get_compile_time_arg_val(3);
    constexpr uint32_t cb_reduction_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_worker_ack_back = get_compile_time_arg_val(5);
    constexpr uint32_t N = get_compile_time_arg_val(6);
    constexpr uint32_t T_out = get_compile_time_arg_val(7);
    constexpr uint32_t H_out = get_compile_time_arg_val(8);
    constexpr uint32_t W_out = get_compile_time_arg_val(9);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(10);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(11);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(12);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(13);
    constexpr uint32_t matmul_M_t = get_compile_time_arg_val(14);
    constexpr uint32_t matmul_K_t = get_compile_time_arg_val(15);
    constexpr uint32_t matmul_N_t = get_compile_time_arg_val(16);
    constexpr uint32_t num_patches_tile_padded = get_compile_time_arg_val(17);
    constexpr uint32_t C_out_block_bytes = get_compile_time_arg_val(19);
    constexpr bool use_bias = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_zero_tiled = get_compile_time_arg_val(22);

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_reducer = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(argidx++);

    experimental::Noc noc;
    experimental::CB cb_out(cb_matmul_result_rm);
    experimental::CB cb_weight(cb_weight_tiled);
    experimental::CB cb_bias(cb_bias_tiled);
    experimental::CB cb_interm(cb_matmul_interm_tiled);
    experimental::CB cb_reduction(cb_reduction_tiled);
    experimental::CB cb_ack(cb_worker_ack_back);
    experimental::Semaphore<> sem(semaphore_id);

    // Reducer coordinates and worker core coordinates are only present when num_workers > 0
    uint32_t reducer_core_x = 0, reducer_core_y = 0;
    tt_l1_ptr uint32_t* worker_core_xs = nullptr;
    tt_l1_ptr uint32_t* worker_core_ys = nullptr;
    if (num_workers > 0) {
        reducer_core_x = get_arg_val<uint32_t>(argidx++);
        reducer_core_y = get_arg_val<uint32_t>(argidx++);
        worker_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(argidx));
        argidx += num_workers;
        worker_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(argidx));
    }

    constexpr uint32_t tile_bytes = get_tile_size(cb_weight_tiled);
    constexpr uint32_t partials_tile_bytes = get_tile_size(cb_matmul_interm_tiled);
    constexpr auto out_args = TensorAccessorArgs<23>();
    constexpr auto weight_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto weight_reader = TensorAccessor(weight_args, weight_addr);
    const auto bias_reader = TensorAccessor(bias_args, bias_addr);

    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;
    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t C_out_t = C_out_num_blocks * matmul_N_t;
    constexpr uint32_t T_out_H_out_W_out = T_out * H_out * W_out;

    // Zero-fill the zero CB for FPU accumulate reduction (DST += tile + 0).
    // The zero tile stays resident for the lifetime of the kernel.
    // Clamp index to avoid constexpr OOB when cb_zero_tiled==32 (fp32 reduction disabled).
    // The if constexpr discards this branch, but WH's compiler still evaluates get_tile_size.
    constexpr uint32_t cb_zero_safe = cb_zero_tiled < 32 ? cb_zero_tiled : 0;
    if constexpr (cb_zero_tiled < 32) {
        experimental::CB cb_zero(cb_zero_tiled);
        cb_zero.reserve_back(1);
        constexpr uint32_t zero_tile_bytes = get_tile_size(cb_zero_safe);
        uint32_t zero_addr = cb_zero.get_write_ptr();
        uint32_t remaining = zero_tile_bytes;
        experimental::set_read_state<MEM_ZEROS_SIZE>(noc, MEM_ZEROS_BASE);
        while (remaining >= MEM_ZEROS_SIZE) {
            experimental::read_with_state(noc, zero_addr, MEM_ZEROS_BASE);
            zero_addr += MEM_ZEROS_SIZE;
            remaining -= MEM_ZEROS_SIZE;
        }
        if (remaining > 0) {
            experimental::UnicastEndpoint self_ep;
            noc.async_read(
                self_ep,
                experimental::CoreLocalMem<uint32_t>(zero_addr),
                remaining,
                experimental::local_addr(MEM_ZEROS_BASE, noc.get_noc_id()),
                {});
        }
        noc.async_read_barrier();
        cb_zero.push_back(1);
    }

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
        for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
            const uint32_t c_in_offset_t = c_in_block * matmul_K_t;
            // Iterate only over assigned C_out blocks
            for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
                const uint32_t c_out_offset_t = c_out_block * matmul_N_t;

                // Read weights and bias for this block
                cb_weight.reserve_back(weight_tiles);
                uint32_t weight_write_offset = 0;

                for (uint32_t row = c_in_offset_t; row < c_in_offset_t + matmul_K_t; row++) {
                    for (uint32_t col = c_out_offset_t; col < c_out_offset_t + matmul_N_t; col++) {
                        uint32_t weight_idx = row * C_out_t + col;
                        noc.async_read(
                            weight_reader,
                            cb_weight,
                            tile_bytes,
                            {.page_id = weight_idx},
                            {.offset_bytes = weight_write_offset});
                        weight_write_offset += tile_bytes;
                    }
                }
                noc.async_read_barrier();
                cb_weight.push_back(weight_tiles);

                if constexpr (use_bias) {
                    if (is_reducer) {
                        cb_bias.reserve_back(matmul_N_t);
                        uint32_t bias_write_offset = 0;
                        for (uint32_t i = c_out_offset_t; i < c_out_offset_t + matmul_N_t; i++) {
                            noc.async_read(
                                bias_reader, cb_bias, tile_bytes, {.page_id = i}, {.offset_bytes = bias_write_offset});
                            bias_write_offset += tile_bytes;
                        }
                        noc.async_read_barrier();
                        cb_bias.push_back(matmul_N_t);
                    }
                }

                // Write output for assigned ranges
                for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
                    const uint32_t t_block_end = std::min(t_block + T_block_size, t_out_end);

                    for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                        const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                        for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                            const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);

                            if (!is_reducer) {
                                // I'm a worker.
                                // Wait for compute to finish
                                cb_reduction.wait_front(output_tiles);

                                // Reset our semaphore
                                sem.set(0);

                                // Signal to reducer that we have data ready
                                sem.up(noc, reducer_core_x, reducer_core_y, 1);

                                // Wait for reducer to ack that it has read our data
                                sem.wait(1);

                                // Handshake with compute so it can continue
                                cb_reduction.pop_front(output_tiles);
                                cb_ack.reserve_back(1);
                                cb_ack.push_back(1);
                            } else {
                                // I'm a reducer.
                                // Wait for all workers to finish
                                sem.wait(num_workers);

                                // Reset our semaphore
                                sem.set(0);

                                const uint32_t worker_output_read_ptr = cb_interm.get_read_ptr();
                                for (uint32_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
                                    // Read data from worker into reduction buffer
                                    // Stall if compute has not cleared buffer
                                    cb_reduction.reserve_back(output_tiles);
                                    uint32_t reduction_offset = 0;
                                    const uint16_t worker_x = worker_core_xs[worker_idx];
                                    const uint16_t worker_y = worker_core_ys[worker_idx];
                                    for (uint32_t tile = 0; tile < output_tiles; tile++) {
                                        experimental::UnicastEndpoint ep;
                                        noc.async_read(
                                            ep,
                                            cb_reduction,
                                            partials_tile_bytes,
                                            {.noc_x = worker_x,
                                             .noc_y = worker_y,
                                             .addr = worker_output_read_ptr + tile * partials_tile_bytes},
                                            {.offset_bytes = reduction_offset});
                                        reduction_offset += partials_tile_bytes;
                                    }
                                    noc.async_read_barrier();
                                    cb_reduction.push_back(output_tiles);

                                    sem.up(noc, worker_x, worker_y, 1);
                                }

                                cb_out.wait_front(output_tiles);
                                uint32_t cb_read_ptr = cb_out.get_read_ptr();

                                for (uint32_t t = t_block; t < t_block_end; ++t) {
                                    for (uint32_t h = h_block; h < h_block_end; ++h) {
                                        for (uint32_t w = w_block; w < w_block_end; ++w) {
                                            uint32_t out_page_idx =
                                                batch_idx * T_out_H_out_W_out + t * H_out * W_out + h * W_out + w;
                                            noc.async_write(
                                                experimental::CoreLocalMem<uint32_t>(cb_read_ptr),
                                                out_writer,
                                                C_out_block_bytes,
                                                {},
                                                {.page_id = out_page_idx,
                                                 .offset_bytes = c_out_block * C_out_block_bytes});
                                            cb_read_ptr += C_out_block_bytes;
                                        }
                                    }
                                }
                                noc.async_write_barrier();
                                cb_out.pop_front(output_tiles);
                            }
                        }
                    }
                }
            }
        }
    }
    noc.async_atomic_barrier();
}
