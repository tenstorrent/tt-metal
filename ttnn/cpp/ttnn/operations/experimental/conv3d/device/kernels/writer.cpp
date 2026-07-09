// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/conv3d_weight_share.hpp"

template <
    uint32_t tile_bytes,
    uint32_t matmul_K_t,
    uint32_t matmul_N_t,
    uint32_t C_out_t,
    typename WeightReader,
    typename WeightCB>
FORCE_INLINE void read_weight_block(
    Noc noc,
    const WeightReader& weight_reader,
    const WeightCB& cb_weight,
    uint32_t c_in_offset_t,
    uint32_t c_out_offset_t) {
    uint32_t weight_write_offset = 0;
    for (uint32_t row = c_in_offset_t; row < c_in_offset_t + matmul_K_t; row++) {
        for (uint32_t col = c_out_offset_t; col < c_out_offset_t + matmul_N_t; col++) {
            const uint32_t weight_idx = row * C_out_t + col;
            noc.async_read(
                weight_reader, cb_weight, tile_bytes, {.page_id = weight_idx}, {.offset_bytes = weight_write_offset});
            weight_write_offset += tile_bytes;
        }
    }
    noc.async_read_barrier();
}

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
    // weight_share_mode (see WeightShareMode in conv3d_weight_share.hpp): Disabled, Chain, or Mcast.
    constexpr WeightShareMode weight_share_mode = static_cast<WeightShareMode>(get_compile_time_arg_val(22));
    constexpr bool enable_weight_chain = weight_share_mode == WeightShareMode::Chain;
    constexpr bool enable_weight_mcast = weight_share_mode == WeightShareMode::Mcast;
    constexpr uint32_t weights_mcast_sender_sem_id = get_compile_time_arg_val(23);
    constexpr uint32_t weights_mcast_receiver_sem_id = get_compile_time_arg_val(24);
    constexpr bool enable_streaming_output = get_compile_time_arg_val(25) == 1;

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
    // weight_share_role: see WeightShareRole in conv3d_weight_share.hpp.
    const WeightShareRole weight_share_role = static_cast<WeightShareRole>(get_arg_val<uint32_t>(argidx++));
    const uint32_t weight_src_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t weight_src_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_succ_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_succ_noc_y = get_arg_val<uint32_t>(argidx++);
    // Mcast bbox + counts. Only mcast sender (role 4) needs the bbox; passive (role 6) needs iters.
    const uint32_t mcast_bbox_start_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_bbox_start_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_bbox_end_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_bbox_end_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_num_dests = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_num_iters = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(argidx++);

    Noc noc;
    experimental::CB cb_out(cb_matmul_result_rm);
    experimental::CB cb_weight(cb_weight_tiled);
    experimental::CB cb_bias(cb_bias_tiled);
    experimental::CB cb_interm(cb_matmul_interm_tiled);
    experimental::CB cb_reduction(cb_reduction_tiled);
    experimental::CB cb_ack(cb_worker_ack_back);
    Semaphore<> sem(semaphore_id);
    Semaphore<> weights_mcast_sender_sem(weights_mcast_sender_sem_id);
    Semaphore<> weights_mcast_receiver_sem(weights_mcast_receiver_sem_id);

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
    constexpr auto out_args = TensorAccessorArgs<26>();
    constexpr auto weight_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto weight_reader = TensorAccessor(weight_args, weight_addr);
    const auto bias_reader = TensorAccessor(bias_args, bias_addr);

    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;
    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t C_out_t = C_out_num_blocks * matmul_N_t;
    constexpr uint32_t T_out_H_out_W_out = T_out * H_out * W_out;

    // Mcast passive participant: this core sits inside the mcast bbox but has no work. It exists
    // only to satisfy the multicast handshake (sender's wait depends on every core in the bbox
    // ack'ing). Run mcast_num_iters handshakes (matching active receivers' iteration count) and
    // exit before any work-dependent code below.
    if constexpr (enable_weight_mcast) {
        if (weight_share_role == WeightShareRole::McastPassive) {
            for (uint32_t i = 0; i < mcast_num_iters; i++) {
                weights_mcast_sender_sem.up(noc, weight_src_noc_x, weight_src_noc_y, 1);
                weights_mcast_receiver_sem.wait(1);
                weights_mcast_receiver_sem.set(0);
            }
            noc.async_atomic_barrier();
            return;
        }
    }

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
        for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
            const uint32_t c_in_offset_t = c_in_block * matmul_K_t;
            // Iterate only over assigned C_out blocks
            for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
                const uint32_t c_out_offset_t = c_out_block * matmul_N_t;

                // Read weights for this block. Strategy depends on weight_share_mode:
                //   - chain (1): each non-injector receives from pred, each non-tail forwards to succ
                //   - mcast (2): sender DRAM-reads then multicasts over the bbox; receivers wait
                //   - off (0): every core reads from DRAM independently
                if constexpr (enable_weight_chain) {
                    cb_weight.reserve_back(weight_tiles);
                    if (weight_share_role == WeightShareRole::Local) {
                        // Local DRAM read (single-core group inside a chain-mode program).
                        read_weight_block<tile_bytes, matmul_K_t, matmul_N_t, C_out_t>(
                            noc, weight_reader, cb_weight, c_in_offset_t, c_out_offset_t);
                    } else {
                        if (weight_share_role == WeightShareRole::ChainInjector) {
                            read_weight_block<tile_bytes, matmul_K_t, matmul_N_t, C_out_t>(
                                noc, weight_reader, cb_weight, c_in_offset_t, c_out_offset_t);
                        } else {
                            weights_mcast_sender_sem.up(noc, weight_src_noc_x, weight_src_noc_y, 1);
                            weights_mcast_receiver_sem.wait(1);
                            weights_mcast_receiver_sem.set(0);
                        }
                        if (weight_share_role != WeightShareRole::ChainTail) {
                            weights_mcast_sender_sem.wait(1);
                            weights_mcast_sender_sem.set(0);

                            const uint32_t weight_block_bytes = weight_tiles * tile_bytes;
                            const uint32_t local_addr = cb_weight.get_write_ptr();
                            UnicastEndpoint ep;
                            noc.async_write(
                                use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_weight),
                                ep,
                                weight_block_bytes,
                                {.offset_bytes = 0},
                                {.noc_x = chain_succ_noc_x, .noc_y = chain_succ_noc_y, .addr = local_addr});
                            noc.async_write_barrier();

                            weights_mcast_receiver_sem.up(noc, chain_succ_noc_x, chain_succ_noc_y, 1);
                            noc.async_atomic_barrier();
                        }
                    }
                    cb_weight.push_back(weight_tiles);
                } else if constexpr (enable_weight_mcast) {
                    cb_weight.reserve_back(weight_tiles);
                    if (weight_share_role == WeightShareRole::McastSender) {
                        // Sender: DRAM read into local L1, then hardware multicast over the
                        // bbox. The mcast call below uses EXCLUDE_SRC; sender keeps the
                        // DRAM-read copy in cb_weight so it doesn't need to receive its own
                        // multicast.
                        read_weight_block<tile_bytes, matmul_K_t, matmul_N_t, C_out_t>(
                            noc, weight_reader, cb_weight, c_in_offset_t, c_out_offset_t);

                        // mcast_num_dests is the number of receivers (= number of acks expected
                        // = num_dests passed to EXCLUDE_SRC mcast API). Host always places the
                        // sender inside the bbox, so mcast_num_dests = bbox_cores - 1.
                        weights_mcast_sender_sem.wait(mcast_num_dests);
                        weights_mcast_sender_sem.set(0);

                        // EXCLUDE_SRC: sender already has the data from its DRAM read, so don't
                        // loopback. linked=true lets the API's internal burst-splitting
                        // (~324 KiB → ~20 × 16 KiB on BH) amortize per-burst setup.
                        const uint32_t weight_block_bytes = weight_tiles * tile_bytes;
                        const uint32_t local_addr = cb_weight.get_write_ptr();
                        MulticastEndpoint mcast_dst;
                        noc.async_write_multicast(
                            use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_weight),
                            mcast_dst,
                            weight_block_bytes,
                            mcast_num_dests,
                            {},
                            {.noc_x_start = mcast_bbox_start_x,
                             .noc_y_start = mcast_bbox_start_y,
                             .noc_x_end = mcast_bbox_end_x,
                             .noc_y_end = mcast_bbox_end_y,
                             .addr = local_addr},
                            /*linked=*/true);

                        // No write_barrier between data and flag mcast (per conv2d pattern):
                        // both go through the same NoC and VC (NOC_CMD_STATIC_VC), so the flag
                        // can never overtake the data on any receiver. Sender's push_back is
                        // also correct without a barrier — EXCLUDE_SRC means sender's own L1 is
                        // not a destination, so the data already in cb_weight from the DRAM read
                        // is what compute consumes.
                        weights_mcast_receiver_sem.set(VALID);
                        weights_mcast_receiver_sem.set_multicast(
                            noc,
                            mcast_bbox_start_x,
                            mcast_bbox_start_y,
                            mcast_bbox_end_x,
                            mcast_bbox_end_y,
                            mcast_num_dests,
                            false);
                    } else if (weight_share_role == WeightShareRole::McastReceiver) {
                        // Active receiver: ack sender, wait for VALID, reset for next iteration.
                        weights_mcast_sender_sem.up(noc, weight_src_noc_x, weight_src_noc_y, 1);
                        weights_mcast_receiver_sem.wait(1);
                        weights_mcast_receiver_sem.set(0);
                    }
                    cb_weight.push_back(weight_tiles);
                } else {
                    cb_weight.reserve_back(weight_tiles);
                    read_weight_block<tile_bytes, matmul_K_t, matmul_N_t, C_out_t>(
                        noc, weight_reader, cb_weight, c_in_offset_t, c_out_offset_t);
                    cb_weight.push_back(weight_tiles);
                }

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
                                // Wait for compute to finish.
                                cb_reduction.wait_front(output_tiles);

                                // Reset our semaphore.
                                sem.set(0);

                                // Signal to reducer that we have data ready.
                                sem.up(noc, reducer_core_x, reducer_core_y, 1);

                                // Wait for reducer to ack that it has read our data.
                                sem.wait(1);

                                // Handshake with compute so it can continue.
                                cb_reduction.pop_front(output_tiles);
                                cb_ack.reserve_back(1);
                                cb_ack.push_back(1);
                            } else {
                                // I'm a reducer.
                                if (num_workers > 0) {
                                    // Wait for all workers to finish.
                                    sem.wait(num_workers);

                                    // Reset our semaphore.
                                    sem.set(0);

                                    const uint32_t worker_output_read_ptr = cb_interm.get_read_ptr();
                                    for (uint32_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
                                        // Read data from worker into reduction buffer.
                                        cb_reduction.reserve_back(output_tiles);
                                        const uint16_t worker_x = worker_core_xs[worker_idx];
                                        const uint16_t worker_y = worker_core_ys[worker_idx];
                                        UnicastEndpoint ep;
                                        noc.async_read(
                                            ep,
                                            cb_reduction,
                                            output_tiles * partials_tile_bytes,
                                            {.noc_x = worker_x, .noc_y = worker_y, .addr = worker_output_read_ptr},
                                            {.offset_bytes = 0});
                                        noc.async_read_barrier();
                                        cb_reduction.push_back(output_tiles);

                                        sem.up(noc, worker_x, worker_y, 1);
                                    }
                                }

                                // Streaming output drains one single-tile C_out row at a time to overlap writes with
                                // the remaining compute tail.
                                constexpr uint32_t row_tiles = matmul_N_t;
                                if constexpr (!enable_streaming_output) {
                                    cb_out.wait_front(output_tiles);
                                }
                                uint32_t patch_idx = 0;
                                uint32_t cb_read_offset = 0;
                                uint32_t rows_waited = 0;
                                for (uint32_t t = t_block; t < t_block_end; ++t) {
                                    for (uint32_t h = h_block; h < h_block_end; ++h) {
                                        for (uint32_t w = w_block; w < w_block_end; ++w) {
                                            if constexpr (enable_streaming_output) {
                                                if (patch_idx % tt::constants::TILE_HEIGHT == 0) {
                                                    rows_waited++;
                                                    cb_out.wait_front(rows_waited * row_tiles);
                                                }
                                            }
                                            uint32_t out_page_idx =
                                                batch_idx * T_out_H_out_W_out + t * H_out * W_out + h * W_out + w;
                                            noc.async_write(
                                                cb_out,
                                                out_writer,
                                                C_out_block_bytes,
                                                {.offset_bytes = cb_read_offset},
                                                {.page_id = out_page_idx,
                                                 .offset_bytes = c_out_block * C_out_block_bytes});
                                            cb_read_offset += C_out_block_bytes;
                                            if constexpr (enable_streaming_output) {
                                                patch_idx++;
                                            }
                                        }
                                    }
                                }
                                if constexpr (enable_streaming_output) {
                                    cb_out.wait_front(output_tiles);
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
