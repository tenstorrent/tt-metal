// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "fabric_bound_matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

// Set by the host only when fusing AG; default to 1 (single whole band per k-block) otherwise.
#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

// Two-NoC output-write split: this writer (NOC_1) handles the block's low M-rows [0, split_rows) from c_2;
// dm_in0 writes [split_rows, M) from c_8 on NOC_0. split_rows = M_block_tiles*AG_SPLIT_NOC1_PCT/100, and
// must match compute/dm_in0. Default 50% (no effect unless SPLIT_OUTPUT_WRITE is set).
#ifndef AG_SPLIT_NOC1_PCT
#define AG_SPLIT_NOC1_PCT 50
#endif

void kernel_main() {
    Noc noc;
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
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    uint32_t in1_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    Semaphore<> in1_sender_sem(get_compile_time_arg_val(14));
    Semaphore<> in1_receiver_sem(get_compile_time_arg_val(15));
    Semaphore<> in1_valid_sem(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t max_defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    // Leading (self/local) k-block positions this device owns; used by receivers to derive the same
    // per-position band count the injector gets from streamed_dir (see dm_in0_sender.cpp).
    // (Read unconditionally to keep the arg layout fixed; only consumed by receivers / IN0_SUB_CHUNKS > 1.)
    [[maybe_unused]] const uint32_t num_local_k_blocks = get_arg_val<uint32_t>(argidx++);

#ifdef FUSE_TERNARY
    // Fuse addcmul - read runtime addresses before setting out_addr_rt_arg_idx
    const uint32_t ternary_a_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t ternary_b_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t broadcast_ternary_b = get_arg_val<uint32_t>(argidx++);
#endif  // FUSE_TERNARY

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here (after ternary if present)

    // Tensor accessor for input tensor
    constexpr auto in1_args = TensorAccessorArgs<21>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in1_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr);
#endif

#ifdef FUSE_TERNARY
// Calculate offset for ternary_a_args
#ifdef FUSE_BIAS
    constexpr uint32_t ternary_a_args_cta_offset = in2_args.next_compile_time_args_offset();
#else
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
#endif
    constexpr uint32_t cb_ternary_a_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_ternary_b_id = tt::CBIndex::c_6;

    constexpr uint32_t ternary_a_tile_size = get_tile_size(cb_ternary_a_id);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(cb_ternary_b_id);

    constexpr auto ternary_a_args = TensorAccessorArgs<ternary_a_args_cta_offset>();
    constexpr auto ternary_b_args = TensorAccessorArgs<ternary_a_args.next_compile_time_args_offset()>();
    const auto ternary_a_reader = TensorAccessor(ternary_a_args, ternary_a_addr);
    const auto ternary_b_reader = TensorAccessor(ternary_b_args, ternary_b_addr);

#endif  // FUSE_TERNARY

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N
    // is half the matmul (weight) N. Compute the halved output geometry once here; the
    // weight-space n ranges (n_tile, N_tiles, ...) are halved at each write call site.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
    // Scratch that holds the whole K_block x N_block once per k-block position so the injector can
    // re-present it to compute once per M-row band (read once from DRAM, pushed nb times) without
    // re-reading DRAM. Only used on the sub-chunked injector path.
    constexpr uint32_t cb_in1_scratch_id = tt::CBIndex::c_7;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_in2_id = tt::CBIndex::c_4;
#endif

    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_out(cb_out_id);
#ifdef FUSE_BIAS
    CircularBuffer cb_in2(cb_in2_id);
#endif

#ifdef FUSE_AG
    // Receiver for ccl fusing
    MinimalMatmulOpReceiver fused_op_receiver;
    uint32_t fused_op_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
    uint32_t num_devices = get_arg_val<uint32_t>(fused_op_rt_args_idx);
    uint32_t num_k_blocks = get_arg_val<uint32_t>(fused_op_rt_args_idx + 1);
    uint32_t num_ag_workers = get_arg_val<uint32_t>(fused_op_rt_args_idx + 9);  // after the 9 scalar args
    uint8_t k_block_device_expected[num_k_blocks]{};
    uint8_t k_block_device_received[num_k_blocks]{};
    uint32_t device_k_block_counts[num_devices]{};
    uint32_t device_k_block_start_ids[num_devices]{};
    uint32_t forward_k_block_schedule[num_k_blocks]{};
    uint32_t backward_sem_addrs[num_ag_workers]{};
    uint32_t forward_sem_addrs[num_ag_workers]{};
    if constexpr (is_injector_core) {
        fused_op_receiver = MinimalMatmulOpReceiver(
            false,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule,
            backward_sem_addrs,
            forward_sem_addrs);
    }
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    // OpSignaler runtime args start after output addresses and optional FUSE_AG args
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
#ifdef FUSE_AG
    // Skip MinimalMatmulFusedOpSignaler::push_matmul_fused_op_rt_args: 9 scalars + num_ag_workers + (2N+1) sems
    srs_fuse_signaler_rt_args_idx += (11 + 2 * get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 9));
#endif
    OpSignaler srs_fuse_signaler;
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
    }
#endif

    in1_valid_sem.set(VALID);

    const uint64_t in1_sender_semaphore_noc_addr =
        get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_semaphore_addr);

    const uint64_t in1_receiver_semaphore_noc_addr =
        get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_receiver_semaphore_addr);

    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                // DeviceZoneScopedN("AVAILABLE");
#if defined(FUSE_AG) && (IN0_SUB_CHUNKS > 1) && defined(AG_INTERLEAVE_BANDS)
                if constexpr (is_injector_core) {
                    // Re-present a forward remote k-block and the following backward one interleaved per band
                    // (A.b0, B.b0, ...), matching dm_in0/compute. Both blocks are read once into the two-slot
                    // scratch; in1 has no M dimension so each band re-presents the whole block. Injectors
                    // never defer_write, so the deferred output flush below never applies on this path.
                    if (n_block_iter == 0 && k_block_iter >= num_local_k_blocks && (k_block_iter + 1) < K_num_blocks &&
                        ((k_block_iter - num_local_k_blocks) & 1u) == 0) {
                        const uint32_t kb_pair[2] = {
                            fused_op_receiver.compute_actual_k_block_iter(true, k_block_iter, k_forward),
                            fused_op_receiver.compute_actual_k_block_iter(true, k_block_iter + 1, k_forward)};
                        const uint32_t in1_block_bytes = in1_block_num_tiles * in1_tile_size;
                        const uint32_t scratch_addr = get_write_ptr(cb_in1_scratch_id);
                        const uint32_t scratch_pair[2] = {scratch_addr, scratch_addr + in1_block_bytes};
                        {
                            // DeviceZoneScopedN("DRAM-Latency");
                            for (uint32_t member = 0; member < 2; member++) {
                                read_in1_block_sync<K_block_tiles, N_block_tiles>(
                                    in1_reader,
                                    in1_shape,
                                    cb_in1_scratch_id,
                                    in1_tile_size,
                                    kb_pair[member] * K_block_tiles,
                                    (kb_pair[member] + 1) * K_block_tiles,
                                    n_tile,
                                    n_tile_end,
                                    /*write_ptr_offset=*/member * in1_block_bytes);
                            }
                        }
                        const uint64_t scratch_noc[2] = {get_noc_addr(scratch_pair[0]), get_noc_addr(scratch_pair[1])};
                        for (uint32_t band = 0; band < (uint32_t)IN0_SUB_CHUNKS; band++) {
                            for (uint32_t member = 0; member < 2; member++) {
                                cb_in1.reserve_back(in1_block_num_tiles);
                                uint32_t in1_start_address = get_write_ptr(cb_in1_id);
                                noc_async_read(scratch_noc[member], in1_start_address, in1_block_bytes);
                                noc.async_read_barrier();
                                cb_in1.push_back(in1_block_num_tiles);
                                if (!is_sink_core) {
                                    // DeviceZoneScopedN("MCAST-SEND");
                                    in1_sender_sem.wait(1);
                                    in1_sender_sem.set(0);
                                    uint32_t fwd_addr = in1_start_address;
                                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                                        uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | fwd_addr;
                                        noc_async_write(fwd_addr, in1_unicast_data_addr, current_N_tiles_bytes);
                                        fwd_addr += full_N_tiles_bytes;
                                    }
#ifdef ARCH_BLACKHOLE
                                    noc.async_writes_flushed();
#endif
                                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                                }
                            }
                        }
#ifdef SRS_FUSE_OP_SIGNALER
                        if constexpr (is_output_writer) {
                            if (not_first_block && k_block_iter == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                            }
                            if (not_first_block && (k_block_iter + 1) == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                            }
                        }
#endif
                        k_block_iter++;  // the backward member of the pair is consumed here too
                        continue;
                    }
                }
#endif
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        // DeviceZoneScopedN("DEFER-WRITE");
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr = get_read_ptr(cb_out_id);
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape_swiglu,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        } else {
                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape_swiglu,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        }
                        cb_out.pop_front(out_block_num_tiles_swiglu);
#else
                        cb_out.wait_front(out_block_num_tiles);
                        uint32_t out_read_ptr = get_read_ptr(cb_out_id);

                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        } else {
                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_out.pop_front(out_block_num_tiles);
#endif  // FUSE_SWIGLU
                    }
                }

                // ---- Read-once, re-present-per-band ----
                // in1 has no M dimension, so an M-row band consumes the whole K_block x N_block. To stay
                // in lockstep with the banded in0 (so the matmul fires per band), the injector reads the
                // block once from DRAM into scratch and re-presents it to compute once per band (pushed nb
                // times, no DRAM re-read); receivers receive/push/forward once per band. nb is computed
                // exactly as in dm_in0_sender so per-band CB and semaphore counts never desync.
                [[maybe_unused]] uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                uint32_t nb;
                if constexpr (is_injector_core) {
#ifdef FUSE_AG
                    k_block = fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                    const uint8_t k_dir = fused_op_receiver.streamed_dir;
                    nb = (n_block_iter == 0 && k_dir != 2) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
#else
                    nb = 1u;
#endif
                    // Read the whole K_block x N_block once into scratch. The scratch CB is used as a
                    // plain fixed L1 region (single slot, single-core produce+consume in this kernel), so
                    // we take its write pointer directly without reserve/push/pop.
                    uint32_t scratch_addr = get_write_ptr(cb_in1_scratch_id);
                    {
                        // DeviceZoneScopedN("DRAM-Latency");
                        read_in1_block_sync<K_block_tiles, N_block_tiles>(
                            in1_reader,
                            in1_shape,
                            cb_in1_scratch_id,
                            in1_tile_size,
                            k_block * K_block_tiles,
                            (k_block + 1) * K_block_tiles,
                            n_tile,
                            n_tile_end);
                    }
                    const uint64_t scratch_noc_addr = get_noc_addr(scratch_addr);
                    const uint32_t in1_block_bytes = in1_block_num_tiles * in1_tile_size;
                    for (uint32_t band = 0; band < nb; band++) {
                        cb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_start_address = get_write_ptr(cb_in1_id);
                        // Re-present from scratch (local L1 copy, no DRAM re-read).
                        noc_async_read(scratch_noc_addr, in1_start_address, in1_block_bytes);
                        noc.async_read_barrier();
                        cb_in1.push_back(in1_block_num_tiles);
                        if (!is_sink_core) {
                            // DeviceZoneScopedN("MCAST-SEND");
                            in1_sender_sem.wait(1);
                            in1_sender_sem.set(0);
                            uint32_t fwd_addr = in1_start_address;
                            for (uint32_t i = 0; i < K_block_tiles; i++) {
                                uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | fwd_addr;
                                noc_async_write(fwd_addr, in1_unicast_data_addr, current_N_tiles_bytes);
                                fwd_addr += full_N_tiles_bytes;
                            }
#ifdef ARCH_BLACKHOLE
                            noc.async_writes_flushed();
#endif
                            noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                        }
                    }
                } else {
                    nb = (n_block_iter == 0 && k_block_iter >= num_local_k_blocks) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
                    for (uint32_t band = 0; band < nb; band++) {
                        cb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_start_address = get_write_ptr(cb_in1_id);
                        {
                            // DeviceZoneScopedN("RECV-WAIT");
                            in1_receiver_sem.set(INVALID);
                            noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                            in1_receiver_sem.wait(VALID);
                        }
                        cb_in1.push_back(in1_block_num_tiles);
                        if (!is_sink_core) {
                            // DeviceZoneScopedN("MCAST-SEND");
                            in1_sender_sem.wait(1);
                            in1_sender_sem.set(0);
                            uint32_t fwd_addr = in1_start_address;
                            for (uint32_t i = 0; i < K_block_tiles; i++) {
                                uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | fwd_addr;
                                noc_async_write(fwd_addr, in1_unicast_data_addr, current_N_tiles_bytes);
                                fwd_addr += full_N_tiles_bytes;
                            }
#ifdef ARCH_BLACKHOLE
                            noc.async_writes_flushed();
#endif
                            noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                        }
                    }
                }
#ifdef SRS_FUSE_OP_SIGNALER
                if constexpr (is_output_writer) {
                    if (not_first_block && k_block_iter == max_defer_write_k_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                    }
                }
#endif
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_in2.reserve_back(N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_in2_id);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc.async_read(
                        in2_reader,
                        CoreLocalMem<uint32_t>(l1_write_addr_in2),
                        in2_tile_size,
                        {.page_id = n_tile_id},
                        {});
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc.async_read_barrier();

                cb_in2.push_back(N_block_tiles);
            }
#endif

#ifdef FUSE_TERNARY
            if constexpr (!is_output_writer) {
                read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                    ternary_a_reader,
                    ternary_b_reader,
                    out_shape,
                    cb_ternary_a_id,
                    cb_ternary_b_id,
                    ternary_a_tile_size,
                    ternary_b_tile_size,
                    broadcast_ternary_b,
                    m_tile,
                    m_tile_end,
                    n_tile,
                    n_tile_end);
            }
#endif  // FUSE_TERNARY

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !is_last_block;
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    // DeviceZoneScopedN("OUT-WRITE");
#ifdef FUSE_SWIGLU
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile / 2,
                            n_tile_end / 2);
                    } else {
                        write_block_sync_granular_split<
                            M_block_tiles,
                            out_N_block_tiles,
                            N_chunks,
                            out_N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape_swiglu,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile / 2,
                            n_tile_end / 2);
                    }
#else
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
#ifdef SPLIT_OUTPUT_WRITE
                        // NOC_1 writer: low rows [0, split_rows) from c_2.
                        constexpr uint32_t split_rows = (M_block_tiles * AG_SPLIT_NOC1_PCT) / 100;
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end,
                            0,
                            split_rows);
#else
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
#endif
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
#endif  // FUSE_SWIGLU
#ifdef SRS_FUSE_OP_SIGNALER
                    if (is_last_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                    }
#endif
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
