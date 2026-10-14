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
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/subchunk_bands.hpp"

// Set by the host only when fusing AG; default to 1 (single whole band per k-block) otherwise.
#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

// Two-NoC output-write split: under SPLIT_OUTPUT_WRITE, dm_in0 becomes a co-writer draining the second output CB
#ifndef AG_OUT_WRITE_CB
#define AG_OUT_WRITE_CB 2
#endif
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
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    Semaphore<> in0_sender_sem(get_compile_time_arg_val(14));
    Semaphore<> in0_receiver_sem(get_compile_time_arg_val(15));
    Semaphore<> in0_valid_sem(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);

    // Load input/output addresses and range parameters
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
    const uint32_t max_defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    // Leading (self/local) k-block positions this device owns
    [[maybe_unused]] const uint32_t num_local_k_blocks = get_arg_val<uint32_t>(argidx++);

#ifdef FUSE_TERNARY
    // Fuse addcmul - read runtime addresses before setting out_addr_rt_arg_idx
    const uint32_t ternary_a_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t ternary_b_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t broadcast_ternary_b = get_arg_val<uint32_t>(argidx++);
#endif  // FUSE_TERNARY

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here (after ternary if present)

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<22>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr);
#endif

#ifdef FUSE_TERNARY
// Calculate offset for ternary_a_args - must account for FUSE_BIAS and potentially FUSE_AG
#if defined(FUSE_AG) && defined(READ_FROM_LOCAL_INPUT)
// If we have FUSE_AG with READ_FROM_LOCAL_INPUT, in3 is defined
#ifdef FUSE_BIAS
    // After in2, then in3, then ternary
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, in2_args_cta_offset>();
#else
    // After outputs, then in3, then ternary
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks + 1, out_tensor_args_cta_offset>();
#endif
#else
// No FUSE_AG, same as dm_in1_sender_out
#ifdef FUSE_BIAS
    constexpr uint32_t ternary_a_args_cta_offset = in2_args.next_compile_time_args_offset();

#else

    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
#endif
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

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N is half the matmul (weight) N
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_id = AG_OUT_WRITE_CB;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_in2_id = tt::CBIndex::c_4;
#endif

    CircularBuffer cb_in0(cb_in0_id);
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
            true,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule,
            backward_sem_addrs,
            forward_sem_addrs);
    }

#ifdef READ_FROM_LOCAL_INPUT
#ifdef FUSE_BIAS
    // in3 (local input) sits right after in2 (bias); advance by in2's real arg count, not a fixed constant.
    constexpr auto in3_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t in3_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in3_args = TensorAccessorArgs<in3_args_cta_offset>();
#endif
    const auto in3_reader = TensorAccessor(in3_args, in3_addr);
#endif
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    // OpSignaler runtime args start after output addresses and optional FUSE_AG args
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
#ifdef FUSE_AG
    // Skip MinimalMatmulFusedOpSignaler::push_matmul_fused_op_rt_args: 9 scalars + num_ag_workers + (2N+1) sems
    srs_fuse_signaler_rt_args_idx += (11 + 2 * get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 9));
#endif
    OpSignaler srs_fuse_signaler;
    uint32_t mm_progress_counters_base = 0;
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
        // Per-core signaling: base L1 address of the RS cores' per-core progress counter array
        mm_progress_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
    }
#endif

    in0_valid_sem.set(VALID);

    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
#if defined(FUSE_AG) && (IN0_SUB_CHUNKS > 1) && defined(AG_INTERLEAVE_BANDS)
                if constexpr (is_injector_core) {
                    // Interleave a forward remote k-block with the following backward one: read/push/mcast A.b0
                    if (n_block_iter == 0 && k_block_iter >= num_local_k_blocks && (k_block_iter + 1) < K_num_blocks &&
                        ((k_block_iter - num_local_k_blocks) & 1u) == 0) {
                        // Resolve both slots up front
                        const uint32_t kb_a =
                            fused_op_receiver.compute_actual_k_block_iter(true, k_block_iter, k_forward);
                        const uint8_t dir_a = fused_op_receiver.streamed_dir;
                        const uint32_t kb_b =
                            fused_op_receiver.compute_actual_k_block_iter(true, k_block_iter + 1, k_forward);
                        const uint32_t kb_pair2[2] = {kb_a, kb_b};
                        const uint8_t dir_pair[2] = {dir_a, fused_op_receiver.streamed_dir};
                        for (uint32_t band = 0; band < (uint32_t)IN0_SUB_CHUNKS; band++) {
                            uint32_t band_lo, band_h;
                            balanced_band(current_M_block_tiles, (uint32_t)IN0_SUB_CHUNKS, band, band_lo, band_h);
                            if (band_h == 0) {
                                break;
                            }
                            // Reserve a uniform M_block_tiles/IN0_SUB_CHUNKS-tile slot per band so each band tiles
                            const uint32_t band_slot_tiles = (M_block_tiles / (uint32_t)IN0_SUB_CHUNKS) * K_block_tiles;
                            const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                            const uint32_t band_start = m_tile + band_lo;
                            const uint32_t band_end = band_start + band_h;
                            for (uint32_t member = 0; member < 2; member++) {
                                cb_in0.reserve_back(band_slot_tiles);
                                uint32_t in0_start_address = get_write_ptr(cb_in0_id);
                                {
                                    // band 0's signal was consumed by compute_actual_k_block_iter above
                                    if (band > 0) {
                                        fused_op_receiver.wait_for_dir(dir_pair[member]);
                                    }
                                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                                        in0_reader,
                                        in0_shape,
                                        cb_in0_id,
                                        in0_tile_size,
#ifdef READ_FROM_LOCAL_INPUT
                                        in3_reader,
                                        fused_op_receiver.local_k_start,
                                        fused_op_receiver.local_k_end,
                                        fused_op_receiver.input_tensor_Wt,
#endif
                                        band_start,
                                        band_end,
                                        kb_pair2[member] * K_block_tiles,
                                        (kb_pair2[member] + 1) * K_block_tiles,
                                        /*issue_barrier=*/true);
                                }
                                cb_in0.push_back(band_slot_tiles);
                                if (!is_sink_core) {
                                    in0_sender_sem.wait(1);
                                    in0_sender_sem.set(0);
                                    uint64_t in0_unicast_data_addr =
                                        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                                    noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                                    noc.async_writes_flushed();
#endif
                                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                                }
                            }
                        }
#ifdef SRS_FUSE_OP_SIGNALER
                        if constexpr (is_output_writer) {
                            if (not_first_block && k_block_iter == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                            }
                            if (not_first_block && (k_block_iter + 1) == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                            }
                        }
#endif
                        k_block_iter++;  // the backward member of the pair is consumed here too
                        continue;
                    }
                }
#endif
#if defined(FUSE_AG) && (IN0_SUB_CHUNKS > 1) && defined(AG_INTERLEAVE_BANDS)
                if constexpr (!is_injector_core) {
                    // Receiver mirror of the injector's paired interleave above
                    if (n_block_iter == 0 && k_block_iter >= num_local_k_blocks && (k_block_iter + 1) < K_num_blocks &&
                        ((k_block_iter - num_local_k_blocks) & 1u) == 0) {
                        // Receivers can defer_write
                        if constexpr (is_output_writer) {
                            if (defer_write &&
                                (k_block_iter == defer_write_k_block || (k_block_iter + 1) == defer_write_k_block)) {
#ifdef FUSE_SWIGLU
                                cb_out.wait_front(out_block_num_tiles_swiglu);
                                uint32_t out_read_ptr_swiglu = get_read_ptr(cb_out_id);
                                if constexpr (N_chunks == 1) {
                                    write_block_sync<M_block_tiles, out_N_block_tiles>(
                                        std::get<0>(outputs_tuple),
                                        out_shape_swiglu,
                                        out_read_ptr_swiglu,
                                        out_tile_size,
                                        defer_write_m_tile,
                                        defer_write_m_tile_end,
                                        defer_write_n_tile / 2,
                                        defer_write_n_tile_end / 2);
                                } else {
                                    write_block_sync_split<
                                        M_block_tiles,
                                        out_N_block_tiles,
                                        N_chunks,
                                        out_N_tiles_per_chunk>(
                                        outputs_tuple,
                                        out0_shape_swiglu,
                                        out_read_ptr_swiglu,
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
                        for (uint32_t band = 0; band < (uint32_t)IN0_SUB_CHUNKS; band++) {
                            uint32_t band_lo, band_h;
                            balanced_band(current_M_block_tiles, (uint32_t)IN0_SUB_CHUNKS, band, band_lo, band_h);
                            if (band_h == 0) {
                                break;
                            }
                            const uint32_t band_slot_tiles = (M_block_tiles / (uint32_t)IN0_SUB_CHUNKS) * K_block_tiles;
                            const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                            for (uint32_t member = 0; member < 2; member++) {
                                cb_in0.reserve_back(band_slot_tiles);
                                uint32_t in0_start_address = get_write_ptr(cb_in0_id);
                                {
                                    in0_receiver_sem.set(INVALID);
                                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                                    in0_receiver_sem.wait(VALID);
                                }
                                cb_in0.push_back(band_slot_tiles);
                                if (!is_sink_core) {
                                    in0_sender_sem.wait(1);
                                    in0_sender_sem.set(0);
                                    uint64_t in0_unicast_data_addr =
                                        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                                    noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                                    noc.async_writes_flushed();
#endif
                                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                                }
                            }
                        }
#ifdef SRS_FUSE_OP_SIGNALER
                        if constexpr (is_output_writer) {
                            if (not_first_block && k_block_iter == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                            }
                            if (not_first_block && (k_block_iter + 1) == max_defer_write_k_block) {
                                noc.async_write_barrier();
                                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
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
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr_swiglu = get_read_ptr(cb_out_id);
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape_swiglu,
                                out_read_ptr_swiglu,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        } else {
                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape_swiglu,
                                out_read_ptr_swiglu,
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

                        // write_block_sync_split is more generic (support multiple output tensors) But for N_chunks
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

                // ---- Sub-chunked (banded) delivery ---- Each k-block position is delivered as `nb` M-row bands
                [[maybe_unused]] uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
#ifdef FUSE_AG
                if constexpr (is_injector_core) {
                    k_block = fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                }
                // streamed_dir is only meaningful on the injector; receivers use num_local_k_blocks below.
                [[maybe_unused]] const uint8_t k_dir = is_injector_core ? fused_op_receiver.streamed_dir : (uint8_t)2;
#else
                [[maybe_unused]] const uint8_t k_dir = 2;
#endif
                uint32_t nb;
                if constexpr (is_injector_core) {
                    nb = (n_block_iter == 0 && k_dir != 2) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
                } else {
                    nb = (n_block_iter == 0 && k_block_iter >= num_local_k_blocks) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
                }
                for (uint32_t band = 0; band < nb; band++) {
                    uint32_t band_lo, band_h;
                    balanced_band(current_M_block_tiles, nb, band, band_lo, band_h);
                    if (band_h == 0) {
                        break;  // only when nb > current_M_block_tiles
                    }
                    // Uniform per-band slot (see the interleave branches above)
                    const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                    const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                    cb_in0.reserve_back(band_slot_tiles);
                    uint32_t in0_start_address = get_write_ptr(cb_in0_id);
                    if constexpr (is_injector_core) {
                        const uint32_t band_start = m_tile + band_lo;
                        const uint32_t band_end = band_start + band_h;
#ifdef FUSE_AG
                        // band 0's signal was already awaited by compute_actual_k_block_iter above
                        if (nb > 1 && band > 0) {
                            fused_op_receiver.wait_for_dir(k_dir);
                        }
#endif
                        read_in0_block_sync<M_block_tiles, K_block_tiles>(
                            in0_reader,
                            in0_shape,
                            cb_in0_id,
                            in0_tile_size,
#ifdef READ_FROM_LOCAL_INPUT
                            in3_reader,
                            fused_op_receiver.local_k_start,
                            fused_op_receiver.local_k_end,
                            fused_op_receiver.input_tensor_Wt,
#endif
                            band_start,
                            band_end,
                            k_block * K_block_tiles,
                            (k_block + 1) * K_block_tiles,
                            /*issue_barrier=*/true);
                    } else {
                        in0_receiver_sem.set(INVALID);
                        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                        in0_receiver_sem.wait(VALID);
                    }

                    cb_in0.push_back(band_slot_tiles);

                    if (!is_sink_core) {
                        in0_sender_sem.wait(1);
                        in0_sender_sem.set(0);
                        uint64_t in0_unicast_data_addr =
                            get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                        noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                        noc.async_writes_flushed();
#endif
                        noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                    }
                }
#ifdef SRS_FUSE_OP_SIGNALER
                if constexpr (is_output_writer) {
                    if (not_first_block && k_block_iter == max_defer_write_k_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
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
#endif

            k_forward = !k_forward;

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
                    if constexpr (N_chunks == 1) {
#ifdef SPLIT_OUTPUT_WRITE
#ifdef AGMM_INTERLEAVED_OUTPUT_WRITE
                        // NOC_0 writer: drains the rows split_row_to_noc1() assigns to NOC_0 from c_8
                        write_block_sync_granular_interleaved<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end,
                            AG_SPLIT_NOC1_PCT,
                            /*own_noc1=*/false);
#else
                        // NOC_0 writer: high rows [split_rows, M) from c_8.
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
                            split_rows,
                            M_block_tiles);
#endif  // AGMM_INTERLEAVED_OUTPUT_WRITE
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
#ifdef SPLIT_OUTPUT_WRITE
                        // NOC_0 writer: high rows [split_rows, M) from c_8, demuxed into the chunk tensors.
                        constexpr uint32_t split_rows = (M_block_tiles * AG_SPLIT_NOC1_PCT) / 100;
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end,
                            split_rows,
                            M_block_tiles);
#else
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
#endif
                    }
#endif  // FUSE_SWIGLU
#ifdef SRS_FUSE_OP_SIGNALER
                    if (is_last_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                    }
#endif
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
