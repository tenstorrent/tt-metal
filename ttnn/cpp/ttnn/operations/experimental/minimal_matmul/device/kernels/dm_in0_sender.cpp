// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

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
        in2_args_cta_offset + tensor_accessor::detail::NUM_TENSOR_ACCESSOR_ARGS() * 2;
#else
    // After outputs, then in3, then ternary
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>() +
        tensor_accessor::detail::NUM_TENSOR_ACCESSOR_ARGS();
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
    constexpr uint32_t cb_id_ternary_a = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_ternary_b = tt::CBIndex::c_6;

    constexpr uint32_t ternary_a_tile_size = get_tile_size(cb_id_ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(cb_id_ternary_b);

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

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

#ifdef FUSE_AG
    // Receiver for ccl fusing
    MinimalMatmulOpReceiver fused_op_receiver;
    uint32_t fused_op_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
    uint32_t num_devices = get_arg_val<uint32_t>(fused_op_rt_args_idx);
    uint32_t num_k_blocks = get_arg_val<uint32_t>(fused_op_rt_args_idx + 1);
    uint8_t k_block_device_expected[num_k_blocks]{};
    uint8_t k_block_device_received[num_k_blocks]{};
    uint32_t device_k_block_counts[num_devices]{};
    uint32_t device_k_block_start_ids[num_devices]{};
    uint32_t forward_k_block_schedule[num_k_blocks]{};
    if constexpr (is_injector_core) {
        fused_op_receiver = MinimalMatmulOpReceiver(
            true,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule);
    }

#ifdef READ_FROM_LOCAL_INPUT
#ifdef FUSE_BIAS
    constexpr auto in3_args =
        TensorAccessorArgs<in2_args_cta_offset + tensor_accessor::detail::NUM_TENSOR_ACCESSOR_ARGS>();
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
    srs_fuse_signaler_rt_args_idx += 12;  // Skip MinimalMatmulFusedOpSignaler::push_matmul_fused_op_rt_args (12 args)
#endif
    OpSignaler srs_fuse_signaler;
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
    }
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

#ifdef CREDIT_FORWARD
    // Credit-based forwarding: cumulative counters (semaphores never reset). The receiver pre-grants
    // CREDIT_FORWARD credits to its predecessor (= # free CB slots), letting the sender run that many
    // blocks ahead; afterwards it replenishes one credit per slot the compute frees. recv_sem counts
    // delivered blocks; sender_sem counts granted credits. Both start at INVALID(=0).
    uint32_t cf_blocks_received = 0;  // blocks consumed from predecessor (this core as receiver)
    uint32_t cf_blocks_sent = 0;      // blocks delivered to successor (this core as sender)
    if constexpr (!is_injector_core) {
        // Pre-grant the initial double-buffer slots so the predecessor can fill them without waiting.
        noc_semaphore_inc(in0_sender_semaphore_noc_addr, CREDIT_FORWARD);
    }
#endif

#ifdef MCAST_BROADCAST
    // Multicast-broadcast prototype args, appended after the output addresses (mcast is incompatible
    // with fused ops, so N_chunks output addresses are the only thing between the fixed args and these).
    const uint32_t mc_base = out_addr_rt_arg_idx + N_chunks;
    const uint32_t in0_mc_start_x = get_arg_val<uint32_t>(mc_base + 0);
    const uint32_t in0_mc_start_y = get_arg_val<uint32_t>(mc_base + 1);
    const uint32_t in0_mc_end_x = get_arg_val<uint32_t>(mc_base + 2);
    const uint32_t in0_mc_end_y = get_arg_val<uint32_t>(mc_base + 3);
    const uint32_t in0_num_recv = get_arg_val<uint32_t>(mc_base + 4);
    const uint32_t in0_inj_noc_x = get_arg_val<uint32_t>(mc_base + 5);
    const uint32_t in0_inj_noc_y = get_arg_val<uint32_t>(mc_base + 6);
    // Receivers signal readiness to the injector's sender semaphore (same L1 offset on every core).
    const uint64_t in0_injector_sender_sem_noc_addr =
        get_noc_addr(in0_inj_noc_x, in0_inj_noc_y, in0_sender_semaphore_addr);
#ifdef MCAST_PIPELINED
    // Per-receiver credit counters live in a small L1 scratch CB (one 16B-strided slot per group
    // member). Slot s is on the injector's core; receiver s incs it (cumulative grant), injector polls.
    constexpr uint32_t cb_id_in0_credit = tt::CBIndex::c_5;
    const uint32_t in0_credit_base = get_write_ptr(cb_id_in0_credit);
    const uint32_t in0_my_group_idx = get_arg_val<uint32_t>(mc_base + 7);
    const uint64_t in0_my_credit_noc_addr =
        get_noc_addr(in0_inj_noc_x, in0_inj_noc_y, in0_credit_base + in0_my_group_idx * MCAST_CREDIT_STRIDE);
    uint32_t mc_blocks_received = 0;
    uint32_t mc_blocks_sent = 0;
    // Startup barrier: injector zeroes its credit slots then mcasts a "go" (reuses the otherwise-unused
    // sender semaphore) so no receiver pre-grants before the slots are zeroed (else a grant is lost).
    if constexpr (is_injector_core) {
        for (uint32_t i = 0; i <= in0_num_recv; i++) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_credit_base + i * MCAST_CREDIT_STRIDE) = 0;
        }
        uint64_t go_mcast = get_noc_multicast_addr(
            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_sender_semaphore_addr);
        noc_semaphore_set_multicast(in0_valid_semaphore_addr, go_mcast, in0_num_recv);
    } else {
        noc_semaphore_wait(in0_sender_semaphore_addr_ptr, VALID);
        // Pre-grant the double-buffer slots so the injector can mcast that many blocks ahead.
        noc_semaphore_inc(in0_my_credit_noc_addr, MCAST_PIPELINED);
    }
#endif
#endif

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;
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
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        // When striding M block, in0 gets no reuse
        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);

#ifdef MCAST_PREFETCH
            // Software-pipelined injector k-loop: issue block (k+1)'s DRAM read BEFORE block k's
            // multicast, overlapping read latency with the mcast handshake+transit on the single DM
            // RISC. Receivers are unchanged (same per-block valid sequence). Simple regime only
            // (mcast => no FUSE_AG/BIAS/TERNARY, injectors never defer-write); reuse handled via k_start.
            if constexpr (is_injector_core) {
                uint32_t k_start = reuse_block ? 1 : 0;
                reuse_block = false;
                if (k_start < K_num_blocks) {
                    // Prologue: reserve a slot and ISSUE the first block's read (no barrier yet).
                    cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                    uint32_t wp = get_write_ptr(cb_id_in0);
                    {
                        uint32_t kb = k_forward ? k_start : (K_num_blocks - 1) - k_start;
                        read_in0_block_sync<M_block_tiles, K_block_tiles, /*issue_only=*/true>(
                            in0_reader,
                            in0_shape,
                            wp,
                            in0_tile_size,
                            m_tile,
                            m_tile_end,
                            kb * K_block_tiles,
                            (kb + 1) * K_block_tiles);
                    }
                    for (uint32_t k_block_iter = k_start; k_block_iter < K_num_blocks; k_block_iter++) {
                        noc_async_read_barrier();                      // current block's read completes
                        uint32_t mcast_wp = get_write_ptr(cb_id_in0);  // slot of current block (pre-push)
                        cb_push_back(cb_id_in0, in0_block_num_tiles);  // hand current block to compute

                        // Prefetch: reserve the next slot and ISSUE the next block's read so it flies
                        // during the mcast below. CB depth >= 2 => next slot is distinct from mcast_wp.
                        if (k_block_iter + 1 < K_num_blocks) {
                            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                            wp = get_write_ptr(cb_id_in0);
                            uint32_t kn = k_block_iter + 1;
                            uint32_t kb = k_forward ? kn : (K_num_blocks - 1) - kn;
                            read_in0_block_sync<M_block_tiles, K_block_tiles, /*issue_only=*/true>(
                                in0_reader,
                                in0_shape,
                                wp,
                                in0_tile_size,
                                m_tile,
                                m_tile_end,
                                kb * K_block_tiles,
                                (kb + 1) * K_block_tiles);
                        }

                        // Multicast the current block (non-pipelined broadcast handshake). The flush
                        // guarantees the source slot is read before it can be reused (block k+2 = same slot).
                        noc_semaphore_wait(in0_sender_semaphore_addr_ptr, in0_num_recv);
                        noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
                        uint64_t mcast_data_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, mcast_wp);
                        noc_async_write_multicast(mcast_wp, mcast_data_addr, current_block_bytes, in0_num_recv);
                        noc_async_writes_flushed();
                        uint64_t mcast_valid_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_receiver_semaphore_addr);
                        noc_semaphore_set_multicast(in0_valid_semaphore_addr, mcast_valid_addr, in0_num_recv);
                    }
                }
            } else
#endif
                for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                    if (defer_write && k_block_iter == defer_write_k_block) {
                        if constexpr (is_output_writer) {
                            cb_wait_front(cb_id_out, out_block_num_tiles);
                            uint32_t out_read_ptr = get_read_ptr(cb_id_out);

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
                            cb_pop_front(cb_id_out, out_block_num_tiles);
                        }
                    }

                    if (reuse_block && k_block_iter == 0) {
                        // We strided an N block and this is the first k block, so we get reuse and do not need to read
                        // in0
                        reuse_block = false;
                        continue;
                    }
                    uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                    cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                    uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                // DIRECT_DRAM_READ (experiment): every core reads its own in0 block straight from DRAM
                // (no systolic forwarding chain). Costs ~Nx more DRAM read bytes but removes the
                // per-hop semaphore handshake latency that dominates small/skewed shapes.
#ifdef DIRECT_DRAM_READ
                    if constexpr (true) {
#else
                if constexpr (is_injector_core) {
#endif
#ifdef FUSE_AG
                    if (is_injector_core) {
                        k_block =
                            fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                    }
#endif
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        in0_tile_size,
#ifdef READ_FROM_LOCAL_INPUT
                        in3_reader,
                        fused_op_receiver.local_k_start,
                        fused_op_receiver.local_k_end,
                        fused_op_receiver.input_tensor_Wt,
#endif
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                // Get from previous device
                // ABLATE_INTERCORE (perf analysis only): skip the inter-core receive handshake;
                // CB still pushed below so compute proceeds on stale L1 (measures pure compute floor).
#ifndef ABLATE_INTERCORE
#ifdef MCAST_BROADCAST
#ifdef MCAST_PIPELINED
                    // Replenish a credit once past the pre-granted depth (this reserve only returned
                    // because compute freed a slot), then wait for the injector's cumulative broadcast.
                    if (mc_blocks_received >= MCAST_PIPELINED) {
                        noc_semaphore_inc(in0_my_credit_noc_addr, 1);
                    }
                    noc_semaphore_wait_min(in0_receiver_semaphore_addr_ptr, mc_blocks_received + 1);
                    mc_blocks_received++;
#else
                    // Signal readiness (slot free) to the injector, then wait for its broadcast.
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_injector_sender_sem_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
#endif
#elif defined(CREDIT_FORWARD)
                    // cb_reserve_back above just returned -> a slot is free. The first CREDIT_FORWARD
                    // slots were already granted up front (pre-grant); beyond that, this reserve only
                    // returned because compute freed a slot, so replenish one credit to the predecessor.
                    if (cf_blocks_received >= CREDIT_FORWARD) {
                        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    }
                    noc_semaphore_wait_min(in0_receiver_semaphore_addr_ptr, cf_blocks_received + 1);
                    cf_blocks_received++;
#else
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
#endif
#endif
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in0, in0_block_num_tiles);

                // ABLATE_INTERCORE / DIRECT_DRAM_READ: skip the L1->L1 unicast forward + handshake.
#if !defined(ABLATE_INTERCORE) && !defined(DIRECT_DRAM_READ)
#ifdef MCAST_BROADCAST
                    // Only the injector broadcasts: wait until all receivers signaled a free slot, then
                    // one multicast write of the block + one multicast set of their valid semaphores.
                    if constexpr (is_injector_core) {
#ifdef MCAST_PIPELINED
                        // Wait until EVERY receiver has granted a credit for this block (min over receivers,
                        // not a sum) so the target slot is free on all of them, then broadcast.
                        for (uint32_t i = 1; i <= in0_num_recv; i++) {
                            volatile tt_l1_ptr uint32_t* credit_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                in0_credit_base + i * MCAST_CREDIT_STRIDE);
                            noc_semaphore_wait_min(credit_ptr, mc_blocks_sent + 1);
                        }
                        uint64_t mcast_data_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_start_address);
                        noc_async_write_multicast(
                            in0_start_address, mcast_data_addr, current_block_bytes, in0_num_recv);
                        noc_async_writes_flushed();

                        // Cumulative valid = block count; receivers wait_min(recv_sem >= their block + 1).
                        noc_semaphore_set(in0_valid_semaphore_addr_ptr, mc_blocks_sent + 1);
                        uint64_t mcast_valid_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_receiver_semaphore_addr);
                        noc_semaphore_set_multicast(in0_valid_semaphore_addr, mcast_valid_addr, in0_num_recv);
                        mc_blocks_sent++;
#else
                        noc_semaphore_wait(in0_sender_semaphore_addr_ptr, in0_num_recv);
                        noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                        uint64_t mcast_data_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_start_address);
                        noc_async_write_multicast(
                            in0_start_address, mcast_data_addr, current_block_bytes, in0_num_recv);
                        // Ensure the source L1 read completed before the buffer can be reused next block.
                        noc_async_writes_flushed();

                        uint64_t mcast_valid_addr = get_noc_multicast_addr(
                            in0_mc_start_x, in0_mc_start_y, in0_mc_end_x, in0_mc_end_y, in0_receiver_semaphore_addr);
                        noc_semaphore_set_multicast(in0_valid_semaphore_addr, mcast_valid_addr, in0_num_recv);
#endif
                    }
#else
                    if (!is_sink_core) {
#ifdef CREDIT_FORWARD
                        // Wait for a credit from the successor (cumulative count >= blocks_sent+1), then send.
                        noc_semaphore_wait_min(in0_sender_semaphore_addr_ptr, cf_blocks_sent + 1);
#else
                        noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                        noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
#endif

                        uint64_t in0_unicast_data_addr =
                            get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);

                        /**
                         * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                         * padded tiles. Use `current_block_bytes`.
                         */
                        noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

#ifdef CREDIT_FORWARD
                    // Signal delivery to the successor (cumulative). It waits recv_sem >= its block count.
                    noc_semaphore_inc(in0_receiver_semaphore_noc_addr, 1);
                    cf_blocks_sent++;
#else
                        noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
#endif
                    }
#endif
#endif
#ifdef SRS_FUSE_OP_SIGNALER
                if constexpr (is_output_writer) {
                    if (not_first_block && k_block_iter == max_defer_write_k_block) {
                        noc_async_write_barrier();
                        srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                    }
                }
#endif
                }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_async_read_page(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

#ifdef FUSE_TERNARY
            if constexpr (!is_output_writer) {
                read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                    ternary_a_reader,
                    ternary_b_reader,
                    out_shape,
                    cb_id_ternary_a,
                    cb_id_ternary_b,
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
            // We get reuse on in0 when striding N block
            reuse_block = true;

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
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
#ifdef SRS_FUSE_OP_SIGNALER
                    if (is_last_block) {
                        noc_async_write_barrier();
                        srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                    }
#endif
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
