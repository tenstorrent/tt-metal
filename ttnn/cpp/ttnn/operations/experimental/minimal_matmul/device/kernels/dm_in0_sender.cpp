// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/subchunk_bands.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"  // TEMP: instrumentation for num_local_k_blocks / num_ag_workers

// Set by the host only when fusing AG; default to 1 (single whole band per k-block) otherwise.
#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

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
    // Leading (self/local) k-block positions this device owns. Receivers don't run the AG scheduler,
    // so they use this to compute the same per-position band count the injector derives from streamed_dir.
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
    uint32_t num_ag_workers = get_arg_val<uint32_t>(fused_op_rt_args_idx + 9);  // after the 9 scalar args
    // TEMP INSTRUMENTATION: confirm the fused-op arg base and num_local_k_blocks are sane. A huge
    // num_local_k_blocks (a DRAM address) confirms the override-callback off-by-one corrupted it.
    DPRINT(
        "AGMM-DM0 fidx={} nlkb={} ndev={} nkb={} nwrk={}\n",
        fused_op_rt_args_idx,
        num_local_k_blocks,
        num_devices,
        num_k_blocks,
        num_ag_workers);
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
    // Skip MinimalMatmulFusedOpSignaler::push_matmul_fused_op_rt_args: 9 scalars + num_ag_workers + (2N+1) sems
    srs_fuse_signaler_rt_args_idx += (11 + 2 * get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 9));
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
                DeviceZoneScopedN("AVAILABLE");
#if defined(FUSE_AG) && (IN0_SUB_CHUNKS > 1) && defined(AG_INTERLEAVE_BANDS)
                if constexpr (is_injector_core) {
                    // Interleave a forward remote k-block with the following backward one: read/push/mcast
                    // A.b0, B.b0, A.b1, B.b1, ... Position-based pairing matches compute and dm_in1 so the
                    // per-band in0 CB counts stay in lockstep. Injectors never defer_write, so the deferred
                    // output flush below never applies on this path.
                    if (n_block_iter == 0 && k_block_iter >= num_local_k_blocks && (k_block_iter + 1) < K_num_blocks &&
                        ((k_block_iter - num_local_k_blocks) & 1u) == 0) {
                        // Resolve both slots up front; each call advances the schedule and updates
                        // streamed_dir, so capture the forward direction before resolving the backward slot.
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
                            // Reserve a uniform M_block_tiles/IN0_SUB_CHUNKS-tile slot per band so each band
                            // tiles the in0 CB exactly (no fifo wrap), but forward only the band_h real tiles.
                            // The receiver mirrors this member-inner order (see the !is_injector_core branch),
                            // so exact band_bytes forwards stay in lockstep -- no slack tiles on the wire.
                            const uint32_t band_slot_tiles = (M_block_tiles / (uint32_t)IN0_SUB_CHUNKS) * K_block_tiles;
                            const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                            const uint32_t band_start = m_tile + band_lo;
                            const uint32_t band_end = band_start + band_h;
                            for (uint32_t member = 0; member < 2; member++) {
                                cb_reserve_back(cb_id_in0, band_slot_tiles);
                                uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                                {
                                    DeviceZoneScopedN("DRAM-Latency");
#ifndef SKIP_IN0_DRAM_READ
                                    // band 0's signal was consumed by compute_actual_k_block_iter above; later
                                    // bands wait this member's own per-direction aggregator signal.
                                    if (band > 0) {
                                        DeviceZoneScopedN("IN0-BAND-WAIT");
                                        fused_op_receiver.wait_for_dir(dir_pair[member]);
                                    }
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
                                        band_start,
                                        band_end,
                                        kb_pair2[member] * K_block_tiles,
                                        (kb_pair2[member] + 1) * K_block_tiles,
                                        /*issue_barrier=*/true);
#endif
                                }
                                cb_push_back(cb_id_in0, band_slot_tiles);
                                if (!is_sink_core) {
                                    DeviceZoneScopedN("MCAST-SEND");
                                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
                                    uint64_t in0_unicast_data_addr =
                                        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                                    noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                                    noc_async_writes_flushed();
#endif
                                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                                }
                            }
                        }
#ifdef SRS_FUSE_OP_SIGNALER
                        if constexpr (is_output_writer) {
                            if (not_first_block && k_block_iter == max_defer_write_k_block) {
                                noc_async_write_barrier();
                                srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                            }
                            if (not_first_block && (k_block_iter + 1) == max_defer_write_k_block) {
                                noc_async_write_barrier();
                                srs_fuse_signaler.synchronize_workers_and_signal_op(0);
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
                    // Receiver mirror of the injector's paired interleave (the is_injector_core block above).
                    // The injector mcasts bands member-inner (A.b0, B.b0, A.b1, B.b1, ...); we must recv AND
                    // forward in that SAME order. The sequential non-interleave loop below walks the two
                    // k-blocks separately (A.b0,A.b1,... then B.b0,...), so it would forward each uniform slot
                    // with the wrong (sequential) band_h and drop the last tile of band 0's backward member.
                    // Same 2*nb recv/forwards, exact band_bytes each -> no extra fabric sends; only the
                    // per-slot band_h order changes to match the slot content.
                    if (n_block_iter == 0 && k_block_iter >= num_local_k_blocks && (k_block_iter + 1) < K_num_blocks &&
                        ((k_block_iter - num_local_k_blocks) & 1u) == 0) {
                        // Receivers (unlike injectors) can defer_write, and this branch consumes both
                        // k_block_iter and k_block_iter+1, bypassing the top-of-loop flush below via continue.
                        // Honor a flush scheduled on either paired position here.
                        if constexpr (is_output_writer) {
                            if (defer_write &&
                                (k_block_iter == defer_write_k_block || (k_block_iter + 1) == defer_write_k_block)) {
                                DeviceZoneScopedN("DEFER-WRITE");
                                cb_wait_front(cb_id_out, out_block_num_tiles);
                                uint32_t out_read_ptr = get_read_ptr(cb_id_out);
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
                        for (uint32_t band = 0; band < (uint32_t)IN0_SUB_CHUNKS; band++) {
                            uint32_t band_lo, band_h;
                            balanced_band(current_M_block_tiles, (uint32_t)IN0_SUB_CHUNKS, band, band_lo, band_h);
                            if (band_h == 0) {
                                break;
                            }
                            const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                            const uint32_t band_slot_tiles = (M_block_tiles / (uint32_t)IN0_SUB_CHUNKS) * K_block_tiles;
                            for (uint32_t member = 0; member < 2; member++) {
                                cb_reserve_back(cb_id_in0, band_slot_tiles);
                                uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                                {
                                    DeviceZoneScopedN("RECV-WAIT");
                                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                                }
                                cb_push_back(cb_id_in0, band_slot_tiles);
                                if (!is_sink_core) {
                                    DeviceZoneScopedN("MCAST-SEND");
                                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
                                    uint64_t in0_unicast_data_addr =
                                        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                                    noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                                    noc_async_writes_flushed();
#endif
                                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                                }
                            }
                        }
#ifdef SRS_FUSE_OP_SIGNALER
                        if constexpr (is_output_writer) {
                            if (not_first_block && k_block_iter == max_defer_write_k_block) {
                                noc_async_write_barrier();
                                srs_fuse_signaler.synchronize_workers_and_signal_op(0);
                            }
                            if (not_first_block && (k_block_iter + 1) == max_defer_write_k_block) {
                                noc_async_write_barrier();
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
                        DeviceZoneScopedN("DEFER-WRITE");
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

                // ---- Sub-chunked (banded) delivery ----
                // Each k-block position is delivered as `nb` M-row bands. Remote positions on the first
                // N-block use IN0_SUB_CHUNKS bands (matching the AG's per-band signalling); local positions
                // and every position on later N-blocks use a single whole-block band. Reserve/read (or
                // recv)/push/forward happen per band so the matmul and the downstream forward pipeline at
                // band granularity. N-stride in0 reuse is intentionally disabled here (every position is
                // delivered fresh), so there is no reuse skip on this path.
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
                    // Reserve a uniform slot of M_block_tiles/nb tiles for every band (not the ragged
                    // band_h) so each band tiles the in0 CB exactly and no reservation wraps fifo_limit
                    // mid-block -- a wrap would straddle a single mcast noc_async_write into the
                    // mailboxes and wedge the mesh. band_h real tiles are read; the slot's remaining tiles
                    // are slack that compute reads deep but never packs.
                    const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                    // Forward exactly band_h tiles. This path serves single-k-block positions (local, or an
                    // unpaired remote tail) where injector and receiver both walk bands sequentially, so the
                    // per-slot band_h already matches -- the paired fwd/bwd interleave (member-inner order)
                    // is handled by the dedicated injector/receiver branches above.
                    const uint32_t band_bytes = band_h * K_block_tiles * in0_tile_size;
                    cb_reserve_back(cb_id_in0, band_slot_tiles);
                    uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                    if constexpr (is_injector_core) {
                        DeviceZoneScopedN("DRAM-Latency");
#ifndef SKIP_IN0_DRAM_READ
                        const uint32_t band_start = m_tile + band_lo;
                        const uint32_t band_end = band_start + band_h;
#ifdef FUSE_AG
                        // band 0's signal was already awaited by compute_actual_k_block_iter above; each
                        // later band waits its own aggregator signal.
                        if (nb > 1 && band > 0) {
                            DeviceZoneScopedN("IN0-BAND-WAIT");
                            fused_op_receiver.wait_for_dir(k_dir);
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
                            band_start,
                            band_end,
                            k_block * K_block_tiles,
                            (k_block + 1) * K_block_tiles,
                            /*issue_barrier=*/true);
#else
                        (void)k_block;
#endif
                    } else {
                        DeviceZoneScopedN("RECV-WAIT");
                        noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                    }

                    cb_push_back(cb_id_in0, band_slot_tiles);

                    if (!is_sink_core) {
                        DeviceZoneScopedN("MCAST-SEND");
                        noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                        noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
                        uint64_t in0_unicast_data_addr =
                            get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                        noc_async_write(in0_start_address, in0_unicast_data_addr, band_bytes);
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                    }
                }
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
                    DeviceZoneScopedN("OUT-WRITE");
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
