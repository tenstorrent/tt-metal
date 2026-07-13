// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_combine.h"
#include "noc_parameters.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Positional compile-time args (fused GroupNorm). Scalars [0..18] mirror the welford GN
    // reader's named args; [19..26] add the fabric all-gather (Chan-merge across cluster_axis);
    // the TensorAccessor blocks (src0, then stats DRAM) start at index 27.
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);
    constexpr uint32_t num_batches = get_compile_time_arg_val(4);
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(6);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t tile_height = get_compile_time_arg_val(9);
    constexpr uint32_t tile_width = get_compile_time_arg_val(10);

    constexpr uint32_t block_h = get_compile_time_arg_val(11);
    constexpr uint32_t block_w = get_compile_time_arg_val(12);

    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(13);

    constexpr uint32_t num_out_blocks = get_compile_time_arg_val(14);
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr uint32_t num_channels_per_group = get_compile_time_arg_val(15);
    constexpr uint32_t num_rows_per_group = get_compile_time_arg_val(16);

    constexpr uint32_t cb_in0_welford_arg = get_compile_time_arg_val(17);
    constexpr bool welford_fp32_alias_arg = get_compile_time_arg_val(18) != 0;

    // Fabric all-gather (cross-device Chan merge). ring_size==1 => local (identity), no fabric.
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t num_chunks_per_device = get_compile_time_arg_val(21);
    constexpr uint32_t fwd_arrival_sem_id = get_compile_time_arg_val(22);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(23);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t stats_local_cb_id = get_compile_time_arg_val(25);
    constexpr uint32_t stats_gathered_cb_id = get_compile_time_arg_val(26);
    // Per-device element count per group (equal across devices for spatial shards). Used as the
    // per-subgroup COUNT_PER_VALUE in the cross-device Welford combine. num_rows_per_group is the
    // PER-CORE row count, so the full-device count multiplies by the mcast-group core count (the
    // spatial rows are split across num_mcast_cores). At single-core (num_mcast_cores==1) this is
    // unchanged.
    constexpr uint32_t count_per_device = num_channels_per_group * num_rows_per_group * num_mcast_cores;
    constexpr bool is_distributed = ring_size > 1;

    constexpr auto src0_args = TensorAccessorArgs<27>();
    constexpr auto stats_dram_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);

    const bool has_mcast_first_group = get_arg_val<uint32_t>(5);
    const bool has_mcast_last_group = get_arg_val<uint32_t>(6);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(7);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(8);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(9);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(10);
    const uint32_t num_mcast_cores_mid_group = get_arg_val<uint32_t>(11);

    // first mcast group
    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    // last mcast group
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;

    tt_l1_ptr uint32_t* noc_coord_x;
    tt_l1_ptr uint32_t* noc_coord_y;

    // number of cores in mcast groups
    uint32_t num_mcast_cores_first_group;
    uint32_t num_mcast_cores_last_group;

    // first and last group mcast coordinates passed directly in async_write_multicast calls below

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(17);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(18);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(19);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(20);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(21);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(22));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(22 + num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else {
        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));
    }

    Noc noc;
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    reduce_sender_sem.set(VALID);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
    // Welford-fp32 alias for cb_in0. Shares SRAM with cb_in0 but has its own buffer index
    // configured with UnpackToDestFp32, plus its own read/write pointers.
    // The Welford section of compute reads the alias to get full fp32 into DEST, while later
    // FPU consumers read cb_in0 directly. When welford_fp32_alias is false, cb_in0_welford_id
    // == cb_in0_id and the gated pushes below are skipped.
    constexpr uint32_t cb_in0_welford_id = cb_in0_welford_arg;
    constexpr bool welford_fp32_alias = welford_fp32_alias_arg;

    CircularBuffer cb_ex_partial(cb_ex_partial_id);
    CircularBuffer cb_ex_global(cb_ex_global_id);
    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_in0_welford(cb_in0_welford_id);
    CircularBuffer cb_repack(cb_repack_id);
    CircularBuffer cb_repack_out(cb_repack_out_id);
    CircularBuffer cb_out0(cb_out0_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    constexpr uint32_t src0_tile_bytes = get_tile_size(cb_in0_id);

    constexpr uint32_t local_stride = 2;
    constexpr uint32_t global_stride = NOC_L1_READ_ALIGNMENT_BYTES / 2;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

    const auto src_a = TensorAccessor(src0_args, src_addr);

    // ---- Fabric all-gather setup (cross-device Welford / Chan merge over cluster_axis) ----
    // The host appends the fabric AG args AFTER the variable-length noc_coord_x / noc_coord_y lists
    // (each num_mcast_cores long). Those lists start at noc_arg_base — 12, 17, or 22 depending on
    // which first/last mcast sub-groups are present — so the fabric block begins at
    // noc_arg_base + 2*num_mcast_cores. (For single-core num_mcast_cores==1 this reduces to 14, the
    // legacy fixed offset.) Only present / used when is_distributed.
    uint32_t stats_dram_addr = 0;
    uint32_t fwd_x = 0, fwd_y = 0, my_slot = 0, my_forwarder_index = 0;
    uint32_t fwd_packet_buf_addr = 0, fwd_arrival_sem_addr = 0, go_sem_addr = 0;
    if constexpr (is_distributed) {
        const uint32_t noc_arg_base = (has_mcast_first_group && has_mcast_last_group)   ? 22u
                                      : (has_mcast_first_group || has_mcast_last_group) ? 17u
                                                                                        : 12u;
        const uint32_t fabric_base = noc_arg_base + 2u * num_mcast_cores;
        stats_dram_addr = get_arg_val<uint32_t>(fabric_base + 0);
        fwd_x = get_arg_val<uint32_t>(fabric_base + 1);
        fwd_y = get_arg_val<uint32_t>(fabric_base + 2);
        my_slot = get_arg_val<uint32_t>(fabric_base + 3);
        my_forwarder_index = get_arg_val<uint32_t>(fabric_base + 4);
        fwd_packet_buf_addr = get_write_ptr(packet_cb_id);
        fwd_arrival_sem_addr = get_semaphore(fwd_arrival_sem_id);
        go_sem_addr = get_semaphore(go_sem_id);
    }
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
    }

    uint32_t index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            } else {
                out_block_h_actual = out_block_h_normal;
            }

#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                    if constexpr (welford_fp32_alias) {
                        // Mirror the cb_in0 push on the alias. They share SRAM (multi-buffer-index
                        // alias) so the noc.async_read above already filled both views; this is
                        // purely bookkeeping so compute's welford section can wait_front
                        // on cb_in0_welford independently of cb_in0.
                        cb_in0_welford.reserve_back(1);
                        cb_in0_welford.push_back(1);
                    }
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }

        cb_ex_partial.wait_front(2);
        auto local_means_ptr = cb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        cb_ex_global.reserve_back(2 * num_groups);
        const auto global_base_ptr = cb_ex_global.get_write_ptr();
        auto global_means_ptr = global_base_ptr;
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        // Per-device stats stick (bf16 [mean, var] per group), staged for the fabric AG.
        volatile tt_l1_ptr uint16_t* stick_u16 = nullptr;
        if constexpr (is_distributed) {
            cb_reserve_back(stats_local_cb_id, 1);
            stick_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(stats_local_cb_id));
        }

        // Batched intra-device handshake: wait ONCE for every receiver to have published ALL its
        // groups' partials (each receiver signals once, after its per-group loop). The mcast-back is
        // deferred to a single batched broadcast after the fabric exchange, so the per-group
        // signal/wait lock-step of stock GN (which would deadlock a single exchange) is batched here
        // and in the receiver. This is a sync-granularity change only — the arithmetic and the
        // mcast'd data are identical to stock, so ring_size==1 stays bit-exact.
        if constexpr (num_mcast_cores > 1) {
            reduce_receiver_sem.wait(num_mcast_cores - 1);
            reduce_receiver_sem.set(0);
        }

        for (uint32_t m = 0; m < num_groups; ++m) {
            // Local (per-core) Welford combine of this core's rows for group m.
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);
            auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);

            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            // Intra-device combine: NoC-read every peer's per-core partial for group m into scratch
            // past our own slot, then Welford-combine all num_mcast_cores partials -> device-global.
            if constexpr (num_mcast_cores > 1) {
                for (uint32_t i = 1; i < num_mcast_cores; ++i) {
                    UnicastEndpoint remote_ep;
                    noc.async_read(
                        remote_ep,
                        CoreLocalMem<uint32_t>(global_means_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES),
                        NOC_L1_READ_ALIGNMENT_BYTES,
                        {.noc_x = noc_coord_x[i], .noc_y = noc_coord_y[i], .addr = global_means_ptr},
                        {});
                    noc.async_read(
                        remote_ep,
                        CoreLocalMem<uint32_t>(global_vars_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES),
                        NOC_L1_READ_ALIGNMENT_BYTES,
                        {.noc_x = noc_coord_x[i], .noc_y = noc_coord_y[i], .addr = global_vars_ptr},
                        {});
                }
                noc.async_read_barrier();
            }

            auto global_result =
                combine_welford_stats<num_mcast_cores, num_channels_per_group * num_rows_per_group, global_stride>(
                    p_global_means, p_global_vars);
            p_global_means[0] = global_result.mean;
            p_global_vars[0] = global_result.variance;

            // Stage the DEVICE-GLOBAL stat (post intra-device combine) for the fabric AG — NOT the
            // per-core local, so the cross-device Chan merge sees full per-device statistics.
            if constexpr (is_distributed) {
                stick_u16[m * 2 + 0] = global_result.mean;
                stick_u16[m * 2 + 1] = global_result.variance;
            }

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

        // ---- Cross-device all-gather + Welford (Chan) merge over cluster_axis ----
        // cb_ex_global holds this device's per-group DEVICE-GLOBAL (mean, var). Publish this master's
        // sub-stick, ring-gather every device's, and Chan-merge into the GLOBAL (mean, var).
        if constexpr (is_distributed) {
            // 1. Publish this master's sub-stick to the forwarder packet buffer + signal arrival.
            cb_push_back(stats_local_cb_id, 1);
            cb_wait_front(stats_local_cb_id, 1);
            const uint32_t local_stick_addr = get_read_ptr(stats_local_cb_id);
            const uint64_t dst_noc = safe_get_noc_addr(fwd_x, fwd_y, fwd_packet_buf_addr + my_slot * stick_bytes, 0);
            noc_async_write(local_stick_addr, dst_noc, stick_bytes);
            noc_async_write_barrier();
            const uint64_t fwd_arrival_noc = safe_get_noc_addr(fwd_x, fwd_y, fwd_arrival_sem_addr, 0);
            noc_semaphore_inc(fwd_arrival_noc, 1);
            noc_async_atomic_barrier();
            cb_pop_front(stats_local_cb_id, 1);

            // 2. Wait for the ring gather to land in DRAM (forwarder increments our go-sem).
            volatile tt_l1_ptr uint32_t* go_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem_addr);
            noc_semaphore_wait_min(go_sem_ptr, 1);
            noc_semaphore_set(go_sem_ptr, 0);

            // 3. Read the ring_size sub-sticks (our slot) from DRAM into the gathered CB.
            cb_reserve_back(stats_gathered_cb_id, ring_size);
            const uint32_t gbase = get_write_ptr(stats_gathered_cb_id);
            for (uint32_t d = 0; d < ring_size; d++) {
                const uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index;
                const uint64_t src = get_noc_addr(page_idx, stats_dram, my_slot * stick_bytes);
                noc_async_read(src, gbase + d * stick_bytes, stick_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(stats_gathered_cb_id, ring_size);

            // 4. Chan-merge the ring_size per-device stats per group; overwrite cb_ex_global.
            cb_wait_front(stats_gathered_cb_id, ring_size);
            auto gathered_u16 = reinterpret_cast<volatile uint16_t*>(gbase);
            constexpr uint32_t stick_stride_u16 = stick_bytes / 2;  // uint16 elements per device stick
            auto out_ptr = global_base_ptr;
            for (uint32_t m = 0; m < num_groups; ++m) {
                auto merged = combine_welford_stats<ring_size, count_per_device, stick_stride_u16>(
                    gathered_u16 + m * 2, gathered_u16 + m * 2 + 1);
                auto p_out_means = reinterpret_cast<volatile uint16_t*>(out_ptr);
                auto p_out_vars = reinterpret_cast<volatile uint16_t*>(out_ptr + single_tile_size_bytes);
                p_out_means[0] = merged.mean;
                p_out_vars[0] = merged.variance;
                out_ptr += 2 * single_tile_size_bytes;
            }
            cb_pop_front(stats_gathered_cb_id, ring_size);
        }

        // Batched mcast-back: broadcast the whole cb_ex_global region (the GLOBAL stat for every
        // group) to the receiver cores in one shot, then release them with one semaphore multicast.
        if constexpr (num_mcast_cores > 1) {
            constexpr uint32_t mcast_bytes = 2 * num_groups * single_tile_size_bytes;
            MulticastEndpoint mcast_dst;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(global_base_ptr),
                mcast_dst,
                mcast_bytes,
                num_mcast_cores_mid_group,
                {},
                {.noc_x_start = mcast_dest_noc_start_x,
                 .noc_y_start = mcast_dest_noc_start_y,
                 .noc_x_end = mcast_dest_noc_end_x,
                 .noc_y_end = mcast_dest_noc_end_y,
                 .addr = global_base_ptr},
                true);
            reduce_sender_sem.set_multicast(
                noc,
                mcast_dest_noc_start_x,
                mcast_dest_noc_start_y,
                mcast_dest_noc_end_x,
                mcast_dest_noc_end_y,
                num_mcast_cores_mid_group,
                false);

            if (has_mcast_first_group) {
                MulticastEndpoint mcast_first_group_dst;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(global_base_ptr),
                    mcast_first_group_dst,
                    mcast_bytes,
                    num_mcast_cores_first_group,
                    {},
                    {.noc_x_start = mcast_first_group_dest_noc_start_x,
                     .noc_y_start = mcast_first_group_dest_noc_start_y,
                     .noc_x_end = mcast_first_group_dest_noc_end_x,
                     .noc_y_end = mcast_first_group_dest_noc_end_y,
                     .addr = global_base_ptr},
                    true);
                reduce_sender_sem.set_multicast(
                    noc,
                    mcast_first_group_dest_noc_start_x,
                    mcast_first_group_dest_noc_start_y,
                    mcast_first_group_dest_noc_end_x,
                    mcast_first_group_dest_noc_end_y,
                    num_mcast_cores_first_group,
                    false);
            }

            if (has_mcast_last_group) {
                MulticastEndpoint mcast_last_group_dst;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(global_base_ptr),
                    mcast_last_group_dst,
                    mcast_bytes,
                    num_mcast_cores_last_group,
                    {},
                    {.noc_x_start = mcast_last_group_dest_noc_start_x,
                     .noc_y_start = mcast_last_group_dest_noc_start_y,
                     .noc_x_end = mcast_last_group_dest_noc_end_x,
                     .noc_y_end = mcast_last_group_dest_noc_end_y,
                     .addr = global_base_ptr},
                    true);
                reduce_sender_sem.set_multicast(
                    noc,
                    mcast_last_group_dest_noc_start_x,
                    mcast_last_group_dest_noc_start_y,
                    mcast_last_group_dest_noc_end_x,
                    mcast_last_group_dest_noc_end_y,
                    num_mcast_cores_last_group,
                    false);
            }
            noc.async_write_barrier();
        }

        cb_ex_partial.pop_front(2);
        cb_ex_global.push_back(2 * num_groups);

        mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            } else {
                out_block_h_actual = out_block_h_normal;
            }
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                    if constexpr (welford_fp32_alias) {
                        // Mirror the cb_in0 push on the alias. They share SRAM (multi-buffer-index
                        // alias) so the noc.async_read above already filled both views; this is
                        // purely bookkeeping so compute's welford section can wait_front
                        // on cb_in0_welford independently of cb_in0.
                        cb_in0_welford.reserve_back(1);
                        cb_in0_welford.push_back(1);
                    }
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
