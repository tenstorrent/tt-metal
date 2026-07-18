// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "noc_parameters.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id =
        get_named_compile_time_arg_val("reduce_receiver_semaphore_id");
    constexpr uint32_t reduce_sender_semaphore_id = get_named_compile_time_arg_val("reduce_sender_semaphore_id");

    constexpr uint32_t num_mcast_cores = get_named_compile_time_arg_val("num_cores_per_mcast_group");
    constexpr uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr uint32_t num_batches = get_named_compile_time_arg_val("num_batches");
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("TILE_HEIGHT");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");

    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr uint32_t num_rows_per_group = get_named_compile_time_arg_val("num_rows_per_group");

    constexpr auto src0_args = TensorAccessorArgs<0>();

#ifdef GN_DISTRIBUTED_AG
    // Fabric all-gather (cross-device Chan merge over cluster_axis). ring_size == 1 never sets
    // GN_DISTRIBUTED_AG, so the fused op's local path compiles this file's stock text verbatim.
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t stick_bytes = get_named_compile_time_arg_val("stick_bytes");
    constexpr uint32_t num_chunks_per_device = get_named_compile_time_arg_val("num_chunks_per_device");
    constexpr uint32_t fwd_arrival_sem_id = get_named_compile_time_arg_val("fwd_arrival_sem_id");
    constexpr uint32_t go_sem_id = get_named_compile_time_arg_val("go_sem_id");
    constexpr uint32_t packet_cb_id = get_named_compile_time_arg_val("packet_cb_id");
    constexpr uint32_t stats_local_cb_id = get_named_compile_time_arg_val("stats_local_cb_id");
    constexpr uint32_t stats_gathered_cb_id = get_named_compile_time_arg_val("stats_gathered_cb_id");
    // Per-device element count per group, the per-subgroup COUNT_PER_VALUE for the cross-device
    // combine. num_rows_per_group is PER CORE, so scale by the mcast-group core count.
    constexpr uint32_t count_per_device = num_channels_per_group * num_rows_per_group * num_mcast_cores;
    // The gathered stats sticks are bf16 [mean, var] pairs; an fp32 stats CB would need a different
    // stick layout and stride. Fail the build rather than silently misread the gather.
    static_assert(
        get_named_compile_time_arg_val("stats_is_fp32") == 0,
        "GN_DISTRIBUTED_AG requires bf16 stats CBs (stats_is_fp32 == 0)");
    // Appended after src0's positional accessor block.
    constexpr auto stats_dram_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#endif

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
    uint32_t noc_arg_base;

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

        noc_arg_base = 22;

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        noc_arg_base = 17;

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_arg_base = 17;

    } else {
        noc_arg_base = 12;
    }

    noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(noc_arg_base));
    noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(noc_arg_base + num_mcast_cores));

    Noc noc;
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    reduce_sender_sem.set(VALID);

    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
    // Welford-fp32 alias for dfb_in0. Shares SRAM with dfb_in0 but has its own buffer index
    // configured with UnpackToDestFp32, plus its own read/write pointers.
    // The Welford section of compute reads the alias to get full fp32 into DEST, while later
    // FPU consumers read dfb_in0 directly. When welford_fp32_alias is false, cb_in0_welford_id
    // == cb_in0_id and the gated pushes below are skipped.
    constexpr uint32_t dfb_in0_welford_id = get_named_compile_time_arg_val("cb_in0_welford");
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    // When set, stats CBs hold fp32; the Welford combine reads/writes them as float not bf16, and the cross-core stride
    // is in fp32 elements.
    constexpr bool stats_is_fp32 = get_named_compile_time_arg_val("stats_is_fp32") != 0;
    constexpr bool sfpu_two_pass = get_named_compile_time_arg_val("sfpu_two_pass") != 0;
    constexpr bool sfpu_two_pass_l1_replay = get_named_compile_time_arg_val("sfpu_two_pass_l1_replay") != 0;

    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_in0_welford(dfb_in0_welford_id);
    DataflowBuffer dfb_repack(dfb_repack_id);
    DataflowBuffer dfb_repack_out(dfb_repack_out_id);
    DataflowBuffer dfb_out0(dfb_out0_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(dfb_ex_partial_id);
    constexpr uint32_t src0_tile_bytes = get_tile_size(dfb_in0_id);

    constexpr uint32_t local_stride = 2;
    // Cross-core stats packed at NOC alignment; element stride = byte gap / element size.
    constexpr uint32_t stats_elem_size = stats_is_fp32 ? 4 : 2;
    constexpr uint32_t global_stride = NOC_L1_READ_ALIGNMENT_BYTES / stats_elem_size;

    // Combine overload picked by pointer type: const float* -> fp32 combine, volatile uint16_t* -> bf16.
    using stats_read_t = std::conditional_t<stats_is_fp32, const float, volatile uint16_t>;
    using stats_write_t = std::conditional_t<stats_is_fp32, float, uint16_t>;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

    const auto src_a = TensorAccessor(src0_args, src_addr);

#ifdef GN_DISTRIBUTED_AG
    // ---- Fabric all-gather setup ----
    const uint32_t fabric_base = noc_arg_base + 2u * num_mcast_cores;
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(fabric_base + 0);
    const uint32_t fwd_x = get_arg_val<uint32_t>(fabric_base + 1);
    const uint32_t fwd_y = get_arg_val<uint32_t>(fabric_base + 2);
    const uint32_t my_slot = get_arg_val<uint32_t>(fabric_base + 3);
    const uint32_t my_forwarder_index = get_arg_val<uint32_t>(fabric_base + 4);
    DataflowBuffer dfb_packet(packet_cb_id);
    DataflowBuffer dfb_stats_local(stats_local_cb_id);
    DataflowBuffer dfb_stats_gathered(stats_gathered_cb_id);
    Semaphore<> fwd_arrival_sem(fwd_arrival_sem_id);
    Semaphore<> go_sem(go_sem_id);
    // The packet CB is created on the whole grid, so the forwarder's copy is at this same address.
    const uint32_t fwd_packet_buf_addr = dfb_packet.get_write_ptr();
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
#endif

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = dfb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = dfb_repack.get_write_ptr();
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
        dfb_repack.push_back(per_core_N);
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
        constexpr uint32_t num_stats_passes = sfpu_two_pass && !sfpu_two_pass_l1_replay ? 2 : 1;
        for (uint32_t stats_pass = 0; stats_pass < num_stats_passes; ++stats_pass) {
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
                        dfb_in0.reserve_back(1);
                        const uint32_t l1_write_addr = dfb_in0.get_write_ptr();
                        noc.async_read(
                            src_a,
                            CoreLocalMem<uint32_t>(l1_write_addr),
                            src0_tile_bytes,
                            {.page_id = start_id + index_b_offset + mt_offset + nt},
                            {});
                        noc.async_read_barrier();
                        dfb_in0.push_back(1);
                        if constexpr (welford_fp32_alias) {
                            dfb_in0_welford.reserve_back(1);
                            dfb_in0_welford.push_back(1);
                        }
                    }
                    mt_offset += num_channels_tiles;
                }
#endif
            }
        }

        dfb_ex_partial.wait_front(2);
        auto local_means_ptr = dfb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        dfb_ex_global.reserve_back(2 * num_groups);
        const auto global_base_ptr = dfb_ex_global.get_write_ptr();
        auto global_means_ptr = global_base_ptr;
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

#ifdef GN_DISTRIBUTED_AG
        // Per-device stats stick (bf16 [mean, var] per group), staged for the fabric AG.
        dfb_stats_local.reserve_back(1);
        volatile tt_l1_ptr uint16_t* stick_u16 =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb_stats_local.get_write_ptr());

        // Batched intra-device handshake: wait ONCE for every receiver to have published ALL of its
        // groups' partials (each receiver signals once, after its per-group loop). The mcast-back is
        // deferred to a single batched broadcast after the fabric exchange, so the per-group
        // signal/wait lock-step below (which would deadlock a single exchange) is batched here and
        // in the receiver. Sync granularity only: the arithmetic and the mcast'd bytes are identical.
        if constexpr (num_mcast_cores > 1) {
            reduce_receiver_sem.wait(num_mcast_cores - 1);
            reduce_receiver_sem.set(0);
        }
#endif

        for (uint32_t m = 0; m < num_groups; ++m) {
            // Read mean and variance arrays from dfb_ex_partial, then combine using Welford
            auto p_local_means = reinterpret_cast<stats_read_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<stats_read_t*>(local_vars_ptr);

            auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);

            // Write this to dfb_ex_global
            auto p_global_means = reinterpret_cast<volatile stats_write_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile stats_write_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            if constexpr (num_mcast_cores > 1) {
#ifndef GN_DISTRIBUTED_AG
                // Wait until all other cores have signaled that their partial data is ready.
                // (GN_DISTRIBUTED_AG hoists this to a single batched wait before the loop.)
                reduce_receiver_sem.wait(num_mcast_cores - 1);
                reduce_receiver_sem.set(0);
#endif

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

            // Read dfb_ex_global through read-typed views; the writes below reuse the write-typed pointers (same L1
            // addresses).
            auto p_global_means_read = reinterpret_cast<stats_read_t*>(global_means_ptr);
            auto p_global_vars_read = reinterpret_cast<stats_read_t*>(global_vars_ptr);
            auto global_result =
                combine_welford_stats<num_mcast_cores, num_channels_per_group * num_rows_per_group, global_stride>(
                    p_global_means_read, p_global_vars_read);

            // Write this to dfb_ex_global
            p_global_means[0] = global_result.mean;
            p_global_vars[0] = global_result.variance;

#ifdef GN_DISTRIBUTED_AG
            // Stage the DEVICE-GLOBAL stat (post intra-device combine) for the fabric AG — not the
            // per-core local, so the cross-device Chan merge sees full per-device statistics.
            stick_u16[m * 2 + 0] = global_result.mean;
            stick_u16[m * 2 + 1] = global_result.variance;
#endif

#ifndef GN_DISTRIBUTED_AG
            if constexpr (num_mcast_cores > 1) {
                // mcast to other cores
                MulticastEndpoint mcast_dst;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(global_means_ptr),
                    mcast_dst,
                    2 * single_tile_size_bytes,
                    num_mcast_cores_mid_group,
                    {},
                    {.noc_x_start = mcast_dest_noc_start_x,
                     .noc_y_start = mcast_dest_noc_start_y,
                     .noc_x_end = mcast_dest_noc_end_x,
                     .noc_y_end = mcast_dest_noc_end_y,
                     .addr = global_means_ptr},
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
                        CoreLocalMem<uint32_t>(global_means_ptr),
                        mcast_first_group_dst,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_first_group,
                        {},
                        {.noc_x_start = mcast_first_group_dest_noc_start_x,
                         .noc_y_start = mcast_first_group_dest_noc_start_y,
                         .noc_x_end = mcast_first_group_dest_noc_end_x,
                         .noc_y_end = mcast_first_group_dest_noc_end_y,
                         .addr = global_means_ptr},
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
                        CoreLocalMem<uint32_t>(global_means_ptr),
                        mcast_last_group_dst,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_last_group,
                        {},
                        {.noc_x_start = mcast_last_group_dest_noc_start_x,
                         .noc_y_start = mcast_last_group_dest_noc_start_y,
                         .noc_x_end = mcast_last_group_dest_noc_end_x,
                         .noc_y_end = mcast_last_group_dest_noc_end_y,
                         .addr = global_means_ptr},
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
#endif

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

#ifdef GN_DISTRIBUTED_AG
        // ---- Cross-device all-gather + Welford (Chan) merge over cluster_axis ----
        // dfb_ex_global holds this device's per-group DEVICE-GLOBAL (mean, var). Publish this
        // master's sub-stick, ring-gather every device's, and Chan-merge into the GLOBAL stat.

        // 1. Publish this master's sub-stick to the forwarder packet buffer + signal arrival.
        dfb_stats_local.push_back(1);
        dfb_stats_local.wait_front(1);
        UnicastEndpoint fwd_ep;
        noc.async_write(
            CoreLocalMem<uint32_t>(dfb_stats_local.get_read_ptr()),
            fwd_ep,
            stick_bytes,
            {},
            {.noc_x = fwd_x, .noc_y = fwd_y, .addr = fwd_packet_buf_addr + my_slot * stick_bytes});
        noc.async_write_barrier();
        fwd_arrival_sem.up(noc, fwd_x, fwd_y, 1);
        dfb_stats_local.pop_front(1);

        // 2. Wait for the ring gather to land in DRAM (forwarder increments our go-sem).
        go_sem.wait_min(1);
        go_sem.set(0);

        // 3. Read the ring_size sub-sticks (our slot) from DRAM into the gathered CB.
        dfb_stats_gathered.reserve_back(ring_size);
        const uint32_t gbase = dfb_stats_gathered.get_write_ptr();
        for (uint32_t d = 0; d < ring_size; d++) {
            const uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index;
            noc.async_read(
                stats_dram,
                CoreLocalMem<uint32_t>(gbase + d * stick_bytes),
                stick_bytes,
                {.page_id = page_idx, .offset_bytes = my_slot * stick_bytes},
                {});
        }
        noc.async_read_barrier();
        dfb_stats_gathered.push_back(ring_size);

        // 4. Chan-merge the ring_size per-device stats per group; overwrite dfb_ex_global.
        dfb_stats_gathered.wait_front(ring_size);
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
        dfb_stats_gathered.pop_front(ring_size);

        // Batched mcast-back: broadcast the whole dfb_ex_global region (the GLOBAL stat for every
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
#endif

        dfb_ex_partial.pop_front(2);
        dfb_ex_global.push_back(2 * num_groups);

        if constexpr (!sfpu_two_pass_l1_replay) {
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
                        dfb_in0.reserve_back(1);
                        const uint32_t l1_write_addr = dfb_in0.get_write_ptr();
                        noc.async_read(
                            src_a,
                            CoreLocalMem<uint32_t>(l1_write_addr),
                            src0_tile_bytes,
                            {.page_id = start_id + index_b_offset + mt_offset + nt},
                            {});
                        noc.async_read_barrier();
                        dfb_in0.push_back(1);
                        if constexpr (welford_fp32_alias) {
                            // Mirror the dfb_in0 push on the alias. They share SRAM (multi-buffer-index
                            // alias) so the noc.async_read above already filled both views; this is
                            // purely bookkeeping so compute's welford section can wait_front
                            // on dfb_in0_welford independently of dfb_in0.
                            dfb_in0_welford.reserve_back(1);
                            dfb_in0_welford.push_back(1);
                        }
                    }
                    mt_offset += num_channels_tiles;
                }
#endif
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = dfb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = dfb_repack_out.get_read_ptr();
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
        dfb_repack_out.pop_front(per_core_N);
    }
#endif
}
