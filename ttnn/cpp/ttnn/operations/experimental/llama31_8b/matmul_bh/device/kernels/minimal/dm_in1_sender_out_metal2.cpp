// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "matmul_dataflow_common_metal2.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"
#ifdef IN1_WMC
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"
#endif
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

#if defined(DIR_SPLIT)
#define K_FORWARD_AT(n) ((n) < DIR_SPLIT)
#elif defined(DIR_PERIOD)
#define K_FORWARD_AT(n) (((((n) % DIR_PERIOD) / DIR_BLOCK) % 2) == 0)
#endif

#if defined(N_ROTATE) && defined(DIR_PERIOD)
#define N_NEXT_TILE n_next_tile
#define IN1_BULK_NBI n_block_rot
#else
#define N_NEXT_TILE (n_tile + N_block_tiles)
#define IN1_BULK_NBI n_block_iter
#endif
void kernel_main() {
    Noc noc;
    constexpr auto M_cores = get_arg(args::M_cores);
    constexpr auto K_tiles = get_arg(args::K_tiles);
    constexpr auto padded_K_tiles = get_arg(args::padded_K_tiles);
    constexpr auto N_tiles = get_arg(args::N_tiles);
    constexpr auto padded_N_tiles = get_arg(args::padded_N_tiles);
    constexpr auto M_block_tiles = get_arg(args::M_block_tiles);
    constexpr auto K_block_tiles = get_arg(args::K_block_tiles);
    constexpr auto N_block_tiles = get_arg(args::N_block_tiles);
    constexpr auto N_blocks_per_core = get_arg(args::N_blocks_per_core);
    constexpr auto in1_tile_size = get_arg(args::in1_tile_size);
    constexpr auto out_tile_size = get_arg(args::out_tile_size);
    constexpr auto in2_tile_size = get_arg(args::in2_tile_size);
    Semaphore in1_sender_semaphore(sem::in1_sender);
    Semaphore in1_receiver_semaphore(sem::in1_receiver);
    Semaphore in1_valid_semaphore(sem::in1_valid);
    constexpr auto is_injector_core = get_arg(args::is_injector_core);
    constexpr auto N_chunks = get_arg(args::N_chunks);
    constexpr auto N_tiles_per_chunk = get_arg(args::N_tiles_per_chunk);

    const auto is_sink_core = get_arg(args::is_sink_core);
    const auto in1_dest_noc_x = get_arg(args::in1_dest_noc_x);
    const auto in1_dest_noc_y = get_arg(args::in1_dest_noc_y);
    const auto in1_sender_noc_x = get_arg(args::in1_sender_noc_x);
    const auto in1_sender_noc_y = get_arg(args::in1_sender_noc_y);
    const auto in0_idx = get_arg(args::in0_idx);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);
    const auto defer_write_k_block = get_arg(args::defer_write_k_block);
    const auto max_defer_write_k_block = get_arg(args::max_defer_write_k_block);

#ifdef FUSE_TERNARY
    const auto broadcast_ternary_b = get_arg(args::broadcast_ternary_b);
#endif

    const uint32_t M_tiles = get_arg(args::M_tiles);
    const auto [M_tiles_per_core, M_blocks_per_core, padded_M_tiles, M_start_tile, M_end_tile] =
        ttnn::runtime_length::uniform_split(M_tiles, M_cores, in0_idx, M_block_tiles);
#if defined(N_ROTATE) && defined(DIR_PERIOD)
    uint32_t n_rot_start = 0;
    while (n_rot_start + 1 < N_blocks_per_core && !K_FORWARD_AT(N_start_tile + n_rot_start * N_block_tiles)) {
        n_rot_start++;
    }
#endif

    const auto in1_reader = TensorAccessor(tensor::in1);
#ifdef IN1_BULK_W
    const uint32_t bulk_segn = get_arg(args::bulk_segn);
    const uint32_t bulk_seg[24] = {
        get_arg(args::bulk_s0),  get_arg(args::bulk_s1),  get_arg(args::bulk_s2),  get_arg(args::bulk_s3),
        get_arg(args::bulk_s4),  get_arg(args::bulk_s5),  get_arg(args::bulk_s6),  get_arg(args::bulk_s7),
        get_arg(args::bulk_s8),  get_arg(args::bulk_s9),  get_arg(args::bulk_s10), get_arg(args::bulk_s11),
        get_arg(args::bulk_s12), get_arg(args::bulk_s13), get_arg(args::bulk_s14), get_arg(args::bulk_s15),
        get_arg(args::bulk_s16), get_arg(args::bulk_s17), get_arg(args::bulk_s18), get_arg(args::bulk_s19),
        get_arg(args::bulk_s20), get_arg(args::bulk_s21), get_arg(args::bulk_s22), get_arg(args::bulk_s23)};
    const bool bulk_ok = bulk_segn > 0 && in1_reader.get_aligned_page_size() == in1_tile_size;
#endif
#ifdef IN1_WMC
    constexpr auto in1_wmc = MCAST_ARGS(in1_wmc);
    auto in1_wsnd = in1_wmc.optional_sender(noc);
    auto in1_wrcv = in1_wmc.optional_receiver(noc);
#endif

    [[maybe_unused]] auto outputs_tuple = make_tensor_accessors(tensor::outputs);

#ifdef FUSE_BIAS
    const auto in2_reader = TensorAccessor(tensor::in2);
#endif

#ifdef FUSE_TERNARY
    const auto ternary_a_reader = TensorAccessor(tensor::ternary_a);
    const auto ternary_b_reader = TensorAccessor(tensor::ternary_b);
#endif

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
#ifdef MM_WINDOW_BLOCKS
    const TensorShape2D out_shape(MM_WINDOW_TOTAL_M_TILES, N_tiles, MM_WINDOW_TOTAL_M_TILES, padded_N_tiles);
#else
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
#endif
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    DataflowBuffer dfb_in1(dfb::in1);
#ifdef IS_OUTPUT_WRITER
    DataflowBuffer dfb_out(dfb::out);
#endif
#if defined(FUSE_BIAS) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_in2(dfb::in2);
#endif

#if defined(FUSE_TERNARY) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_ternary_a(dfb::ternary_a);
    DataflowBuffer dfb_ternary_b(dfb::ternary_b);
    constexpr uint32_t ternary_a_tile_size = get_tile_size(dfb::ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(dfb::ternary_b);
#endif

#ifdef FUSE_AG
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
            false,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule);
    }
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
#ifdef FUSE_AG
    srs_fuse_signaler_rt_args_idx += 12;
#endif
    OpSignaler srs_fuse_signaler;
    uint32_t mm_progress_counters_base = 0;
#ifdef MM_WINDOW_BLOCKS
    uint32_t M_window_start_tile = 0;
    uint32_t rs_credit_counters_base = 0;
    uint32_t num_rs_readers = 0;
#endif
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
        mm_progress_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
#ifdef MM_WINDOW_BLOCKS
        M_window_start_tile = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        rs_credit_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        num_rs_readers = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        if constexpr (is_output_writer) {
            volatile tt_l1_ptr uint32_t* credits =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_credit_counters_base);
            for (uint32_t r = 0; r < num_rs_readers; r++) {
                credits[r] = 0;
            }
        }
#endif
    }
#endif

    in1_valid_semaphore.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    [[maybe_unused]] uint32_t defer_write_m_tile = 0;
    [[maybe_unused]] uint32_t defer_write_m_tile_end = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile_end = 0;
    [[maybe_unused]] uint32_t defer_write_nbi = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
#ifdef MM_WINDOW_BLOCKS
        if constexpr (is_output_writer) {
            if (m_block_iter >= MM_WINDOW_BLOCKS) {
                const uint32_t blocks_released_needed = m_block_iter - MM_WINDOW_BLOCKS + 1;
                volatile tt_l1_ptr uint32_t* credits =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_credit_counters_base);
                for (uint32_t r = 0; r < num_rs_readers; r++) {
                    if (credits[r] < blocks_released_needed) {
                        noc_semaphore_wait_min(&credits[r], blocks_released_needed);
                    }
                }
            }
            const uint32_t rows_in_block = m_tile_end - m_tile;
            m_tile = M_window_start_tile + (m_block_iter % MM_WINDOW_BLOCKS) * M_block_tiles;
            m_tile_end = m_tile + rows_in_block;
        }
#endif
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        k_forward = true;
#ifdef DIR_PERIOD
#if defined(N_ROTATE)
        k_forward = K_FORWARD_AT(N_start_tile + n_rot_start * N_block_tiles);
#else
        k_forward = K_FORWARD_AT(N_start_tile);
#endif
#endif

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
#if defined(N_ROTATE) && defined(DIR_PERIOD)
            [[maybe_unused]] const uint32_t n_block_rot = (n_block_iter + n_rot_start) % N_blocks_per_core;
            uint32_t n_tile = N_start_tile + n_block_rot * N_block_tiles;
            [[maybe_unused]] const uint32_t n_next_tile =
                N_start_tile + ((n_block_iter + 1 + n_rot_start) % N_blocks_per_core) * N_block_tiles;
#else
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
#endif
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                    dfb_out.wait_front(out_block_num_tiles_swiglu);
                    uint32_t out_read_ptr = dfb_out.get_read_ptr();
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
                    dfb_out.pop_front(out_block_num_tiles_swiglu);
#else
                    dfb_out.wait_front(out_block_num_tiles);
                    uint32_t out_read_ptr = dfb_out.get_read_ptr();

                    if constexpr (N_chunks == 1) {
#ifdef DIR_SPLIT
                        for (uint32_t m_row = defer_write_m_tile;
                             m_row < defer_write_m_tile_end && m_row < out_shape.logical_d0;
                             m_row++) {
                            const uint32_t row_ptr =
                                out_read_ptr + (m_row - defer_write_m_tile) * N_block_tiles * out_tile_size;
                            for (uint32_t q = 0; q < bulk_segn; q++) {
                                const uint32_t seg = bulk_seg[q];
                                if ((seg >> 24) != defer_write_nbi) {
                                    continue;
                                }
                                const uint32_t col = (seg >> 8) & 0xFFu;
                                const uint32_t width = seg & 0xFFu;
                                for (uint32_t t = 0; t < width; t++) {
                                    noc.async_write(
                                        CoreLocalMem<uint32_t>(row_ptr + (col + t) * out_tile_size),
                                        std::get<0>(outputs_tuple),
                                        out_tile_size,
                                        {},
                                        {.page_id = m_row * out_shape.logical_d1 + ((seg >> 16) & 0xFFu) + t});
                                }
                            }
                        }
                        noc.async_writes_flushed();
#else
                        write_block_sync<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end);
#endif
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
                    dfb_out.pop_front(out_block_num_tiles);
#endif
#endif
                }

                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                dfb_in1.reserve_back(in1_block_num_tiles);

                uint32_t in1_start_address = dfb_in1.get_write_ptr();
                if constexpr (is_injector_core) {
#ifdef FUSE_AG
                    if (is_injector_core) {
                        k_block =
                            fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                    }
#endif
#ifdef IN1_BULK_W
                    if (bulk_ok) {
                        const uint32_t k0 = k_block * K_block_tiles;
                        const uint32_t rows = k0 >= in1_shape.logical_d0
                                                  ? 0u
                                                  : std::min<uint32_t>(K_block_tiles, in1_shape.logical_d0 - k0);
                        const uint32_t src_row = IN1_BULK_W * in1_tile_size;
                        const uint32_t dst_row = N_block_tiles * in1_tile_size;
                        for (uint32_t q = 0; q < bulk_segn; q++) {
                            const uint32_t seg = bulk_seg[q];
#ifdef DIR_SPLIT
                            if ((seg >> 24) != IN1_BULK_NBI) {
                                continue;
                            }
#else
                            if ((seg >> 16) != IN1_BULK_NBI) {
                                continue;
                            }
#endif
                            const uint32_t col = (seg >> 8) & 0xFFu;
                            const uint32_t width = seg & 0xFFu;
                            const uint32_t dst = in1_start_address + col * in1_tile_size;
                            if (rows > 0) {
#ifdef DIR_SPLIT
                                const uint64_t src =
                                    in1_reader.get_noc_addr(k0 * in1_shape.logical_d1 + ((seg >> 16) & 0xFFu));
#else
                                const uint64_t src = in1_reader.get_noc_addr(k0 * in1_shape.logical_d1 + n_tile + col);
#endif
#if IN1_BULK_MODE == 2
                                noc_async_read_one_packet_set_state(src, in1_tile_size);
                                for (uint32_t r = 0; r < rows; r++) {
                                    for (uint32_t t = 0; t < width; t++) {
                                        noc_async_read_one_packet_with_state(
                                            (uint32_t)(src + (uint64_t)r * src_row) + t * in1_tile_size,
                                            dst + r * dst_row + t * in1_tile_size);
                                    }
                                }
#else
                                noc_async_read_set_state(src);
                                if (width == N_block_tiles && width == IN1_BULK_W && rows == K_block_tiles) {
                                    noc_async_read_with_state(
                                        (uint32_t)src, dst, K_block_tiles * width * in1_tile_size);
                                } else {
                                    for (uint32_t r = 0; r < rows; r++) {
                                        noc_async_read_with_state(
                                            (uint32_t)(src + (uint64_t)r * src_row),
                                            dst + r * dst_row,
                                            width * in1_tile_size);
                                    }
                                }
#endif
                            }
                            for (uint32_t r = rows; r < K_block_tiles; r++) {
                                for (uint32_t t = 0; t < width; t++) {
                                    fill_zeros_async(
                                        noc, dfb_in1, in1_tile_size, r * dst_row + (col + t) * in1_tile_size);
                                }
                            }
                        }
                        noc.async_read_barrier();
                        noc.write_zeros_l1_barrier();
                    } else
#endif
                        read_in1_block_sync<K_block_tiles, N_block_tiles>(
                            in1_reader,
                            in1_shape,
                            dfb_in1,
                            in1_tile_size,
                            k_block * K_block_tiles,
                            (k_block + 1) * K_block_tiles,
                            n_tile,
                            n_tile_end);
                } else {
#ifdef IN1_WMC
                    in1_wrcv->receive();
#else
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    in1_receiver_semaphore.wait(VALID);
#endif
                }

                dfb_in1.push_back(in1_block_num_tiles);

#ifdef IN1_WMC
                if constexpr (is_injector_core) {
                    in1_wsnd->send(in1_start_address, in1_start_address, K_block_tiles * full_N_tiles_bytes);
                }
                if (false) {
#else
                if (!is_sink_core) {
#endif
                    in1_sender_semaphore.wait(1);
                    in1_sender_semaphore.set(0);

                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(in1_start_address),
                            UnicastEndpoint{},
                            current_N_tiles_bytes,
                            {},
                            {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
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
#ifndef IS_OUTPUT_WRITER
            dfb_in2.reserve_back(N_block_tiles);

            uint32_t l1_write_addr_in2 = dfb_in2.get_write_ptr();
            for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                noc.async_read(
                    in2_reader, CoreLocalMem<uint32_t>(l1_write_addr_in2), in2_tile_size, {.page_id = n_tile_id}, {});
                l1_write_addr_in2 += in2_tile_size;
            }
            noc.async_read_barrier();

            dfb_in2.push_back(N_block_tiles);
#endif
#endif

#ifdef FUSE_TERNARY
#ifndef IS_OUTPUT_WRITER
            read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                ternary_a_reader,
                ternary_b_reader,
                out_shape,
                dfb_ternary_a,
                dfb_ternary_b,
                ternary_a_tile_size,
                ternary_b_tile_size,
                broadcast_ternary_b,
                m_tile,
                m_tile_end,
                n_tile,
                n_tile_end);
#endif
#endif

#ifdef DIR_PERIOD
            k_forward = K_FORWARD_AT(N_NEXT_TILE);
#else
            k_forward = !k_forward;
#endif

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_nbi = IN1_BULK_NBI;
            defer_write_n_tile_end = n_tile_end;
            defer_write = !is_last_block;

            if (!defer_write) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                if constexpr (N_chunks == 1) {
                    write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                } else {
                    write_block_sync_granular_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                        outputs_tuple,
                        out0_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                }
#else
                if constexpr (N_chunks == 1) {
#ifdef DIR_SPLIT
                    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
                        dfb_out.wait_front(N_block_tiles);
                        const uint32_t m_row = m_tile + m_id;
                        if (m_row < m_tile_end && m_row < out_shape.logical_d0) {
                            const uint32_t row_ptr = dfb_out.get_read_ptr();
                            for (uint32_t q = 0; q < bulk_segn; q++) {
                                const uint32_t seg = bulk_seg[q];
                                if ((seg >> 24) != IN1_BULK_NBI) {
                                    continue;
                                }
                                const uint32_t col = (seg >> 8) & 0xFFu;
                                const uint32_t width = seg & 0xFFu;
                                for (uint32_t t = 0; t < width; t++) {
                                    noc.async_write(
                                        CoreLocalMem<uint32_t>(row_ptr + (col + t) * out_tile_size),
                                        std::get<0>(outputs_tuple),
                                        out_tile_size,
                                        {},
                                        {.page_id = m_row * out_shape.logical_d1 + ((seg >> 16) & 0xFFu) + t});
                                }
                            }
                        }
                        noc.async_writes_flushed();
                        dfb_out.pop_front(N_block_tiles);
                    }
#else
                    write_block_sync_granular<M_block_tiles, N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile,
                        n_tile_end);
#endif
                } else {
                    write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                        outputs_tuple, out0_shape, dfb_out, out_tile_size, m_tile, m_tile_end, n_tile, n_tile_end);
                }
#endif
#ifdef SRS_FUSE_OP_SIGNALER
                if (is_last_block) {
                    noc.async_write_barrier();
                    srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                }
#endif
#endif
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
