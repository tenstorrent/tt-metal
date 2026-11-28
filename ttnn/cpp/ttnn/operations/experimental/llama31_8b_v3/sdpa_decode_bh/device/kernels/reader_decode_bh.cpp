// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include <vector>

#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/rt_args_common_dec.hpp"
#include "dataflow_common_dec.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr bool is_q_sharded = get_compile_time_arg_val(7);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(8);
    constexpr bool is_paged_attention = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(10);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(11);
    constexpr uint32_t Bkv = get_compile_time_arg_val(12);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(14);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(16);
    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(18) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(20);
    constexpr bool tilize_q = get_compile_time_arg_val(21) == 1;
    constexpr bool reuse_k = get_compile_time_arg_val(22) == 1;
    constexpr bool use_half_tile = get_compile_time_arg_val(23);
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(24);
    constexpr bool is_cur_pos_tensor_sharded = get_compile_time_arg_val(25);
    constexpr bool is_page_table_sharded = get_compile_time_arg_val(26);
    constexpr uint32_t q_page_size_bytes = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(29);
    constexpr bool has_block_padding = is_paged_attention && original_block_size > 0 && original_block_size < 32;
    constexpr uint32_t k_mcast_semaphore_id = get_compile_time_arg_val(30);
    constexpr bool q_locally_available = get_compile_time_arg_val(31) == 1;
    constexpr bool use_k_mcast = get_compile_time_arg_val(32) == 1;
    constexpr uint32_t Bmask = get_compile_time_arg_val(33);

    constexpr uint32_t capacity_t = get_compile_time_arg_val(34);

    constexpr uint32_t spec_multi_pos_T = get_compile_time_arg_val(35);
    constexpr bool spec_multi_pos = spec_multi_pos_T > 0;

    constexpr auto q_args = TensorAccessorArgs<36>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;

    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);
    const bool do_k_mcast = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_dests = get_arg_val<uint32_t>(arg_idx++);

    if (q_addr == 0) {
        return;
    }

    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;

#ifdef SDB_R4
    bool sdk_hdo_pending = false;
#define SDB_HDO_FLUSH()                                      \
    do {                                                     \
        if (sdk_hdo_pending) {                               \
            noc.async_read_barrier();                        \
            CircularBuffer(cb_compute_cur_pos).push_back(1); \
            sdk_hdo_pending = false;                         \
        }                                                    \
    } while (0)
#else
#define SDB_HDO_FLUSH() \
    do {                \
    } while (0)
#endif
#ifdef SDB_M
    uint32_t sdk_vr_pos = UINT32_MAX;
#endif
    if constexpr (is_causal) {
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            CircularBuffer cb_writer(cb_writer_cur_pos);
            cb_writer.reserve_back(1);
            uint32_t index_cb_wr_ptr = cb_writer.get_write_ptr();
            if constexpr (!is_cur_pos_tensor_sharded) {
                const auto addrg = TensorAccessor(pos_args, pos_addr);

                noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
                noc.async_read_barrier();
            }
            CircularBuffer cb_compute(cb_compute_cur_pos);
            cb_compute.reserve_back(1);
            uint32_t index_cb_compute_wr_ptr = cb_compute.get_write_ptr();
            const uint8_t noc_id = noc.get_noc_id();
            const uint32_t my_noc_x = my_x[noc_id];
            const uint32_t my_noc_y = my_y[noc_id];
            UnicastEndpoint pos_src;
            noc.async_read(
                pos_src,
                CoreLocalMem<uint32_t>(index_cb_compute_wr_ptr),
                index_stick_size_B,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = index_cb_wr_ptr},
                {});
#ifdef SDB_R4
            cb_writer.push_back(1);
            sdk_hdo_pending = true;
#else
            noc.async_read_barrier();
            cb_writer.push_back(1);
            cb_compute.push_back(1);
#endif
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);

            if constexpr (spec_multi_pos) {
                cur_pos = index_ptr[cur_batch * spec_multi_pos_T + (spec_multi_pos_T - 1)];
#if ASSERT_ENABLED

                const uint32_t spec_pos_base = cur_batch * spec_multi_pos_T;
                for (uint32_t j = 0; j + 1 < spec_multi_pos_T; ++j) {
                    ASSERT(index_ptr[spec_pos_base + j] <= index_ptr[spec_pos_base + j + 1]);
                }
#endif
            } else {
                cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
            }
#ifdef SDB_M
            sdk_vr_pos = cur_pos;
            for (uint32_t b = 1; b < SDB_M; ++b) {
                const uint32_t v = index_ptr[b];
                if (v != UINT32_MAX && v > sdk_vr_pos) {
                    sdk_vr_pos = v;
                }
            }
#endif
        }
        if (cur_pos == UINT32_MAX) {
            SDB_HDO_FLUSH();
            return;
        }
    }

    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }
#ifdef SDB_M
    if (sdk_vr_pos == UINT32_MAX) {
        sdk_vr_pos = cur_pos;
    } else if constexpr (has_block_padding && is_causal) {
        sdk_vr_pos = (sdk_vr_pos / original_block_size) * 32 + (sdk_vr_pos % original_block_size);
    }
    const uint32_t sdk_vr_bound = sdk_vr_pos;
#else
#define sdk_vr_bound cur_pos
#endif

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        SDB_HDO_FLUSH();
        return;
    }

    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;
    uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t_dynamic;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr uint32_t attention_sink_tile_bytes = get_tile_size(cb_attention_sink);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    const uint32_t q_batch_offset = cur_batch * q_chunk_tiles;

#ifdef SDB_M
    {
        static_assert(is_q_sharded && !tilize_q && !use_half_tile && PNHt == 1 && q_tile_bytes == 2048);
        constexpr uint32_t m_qxy[] = {SDB_MQ};
        static_assert(sizeof(m_qxy) / sizeof(m_qxy[0]) == 2 * SDB_M);
        constexpr uint32_t m_face = q_tile_bytes / 4;
        constexpr uint32_t m_line = m_face / 16;
        const uint32_t s_row = cur_head_group * SDB_MG;
        const uint32_t s_off = (s_row < 16 ? 0 : 2 * m_face) + (s_row % 16) * m_line;
        CircularBuffer m_cb(cb_q_in);
        m_cb.reserve_back(q_chunk_tiles);
        const uint32_t m_dst = m_cb.get_write_ptr();
        for (uint32_t t = 0; t < q_chunk_tiles; ++t) {
            noc.async_write_zeros(m_cb, q_tile_bytes, {.offset_bytes = t * q_tile_bytes});
        }
        noc.write_zeros_l1_barrier();
        UnicastEndpoint m_src;
        for (uint32_t b = 0; b < SDB_M; ++b) {
            const uint32_t d_row = b * SDB_MG;
            const uint32_t d_off = (d_row < 16 ? 0 : 2 * m_face) + (d_row % 16) * m_line;
            for (uint32_t t = 0; t < q_chunk_tiles; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(
                        m_src,
                        CoreLocalMem<uint32_t>(m_dst + t * q_tile_bytes + d_off + f * m_face),
                        SDB_MG * m_line,
                        {.noc_x = m_qxy[2 * b],
                         .noc_y = m_qxy[2 * b + 1],
                         .addr = q_addr + t * q_tile_bytes + s_off + f * m_face},
                        {});
                }
            }
        }
        noc.async_read_barrier();
        m_cb.push_back(q_chunk_tiles);
    }
#else
#ifdef SDB_R2
    if constexpr (
        is_q_sharded && !tilize_q && !use_half_tile && !q_locally_available && PNHt == 1 && num_heads_per_core == 1 &&
        q_tile_bytes == 2048) {
        constexpr uint32_t qr_face_bytes = q_tile_bytes / 4;
        constexpr uint32_t qr_row_bytes = qr_face_bytes / 16;
        const uint32_t qr_row0 = cur_head_group * SDB_R2;
        const uint32_t qr_off = (qr_row0 < 16 ? 0 : 2 * qr_face_bytes) + (qr_row0 % 16) * qr_row_bytes;
        const uint8_t qr_noc = noc.get_noc_id();
        const uint32_t qr_x = is_output_core ? my_x[qr_noc] : output_core_noc_x;
        const uint32_t qr_y = is_output_core ? my_y[qr_noc] : output_core_noc_y;
        CircularBuffer qr_cb(cb_q_in);
        qr_cb.reserve_back(q_chunk_tiles);
        const uint32_t qr_dst = qr_cb.get_write_ptr();
        for (uint32_t t = 0; t < q_chunk_tiles; ++t) {
            noc.async_write_zeros(qr_cb, q_tile_bytes, {.offset_bytes = t * q_tile_bytes});
        }
        noc.write_zeros_l1_barrier();
        UnicastEndpoint qr_src;
        for (uint32_t t = 0; t < q_chunk_tiles; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t o = t * q_tile_bytes + qr_off + f * qr_face_bytes;
                noc.async_read(
                    qr_src,
                    CoreLocalMem<uint32_t>(qr_dst + o),
                    SDB_R2 * qr_row_bytes,
                    {.noc_x = qr_x, .noc_y = qr_y, .addr = q_addr + o},
                    {});
            }
        }
#ifndef SDB_R3
        noc.async_read_barrier();
        qr_cb.push_back(q_chunk_tiles);
#endif
    } else
#endif
        read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles, is_q_sharded, tilize_q, use_half_tile, barrier_threshold>(
            q_locally_available,
            is_output_core,
            q_addr,
            output_core_noc_x,
            output_core_noc_y,
            q_chunk_size_bytes,
            q_args,
            q_page_size_bytes,
            q_batch_offset);
#endif

    const auto k_reader = TensorAccessor(k_args, k_addr);

    const auto v_reader = TensorAccessor(v_args, v_addr);

    const auto mask_reader = TensorAccessor(mask_args, mask_addr);

    if constexpr (use_attention_sink) {
        const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);

        CircularBuffer cb_sink(cb_attention_sink);
        cb_sink.reserve_back(PNHt);
        uint32_t attention_sink_write_ptr = cb_sink.get_write_ptr();

        for (uint32_t tile = 0; tile < PNHt; ++tile) {
            noc.async_read(
                attention_sink_reader,
                CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                attention_sink_tile_bytes,
                {.page_id = tile},
                {});
            attention_sink_write_ptr += attention_sink_tile_bytes;
        }
        noc.async_read_barrier();
        cb_sink.push_back(PNHt);
    }

    volatile tt_l1_ptr uint32_t* page_table_ptr;
    uint32_t page_table_cb_wr_ptr = 0;
    volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
    volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
    if constexpr (is_paged_attention) {
        CircularBuffer cb_page_table(cb_id_page_table);
        uint32_t num_pages_to_read = is_page_table_sharded ? B : 1;
        cb_page_table.reserve_back(num_pages_to_read);

        if constexpr (!is_page_table_sharded) {
#ifdef SDB_R1
            if constexpr (capacity_t == 0) {
                const uint32_t pts_b0 = (k_chunk_start * Sk_chunk_t_dynamic) / block_size_t;
                const uint32_t pts_b1 = (k_chunk_end * Sk_chunk_t_dynamic - 1) / block_size_t + 1;
                const uint32_t pts_a0 = (pts_b0 * sizeof(uint32_t)) & ~63u;
                uint32_t pts_a1 = (pts_b1 * sizeof(uint32_t) + 63u) & ~63u;
                if (pts_a1 > page_table_page_size) {
                    pts_a1 = page_table_page_size;
                }
                const uint32_t pts_dst = cb_page_table.get_write_ptr();
                const auto pts_reader = TensorAccessor(page_table_args, page_table_addr, page_table_page_size);
                noc.async_read(
                    pts_reader,
                    CoreLocalMem<uint32_t>(pts_dst),
                    pts_a1 - pts_a0,
                    {.page_id = cur_batch / q_heads_parallel_factor, .offset_bytes = pts_a0},
                    {});
                noc.async_read_barrier();
                page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pts_dst) - pts_a0 / sizeof(uint32_t);
            } else
#endif
                page_table_ptr = read_page_table_for_batch(
                    noc,
                    cb_id_page_table,
                    cur_batch / q_heads_parallel_factor,
                    page_table_args,
                    page_table_addr,
                    page_table_page_size);
            page_table_ptr_u32 = page_table_ptr;
        } else {
            page_table_cb_wr_ptr =
                cb_page_table.get_write_ptr() + (cur_batch / q_heads_parallel_factor) * page_table_page_size;
            page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_cb_wr_ptr);
        }
        cb_page_table.push_back(num_pages_to_read);
    }
#if defined(SDB_R2) && defined(SDB_R3)
    if constexpr (
        is_q_sharded && !tilize_q && !use_half_tile && !q_locally_available && PNHt == 1 && num_heads_per_core == 1 &&
        q_tile_bytes == 2048) {
        noc.async_read_barrier();
        CircularBuffer(cb_q_in).push_back(q_chunk_tiles);
    }
#endif
    SDB_HDO_FLUSH();

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        const uint32_t mask_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bmask) * PNHt * St;
        const uint32_t mask_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic;
        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;

        KMcastParams k_mcast_params = {
            .do_mcast = do_k_mcast,
            .mcast_x = mcast_x,
            .mcast_y0 = mcast_y0,
            .mcast_y1 = mcast_y1,
            .num_dests = num_dests,
            .mcast_sem_id = k_mcast_semaphore_id};

#if 0 || defined(SDB_R0)
        if constexpr (is_paged_attention && !use_k_mcast && !reuse_k && !use_attention_mask) {
            constexpr bool one_bar = false;
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                const uint32_t row0 = k_chunk * Sk_chunk_t_dynamic;
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                cb_k.reserve_back(k_chunk_tiles);
                const uint32_t k_wr = cb_k.get_write_ptr();
                bool zeroed = false;
                uint32_t outstanding = 0;
                for (uint32_t row = 0; row < Sk_chunk_t_dynamic; ++row) {
#ifdef SDB_R0
                    const bool valid = (row0 + row) * 32 <= sdk_vr_bound;
#else
                    const bool valid = true;
#endif
                    if (!valid) {
                        for (uint32_t col = 0; col < DHt; ++col) {
                            noc.async_write_zeros(cb_k, k_tile_bytes, {.offset_bytes = (row + col * Sk_chunk_t_dynamic) * k_tile_bytes});
                        }
                        zeroed = true;
                        continue;
                    }
                    uint32_t k_col = k_wr + row * k_tile_bytes;
                    uint32_t k_id =
                        (is_page_table_sharded)
                            ? virtual_seq_tile_id_to_physical_tile_id<uint16_t, num_kv_heads, block_size_t, DHt, capacity_t>(
                                  row0 + row, cur_head, page_table_ptr_u16)
                            : virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_kv_heads, block_size_t, DHt, capacity_t>(
                                  row0 + row, cur_head, page_table_ptr_u32);
                    for (uint32_t col = 0; col < DHt; ++col) {
                        noc.async_read(k_reader, CoreLocalMem<uint32_t>(k_col), k_tile_bytes, {.page_id = k_id}, {});
                        k_id += 1;
                        k_col += Sk_chunk_t_dynamic * k_tile_bytes;
                        if (++outstanding == barrier_threshold) {
                            noc.async_read_barrier();
                            outstanding = 0;
                        }
                    }
                }
                if constexpr (!one_bar) {
                    noc.async_read_barrier();
                    if (zeroed) {
                        noc.write_zeros_l1_barrier();
                    }
                    cb_k.push_back(k_chunk_tiles);
                    outstanding = 0;
                    zeroed = false;
                }
                cb_v.reserve_back(v_chunk_tiles);
                uint32_t v_wr = cb_v.get_write_ptr();
                for (uint32_t row = 0; row < Sk_chunk_t_dynamic; ++row) {
#ifdef SDB_R0
                    const bool valid = (row0 + row) * 32 <= sdk_vr_bound;
#else
                    const bool valid = true;
#endif
                    if (!valid) {
                        for (uint32_t col = 0; col < vDHt; ++col) {
                            noc.async_write_zeros(cb_v, v_tile_bytes, {.offset_bytes = (row * vDHt + col) * v_tile_bytes});
                        }
                        v_wr += vDHt * v_tile_bytes;
                        zeroed = true;
                        continue;
                    }
                    uint32_t v_id =
                        (is_page_table_sharded)
                            ? virtual_seq_tile_id_to_physical_tile_id<uint16_t, num_kv_heads, block_size_t, vDHt, capacity_t>(
                                  row0 + row, cur_head, page_table_ptr_u16)
                            : virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_kv_heads, block_size_t, vDHt, capacity_t>(
                                  row0 + row, cur_head, page_table_ptr_u32);
                    for (uint32_t col = 0; col < vDHt; ++col) {
                        noc.async_read(v_reader, CoreLocalMem<uint32_t>(v_wr), v_tile_bytes, {.page_id = v_id}, {});
                        v_id += 1;
                        v_wr += v_tile_bytes;
                        if (++outstanding == barrier_threshold) {
                            noc.async_read_barrier();
                            outstanding = 0;
                        }
                    }
                }
                noc.async_read_barrier();
                if (zeroed) {
                    noc.write_zeros_l1_barrier();
                }
                if constexpr (one_bar) {
                    cb_k.push_back(k_chunk_tiles);
                }
                cb_v.push_back(v_chunk_tiles);
            }
            continue;
        }
#endif
        if constexpr (is_paged_attention) {
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t_dynamic;
                uint64_t k_base_read_ptr;

                k_base_read_ptr = read_k<
                    cb_k_in,
                    DHt,
                    num_kv_heads,
                    block_size_t,
                    k_tile_bytes,
                    barrier_threshold,
                    is_page_table_sharded,
                    use_k_mcast,
                    capacity_t>(
                    k_chunk_tiles,
                    cur_head,
                    Sk_chunk_t_dynamic,
                    k_chunk_start_row_num,
                    k_reader,
                    page_table_ptr_u16,
                    page_table_ptr_u32,
                    barrier_count,
                    k_mcast_params);

                if constexpr (use_attention_mask) {
                    mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                        PSt, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);
                }

                read_v<
                    cb_v_in,
                    vDHt,
                    num_kv_heads,
                    block_size_t,
                    v_tile_bytes,
                    barrier_threshold,
                    is_page_table_sharded,
                    reuse_k,
                    capacity_t>(
                    v_chunk_tiles,
                    cur_head,
                    Sk_chunk_t_dynamic,
                    k_chunk_start_row_num,
                    v_reader,
                    page_table_ptr_u16,
                    page_table_ptr_u32,
                    barrier_count,
                    k_base_read_ptr,
                    k_tile_bytes);
            }
        } else {
            const uint32_t k_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * DHt;
            const uint32_t k_head_offset = cur_head * St * DHt;

            const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * DHt;
            uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;

            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * vDHt;
            const uint32_t v_head_offset = cur_head * St * vDHt;
            const uint32_t v_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * vDHt;
            uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset;

            read_kv_mask_chunks<
                DHt,
                vDHt,
                barrier_threshold,
                mask_tile_bytes,
                PNHt,
                use_attention_mask,
                cb_k_in,
                cb_v_in,
                cb_mask_in,
                reuse_k>(
                k_chunk_start,
                k_chunk_end,
                k_start_tile_id,
                v_start_tile_id,
                mask_start_tile_id,
                Sk_chunk_t_dynamic,
                k_chunk_tiles,
                v_chunk_tiles,
                mask_chunk_tiles,
                k_reader,
                v_reader,
                mask_reader,
                k_tile_bytes,
                v_tile_bytes,
                PSt);
        }
    }
}
