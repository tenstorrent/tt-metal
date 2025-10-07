// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/array_view.hpp"
#include "accessor/tensor_accessor.h"

#if defined(WATCHER_OVERHEAD_OK)
template <bool is_output_cb, bool is_wr_ptr>
void dprint_cb_tile(uint32_t cb_id, uint32_t tile_id) {
    noc_async_read_barrier();
    noc_async_write_barrier();
    for (uint8_t i = 0; i < 32; ++i) {
        DPRINT << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{.h0 = i, .h1 = (uint8_t)(i + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                      is_output_cb ? TSLICE_OUTPUT_CB : TSLICE_INPUT_CB,
                      is_wr_ptr ? TSLICE_WR_PTR : TSLICE_RD_PTR,
                      true,
                      true)
               << ENDL();
    }
}
#endif

template <uint32_t tile_bytes>
void clear_mantissa_range(
    ArrayView<uint32_t, CBAccessType::CB_BACK_RW>& uint32_arr,
    uint32_t face_offset,
    uint32_t row_start_idx,
    uint32_t row_end_idx,
    uint32_t col_start_idx,
    uint32_t col_end_idx,
    uint32_t uint32_datums_per_face_row,
    uint32_t bf4_datums_per_uint32) {
    /**
     * [INFO] row_start_idx, row_end_idx, col_start_idx, col_end_idx are the coordinates within the face;
     * their values are relative to the top left corner -- (0,0) element -- of the face
     */

    uint32_t start_col_group_idx = col_start_idx / bf4_datums_per_uint32;
    uint32_t start_pos_in_group = col_start_idx % bf4_datums_per_uint32;
    uint32_t end_col_group_idx = col_end_idx / bf4_datums_per_uint32;
    uint32_t end_pos_in_group = col_end_idx % bf4_datums_per_uint32;
    uint32_t nbits_per_datum = 32 / bf4_datums_per_uint32;

    for (uint32_t row = row_start_idx; row < row_end_idx; row++) {
        uint32_t row_offset = row * uint32_datums_per_face_row;

        // little endian mask to clear the bits after start_pos_in_group to zero
        uint32_t start_mask = ~(0xFFFFFFFF << (start_pos_in_group * nbits_per_datum));

        // little endian mask to clear the bits before end_pos_in_group to zero
        uint32_t end_mask = 0xFFFFFFFF << (end_pos_in_group * nbits_per_datum);

        if (start_col_group_idx == end_col_group_idx) {
            // handle corner case where the range is a single uint32
            uint32_arr[face_offset + row_offset + start_col_group_idx] &= (start_mask | end_mask);
            continue;
        }

        // Clear bits after start position in the first uint32
        uint32_arr[face_offset + row_offset + start_col_group_idx] &= start_mask;
        // Set entire uint32s to 0 in the middle
        for (uint32_t col_group_idx = start_col_group_idx + 1; col_group_idx < end_col_group_idx; ++col_group_idx) {
            uint32_arr[face_offset + row_offset + col_group_idx] = 0;
        }
        // Clear bits before end position in the last uint32
        // [INFO] end_pos_in_group == 0 means that the next uint32 is not part of the range to be cleared
        if (end_pos_in_group > 0) {
            uint32_arr[face_offset + row_offset + end_col_group_idx] &= end_mask;
        }
    }
}

template <uint32_t tile_bytes>
inline void fill_diag_subtile_zeros_bfp4(
    uint32_t cb_id,
    uint32_t tile_id,
    uint32_t row_start_idx,
    uint32_t row_end_idx,
    uint32_t col_start_idx,
    uint32_t col_end_idx) {
    /**
     * bfp4_b tile is laid out in memory as:
     * [face0 exp][face1 exp][face2 exp][face3 exp][face0 mant][face1 mant][face2 mant][face3 mant]
     * where each face's exp is 16 bytes and each face's mant is 16x16x.5B = 128B, stored in row-major order.
     *
     * [INFO] face 0 and face 3 are diagonal faces
     * [INFO] this function requires that FACE_HEIGHT == FACE_WIDTH
     * [INFO] row_start_idx, row_end_idx, col_start_idx, col_end_idx are the coordinates of the subtile within the tile;
     * their values are relative to the top left corner -- (0,0) element -- of the tile
     */

    constexpr uint32_t bf4_mant_per_uint32 = 8;  // 8 mantissas per uint32
    constexpr uint32_t bf4_exp_per_uint32 = 4;   // 4 exponents per uint32

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    // [INFO] Only setting mantissa to 0 is needed to set the whole value to 0
    // Calculate face offsets in uint32 words
    constexpr uint32_t face_offsets[4] = {
        uint32_exp_per_face * 4,
        uint32_exp_per_face * 4 + uint32_datums_per_face,
        uint32_exp_per_face * 4 + uint32_datums_per_face * 2,
        uint32_exp_per_face * 4 + uint32_datums_per_face * 3};

    auto uint32_arr = ArrayView<uint32_t, CBAccessType::CB_BACK_RW>(cb_id, tile_id);

    // fill face 0 with both cases where the subtile is contained completely in the face or partially in the face
    if (row_start_idx < tt::constants::FACE_HEIGHT && col_start_idx < tt::constants::FACE_WIDTH) {
        clear_mantissa_range<tile_bytes>(
            uint32_arr,
            face_offsets[0],
            row_start_idx,
            std::min(row_end_idx, tt::constants::FACE_HEIGHT),
            col_start_idx,
            std::min(col_end_idx, tt::constants::FACE_WIDTH),
            uint32_datums_per_face_row,
            bf4_mant_per_uint32);
    }

    // fill face 1
    if (row_start_idx < tt::constants::FACE_HEIGHT && col_end_idx > tt::constants::FACE_WIDTH) {
        clear_mantissa_range<tile_bytes>(
            uint32_arr,
            face_offsets[1],
            row_start_idx,
            std::min(row_end_idx, tt::constants::FACE_HEIGHT),
            (col_start_idx > tt::constants::FACE_WIDTH ? col_start_idx - tt::constants::FACE_WIDTH : 0),
            col_end_idx - tt::constants::FACE_WIDTH,
            uint32_datums_per_face_row,
            bf4_mant_per_uint32);
    }

    // fill face 2
    if (row_end_idx > tt::constants::FACE_HEIGHT && col_start_idx < tt::constants::FACE_WIDTH) {
        clear_mantissa_range<tile_bytes>(
            uint32_arr,
            face_offsets[2],
            (row_start_idx > tt::constants::FACE_HEIGHT ? row_start_idx - tt::constants::FACE_HEIGHT : 0),
            row_end_idx - tt::constants::FACE_HEIGHT,
            col_start_idx,
            std::min(col_end_idx, tt::constants::FACE_WIDTH),
            uint32_datums_per_face_row,
            bf4_mant_per_uint32);
    }

    // fill face 3 with both cases where the subtile is contained completely in the face or partially in the face
    if (row_end_idx > tt::constants::FACE_HEIGHT && col_end_idx > tt::constants::FACE_WIDTH) {
        clear_mantissa_range<tile_bytes>(
            uint32_arr,
            face_offsets[3],
            row_start_idx > tt::constants::FACE_HEIGHT ? row_start_idx - tt::constants::FACE_HEIGHT : 0,
            row_end_idx - tt::constants::FACE_HEIGHT,
            col_start_idx > tt::constants::FACE_WIDTH ? col_start_idx - tt::constants::FACE_WIDTH : 0,
            col_end_idx - tt::constants::FACE_WIDTH,
            uint32_datums_per_face_row,
            bf4_mant_per_uint32);
    }
}

template <uint32_t tile_bytes>
void async_fill_tile_zeros(uint32_t cb_id, uint32_t tile_id) {
    static_assert(tile_bytes % 4 == 0, "tile_bytes must be a multiple of 4");
    uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
    fill_zeros_async(write_addr, tile_bytes);
}

template <uint32_t tile_bytes, typename TensorAccessorType>
uint32_t async_read_chunk_with_padding(
    const TensorAccessorType& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold,
    const bool transpose = false) {
    /*
    Method always reads tiles from memory in row-major order.
    It assumes that the block of rows x cols in stored in contiguous tile order.
    That means, it won't work if the chunk to read is a slice of the last dimension.

    This handles the case where the dst CB is larger than the src CB, with some padding on the
    rows or cols of the DST CB.
    */
    // Read Q chunk
    const uint32_t num_tiles = dst_rows * dst_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);
    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            uint64_t noc_addr = reader.get_noc_addr(start_tile_id);
            noc_async_read(noc_addr, write_ptr, tile_bytes);
            start_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
    }

    // Zero out the padding
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_tile_zeros<tile_bytes>(cb_id, tile_id);
        }
    }

    return num_tiles;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(6);
    constexpr uint32_t DHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t cu_window_seqlens_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t cu_window_seqlens_eles = get_arg_val<uint32_t>(argidx++);
    const uint32_t i = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    // Set up tensor accessor args for each buffer
    // Base index for compile-time args after the fixed parameters (0-10)
    constexpr uint32_t base_cta_idx = 11;
    constexpr auto q_args = TensorAccessorArgs<base_cta_idx>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto cu_window_seqlens_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_cu_window_seqlens_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_4;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);
    constexpr uint32_t cu_window_seqlens_tile_bytes = get_tile_size(cb_cu_window_seqlens_in);
    constexpr DataFormat cu_window_seqlens_data_format = get_dataformat(cb_cu_window_seqlens_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr DataFormat mask_data_format = get_dataformat(cb_mask_in);

    constexpr uint32_t q_heads_per_kv = NQH / NKH;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto cu_window_seqlens_reader =
        TensorAccessor(cu_window_seqlens_args, cu_window_seqlens_addr, cu_window_seqlens_tile_bytes);

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);
    const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);

    // load the entire cu_window_seqlens tensor into a circular buffer
    cb_reserve_back(cb_cu_window_seqlens_in, 1);
    uint64_t cu_window_noc_addr = cu_window_seqlens_reader.get_noc_addr(0);
    noc_async_read(cu_window_noc_addr, get_write_ptr(cb_cu_window_seqlens_in), cu_window_seqlens_tile_bytes);
    auto cb_cu_window_seqlens_ptr = ArrayView<uint32_t, CBAccessType::CB_BACK_RO>(cb_cu_window_seqlens_in);
    auto get_cu_window_seqlens = [&](uint32_t idx) -> uint32_t {
        if constexpr (cu_window_seqlens_data_format == DataFormat::UInt32) {
            return cb_cu_window_seqlens_ptr[idx];
        } else if constexpr (cu_window_seqlens_data_format == DataFormat::Int32) {
            return (uint32_t)cb_cu_window_seqlens_ptr[idx];
        } else {
            ASSERT(false);
        }
    };
    auto get_window_indices = [&](uint32_t local_mask_idx) {
        if (local_mask_idx < cu_window_seqlens_eles) {
            auto low = get_cu_window_seqlens(local_mask_idx);
            auto high = local_mask_idx == cu_window_seqlens_eles - 1 ? low : get_cu_window_seqlens(local_mask_idx + 1);
            return std::make_pair(low, high);
        }
        auto low = get_cu_window_seqlens(cu_window_seqlens_eles - 1);
        return std::make_pair(low, low);
    };

    // find the windows covered by the current q_chunk and k_chunk
    uint32_t q_low_idx_in_tokens = local_q_start * Sq_chunk_t * tt::constants::TILE_HEIGHT;
    uint32_t q_high_idx_in_tokens =
        std::min((local_q_start + q_chunks_per_core) * Sq_chunk_t, valid_Sqt) * tt::constants::TILE_HEIGHT;
    uint32_t mask_windows_low_idx = 0;
    bool found_mask_windows = false;
    noc_async_read_barrier();  // Wait until reads are done
    cb_push_back(cb_cu_window_seqlens_in, 1);
    DPRINT_ARRAY_VIEW({
        DPRINT << "cu_window_seqlens_eles: " << cu_window_seqlens_eles << ENDL();
        DPRINT << "cu_window_seqlens: " << ENDL();
        cb_cu_window_seqlens_ptr.print();
    });
    // [INFO] all windows are diagonal
    for (uint32_t w = 0; w < cu_window_seqlens_eles - 1; ++w) {
        auto window_start = get_cu_window_seqlens(w);
        auto window_end = get_cu_window_seqlens(w + 1);
        if ((q_low_idx_in_tokens >= window_start && q_low_idx_in_tokens < window_end) ||
            (q_high_idx_in_tokens > window_start && q_high_idx_in_tokens <= window_end)) {
            mask_windows_low_idx = w;
            found_mask_windows = true;
            break;
        }
    }

    uint32_t q_high_idx_in_tiles = Skt;
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        const uint32_t mask_batch_offset = nb * Sqt * Skt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // [INFO] each user or head tracks its own mask_windows_low_idx
            auto local_mask_windows_low_idx = mask_windows_low_idx;
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk = local_q_start + q_iter;

                const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
                const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
                const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
                const uint32_t q_tile_id = q_tile_shape.id_of(nb, nq, q_row_start_tile, 0);

                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);

                uint32_t q_low_idx_in_tiles = q_chunk * Sq_chunk_t;

                const uint32_t kv_head = nq / q_heads_per_kv;

                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx_in_tiles; ++k_chunk) {
                    const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);
                    const uint32_t k_row_end_tile = std::min(k_row_start_tile + Sk_chunk_t, valid_Skt);
                    const uint32_t k_row_tile_count = k_row_end_tile - k_row_start_tile;
                    const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, kv_head, k_row_start_tile, 0);

                    // Read K chunk
                    auto k_chunk_ntiles = async_read_chunk_with_padding<k_tile_bytes>(
                        k_reader,
                        cb_k_in,
                        k_start_tile_id,
                        k_row_tile_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        64,   // optimized for Sq_chunk_t = 8 and Sk_chunk_t = 8
                        true  // transpose=true for K reads
                    );

                    // [INFO] Generate windowed attention mask on-the-fly for q_row_tile_count x k_row_tile_count tiles
                    // [INFO] q_chunk_size and k_chunk_size can differ
                    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                    uint32_t mask_write_ptr_base = get_write_ptr(cb_mask_in);
                    uint64_t noc_write_addr_base = get_noc_addr(mask_write_ptr_base);

                    int zero_tile_idx = -1;
                    int inf_tile_idx = -1;
                    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                        uint32_t q_start_idx = (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
                        uint32_t q_end_idx = q_start_idx + tt::constants::TILE_HEIGHT;

                        // [INFO]:
                        // A window is defined by two pairs of indices (window_low_idx, window_low_idx) and
                        // (window_high_idx, window_high_idx) -- the top left and bottom right corners of the window.
                        // A tile is defined by two pairs of indices (q_start_idx, k_start_idx) and (q_end_idx,
                        // k_end_idx) -- the top left and bottom right corners of the tile.
                        // A covered window by a tile is defined by two pairs of indices (max(q_start_idx,
                        // window_low_idx), max(k_start_idx, window_low_idx)) and (min(q_end_idx, window_high_idx),
                        // min(k_end_idx, window_high_idx)) -- the top left and bottom right corners of the covered
                        // window. A whole window can be covered partially and it is possible to have multiple small
                        // windows that are completely covered by the same tile.
                        auto result = get_window_indices(local_mask_windows_low_idx);
                        uint32_t window_low_idx = result.first;
                        uint32_t window_high_idx = result.second;

                        // loop invariant: local_mask_windows_low_idx is covered by the some of the tiles in the current
                        // row unless the current row is empty, i.e., window_low_idx == window_high_idx
                        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                            uint32_t k_start_idx = (k_row_start_tile + col) * tt::constants::TILE_HEIGHT;
                            uint32_t k_end_idx = k_start_idx + tt::constants::TILE_HEIGHT;

                            uint32_t in_mask_tile_id = row * Sk_chunk_t + col;

                            // case: the tile covers a single window --> fill the tile with zeros
                            // [INFO] if the window is empty, the following condition will be false
                            if (q_start_idx >= window_low_idx && q_end_idx <= window_high_idx &&
                                k_start_idx >= window_low_idx && k_end_idx <= window_high_idx) {
                                if (zero_tile_idx == -1) {
                                    async_fill_tile_zeros<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                                } else {
                                    copy_tile<mask_tile_bytes>(
                                        noc_write_addr_base, mask_write_ptr_base, zero_tile_idx, in_mask_tile_id);
                                }
                                // save most recent zero'ed tile as the source of copy_tile in the future
                                zero_tile_idx = in_mask_tile_id;
                                continue;
                            }

                            // cases: the tile does not cover any window or the window is empty or the tile covers at
                            // least a window partially --> fill the tile with inf
                            // no windows to fill anymore; fill all tiles with inf
                            if (inf_tile_idx == -1) {
                                fill_neginf_tile_bfp4<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                            } else {
                                copy_tile<mask_tile_bytes>(
                                    noc_write_addr_base, mask_write_ptr_base, inf_tile_idx, in_mask_tile_id);
                            }
                            if (!found_mask_windows || k_end_idx <= window_low_idx || k_start_idx >= window_high_idx ||
                                window_low_idx >= window_high_idx) {
                                // case: the tile does not cover any window or the window is empty
                                // save most recent inf'ed tile as the source of copy_tile in the future
                                inf_tile_idx = in_mask_tile_id;
                                continue;
                            }

                            // cases: the tile covers at least a window (potentailly multiple windows)
                            uint32_t covered_window_q_start_idx, covered_window_k_start_idx, covered_window_q_end_idx,
                                covered_window_k_end_idx;
                            do {
                                covered_window_q_start_idx = std::max(q_start_idx, window_low_idx);
                                covered_window_k_start_idx = std::max(k_start_idx, window_low_idx);
                                covered_window_q_end_idx = std::min(q_end_idx, window_high_idx);
                                covered_window_k_end_idx = std::min(k_end_idx, window_high_idx);

                                if (covered_window_q_start_idx < covered_window_q_end_idx &&
                                    covered_window_k_start_idx < covered_window_k_end_idx) {
                                    // only work on the covered window when it is not empty
                                    fill_diag_subtile_zeros_bfp4<mask_tile_bytes>(
                                        cb_mask_in,
                                        in_mask_tile_id,
                                        covered_window_q_start_idx - q_start_idx,
                                        covered_window_q_end_idx - q_start_idx,
                                        covered_window_k_start_idx - k_start_idx,
                                        covered_window_k_end_idx - k_start_idx);
                                }

                                if (covered_window_q_end_idx >= window_high_idx &&
                                    covered_window_k_end_idx >= window_high_idx) {
                                    // get the next window when the covering of the current window is complete
                                    local_mask_windows_low_idx += 1;
                                    auto result = get_window_indices(local_mask_windows_low_idx);
                                    window_low_idx = result.first;
                                    window_high_idx = result.second;
                                }
                            } while (window_low_idx < window_high_idx && covered_window_q_end_idx < q_end_idx &&
                                     covered_window_k_end_idx < k_end_idx);

                            DPRINT_ARRAY_VIEW({
                                DPRINT << "  [COL ITER] WINDOW tile: in_mask_tile_id: " << in_mask_tile_id << ENDL();
                                (dprint_cb_tile<true, true>(cb_mask_in, in_mask_tile_id));
                            });
                        }
                    }
                    // sync up the read and writes, push back the mask cb, after processing each chunk
                    noc_async_read_barrier();  // syncs up the reads of k_chunk and the ones used by mask generation
                    cb_push_back(cb_k_in, k_chunk_ntiles);
                    cb_push_back(cb_mask_in, mask_chunk_tiles);

                    // Read V chunk
                    read_chunk_with_padding<v_tile_bytes>(
                        v_reader,
                        cb_v_in,
                        k_start_tile_id,
                        k_row_tile_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        barrier_threshold,
                        false  // transpose=false for V reads
                    );
                }
            }
        }
    }
}
