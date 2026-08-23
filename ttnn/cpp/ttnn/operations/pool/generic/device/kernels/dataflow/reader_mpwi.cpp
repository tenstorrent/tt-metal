// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp>

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "api/debug/dprint_pages.h"
#endif

template <bool B>
struct IndexType;

template <>
struct IndexType<true> {
    using type = uint32_t;
};

template <>
struct IndexType<false> {
    using type = uint16_t;
};

template <
    typename IndexType,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t fill_c,
    uint32_t column_stride,
    uint32_t row_stride,
    bool is_large_kernel,
    uint32_t sticks_per_chunk,
    uint32_t in_idx_cb_id>
void fill_indexes(uint32_t init_index) {
    DataflowBuffer idx_cb(in_idx_cb_id);
    volatile tt_l1_ptr IndexType* idx_ptr = reinterpret_cast<volatile tt_l1_ptr IndexType*>(idx_cb.get_write_ptr());
    uint32_t kernel_idx = 0;

    for (uint32_t h = 0; h < kernel_h; ++h) {
        uint32_t hw = h * kernel_w;
        for (uint32_t w = 0; w < kernel_w; ++w, ++hw) {
            if (!is_large_kernel || hw < sticks_per_chunk) {
                volatile tt_l1_ptr IndexType* base_ptr = &idx_ptr[hw * TILE_WIDTH];
                IndexType index = static_cast<IndexType>(init_index + kernel_idx);

                for (uint32_t c = 0; c < fill_c; ++c) {
                    base_ptr[c] = index;
                }
            }
            kernel_idx += column_stride;
        }
        kernel_idx += row_stride;
    }
}

// Initialize indices and increment tiles for return_indices functionality
template <
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w,
    uint32_t in_c,
    uint32_t pad_t,
    uint32_t pad_l,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool is_large_kernel,
    uint32_t sticks_per_chunk,
    uint32_t right_inc,
    uint32_t down_left_wrap_inc,
    uint32_t up_left_wrap_inc,
    uint32_t intra_kernel_right_inc,
    uint32_t intra_kernel_down_left_wrap_inc,
    uint32_t in_idx_cb_id,
    uint32_t right_inc_cb_id,
    uint32_t down_left_wrap_inc_cb_id,
    uint32_t up_left_wrap_inc_cb_id,
    uint32_t intra_kernel_right_inc_cb_id,
    uint32_t intra_kernel_down_left_wrap_inc_cb_id,
    bool indexes_32_bit>
ALWI void initialize_return_indices_data(uint32_t start_row, uint32_t start_col) {
    // since kernels can start in padded regions we need to have "indexes" in these regions
    // we choose a paradigm where we padding indexes must satisfy the following conditions:
    //   1. they are 1 less than the index to the right (whether it's padding or not)
    //   2. they are in_w less than the index below (whether it's padding or not)
    // this results in repeat indexes, negative indexes and other such effects, but since
    // padding is never chosen as a max index, validity of the padding index is unimportant
    // we only care that when they are incremented they result in the correct index which
    // this paradigm guarantees
    // Note we also use unsigned integers and allow wrapping for negatives to preserve range
    // and since all negative values correspond to padding indexes which will never be a max

    // Calculate initial index based on padding conditions
    uint32_t init_index = 0;
    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr uint32_t eff_kernel_w = (kernel_w - 1) * dilation_w + 1;
    if (start_row <= pad_t) {
        // top left is in top padding, we increment from the padding index in the top left
        // of the padded tensor
        uint32_t global_top_left_pad_idx = -pad_l - pad_t * in_w;
        init_index = global_top_left_pad_idx + start_col + start_row * in_w;
    } else if (start_col <= pad_l) {
        // top left is in left padding, we increment from the padding index in the leftmost
        // column of the starting row of the padded tensor
        uint32_t leftmost_valid_index = (start_row - pad_t) * in_w;
        uint32_t start_row_left_pad_idx = leftmost_valid_index - pad_l;
        init_index = start_row_left_pad_idx + start_col;
    } else {
        // top left is in valid region, we choose the valid index
        init_index = (start_row - pad_t) * in_w + (start_col - pad_l);
    }

    constexpr uint32_t fill_c = in_c <= TILE_WIDTH ? in_c : TILE_WIDTH;
    constexpr uint32_t fill_c_32_bit = fill_c % 2 == 0 ? fill_c / 2 : (fill_c / 2) + 1;
    constexpr uint32_t HALF_TILE_WIDTH = TILE_WIDTH / 2;
    constexpr uint32_t column_stride = dilation_w;
    constexpr uint32_t row_stride = dilation_h * in_w - eff_kernel_w - (dilation_w - 1);

    // initialize the index CB
    DataflowBuffer idx_cb(in_idx_cb_id);
    idx_cb.reserve_back(1);
    fill_indexes<
        typename IndexType<indexes_32_bit>::type,
        kernel_h,
        kernel_w,
        fill_c,
        column_stride,
        row_stride,
        is_large_kernel,
        sticks_per_chunk,
        in_idx_cb_id>(init_index);
    idx_cb.push_back(1);

    // initialize the increment CBs
    // TODO we used to fill the 16 bit values two at a time, but this technically resulted in overflow with odd
    // c dimensions so for now we do it one at a time for both 16 and 32 bit indexes
    if constexpr (indexes_32_bit) {
        auto fill_inc_32 = [&](DataflowBuffer& inc_cb, uint32_t inc) __attribute__((always_inline)) {
            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inc_cb.get_write_ptr());
            for (uint32_t k = 0; k < window_size_hw; ++k) {
                for (uint32_t c = 0; c < fill_c; ++c) {
                    ptr[k * TILE_WIDTH + c] = inc;
                }
            }
        };

        DataflowBuffer right_inc_cb(right_inc_cb_id);
        right_inc_cb.reserve_back(1);
        fill_inc_32(right_inc_cb, right_inc);
        right_inc_cb.push_back(1);

        DataflowBuffer down_left_wrap_inc_cb(down_left_wrap_inc_cb_id);
        down_left_wrap_inc_cb.reserve_back(1);
        fill_inc_32(down_left_wrap_inc_cb, down_left_wrap_inc);
        down_left_wrap_inc_cb.push_back(1);

        DataflowBuffer up_left_wrap_inc_cb(up_left_wrap_inc_cb_id);
        up_left_wrap_inc_cb.reserve_back(1);
        fill_inc_32(up_left_wrap_inc_cb, up_left_wrap_inc);
        up_left_wrap_inc_cb.push_back(1);

        if constexpr (is_large_kernel) {
            DataflowBuffer intra_kernel_right_inc_cb(intra_kernel_right_inc_cb_id);
            intra_kernel_right_inc_cb.reserve_back(1);
            fill_inc_32(intra_kernel_right_inc_cb, intra_kernel_right_inc);
            intra_kernel_right_inc_cb.push_back(1);

            DataflowBuffer intra_kernel_down_left_wrap_inc_cb(intra_kernel_down_left_wrap_inc_cb_id);
            intra_kernel_down_left_wrap_inc_cb.reserve_back(1);
            fill_inc_32(intra_kernel_down_left_wrap_inc_cb, intra_kernel_down_left_wrap_inc);
            intra_kernel_down_left_wrap_inc_cb.push_back(1);
        }
    } else {
        auto fill_inc = [&](DataflowBuffer& inc_cb, uint32_t inc) __attribute__((always_inline)) {
            uint16_t inc_16 = (uint16_t)inc;
            uint32_t inc_32_bit = (uint32_t)inc_16 | ((uint32_t)inc_16 << 16);
            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inc_cb.get_write_ptr());
            for (uint32_t k = 0; k < window_size_hw; ++k) {
                for (uint32_t c = 0; c < fill_c_32_bit; ++c) {
                    ptr[k * HALF_TILE_WIDTH + c] = inc_32_bit;
                }
            }
        };

        DataflowBuffer right_inc_cb(right_inc_cb_id);
        right_inc_cb.reserve_back(1);
        fill_inc(right_inc_cb, right_inc);
        right_inc_cb.push_back(1);

        DataflowBuffer down_left_wrap_inc_cb(down_left_wrap_inc_cb_id);
        down_left_wrap_inc_cb.reserve_back(1);
        fill_inc(down_left_wrap_inc_cb, down_left_wrap_inc);
        down_left_wrap_inc_cb.push_back(1);

        DataflowBuffer up_left_wrap_inc_cb(up_left_wrap_inc_cb_id);
        up_left_wrap_inc_cb.reserve_back(1);
        fill_inc(up_left_wrap_inc_cb, up_left_wrap_inc);
        up_left_wrap_inc_cb.push_back(1);

        if constexpr (is_large_kernel) {
            DataflowBuffer intra_kernel_right_inc_cb(intra_kernel_right_inc_cb_id);
            intra_kernel_right_inc_cb.reserve_back(1);
            fill_inc(intra_kernel_right_inc_cb, intra_kernel_right_inc);
            intra_kernel_right_inc_cb.push_back(1);

            DataflowBuffer intra_kernel_down_left_wrap_inc_cb(intra_kernel_down_left_wrap_inc_cb_id);
            intra_kernel_down_left_wrap_inc_cb.reserve_back(1);
            fill_inc(intra_kernel_down_left_wrap_inc_cb, intra_kernel_down_left_wrap_inc);
            intra_kernel_down_left_wrap_inc_cb.push_back(1);
        }
    }
}

// Read kernel data for MPWI (Max Pool With Indices)
// This handles reading input data and managing index tracking for max pooling with index output
template <
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_leftover,
    uint32_t in_c,
    uint32_t sticks_per_chunk,
    uint32_t total_elems_to_reduce,
    bool wide_reduction,
    uint32_t clear_value_cb_id,
    uint32_t in_cb_ntiles,
    uint32_t in_nbytes_c,
    uint32_t shard_width_bytes,
    bool is_large_kernel,
    bool last_tile_is_partial,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool zero_pages,
    uint32_t out_cb_id,
    uint32_t out_idx_cb_id,
    uint32_t reader_id,
    uint32_t pack_tmp_cb_id,
    uint32_t pack_idx_tmp_cb_id,
    uint32_t indexes_32_bit>
ALWI void read_kernel_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // MPWI requires 1 tile at a time for max reduction with indices
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 1;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = (in_c + TILE_WIDTH - 1) / TILE_WIDTH;
    static_assert(MAX_TILES_PER_REDUCTION == 1, "MAX_TILES_PER_REDUCTION must be 1 for MPWI");
    constexpr uint32_t max_write_inc = TILE_WIDTH * BYTES_PER_ELEM;

    // Each reader constructs only the CBs it drives. Accessor aliases keep the common template
    // arguments valid while reader_id constexpr-discarding prevents the inactive role from using them.
    DataflowBuffer in_cb(in_cb_id);
    DataflowBuffer out_cb(out_cb_id);
    DataflowBuffer out_idx_cb(out_idx_cb_id);
    DataflowBuffer pack_tmp_cb(pack_tmp_cb_id);
    DataflowBuffer pack_idx_tmp_cb(pack_idx_tmp_cb_id);
    Noc noc;
    UnicastEndpoint self_ep;

    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        uint32_t read_bytes = in_nbytes_c;
        if constexpr (wide_reduction) {
            read_bytes =
                (c_i == in_nblocks_c - 1) ? in_nbytes_c - c_i * MAX_BYTES_PER_REDUCTION : MAX_BYTES_PER_REDUCTION;
        }

        uint32_t write_offset = 0;
        // reader_id is a named constexpr argument, so the inactive reader role is discarded.
        if constexpr (reader_id == 0) {
            in_cb.reserve_back(1);
            // page zeroing is only necessary for tiled block output format so that scale is not affected by
            // junk/padding data
            if constexpr (zero_pages) {
                if (c_i == in_nblocks_c - 1 && last_tile_is_partial) {
                    zero_out_page(noc, in_cb);
                }
            }
        }
        for (uint32_t h = 0; h < kernel_h; ++h) {
            auto process_h = [&](uint32_t w, uint32_t w_multiple) __attribute__((always_inline)) {
                uint32_t w_offset = w * dilation_w;
                if constexpr (reader_id == 0) {
                    const uint32_t stick_offset = ind + w_offset + h * dilation_h * in_w_padded;
                    const uint32_t read_offset =
                        in_l1_read_base_addr + (stick_offset * shard_width_bytes + c_i * MAX_BYTES_PER_REDUCTION);
                    noc.async_read(
                        self_ep,
                        in_cb,
                        read_bytes * w_multiple,
                        experimental::local_addr(read_offset),
                        {.offset_bytes = write_offset});
                    write_offset += max_write_inc * w_multiple;
                }
                bool kernel_complete = h == kernel_h - 1 && w == kernel_w - 1;
                bool push_chunk =
                    kernel_complete || (is_large_kernel && ((w + 1) % sticks_per_chunk == 0 || w == (kernel_w - 1)));
                if (push_chunk) {
                    if constexpr (reader_id == 0) {
                        noc.async_read_barrier();
                        in_cb.push_back(1);
                        if (!kernel_complete) {
                            in_cb.reserve_back(1);
                            write_offset = 0;
                        }
                    } else {
                        if (kernel_complete) {  // write output once all chunks are done
                            // Mirror compute_pool_2d.cpp: pack 1 face for "single partial tile
                            // fits in one face" or "last tile has exactly FACE_WIDTH valid".
                            constexpr bool single_partial_fits_in_face = last_tile_is_partial && in_c <= FACE_WIDTH;
                            constexpr uint32_t num_faces_in_output_tile = single_partial_fits_in_face ? 1 : 2;
                            constexpr uint32_t num_faces_in_last_output_tile =
                                last_tile_is_partial && (in_c % TILE_WIDTH == FACE_WIDTH || single_partial_fits_in_face)
                                    ? 1
                                    : 2;
                            uint32_t output_faces =
                                c_i == in_nblocks_c - 1 ? num_faces_in_last_output_tile : num_faces_in_output_tile;

                            out_cb.reserve_back(output_faces);
                            out_idx_cb.reserve_back(output_faces);

                            pack_tmp_cb.wait_front(1);
                            noc.async_read(
                                self_ep,
                                out_cb,
                                output_faces * FACE_WIDTH * BYTES_PER_ELEM,
                                experimental::local_addr(pack_tmp_cb.get_read_ptr()),
                                {});

                            pack_idx_tmp_cb.wait_front(1);
                            constexpr uint32_t BYTES_PER_ELEM_IDX = indexes_32_bit ? 4 : 2;
                            noc.async_read(
                                self_ep,
                                out_idx_cb,
                                output_faces * FACE_WIDTH * BYTES_PER_ELEM_IDX,
                                experimental::local_addr(pack_idx_tmp_cb.get_read_ptr()),
                                {});
                            noc.async_read_barrier();
                            pack_tmp_cb.pop_front(1);
                            pack_idx_tmp_cb.pop_front(1);

                            out_cb.push_back(output_faces);
                            out_idx_cb.push_back(output_faces);
                        }
                    }
                }
            };

            // TODO - contiguous reads for some cases
            for (uint32_t w = 0; w < kernel_w; ++w) {
                process_h(w, 1);
            }
        }
    }
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
template <
    uint32_t reader_nindices,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t pad_w,
    uint32_t in_nbytes_leftover,
    uint32_t in_w,
    uint32_t in_c,
    uint32_t split_reader,
    uint32_t reader_id,
    uint32_t bf16_scalar,
    uint32_t bf16_init_value,
    uint32_t in_nblocks_c,
    uint32_t in_cb_sz,
    uint32_t max_sticks_for_reduction,
    uint32_t ceil_pad_w,
    uint32_t pool_type_is_avg,
    uint32_t one_scalar_per_core,
    uint32_t in_nbytes_c,
    uint32_t shard_width_bytes,
    uint32_t multi_buffering_factor,
    uint32_t stride_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t zero_pages,
    uint32_t config_in_dram,
    uint32_t config_page_size,
    uint32_t reader_page_size,
    uint32_t pad_t,
    uint32_t pad_l,
    uint32_t right_inc,
    uint32_t down_left_wrap_inc,
    uint32_t up_left_wrap_inc,
    uint32_t intra_kernel_right_inc,
    uint32_t intra_kernel_down_left_wrap_inc,
    uint32_t indexes_32_bit>
TT_KERNEL void reader_mpwi(uint32_t core_nhw_index, uint32_t start_row, uint32_t start_col) {
    // CB ids now come from Metal 2.0 DFB bindings. reader0 and reader1 run on the SAME nodes,
    // so a DFB endpoint cannot be bound on both readers — each reader binary must reference
    // only the DFBs it actually drives. The host supplies reader_id as a named constexpr argument;
    // accessor aliases resolve common entry-point arguments for the inactive reader role.
    //
    // reader0-driven (produced by reader0, consumed by COMPUTE): in_cb, in_scalar_cb, clear_value_cb,
    //   in_idx_cb, right_inc_cb, down_left_wrap_inc_cb, up_left_wrap_inc_cb, intra_* (large-kernel).
    // reader1-driven (writer face): out_cb, out_idx_cb, pack_tmp_cb, pack_idx_tmp_cb.
    // Shared (both readers reference): in_shard_cb (input-shard base read), reader_indices_cb
    //   (reader0 produces in DRAM path / reader1 wait_fronts). Inactive accessor names alias a
    //   bound DFB so common entry-point template arguments still resolve without extra storage.
    constexpr uint32_t in_cb_id = dfb::in_cb;
    constexpr uint32_t in_scalar_cb_id_0 = dfb::in_scalar_cb;
    constexpr uint32_t clear_value_cb_id = dfb::clear_value_cb;
    constexpr uint32_t in_shard_cb_id = dfb::in_shard_cb;
    constexpr uint32_t in_reader_indices_cb_id = dfb::reader_indices_cb;
    // MPWI is max-pool-only with one scalar per core, but the common reader schema still
    // carries the regular-pool specialization parameters as entry-point template arguments.
    // MPWI-specific args start here. CB-id usage is role-gated by reader_id:
    // the index/increment CBs are reader0-driven (consumed by COMPUTE); the output/pack CBs are
    // reader1-driven. Non-driven accessor names alias an existing binding.
    constexpr uint32_t in_idx_cb_id = dfb::in_idx_cb;
    constexpr uint32_t right_inc_cb_id = dfb::right_inc_cb;
    constexpr uint32_t down_left_wrap_inc_cb_id = dfb::down_left_wrap_inc_cb;
    constexpr uint32_t up_left_wrap_inc_cb_id = dfb::up_left_wrap_inc_cb;
    constexpr uint32_t intra_kernel_right_inc_cb_id = dfb::intra_kernel_right_inc_cb;
    constexpr uint32_t intra_kernel_down_left_wrap_inc_cb_id = dfb::intra_kernel_down_left_wrap_inc_cb;
    constexpr uint32_t out_cb_id = dfb::out_cb;
    constexpr uint32_t out_idx_cb_id = dfb::out_idx_cb;
    constexpr uint32_t pack_tmp_cb_id = dfb::pack_tmp_cb;
    constexpr uint32_t pack_idx_tmp_cb_id = dfb::pack_idx_tmp_cb;
    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t sticks_per_chunk = kernel_w <= max_sticks_for_reduction ? kernel_w : max_sticks_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t in_cb_ntiles = in_cb_sz / (TILE_WIDTH * TILE_HEIGHT);  // only use the non-multi buffering size

    // Fill the clear CB only on reader0; reader_id is compile-time constant.
    if constexpr (reader_id == 0) {
        DataflowBuffer clear_value_cb(clear_value_cb_id);
        fill_with_val(clear_value_cb.get_write_ptr(), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);
        clear_value_cb.push_back(1);
        clear_out_tiles<in_cb_id, clear_value_cb_id>(Noc(), DataflowBuffer(in_cb_id), clear_value_cb);
        initialize_return_indices_data<
            kernel_h,
            kernel_w,
            in_w,
            in_c,
            pad_t,
            pad_l,
            dilation_h,
            dilation_w,
            is_large_kernel,
            sticks_per_chunk,
            right_inc,
            down_left_wrap_inc,
            up_left_wrap_inc,
            intra_kernel_right_inc,
            intra_kernel_down_left_wrap_inc,
            in_idx_cb_id,
            right_inc_cb_id,
            down_left_wrap_inc_cb_id,
            up_left_wrap_inc_cb_id,
            intra_kernel_right_inc_cb_id,
            intra_kernel_down_left_wrap_inc_cb_id,
            indexes_32_bit>(start_row, start_col);

        // initialize the scalar CB (reader0-only: in_scalar_cb is reader0-driven in MPWI).
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        DataflowBuffer in_scalar_cb(in_scalar_cb_id_0);
        fill_with_val(in_scalar_cb.get_write_ptr(), FACE_WIDTH, bf16_scalar >> 16);
        in_scalar_cb.push_back(1);
    }
    DataflowBuffer in_shard_cb(in_shard_cb_id);
    const uint32_t in_l1_read_base_addr = in_shard_cb.get_read_ptr();
    DataflowBuffer in_reader_indices_cb(in_reader_indices_cb_id);
    if constexpr (config_in_dram) {
        // reader_indices_cb is a shared DFB (reader0 produces in the DRAM path / reader1 consumes).
        // Only reader0 executes the DRAM read. Both reader specializations bind tensor::reader_indices
        // so the name resolves while parsing this constexpr-discarded branch.
        if constexpr (reader_id == 0) {
            // Inlined load_config_tensor_if_in_dram: the reader-indices tensor flows in via its
            // Metal 2.0 TensorBinding (tensor::reader_indices) instead of a CTA-baked DRAM address.
            Noc cfg_noc;
            const auto reader_indices_accessor = TensorAccessor(tensor::reader_indices);
            cfg_noc.async_read(
                reader_indices_accessor, in_reader_indices_cb, reader_page_size, {.page_id = core_nhw_index}, {});
            cfg_noc.async_read_barrier();
            in_reader_indices_cb.push_back(1);
        } else {
            (void)core_nhw_index;
            in_reader_indices_cb.wait_front(1);
        }
    }
    uint32_t reader_indices_l1_addr = in_reader_indices_cb.get_read_ptr();
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);

    uint32_t segments_counter = 1;
    constexpr uint32_t total_elems_to_reduce = kernel_h * kernel_w;

    uint16_t num_segments = reader_indices_ptr[0] & 0xffff;

    while (num_segments--) {
        uint32_t start_end_segment = reader_indices_ptr[segments_counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;

        constexpr uint32_t stride_multiple = 1;
        for (uint16_t ind = start; ind <= end; ind += stride_multiple * stride_w) {
            read_kernel_with_top_left_index<
                in_nblocks_c,
                in_cb_id,
                kernel_h,
                kernel_w,
                in_w_padded,
                in_nbytes_leftover,
                in_c,
                sticks_per_chunk,
                total_elems_to_reduce,
                wide_reduction,
                clear_value_cb_id,
                in_cb_ntiles,
                in_nbytes_c,
                shard_width_bytes,
                is_large_kernel,
                last_tile_is_partial,
                dilation_h,
                dilation_w,
                zero_pages,
                out_cb_id,
                out_idx_cb_id,
                reader_id,
                pack_tmp_cb_id,
                pack_idx_tmp_cb_id,
                indexes_32_bit>(ind, in_l1_read_base_addr);
        }
    }
}  // TT_KERNEL
