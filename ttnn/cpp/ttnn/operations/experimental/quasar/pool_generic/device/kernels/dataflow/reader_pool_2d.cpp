// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/pool_kernels_common.hpp>

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

// Read kernel data for normal max/average pooling (without indices)
template <
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_leftover,
    uint32_t in_c,
    uint32_t max_sticks_for_reduction,
    uint32_t total_elems_to_reduce,
    bool is_avg_pool,
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
    uint32_t in_cb_sz,
    uint32_t bf16_init_value>
ALWI void read_kernel_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = (in_c + TILE_WIDTH - 1) / TILE_WIDTH;
    constexpr uint32_t num_tilized_rows =
        wide_reduction ? (in_cb_sz / (MAX_TILES_PER_REDUCTION * TILE_WIDTH)) : (in_cb_sz / (in_ntiles_c * TILE_WIDTH));
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     (kernel_h * kernel_w) <= 16 && !last_tile_is_partial;

    DataflowBuffer in_cb(in_cb_id);
    DataflowBuffer clear_cb(clear_value_cb_id);
    Noc noc;
    UnicastEndpoint self_ep;

    uint32_t max_write_inc = wide_reduction ? MAX_BYTES_PER_REDUCTION : in_nbytes_leftover;
    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        uint32_t read_bytes = in_nbytes_c;
        if constexpr (wide_reduction) {
            read_bytes =
                (c_i == in_nblocks_c - 1) ? in_nbytes_c - c_i * MAX_BYTES_PER_REDUCTION : MAX_BYTES_PER_REDUCTION;
        }

        in_cb.reserve_back(1);
        uint32_t write_offset = 0;
        uint32_t processed_sticks = 0;
        // page zeroing is only necessary for tiled block output format so that scale is not affected by
        // junk/padding data
        if constexpr (zero_pages) {
            if (c_i == in_nblocks_c - 1 && last_tile_is_partial) {
                zero_out_page(noc, in_cb);
            }
        }
        // When the CB intentionally holds more rows than the kernel window (medium kernels,
        // FACE_WIDTH < kernel_size_hw < TILE_HEIGHT), the rows in
        // [total_elems_to_reduce, num_tilized_rows) are never overwritten by the async_reads
        // below and would otherwise contribute junk to the reduce. Fill only that tail region
        // with the init value -- the leading rows will be fully overwritten by process_h().
        if constexpr (!is_large_kernel) {
            if constexpr (num_tilized_rows > total_elems_to_reduce) {
                constexpr uint32_t row_stride_elems =
                    wide_reduction ? (MAX_TILES_PER_REDUCTION * TILE_WIDTH) : (in_ntiles_c * TILE_WIDTH);
                constexpr uint32_t tail_offset_bytes = total_elems_to_reduce * row_stride_elems * BYTES_PER_ELEM;
                constexpr uint32_t tail_elems = (num_tilized_rows - total_elems_to_reduce) * row_stride_elems;
                fill_with_val(
                    in_cb.get_write_ptr() + tail_offset_bytes, tail_elems, static_cast<uint16_t>(bf16_init_value));
            }
        }
        for (uint32_t h = 0; h < kernel_h; ++h) {
            auto process_h = [&](uint32_t w_offset, uint32_t w_multiple) __attribute__((always_inline)) {
                const uint32_t stick_offset = ind + w_offset + h * dilation_h * in_w_padded;
                const uint32_t read_offset =
                    in_l1_read_base_addr + (stick_offset * shard_width_bytes + c_i * MAX_BYTES_PER_REDUCTION);
                noc.async_read(
                    self_ep,
                    in_cb,
                    read_bytes * w_multiple,
                    experimental::local_addr(read_offset),
                    {.offset_bytes = write_offset});
                // if compute is using tilize_reconfig we will only untilize the needed number of tiles rather
                // than the entire MAX_TILES_PER_REDUCTION, thus we use a different offset for the write address
                if constexpr (tilize_reconfig) {
                    write_offset += read_bytes * w_multiple;
                } else {
                    write_offset += max_write_inc * w_multiple;
                }
                processed_sticks += w_multiple;
                if constexpr (is_large_kernel) {
                    if ((processed_sticks % max_sticks_for_reduction) == 0 ||
                        processed_sticks == total_elems_to_reduce) {
                        noc.async_read_barrier();
                        in_cb.push_back(1);
                        in_cb.reserve_back(1);
                        write_offset = 0;
                        // If next is last chunk, fill whole buffer with the init_value. note for max pool we do
                        // not need to fill the CB for the partial chunk since as long as we have N>1 chunks we
                        // are guaranteed that the junk data remaining from chunk N-1 will fill the entire CB and
                        // cannot contain values greater than the max value, and if we have N=1 chunks we already
                        // initialized the entire CB with the init value, but for avg pool we need to fill the
                        // entire CB with the init value since the junk data will contribute to the average.
                        if constexpr (is_avg_pool) {
                            // clear the in CB
                            if ((total_elems_to_reduce - processed_sticks) < max_sticks_for_reduction &&
                                processed_sticks != total_elems_to_reduce) {
                                clear_out_tiles<clear_value_cb_id>(noc, in_cb, clear_cb, in_cb_ntiles);
                            }
                        }
                    }
                }
            };

            // Case where in_nbytes_leftover and in_nbytes_c is different is when we are dealing with
            // tesnors that have last tile as partial. Cb page size is multiple of tile but when the last
            // tile is partial we have to read the smaller stick width. Therefore we need to write out the next
            // stick right below the previous one and this is when increment of the write pointer and the read
            // stick size is not compliant.
            bool use_contiguous_read = !wide_reduction && in_nbytes_leftover == in_nbytes_c &&
                                       dilation_w == 1;  // read entire row as one chunk (only if no width dilation)
            if constexpr (is_large_kernel) {
                bool whole_row_remaining =
                    kernel_w <= max_sticks_for_reduction - (processed_sticks % max_sticks_for_reduction);
                use_contiguous_read &= whole_row_remaining;
            }

            if (use_contiguous_read) {
                process_h(0, kernel_w);
            } else {  // read rows stick by stick with dilation
                for (uint32_t w = 0; w < kernel_w; ++w) {
                    process_h(w * dilation_w, 1);
                }
            }
        }
        if constexpr (!is_large_kernel) {
            noc.async_read_barrier();
            in_cb.push_back(1);
        }
    }
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    constexpr uint32_t reader_nindices = get_arg(args::reader_nindices);
    constexpr uint32_t kernel_h = get_arg(args::kernel_h);
    constexpr uint32_t kernel_w = get_arg(args::kernel_w);

    constexpr int32_t pad_w = get_arg(args::pad_w);

    // channel size in bytes
    constexpr uint32_t in_nbytes_leftover = get_arg(args::in_nbytes_leftover);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_arg(args::in_w);

    constexpr uint32_t in_c = get_arg(args::in_c);

    constexpr uint32_t split_reader = get_arg(args::split_reader);
    constexpr uint32_t reader_id = get_arg(args::reader_id);

    constexpr uint32_t bf16_scalar = get_arg(args::bf16_scalar);
    constexpr uint32_t bf16_init_value = get_arg(args::bf16_init_value);

    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t in_cb_sz = get_arg(args::in_cb_sz);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);
    constexpr uint32_t ceil_pad_w = get_arg(args::ceil_pad_w);

    // CB ids now come from Metal 2.0 DFB bindings. Split-reader uses per-reader input/scalar
    // DFBs bound under the same accessor names, so the kernel references one name regardless
    // of reader_id (the host binds the right DFB per reader KernelSpec).
    constexpr uint32_t in_cb_id = dfb::in_cb;
    constexpr uint32_t in_shard_cb_id = dfb::in_shard_cb;
    constexpr uint32_t in_reader_indices_cb_id = dfb::reader_indices_cb;
    constexpr uint32_t in_scalar_cb_id = dfb::in_scalar_cb;
    constexpr uint32_t clear_value_cb_id = dfb::clear_value_cb;
    constexpr bool is_avg_pool = (bool)get_arg(args::pool_type_is_avg);
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
    // The avg-pool scalar config DFB + tensor::config only exist when !one_scalar_per_core; the
    // host emits HAS_CONFIG to this kernel's defines exactly then. Gate every dfb::config_cb /
    // tensor::config reference: `if constexpr (!one_scalar_per_core)` is not enough since the
    // discarded branch still name-looks-up the (then-undeclared) tokens.
#ifdef HAS_CONFIG
    constexpr uint32_t config_cb_id = dfb::config_cb;
    constexpr uint32_t config_page_size = get_arg(args::config_page_size);
#endif
    constexpr uint32_t in_nbytes_c = get_arg(args::in_nbytes_c);
    constexpr uint32_t shard_width_bytes = get_arg(args::shard_width_bytes);
    constexpr uint32_t multi_buffering_factor = get_arg(args::multi_buffering_factor);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr bool zero_pages = (bool)get_arg(args::zero_pages);
    constexpr uint32_t config_in_dram = get_arg(args::config_in_dram);
    constexpr uint32_t reader_page_size = get_arg(args::reader_page_size);

    constexpr bool use_split_reader = split_reader;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    // The per-reader scalar DFB selection that legacy code did here (in_scalar_cb_id_1 for
    // reader1 when !one_scalar_per_core) is now done on the host: each reader KernelSpec binds
    // its own scalar DFB under accessor name "in_scalar_cb", so in_scalar_cb_id == dfb::in_scalar_cb.

    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT ? window_size_hw : FACE_HEIGHT;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_WIDTH || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;
    // we only need to initialize the in_cb if we will not fill each reduction chunk with valid data
    constexpr bool need_to_initialize_in_cb =
        (remaining_elems && face_r_dim == FACE_HEIGHT && (num_faces_in_input_tile == 4 || last_tile_is_partial) &&
         interm_reduction_chunks <= multi_buffering_factor);
    constexpr uint32_t in_cb_ntiles = in_cb_sz / (TILE_WIDTH * TILE_HEIGHT);  // only use the non-multi buffering size

    DataflowBuffer clear_value_cb(clear_value_cb_id);
    DataflowBuffer in_scalar_cb(in_scalar_cb_id);
    DataflowBuffer in_shard_cb(in_shard_cb_id);
    DataflowBuffer reader_indices_cb(in_reader_indices_cb_id);
#ifdef HAS_CONFIG
    DataflowBuffer config_cb(config_cb_id);
#endif

    // QSR max_pool fix: for a partial-face window (face_r_dim < 16, e.g. 3x3 -> 9), need_to_initialize_in_cb
    // is false, but the quasar reduce reads the FULL 16-row face while the reader fills only the populated
    // rows -> the unwritten face rows leak stale L1 into the max (value inflation; masked only when the L1
    // residue happens to be <= the data). Force the -inf pre-clear for MAX pool. -inf is the max identity,
    // so pre-clearing can never change a correct max; the once-at-init clear persists across the in_cb ring
    // because the reader never overwrites those rows. (Real fix: make the quasar reduce respect face_r_dim.)
    constexpr bool force_max_clear = !is_avg_pool;
    // fill the clear cb
    if constexpr (is_avg_pool || need_to_initialize_in_cb || force_max_clear) {
        if constexpr (reader_id == 0) {
            fill_with_val(clear_value_cb.get_write_ptr(), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);
            clear_value_cb.push_back(1);
        }
        if constexpr (reader_id == 1) {
            clear_value_cb.wait_front(1);
        }
        // for average pool clear out tiles runs in loop, no need to initialize here
        if constexpr (!is_avg_pool || !is_large_kernel) {
            clear_out_tiles<in_cb_id, clear_value_cb_id>(Noc(), DataflowBuffer(in_cb_id), clear_value_cb);
        }
    }

    // initialize the scalar CB
    if constexpr (reader_id == 0 && one_scalar_per_core) {
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        fill_with_val(in_scalar_cb.get_write_ptr(), FACE_WIDTH, bf16_scalar >> 16);
        in_scalar_cb.push_back(1);
    }
    const uint32_t core_nhw_index = get_arg(args::core_nhw_index);

    const uint32_t in_l1_read_base_addr = in_shard_cb.get_read_ptr();
    if constexpr (config_in_dram) {
        if (reader_id == 0) {
            // Inlined load_config_tensor_if_in_dram: the reader-indices tensor flows in via its
            // Metal 2.0 TensorBinding (tensor::reader_indices) instead of a CTA-baked DRAM address.
            Noc cfg_noc;
            const auto reader_indices_accessor = TensorAccessor(tensor::reader_indices);
            cfg_noc.async_read(
                reader_indices_accessor, reader_indices_cb, reader_page_size, {.page_id = core_nhw_index}, {});
            cfg_noc.async_read_barrier();
            reader_indices_cb.push_back(1);
        } else {
            reader_indices_cb.wait_front(1);
        }
    }
    uint32_t reader_indices_l1_addr = reader_indices_cb.get_read_ptr();
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);

    uint32_t segments_counter = 1;
    constexpr uint32_t total_elems_to_reduce = kernel_h * kernel_w;

    volatile tt_l1_ptr uint16_t* config_ptr;
    uint32_t scalar_index = 0;
    uint32_t scalar_start;
    uint32_t scalar_value;
    uint32_t scalar_end;
    uint32_t counter = reader_id;
    // HAS_CONFIG <=> !one_scalar_per_core (host-emitted). Gated rather than `if constexpr` because
    // the body references dfb::config_cb / tensor::config, which are only declared when HAS_CONFIG.
#ifdef HAS_CONFIG
    {
        uint32_t config_l1_addr = config_cb.get_read_ptr();
        if constexpr (config_in_dram) {
            if (reader_id == 0) {
                // Inlined load_config_tensor_if_in_dram: the scalar config tensor flows in via its
                // Metal 2.0 TensorBinding (tensor::config) instead of a CTA-baked DRAM address.
                Noc cfg_noc;
                const auto config_accessor = TensorAccessor(tensor::config);
                cfg_noc.async_read(config_accessor, config_cb, config_page_size, {.page_id = core_nhw_index}, {});
                cfg_noc.async_read_barrier();
                config_cb.push_back(1);
            } else {
                config_cb.wait_front(1);
            }
        }
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[0];
        scalar_value = config_ptr[1];
        scalar_end = config_ptr[2];
    }
#endif

    uint16_t num_segments = reader_indices_ptr[0] & 0xffff;
    bool first_row_value = reader_id == 0 || !use_split_reader;

    // [#47797 DEBUG] If POOL hangs at waypoint R, dump the loop-control values. A garbage num_segments
    // (e.g. unwritten reader_indices config) or stride_w==0 makes while(num_segments--)/the inner stride
    // loop spin forever. Compare these against the host sliding-window config for this pool.
    DPRINT(
        "POOL rdr id={} nseg={} strW={} kH={} kW={}\n",
        (uint32_t)reader_id,
        (uint32_t)num_segments,
        (uint32_t)stride_w,
        (uint32_t)kernel_h,
        (uint32_t)kernel_w);

    while (num_segments--) {
        uint32_t start_end_segment = reader_indices_ptr[segments_counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;
        DPRINT("POOL seg start={} end={}\n", (uint32_t)start, (uint32_t)end);  // [#47797 DEBUG]

        if (!first_row_value) {
            start += stride_w;
            first_row_value = true;
        }

        constexpr uint32_t stride_multiple = use_split_reader ? 2 : 1;
        for (uint16_t ind = start; ind <= end; ind += stride_multiple * stride_w) {
            if constexpr (!one_scalar_per_core) {
                fill_scalar<
                    one_scalar_per_core,
                    in_scalar_cb_id,
                    reader_nindices,
                    use_split_reader,
                    multi_buffering_factor>(
                    in_scalar_cb, scalar_start, scalar_end, scalar_value, scalar_index, counter, config_ptr);
            }
            read_kernel_with_top_left_index<
                in_nblocks_c,
                in_cb_id,
                kernel_h,
                kernel_w,
                in_w_padded,
                in_nbytes_leftover,
                in_c,
                max_sticks_for_reduction,
                total_elems_to_reduce,
                is_avg_pool,
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
                in_cb_sz,
                bf16_init_value>(ind, in_l1_read_base_addr);
            if (use_split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }
}  // kernel_main()
