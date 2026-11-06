// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Copies the stride_w pixels of one chunk: pixel_size bytes from each source pixel, which are aligned_pixel_size apart,
// packed back to back at dst. T is the widest unit that divides pixel_size, so every access stays aligned.
template <typename T, uint32_t pixel_size, uint32_t aligned_pixel_size, uint32_t stride_w>
FORCE_INLINE void copy_chunk(uint32_t src_addr, uint32_t dst_addr) {
    const T* src = reinterpret_cast<const T*>(src_addr);
    T* dst = reinterpret_cast<T*>(dst_addr);
    for (uint32_t w = 0; w < stride_w; ++w) {
        for (uint32_t i = 0; i < pixel_size / sizeof(T); ++i) {
            dst[i] = src[i];
        }
        src += aligned_pixel_size / sizeof(T);
        dst += pixel_size / sizeof(T);
    }
}

void kernel_main() {
    // src0 / dst0 are borrowed-memory DFBs.
    constexpr uint32_t pixel_size = get_arg(args::pixel_size);
    constexpr uint32_t aligned_pixel_size = get_arg(args::aligned_pixel_size);
    constexpr uint32_t aligned_dst_pixel_size = get_arg(args::aligned_dst_pixel_size);
    constexpr uint32_t aligned_chunk_size = get_arg(args::aligned_chunk_size);
    constexpr uint32_t aligned_row_size = get_arg(args::aligned_row_size);
    constexpr uint32_t stride_h = get_arg(args::stride_h);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t num_dst_rows = get_arg(args::num_dst_rows);
    constexpr uint32_t num_dst_cols = get_arg(args::num_dst_cols);
    constexpr uint32_t dst_row_offset = get_arg(args::dst_row_offset);
    constexpr uint32_t is_reader = get_arg(args::is_reader);

    constexpr uint32_t dst_row_size = num_dst_cols * aligned_dst_pixel_size;
    constexpr uint32_t cols_per_core = num_dst_cols / 2;
    constexpr uint32_t process_cols = cols_per_core + ((num_dst_cols % 2) & is_reader);
    constexpr uint32_t core_col_offset = is_reader ? 0 : aligned_chunk_size;
    constexpr uint32_t core_dst_offset = is_reader ? 0 : aligned_dst_pixel_size;

    // pixel_size is the logical channels' bytes; aligned_pixel_size is the pitch of the input's pixels, which is wider
    // when the pixel is not 16-byte aligned or the shard is wider than the channels.
    constexpr bool is_aligned = (pixel_size == aligned_pixel_size);
    // A pixel narrower than its pitch but L1-aligned (e.g. C = 16 bf16 in 32-wide shards) moves with one NoC read.
    constexpr bool read_each_pixel = !is_aligned && pixel_size % L1_ALIGNMENT == 0 && pixel_size <= NOC_MAX_BURST_SIZE;
    // Otherwise the RISC copies it. The pitch is a multiple of the L1 alignment, so pixel_size alone sets the unit.
    static_assert(aligned_pixel_size % L1_ALIGNMENT == 0);
    constexpr uint32_t copy_unit = pixel_size % 4 == 0 ? 4 : (pixel_size % 2 == 0 ? 2 : 1);

    Noc noc;
    DataflowBuffer dfb_src0(dfb::src0);
    DataflowBuffer dfb_dst0(dfb::dst0);

    const uint32_t dst_addr_base = dfb_dst0.get_write_ptr();
    uint32_t src_addr_base = dfb_src0.get_read_ptr();

    if constexpr (is_aligned) {
        experimental::set_read_state<aligned_chunk_size>(noc, src_addr_base);
    } else if constexpr (read_each_pixel) {
        experimental::set_read_state<pixel_size>(noc, src_addr_base);
    }

    // Process each destination row
    for (uint32_t row = 0; row < num_dst_rows; ++row) {
        uint32_t src_col_offset = core_col_offset;
        uint32_t dst_addr = dst_addr_base + (row * dst_row_size) + core_dst_offset;

        // Process columns assigned to this core
        for (uint32_t col = 0; col < process_cols; ++col) {
            uint32_t dst_pixel_addr = dst_addr;
            // Gather pixels along stride_h dimension
            for (uint32_t h = 0; h < stride_h; ++h) {
                const uint32_t h_offset = h * aligned_row_size;
                if constexpr (is_aligned) {
                    // Fast path: aligned NOC read
                    experimental::read_with_state(noc, dst_pixel_addr, src_addr_base + src_col_offset + h_offset);
                    dst_pixel_addr += aligned_chunk_size;
                } else if constexpr (read_each_pixel) {
                    uint32_t src_pixel_addr = src_addr_base + src_col_offset + h_offset;
                    for (uint32_t w = 0; w < stride_w; ++w) {
                        experimental::read_with_state(noc, dst_pixel_addr, src_pixel_addr);
                        src_pixel_addr += aligned_pixel_size;
                        dst_pixel_addr += pixel_size;
                    }
                } else {
                    // Slow path: copy each pixel's logical bytes and pack them.
                    const uint32_t src_chunk_addr = src_addr_base + src_col_offset + h_offset;
                    if constexpr (copy_unit == 4) {
                        copy_chunk<uint32_t, pixel_size, aligned_pixel_size, stride_w>(src_chunk_addr, dst_pixel_addr);
                    } else if constexpr (copy_unit == 2) {
                        copy_chunk<uint16_t, pixel_size, aligned_pixel_size, stride_w>(src_chunk_addr, dst_pixel_addr);
                    } else {
                        copy_chunk<uint8_t, pixel_size, aligned_pixel_size, stride_w>(src_chunk_addr, dst_pixel_addr);
                    }
                    dst_pixel_addr += pixel_size * stride_w;
                }
            }
            // Move to next column (2 cores interleaved)
            src_col_offset += aligned_chunk_size * 2;
            dst_addr += aligned_dst_pixel_size * 2;
        }
        // Move to next row
        src_addr_base += dst_row_offset;
    }

    noc.async_read_barrier();
}
