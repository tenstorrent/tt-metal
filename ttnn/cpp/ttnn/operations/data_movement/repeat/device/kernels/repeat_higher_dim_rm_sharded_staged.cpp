// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    if (get_arg(args::nop) == 1) {
        return;
    }
    const auto higher_dim_start = get_arg(args::higher_dim_start);
    const auto higher_dim_end = get_arg(args::higher_dim_end);
    const auto lower_dim_start = get_arg(args::lower_dim_start);
    const auto lower_dim_end = get_arg(args::lower_dim_end);
    const auto repetitions = get_arg(args::repetitions);
    constexpr auto row_bytes = get_arg(args::original_page_size_bytes);
    constexpr auto src_page_bytes = get_arg(args::source_page_size_bytes);
    constexpr auto dst_page_bytes = get_arg(args::dest_page_size_bytes);
    constexpr auto transfer_bytes = get_arg(args::staging_page_size_bytes);
    constexpr auto lower_dims = get_arg(args::LOWER_DIMS);
    constexpr auto rep_dim = get_arg(args::REP_DIM);
    constexpr auto src_pages_per_row = div_up<row_bytes, src_page_bytes>();
    constexpr auto dst_pages_per_row = div_up<row_bytes, dst_page_bytes>();

    const auto src = TensorAccessor(tensor::src);
    const auto dst = TensorAccessor(tensor::dst);
    Scratchpad<uint8_t> read_scratch(scratch::in0);
    Scratchpad<uint8_t> write_scratch(scratch::in1);
    const uint32_t read_buffer = align_address<64>(read_scratch.local_mem(), MASK_64);
    const uint32_t write_buffer = align_address<16>(write_scratch.local_mem(), MASK_16);
    constexpr auto read_mask = decltype(src)::is_dram ? OFFSET_64 : OFFSET_16;
    Noc noc;

    for (uint32_t h = higher_dim_start; h < higher_dim_end; ++h) {
        for (uint32_t r = 0; r < rep_dim; ++r) {
            for (uint32_t l = lower_dim_start; l < lower_dim_end; ++l) {
                const uint32_t input_row = h * lower_dims * rep_dim + r * lower_dims + l;
                for (uint32_t offset = 0; offset < row_bytes;) {
                    // Payload widths may differ. End each transfer at both page boundaries, excluding padding.
                    const uint32_t src_offset = offset % src_page_bytes;
                    const uint32_t dst_offset = offset % dst_page_bytes;
                    const uint32_t bytes = std::min(
                        std::min(transfer_bytes, row_bytes - offset),
                        std::min(src_page_bytes - src_offset, dst_page_bytes - dst_offset));
                    const uint32_t src_page = input_row * src_pages_per_row + offset / src_page_bytes;
                    const uint32_t aligned_source =
                        read_buffer + (noc_traits_t<std::remove_cv_t<decltype(src)>>::src_addr<Noc::AddressType::NOC>(
                                           src, noc, {.page_id = src_page, .offset_bytes = src_offset}) &
                                       read_mask);
                    noc.async_read(
                        src,
                        CoreLocalMem<uint32_t>(aligned_source),
                        bytes,
                        {.page_id = src_page, .offset_bytes = src_offset},
                        {});
                    noc.async_read_barrier();
                    for (uint32_t n = 0; n < repetitions; ++n) {
                        const uint32_t output_row = (h * repetitions + n) * lower_dims * rep_dim + r * lower_dims + l;
                        const uint32_t dst_page = output_row * dst_pages_per_row + offset / dst_page_bytes;
                        const uint32_t aligned_dest =
                            write_buffer +
                            (noc_traits_t<std::remove_cv_t<decltype(dst)>>::dst_addr<Noc::AddressType::NOC>(
                                 dst, noc, {.page_id = dst_page, .offset_bytes = dst_offset}) &
                             OFFSET_16);
                        tt_memmove<false, false, false, transfer_bytes>(noc, aligned_dest, aligned_source, bytes);
                        noc.async_write(
                            CoreLocalMem<uint32_t>(aligned_dest),
                            dst,
                            bytes,
                            {},
                            {.page_id = dst_page, .offset_bytes = dst_offset});
                        // The next replica or chunk reuses the staging buffers.
                        noc.async_write_barrier();
                    }
                    offset += bytes;
                }
            }
        }
    }
}
