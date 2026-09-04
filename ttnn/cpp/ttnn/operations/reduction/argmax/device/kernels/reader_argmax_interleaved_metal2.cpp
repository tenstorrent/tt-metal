// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_argmax_interleaved.cpp, which lives beside this file and still serves
// consumers on the legacy ProgramDescriptor API. Keep the two in sync until the legacy copy is
// retired; the only differences should be the named arguments and the named resource bindings.

#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <stdint.h>

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr auto src_page_size = get_arg(args::src_page_size);
    constexpr auto dst_page_size = get_arg(args::dst_page_size);

    // This is the number of elements in the output, excluding the last two dimensions.
    // i.e. for an input tensor of shape (.., N, C, H, W), this is (.. * N * C)
    // It also depends on the `keepdim`
    constexpr auto outer_dim_units = get_arg(args::outer_dim_units);

    // This is the number of elements in the last dimension of the output
    // i.e. for an input tensor of shape (.., N, C, H, W), this is H.
    // This dictates the page size in the output dfb
    constexpr auto inner_dim_units = get_arg(args::inner_dim_units);

    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr auto red_dim_units = get_arg(args::red_dim_units);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_arg(args::reduce_all);

    //-------------------------------------------------------------------------
    const auto s_src = TensorAccessor(tensor::src);
    const auto s_dst = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer src_dfb(dfb::src);
    DataflowBuffer dst_dfb(dfb::dst);

    // DFB in L1 memory for storing input
    const uint32_t src_dfb_addr = src_dfb.get_write_ptr();
    constexpr DataFormat src_dfb_addr_data_format = get_dataformat(dfb::src);

    // DFB in L1 memory for storing output
    const uint32_t dst_dfb_addr = dst_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_dfb_addr);

    uint32_t max_idx = 0;
    auto max_val = get_default_value<src_dfb_addr_data_format>();

    //-------------------------------------------------------------------------
    // Main loop - run by all cores
    for (uint32_t k = 0; k < outer_dim_units; ++k) {
        for (uint32_t j = 0; j < inner_dim_units; ++j) {
            noc.async_read(s_src, src_dfb, src_page_size, {.page_id = k * inner_dim_units + j}, {.offset_bytes = 0});
            noc.async_read_barrier();

            // Reset max_val for each new output
            if constexpr (not reduce_all) {
                max_idx = 0;
                max_val = get_default_value<src_dfb_addr_data_format>();
            }

            for (uint32_t i = 0; i < red_dim_units; ++i) {
                compare_values<src_dfb_addr_data_format>(
                    src_dfb_addr, max_val, max_idx, i, j, k, red_dim_units, reduce_all, inner_dim_units);
            }
            if constexpr (not reduce_all) {
                out_idxs[j] = max_idx;
            }
        }

        // The results were written at dst_dfb's write pointer, and that is the address this send
        // reads from: nothing on this DFB ever calls reserve_back/push_back/wait_front/pop_front, so
        // its read and write pointers both stay at the buffer base for the whole kernel.
        if constexpr (not reduce_all) {
            noc.async_write(dst_dfb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = k});
            noc.async_write_barrier();
        }
    }

    // TODO: Generalize write for argmax for other dims
    if constexpr (reduce_all) {
        out_idxs[0] = max_idx;
        noc.async_write(dst_dfb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = 0});
        noc.async_write_barrier();
    }
}
