// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

template <uint32_t copy_words>
FORCE_INLINE void copy_rows(
    volatile tt_l1_ptr uint32_t* src,
    uint32_t src_step_words,
    volatile tt_l1_ptr uint32_t* dst,
    uint32_t dst_step_words,
    uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
#pragma GCC unroll 8
        for (uint32_t w = 0; w < copy_words; ++w) {
            dst[w] = src[w];
        }
        src += src_step_words;
        dst += dst_step_words;
    }
}

void kernel_main() {
    constexpr uint32_t segment_rows = get_arg(args::segment_rows);
    constexpr uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t nb = get_arg(args::num_banks);
    constexpr uint32_t copy_words = get_arg(args::copy_words);
    constexpr bool producer = get_arg(args::producer) != 0;
#ifdef NUM_DRAM_BANKS
    static_assert(nb == NUM_DRAM_BANKS);
#endif

    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t row_begin = get_arg(args::row_begin);
    const uint32_t row_end = get_arg(args::row_end);
    const uint32_t dst_stride = get_arg(args::dst_stride);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    Scratchpad<uint8_t> scratch_pad(scratch::scratch);
    const auto s0 = TensorAccessor(tensor::src);

    const LocalTensorAccessor<uint32_t> dst(tensor::dst);
    if constexpr (producer) {
        dfb.reserve_back(block_height);
    }
    volatile tt_l1_ptr uint32_t* const dst_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&dst[0]);
    volatile tt_l1_ptr uint32_t* const scratch_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&scratch_pad[0]);
    const uint32_t dst_step_words = dst_stride / sizeof(uint32_t);

    constexpr uint32_t pass_rows = segment_rows * nb;
    constexpr uint32_t half_bytes = pass_rows * page_stride;
    uint32_t first[2][nb];
    uint32_t count[2][nb];
    uint32_t stage_off[2][nb];
    auto issue = [&](uint32_t h, uint32_t r0, uint32_t n) {
        const uint32_t first_id = start_id + r0;
        const uint32_t end_id = first_id + n;
        const uint32_t phase = first_id % nb;
        uint32_t off = h * half_bytes;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t f = first_id + (b + nb - phase) % nb;
            const uint32_t c = (f < end_id) ? (end_id - 1 - f) / nb + 1 : 0;
            first[h][b] = f;
            count[h][b] = c;
            stage_off[h][b] = off;
            if (c) {
                noc.async_read<NocOptions::TXN_ID>(
                    s0,
                    scratch_pad,
                    c * page_stride,
                    {.page_id = f, .offset_bytes = 0},
                    {.offset_bytes = off},
                    NocOptVals{.trid = static_cast<uint8_t>(1 + h)});
            }
            off += c * page_stride;
        }
    };
    auto drain = [&](uint32_t h) {
        noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = static_cast<uint8_t>(1 + h)});
        for (uint32_t b = 0; b < nb; ++b) {
            copy_rows<copy_words>(
                scratch_words + stage_off[h][b] / sizeof(uint32_t),
                page_stride / sizeof(uint32_t),
                dst_words + (first[h][b] - start_id) * dst_step_words,
                nb * dst_step_words,
                count[h][b]);
        }
    };
    uint32_t cur_r0 = row_begin;
    uint32_t cur_n = (row_end - row_begin < pass_rows) ? (row_end - row_begin) : pass_rows;
    uint32_t cur_h = 0;
    if (cur_n) {
        issue(cur_h, cur_r0, cur_n);
    }
    while (cur_n) {
        const uint32_t nxt_r0 = cur_r0 + cur_n;
        const uint32_t left = row_end - nxt_r0;
        const uint32_t nxt_n = left < pass_rows ? left : pass_rows;
        if (nxt_n) {
            issue(cur_h ^ 1, nxt_r0, nxt_n);
        }
        drain(cur_h);
        cur_r0 = nxt_r0;
        cur_n = nxt_n;
        cur_h ^= 1;
    }

    UnicastEndpoint self_ep;
    noc.set_async_read_state<NocOptions::TXN_ID>(
        self_ep,
        /*size_bytes=*/0,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = 0},
        NocOptVals{.trid = 0});

    if constexpr (producer) {
        dfb.push_back(block_height);
    } else {
        dfb.wait_front(block_height);
        dfb.pop_front(block_height);
    }
}
