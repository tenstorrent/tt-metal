// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_writer.cpp — BRISC1 / writer for device-side BATCH FFT.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "batch_fft_common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_tile_idx = get_arg(args::base_tile_idx);
    const uint32_t batch_per_core = get_arg(args::batch_per_core);
    // OUTPUT_BF16: when defined, convert fp32 STATE → bf16 in CB_OUT_*_BF16 and
    // write bf16 tiles (2048 B) to the output buffers. Default 0 preserves
    // the legacy fp32 fast path.
    constexpr uint32_t SUB_N = get_arg(args::sub_n);

    // Output generators.  See reader for full rationale: we MUST use
    // computed from page_size (aligned to dram_alignment) instead of the
    // hardcoded tile size.  Otherwise ROW_MAJOR tensors with page_size <
    // tile_size scribble at the wrong bank offset once tile_idx wraps
    // past the number of DRAM banks (12 on WH, 8 on BH).
    //
    // no operator= ; construct directly with the right page_size.

    const auto out_r_gen = TensorAccessor(tensor::out_r);
    const auto out_i_gen = TensorAccessor(tensor::out_i);

    Noc noc;
    DataflowBuffer cb_sync(dfb::sync);
    DataflowBuffer cb_state_r(dfb::state_r);
    DataflowBuffer cb_state_i(dfb::state_i);
#ifdef OUTPUT_BF16
    DataflowBuffer cb_out_r_bf16(dfb::out_r_bf16);
    DataflowBuffer cb_out_i_bf16(dfb::out_i_bf16);
#endif

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_sync.wait_front(1);
        cb_state_r.wait_front(1);
        cb_state_i.wait_front(1);

#ifdef OUTPUT_BF16
        {
            // Convert fp32 STATE → bf16 in CB_OUT_*_BF16, then DMA bf16 tile.
            cb_out_r_bf16.reserve_back(1);
            cb_out_i_bf16.reserve_back(1);
            const uint32_t out_r_bf16_l1 = cb_out_r_bf16.get_write_ptr();
            const uint32_t out_i_bf16_l1 = cb_out_i_bf16.get_write_ptr();

            volatile tt_l1_ptr uint32_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_state_r.get_read_ptr());
            volatile tt_l1_ptr uint32_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_state_i.get_read_ptr());
            volatile tt_l1_ptr uint16_t* const dst_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            // Truncation (drop low 16 bits). Round-to-nearest-even costs
            // one add + one mask per element; truncation is fine for the
            // first cut and avoids any RNE corner cases.
            for (uint32_t i = 0; i < SUB_N; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_out_r_bf16.push_back(1);
            cb_out_i_bf16.push_back(1);

            noc.async_write(cb_out_r_bf16, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write(cb_out_i_bf16, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write_barrier();

            cb_out_r_bf16.pop_front(1);
            cb_out_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_write(cb_state_r, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write(cb_state_i, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write_barrier();
        }
#endif

        cb_sync.pop_front(1);
        cb_state_r.pop_front(1);
        cb_state_i.pop_front(1);
    }
}
