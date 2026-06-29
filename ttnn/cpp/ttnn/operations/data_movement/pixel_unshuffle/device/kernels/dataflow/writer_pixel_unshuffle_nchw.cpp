// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for pixel_unshuffle on NCHW ROW_MAJOR output.
//
// Pops each CB page (a full W-element input row) and scatters every r-th
// element starting at offset rw = c_out % r into the output stick.
// The Wo packed output elements are written contiguously to DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_nbytes_in = get_compile_time_arg_val(0);       // W * datum_size (CB page)
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);             // input CB index
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(2);  // aligned input stick size
    constexpr uint32_t r = get_compile_time_arg_val(3);                     // downscale factor
    constexpr uint32_t Ho = get_compile_time_arg_val(4);                    // H / r
    constexpr uint32_t W = get_compile_time_arg_val(5);                     // input width
    constexpr uint32_t C = get_compile_time_arg_val(6);                     // input channels
    constexpr uint32_t H = get_compile_time_arg_val(7);                     // input height
    constexpr uint32_t N = get_compile_time_arg_val(8);                     // batch size
    constexpr uint32_t stick_nbytes_out = get_compile_time_arg_val(9);      // Wo * datum_size
    constexpr uint32_t cb_id_scratch = get_compile_time_arg_val(10);        // scratch CB index (1)
    constexpr auto dst_args = TensorAccessorArgs<11>();                     // TensorAccessor at CTA[11]
    // Derive bytes-per-element from existing CTAs — no extra CTA needed.
    // stick_nbytes_in = W * datum_sz  →  datum_nbytes = stick_nbytes_in / W
    constexpr uint32_t datum_nbytes = stick_nbytes_in / W;

    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);   // starting output stick index
    uint32_t num_sticks = get_arg_val<uint32_t>(2);  // sticks for this core

    const auto s_out = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    experimental::CB cb_in0(cb_id_in0);
    experimental::CB cb_scratch(cb_id_scratch);

    constexpr uint32_t r2 = r * r;
    constexpr uint32_t Wo = W / r;  // output width
    const uint32_t C_out = C * r2;

    // Decode start_idx -> (n, c_out, h_out)
    uint32_t c_out = (start_idx % (C_out * Ho)) / Ho;
    uint32_t h_out = start_idx % Ho;
    uint32_t rw = c_out % r;

    uint32_t dst_page = start_idx;  // output pages are in order

    for (uint32_t i = 0; i < num_sticks; i++) {
        cb_in0.wait_front(1);
        uint32_t src_l1 = cb_in0.get_read_ptr();

        // Scatter: pick every r-th element starting at rw.
        // Element width differs by dtype: 2 bytes (bfloat16) or 4 bytes (float32).
        if constexpr (datum_nbytes == 2) {
            volatile tt_l1_ptr uint16_t* scratch = (volatile tt_l1_ptr uint16_t*)cb_scratch.get_write_ptr();
            volatile tt_l1_ptr uint16_t* src = (volatile tt_l1_ptr uint16_t*)src_l1;
            for (uint32_t w = 0; w < Wo; w++) {
                scratch[w] = src[w * r + rw];
            }
        } else {
            // 4-byte elements (float32, uint32, int32)
            volatile tt_l1_ptr uint32_t* scratch = (volatile tt_l1_ptr uint32_t*)cb_scratch.get_write_ptr();
            volatile tt_l1_ptr uint32_t* src = (volatile tt_l1_ptr uint32_t*)src_l1;
            for (uint32_t w = 0; w < Wo; w++) {
                scratch[w] = src[w * r + rw];
            }
        }

        // Write packed output stick to DRAM
        noc.async_write(
            use<experimental::CB::AddrSelector::WRITE_PTR>(cb_scratch),
            s_out,
            stick_nbytes_out,
            {},
            {.page_id = dst_page});
        noc.async_write_barrier();

        cb_in0.pop_front(1);
        dst_page++;

        // Advance indices
        h_out++;
        if (h_out == Ho) {
            h_out = 0;
            c_out++;
            if (c_out == C_out) {
                c_out = 0;
            }
            rw = c_out % r;
        }
    }
}
