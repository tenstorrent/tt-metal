// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for pixel_unshuffle on NCHW ROW_MAJOR input.
//
// For each assigned output stick (n, c_out, h_out), decode (c_in, rh) from c_out
// according to the channel ordering (CTA[9]):
//   CHANNEL_MAJOR (0):  c_in = c_out / r²      rh = (c_out % r²) / r
//   SPATIAL_MAJOR (1):  c_in = c_out % C       rh = (c_out / C) / r
//   input_page = n * C * H + c_in * H + h_out * r + rh   (same formula for both)
//
// Reads the full input row (W elements) into CB. The writer scatters every r-th
// element starting at rw, which it decodes from c_out with the matching ordering.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_nbytes_in = get_compile_time_arg_val(0);       // W * datum_size
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);             // CB index (0)
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(2);  // aligned W * datum_size
    constexpr uint32_t r = get_compile_time_arg_val(3);                     // downscale factor
    constexpr uint32_t Ho = get_compile_time_arg_val(4);                    // H / r
    constexpr uint32_t W = get_compile_time_arg_val(5);                     // input width (unused here)
    constexpr uint32_t C = get_compile_time_arg_val(6);                     // input channels
    constexpr uint32_t H = get_compile_time_arg_val(7);                     // input height
    constexpr uint32_t N = get_compile_time_arg_val(8);                     // batch size
    constexpr uint32_t channel_order = get_compile_time_arg_val(9);         // 0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR
    constexpr auto src_args = TensorAccessorArgs<10>();                     // TensorAccessor starts at CTA[10]

    constexpr uint32_t CHANNEL_MAJOR = 0;
    constexpr uint32_t SPATIAL_MAJOR = 1;

    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);   // starting output stick index
    uint32_t num_sticks = get_arg_val<uint32_t>(2);  // number of output sticks for this core

    const auto s_in = TensorAccessor(src_args, src_addr);
    experimental::Noc noc;
    experimental::CB cb_in0(cb_id_in0);

    constexpr uint32_t r2 = r * r;
    constexpr uint32_t C_out_per_C = r2;  // C_out = C * r^2; per input channel: r^2 output channels
    const uint32_t C_out = C * r2;

    // Decode start_idx -> (n, c_out, h_out)
    uint32_t n = start_idx / (C_out * Ho);
    uint32_t rem = start_idx % (C_out * Ho);
    uint32_t c_out = rem / Ho;
    uint32_t h_out = rem % Ho;

    // Precompute c_in, rh for starting c_out (branch resolved at compile time)
    uint32_t c_in, rh;
    if constexpr (channel_order == SPATIAL_MAJOR) {
        c_in = c_out % C;
        rh = (c_out / C) / r;
    } else {  // CHANNEL_MAJOR
        c_in = c_out / r2;
        rh = (c_out % r2) / r;
    }

    for (uint32_t i = 0; i < num_sticks; i++) {
        // input_page = n*C*H + c_in*H + h_out*r + rh
        uint32_t input_page = n * (C * H) + c_in * H + h_out * r + rh;

        cb_in0.reserve_back(1);
        noc.async_read(s_in, cb_in0, stick_nbytes_in, {.page_id = input_page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);

        // Advance indices
        h_out++;
        if (h_out == Ho) {
            h_out = 0;
            c_out++;
            if (c_out == C_out) {
                c_out = 0;
                n++;
            }
            // Recompute c_in and rh only when c_out changes
            if constexpr (channel_order == SPATIAL_MAJOR) {
                c_in = c_out % C;
                rh = (c_out / C) / r;
            } else {  // CHANNEL_MAJOR
                c_in = c_out / r2;
                rh = (c_out % r2) / r;
            }
        }
    }
}
