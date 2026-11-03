// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

// PLACEHOLDER compute kernel: synchronization only, no decode. It consumes one
// packed tile and one scale page per iteration and publishes `planes` output
// tiles WITHOUT writing them, so outputs are undefined and the accuracy gate
// must reject it. It exists so the host operation, dataflow kernels, tests and
// harness can be exercised before a real decoder is written.
//
// Compile-time args: 0 planes (output tiles per packed tile: 1 for UINT8,
// 2 for UINT16), 1 UINT8 input, 2 BFLOAT16 output, 3 fp32_dest_acc_en.
// Runtime arg 0: packed tiles assigned to this core.
//
// The harness's isolated-math diagnostic repeats the code between the
// MXFP8_MATH_SWEEP_BEGIN/END markers; keep both markers around the complete
// per-packed-tile decode sweep in a real kernel.
void kernel_main() {
    constexpr uint32_t planes = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer data_cb(0), scale_cb(1), output_cb(16);
    for (uint32_t tile = 0; tile < count; ++tile) {
        data_cb.wait_front(1);
        scale_cb.wait_front(1);
        output_cb.reserve_back(planes);
        // MXFP8_MATH_SWEEP_BEGIN
        // MXFP8_MATH_SWEEP_END
        output_cb.push_back(planes);
        data_cb.pop_front(1);
        scale_cb.pop_front(1);
    }
}
