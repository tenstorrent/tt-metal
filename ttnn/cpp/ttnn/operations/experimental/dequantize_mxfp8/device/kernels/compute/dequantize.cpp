// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "mxfp8_sfpu.hpp"
#endif

// SFPU reference decoder. Per packed tile: copy the integer words into D0, then
// for each plane decode one 32x32 output tile into D(plane + 1) and pack it.
//
// Compile-time args: 0 planes (output tiles per packed tile: 1 for UINT8,
// 2 for UINT16), 1 UINT8 input, 2 BFLOAT16 output, 3 fp32_dest_acc_en.
// Runtime arg 0: packed tiles assigned to this core.
//
// The harness's isolated-math diagnostic repeats the code between the
// MXFP8_MATH_SWEEP_BEGIN/END markers; keep both markers around the complete
// per-packed-tile decode sweep.
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    constexpr uint32_t planes = get_compile_time_arg_val(0);
    constexpr bool byte_input = get_compile_time_arg_val(1);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer data_cb(0), scale_cb(1), output_cb(16);
    copy_init(0);
    for (uint32_t tile = 0; tile < count; ++tile) {
        data_cb.wait_front(1);
        scale_cb.wait_front(1);
        // Hand the L1 pointer from UNPACK to MATH through the mailbox API.
        const uint32_t scales_address = scale_cb.get_tile_address(0);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        // MXFP8_MATH_SWEEP_BEGIN
        for (uint32_t plane = 0; plane < planes; ++plane) {
            MATH((sfpu::mxfp8_decode_plane<byte_input, DST_ACCUM_MODE>(
                reinterpret_cast<const volatile uint32_t*>(scales_address + plane * 32), plane)));
        }
        // MXFP8_MATH_SWEEP_END
        tile_regs_commit();
        output_cb.reserve_back(planes);
        tile_regs_wait();
        for (uint32_t plane = 0; plane < planes; ++plane) {
            pack_tile(plane + 1, 16);
        }
        tile_regs_release();
        output_cb.push_back(planes);
        data_cb.pop_front(1);
        scale_cb.pop_front(1);
    }
}
