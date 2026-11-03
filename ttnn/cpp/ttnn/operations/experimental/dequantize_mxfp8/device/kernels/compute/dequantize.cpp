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
// The harness's isolated-math diagnostic repeats the code between the sweep
// BEGIN/END marker comments below (each must appear exactly once); keep them
// around the complete per-packed-tile decode sweep.
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    constexpr uint32_t planes = get_compile_time_arg_val(0);
    constexpr bool byte_input = get_compile_time_arg_val(1);
    // One E8M0 byte per output-tile row: 32 bytes = 8 words per plane.
    constexpr uint32_t scale_words_per_tile = 8 * planes;
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer data_cb(0), scale_cb(1), output_cb(16);
    copy_init(0);
    for (uint32_t tile = 0; tile < count; ++tile) {
        data_cb.wait_front(1);
        scale_cb.wait_front(1);
        // UNPACK reads the scale words and hands them to MATH (and PACK) through the
        // mailbox FIFO before it pops the page. MATH must not read the page from L1
        // directly: UNPACK runs up to two tiles ahead and the reader then refills it.
        uint32_t scale_words[scale_words_per_tile];
        for (uint32_t word = 0; word < scale_words_per_tile; ++word) {
            scale_words[word] = scale_cb.read_tile_value(0, word);
        }
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        // MXFP8_MATH_SWEEP_BEGIN
        for (uint32_t plane = 0; plane < planes; ++plane) {
            MATH((sfpu::mxfp8_decode_plane<byte_input, DST_ACCUM_MODE>(scale_words + plane * 8, plane)));
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
