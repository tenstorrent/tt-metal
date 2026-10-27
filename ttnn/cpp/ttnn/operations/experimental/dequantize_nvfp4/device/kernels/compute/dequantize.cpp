// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "nvfp4_sfpu.hpp"
#endif

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    constexpr uint32_t planes = get_compile_time_arg_val(0);
    constexpr bool byte_input = get_compile_time_arg_val(1);
    constexpr bool e2m1_scale = get_compile_time_arg_val(2);
    constexpr uint32_t global_scale_bits = get_compile_time_arg_val(3);
    constexpr bool bf16_output = get_compile_time_arg_val(4);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer data_cb(0), scale_cb(1), output_cb(16);
    copy_init(0);
    for (uint32_t tile = 0; tile < count; ++tile) {
        data_cb.wait_front(1);
        scale_cb.wait_front(1);
        // Hand the L1 pointer from UNPACK to MATH through the existing mailbox API.
        const uint32_t scales_address = scale_cb.get_tile_address(0);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        // Full DEST synchronization makes room for the input plus all four
        // expanded planes, including when DEST uses 32 bits per element.
        for (uint32_t plane = 0; plane < planes; ++plane) {
            MATH((sfpu::nvfp4_decode_plane<byte_input, e2m1_scale, DST_ACCUM_MODE, bf16_output>(
                reinterpret_cast<const volatile uint32_t*>(scales_address + plane * 64),
                plane,
                sfpu::Converter::as_float(global_scale_bits))));
        }
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
