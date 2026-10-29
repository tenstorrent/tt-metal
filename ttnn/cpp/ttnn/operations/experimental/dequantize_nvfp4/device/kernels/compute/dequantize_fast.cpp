// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "nvfp4_sfpu.hpp"
#include "nvfp4_tti.hpp"
#endif

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    constexpr uint32_t planes = get_compile_time_arg_val(0);
    constexpr bool fpu_tensor_scale = get_compile_time_arg_val(5);
    constexpr bool transpose_output = get_compile_time_arg_val(6);
    constexpr uint32_t global_bits = get_compile_time_arg_val(3);
    constexpr uint32_t global_bf16_bits = (global_bits + 0x7fff + ((global_bits >> 16) & 1)) & 0xffff0000;
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer data_cb(0), scale_cb(1), global_cb(3), output_cb(16);
    if constexpr (fpu_tensor_scale) {
        global_cb.wait_front(1);
    }
    MATH((sfpu::nvfp4_configure_raw_mask()));
    for (uint32_t tile = 0; tile < count; ++tile) {
        data_cb.wait_front(1);
        scale_cb.wait_front(planes * 4);
        tile_regs_acquire();
        reconfig_full_operand(0, 0);
        copy_init(0);
        copy_tile(0, 0, 0);
        reconfig_full_operand(1, 1);
        copy_init(1);
        // Sixteen contiguous 32-byte CB pages form one 16x16 scale tile.
        // Expand its rows once in DEST instead of broadcasting every face.
        copy_tile(1, 0, 1);
        MATH((sfpu::nvfp4_convert_begin()));
        MATH((sfpu::nvfp4_expand_scale_vectors()));
        MATH((sfpu::nvfp4_init_raw()));
#pragma GCC unroll 4
        for (uint32_t plane = 0; plane < planes; ++plane) {
#pragma GCC unroll 4
            for (uint32_t face = 0; face < 4; ++face) {
                MATH((sfpu::nvfp4_decode_column_face_raw<fpu_tensor_scale, global_bf16_bits>(plane, face)));
            }
        }
        MATH((sfpu::nvfp4_convert_end()));
        if constexpr (!transpose_output) {
            // Restore ordinary tiles; transposed output keeps these orientations
            // and lets the writer exchange the tile-grid coordinates instead.
            transpose_dest_init();
            for (uint32_t plane = 0; plane < planes; ++plane) {
                transpose_dest(plane + 2);
            }
        }
        // FPU multiplication uses a BF16 tensor scale, with decoded output
        // reused directly from DEST. No intermediate L1 pack/unpack is needed.
        if constexpr (fpu_tensor_scale) {
            reconfig_full_operand(3, 3);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(3);
            for (uint32_t plane = 0; plane < planes; ++plane) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(3, 0, plane + 2);
            }
        }
        tile_regs_commit();
        output_cb.reserve_back(planes);
        tile_regs_wait();
        for (uint32_t plane = 0; plane < planes; ++plane) {
            pack_tile(plane + 2, 16);
        }
        tile_regs_release();
        output_cb.push_back(planes);
        data_cb.pop_front(1);
        scale_cb.pop_front(planes * 4);
    }
}
