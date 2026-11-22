// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.unpack_fp8_rows (fp8_rows.py), compute. Per band: the 4 FP8_E4M3 tiles copied into a fp32 DEST
// (exact), the scale tile copied in at fp32 (UnpackToDestFp32), every value times its row's scale on the SFPU (fp32),
// then rounded to nearest even onto the bfloat8_b grid (ckv_bfp4_ttnn's ckv_round_bfp_face<7>, the host packer's
// rule) and packed as bfloat8_b (the pack only copies bits: the values are on the grid).
// runtime (value slot): nbands; the band split as the reader's.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "../../ckv_bfp4_ttnn/kernels/ckv_round_bfp4.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_e = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t NCORES = get_compile_time_arg_val(4);
    constexpr uint32_t SC = 4;  // the scale's DEST tile
#ifdef UNPACK_COMMON_VALUES     // (option common_values: common arg 0)
    const uint32_t nbands = get_common_arg_val<uint32_t>(0);
#else
    const uint32_t nbands = get_arg_val<uint32_t>(0);
#endif
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t count = nbands / NCORES + (c < nbands % NCORES ? 1u : 0u);
    if (count == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_e, cb_out);
    pack_init(cb_out);
    for (uint32_t b = 0; b < count; ++b) {
        cb_wait_front(cb_e, 4);
        cb_wait_front(cb_s, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_e);  // (copy_init only checks the state; the unpacker is reconfigured here)
        copy_init(cb_e);
        for (uint32_t t = 0; t < 4; ++t) {
            copy_tile(cb_e, t, t);
        }
        reconfig_data_format_srca(cb_s);  // fp32, unpacked straight to DEST (UnpackToDestFp32)
        copy_init(cb_s);
        copy_tile(cb_s, 0, SC);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < 4; ++t) {
            mul_binary_tile(t, SC, t);
        }
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
        for (uint32_t t = 0; t < 4; ++t) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ckv_round_bfp_face<7>, t, VectorMode::RC)));
        }
        tile_regs_commit();
        cb_pop_front(cb_e, 4);
        cb_pop_front(cb_s, 1);
        cb_reserve_back(cb_out, 4);
        tile_regs_wait();
        for (uint32_t t = 0; t < 4; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, 4);
    }
}
