// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gelu_alt_approx isolated bench: math thread copies a block of fp32 tiles into DEST (UnpackToDestFp32,
// lossless); the PACK thread applies the GELU variant on the SFPU (same sync as the op's
// apply_activation_from_pack: semwait, dest-offset flip, per-tile SFPU, STALLWAIT) and packs the block.
// Raw sfpi call via _llk_math_eltwise_unary_sfpu_params_ bypasses gelu_tanh_tile_pack on purpose: the
// variants are alternative instruction sequences for the same function.
//
// CT args: [variant, tiles_per_core, block, reps]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_PACK
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "gelu_variants.h"
#endif

namespace {

#ifdef TRISC_PACK
template <uint32_t variant>
inline void variant_init() {
    using namespace gelu_alt;
    if constexpr (variant == 1) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_base_init);
    } else if constexpr (variant == 2 || variant == 5) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_t21_init);
    } else if constexpr (variant == 3 || variant == 6) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_mag_init);
    } else if constexpr (variant == 7) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_magb_init);
    } else if constexpr (variant == 8 || variant == 9) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_t21s_init);
    } else if constexpr (variant == 4) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(gelu_hwx_init);
    } else if constexpr (variant == 12) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(probe_round_init);
    } else {
        llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(probe_none_init);
    }
}

template <uint32_t variant>
inline void variant_apply(uint32_t idst) {
    using namespace gelu_alt;
    constexpr VectorMode rc = VectorMode::RC;
    if constexpr (variant == 1) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_base<8>, idst, rc);
    } else if constexpr (variant == 2) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_t21<1, 8>, idst, rc);
    } else if constexpr (variant == 3) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_mag<1, 8>, idst, rc);
    } else if constexpr (variant == 5) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_t21<2, 8>, idst, rc);
    } else if constexpr (variant == 6) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_mag<2, 8>, idst, rc);
    } else if constexpr (variant == 7) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_magb<8>, idst, rc);
    } else if constexpr (variant == 8) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_t21s<false, 8>, idst, rc);
    } else if constexpr (variant == 9) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_t21s<true, 8>, idst, rc);
    } else if constexpr (variant == 4) {
        _llk_math_eltwise_unary_sfpu_params_(gelu_hwx<8>, idst, rc);
    } else if constexpr (variant == 10) {
        _llk_math_eltwise_unary_sfpu_params_(probe_exp<8>, idst, rc);
    } else if constexpr (variant == 11) {
        _llk_math_eltwise_unary_sfpu_params_(probe_recip<8>, idst, rc);
    } else if constexpr (variant == 12) {
        _llk_math_eltwise_unary_sfpu_params_(probe_round<8>, idst, rc);
    }
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr uint32_t reps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    if constexpr (variant != 0) {
        PACK((variant_init<variant>()));
    }

    cb_wait_front(cb_in, tiles_per_core);  // resident sharded input (pushed by the sink kernel)

    for (uint32_t rep = 0; rep < reps; ++rep) {
        for (uint32_t base = 0; base < tiles_per_core; base += block) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < block; ++i) {
                copy_tile(cb_in, base + i, i);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, block);
            if constexpr (variant != 0) {
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t i = 0; i < block; ++i) {
                    PACK((variant_apply<variant>(i)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            } else {
                tile_regs_wait();
            }
            for (uint32_t i = 0; i < block; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, block);
        }
    }
}
