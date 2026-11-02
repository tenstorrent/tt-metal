// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// End-to-end swap for the op's compute kernel copy: GeluTanhActivation<ID> with ID >= 100 routes the packer
// GELU to a gelu_alt variant via explicit specializations of the kernel_lib activation helpers (the shared
// header is untouched; only a kernel that names these IDs sees them). Include after
// sfpu_activation_helpers.hpp.
#pragma once

#ifdef TRISC_PACK
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "gelu_variants.h"
#endif

namespace compute_kernel_lib {

#define GA_SPECIALIZE(ID, APPLY, INIT)                                                                           \
    template <>                                                                                                  \
    struct ActivationInitHelper<KernelActivation::GELU_TANH, ID, 0> {                                            \
        FORCE_INLINE static void init() { PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(INIT))); } \
    };                                                                                                           \
    template <>                                                                                                  \
    struct ActivationApplyHelper<KernelActivation::GELU_TANH, ID, 0, 0> {                                        \
        FORCE_INLINE static void apply(uint32_t tile_index) {                                                    \
            PACK((_llk_math_eltwise_unary_sfpu_params_(APPLY, tile_index, VectorMode::RC)));                     \
        }                                                                                                        \
    };

#ifdef TRISC_PACK
namespace ga_entry {
inline void base() { gelu_alt::gelu_base<8>(); }
inline void t21() { gelu_alt::gelu_t21<1, 8>(); }
inline void t21s() { gelu_alt::gelu_t21s<false, 8>(); }
inline void t21sh() { gelu_alt::gelu_t21s<true, 8>(); }
inline void magb() { gelu_alt::gelu_magb<8>(); }
}  // namespace ga_entry
#endif

GA_SPECIALIZE(100, ga_entry::base, gelu_alt::gelu_base_init)
GA_SPECIALIZE(101, ga_entry::t21, gelu_alt::gelu_t21_init)
GA_SPECIALIZE(102, ga_entry::t21s, gelu_alt::gelu_t21s_init)
GA_SPECIALIZE(103, ga_entry::t21sh, gelu_alt::gelu_t21s_init)
GA_SPECIALIZE(104, ga_entry::magb, gelu_alt::gelu_magb_init)

#undef GA_SPECIALIZE

}  // namespace compute_kernel_lib
