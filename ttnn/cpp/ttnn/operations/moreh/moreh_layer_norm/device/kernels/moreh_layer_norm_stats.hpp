// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "api/compute/bcast.h"
#include "api/compute/mask.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

constexpr bool kFp32Dest = DST_ACCUM_MODE;

constexpr uint32_t kA1 = 0;
constexpr uint32_t kScr = kFp32Dest ? 2u : 4u;
constexpr uint32_t kGroupTiles = kScr;

constexpr uint32_t ln_isqrt(uint32_t v) {
    uint32_t r = 0;
    while ((r + 1) * (r + 1) <= v) {
        ++r;
    }
    return r;
}
constexpr uint32_t ln_level_tiles(uint32_t n) {
    const uint32_t b = (ln_isqrt(n) + kGroupTiles - 1) / kGroupTiles * kGroupTiles;
    return b < kGroupTiles ? kGroupTiles : b;
}

template <bool kLastDim>
ALWI void ln_sub_bcast_init(uint32_t cb_a, uint32_t cb_b) {
    constexpr auto bt = kLastDim ? ckernel::BroadcastType::COL : ckernel::BroadcastType::SCALAR;
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWSUB, bt, MathFidelity::LoFi>(cb_a, cb_b, false)));
    UNPACK((llk_unpack_AB_init<bt>(cb_a, cb_b)));
}

template <bool kLastDim>
ALWI void ln_sub_bcast(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t dst) {
    constexpr auto bt = kLastDim ? ckernel::BroadcastType::COL : ckernel::BroadcastType::SCALAR;
    sub_tiles_bcast<bt>(cb_a, cb_b, tile_a, tile_b, dst);
}

ALWI void ln_mask(uint32_t cb_mask, uint32_t slot) {
#if defined(FP32_DEST_ACC_EN)
    reconfig_data_format_srca(cb_mask);
#endif
    copy_init(cb_mask);
    copy_tile(cb_mask, 0, slot + 1);
    mask_tile_init();
    mask_tile(slot, slot + 1);
}

ALWI void ln_acc(uint32_t acc, uint32_t src) {
    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(acc, src, acc);
}

template <bool kLastDim>
__attribute__((noinline)) void ln_group(
    uint32_t cb_x, uint32_t x_tile, uint32_t n, uint32_t cb_r, bool mh, uint32_t cb_mh, bool mw, uint32_t cb_mw) {
    ln_sub_bcast_init<kLastDim>(cb_x, cb_r);
    for (uint32_t j = 0; j < n; ++j) {
        ln_sub_bcast<kLastDim>(cb_x, cb_r, x_tile + j, 0, kScr + j);
    }
    if (mh) {
        ln_mask(cb_mh, kScr);
    }
    if (mw) {
        ln_mask(cb_mw, kScr);
    }
    add_binary_tile_init();
    for (uint32_t j = 0; j < n; ++j) {
        ln_acc(kA1, kScr + j);
    }
}

constexpr uint32_t kOuterBf16 = 1;
constexpr uint32_t kSumSlot = kFp32Dest ? kA1 : kOuterBf16;

template <bool kLastDim>
ALWI void ln_sub_init_math(uint32_t cb_a, uint32_t cb_b, bool acc) {
    constexpr auto bt = kLastDim ? ckernel::BroadcastType::COL : ckernel::BroadcastType::SCALAR;
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWSUB, bt, MathFidelity::LoFi>(cb_a, cb_b, acc)));
}
template <bool kLastDim>
ALWI void ln_sub_init_unpack(uint32_t cb_a, uint32_t cb_b) {
    constexpr auto bt = kLastDim ? ckernel::BroadcastType::COL : ckernel::BroadcastType::SCALAR;
    UNPACK((llk_unpack_AB_init<bt>(cb_a, cb_b)));
}
ALWI void ln_zero_slot(uint32_t s) {
    fill_tile_init();
    fill_tile(s, 0.0f);
}

template <bool kLastDim>
struct LnFpuSum {
    uint32_t cb_x;
    uint32_t cb_r;
    uint32_t cb_mh;
    uint32_t cb_mw;
    uint32_t level_tiles;
    uint32_t count[4];
    uint32_t mode;
    uint32_t outer_live;

    __attribute__((noinline)) void set_mode(uint32_t m) {
        if (mode == 0) {
            ln_sub_init_unpack<kLastDim>(cb_x, cb_r);
        }
        ln_sub_init_math<kLastDim>(cb_x, cb_r, m == 2);
        mode = m;
    }
    __attribute__((noinline)) void mask_bin(uint32_t b) {
        const uint32_t s = 2 * b;
        if (b & 2u) {
            ln_mask(cb_mh, s);
        }
        if (b & 1u) {
            ln_mask(cb_mw, s);
        }
        mode = 0;
    }
    __attribute__((noinline)) void flush_bf16(uint32_t b) {
        mask_bin(b);
        if (!outer_live) {
            ln_zero_slot(kOuterBf16);
            outer_live = 1;
        }

        add_binary_tile_init();
        ln_acc(kOuterBf16, 2 * b);
        count[b] = 0;
    }
    __attribute__((noinline)) void add(uint32_t x_tile, bool mh, bool mw) {
        if constexpr (kFp32Dest) {
            if (!mh && !mw) {
                if (mode != 2) {
                    set_mode(2);
                }
                ln_sub_bcast<kLastDim>(cb_x, cb_r, x_tile, 0, kA1);
            } else {
                ln_group<kLastDim>(cb_x, x_tile, 1, cb_r, mh, cb_mh, mw, cb_mw);
                mode = 0;
            }
        } else {
            const uint32_t b = (mh ? 2u : 0u) | (mw ? 1u : 0u);
            const uint32_t m = count[b] == 0 ? 1u : 2u;
            if (mode != m) {
                set_mode(m);
            }
            ln_sub_bcast<kLastDim>(cb_x, cb_r, x_tile, 0, 2 * b);
            if (++count[b] == level_tiles) {
                flush_bf16(b);
            }
        }
    }
    uint32_t finish() {
        if constexpr (kFp32Dest) {
            return kA1;
        } else {
            if (!outer_live) {
                uint32_t first = 4;
                for (uint32_t b = 0; b < 4; ++b) {
                    if (count[b] == 0) {
                        continue;
                    }
                    mask_bin(b);
                    if (first == 4) {
                        first = b;
                    } else {
                        add_binary_tile_init();
                        add_binary_tile(2 * first, 2 * b, 2 * first);
                    }
                }
                return 2 * first;
            }

            for (uint32_t b = 0; b < 4; ++b) {
                if (count[b] != 0) {
                    flush_bf16(b);
                }
            }
            return kOuterBf16;
        }
    }
};

template <bool kLastDim>
__attribute__((noinline)) void ln_xmm_tiles(
    uint32_t cb_x, uint32_t x_tile, uint32_t n, uint32_t cb_r, uint32_t cb_zero, uint32_t cb_e) {
#if defined(FP32_DEST_ACC_EN)
    reconfig_data_format(cb_x, cb_r);
#endif
    ln_sub_bcast_init<kLastDim>(cb_x, cb_r);
    for (uint32_t j = 0; j < n; ++j) {
        ln_sub_bcast<kLastDim>(cb_x, cb_r, x_tile + j, 0, j);
    }
#if defined(FP32_DEST_ACC_EN)
    reconfig_data_format(cb_zero, cb_e);
#endif
    ln_sub_init_unpack<kLastDim>(cb_zero, cb_e);
    ln_sub_init_math<kLastDim>(cb_zero, cb_e, true);
    for (uint32_t j = 0; j < n; ++j) {
        ln_sub_bcast<kLastDim>(cb_zero, cb_e, 0, 0, j);
    }
}
