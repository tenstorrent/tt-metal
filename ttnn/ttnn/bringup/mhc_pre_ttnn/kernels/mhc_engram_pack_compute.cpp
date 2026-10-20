// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing_pack engram compute (host: device/mhc_engram_pack_program_factory.cpp). Per unit (stream i, column
// block g, token tile-row r; mhc_engram_pack_common.hpp), one DEST window (full sync): per column tile c, h_i[r, c] ->
// DEST 3, k_i[r, c] -> DEST 4, the weight tile (the weight row on all 32 rows) -> DEST 5 (UnpackToDestFp32, exact);
// SFPU lane-wise fp32: DEST 0 += (h w) k (the product h w rounded, then one fused multiply-add), DEST 1 += h h,
// DEST 2 += k k (the first column assigns) -> cb_acc. Then, on the writer's coefficient-major copy of the three
// (cb_soa, lane = token row), slot 0 of each = the sum of its 32 column slots in order -> cb_soa_out.
// Root phase: per rooted row, its n G partial row tiles (cb_part, unit order) summed lane-wise in fp32 -> cb_fin.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "mhc_engram_pack_common.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t n_streams = get_compile_time_arg_val(0);
constexpr uint32_t ct = get_compile_time_arg_val(1);
constexpr uint32_t rt = get_compile_time_arg_val(3);
constexpr uint32_t bw = get_compile_time_arg_val(4);
constexpr uint32_t groups = get_compile_time_arg_val(5);
constexpr uint32_t cb_in = get_compile_time_arg_val(7);
constexpr uint32_t cb_w = get_compile_time_arg_val(8);
constexpr uint32_t cb_acc = get_compile_time_arg_val(9);
constexpr uint32_t cb_soa = get_compile_time_arg_val(10);
constexpr uint32_t cb_soa_out = get_compile_time_arg_val(11);
constexpr uint32_t cb_part = get_compile_time_arg_val(12);
constexpr uint32_t cb_fin = get_compile_time_arg_val(13);

#ifdef TRISC_MATH
namespace engram_sfpu {

using namespace sfpi;
constexpr int T = 32;

template <bool FIRST>
void accumulate() {
#pragma GCC unroll 8
    for (int v = 0; v < T; ++v) {
        const vFloat h = dst_reg[3 * T + v];
        const vFloat k = dst_reg[4 * T + v];
        const vFloat w = dst_reg[5 * T + v];
        const vFloat hw = h * w;
        if constexpr (FIRST) {
            dst_reg[v] = hw * k;
            dst_reg[T + v] = h * h;
            dst_reg[2 * T + v] = k * k;
        } else {
            dst_reg[v] = hw * k + vFloat(dst_reg[v]);
            dst_reg[T + v] = h * h + vFloat(dst_reg[T + v]);
            dst_reg[2 * T + v] = k * k + vFloat(dst_reg[2 * T + v]);
        }
    }
}

// DEST tiles 0..2: slot 0 <- the sum of slots 0..31 (column order), lane-wise.
void totals3() {
#pragma GCC unroll 1
    for (int t = 0; t < 3; ++t) {
        vFloat s = dst_reg[t * T];
#pragma GCC unroll 31
        for (int k = 1; k < T; ++k) {
            s = s + vFloat(dst_reg[t * T + k]);
        }
        dst_reg[t * T] = s;
    }
}

void add01() {
#pragma GCC unroll 8
    for (int v = 0; v < T; ++v) {
        dst_reg[v] = vFloat(dst_reg[v]) + vFloat(dst_reg[T + v]);
    }
}

}  // namespace engram_sfpu
#endif

using namespace ckernel;

ALWI void sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

#define ENGRAM_SFPU(fn)                                                        \
    do {                                                                       \
        sfpu_init();                                                           \
        MATH((_llk_math_eltwise_unary_sfpu_params_(fn, 0, VectorMode::None))); \
    } while (0)

void kernel_main() {
    constexpr uint32_t nu = n_streams * groups;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t n_rooted = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_acc);

    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_engram::Unit U = mhc_engram::unit_of(u, rt, groups, bw, ct);
        if (U.bu != cur) {
            if (cur != 0xFFFFFFFFu) {
                cb_pop_front(cb_w, bw);
            }
            cb_wait_front(cb_w, bw);
            cur = U.bu;
        }
        cb_wait_front(cb_in, 2 * bw);
        cb_reserve_back(cb_acc, 3);
        tile_regs_acquire();
        for (uint32_t c = 0; c < U.valid; ++c) {
            copy_init(cb_in);
            copy_tile(cb_in, 2 * c, 3);
            copy_tile(cb_in, 2 * c + 1, 4);
            copy_init(cb_w);
            copy_tile(cb_w, c, 5);
            if (c == 0) {
                ENGRAM_SFPU(engram_sfpu::accumulate<true>);
            } else {
                ENGRAM_SFPU(engram_sfpu::accumulate<false>);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < 3; ++t) {
            pack_tile(t, cb_acc);
        }
        tile_regs_release();
        cb_push_back(cb_acc, 3);
        cb_pop_front(cb_in, 2 * bw);

        cb_wait_front(cb_soa, 3);
        cb_reserve_back(cb_soa_out, 3);
        tile_regs_acquire();
        copy_init(cb_soa);
        for (uint32_t t = 0; t < 3; ++t) {
            copy_tile(cb_soa, t, t);
        }
        ENGRAM_SFPU(engram_sfpu::totals3);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < 3; ++t) {
            pack_tile(t, cb_soa_out);
        }
        tile_regs_release();
        cb_push_back(cb_soa_out, 3);
        cb_pop_front(cb_soa, 3);
    }
    if (cur != 0xFFFFFFFFu) {
        cb_pop_front(cb_w, bw);
    }

    for (uint32_t j = 0; j < n_rooted; ++j) {
        cb_reserve_back(cb_fin, 1);
        tile_regs_acquire();
        copy_init(cb_part);
        cb_wait_front(cb_part, nu);
        copy_tile(cb_part, 0, 0);
        for (uint32_t b = 1; b < nu; ++b) {
            copy_tile(cb_part, b, 1);
            ENGRAM_SFPU(engram_sfpu::add01);
        }
        cb_pop_front(cb_part, nu);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_fin);
        tile_regs_release();
        cb_push_back(cb_fin, 1);
    }
}
