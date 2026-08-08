// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/tile_move_copy.h"

namespace ckernel {

namespace stable_logaddexp_detail {
constexpr uint32_t kStableLogaddexpMaxDstOffset = 4;
constexpr uint32_t kStableLogaddexpRhsDstOffset = 8;
constexpr uint32_t kInvLn2Bits = 0x3fb8aa3bu;

ALWI void copy_tile_to_dst_init_short_cross_format(uint32_t old_cb, uint32_t new_cb) {
#ifndef ARCH_QUASAR
    copy_tile_to_dst_init_short_with_dt(old_cb, new_cb);
#else
    copy_tile_to_dst_init_short(new_cb);
#endif
}
}  // namespace stable_logaddexp_detail

ALWI void stable_logaddexp_tile_init() {}

ALWI void stable_logaddexp2_tile_init() {}

ALWI void stable_logaddexp_tile(uint32_t lhs_dst, uint32_t rhs_dst, uint32_t out_dst) {
    const uint32_t max_dst = out_dst + stable_logaddexp_detail::kStableLogaddexpMaxDstOffset;

    binary_max_tile_init();
    binary_max_tile(lhs_dst, rhs_dst, max_dst);
    binary_min_tile_init();
    binary_min_tile(lhs_dst, rhs_dst, rhs_dst);
    sub_binary_tile_init();
    sub_binary_tile(rhs_dst, max_dst, rhs_dst);
    exp_tile_init<false>();
    exp_tile<false>(rhs_dst);
    log1p_tile_init<false>();
    log1p_tile<false>(rhs_dst);
    add_binary_tile_init();
    add_binary_tile(max_dst, rhs_dst, out_dst);
}

ALWI void stable_logaddexp2_tile(uint32_t lhs_dst, uint32_t rhs_dst, uint32_t out_dst) {
    const uint32_t max_dst = out_dst + stable_logaddexp_detail::kStableLogaddexpMaxDstOffset;

    binary_max_tile_init();
    binary_max_tile(lhs_dst, rhs_dst, max_dst);
    binary_min_tile_init();
    binary_min_tile(lhs_dst, rhs_dst, rhs_dst);
    sub_binary_tile_init();
    sub_binary_tile(rhs_dst, max_dst, rhs_dst);
    exp2_tile_init();
    exp2_tile(rhs_dst);
    log1p_tile_init<false>();
    log1p_tile<false>(rhs_dst);
    binop_with_scalar_tile_init();
    mul_unary_tile(rhs_dst, stable_logaddexp_detail::kInvLn2Bits);
    add_binary_tile_init();
    add_binary_tile(max_dst, rhs_dst, out_dst);
}

ALWI void stable_logaddexp_tiles(
    uint32_t lhs_cb, uint32_t rhs_cb, uint32_t lhs_tile, uint32_t rhs_tile, uint32_t out_dst) {
    const uint32_t rhs_dst = out_dst + stable_logaddexp_detail::kStableLogaddexpRhsDstOffset;

    copy_tile_to_dst_init_short(lhs_cb);
    copy_tile(lhs_cb, lhs_tile, out_dst);
    stable_logaddexp_detail::copy_tile_to_dst_init_short_cross_format(lhs_cb, rhs_cb);
    copy_tile(rhs_cb, rhs_tile, rhs_dst);

    stable_logaddexp_tile(out_dst, rhs_dst, out_dst);
}

ALWI void stable_logaddexp2_tiles(
    uint32_t lhs_cb, uint32_t rhs_cb, uint32_t lhs_tile, uint32_t rhs_tile, uint32_t out_dst) {
    const uint32_t rhs_dst = out_dst + stable_logaddexp_detail::kStableLogaddexpRhsDstOffset;

    copy_tile_to_dst_init_short(lhs_cb);
    copy_tile(lhs_cb, lhs_tile, out_dst);
    stable_logaddexp_detail::copy_tile_to_dst_init_short_cross_format(lhs_cb, rhs_cb);
    copy_tile(rhs_cb, rhs_tile, rhs_dst);

    stable_logaddexp2_tile(out_dst, rhs_dst, out_dst);
}

}  // namespace ckernel
