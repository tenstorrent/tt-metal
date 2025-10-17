// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file welford_combine_compute.h
 * @brief Combines sets of Welford partial mean and variance
 *        results using the compute API (invoked from compute kernels)
 */

#pragma once

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/reg_api.h"

namespace norm::kernel_util::compute {
namespace detail {
/**
 * @brief C++17 compatible bit_cast replacement using union
 * @tparam To The type to cast to
 * @tparam From The type to cast from
 * @param from The value to cast from
 * @return The casted value
 */
template <typename To, typename From>
inline To bit_cast(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Types must have same size");
    static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");

    union {
        From f;
        To t;
    } u;

    u.f = from;
    return u.t;
}
}  // namespace detail

/**
 * @brief Policy used to optionally compute 1/sqrt(Var[x] + eps)
 *        when combining Welford partial results
 */
struct RSqrtPolicy {
    bool compute = false;  // Specifies if the computation should be performed
    uint32_t eps = 0;      // If compute is true, this is the epsilon value
                           // to add to the variance
};

/**
 * @brief Combine partial mean and variance results for
 *        multiple sets into a single mean and variance
 * @param cb_partials CB containing interleaved [mean, variance] tiles for
 *                    all sets to be combined. This CB will be consumed
 * @param cb_combined CB to store the combined mean and variance results(size 2)
 * @param num_sets Number of sets to combine
 * @param rsqrt_policy Policy to specify if 1/sqrt(Var[x] + eps) should be computed
 *                     and pushed to the combined CB instead of variance
 * @tparam NextSetSizeFn Functor that takes the set index and returns the
 *                       size of the next set
 */
template <typename NextSetSizeFn>
inline void combine_welford_partials(
    uint32_t cb_partials,
    uint32_t cb_combined,
    uint32_t num_sets,
    NextSetSizeFn&& next_set_size_fn,
    RSqrtPolicy rsqrt_policy = RSqrtPolicy{}) {
    // Accumulate mean and M2 in dst regs.
    // Use 2 extra dst regs to help with the math
    constexpr uint32_t tmp_dst0 = 0;
    constexpr uint32_t tmp_dst1 = 1;
    constexpr uint32_t mean_acc_dst = 2;
    constexpr uint32_t m2_acc_dst = 3;

    // Work with two tiles (1 mean and 1 var) at a time
    constexpr uint32_t mean_cb_idx = 0;
    constexpr uint32_t var_cb_idx = 1;

    tile_regs_acquire();

    // Make sure that the registers are zeroed out
    fill_tile_init();
    fill_tile(tmp_dst0, 0.f);
    fill_tile(tmp_dst1, 0.f);
    fill_tile(mean_acc_dst, 0.f);
    fill_tile(m2_acc_dst, 0.f);

    uint32_t acc_n = 0;
    for (uint32_t b = 0; b < num_sets; b++) {
        // Wait for 1 mean tile and 1 var tile
        cb_wait_front(cb_partials, 2);

        const float n_a = acc_n;
        const float n_b = next_set_size_fn(b);

        const float n_a_norm = static_cast<float>(n_a) / (n_a + n_b);
        const float n_b_norm = static_cast<float>(n_b) / (n_a + n_b);

        // Copy x_b to dst1
        copy_tile_to_dst_init_short(cb_partials);
        copy_tile(cb_partials, mean_cb_idx, tmp_dst1);

        // Compute delta = x_b - x_a, store in dst0
        sub_binary_tile_init();
        sub_binary_tile(tmp_dst1, mean_acc_dst, tmp_dst0);

        // Multiply accumulated mean by n_a_norm
        binop_with_scalar_tile_init();
        mul_unary_tile(mean_acc_dst, detail::bit_cast<uint32_t>(n_a_norm));

        // Multiply x_b by n_b_norm
        binop_with_scalar_tile_init();
        mul_unary_tile(tmp_dst1, detail::bit_cast<uint32_t>(n_b_norm));

        // Accumulate n_b_norm * x_b into mean
        add_binary_tile_init();
        add_binary_tile(mean_acc_dst, tmp_dst1, mean_acc_dst);

        // Square delta
        square_tile_init();
        square_tile(tmp_dst0);

        // Multiply delta^2 by n_a * n_b_norm
        binop_with_scalar_tile_init();
        mul_unary_tile(tmp_dst0, detail::bit_cast<uint32_t>(n_a * n_b_norm));

        // Accumulate into M2
        add_binary_tile_init();
        add_binary_tile(m2_acc_dst, tmp_dst0, m2_acc_dst);

        // Copy var_b into dst0
        copy_tile_to_dst_init_short(cb_partials);
        copy_tile(cb_partials, var_cb_idx, tmp_dst0);

        // Multiply var_b by n_b to get M2_b, store in dst0
        binop_with_scalar_tile_init();
        mul_unary_tile(tmp_dst0, detail::bit_cast<uint32_t>(n_b));

        // Accumulate into M2
        add_binary_tile_init();
        add_binary_tile(m2_acc_dst, tmp_dst0, m2_acc_dst);

        acc_n += n_b;
        cb_pop_front(cb_partials, 2);
    }

    // Convert final M2 to var
    binop_with_scalar_tile_init();
    mul_unary_tile(m2_acc_dst, detail::bit_cast<uint32_t>(1.f / acc_n));

    // Compute 1/sqrt(Var[x] + eps) if enabled
    if (rsqrt_policy.compute) {
        // Add eps to var
        binop_with_scalar_tile_init();
        add_unary_tile(m2_acc_dst, rsqrt_policy.eps);

        // Compute 1/sqrt(Var[x] + eps)
        rsqrt_tile_init();
        rsqrt_tile(m2_acc_dst);
    }

    // Pack the results into the combined CB
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(mean_acc_dst, cb_combined);
    pack_tile(m2_acc_dst, cb_combined);
    tile_regs_release();
}

}  // namespace norm::kernel_util::compute
