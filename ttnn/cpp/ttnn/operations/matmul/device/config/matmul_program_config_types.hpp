// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N);
    }
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool transpose_mcast{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool fuse_batch = true;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "per_core_M",
        "per_core_N",
        "transpose_mcast",
        "fused_activation",
        "fuse_batch");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            transpose_mcast,
            fused_activation,
            fuse_batch);
    }
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool fuse_batch{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool mcast_in0{};
    bool gather_in0{};
    CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers{};
    bool untilize_out{};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "per_core_M",
        "per_core_N",
        "fuse_batch",
        "fused_activation",
        "mcast_in0",
        "gather_in0",
        "hop_cores",
        "num_global_cb_receivers",
        "untilize_out");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fuse_batch,
            fused_activation,
            mcast_in0,
            gather_in0,
            hop_cores,
            num_global_cb_receivers,
            untilize_out);
    }
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;

    static constexpr auto attribute_names =
        std::forward_as_tuple("in0_block_w", "per_core_M", "per_core_N", "fused_activation");
    auto attribute_values() const {
        return std::forward_as_tuple(in0_block_w, per_core_M, per_core_N, fused_activation);
    }
};

struct MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;

    static constexpr auto attribute_names =
        std::forward_as_tuple("in0_block_w", "per_core_M", "per_core_N", "fused_activation");
    auto attribute_values() const {
        return std::forward_as_tuple(in0_block_w, per_core_M, per_core_N, fused_activation);
    }
};

struct MatmulMultiCoreProgramConfig {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const { return std::make_tuple(); }
};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>;

}  // namespace ttnn::operations::matmul
