// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::pool {

// Mirror of PyTorch behavior: there is no dedicated global_avg_pool2d op, it always routes
// through pool2d/adaptive_avg_pool2d. We delegate to avg_pool2d, whose pool2d() entry point
// detects the global-pool case (kernel == input spatial, no padding/dilation) and runs a
// single pool_sum reduction instead of the sliding-window kernels.
Tensor global_avg_pool2d(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");

    const auto& logical_shape = input.logical_shape();
    const auto& padded_shape = input.padded_shape();
    uint32_t rank = logical_shape.rank();

    Tensor input_4d = input;
    if (rank == 3) {
        log_debug(
            tt::LogOp,
            "GlobalAvgPool2D: Rank-3 input tensor detected, assuming [H, W, C] format and reshaping to [1, H, W, C]");
        ttnn::Shape reshaped_logical({1, logical_shape[0], logical_shape[1], logical_shape[2]});
        ttnn::Shape reshaped_padded({1, padded_shape[0], padded_shape[1], padded_shape[2]});
        input_4d = ttnn::reshape(input, reshaped_logical, reshaped_padded);
    } else if (rank == 2) {
        log_debug(
            tt::LogOp,
            "GlobalAvgPool2D: Rank-2 input tensor detected, assuming [H, W] format and reshaping to [1, H, W, 1]");
        ttnn::Shape reshaped_logical({1, logical_shape[0], logical_shape[1], 1});
        ttnn::Shape reshaped_padded({1, padded_shape[0], padded_shape[1], 1});
        input_4d = ttnn::reshape(input, reshaped_logical, reshaped_padded);
    } else if (rank != 4) {
        TT_THROW("Input tensor must be rank 2, 3, or 4, got rank {}", rank);
    }

    const auto& in_4d_logical = input_4d.logical_shape();
    const auto& in_4d_padded = input_4d.padded_shape();
    uint32_t N = in_4d_logical[0];
    uint32_t H = in_4d_logical[1];
    uint32_t W = in_4d_logical[2];
    uint32_t C = in_4d_logical[3];

    // avg_pool2d expects flat (1, 1, N*H*W, C) input. For 4D NHWC tensors only the last two
    // dims pad in TILE layout, so padded_shape[0..2] match logical N, H, W and the flat reshape
    // is metadata-only.
    ttnn::Shape flat_logical({1, 1, N * H * W, C});
    ttnn::Shape flat_padded({1, 1, in_4d_padded[0] * in_4d_padded[1] * in_4d_padded[2], in_4d_padded[3]});
    Tensor flat_input = ttnn::reshape(input_4d, flat_logical, flat_padded);

    auto memory_config = memory_config_arg.value_or(input.memory_config());
    DataType out_dtype = output_dtype.value_or(DataType::BFLOAT16);

    Tensor output = ttnn::operations::pool::avg_pool2d(
        flat_input,
        N,
        H,
        W,
        C,
        std::array<uint32_t, 2>{H, W},
        std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/true,
        /*divisor_override=*/std::nullopt,
        memory_config,
        /*dram_slice_config=*/std::nullopt,
        /*applied_shard_scheme=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*deallocate_input=*/false,
        /*reallocate_halo_output=*/true,
        out_dtype,
        Layout::TILE,  // match the legacy global_avg_pool2d's output layout
        /*config_tensor_in_dram=*/false);

    // avg_pool2d returns flat NHW format (1, 1, N*out_H*out_W, C) = (1, 1, N, C). Reshape to
    // (N, 1, 1, C) — the historical global_avg_pool2d output shape — for backward compatibility.
    const auto& out_padded = output.padded_shape();
    ttnn::Shape final_logical({N, 1, 1, C});
    ttnn::Shape final_padded({N, 1, 1, out_padded[3]});
    output = ttnn::reshape(output, final_logical, final_padded);

    return output;
}

}  // namespace ttnn::operations::pool
