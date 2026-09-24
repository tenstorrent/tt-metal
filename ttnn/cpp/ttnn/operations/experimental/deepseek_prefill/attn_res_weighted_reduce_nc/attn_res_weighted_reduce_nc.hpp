// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc {

// out[r][0][h][w] = sum_c input[0][c][h][w] * weight[r][c][h][0]
//
// A reduction over `dim` fused with the per-row weighting that would otherwise
// need a separate `mul` and its full-size intermediate. The product is MAC'd
// into the accumulator, so the tensor is read once instead of three times.
//
// `weight` carries one scalar per (r, c, h) and is broadcast along the last dim,
// which is what `BroadcastType::COL` does natively — no transpose, and no
// padding beyond the tile width the layout already imposes.
//
// The weight's dim 0 batches the output: R weight sets reduce the same input
// into R planes. A caller holding R sets gets them in one dispatch, and the
// input costs one read per group of sets rather than one per set — which is the
// difference between the op being bound by its own arithmetic and being bound by
// re-reading a tensor it has already seen.
//
// Only `dim == 1` is implemented. See the device op's validation.
//
// Not `fast_reduce_nc` with a weight operand, despite the neighbouring name. That op
// sums into one output plane and splits its work over output tiles, because each output
// tile owns a private set of input tiles; this one produces R planes from a single pass
// and splits over input positions, because every plane reads the same input and reading
// it once is the whole point. The two therefore share no work-splitting strategy and no
// output shape contract, which is why the weighting is not a flag on that op. The
// constraints do not transfer either: this op is Blackhole-only and takes a bf16 input,
// where `fast_reduce_nc` is general across architectures.
ttnn::Tensor attn_res_weighted_reduce_nc(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc
