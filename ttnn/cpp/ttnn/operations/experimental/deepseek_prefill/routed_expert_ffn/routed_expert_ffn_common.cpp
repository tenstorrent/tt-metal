// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"
#include "routed_expert_ffn_common.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
// TODO(nuked-op matmul): removed #include "ttnn/operations/matmul/matmul.hpp"
// TODO(nuked-op matmul): removed #include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace detail {

uint32_t best_in0_block_w(
    uint32_t K_tiles,
    uint32_t per_core_M,
    uint32_t per_core_N,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype,
    float l1_safety_margin) {
    // TODO(nuked-op matmul): the original body relied on the deleted
    // ttnn::operations::matmul::utilities helpers (get_max_l1_space,
    // estimate_interm_tile_size, get_estimated_size_of_cbs) to L1-budget the
    // in0_block_w. With the matmul op nuked, return a safe divisor (1) so this
    // compiles. Restore the real CB-budgeting search when the op is recreated.
    (void)K_tiles;
    (void)per_core_M;
    (void)per_core_N;
    (void)input_tensor_a;
    (void)input_tensor_b;
    (void)compute_kernel_config;
    (void)output_dtype;
    (void)l1_safety_margin;
    return 1;
}

ttnn::Tensor routed_expert_ffn_chunked(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    uint32_t chunk_M_tiles,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // x is (..., M, K). Split M into chunks of `chunk_M_tiles` tiles, run the
    // BH routed_expert_ffn per chunk, and write each chunk's output into a
    // narrow view of the pre-allocated output tensor. ttnn::narrow is
    // zero-copy: it returns a Tensor that shares the underlying buffer with an
    // offset, so both the input slice and the output target are device-op-free.
    // The last chunk may be smaller (handled by the BH path automatically).
    const auto& x_shape = x.padded_shape();
    const auto x_rank = x_shape.rank();
    TT_FATAL(x_rank >= 2, "routed_expert_ffn_chunked: x must have rank >= 2, got {}", x_rank);
    // For rank > 2, require all leading dims == 1 (we treat x as (M, K)).
    for (int i = 0; i < static_cast<int>(x_rank) - 2; ++i) {
        TT_FATAL(x_shape[i] == 1, "routed_expert_ffn_chunked: x leading dim {} must be 1, got {}", i, x_shape[i]);
    }
    const uint32_t M = x_shape[-2];
    const uint32_t N = down_proj.padded_shape()[-1];
    const uint32_t M_tiles = M / ttnn::TILE_SIZE;
    const uint32_t chunk_M = chunk_M_tiles * ttnn::TILE_SIZE;
    const uint32_t num_chunks = tt::div_up(M_tiles, chunk_M_tiles);
    const int32_t narrow_dim = static_cast<int32_t>(x_rank) - 2;

    // Allocate the full output tensor once (if not supplied). All chunks write
    // into views of this buffer. Preserve x's rank — leading dims are all 1.
    ttnn::Tensor full_output;
    if (output.has_value()) {
        TT_FATAL(
            output->padded_shape()[-2] == M && output->padded_shape()[-1] == N,
            "routed_expert_ffn_chunked: supplied output tensor shape {} does not match expected ({}, {})",
            output->padded_shape(),
            M,
            N);
        full_output = *output;
    } else {
        ttnn::SmallVector<uint32_t> out_dims(x_rank, 1u);
        out_dims[x_rank - 2] = M;
        out_dims[x_rank - 1] = N;
        full_output = ttnn::empty(
            ttnn::Shape(out_dims),
            x.dtype(),
            ttnn::TILE_LAYOUT,
            x.device(),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    }

    for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t begin_m = i * chunk_M;
        const uint32_t end_m = std::min(begin_m + chunk_M, M);
        const uint32_t this_len = end_m - begin_m;
        auto chunk_x = ttnn::narrow(
            /*input_tensor=*/x,
            /*narrow_dim=*/narrow_dim,
            /*narrow_start=*/static_cast<int32_t>(begin_m),
            /*length=*/this_len);
        auto chunk_out_view = ttnn::narrow(
            /*input_tensor=*/full_output,
            /*narrow_dim=*/narrow_dim,
            /*narrow_start=*/static_cast<int32_t>(begin_m),
            /*length=*/this_len);
        (void)routed_expert_ffn_bh(
            /*x=*/chunk_x,
            /*gate_proj=*/gate_proj,
            /*up_proj=*/up_proj,
            /*down_proj=*/down_proj,
            /*compute_kernel_config=*/compute_kernel_config,
            /*output=*/chunk_out_view);
    }

    return full_output;
}

ttnn::Tensor routed_expert_ffn_default(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    (void)gate_proj;
    (void)up_proj;
    (void)down_proj;
    (void)compute_kernel_config;
    (void)output;
    // TODO(nuked-op matmul): restore real call (was ttnn::matmul(x, gate_proj, ..., "silu"))
    auto gate_result = x;
    // TODO(nuked-op matmul): restore real call (was ttnn::matmul(x, up_proj, ...))
    auto up_result = x;

    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);

    // TODO(nuked-op matmul): restore real call (was ttnn::matmul(gate_result, down_proj, ...))
    return gate_result;
}

}  // namespace detail

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    const uint32_t M_tiles = x.padded_shape()[-2] / ttnn::TILE_SIZE;
    const bool is_wormhole = x.device()->arch() == tt::ARCH::WORMHOLE_B0;

    // Fall back to default (auto-configured) matmuls only on Wormhole for very
    // large M. Blackhole has the unified chunked path below which handles any M.
    if (is_wormhole && M_tiles > 64) {
        return detail::routed_expert_ffn_default(
            /*x=*/x,
            /*gate_proj=*/gate_proj,
            /*up_proj=*/up_proj,
            /*down_proj=*/down_proj,
            /*compute_kernel_config=*/compute_kernel_config,
            /*output=*/std::move(output));
    }

    TT_FATAL(
        x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::BFLOAT8_B,
        "routed_expert_ffn: x must be BFLOAT16 or BFLOAT8_B, got {}",
        x.dtype());

    if (is_wormhole) {
        return detail::routed_expert_ffn_wh(x, gate_proj, up_proj, down_proj, compute_kernel_config, output);
    }

    // Blackhole: the optimized BH path is L1-bound at per-core scratch +
    // partials. Per-core L1 for DeepSeek V3 dims (emb=7168, hidden=2048)
    // holds up to M_tiles=128. We chunk at 64 so the same JIT-cached
    // program-config services M sizes from 2k tokens up to the input cap:
    // M_tiles<=64 runs in one chunk, larger M splits into ceil(M/64) chunks
    // that each reuse the same routed_expert_ffn_bh program.
    constexpr uint32_t MAX_CHUNK_M_TILES = 64;
    if (M_tiles <= MAX_CHUNK_M_TILES) {
        return detail::routed_expert_ffn_bh(
            x, gate_proj, up_proj, down_proj, compute_kernel_config, std::move(output));
    }

    return detail::routed_expert_ffn_chunked(
        x, gate_proj, up_proj, down_proj, MAX_CHUNK_M_TILES, compute_kernel_config, std::move(output));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
