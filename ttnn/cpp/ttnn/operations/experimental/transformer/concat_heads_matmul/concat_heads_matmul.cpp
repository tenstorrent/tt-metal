// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/concat_heads_matmul/concat_heads_matmul.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::experimental {

ttnn::Tensor concat_heads_matmul(
    const Tensor& attn,
    const Tensor& weight,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config) {
    using namespace tt::constants;

    TT_FATAL(attn.storage_type() == StorageType::DEVICE, "attn must be on device");
    TT_FATAL(attn.padded_shape().rank() == 4, "attn must be rank-4 [1, nh, seq, hd]");
    TT_FATAL(
        attn.padded_shape()[2] == TILE_HEIGHT,
        "concat_heads_matmul requires seq <= one tile; got padded seq {}",
        attn.padded_shape()[2]);

    // Free concat-heads: for seq <= 1 tile, concat-heads is exactly attn's contiguous tile order, so
    // reinterpreting its buffer as [1, 1, seq, nh * hd] is a metadata-only view (no device op). The
    // O-projection is then a single ttnn::matmul dispatch with the caller's program config; the kernel
    // config is ttnn::matmul's own default unless one is given, so this matches the unfused O-proj.
    const uint32_t seq = attn.padded_shape()[2];
    const uint32_t K = attn.padded_shape()[1] * attn.padded_shape()[3];  // nh * hd
    ttnn::Shape in0_shape({1, 1, seq, K});
    Tensor in0 = tt::tt_metal::view(attn, in0_shape, in0_shape);

    return ttnn::matmul(
        in0,
        weight,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        memory_config.value_or(attn.memory_config()),
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        program_config,
        /*activation=*/std::nullopt,
        compute_kernel_config);
}

}  // namespace ttnn::experimental
