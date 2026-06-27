// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/concat_heads_matmul_decode/concat_heads_matmul_decode.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/matmul_decode/device/matmul_decode_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor concat_heads_matmul_decode(
    const Tensor& attn,
    const Tensor& weight,
    std::optional<tt::tt_metal::DataType> output_dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t reshard_cores) {
    using namespace tt::constants;
    using tt::tt_metal::BufferType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::MemoryConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpec;
    using tt::tt_metal::StorageType;
    using tt::tt_metal::TensorMemoryLayout;

    TT_FATAL(attn.storage_type() == StorageType::DEVICE, "attn must be on device");
    TT_FATAL(weight.storage_type() == StorageType::DEVICE, "weight must be on device");
    TT_FATAL(attn.layout() == Layout::TILE, "attn must be tilized");
    TT_FATAL(attn.padded_shape().rank() == 4, "attn must be rank-4 [1, nh, seq, hd]");
    TT_FATAL(
        attn.padded_shape()[2] == TILE_HEIGHT,
        "concat_heads_matmul_decode requires seq <= one tile (Mt==1); got seq {}",
        attn.padded_shape()[2]);
    TT_FATAL(reshard_cores >= 1, "reshard_cores must be >= 1");

    const uint32_t seq = attn.padded_shape()[2];
    const uint32_t K = attn.padded_shape()[1] * attn.padded_shape()[3];  // nh * hd
    TT_FATAL(
        K % reshard_cores == 0,
        "concat_heads_matmul_decode: K ({}) must be divisible by reshard_cores ({})",
        K,
        reshard_cores);
    const uint32_t shard_w = K / reshard_cores;
    TT_FATAL(
        shard_w % TILE_WIDTH == 0,
        "concat_heads_matmul_decode: per-core shard width K/reshard_cores ({}) must be tile-aligned",
        shard_w);

    // --- FREE concat-heads: reinterpret attn's buffer as [1, 1, seq, K] (build-time-only view).
    // For seq <= 1 tile the concat-heads is exactly the contiguous tile order of attn, so this is
    // a pure metadata change -- NOT a traced op (same trick as concat_heads_matmul).
    ttnn::Shape in0_shape({1, 1, seq, K});
    Tensor viewed = tt::tt_metal::view(attn, in0_shape, in0_shape);

    // --- O-projection via matmul_decode with READER-SIDE input reshard. Pass the INTERLEAVED view
    // directly; matmul_decode (reshard_input=true) reshards it to WIDTH_SHARDED input-A over
    // reshard_cores cores inside its reader (overlapped with the matmul), so there is NO standalone
    // interleaved->sharded reshard op. The `shard_w` tile-alignment check above still applies
    // (reshard_input requires K/reshard_cores to be tile-aligned). Interleaved L1 out.
    Tensor out = ttnn::prim::matmul_decode(
        viewed,
        weight,
        /*partial_width_sharded=*/true,
        output_dtype.has_value() ? std::optional<const tt::tt_metal::DataType>(*output_dtype)
                                 : std::optional<const tt::tt_metal::DataType>(tt::tt_metal::DataType::BFLOAT16),
        compute_kernel_config,
        /*fused_gelu=*/false,
        /*interleaved_output=*/true,
        /*fused_gelu_approx=*/false,
        /*reshard_input=*/true,
        /*reshard_cores=*/reshard_cores);

    return out;
}

}  // namespace ttnn::experimental
