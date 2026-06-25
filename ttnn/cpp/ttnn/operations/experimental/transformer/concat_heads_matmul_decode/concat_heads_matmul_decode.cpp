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

    // --- Reshard the viewed activation to WIDTH_SHARDED input-A over reshard_cores cores.
    // matmul_decode(partial_width_sharded=true) HARD-REQUIRES a width-sharded input A
    // (matmul_decode_device_operation.cpp validate). Shard shape is [seq, K/reshard_cores],
    // ROW_MAJOR, L1 -- equivalent to the test's `o_in_mc`.
    CoreRangeSet a_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
        reshard_cores, attn.device()->compute_with_storage_grid_size(), /*row_wise=*/true);
    std::array<uint32_t, 2> a_shard_shape = {seq, shard_w};
    ShardSpec a_shard_spec(a_core_range_set, a_shard_shape, ShardOrientation::ROW_MAJOR);
    MemoryConfig a_width_sharded(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, a_shard_spec);

    Tensor in0 = ttnn::to_memory_config(viewed, a_width_sharded);

    // --- O-projection via matmul_decode: partial-width-sharded resident-L1 B, interleaved L1 out.
    Tensor out = ttnn::prim::matmul_decode(
        in0,
        weight,
        /*partial_width_sharded=*/true,
        output_dtype.has_value() ? std::optional<const tt::tt_metal::DataType>(*output_dtype)
                                 : std::optional<const tt::tt_metal::DataType>(tt::tt_metal::DataType::BFLOAT16),
        compute_kernel_config,
        /*fused_gelu=*/false,
        /*interleaved_output=*/true,
        /*fused_gelu_approx=*/false);

    in0.deallocate();
    return out;
}

}  // namespace ttnn::experimental
