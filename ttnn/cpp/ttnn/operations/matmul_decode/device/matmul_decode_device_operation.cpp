// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::matmul_decode {

namespace {

// Dimensions recovered from a "partial width-sharded" B tensor.
//
// In this layout the caller has already reshaped + permuted B so that a [K, N]
// weight becomes a width-sharded tensor whose shard shape is [Kc, Nc] across
// K_blocks * N_blocks cores, where Kc = K / K_blocks and Nc = N / N_blocks.
// Because the reshape folds the K-block dimension into the (width-shardable) last
// dimension, the partial-sharded B's *logical* shape no longer matches [K, N];
// instead its height is Kc and its width is K_blocks * N_blocks * Nc. We therefore
// recover the real matmul dims from the shard spec plus A's K dimension.
struct PartialDims {
    int K_blocks;
    int N_blocks;
    int Kc;  // shard height (== K / K_blocks)
    int Nc;  // shard width  (== N / N_blocks)
    int N;   // recovered output width (== N_blocks * Nc)
};

// std::optional<PartialDims> detect_partial_width_sharded(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
//     const auto& b_mem = input_tensor_b.memory_config();
//     if (b_mem.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
//         return std::nullopt;
//     }
//     const auto& shard_spec = b_mem.shard_spec();
//     if (!shard_spec.has_value()) {
//         return std::nullopt;
//     }
//     const int K = input_tensor_a.logical_shape()[-1];
//     const int Kc = static_cast<int>(shard_spec->shape[0]);
//     const int Nc = static_cast<int>(shard_spec->shape[1]);
//     // Full width-sharded keeps the entire K dimension per shard (Kc == K); only treat
//     // B as partial when its shard height is a strict, even divisor of K.
//     if (Kc <= 0 || Kc >= K || K % Kc != 0) {
//         return std::nullopt;
//     }
//     const int K_blocks = K / Kc;
//     const int num_cores = static_cast<int>(shard_spec->grid.num_cores());
//     if (K_blocks <= 0 || num_cores % K_blocks != 0) {
//         return std::nullopt;
//     }
//     const int N_blocks = num_cores / K_blocks;
//     return PartialDims{K_blocks, N_blocks, Kc, Nc, N_blocks * Nc};
// }

}  // namespace

MatmulDecodeDeviceOperation::program_factory_t MatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // The flag explicitly requests the partial factory; otherwise fall back to detecting
    // the partial layout from the inputs (B sharded along both K and N).
    if (operation_attributes.partial_width_sharded) {
        return PartialWidthSharded{};
    }
    return FullWidthSharded{};
}

void MatmulDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input tensor A must be in tile layout");
    TT_FATAL(input_tensor_b.layout() == Layout::TILE, "Input tensor B must be in tile layout");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor A must be in width sharded memory layout, but got {}",
        input_tensor_a.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor B must be in width sharded memory layout, but got {}",
        input_tensor_b.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_a.logical_shape()[-1] == operation_attributes.K,
        "Input tensor A must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_a.logical_shape()[-2] == operation_attributes.M,
        "Input tensor A must have the same M dimension as the operation attributes");

    // const auto partial = detect_partial_width_sharded(input_tensor_a, input_tensor_b);
    if (operation_attributes.partial_width_sharded) {
        // Partial width-sharded B: validate the recovered 2D (K x N) block sharding.
        // TT_FATAL(
        //     operation_attributes.M <= tt::constants::TILE_HEIGHT,
        //     "partial_width_sharded matmul_decode currently supports a single M tile (decode), but got M={}",
        //     operation_attributes.M);
        // TT_FATAL(
        //     partial->K_blocks % 2 == 0,
        //     "partial_width_sharded matmul_decode requires an even number of K-blocks (cross-core reduction is done "
        //     "pairwise), but got K_blocks={}",
        //     partial->K_blocks);
        // TT_FATAL(
        //     partial->Kc % tt::constants::TILE_WIDTH == 0 && partial->Nc % tt::constants::TILE_WIDTH == 0,
        //     "partial_width_sharded matmul_decode requires B shard dims [{}, {}] to be tile-aligned",
        //     partial->Kc,
        //     partial->Nc);
        // TT_FATAL(
        //     partial->N == operation_attributes.N,
        //     "partial_width_sharded matmul_decode recovered N={} does not match operation attribute N={}",
        //     partial->N,
        //     operation_attributes.N);
        return;
    }

    // Full width-sharded B: each shard holds the full K dimension for its N-slice.
    if (input_tensor_a.logical_shape().rank() > 2) {
        for (int i = 0; i < input_tensor_a.logical_shape().rank() - 2; i++) {
            TT_FATAL(
                input_tensor_a.logical_shape()[i] == input_tensor_b.logical_shape()[i],
                "Input tensor A and B must have the same shape for all dimensions except the last two, but got {} and "
                "{}",
                input_tensor_a.logical_shape(),
                input_tensor_b.logical_shape());
        }
    }
    TT_FATAL(
        input_tensor_b.logical_shape()[-2] == operation_attributes.K,
        "Input tensor B must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_b.logical_shape()[-1] == operation_attributes.N,
        "Input tensor B must have the same N dimension as the operation attributes");
}

MatmulDecodeDeviceOperation::spec_return_value_t MatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;

    // Output shape is the LHS shape with the last dim replaced by N. We use the
    // operation attribute N (rather than B's last logical dim) so that the partial
    // width-sharded layout -- whose reshaped/permuted B has a different logical
    // shape -- still produces a correct [..., M, N] output.
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());
    int output_num_cores = tt::div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
    CoreRangeSet output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
        output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size());
    int per_core_output_width = tt::div_up(operation_attributes.N, output_num_cores);
    std::array<uint32_t, 2> shard_shape = {operation_attributes.M, per_core_output_width};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);
    log_info(tt::LogOp, "matmul_decode with output memory_config: {}", memory_config);
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

MatmulDecodeDeviceOperation::tensor_return_value_t MatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::matmul_decode

namespace ttnn::prim {
ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype) {
    using OperationType = ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation;

    // For the partial width-sharded layout the caller reshapes/permutes B, so its
    // last logical dim is no longer N; recover N from the shard spec in that case.
    int M, N, K;
    if (partial_width_sharded) {
        M = input_tensor_a.logical_shape()[-2];
        int K_a = input_tensor_a.logical_shape()[-1];
        int K_b = input_tensor_b.logical_shape()[-2];
        N = input_tensor_b.logical_shape()[-1];
        if (K_a >= K_b) {
            TT_FATAL(K_a % K_b == 0, "K_a must be divisible by K_b");
            int K_ratio = K_a / K_b;
            N = N / K_ratio;
        }
        K = K_a;
    } else {
        M = input_tensor_a.logical_shape()[-2];
        N = input_tensor_b.logical_shape()[-1];
        K = input_tensor_a.logical_shape()[-1];
    }
    log_info(tt::LogOp, "matmul_decode partial_width_sharded={} with M={}, N={}, K={}", partial_width_sharded, M, N, K);
    auto operation_attributes = OperationType::operation_attributes_t{
        M,
        N,
        K,
        input_tensor_a.memory_config(),
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        partial_width_sharded,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
