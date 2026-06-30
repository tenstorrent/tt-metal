// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::experimental::matmul_decode {

MatmulDecodeDeviceOperation::program_factory_t MatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
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

    if (operation_attributes.partial_width_sharded) {
        // Partial width-sharded B: the 2D (K x N) block-sharding geometry is recovered and
        // validated in PartialWidthSharded::create_descriptor.
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
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    // Output shape is the LHS shape with the last dim replaced by N. We use the
    // operation attribute N (rather than B's last logical dim) so that the partial
    // width-sharded layout -- whose reshaped/permuted B has a different logical
    // shape -- still produces a correct [..., M, N] output.
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());

    CoreRangeSet output_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;
    int output_num_cores = output_core_range_set.num_cores();
    if (operation_attributes.partial_width_sharded) {
        // The partial layout reduces across K_blocks cores, so the output is sharded
        // across N_blocks cores (one N-slice per block). Mirror the factory: each B
        // shard is [Kc, Nc], so the cores spanning N is N_tiles / Nc_tiles and that
        // equals N_blocks.
        const auto& b_shard_spec = input_tensor_b.memory_config().shard_spec().value();
        const int N_tiles = tt::div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const int Nc_tiles = static_cast<int>(b_shard_spec.shape[1]) / tt::constants::TILE_WIDTH;
        const int N_blocks = N_tiles / Nc_tiles;
        output_num_cores = N_blocks;
        output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
            output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
    }
    int per_core_output_width = tt::div_up(operation_attributes.N, output_num_cores);
    std::array<uint32_t, 2> shard_shape = {operation_attributes.M, per_core_output_width};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input_tensor_a.tensor_spec().tile()),
            memory_config));
}

MatmulDecodeDeviceOperation::tensor_return_value_t MatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::experimental::matmul_decode

namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype) {
    using OperationType = ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation;

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
    log_debug(
        tt::LogOp, "matmul_decode partial_width_sharded={} with M={}, N={}, K={}", partial_width_sharded, M, N, K);
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
