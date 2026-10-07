// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_large_k_device_operation.hpp"

#include "tt-metalium/constants.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::matmul_decode {

namespace {

// tt::tt_metal::NUM_SEMAPHORES, which no public header exposes to a ttnn op.
constexpr uint32_t kLargeKSemaphoresPerCore = 16;

// Product of all dims except the last two; the op only handles a single [M, K] x [K, N].
uint32_t large_k_leading_volume(const ttnn::Shape& shape) {
    uint32_t volume = 1;
    for (int i = 0; i < static_cast<int>(shape.rank()) - 2; ++i) {
        volume *= shape[i];
    }
    return volume;
}

// The root of the reduction tree and the only core that holds the output.
CoreRangeSet large_k_root_core(const Tensor& input_tensor_a) {
    const auto& grid = input_tensor_a.memory_config().shard_spec().value().grid;
    const CoreCoord root = tt::tt_metal::corerange_to_cores(grid, 1, /*row_wise=*/true).front();
    return CoreRangeSet(CoreRange(root, root));
}

}  // namespace

MatmulDecodeLargeKDeviceOperation::program_factory_t MatmulDecodeLargeKDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return TreeReduce{};
}

void MatmulDecodeLargeKDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const uint32_t M = operation_attributes.M;
    const uint32_t K = operation_attributes.K;
    const uint32_t N = operation_attributes.N;

    TT_FATAL(
        a.layout() == Layout::ROW_MAJOR, "matmul_decode_large_k requires input A in ROW_MAJOR, but got {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16, "matmul_decode_large_k requires input A in BFLOAT16, but got {}", a.dtype());
    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            a.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
        "matmul_decode_large_k requires input A WIDTH_SHARDED in L1, but got {}",
        a.memory_config());
    const auto& a_shard = a.memory_config().shard_spec().value();
    TT_FATAL(
        a_shard.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR,
        "matmul_decode_large_k requires input A's shards in ROW_MAJOR orientation (shard i is K-slice i)");
    TT_FATAL(
        large_k_leading_volume(a.logical_shape()) == 1,
        "matmul_decode_large_k requires input A's leading dims to be 1, but got shape {}",
        a.logical_shape());
    TT_FATAL(
        M >= 1 && M <= tt::constants::TILE_HEIGHT,
        "matmul_decode_large_k treats each row of A as a 1x32 tile, so M must be in [1, {}], but got {}",
        tt::constants::TILE_HEIGHT,
        M);
    const uint32_t num_cores = a_shard.grid.num_cores();
    const uint32_t Kc = a_shard.shape[1];
    TT_FATAL(
        a_shard.shape[0] == M,
        "matmul_decode_large_k requires input A's shard height {} to equal M {}",
        a_shard.shape[0],
        M);
    TT_FATAL(
        Kc % tt::constants::TILE_WIDTH == 0,
        "matmul_decode_large_k requires input A's shard width {} to be a multiple of {}",
        Kc,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        Kc * num_cores == K,
        "matmul_decode_large_k requires A's {} shards of width {} to cover K = {} exactly",
        num_cores,
        Kc,
        K);

    TT_FATAL(b.layout() == Layout::TILE, "matmul_decode_large_k requires input B in TILE, but got {}", b.layout());
    const auto& b_tile = b.tensor_spec().tile();
    TT_FATAL(
        b_tile.get_height() == tt::constants::TILE_HEIGHT && b_tile.get_width() == tt::constants::TILE_WIDTH,
        "matmul_decode_large_k requires input B in 32x32 tiles, but got {}x{}",
        b_tile.get_height(),
        b_tile.get_width());
    TT_FATAL(
        b.dtype() == DataType::BFLOAT16 || b.dtype() == DataType::BFLOAT8_B || b.dtype() == DataType::BFLOAT4_B,
        "matmul_decode_large_k requires input B in BFLOAT16, BFLOAT8_B or BFLOAT4_B, but got {}",
        b.dtype());
    TT_FATAL(
        b.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            b.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
        "matmul_decode_large_k requires input B HEIGHT_SHARDED in L1, but got {}",
        b.memory_config());
    const auto& b_shard = b.memory_config().shard_spec().value();
    TT_FATAL(
        b_shard.grid == a_shard.grid && b_shard.orientation == a_shard.orientation,
        "matmul_decode_large_k requires input B on the same cores and shard orientation as input A, but got {} vs {}",
        b_shard.grid.str(),
        a_shard.grid.str());
    TT_FATAL(
        b_shard.shape[0] == Kc && b_shard.shape[1] == N,
        "matmul_decode_large_k requires input B's shard to be [Kc, N] = [{}, {}], but got [{}, {}]",
        Kc,
        N,
        b_shard.shape[0],
        b_shard.shape[1]);
    TT_FATAL(
        large_k_leading_volume(b.logical_shape()) == 1 && b.logical_shape()[-2] == K,
        "matmul_decode_large_k requires input B to be [K, N] with K = {}, but got shape {}",
        K,
        b.logical_shape());
    TT_FATAL(
        N % tt::constants::TILE_WIDTH == 0,
        "matmul_decode_large_k requires N {} to be a multiple of {}",
        N,
        tt::constants::TILE_WIDTH);

    const DataType out_dtype = operation_attributes.output_dtype;
    TT_FATAL(
        out_dtype == DataType::BFLOAT16 || out_dtype == DataType::FLOAT32,
        "matmul_decode_large_k produces a ROW_MAJOR output, so its dtype must be BFLOAT16 or FLOAT32, but got {}",
        out_dtype);

    const uint32_t fan_in = operation_attributes.reduce_fan_in;
    TT_FATAL(fan_in >= 2, "matmul_decode_large_k requires reduce_fan_in >= 2, but got {}", fan_in);
    const uint32_t num_levels = large_k_tree_num_levels(num_cores, fan_in);
    TT_FATAL(
        num_levels <= kLargeKSemaphoresPerCore,
        "matmul_decode_large_k needs one semaphore per tree level, but {} cores at fan-in {} make {} levels (max {})",
        num_cores,
        fan_in,
        num_levels,
        kLargeKSemaphoresPerCore);
}

MatmulDecodeLargeKDeviceOperation::spec_return_value_t MatmulDecodeLargeKDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    ttnn::Shape output_shape(a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto shard_spec = tt::tt_metal::ShardSpec(
        large_k_root_core(a),
        {static_cast<uint32_t>(operation_attributes.M), static_cast<uint32_t>(operation_attributes.N)},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const auto memory_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            memory_config));
}

MatmulDecodeLargeKDeviceOperation::tensor_return_value_t MatmulDecodeLargeKDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::experimental::matmul_decode

namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeLargeKDeviceOperation::tensor_return_value_t
matmul_decode_large_k(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const DataType> dtype,
    uint32_t reduce_fan_in) {
    using OperationType = ttnn::operations::experimental::matmul_decode::MatmulDecodeLargeKDeviceOperation;

    // compute_output_specs runs ahead of validation and reads A's shard grid, so that much has to
    // hold before launching.
    TT_FATAL(
        input_tensor_a.memory_config().is_sharded() && input_tensor_a.memory_config().shard_spec().has_value(),
        "matmul_decode_large_k requires input A to be sharded with a shard spec, but got {}",
        input_tensor_a.memory_config());

    const auto& a_shape = input_tensor_a.logical_shape();
    const auto& b_shape = input_tensor_b.logical_shape();
    TT_FATAL(
        a_shape.rank() >= 2 && b_shape.rank() >= 2,
        "matmul_decode_large_k requires inputs of rank >= 2, but got {} and {}",
        a_shape,
        b_shape);
    const auto operation_attributes = OperationType::operation_attributes_t{
        .M = static_cast<int>(a_shape[-2]),
        .N = static_cast<int>(b_shape[-1]),
        .K = static_cast<int>(a_shape[-1]),
        .output_dtype = dtype.has_value() ? *dtype : input_tensor_a.dtype(),
        .reduce_fan_in = reduce_fan_in,
    };
    const auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
