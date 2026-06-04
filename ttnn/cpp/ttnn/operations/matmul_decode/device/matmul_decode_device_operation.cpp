// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::matmul_decode {

MatmulDecodeDeviceOperation::program_factory_t MatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // TEMPLATE: always pick the full width-sharded skeleton. A real implementation
    // would choose between factories based on shapes / memory layout / core grid.
    if (operation_attributes.N >= operation_attributes.K) {
        return FullWidthSharded{};
    }
    return FullWidthSharded{};
}

void MatmulDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TEMPLATE: no validation. A real implementation would check dtypes, layouts,
    // and that the inner dimensions of the two inputs match (K_a == K_b).
    if (tensor_args.input_tensor_a.logical_shape().rank() > 2) {
        for (int i = 0; i < tensor_args.input_tensor_a.logical_shape().rank() - 2; i++) {
            TT_FATAL(
                tensor_args.input_tensor_a.logical_shape()[i] == tensor_args.input_tensor_b.logical_shape()[i],
                "Input tensor A and B must have the same shape for all dimensions except the last two, but got {} and "
                "{}",
                tensor_args.input_tensor_a.logical_shape(),
                tensor_args.input_tensor_b.logical_shape());
        }
    }
    TT_FATAL(
        tensor_args.input_tensor_a.logical_shape()[-1] == operation_attributes.K,
        "Input tensor A must have the same K dimension as the operation attributes");
    TT_FATAL(
        tensor_args.input_tensor_b.logical_shape()[-2] == operation_attributes.K,
        "Input tensor B must have the same K dimension as the operation attributes");
    TT_FATAL(
        tensor_args.input_tensor_a.logical_shape()[-2] == operation_attributes.M,
        "Input tensor A must have the same M dimension as the operation attributes");
    TT_FATAL(
        tensor_args.input_tensor_b.logical_shape()[-1] == operation_attributes.N,
        "Input tensor B must have the same N dimension as the operation attributes");
    TT_FATAL(tensor_args.input_tensor_a.layout() == Layout::TILE, "Input tensor A must be in tile layout");
    TT_FATAL(tensor_args.input_tensor_b.layout() == Layout::TILE, "Input tensor B must be in tile layout");
    TT_FATAL(
        tensor_args.input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor A must be in width sharded memory layout, but got {}",
        tensor_args.input_tensor_a.memory_config().memory_layout());
    TT_FATAL(
        tensor_args.input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor B must be in width sharded memory layout, but got {}",
        tensor_args.input_tensor_b.memory_config().memory_layout());
}

MatmulDecodeDeviceOperation::spec_return_value_t MatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    // Output shape is the LHS shape with the last dim replaced by the RHS's last dim:
    // [..., M, K] x [..., K, N] -> [..., M, N]
    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = input_tensor_b.logical_shape()[-1];

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
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, std::optional<const DataType> dtype) {
    using OperationType = ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        input_tensor_a.logical_shape()[-2],
        input_tensor_b.logical_shape()[-1],
        input_tensor_a.logical_shape()[-1],
        input_tensor_a.memory_config(),
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
