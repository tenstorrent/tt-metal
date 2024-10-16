// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::binary {

BinaryDeviceOperation::program_factory_t BinaryDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    ZoneScopedN("BinaryDeviceOperation::select_program_factory");
    const auto& input_shape_a = tensor_args.input_tensor_a.tensor_attributes->shape;
    const auto& input_shape_b = tensor_args.input_tensor_b.tensor_attributes->shape;

    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    if (height_a == height_b and width_a == width_b) {
        return ElementWiseMultiCore{};
    } else if (height_b == 1 or width_b == 1) {
        if (height_b == 1 and width_b == 1) {
            return BroadcastHeightAndWidthMultiCore{};
        } else if (height_b == 1) {
            if (tensor_args.input_tensor_a.is_sharded()) {
                if (tensor_args.input_tensor_a.get_legacy_shape()[0] ==
                        tensor_args.input_tensor_b.get_legacy_shape()[0] ||
                    tensor_args.input_tensor_a.get_legacy_shape()[0] > 1 and
                        tensor_args.input_tensor_b.get_legacy_shape()[0] == 1) {
                    return BroadcastHeightMultiCoreShardedOptimized{};
                } else {
                    return BroadcastHeightMultiCoreSharded{};
                }
            }
            return BroadcastHeightMultiCore{};
        } else if (width_b == 1) {
            return BroadcastWidthMultiCore{};
        }
    }
    TT_THROW("ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast");
}

void BinaryDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& attributes,
                                                           const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    BinaryDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(),
             "Operands to eltwise binary need to be on the same device!");
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
             "Inputs to eltwise binary must be tilized");
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()), "Error");
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1, "Error");
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout,
                     "Error");
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value(), "Error");
        }
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == attributes.memory_config.memory_layout, "Error");
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == attributes.memory_config.memory_layout, "Error");
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0, "Error");

        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    }

    auto program_factory = select_program_factory(attributes, tensor_args);
    std::visit(
        [&attributes](auto&& program_factory) {
            if constexpr (std::is_same_v<decltype(program_factory), ElementWiseMultiCore>) {
                TT_FATAL(not attributes.activations.has_value(), "Error");
            }
        },
        program_factory);
}
void BinaryDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& attributes,
                                                          const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        TT_ASSERT(batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
                  "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
                  "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
    }
    if (height_a != height_b) {
        TT_ASSERT(height_a > height_b and height_b == 1,
                  "ttnn::operations::binary::BinaryDeviceOperation: height mismatch");
    }
    if (width_a != width_b) {
        TT_ASSERT(width_a > width_b and width_b == 1,
                  "ttnn::operations::binary::BinaryDeviceOperation: width mismatch");
    }
}

BinaryDeviceOperation::shape_return_value_t BinaryDeviceOperation::compute_output_shapes(
    const operation_attributes_t&,
    const tensor_args_t& tensor_args) {
    const auto input_shape_a = tensor_args.input_tensor_a.tensor_attributes->shape;
    const auto input_shape_b = tensor_args.input_tensor_b.tensor_attributes->shape;

    const int rank_a = input_shape_a.rank();
    const int rank_b = input_shape_b.rank();
    const int larger_rank = std::max(rank_a, rank_b);

    // -------------------------------------------------------------------------
    // This lambda function computes the broadcasted output shape between two tensors.
    // It follows the broadcasting rules to determine the shape of the result
    // when performing binary operations on tensors of potentially different shapes and ranks.
    //
    // Broadcasting Rules Overview:
    // - If the two tensors have different ranks, we virtually pad the smaller-rank tensor's shape
    //   with ones on the left (i.e., higher-order dimensions) until both shapes have the same length.
    // - For each dimension (starting from the rightmost), the sizes are compatible if:
    //     - They are equal, or
    //     - One of them is 1 (the dimension can be broadcast to match the other size).
    // - The result dimension is the maximum of the two sizes.
    //
    // Key Points:
    // - Negative indexing simplifies dimension alignment from the right (least significant dimensions),
    //   thats essential for correct broadcasting.
    // - By defaulting to 1 for missing dimensions, we correctly handle tensors of different ranks.
    // - The use of 'std::max' ensures that when one of the dimensions is 1, the other dimension size
    //   is used, adhering to broadcasting rules. Important! Code assumes that shapes are validated beforehand.
    // - The lambda is reused for both logical shapes and padded shapes, ensuring consistency.
    // -------------------------------------------------------------------------
    auto compute_broadcasted_output = [rank_a, rank_b, larger_rank](const auto& shape_a, const auto& shape_b) {
        std::vector<uint32_t> output_shape(larger_rank, 1);
        for (int i = -1; i >= -larger_rank; --i) {
            auto dim_a = (i >= -rank_a) ? shape_a[i] : 1;
            auto dim_b = (i >= -rank_b) ? shape_b[i] : 1;
            output_shape[i + larger_rank] = std::max(dim_a, dim_b);
        }
        return output_shape;
    };

    const auto logical_shape_a = input_shape_a.logical_shape();
    const auto logical_shape_b = input_shape_b.logical_shape();
    const auto output_shape = compute_broadcasted_output(logical_shape_a, logical_shape_b);

    const auto padded_shape_a = input_shape_a.padded_shape();
    const auto padded_shape_b = input_shape_b.padded_shape();
    const auto output_shape_with_tile_padding = compute_broadcasted_output(padded_shape_a, padded_shape_b);

    return ttnn::Shape(output_shape, output_shape_with_tile_padding);
}

BinaryDeviceOperation::tensor_return_value_t BinaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    if (output_tensor.has_value()) {
        return output_tensor.value();
    }

    auto program_factory = select_program_factory(operation_attributes, tensor_args);
    if (std::holds_alternative<ElementWiseMultiCore>(program_factory)) {
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                shard_spec = input_tensor_a.shard_spec().value();
            } else if (input_tensor_b.memory_config().is_sharded()) {
                shard_spec = input_tensor_b.shard_spec().value();
            } else {
                uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                uint32_t num_grid_cores = core_grid.x * core_grid.y;
                uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
                shard_spec.grid = tt::tt_metal::num_cores_to_corerange_set(target_num_cores, core_grid, true);
                shard_spec.shape = {num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
                shard_spec.orientation = ShardOrientation::ROW_MAJOR;
            }
            auto memory_config = operation_attributes.memory_config;
            memory_config.shard_spec = shard_spec;
            return create_device_tensor(
                output_shape, operation_attributes.dtype, Layout::TILE, input_tensor_a.device(), memory_config);
        }
    } else {
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                // Derive output shard_spec based on input
                shard_spec = input_tensor_a.shard_spec().value();
            }
            auto memory_config = operation_attributes.memory_config;
            memory_config.shard_spec = shard_spec;
            return create_device_tensor(
                output_shape, operation_attributes.dtype, Layout::TILE, input_tensor_a.device(), memory_config);
        }
    }
    return create_device_tensor(output_shape,
                                operation_attributes.dtype,
                                Layout::TILE,
                                input_tensor_a.device(),
                                operation_attributes.memory_config);
}

tt::stl::hash::hash_t BinaryDeviceOperation::compute_program_hash(const operation_attributes_t& attributes,
                                                                  const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    auto program_factory = select_program_factory(attributes, tensor_args);
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_a.get_storage()),
              "Unexpected type {}",
              tt::stl::get_active_type_name_in_variant(input_tensor_a.get_storage()));
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_b.get_storage()),
              "Unexpected type {}",
              tt::stl::get_active_type_name_in_variant(input_tensor_b.get_storage()));

    operation::Hash hash = operation::hash_operation<BinaryDeviceOperation>(
        attributes,
        program_factory.index(),
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config());
    return hash;
}

operation::OpPerformanceModel BinaryDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_return_value;
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr int num_cores = 9 * 12;

    int total_bytes = 0;
    total_bytes += input_tensor_a.volume() * input_tensor_a.element_size();
    total_bytes += input_tensor_b.volume() * input_tensor_b.element_size();
    int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    // TODO: update OpPerformanceModel to work on variadic arguments
    operation::OpPerformanceModel result({input_tensor_a, input_tensor_b}, {output_tensor}, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "BinaryDeviceOperation PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

std::tuple<BinaryDeviceOperation::operation_attributes_t, BinaryDeviceOperation::tensor_args_t> BinaryDeviceOperation::
    invoke(const Tensor& input_tensor_a_arg,
           const Tensor& input_tensor_b_arg,
           BinaryOpType binary_op_type,
           const std::optional<const DataType>& output_dtype,
           const std::optional<MemoryConfig>& memory_config,
           std::optional<Tensor> optional_output_tensor,
           std::optional<unary::FusedActivations> activations,
           std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(output_dtype.value() == optional_output_tensor.value().get_dtype(),
                 "If both output dtype and output tensor provided dtype should match");
    }

    return {operation_attributes_t{binary_op_type,
                                   activations,
                                   input_tensor_a_activation,
                                   memory_config.value_or(input_tensor_a_arg.memory_config()),
                                   output_dtype.value_or(input_tensor_a_arg.get_dtype()),
                                   std::nullopt},
            tensor_args_t{input_tensor_a_arg, input_tensor_b_arg, optional_output_tensor}};
}

}  // namespace ttnn::operations::binary
