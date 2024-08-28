// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::binary {

BinaryDeviceOperation::program_factory_t BinaryDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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
            if(tensor_args.input_tensor_a.is_sharded()){
                if (tensor_args.input_tensor_a.get_legacy_shape()[0] == tensor_args.input_tensor_b.get_legacy_shape()[0]
                        || tensor_args.input_tensor_a.get_legacy_shape()[0] > 1
                        and tensor_args.input_tensor_b.get_legacy_shape()[0] == 1){
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

void BinaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    BinaryDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to eltwise binary need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to eltwise binary must be tilized");
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()));
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == attributes.memory_config.memory_layout);
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == attributes.memory_config.memory_layout);
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0);

        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }

    auto program_factory = select_program_factory(attributes, tensor_args);
    std::visit(
        [&attributes](auto&& program_factory) {
            if constexpr (std::is_same_v<decltype(program_factory), ElementWiseMultiCore>) {
                TT_FATAL(not attributes.activations.has_value());
            }
        },
        program_factory);

}
void BinaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
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
        TT_ASSERT(
            batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
            "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(
            batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
            "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
    }
    if (height_a != height_b) {
        TT_ASSERT(
            height_a > height_b and height_b == 1, "ttnn::operations::binary::BinaryDeviceOperation: height mismatch");
    }
    if (width_a != width_b) {
        TT_ASSERT(
            width_a > width_b and width_b == 1, "ttnn::operations::binary::BinaryDeviceOperation: width mismatch");
    }
}

BinaryDeviceOperation::shape_return_value_t BinaryDeviceOperation::compute_output_shapes(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto input_shape_a = tensor_args.input_tensor_a.tensor_attributes->shape;
    const auto input_shape_b = tensor_args.input_tensor_b.tensor_attributes->shape;

    auto rank = std::max(input_shape_a.rank(), input_shape_b.rank());
    std::vector<uint32_t> output_shape(rank, 0);
    std::vector<uint32_t> output_shape_with_tile_padding(rank, 0);

    for (int i = -1; i >= -rank; --i) {
        auto dim_a = i + input_shape_a.rank() < input_shape_a.rank() ? input_shape_a[i] : 1;
        auto dim_b = i + input_shape_b.rank() < input_shape_b.rank() ? input_shape_b[i] : 1;
        output_shape[i + rank] = std::max(dim_a, dim_b);

        auto dim_a_with_tile_padding =
            i + input_shape_a.rank() < input_shape_a.rank() ? input_shape_a.with_tile_padding()[i] : 1;
        auto dim_b_with_tile_padding =
            i + input_shape_b.rank() < input_shape_b.rank() ? input_shape_b.with_tile_padding()[i] : 1;
        output_shape_with_tile_padding[i + rank] = std::max(dim_a_with_tile_padding, dim_b_with_tile_padding);
    }
    return ttnn::Shape(output_shape, output_shape_with_tile_padding);
}

BinaryDeviceOperation::tensor_return_value_t BinaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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
                shard_spec.grid = num_cores_to_corerange_set(target_num_cores, core_grid, true);
                shard_spec.shape = {
                    num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
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
    return create_device_tensor(
        output_shape,
        operation_attributes.dtype,
        Layout::TILE,
        input_tensor_a.device(),
        operation_attributes.memory_config);

}

tt::stl::hash::hash_t BinaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    auto program_factory = select_program_factory(attributes, tensor_args);
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.get_storage()),
        fmt::format(
            "Unexpected type {} in {}:{} ",
            tt::stl::get_active_type_name_in_variant(input_tensor_a.get_storage()),
            __FILE__,
            __LINE__));
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_b.get_storage()),
        fmt::format(
            "Unexpected type {} in {}:{} ",
            tt::stl::get_active_type_name_in_variant(input_tensor_b.get_storage()),
            __FILE__,
            __LINE__));
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



std::tuple<BinaryDeviceOperation::operation_attributes_t, BinaryDeviceOperation::tensor_args_t> BinaryDeviceOperation::invoke(
    const Tensor &input_tensor_a_arg,
    const Tensor &input_tensor_b_arg,
    BinaryOpType binary_op_type,
    const std::optional<const DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<unary::FusedActivations> activations,
    std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == optional_output_tensor.value().get_dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    return {
        operation_attributes_t{
            binary_op_type,
            activations,
            input_tensor_a_activation,
            memory_config.value_or(input_tensor_a_arg.memory_config()),
            output_dtype.value_or(input_tensor_a_arg.get_dtype()),
            std::nullopt},
        tensor_args_t{input_tensor_a_arg, input_tensor_b_arg, optional_output_tensor}};
    }

}  // namespace ttnn::operations::binary
