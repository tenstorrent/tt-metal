// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"

#include <utility>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::binary {

namespace utils {
bool is_binary_sfpu_op(BinaryOpType val, DataType a, DataType b) {
    switch (val) {
        case BinaryOpType::ADD:
        case BinaryOpType::SUB:
        case BinaryOpType::MUL:
        case BinaryOpType::EQ:
        case BinaryOpType::NE:
        case BinaryOpType::LOGICAL_AND:
        case BinaryOpType::LOGICAL_OR:
        case BinaryOpType::LOGICAL_XOR:
        case BinaryOpType::SQUARED_DIFFERENCE:
            return a == b &&
                   (a == DataType::FLOAT32 || a == DataType::INT32 || a == DataType::UINT32 || a == DataType::UINT16);
        case BinaryOpType::LOGADDEXP:
        case BinaryOpType::LOGADDEXP2:
        case BinaryOpType::LDEXP:
        case BinaryOpType::BIAS_GELU:
        case BinaryOpType::HYPOT: return (a == DataType::FLOAT32 && b == DataType::FLOAT32);
        case BinaryOpType::DIV:
        case BinaryOpType::RSUB:
        case BinaryOpType::GT:
        case BinaryOpType::LT:
        case BinaryOpType::GE:
        case BinaryOpType::LE:
            return (
                (a == DataType::FLOAT32 && b == DataType::FLOAT32) || (a == DataType::INT32 && b == DataType::INT32));
        case BinaryOpType::GCD:
        case BinaryOpType::LCM:
        case BinaryOpType::LEFT_SHIFT:
        case BinaryOpType::RIGHT_SHIFT:
        case BinaryOpType::LOGICAL_RIGHT_SHIFT:
            return ((a == DataType::INT32 && b == DataType::INT32) || (a == DataType::UINT32 && b == DataType::UINT32));
        case BinaryOpType::BITWISE_XOR:
        case BinaryOpType::BITWISE_OR:
        case BinaryOpType::BITWISE_AND:
            return a == b && (a == DataType::INT32 || a == DataType::UINT32 || a == DataType::UINT16);
        case BinaryOpType::MAXIMUM:
        case BinaryOpType::MINIMUM:
        case BinaryOpType::XLOGY:
        case BinaryOpType::POWER: return true;
        default: return false;
    }
    return false;
}
}  // namespace utils

BinaryDeviceOperation::program_factory_t BinaryDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    ZoneScopedN("BinaryDeviceOperation::select_program_factory");
    const auto input_shape_a = tensor_args.input_tensor_a.logical_shape();

    if (operation_attributes.scalar.has_value()) {
        return BroadcastHeightAndWidthMultiCore{};
    }

    const auto input_shape_b = tensor_args.input_tensor_b->logical_shape();

    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    if (height_a == height_b and width_a == width_b) {
        BinaryOpType op = operation_attributes.binary_op_type;
        DataType dtype1 = tensor_args.input_tensor_a.dtype();
        DataType dtype2 = tensor_args.input_tensor_b->dtype();
        bool sfpu_op_check = utils::is_binary_sfpu_op(op, dtype1, dtype2);
        if (sfpu_op_check) {
            return ElementWiseMultiCoreSfpu{};
        }
        return ElementWiseMultiCore{};
    }
    if (height_b == 1 or width_b == 1) {
        if (height_b == 1 and width_b == 1) {
            return BroadcastHeightAndWidthMultiCore{};
        }
        if (height_b == 1) {
            if (tensor_args.input_tensor_a.is_sharded()) {
                if (tensor_args.input_tensor_a.padded_shape()[0] == tensor_args.input_tensor_b->padded_shape()[0] ||
                    (tensor_args.input_tensor_a.padded_shape()[0] > 1 &&
                     tensor_args.input_tensor_b->padded_shape()[0] == 1)) {
                    return BroadcastHeightMultiCoreShardedOptimized{};
                }
                return BroadcastHeightMultiCoreSharded{};
            }
            return BroadcastHeightMultiCore{};
        }
        if (width_b == 1) {
            return BroadcastWidthMultiCore{};
        }
    }
    TT_THROW("ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast");
}

void BinaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a.storage_type());

    if (input_tensor_b.has_value()) {
        TT_FATAL(
            input_tensor_b->storage_type() == StorageType::DEVICE,
            "Input tensor B must be on device, got storage type: {}",
            input_tensor_b->storage_type());
    }

    TT_FATAL(
        input_tensor_b.has_value() != attributes.scalar.has_value(), "Either the tensor b or scalar should be set");

    BinaryDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input to eltwise binary must be tilized");

    bool tensor_b_sharded = false;

    if (input_tensor_b.has_value()) {
        tensor_b_sharded = input_tensor_b->memory_config().is_sharded();
        if (input_tensor_a.device() != input_tensor_b->device()) {
            TT_FATAL(
                input_tensor_a.device() == input_tensor_b->device(),
                "Operands to eltwise binary need to be on the same device!");
        }
        TT_FATAL(input_tensor_b->layout() == Layout::TILE, "Inputs to eltwise binary must be tilized");
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        if (tensor_b_sharded) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == input_tensor_b->memory_config().memory_layout(),
                "Input tensor A and input tensor B must have the same memory layout");
            TT_FATAL(
                input_tensor_a.shard_spec().value() == input_tensor_b->shard_spec().value(),
                "Input tensor A and input tensor B must have the same shard spec");
        }
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == attributes.memory_config.memory_layout(),
                "Input tensor A and output tensor must have the same memory layout");
        } else {
            TT_FATAL(
                attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Output tensor must be interleaved");
        }
    } else if (tensor_b_sharded) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor A must be interleaved");
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(
                input_tensor_b->memory_config().memory_layout() == attributes.memory_config.memory_layout(),
                "Input tensor B and output tensor must have the same memory layout");
        } else {
            TT_FATAL(
                attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Output tensor must be interleaved");
        }
    } else {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor A must be interleaved");
        if (input_tensor_b.has_value()) {
            TT_FATAL(
                input_tensor_b->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input tensor B must be interleaved");
        }
        if (!attributes.memory_config.is_sharded()) {
            TT_FATAL(
                attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Output tensor must be interleaved");
        }
    }

    auto program_factory = select_program_factory(attributes, tensor_args);
    std::visit(
        [&attributes](auto&& program_factory) {
            if constexpr (std::is_same_v<decltype(program_factory), ElementWiseMultiCore>) {
                TT_FATAL(not attributes.activations.has_value(), "Activations are not supported for binary operations");
            }
        },
        program_factory);
}

void BinaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;

    const auto& input_shape_a = input_tensor_a.logical_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    const auto input_shape_b =
        tensor_args.input_tensor_b.has_value() ? tensor_args.input_tensor_b->logical_shape() : ttnn::Shape{1, 1};
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

    TT_FATAL(
        height_a == height_b || height_a == 1 || height_b == 1,
        "ttnn::operations::binary::BinaryDeviceOperation: height mismatch");
    TT_FATAL(
        width_a == width_b || width_a == 1 || width_b == 1,
        "ttnn::operations::binary::BinaryDeviceOperation: width mismatch");
}

BinaryDeviceOperation::spec_return_value_t BinaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor->tensor_spec();
    }

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto input_shape_a = input_tensor_a.logical_shape();
    const auto& tensor_b = tensor_args.input_tensor_b;
    const auto input_shape_b = tensor_b.has_value() ? tensor_b->logical_shape() : ttnn::Shape{};

    const int rank_a = input_shape_a.rank();
    const int rank_b = input_shape_b.rank();
    const int larger_rank = std::max(rank_a, rank_b);

    // Broadcasting Rules Overview:
    // - If the two tensors have different ranks, we virtually pad the smaller-rank tensor's shape
    //   with ones on the left (i.e., higher-order dimensions) until both shapes have the same length.
    // - For each dimension (starting from the rightmost), the sizes are compatible if:
    //     - They are equal, or
    //     - One of them is 1 (the dimension can be broadcast to match the other size).
    auto compute_broadcasted_output = [rank_a, rank_b, larger_rank](const auto& shape_a, const auto& shape_b) {
        SmallVector<uint32_t> output_shape(larger_rank, 1);
        for (int i = -1; i >= -larger_rank; --i) {
            auto dim_a = (i >= -rank_a) ? shape_a[i] : 1;
            auto dim_b = (i >= -rank_b) ? shape_b[i] : 1;
            if (dim_a != 1 && dim_b != 1) {
                TT_FATAL(dim_a == dim_b, "Incompatible dimensions {} and {}", dim_a, dim_b);
                output_shape[i + larger_rank] = dim_a;
            } else {
                // One of the dimension is one, calculating the other one
                output_shape[i + larger_rank] = dim_a + dim_b - 1;
            }
        }
        return ttnn::Shape(output_shape);
    };
    auto output_shape = compute_broadcasted_output(input_shape_a, input_shape_b);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);
    if (std::holds_alternative<ElementWiseMultiCore>(program_factory) or
        std::holds_alternative<ElementWiseMultiCoreSfpu>(program_factory)) {
        const auto& input_tensor_b = *tensor_args.input_tensor_b;
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                shard_spec = input_tensor_a.shard_spec().value();
            } else if (input_tensor_b.memory_config().is_sharded()) {
                shard_spec = input_tensor_b.shard_spec().value();
            } else {
                shard_spec = operation_attributes.memory_config.shard_spec().value();
            }
            auto memory_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape, TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), memory_config));
        }
    } else {
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                // Derive output shard_spec based on input
                shard_spec = input_tensor_a.shard_spec().value();
            }
            auto memory_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape, TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), memory_config));
        }
    }
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
}

BinaryDeviceOperation::tensor_return_value_t BinaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return *tensor_args.output_tensor;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t BinaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    auto program_factory = select_program_factory(attributes, tensor_args);
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.storage()));

    if (input_tensor_b.has_value()) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_tensor_b->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_tensor_b->storage()));

        return operation::hash_operation<BinaryDeviceOperation>(
            attributes,
            program_factory.index(),
            input_tensor_a.dtype(),
            input_tensor_a.memory_config(),
            input_tensor_b->dtype(),
            input_tensor_b->memory_config());
    }

    return operation::hash_operation<BinaryDeviceOperation>(
        attributes, program_factory.index(), input_tensor_a.dtype(), input_tensor_a.memory_config());
}

operation::OpPerformanceModelGeneral<BinaryDeviceOperation::tensor_return_value_t>
BinaryDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_return_value;
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr uint32_t num_cores = 9 * 12;

    uint32_t total_bytes = 0;
    std::vector<Tensor> input_tensors = {input_tensor_a};
    total_bytes += input_tensor_a.physical_volume() * input_tensor_a.element_size();
    if (input_tensor_b.has_value()) {
        input_tensors.push_back(*input_tensor_b);
        total_bytes += input_tensor_b->physical_volume() * input_tensor_b->element_size();
    }
    uint32_t ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    // TODO: update OpPerformanceModel to work on variadic arguments
    operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        input_tensors, output_tensor, ideal_eltwise_cycles);
#if 0
        log_info(tt::LogOp, "BinaryDeviceOperation PerfModel:");
        log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

bool BinaryDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::binary

namespace ttnn::prim {

ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t binary(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    ttnn::operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<ttnn::operations::unary::EltwiseFusedActivations> activations,
    std::optional<ttnn::operations::unary::EltwiseUnaryWithParam> input_tensor_a_activation) {
    using OperationType = ttnn::operations::binary::BinaryDeviceOperation;
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    TT_FATAL(
        input_tensor_a_arg.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a_arg.storage_type());

    TT_FATAL(
        input_tensor_b_arg.storage_type() == StorageType::DEVICE,
        "Input tensor B must be on device, got storage type: {}",
        input_tensor_b_arg.storage_type());

    CoreRangeSet worker_grid;
    // We assert all shard specs are the same if sharded, so only need to check the first shard spec
    // This will create the worker grid based on the used sub-devices when sharded
    // Otherwise this will use all cores of the sub-devices
    // TODO #13655: Note that the current program ingfrastructure still only supports a single sub-device per program
    if (input_tensor_a_arg.is_sharded()) {
        const auto& input_grid = input_tensor_a_arg.shard_spec().value().grid;
        auto* device = input_tensor_a_arg.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(input_grid)) {
                worker_grid = worker_grid.merge(sub_device_workers);
            }
        }
    } else if (input_tensor_b_arg.is_sharded()) {
        const auto& input_grid = input_tensor_b_arg.shard_spec().value().grid;
        auto* device = input_tensor_b_arg.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(input_grid)) {
                worker_grid = worker_grid.merge(sub_device_workers);
            }
        }
    } else if (optional_output_tensor.has_value() && optional_output_tensor->is_sharded()) {
        const auto& output_grid = optional_output_tensor->shard_spec().value().grid;
        auto* device = optional_output_tensor->device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(output_grid)) {
                worker_grid = worker_grid.merge(sub_device_workers);
            }
        }
    } else {
        auto* device = input_tensor_a_arg.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            worker_grid = worker_grid.merge(sub_device_workers);
        }
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        binary_op_type,
        std::move(activations),
        std::move(input_tensor_a_activation),
        std::nullopt,
        memory_config.value_or(
            optional_output_tensor.has_value() ? optional_output_tensor->memory_config()
                                               : input_tensor_a_arg.memory_config()),
        output_dtype.value_or(input_tensor_a_arg.dtype()),
        std::move(worker_grid),
        std::nullopt};
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a_arg, input_tensor_b_arg, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t binary(
    const Tensor& input_tensor_a_arg,
    float scalar,
    ttnn::operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<ttnn::operations::unary::EltwiseFusedActivations> activations,
    std::optional<ttnn::operations::unary::EltwiseUnaryWithParam> input_tensor_a_activation) {
    using OperationType = ttnn::operations::binary::BinaryDeviceOperation;
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    // Currently unused/unsupported
    CoreRangeSet worker_grid = CoreRangeSet();
    auto operation_attributes = OperationType::operation_attributes_t{
        binary_op_type,
        std::move(activations),
        std::move(input_tensor_a_activation),
        scalar,
        memory_config.value_or(input_tensor_a_arg.memory_config()),
        output_dtype.value_or(input_tensor_a_arg.dtype()),
        std::move(worker_grid),
        std::nullopt};
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a_arg, std::nullopt, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
