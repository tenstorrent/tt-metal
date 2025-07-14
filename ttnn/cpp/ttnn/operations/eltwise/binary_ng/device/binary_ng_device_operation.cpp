// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_ng {

namespace utils {
bool is_binary_sfpu_op(BinaryOpType val, DataType a, DataType b) {
    using enum BinaryOpType;
    using enum DataType;
    switch (val) {
        case ADD:
            return (
                (a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32) || (a == UINT32 && b == UINT32) ||
                (a == UINT16 && b == UINT16));
        case SUB: return ((a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32) || (a == UINT16 && b == UINT16));
        case MUL: return ((a == FLOAT32 && b == FLOAT32) || (a == UINT16 && b == UINT16));
        case DIV:
        case RSUB:
        case LOGADDEXP:
        case LOGADDEXP2:
        case LDEXP:
        case SQUARED_DIFFERENCE:
        case LOGICAL_AND:
        case BIAS_GELU: return (a == FLOAT32 && b == FLOAT32);
        case LOGICAL_OR:
        case LOGICAL_XOR:
        case GT:
        case LT:
        case GE:
        case LE:
        case EQ:
        case NE: return ((a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32));
        case LCM:
        case GCD: return (a == INT32 && b == INT32);
        case LEFT_SHIFT:
        case RIGHT_SHIFT:
        case LOGICAL_RIGHT_SHIFT: return ((a == INT32 || a == UINT32) && (b == INT32 || b == UINT32));
        case BITWISE_XOR:
        case BITWISE_OR:
        case BITWISE_AND: return ((a == INT32 && b == INT32) || (a == UINT16 && b == UINT16));
        case QUANT:
        case REQUANT:
        case DEQUANT:
        case MAXIMUM:
        case MINIMUM:
        case POWER: return true;
        default: return false;
    }
    return false;
}

bool is_quant_op(const BinaryOpType val) {
    return (val == BinaryOpType::QUANT) || (val == BinaryOpType::DEQUANT) || (val == BinaryOpType::REQUANT);
}
}  // namespace utils

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor_a, const Tensor* input_tensor_b, const std::optional<Tensor>& output_tensor) {
    auto get_tensor_grid = [](const Tensor& tensor) -> CoreRangeSet {
        const auto& grid = tensor.shard_spec()->grid;
        auto device = tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
        __builtin_unreachable();
    };

    if (input_tensor_a.is_sharded()) {
        return get_tensor_grid(input_tensor_a);
    } else if (input_tensor_b && input_tensor_b->is_sharded()) {
        return get_tensor_grid(*input_tensor_b);
    } else if (output_tensor.has_value() && output_tensor->is_sharded()) {
        return get_tensor_grid(*output_tensor);
    }

    auto device = input_tensor_a.device();
    return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
}

SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w) {
    if (a_h == b_h && a_w == b_w) {
        return SubtileBroadcastType::NONE;
    }
    if (a_h == 1 && a_w == 1) {
        return SubtileBroadcastType::SCALAR_A;
    }
    if (b_h == 1 && b_w == 1) {
        return SubtileBroadcastType::SCALAR_B;
    }
    if (a_h == 1 /* && a_w != 1 */ && b_w == 1 /* && b_h != 1 */) {
        return SubtileBroadcastType::ROW_A_COL_B;
    }
    if (a_w == 1 /* && a_h != 1 */ && b_h == 1 /* && b_w != 1 */) {
        return SubtileBroadcastType::ROW_B_COL_A;
    }
    if (a_h == 1) {
        return SubtileBroadcastType::ROW_A;
    }
    if (a_w == 1) {
        return SubtileBroadcastType::COL_A;
    }
    if (b_h == 1) {
        return SubtileBroadcastType::ROW_B;
    }
    if (b_w == 1) {
        return SubtileBroadcastType::COL_B;
    }

    TT_THROW("Invalid subtile broadcast type");
}

tt::stl::hash::hash_t BinaryNgDeviceOperation::operation_attributes_t::to_hash() const {
    // TODO: a more generalized way to skip the hashing of an UnaryWithParam?
    // Don't hash the quantization scale, otherwise we build the kernel for each different scale
    return tt::stl::hash::hash_objects_with_default_seed(
        binary_op_type,
        lhs_activations,
        rhs_activations,
        is_quant_op ? ttnn::SmallVector<unary::UnaryWithParam>{} : post_activations,
        memory_config,
        get_dtype(),
        compute_kernel_config,
        subtile_broadcast_type,
        is_sfpu,
        is_quant_op);
}

DataType BinaryNgDeviceOperation::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
}

void validate_sharding(
    TensorMemoryLayout memory_layout_x,
    const ShardSpec& shard_spec_x,
    TensorMemoryLayout memory_layout_y,
    const ShardSpec& shard_spec_y,
    SubtileBroadcastType subtile_broadcast_type) {
    TT_FATAL(memory_layout_x == memory_layout_y, "Operands to eltwise binary need to have the same memory layout");

    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            TT_FATAL(shard_spec_x == shard_spec_y, "Operands to eltwise binary need to have the same shard spec");
            break;
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::COL_B:
            TT_FATAL(
                memory_layout_x == TensorMemoryLayout::HEIGHT_SHARDED,
                "Operands to eltwise binary must be height sharded when broadcasting on W");
            TT_FATAL(
                memory_layout_y == TensorMemoryLayout::HEIGHT_SHARDED,
                "Operands to eltwise binary must be height sharded when broadcasting on W");
            TT_FATAL(
                shard_spec_x.shape[0] == shard_spec_y.shape[0],
                "Operands to eltwise binary need to have the same"
                "shard height when broadcasting on W");
            TT_FATAL(
                shard_spec_x.orientation == shard_spec_y.orientation,
                "Operands to eltwise binary must have same shard orientation");
            break;
        default: TT_THROW("Invalid subtile broadcast type for sharding validation");
    }
}

void BinaryNgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // We don't support sharding for now
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    TT_FATAL(
        input_tensor_b.has_value() != attributes.scalar.has_value(), "Either the tensor b or scalar should be set");

    BinaryNgDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    if (attributes.dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(
            *attributes.dtype == output_tensor->dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "First operand to eltwise binary must be tilized");

    bool tensor_a_sharded = input_tensor_a.memory_config().is_sharded();
    if (not tensor_a_sharded) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "LHS operand must be either sharded or interleaved");
    }

    bool output_sharded = attributes.memory_config.is_sharded();
    if (not output_sharded) {
        TT_FATAL(
            attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output must be interleaved or sharded");
    }

    bool tensor_b_sharded = false;

    if (input_tensor_b.has_value()) {
        tensor_b_sharded = input_tensor_b->memory_config().is_sharded();
        TT_FATAL(
            input_tensor_a.device() == input_tensor_b->device(),
            "Operands to eltwise binary need to be on the same device!");
        TT_FATAL(input_tensor_b->layout() == Layout::TILE, "Second operand to eltwise binary must be tilized");

        if (not tensor_b_sharded) {
            TT_FATAL(
                input_tensor_b->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "RHS operand must be either sharded or interleaved");
        }
    }

    // Validate that all shard specs match
    if (tensor_a_sharded) {
        if (tensor_b_sharded) {
            validate_sharding(
                input_tensor_a.memory_config().memory_layout(),
                *input_tensor_a.shard_spec(),
                input_tensor_b->memory_config().memory_layout(),
                *input_tensor_b->shard_spec(),
                attributes.subtile_broadcast_type);
        }
        if (output_sharded) {
            validate_sharding(
                input_tensor_a.memory_config().memory_layout(),
                *input_tensor_a.shard_spec(),
                attributes.memory_config.memory_layout(),
                attributes.memory_config.shard_spec().value_or(*input_tensor_a.shard_spec()),
                attributes.subtile_broadcast_type);
        }
    } else if (tensor_b_sharded) {
        if (output_sharded) {
            validate_sharding(
                input_tensor_b->memory_config().memory_layout(),
                *input_tensor_b->shard_spec(),
                attributes.memory_config.memory_layout(),
                attributes.memory_config.shard_spec().value_or(*input_tensor_b->shard_spec()),
                attributes.subtile_broadcast_type);
        }
    } else if (output_sharded) {
        TT_FATAL(
            attributes.memory_config.shard_spec().has_value(),
            "Sharded output memory config must have shard spec if neither input is sharded");
    }
}

void BinaryNgDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    bool has_shard_spec = input_tensor_a.memory_config().is_sharded() ||
                          (input_tensor_b.has_value() && input_tensor_b->memory_config().is_sharded()) ||
                          attributes.memory_config.is_sharded();

    if (output_tensor.has_value() && !has_shard_spec) {
        compute_output_specs(attributes, tensor_args);
    }

    const auto& input_shape_a = input_tensor_a.logical_shape();
    const auto& input_shape_b = input_tensor_b.has_value() ? input_tensor_b->logical_shape() : input_shape_a;

    const int rank_a = input_shape_a.rank();
    const int rank_b = input_shape_b.rank();
    const int larger_rank = std::max(rank_a, rank_b);

    for (int i = -1; i >= -larger_rank; --i) {
        auto a_dim = (i >= -rank_a) ? input_shape_a[i] : 1;
        auto b_dim = (i >= -rank_b) ? input_shape_b[i] : 1;
        TT_FATAL(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "Broadcasting rule violation for rank {}, dim a: {}, dim b: {}",
            i,
            a_dim,
            b_dim);

        if (i <= -6) {
            TT_FATAL(
                a_dim == b_dim,
                "Broadcasting rule violation for rank >= 6 : dim {}, Broadcast is supported up to rank 5, dim a: {}, "
                "dim b: {}",
                i,
                a_dim,
                b_dim);
        }

        if (has_shard_spec and i != -1) {
            TT_FATAL(
                a_dim == b_dim,
                "Cannot broadcast sharded tensors on dims other than W, violation for rank {}, dim a: {}, dim b: "
                "{}",
                i,
                a_dim,
                b_dim);
        }
    }
}

BinaryNgDeviceOperation::spec_return_value_t BinaryNgDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;

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
                output_shape[i + larger_rank] = dim_a;
            } else {
                output_shape[i + larger_rank] = dim_a + dim_b - 1;
            }
        }
        return ttnn::Shape(output_shape);
    };

    auto output_shape = compute_broadcasted_output(input_shape_a, input_shape_b);

    if (output_tensor.has_value()) {
        auto shapes_equal = [=](const auto& shape_a, const auto& shape_b) {
            const auto smaller_rank = std::min(shape_a.rank(), shape_b.rank());
            for (int i = 0; i < smaller_rank; ++i) {
                auto dim = -1 - i;
                if (shape_a[dim] != shape_b[dim]) {
                    return false;
                }
            }
            const auto& larger_shape = shape_a.rank() > shape_b.rank() ? shape_a : shape_b;
            for (int i = smaller_rank; i < larger_rank; ++i) {
                auto dim = -1 - i;
                if (larger_shape[dim] != 1) {
                    return false;
                }
            }
            return true;
        };
        auto shape = output_tensor.value().logical_shape();
        TT_FATAL(
            shapes_equal(shape, output_shape),
            "Shape of Output tensor {} provided does not match the broadcasted output shape {}",
            shape,
            output_shape);
        return output_tensor->tensor_spec();
    }

    if (attributes.memory_config.is_sharded()) {
        const auto& memory_layout = attributes.memory_config.memory_layout();
        const auto& buffer_type = attributes.memory_config.buffer_type();
        const auto& shard_spec = attributes.memory_config.shard_spec();
        const auto& input_a_shard_spec = input_tensor_a.memory_config().shard_spec();
        const auto& input_b_shard_spec = tensor_b.has_value() ? tensor_b->memory_config().shard_spec() : std::nullopt;
        const auto& output_shard_spec = shard_spec.has_value()           ? *shard_spec
                                        : input_a_shard_spec.has_value() ? *input_a_shard_spec
                                                                         : *input_b_shard_spec;
        return TensorSpec(
            output_shape,
            TensorLayout(
                attributes.get_dtype(),
                PageConfig(Layout::TILE),
                MemoryConfig(memory_layout, buffer_type, output_shard_spec)));
    }

    return TensorSpec(
        output_shape, TensorLayout(attributes.get_dtype(), PageConfig(Layout::TILE), attributes.memory_config));
}

BinaryNgDeviceOperation::program_factory_t BinaryNgDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

BinaryNgDeviceOperation::tensor_return_value_t BinaryNgDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor.value();
    }

    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t BinaryNgDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.storage()));

    if (input_tensor_b.has_value()) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_tensor_b->get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_tensor_b->get_storage()));

        return operation::hash_operation<BinaryNgDeviceOperation>(
            attributes,
            input_tensor_a.dtype(),
            input_tensor_a.memory_config(),
            input_tensor_b->dtype(),
            input_tensor_b->memory_config());
    }

    return operation::hash_operation<BinaryNgDeviceOperation>(
        attributes, input_tensor_a.dtype(), input_tensor_a.memory_config());
}

bool BinaryNgDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<BinaryNgDeviceOperation::operation_attributes_t, BinaryNgDeviceOperation::tensor_args_t>
BinaryNgDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    auto subtile_broadcast_type = get_subtile_broadcast_type(
        input_tensor_a.logical_shape()[-2],
        input_tensor_a.logical_shape()[-1],
        input_tensor_b.logical_shape()[-2],
        input_tensor_b.logical_shape()[-1]);

    DataType dtype_a = input_tensor_a.dtype();
    DataType dtype_b = input_tensor_b.dtype();
    bool is_sfpu_op = (utils::is_binary_sfpu_op(binary_op_type, dtype_a, dtype_b));
    bool is_quant_op = utils::is_quant_op(binary_op_type);
    return {
        operation_attributes_t{
            binary_op_type,
            {lhs_activations.begin(), lhs_activations.end()},
            {rhs_activations.begin(), rhs_activations.end()},
            {post_activations.begin(), post_activations.end()},
            std::nullopt,
            memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config()),
            input_tensor_a.dtype(),
            output_dtype,
            get_worker_grid(input_tensor_a, &input_tensor_b, output_tensor),
            std::nullopt,
            subtile_broadcast_type,
            is_sfpu_op,
            is_quant_op},
        tensor_args_t{input_tensor_a, input_tensor_b, std::move(output_tensor)}};
}

std::tuple<BinaryNgDeviceOperation::operation_attributes_t, BinaryNgDeviceOperation::tensor_args_t>
BinaryNgDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> post_activations) {
    DataType dtype_a = input_tensor_a.dtype();
    bool is_sfpu_op = (utils::is_binary_sfpu_op(binary_op_type, dtype_a, dtype_a));
    bool is_quant_op = utils::is_quant_op(binary_op_type);
    return {
        operation_attributes_t{
            binary_op_type,
            {lhs_activations.begin(), lhs_activations.end()},
            {rhs_activations.begin(), rhs_activations.end()},
            {post_activations.begin(), post_activations.end()},
            scalar,
            memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config()),
            input_tensor_a.dtype(),
            output_dtype,
            get_worker_grid(input_tensor_a, nullptr, output_tensor),
            std::nullopt,
            SubtileBroadcastType::NONE,
            is_sfpu_op,
            is_quant_op},
        tensor_args_t{input_tensor_a, std::nullopt, std::move(output_tensor)}};
}

}  // namespace ttnn::operations::binary_ng
