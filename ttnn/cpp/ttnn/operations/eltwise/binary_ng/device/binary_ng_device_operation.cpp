// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_ng {

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
    return tt::stl::hash::hash_objects_with_default_seed(
        binary_op_type,
        lhs_activations,
        rhs_activations,
        post_activations,
        memory_config,
        get_dtype(),
        compute_kernel_config,
        subtile_broadcast_type);
}

DataType BinaryNgDeviceOperation::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
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
            *attributes.dtype == output_tensor->get_dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "First operand to eltwise binary must be tilized");

    bool tensor_a_sharded = input_tensor_a.memory_config().is_sharded();
    if (not tensor_a_sharded) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "LHS operand must be either sharded or interleaved");
    }

    bool output_sharded = attributes.memory_config.is_sharded();
    if (not output_sharded) {
        TT_FATAL(
            attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Output must be interleaved or sharded");
    }

    bool tensor_b_sharded = false;

    if (input_tensor_b.has_value()) {
        tensor_b_sharded = input_tensor_b->memory_config().is_sharded();
        TT_FATAL(
            input_tensor_a.device() == input_tensor_b->device(),
            "Operands to eltwise binary need to be on the same device!");
        TT_FATAL(input_tensor_b->get_layout() == Layout::TILE, "Second operand to eltwise binary must be tilized");

        if (not tensor_b_sharded) {
            TT_FATAL(
                input_tensor_b->memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
                "RHS operand must be either sharded or interleaved");
        }
    }

    // Validate that all shard specs match
    if (tensor_a_sharded) {
        if (tensor_b_sharded) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout == input_tensor_b->memory_config().memory_layout,
                "Operands to eltwise binary need to have the same memory layout");
            TT_FATAL(
                input_tensor_a.shard_spec().value() == input_tensor_b->shard_spec().value(),
                "Operands to eltwise binary need to have the same shard spec");
        }
        if (output_sharded) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout == attributes.memory_config.memory_layout,
                "LHS operand and output to eltwise binary need to have the same memory layout");
            TT_FATAL(
                input_tensor_a.shard_spec().value() == attributes.memory_config.shard_spec.value(),
                "LHS operand and output to eltwise binary need to have the same shard spec");
        }
    } else if (tensor_b_sharded and output_sharded) {
        TT_FATAL(
            input_tensor_b->memory_config().memory_layout == attributes.memory_config.memory_layout,
            "RHS operand and output to eltwise binary need to have the same memory layout");
        TT_FATAL(
            input_tensor_b->shard_spec().value() == attributes.memory_config.shard_spec.value(),
            "RHS operand and output to eltwise binary need to have the same shard spec");
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

    const auto& input_shape_a = input_tensor_a.get_logical_shape();
    const auto input_shape_b = input_tensor_b.has_value() ? input_tensor_b->get_logical_shape() : ttnn::Shape{1, 1};

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

        if (has_shard_spec) {
            TT_FATAL(
                a_dim == b_dim,
                "Cannot broadcast sharded tensors, violation for rank {}, dim a: {}, dim b: {}",
                i,
                a_dim,
                b_dim);
        }
    }
}

BinaryNgDeviceOperation::spec_return_value_t BinaryNgDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor->get_tensor_spec();
    }

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto input_shape_a = input_tensor_a.logical_shape();
    const auto& tensor_b = tensor_args.input_tensor_b;
    const auto input_shape_b = tensor_b.has_value() ? tensor_b->logical_shape() : ttnn::SimpleShape{};

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
        return ttnn::SimpleShape(output_shape);
    };

    auto output_shape = compute_broadcasted_output(input_shape_a, input_shape_b);

    if (attributes.memory_config.is_sharded()) {
        ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
        if (input_tensor_a.memory_config().is_sharded()) {
            shard_spec = input_tensor_a.shard_spec().value();
        } else if (tensor_b.has_value() and tensor_b->memory_config().is_sharded()) {
            shard_spec = tensor_b->shard_spec().value();
        } else {
            shard_spec = attributes.memory_config.shard_spec.value();
        }
        auto memory_config = attributes.memory_config;
        memory_config.shard_spec = shard_spec;
        return TensorSpec(output_shape, TensorLayout(attributes.get_dtype(), PageConfig(Layout::TILE), memory_config));
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
        std::holds_alternative<DeviceStorage>(input_tensor_a.get_storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.get_storage()));

    if (input_tensor_b.has_value()) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_tensor_b->get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_tensor_b->get_storage()));

        return operation::hash_operation<BinaryNgDeviceOperation>(
            attributes,
            input_tensor_a.dtype(),
            std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
            input_tensor_b->dtype(),
            std::get<DeviceStorage>(input_tensor_b->storage()).memory_config());
    }

    return operation::hash_operation<BinaryNgDeviceOperation>(
        attributes, input_tensor_a.dtype(), std::get<DeviceStorage>(input_tensor_a.storage()).memory_config());
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
    std::optional<Tensor> output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    auto subtile_broadcast_type = get_subtile_broadcast_type(
        input_tensor_a.get_logical_shape()[-2],
        input_tensor_a.get_logical_shape()[-1],
        input_tensor_b.get_logical_shape()[-2],
        input_tensor_b.get_logical_shape()[-1]);

    return {
        operation_attributes_t{
            binary_op_type,
            {lhs_activations.begin(), lhs_activations.end()},
            {rhs_activations.begin(), rhs_activations.end()},
            {post_activations.begin(), post_activations.end()},
            std::nullopt,
            memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config()),
            input_tensor_a.get_dtype(),
            output_dtype,
            get_worker_grid(input_tensor_a, &input_tensor_b, output_tensor),
            std::nullopt,
            subtile_broadcast_type},
        tensor_args_t{input_tensor_a, input_tensor_b, std::move(output_tensor)}};
}

std::tuple<BinaryNgDeviceOperation::operation_attributes_t, BinaryNgDeviceOperation::tensor_args_t>
BinaryNgDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor,
    tt::stl::Span<const unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const unary::UnaryOpType> post_activations) {
    return {
        operation_attributes_t{
            binary_op_type,
            {lhs_activations.begin(), lhs_activations.end()},
            {rhs_activations.begin(), rhs_activations.end()},
            {post_activations.begin(), post_activations.end()},
            scalar,
            memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config()),
            input_tensor_a.get_dtype(),
            output_dtype,
            get_worker_grid(input_tensor_a, nullptr, output_tensor),
            std::nullopt},
        tensor_args_t{input_tensor_a, std::nullopt, std::move(output_tensor)}};
}

}  // namespace ttnn::operations::binary_ng
