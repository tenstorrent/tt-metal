// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hypot_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_ng {

namespace utils {
bool is_binary_sfpu_op(DataType a, DataType b) {
    using enum DataType;
    return (a == FLOAT32 && b == FLOAT32);
}
}  // namespace utils

tt::stl::hash::hash_t HypotNgDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        memory_config,
        get_dtype(),
        compute_kernel_config,
        subtile_broadcast_type,
        is_sfpu);
}

DataType HypotNgDeviceOperation::operation_attributes_t::get_dtype() const { return this->input_dtype; }

void HypotNgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // We don't support sharding for now
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    TT_FATAL(
        input_tensor_b.has_value() != attributes.scalar.has_value(), "Either the tensor b or scalar should be set");

    HypotNgDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

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
            validate_sharding(
                input_tensor_a.memory_config().memory_layout,
                *input_tensor_a.shard_spec(),
                input_tensor_b->memory_config().memory_layout,
                *input_tensor_b->shard_spec(),
                attributes.subtile_broadcast_type);
        }
        if (output_sharded) {
            validate_sharding(
                input_tensor_a.memory_config().memory_layout,
                *input_tensor_a.shard_spec(),
                attributes.memory_config.memory_layout,
                *attributes.memory_config.shard_spec,
                attributes.subtile_broadcast_type);
        }
    } else if (tensor_b_sharded and output_sharded) {
        validate_sharding(
            input_tensor_b->memory_config().memory_layout,
            *input_tensor_b->shard_spec(),
            attributes.memory_config.memory_layout,
            *attributes.memory_config.shard_spec,
            attributes.subtile_broadcast_type);
    }
}

void HypotNgDeviceOperation::validate_on_program_cache_hit(
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

        if (has_shard_spec and i != -1) {
            TT_FATAL(
                a_dim == b_dim,
                "Cannot broadcast sharded tensors on dims other than W, violation for rank {}, dim a: {}, dim b: {}",
                i,
                a_dim,
                b_dim);
        }
    }
}

HypotNgDeviceOperation::spec_return_value_t HypotNgDeviceOperation::compute_output_specs(
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
        auto shape = output_tensor.value().logical_shape();
        TT_FATAL(
            shape == output_shape,
            "Shape of Output tensor {} provided does not match the broadcasted output shape {}",
            shape,
            output_shape);
        return output_tensor->get_tensor_spec();
    }

    if (attributes.memory_config.is_sharded()) {
        return TensorSpec(
            output_shape, TensorLayout(attributes.get_dtype(), PageConfig(Layout::TILE), attributes.memory_config));
    }

    return TensorSpec(
        output_shape, TensorLayout(attributes.get_dtype(), PageConfig(Layout::TILE), attributes.memory_config));
}

HypotNgDeviceOperation::program_factory_t HypotNgDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return HypotProgramFactory{};
}

HypotNgDeviceOperation::tensor_return_value_t HypotNgDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t HypotNgDeviceOperation::compute_program_hash(
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

        return operation::hash_operation<HypotNgDeviceOperation>(
            attributes,
            input_tensor_a.dtype(),
            std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
            input_tensor_b->dtype(),
            std::get<DeviceStorage>(input_tensor_b->storage()).memory_config());
    }

    return operation::hash_operation<HypotNgDeviceOperation>(
        attributes, input_tensor_a.dtype(), std::get<DeviceStorage>(input_tensor_a.storage()).memory_config());
}

bool HypotNgDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<HypotNgDeviceOperation::operation_attributes_t, HypotNgDeviceOperation::tensor_args_t>
HypotNgDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    auto subtile_broadcast_type = get_subtile_broadcast_type(
        input_tensor_a.get_logical_shape()[-2],
        input_tensor_a.get_logical_shape()[-1],
        input_tensor_b.get_logical_shape()[-2],
        input_tensor_b.get_logical_shape()[-1]);

    DataType dtype1 = input_tensor_a.get_dtype();
    DataType dtype2 = input_tensor_a.get_dtype();
    bool device_check = input_tensor_a.device()->arch() != tt::ARCH::GRAYSKULL;
    bool is_sfpu_op = (utils::is_binary_sfpu_op(dtype1, dtype2) && device_check);
    return {
        operation_attributes_t{
            std::nullopt,
            memory_config.value_or(output_tensor.has_value() ? output_tensor->memory_config() : MemoryConfig{}),
            input_tensor_a.get_dtype(),
            get_worker_grid(input_tensor_a, &input_tensor_b, output_tensor),
            std::nullopt,
            subtile_broadcast_type,
            is_sfpu_op},
        tensor_args_t{input_tensor_a, input_tensor_b, std::move(output_tensor)}};
}

}  // namespace ttnn::operations::binary_ng
