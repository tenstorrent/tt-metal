// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ternary {

// Helper function to get worker grid for sharded tensors
CoreRangeSet get_worker_grid_for_sharded(
    const Tensor& predicate,
    const Tensor* value_true,
    const Tensor* value_false,
    const std::optional<Tensor>& output_tensor) {
    auto get_tensor_grid = [](const Tensor& tensor) -> CoreRangeSet {
        const auto& grid = tensor.shard_spec()->grid;
        auto device = tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
        __builtin_unreachable();
    };

    if (predicate.is_sharded()) {
        return get_tensor_grid(predicate);
    } else if (value_true && value_true->is_sharded()) {
        return get_tensor_grid(*value_true);
    } else if (value_false && value_false->is_sharded()) {
        return get_tensor_grid(*value_false);
    } else if (output_tensor.has_value() && output_tensor->is_sharded()) {
        return get_tensor_grid(*output_tensor);
    }

    auto device = predicate.device();
    return device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
}

// Helper function to validate sharding compatibility for TTT variant
void validate_sharding_ttt(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const MemoryConfig& memory_config,
    WhereBroadcastType broadcast_type) {
    bool predicate_sharded = predicate.is_sharded();
    bool value_true_sharded = value_true.is_sharded();
    bool value_false_sharded = value_false.is_sharded();
    bool output_sharded = memory_config.is_sharded();

    // If any input is sharded, output must be sharded
    TT_FATAL(!predicate_sharded || output_sharded, "Predicate sharded but output not sharded");
    TT_FATAL(!value_true_sharded || output_sharded, "Value true sharded but output not sharded");
    TT_FATAL(!value_false_sharded || output_sharded, "Value false sharded but output not sharded");

    if (!output_sharded) {
        return;  // No sharding validation needed for interleaved case
    }

    // Validate that all sharded tensors have the same memory layout
    TensorMemoryLayout memory_layout = memory_config.memory_layout();
    TT_FATAL(
        predicate.memory_config().memory_layout() == memory_layout,
        "Predicate and output must have same memory layout");
    TT_FATAL(
        value_true.memory_config().memory_layout() == memory_layout,
        "Value true and output must have same memory layout");
    TT_FATAL(
        value_false.memory_config().memory_layout() == memory_layout,
        "Value false and output must have same memory layout");

    // Validate buffer type for sharded tensors
    TT_FATAL(predicate.memory_config().buffer_type() == BufferType::L1, "Sharded predicate must be in L1");
    TT_FATAL(value_true.memory_config().buffer_type() == BufferType::L1, "Sharded value true must be in L1");
    TT_FATAL(value_false.memory_config().buffer_type() == BufferType::L1, "Sharded value false must be in L1");
    TT_FATAL(memory_config.buffer_type() == BufferType::L1, "Sharded output must be in L1");

    // For non-broadcast cases, shard specs must match
    if (broadcast_type == WhereBroadcastType::NONE) {
        TT_FATAL(
            predicate.memory_config().shard_spec() == value_true.memory_config().shard_spec(),
            "Predicate and value true must have same shard spec for non-broadcast case");
        TT_FATAL(
            predicate.memory_config().shard_spec() == value_false.memory_config().shard_spec(),
            "Predicate and value false must have same shard spec for non-broadcast case");
        TT_FATAL(
            predicate.memory_config().shard_spec() == memory_config.shard_spec(),
            "All tensors must have same shard spec for non-broadcast case");
    } else {
        // For broadcast cases, all tensors must have the same orientation
        TT_FATAL(
            predicate.memory_config().shard_spec()->orientation == value_true.memory_config().shard_spec()->orientation,
            "All tensors must have same shard orientation for broadcast case");
        TT_FATAL(
            predicate.memory_config().shard_spec()->orientation ==
                value_false.memory_config().shard_spec()->orientation,
            "All tensors must have same shard orientation for broadcast case");
        TT_FATAL(
            predicate.memory_config().shard_spec()->orientation == memory_config.shard_spec()->orientation,
            "All tensors must have same shard orientation for broadcast case");
    }
}

// Helper function to compute output shape
ttnn::Shape compute_output_shape(
    const WhereDeviceOperation::operation_attributes_t& args, const WhereDeviceOperation::tensor_args_t& tensor_args) {
    auto broadcast_type = args.broadcast_type;
    auto output_shape = tensor_args.predicate.logical_shape();

    if (broadcast_type == WhereBroadcastType::NONE) {
        return output_shape;
    }

    const auto compute_broadcasted_output_ternary = [&](const auto& pred_shape,
                                                        const auto& true_shape,
                                                        const auto& false_shape) {
        const int rank_a = pred_shape.rank();
        const int rank_b = true_shape.rank();
        const int rank_c = false_shape.rank();
        const int largest_rank = std::max({rank_a, rank_b, rank_c});

        SmallVector<uint32_t> output_shape(largest_rank, 1);

        for (int i = -1; i >= -largest_rank; --i) {
            auto dim_a = (i >= -rank_a) ? pred_shape[i] : 1;
            auto dim_b = (i >= -rank_b) ? true_shape[i] : 1;
            auto dim_c = (i >= -rank_c) ? false_shape[i] : 1;

            // Find the maximum dimension size (ignoring 1s which can be broadcast)
            uint32_t max_dim = 1;
            if (dim_a != 1) {
                max_dim = std::max(max_dim, dim_a);
            }
            if (dim_b != 1) {
                max_dim = std::max(max_dim, dim_b);
            }
            if (dim_c != 1) {
                max_dim = std::max(max_dim, dim_c);
            }

            // Validate broadcasting compatibility
            bool compatible = true;
            if (dim_a != 1 && dim_a != max_dim) {
                compatible = false;
            }
            if (dim_b != 1 && dim_b != max_dim) {
                compatible = false;
            }
            if (dim_c != 1 && dim_c != max_dim) {
                compatible = false;
            }

            TT_FATAL(
                compatible,
                "Broadcasting rule violation for rank {}, dim a: {}, dim b: {}, dim c: {}",
                i,
                dim_a,
                dim_b,
                dim_c);

            // For ranks >= 6, ensure exact match (following existing pattern)
            if (i <= -6) {
                TT_FATAL(
                    dim_a == dim_b && dim_b == dim_c,
                    "Broadcasting rule violation for rank >= 6 : dim {}, Broadcast is supported up to rank 5, "
                    "dim a: {}, dim b: {}, dim c: {}",
                    i,
                    dim_a,
                    dim_b,
                    dim_c);
            }

            output_shape[i + largest_rank] = max_dim;
        }
        return ttnn::Shape(output_shape);
    };

    const auto compute_broadcasted_output_binary = [&](const auto& pred_shape, const auto& b_shape) {
        const int rank_a = pred_shape.rank();
        const int rank_b = b_shape.rank();
        const int largest_rank = std::max(rank_a, rank_b);
        SmallVector<uint32_t> output_shape(largest_rank, 1);

        for (int i = -1; i >= -largest_rank; --i) {
            auto a_dim = (i >= -rank_a) ? pred_shape[i] : 1;
            auto b_dim = (i >= -rank_b) ? b_shape[i] : 1;

            // Standard broadcasting validation for all binary cases
            TT_FATAL(
                a_dim == b_dim || a_dim == 1 || b_dim == 1,
                "Broadcasting rule violation for rank {}, dim a: {}, dim b: {}",
                i,
                a_dim,
                b_dim);

            if (i <= -6) {
                TT_FATAL(
                    a_dim == b_dim,
                    "Broadcasting rule violation for rank >= 6 : dim {}, Broadcast is supported up to rank 5, dim a: "
                    "{}, "
                    "dim b: {}",
                    i,
                    a_dim,
                    b_dim);
            }

            // Determine the resulting dimension for this axis
            uint32_t out_dim = std::max<uint32_t>(a_dim, b_dim);
            output_shape[i + largest_rank] = out_dim;
        }
        return ttnn::Shape(output_shape);
    };

    if (args.where_variant == WhereVariant::TTT) {
        auto pred_shape = tensor_args.predicate.logical_shape();
        auto true_shape = tensor_args.value_true.value().logical_shape();
        auto false_shape = tensor_args.value_false.value().logical_shape();

        output_shape = compute_broadcasted_output_ternary(pred_shape, true_shape, false_shape);
    } else if (args.where_variant == WhereVariant::TTS) {
        // Use binary function for TTS (handles both outer broadcast and column broadcast)
        output_shape = compute_broadcasted_output_binary(
            tensor_args.predicate.logical_shape(), tensor_args.value_true.value().logical_shape());
    } else if (args.where_variant == WhereVariant::TST) {
        output_shape = compute_broadcasted_output_binary(
            tensor_args.predicate.logical_shape(), tensor_args.value_false.value().logical_shape());
    }

    return output_shape;
}

DataType ttnn::operations::ternary::WhereDeviceOperation::operation_attributes_t::get_dtype() const {
    return dtype.value_or(input_dtype);
}

ttnn::operations::ternary::WhereDeviceOperation::program_factory_t
ttnn::operations::ternary::WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Support sharded tensors for TTT variant only for now
    if (args.where_variant == WhereVariant::TTT) {
        bool any_sharded = tensor_args.predicate.is_sharded() ||
                           (tensor_args.value_true.has_value() && tensor_args.value_true->is_sharded()) ||
                           (tensor_args.value_false.has_value() && tensor_args.value_false->is_sharded());
        TT_FATAL(
            !any_sharded || args.memory_config.is_sharded(),
            "For sharded TTT variant, output memory config must also be sharded");
    } else {
        TT_FATAL(
            !tensor_args.predicate.is_sharded(), "WhereDeviceOperation only supports sharded tensors for TTT variant");
    }
    return WhereProgramFactory{};
}

tt::stl::hash::hash_t ttnn::operations::ternary::WhereDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        where_variant, broadcast_type, memory_config, get_dtype(), compute_kernel_config);
}

void ttnn::operations::ternary::WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ttnn::operations::ternary::WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true_tensor = tensor_args.value_true;
    const auto& value_false_tensor = tensor_args.value_false;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto out_memory_config = args.memory_config;
    // For TTT, allow exact shape match or broadcast-compatible shapes
    auto broadcast_type = args.broadcast_type;

    if (optional_output_tensor.has_value()) {
        out_memory_config = optional_output_tensor->memory_config();
    }

    TT_FATAL(
        predicate_tensor.storage_type() == StorageType::DEVICE,
        "Where operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(predicate_tensor.storage_type()));

    TT_FATAL(
        predicate_tensor.buffer() != nullptr,
        "Operands to eltwise where need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        predicate_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Where operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(predicate_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    // Validate tensor shapes based on variant and scalar broadcast compatibility
    if (args.where_variant == WhereVariant::TTT) {
        TT_FATAL(
            value_true_tensor.has_value() && value_false_tensor.has_value(),
            "TTT variant requires both value_true and value_false tensors");

        TT_FATAL(
            ((broadcast_type != WhereBroadcastType::SCALAR_A_BCAST) &&
             (broadcast_type != WhereBroadcastType::SCALAR_B_BCAST)),
            "Unsupported broadcast type for TTT operation. scalar broadcast for TTT requires SCALAR_BCAST");

        // Add sharding validation for TTT variant
        validate_sharding_ttt(
            predicate_tensor, *value_true_tensor, *value_false_tensor, out_memory_config, broadcast_type);

    } else if (args.where_variant == WhereVariant::TTS) {
        TT_FATAL(
            value_true_tensor.has_value() && !value_false_tensor.has_value(),
            "TTS variant requires value_true tensor and value_false scalar");
        TT_FATAL(
            args.value_false_scalar.has_value(),
            "Where TTS operation requires value_false_scalar to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != WhereBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TTS operation. scalar broadcast for TTS requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");

    } else if (args.where_variant == WhereVariant::TST) {
        TT_FATAL(
            !value_true_tensor.has_value() && value_false_tensor.has_value(),
            "TST variant requires value_true scalar and value_false tensor");
        TT_FATAL(
            args.value_true_scalar.has_value(),
            "Where TST operation requires value_true_scalar to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != WhereBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TST operation. scalar broadcast for TST requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");
    }

    if (!predicate_tensor.is_sharded()) {
        TT_FATAL(
            predicate_tensor.layout() == Layout::TILE,
            "Where operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(predicate_tensor.layout()));

        TT_FATAL(
            predicate_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Where operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(predicate_tensor.memory_config().memory_layout()));
    }

    if (optional_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto optional_output_tensor_shape = optional_output_tensor.value().logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == computed_output_shape,
            "When preallocted output tensor is used, Where operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            optional_output_tensor_shape);

        if (!predicate_tensor.is_sharded()) {
            TT_FATAL(
                (optional_output_tensor.value().layout() == Layout::TILE),
                "Where operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

ttnn::operations::ternary::WhereDeviceOperation::spec_return_value_t
ttnn::operations::ternary::WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.predicate.layout();
    }

    // For sharded TTT variant, ensure output shard spec is properly set
    if (args.where_variant == WhereVariant::TTT && args.memory_config.is_sharded()) {
        const auto& predicate_shard_spec = tensor_args.predicate.memory_config().shard_spec();
        const auto& value_true_shard_spec = tensor_args.value_true.value().memory_config().shard_spec();
        const auto& value_false_shard_spec = tensor_args.value_false.value().memory_config().shard_spec();

        // For non-broadcast cases, use predicate's shard spec
        if (args.broadcast_type == WhereBroadcastType::NONE) {
            TT_FATAL(
                predicate_shard_spec == value_true_shard_spec,
                "Predicate and value_true must have same shard spec for non-broadcast TTT");
            TT_FATAL(
                predicate_shard_spec == value_false_shard_spec,
                "Predicate and value_false must have same shard spec for non-broadcast TTT");
        }

        // Use predicate's shard spec for output (most common case)
        MemoryConfig output_memory_config = args.memory_config;
        if (args.memory_config.shard_spec().has_value()) {
            output_memory_config = MemoryConfig(
                args.memory_config.memory_layout(),
                args.memory_config.buffer_type(),
                args.memory_config.shard_spec().value());
        } else {
            output_memory_config = MemoryConfig(
                args.memory_config.memory_layout(), args.memory_config.buffer_type(), *predicate_shard_spec);
        }

        return TensorSpec(
            compute_output_shape(args, tensor_args),
            TensorLayout(args.get_dtype(), output_layout, output_memory_config));
    }

    return TensorSpec(
        compute_output_shape(args, tensor_args),
        tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

ttnn::operations::ternary::WhereDeviceOperation::tensor_return_value_t
ttnn::operations::ternary::WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.predicate.device());
}

tt::stl::hash::hash_t ttnn::operations::ternary::WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true = tensor_args.value_true;
    const auto& value_false = tensor_args.value_false;
    const auto& predicate_shape = predicate_tensor.padded_shape();
    WhereVariant variant = args.where_variant;

    auto program_factory = select_program_factory(args, tensor_args);

    // For TTT variant with sharding, include all memory configs in hash
    if (variant == WhereVariant::TTT && args.memory_config.is_sharded()) {
        return tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.value().dtype(),
            value_true.value().memory_config(),
            value_false.value().dtype(),
            value_false.value().memory_config(),
            predicate_shape.volume());
    }

    tt::stl::hash::hash_t hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
        args,
        program_factory.index(),
        predicate_tensor.dtype(),
        predicate_tensor.memory_config(),
        predicate_shape.volume());

    if (variant == WhereVariant::TTT) {
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.value().dtype(),
            value_true.value().memory_config(),
            value_false.value().dtype(),
            value_false.value().memory_config(),
            predicate_shape.volume());
    } else if (variant == WhereVariant::TTS) {
        // For TTS, include the scalar value in hash
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.value().dtype(),
            value_true.value().memory_config(),
            predicate_shape.volume());
    } else if (variant == WhereVariant::TST) {
        // For TST, include the scalar value in hash
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_false.value().dtype(),
            value_false.value().memory_config(),
            predicate_shape.volume());
    }

    return hash;
}

bool ttnn::operations::ternary::WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<
    ttnn::operations::ternary::WhereDeviceOperation::operation_attributes_t,
    ttnn::operations::ternary::WhereDeviceOperation::tensor_args_t>
ttnn::operations::ternary::WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Detect broadcast type for TTT variant
    WhereBroadcastType broadcast_type =
        get_broadcast_type(predicate.logical_shape(), value_true.logical_shape(), value_false.logical_shape());

    operation_attributes_t attributes{
        .where_variant = WhereVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(value_true.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_true.dtype()),
        .compute_kernel_config = std::nullopt,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = value_false,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<
    ttnn::operations::ternary::WhereDeviceOperation::operation_attributes_t,
    ttnn::operations::ternary::WhereDeviceOperation::tensor_args_t>
ttnn::operations::ternary::WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    float value_false_scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Detect broadcast type for TTS variant
    WhereBroadcastType broadcast_type = get_broadcast_type(predicate.logical_shape(), value_true.logical_shape());

    operation_attributes_t attributes{
        .where_variant = WhereVariant::TTS,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(value_true.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_true.dtype()),
        .compute_kernel_config = std::nullopt,
        .value_false_scalar = value_false_scalar,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = std::nullopt,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<
    ttnn::operations::ternary::WhereDeviceOperation::operation_attributes_t,
    ttnn::operations::ternary::WhereDeviceOperation::tensor_args_t>
ttnn::operations::ternary::WhereDeviceOperation::invoke(
    const Tensor& predicate,
    float value_true_scalar,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    WhereBroadcastType broadcast_type = get_broadcast_type(predicate.logical_shape(), value_false.logical_shape());

    operation_attributes_t attributes{
        .where_variant = WhereVariant::TST,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(value_false.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_false.dtype()),
        .compute_kernel_config = std::nullopt,
        .value_true_scalar = value_true_scalar,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = std::nullopt,
        .value_false = value_false,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

}  // namespace ttnn::operations::ternary
