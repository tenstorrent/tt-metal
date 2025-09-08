// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ternary {

DataType WhereDeviceOperation::operation_attributes_t::get_dtype() const { return dtype.value_or(input_dtype); }

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.predicate.is_sharded(), "WhereDeviceOperation is not implemented for sharded tensors");
    return WhereProgramFactory{};
}

tt::stl::hash::hash_t WhereDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        where_variant, broadcast_type, memory_config, get_dtype(), compute_kernel_config);
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true_tensor = tensor_args.value_true;
    const auto& value_false_tensor = tensor_args.value_false;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto& predicate_shape = predicate_tensor.logical_shape();

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

    TT_FATAL(
        broadcast_type != ttnn::operations::ternary::WhereBroadcastType::INVALID_BCAST,
        "Invalid broadcast type for Where device operation. Supported bcast dims for TTT: -5, -4, -3, -1, for TTS/TST: "
        "-5, -4, -3");

    // Validate tensor shapes based on variant
    if (args.where_variant == WhereVariant::TTT) {
        auto& true_shape = value_true_tensor.value().logical_shape();
        auto& false_shape = value_false_tensor.value().logical_shape();
        if (broadcast_type == ttnn::operations::ternary::WhereBroadcastType::NONE ||
            broadcast_type == ttnn::operations::ternary::WhereBroadcastType::OUTER_BCAST) {
            const bool is_W_same = (predicate_shape[-1] == true_shape[-1]) && (predicate_shape[-1] == false_shape[-1]);
            const bool is_H_same = (predicate_shape[-2] == true_shape[-2]) && (predicate_shape[-2] == false_shape[-2]);
            // Check for exact shape match as fallback
            TT_FATAL(
                (is_H_same && is_W_same),
                "Where TTT operation requires H and W to match when there is no subtile broadcast. "
                "Predicate: {}, True_tensor: {}, False_tensor: {}",
                predicate_shape,
                true_shape,
                false_shape);
        }

    } else if (args.where_variant == WhereVariant::TTS) {
        TT_FATAL(
            (predicate_tensor.logical_shape() == value_true_tensor.value().logical_shape() ||
             broadcast_type == ttnn::operations::ternary::WhereBroadcastType::OUTER_BCAST),
            "Where TTS operation requires predicate and value_true to have same shape. Predicate: {}, Value true: {}",
            predicate_tensor.logical_shape(),
            value_true_tensor.value().logical_shape());
        TT_FATAL(
            args.value_false_scalar.has_value(),
            "Where TTS operation requires value_false_scalar to be set in operation attributes");
    } else if (args.where_variant == WhereVariant::TST) {
        TT_FATAL(
            (predicate_tensor.logical_shape() == value_false_tensor.value().logical_shape() ||
             broadcast_type == ttnn::operations::ternary::WhereBroadcastType::OUTER_BCAST),
            "Where TST operation requires predicate and value_false to have same shape. Predicate: {}, Value false: {}",
            predicate_tensor.logical_shape(),
            value_false_tensor.value().logical_shape());
        TT_FATAL(
            args.value_true_scalar.has_value(),
            "Where TST operation requires value_true_scalar to be set in operation attributes");
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

TensorSpec WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.predicate.layout();
    }

    // Determine output shape based on broadcast pattern
    // For TST/TTS variants, one of the values is a scalar, so we need to handle that case

    auto broadcast_type = args.broadcast_type;

    auto output_shape = tensor_args.predicate.logical_shape();

    if (broadcast_type == WhereBroadcastType::NONE) {
        return TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
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
        output_shape = compute_broadcasted_output_binary(
            tensor_args.predicate.logical_shape(), tensor_args.value_true.value().logical_shape());
    } else if (args.where_variant == WhereVariant::TST) {
        output_shape = compute_broadcasted_output_binary(
            tensor_args.predicate.logical_shape(), tensor_args.value_false.value().logical_shape());
    }

    return TensorSpec(output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

Tensor WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.predicate.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true = tensor_args.value_true;
    const auto& value_false = tensor_args.value_false;
    const auto& predicate_shape = predicate_tensor.padded_shape();
    WhereVariant variant = args.where_variant;

    auto program_factory = select_program_factory(args, tensor_args);

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

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
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

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    float value_false_scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
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

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
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
