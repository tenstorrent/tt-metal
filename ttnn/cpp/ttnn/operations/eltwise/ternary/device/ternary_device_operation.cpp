// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ternary_op_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;
namespace ttnn::operations::ternary {

static ttnn::Shape compute_broadcasted_output_ternary(
    const ttnn::Shape& a_shape, const ttnn::Shape& b_shape, const ttnn::Shape& c_shape) {
    const int rank_a = a_shape.rank();
    const int rank_b = b_shape.rank();
    const int rank_c = c_shape.rank();
    const int largest_rank = std::max({rank_a, rank_b, rank_c});

    SmallVector<uint32_t> output_shape(largest_rank, 1);

    for (int i = -1; i >= -largest_rank; --i) {
        auto dim_a = (i >= -rank_a) ? a_shape[i] : 1;
        auto dim_b = (i >= -rank_b) ? b_shape[i] : 1;
        auto dim_c = (i >= -rank_c) ? c_shape[i] : 1;

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
}

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor_a,
    const Tensor* input_tensor_b,
    const Tensor* input_tensor_c,
    const std::optional<Tensor>& output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // If sub_core_grids is provided, use it directly
    if (sub_core_grids.has_value()) {
        return sub_core_grids.value();
    }

    auto get_tensor_grid = [](const Tensor& tensor) -> CoreRangeSet {
        const auto& grid = tensor.shard_spec()->grid;
        auto* device = tensor.device();
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
    }
    if (input_tensor_b && input_tensor_b->is_sharded()) {
        return get_tensor_grid(*input_tensor_b);
    }
    if (input_tensor_c && input_tensor_c->is_sharded()) {
        return get_tensor_grid(*input_tensor_c);
    }
    if (output_tensor.has_value() && output_tensor->is_sharded()) {
        return get_tensor_grid(*output_tensor);
    }

    auto* device = input_tensor_a.device();
    return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
}

static ttnn::Shape compute_broadcasted_output_binary(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    const int rank_a = a_shape.rank();
    const int rank_b = b_shape.rank();
    const int largest_rank = std::max(rank_a, rank_b);
    SmallVector<uint32_t> output_shape(largest_rank, 1);

    for (int i = -1; i >= -largest_rank; --i) {
        auto a_dim = (i >= -rank_a) ? a_shape[i] : 1;
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

        uint32_t out_dim = std::max<uint32_t>(a_dim, b_dim);
        output_shape[i + largest_rank] = out_dim;
    }
    return ttnn::Shape(output_shape);
}

static ShardSpec compute_output_shard_spec(
    const std::optional<ShardSpec>& shard_spec,
    const std::optional<ShardSpec>& input_a_shard_spec,
    const std::optional<ShardSpec>& input_b_shard_spec,
    const std::optional<ShardSpec>& input_c_shard_spec,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    const ttnn::Shape& output_shape) {
    ShardSpec output_shard_spec{CoreRangeSet(), {0, 0}};
    // Check if memory config was inherited from an input (needs adjustment)
    // or explicitly provided by user (use as-is)
    bool inherited_from_input_a =
        input_a_shard_spec.has_value() && shard_spec.has_value() && *shard_spec == *input_a_shard_spec;
    bool inherited_from_input_b =
        input_b_shard_spec.has_value() && shard_spec.has_value() && *shard_spec == *input_b_shard_spec;
    bool inherited_from_input_c =
        input_c_shard_spec.has_value() && shard_spec.has_value() && *shard_spec == *input_c_shard_spec;

    if (shard_spec.has_value() && !inherited_from_input_a && !inherited_from_input_b && !inherited_from_input_c) {
        // User explicitly provided a shard spec that differs from all inputs - use as-is
        output_shard_spec = *shard_spec;
    } else if (input_a_shard_spec.has_value() && !inherited_from_input_b && !inherited_from_input_c) {
        // A has a spec AND we're not using B's or C's spec → adjust from A
        auto padded_output_shape =
            tensor_args.input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(output_shape);
        output_shard_spec =
            adjust_to_shape(*input_a_shard_spec, tensor_args.input_tensor_a.padded_shape(), padded_output_shape);
    } else if (input_b_shard_spec.has_value() && !inherited_from_input_c) {
        // B has a spec (either inherited from B or fallback to B) AND we're not using C's spec → adjust from B
        TT_FATAL(tensor_args.input_tensor_b.has_value(), "Cannot adjust from input_b when tensor_b is not present");
        auto padded_output_shape =
            tensor_args.input_tensor_b->tensor_spec().tensor_layout().compute_padded_shape(output_shape);
        output_shard_spec =
            adjust_to_shape(*input_b_shard_spec, tensor_args.input_tensor_b->padded_shape(), padded_output_shape);
    } else if (input_c_shard_spec.has_value()) {
        // C has a spec (either inherited from C or fallback to C) → adjust from C
        TT_FATAL(tensor_args.input_tensor_c.has_value(), "Cannot adjust from input_c when tensor_c is not present");
        auto padded_output_shape =
            tensor_args.input_tensor_c->tensor_spec().tensor_layout().compute_padded_shape(output_shape);
        output_shard_spec =
            adjust_to_shape(*input_c_shard_spec, tensor_args.input_tensor_c->padded_shape(), padded_output_shape);
    } else {
        TT_FATAL(shard_spec.has_value(), "Sharded memory config specified but no shard spec available");
        output_shard_spec = *shard_spec;
    }
    return output_shard_spec;
}

DataType TernaryDeviceOperation::operation_attributes_t::get_dtype() const { return dtype.value_or(input_dtype); }

TernaryDeviceOperation::program_factory_t TernaryDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return TernaryProgramFactory{};
}

tt::stl::hash::hash_t TernaryDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        ternary_op_type,
        ternary_variant,
        broadcast_type,
        memory_config,
        get_dtype(),
        compute_kernel_config,
        sub_core_grids);
}

void TernaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TernaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;
    const auto& input_c = tensor_args.input_tensor_c;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto out_memory_config = args.memory_config;
    auto broadcast_type = args.broadcast_type;

    if (optional_output_tensor.has_value()) {
        out_memory_config = optional_output_tensor->memory_config();
    }

    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_a.storage_type()));

    TT_FATAL(
        input_a.buffer() != nullptr,
        "Operands to eltwise ternary operation need to be allocated in buffers on the device. Buffer is null.");

    // Validate each tensor individually
    bool input_a_sharded = input_a.memory_config().is_sharded();
    if (not input_a_sharded) {
        TT_FATAL(
            input_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input A must be either sharded or interleaved");
    }

    bool output_sharded = out_memory_config.is_sharded();
    if (not output_sharded) {
        TT_FATAL(
            out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output must be either sharded or interleaved");
    }

    // Validate tensor shapes based on variant
    if (args.ternary_variant == TernaryVariant::TTT) {
        TT_FATAL(input_b.has_value() && input_c.has_value(), "TTT variant requires both input_b and input_c tensors");

        // Validate input_b (true tensor)
        bool input_b_sharded = input_b->memory_config().is_sharded();
        if (not input_b_sharded) {
            TT_FATAL(
                input_b->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input B must be either sharded or interleaved");
        }

        // Validate input_c (false tensor)
        bool input_c_sharded = input_c->memory_config().is_sharded();
        if (not input_c_sharded) {
            TT_FATAL(
                input_c->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input C must be either sharded or interleaved");
        }

        TT_FATAL(
            ((broadcast_type != TernaryBroadcastType::SCALAR_A_BCAST) &&
             (broadcast_type != TernaryBroadcastType::SCALAR_B_BCAST)),
            "Unsupported broadcast type for TTT operation. scalar broadcast for TTT requires SCALAR_BCAST");

    } else if (args.ternary_variant == TernaryVariant::TTS) {
        TT_FATAL(input_b.has_value() && !input_c.has_value(), "TTS variant requires input_b tensor and input_c scalar");
        TT_FATAL(
            args.scalar_input_b.has_value(),
            "Ternary TTS operation requires scalar_input_b to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != TernaryBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TTS operation. scalar broadcast for TTS requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");

    } else if (args.ternary_variant == TernaryVariant::TST) {
        TT_FATAL(!input_b.has_value() && input_c.has_value(), "TST variant requires input_b scalar and input_c tensor");
        TT_FATAL(
            args.scalar_input_a.has_value(),
            "Ternary TST operation requires scalar_input_a to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != TernaryBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TST operation. scalar broadcast for TST requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");
    }

    if (!input_a.is_sharded()) {
        TT_FATAL(
            input_a.layout() == Layout::TILE,
            "Ternary operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_a.layout()));

        TT_FATAL(
            input_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Ternary operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_a.memory_config().memory_layout()));
    }

    if (optional_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto optional_output_tensor_shape = optional_output_tensor.value().logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == computed_output_shape,
            "When preallocted output tensor is used, Ternary operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            optional_output_tensor_shape);

        if (!input_a.is_sharded()) {
            TT_FATAL(
                (optional_output_tensor.value().layout() == Layout::TILE),
                "Ternary operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

TensorSpec TernaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor_a.layout();
    }

    auto broadcast_type = args.broadcast_type;
    auto output_shape = tensor_args.input_tensor_a.logical_shape();

    if (broadcast_type == TernaryBroadcastType::NONE && !args.memory_config.is_sharded()) {
        // Early return for NONE broadcast with non-sharded memory config
        return TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
    }

    if (broadcast_type != TernaryBroadcastType::NONE) {
        if (args.ternary_variant == TernaryVariant::TTT) {
            auto a_shape = tensor_args.input_tensor_a.logical_shape();
            auto b_shape = tensor_args.input_tensor_b.value().logical_shape();
            auto c_shape = tensor_args.input_tensor_c.value().logical_shape();

            output_shape = compute_broadcasted_output_ternary(a_shape, b_shape, c_shape);
        } else if (args.ternary_variant == TernaryVariant::TTS) {
            output_shape = compute_broadcasted_output_binary(
                tensor_args.input_tensor_a.logical_shape(), tensor_args.input_tensor_b.value().logical_shape());
        } else if (args.ternary_variant == TernaryVariant::TST) {
            output_shape = compute_broadcasted_output_binary(
                tensor_args.input_tensor_a.logical_shape(), tensor_args.input_tensor_c.value().logical_shape());
        }
    }

    if (args.memory_config.is_sharded()) {
        const auto& memory_layout = args.memory_config.memory_layout();
        const auto& buffer_type = args.memory_config.buffer_type();
        const auto& shard_spec = args.memory_config.shard_spec();
        const auto& input_a_shard_spec = tensor_args.input_tensor_a.memory_config().shard_spec();
        const auto& input_b_shard_spec = tensor_args.input_tensor_b.has_value()
                                             ? tensor_args.input_tensor_b->memory_config().shard_spec()
                                             : std::nullopt;
        const auto& input_c_shard_spec = tensor_args.input_tensor_c.has_value()
                                             ? tensor_args.input_tensor_c->memory_config().shard_spec()
                                             : std::nullopt;

        ShardSpec output_shard_spec = compute_output_shard_spec(
            shard_spec, input_a_shard_spec, input_b_shard_spec, input_c_shard_spec, tensor_args, output_shape);

        return TensorSpec(
            output_shape,
            TensorLayout(
                args.dtype.value(),
                PageConfig(output_layout),
                MemoryConfig(memory_layout, buffer_type, output_shard_spec)));
    }

    // If not sharded, use the memory config from attributes
    return TensorSpec(output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

Tensor TernaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t TernaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;
    const auto& input_c = tensor_args.input_tensor_c;
    const auto& a_shape = input_a.padded_shape();
    TernaryVariant variant = args.ternary_variant;

    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_a.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_a.storage()));

    auto program_factory = select_program_factory(args, tensor_args);

    tt::stl::hash::hash_t hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
        args, program_factory.index(), input_a.dtype(), input_a.memory_config(), a_shape.volume());

    if (variant == TernaryVariant::TTT) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_b->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_b->storage()));
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_c->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_c->storage()));

        const auto shard_volumes = get_shard_volumes(
            input_a.tensor_spec(),
            input_b->tensor_spec(),
            input_c->tensor_spec(),
            compute_output_specs(args, tensor_args));

        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_b.value().dtype(),
            input_b.value().memory_config(),
            input_c.value().dtype(),
            input_c.value().memory_config(),
            a_shape.volume(),
            shard_volumes);

    } else if (variant == TernaryVariant::TTS) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_b->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_b->storage()));

        const auto shard_volumes = get_shard_volumes(
            input_a.tensor_spec(), input_b->tensor_spec(), std::nullopt, compute_output_specs(args, tensor_args));

        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_b.value().dtype(),
            input_b.value().memory_config(),
            a_shape.volume(),
            shard_volumes);
    } else if (variant == TernaryVariant::TST) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_c->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_c->storage()));

        const auto shard_volumes = get_shard_volumes(
            input_a.tensor_spec(), std::nullopt, input_c->tensor_spec(), compute_output_specs(args, tensor_args));

        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_c.value().dtype(),
            input_c.value().memory_config(),
            a_shape.volume(),
            shard_volumes);
    }

    return hash;
}

bool TernaryDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::ternary

namespace ttnn::prim {

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::ternary::TernaryDeviceOperation;

    // Detect broadcast type for TTT variant
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    OperationType::operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = ttnn::operations::ternary::TernaryVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .worker_grid = ttnn::operations::ternary::get_worker_grid(
            input_a, &input_b, &input_c, optional_output_tensor, sub_core_grids),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .sub_core_grids = sub_core_grids,
        .scalar_input_a = std::nullopt,
        .scalar_input_b = std::nullopt,
    };

    OperationType::tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(attributes, args);
}

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::ternary::TernaryDeviceOperation;

    // This variant is only for operations that need a scalar parameter with TTT variant
    TT_FATAL(
        op_type == ttnn::operations::ternary::TernaryOpType::ADDCMUL,
        "This variant with scalar parameter is only supported for ADDCMUL operation");

    // Detect broadcast type for TTT variant
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    OperationType::operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = ttnn::operations::ternary::TernaryVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .worker_grid = ttnn::operations::ternary::get_worker_grid(
            input_a, &input_b, &input_c, optional_output_tensor, sub_core_grids),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .sub_core_grids = sub_core_grids,
        .scalar_input_a = scalar,  // Reuse scalar_input_a for ADDCMUL scalar value
        .scalar_input_b = std::nullopt,
    };

    OperationType::tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(attributes, args);
}

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    float scalar_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::ternary::TernaryDeviceOperation;

    // Detect broadcast type for TTS variant
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(input_a.logical_shape(), input_b.logical_shape());

    OperationType::operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = ttnn::operations::ternary::TernaryVariant::TTS,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .worker_grid = ttnn::operations::ternary::get_worker_grid(
            input_a, &input_b, nullptr, optional_output_tensor, sub_core_grids),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .sub_core_grids = sub_core_grids,
        .scalar_input_b = scalar_c,
    };

    OperationType::tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = std::nullopt,
        .optional_output_tensor = optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(attributes, args);
}

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    float scalar_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::ternary::TernaryDeviceOperation;

    ttnn::operations::ternary::TernaryBroadcastType broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(input_a.logical_shape(), input_c.logical_shape());

    OperationType::operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = ttnn::operations::ternary::TernaryVariant::TST,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_c.memory_config()),
        .input_dtype = input_a.dtype(),
        .worker_grid = ttnn::operations::ternary::get_worker_grid(
            input_a, nullptr, &input_c, optional_output_tensor, sub_core_grids),
        .dtype = output_dtype.value_or(input_c.dtype()),
        .compute_kernel_config = std::nullopt,
        .sub_core_grids = sub_core_grids,
        .scalar_input_a = scalar_b,
    };

    OperationType::tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = std::nullopt,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(attributes, args);
}

}  // namespace ttnn::prim
