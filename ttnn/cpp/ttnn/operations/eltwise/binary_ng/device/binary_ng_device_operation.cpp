// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "binary_ng_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_ng {

namespace utils {
bool is_binary_sfpu_op(BinaryOpType val, DataType a, DataType b, bool fast_and_approximate_mode = false) {
    using enum BinaryOpType;
    using enum DataType;
    switch (val) {
        case ADD:
        case SUB:
        case MUL:
        case EQ:
        case NE:
        case LOGICAL_AND:
        case LOGICAL_OR:
        case LOGICAL_XOR:
        case SQUARED_DIFFERENCE:
        case RSUB: return a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16);
        case LOGADDEXP:
        case LOGADDEXP2:
        case LDEXP:
        case BIAS_GELU:
        case HYPOT: return (a == FLOAT32 && b == FLOAT32);
        case GT:
        case LT:
        case GE:
        case LE: return ((a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32));
        case LCM:
        case GCD: return (a == INT32 && b == INT32);
        case LEFT_SHIFT:
        case RIGHT_SHIFT:
        case LOGICAL_RIGHT_SHIFT: return ((a == INT32 || a == UINT32) && (b == INT32 || b == UINT32));
        case BITWISE_XOR:
        case BITWISE_OR:
        case BITWISE_AND: return a == b && (a == INT32 || a == UINT32 || a == UINT16);
        case DIV_FLOOR:
        case DIV_TRUNC: return (a == INT32 && b == INT32);
        case QUANT:
        case REQUANT:
        case DEQUANT:
        case MAXIMUM:
        case MINIMUM:
        case XLOGY:
        case POWER:
        case WHERE_TST:
        case WHERE_TTS: return true;
        case DIV: return !fast_and_approximate_mode || (a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32);
        default: return false;
    }
    return false;
}

bool is_quant_op(const BinaryOpType val) {
    return (val == BinaryOpType::QUANT) || (val == BinaryOpType::DEQUANT) || (val == BinaryOpType::REQUANT);
}
}  // namespace utils

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor_a,
    const Tensor* input_tensor_b,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const MemoryConfig& memory_config_actual) {
    // If sub_core_grids is provided, use it directly
    if (sub_core_grids.has_value()) {
        log_debug(tt::LogOp, "Using provided sub_core_grids for worker grid {}", sub_core_grids->str());
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

    if (output_tensor.has_value() && output_tensor->is_sharded()) {
        log_debug(tt::LogOp, "Using output tensor grid for worker grid {}", output_tensor->shard_spec()->grid.str());
        return get_tensor_grid(*output_tensor);
    }

    if (memory_config.has_value()) {
        if (memory_config->is_sharded()) {
            // Use the shard spec from memory config if provided
            const auto& shard_spec_opt = memory_config->shard_spec();
            if (shard_spec_opt.has_value()) {
                log_debug(
                    tt::LogOp, "Using memory config shard spec grid for worker grid {}", shard_spec_opt->grid.str());
                auto* device = input_tensor_a.device();
                for (const auto& sub_device_id : device->get_sub_device_ids()) {
                    const auto& sub_device_workers =
                        device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
                    if (sub_device_workers.intersects(shard_spec_opt->grid)) {
                        return sub_device_workers;
                    }
                }
            }
        } else {
            log_debug(tt::LogOp, "Memory config not sharded");
        }
    } else {
        log_debug(tt::LogOp, "No memory config provided");
    }
    if (output_tensor.has_value() || memory_config.has_value()) {
        // If output tensor or memory config is provided but not sharded, use all worker cores
        log_debug(
            tt::LogOp, "Using all worker cores of the device for worker grid, output or memory config not sharded");
        auto* device = input_tensor_a.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    }

    if (is_native_L1_sharding(
            input_tensor_a.tensor_spec(),
            input_tensor_b ? std::optional<TensorSpec>{input_tensor_b->tensor_spec()} : std::nullopt,
            memory_config_actual)) {
        if (input_tensor_a.is_sharded()) {
            log_debug(
                tt::LogOp,
                "Native L1 sharding using input tensor A grid for worker grid {}",
                input_tensor_a.shard_spec()->grid.str());
            return get_tensor_grid(input_tensor_a);
        }
        if (input_tensor_b && input_tensor_b->is_sharded()) {
            log_debug(
                tt::LogOp,
                "Native L1 sharding using input tensor B grid for worker grid {}",
                input_tensor_b->shard_spec()->grid.str());
            return get_tensor_grid(*input_tensor_b);
        }
    }
    // use all worker cores of the device
    log_debug(tt::LogOp, "Using all worker cores of the device for worker grid");
    auto* device = input_tensor_a.device();
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
    // TODO: a more generalized way to skip the hashing of an EltwiseUnaryWithParam?
    // Don't hash the quantization scale, otherwise we build the kernel for each different scale
    return tt::stl::hash::hash_objects_with_default_seed(
        binary_op_type,
        lhs_activations,
        rhs_activations,
        (is_where_op || is_quant_op) ? ttnn::SmallVector<unary::EltwiseUnaryWithParam>{} : post_activations,
        memory_config,
        get_dtype(),
        compute_kernel_config,
        sub_core_grids,
        subtile_broadcast_type,
        is_sfpu,
        is_quant_op,
        is_where_op);
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

    // Validate storage type for input tensors
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
    if (attributes.binary_op_type == BinaryOpType::WHERE_TST || attributes.binary_op_type == BinaryOpType::WHERE_TTS) {
        TT_FATAL(
            input_tensor_b.has_value() && attributes.scalar.has_value(), "Input tensor B and scalar value must be set");
    } else {
        TT_FATAL(
            input_tensor_b.has_value() != attributes.scalar.has_value(), "Either the tensor b or scalar should be set");
    }

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
    auto output_dtype = attributes.get_dtype();

    // Integer division results in FP32 outputs.
    if (attributes.binary_op_type == BinaryOpType::DIV && input_tensor_a.dtype() == DataType::INT32) {
        if (!tensor_b.has_value() || tensor_b->dtype() == DataType::INT32) {
            output_dtype = DataType::FLOAT32;
        }
    }

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
        auto shard_spec_opt = attributes.memory_config.shard_spec();

        // If no shard spec is provided, inherit from input tensor
        if (!shard_spec_opt.has_value()) {
            if (input_tensor_a.is_sharded()) {
                // Adjust shard spec from input A to match output shape
                const auto& padded_a_shape = input_tensor_a.padded_shape();
                const auto& padded_out_shape =
                    input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(output_shape);
                shard_spec_opt = ttnn::operations::binary_ng::adjust_to_shape(
                    *input_tensor_a.memory_config().shard_spec(), padded_a_shape, padded_out_shape);
            } else if (tensor_b.has_value() && tensor_b->is_sharded()) {
                // Adjust shard spec from input B to match output shape
                const auto& padded_b_shape = tensor_b->padded_shape();
                const auto& padded_out_shape =
                    tensor_b->tensor_spec().tensor_layout().compute_padded_shape(output_shape);
                shard_spec_opt = ttnn::operations::binary_ng::adjust_to_shape(
                    *tensor_b->memory_config().shard_spec(), padded_b_shape, padded_out_shape);
            } else {
                TT_THROW("Output memory config is sharded but has no shard spec, and no input tensors are sharded");
            }
        }

        return TensorSpec(
            output_shape,
            TensorLayout(
                output_dtype, PageConfig(Layout::TILE), MemoryConfig(memory_layout, buffer_type, shard_spec_opt)));
    }

    // If not sharded, use the memory config from input a that is interleaved
    return TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::TILE), attributes.memory_config));
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
            std::holds_alternative<DeviceStorage>(input_tensor_b->storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_tensor_b->storage()));

        const auto shard_volumes = get_shard_volumes(
            input_tensor_a.tensor_spec(), input_tensor_b->tensor_spec(), compute_output_specs(attributes, tensor_args));

        return operation::hash_operation<BinaryNgDeviceOperation>(
            attributes,
            input_tensor_a.dtype(),
            input_tensor_a.memory_config(),
            input_tensor_b->dtype(),
            input_tensor_b->memory_config(),
            shard_volumes);
    }

    return operation::hash_operation<BinaryNgDeviceOperation>(
        attributes, input_tensor_a.dtype(), input_tensor_a.memory_config());
}

bool BinaryNgDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::binary_ng

namespace ttnn::prim {

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<bool>& fast_and_approximate_mode,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::binary_ng::BinaryNgDeviceOperation;
    // Validate storage type for input tensors
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a.storage_type());

    TT_FATAL(
        input_tensor_b.storage_type() == StorageType::DEVICE,
        "Input tensor B must be on device, got storage type: {}",
        input_tensor_b.storage_type());

    auto subtile_broadcast_type = ttnn::operations::binary_ng::get_subtile_broadcast_type(
        input_tensor_a.logical_shape()[-2],
        input_tensor_a.logical_shape()[-1],
        input_tensor_b.logical_shape()[-2],
        input_tensor_b.logical_shape()[-1]);

    DataType dtype_a = input_tensor_a.dtype();
    DataType dtype_b = input_tensor_b.dtype();
    bool is_sfpu_op = (ttnn::operations::binary_ng::utils::is_binary_sfpu_op(
        binary_op_type, dtype_a, dtype_b, fast_and_approximate_mode.value_or(false)));
    bool is_quant_op = ttnn::operations::binary_ng::utils::is_quant_op(binary_op_type);
    bool is_where_op =
        (binary_op_type == ttnn::operations::binary_ng::BinaryOpType::WHERE_TTS ||
         binary_op_type == ttnn::operations::binary_ng::BinaryOpType::WHERE_TST);

    MemoryConfig mem_config_actual = input_tensor_a.memory_config();
    if (input_tensor_a.is_sharded()) {
        mem_config_actual =
            operations::binary_ng::compute_mem_config_actual(input_tensor_a, input_tensor_b.logical_shape());
    }
    if (!memory_config.has_value() && !output_tensor.has_value()) {
        if (input_tensor_b.memory_config().is_sharded()) {
            // if a is interleaved but in L1 (not DRAM), still use a's memory config
            if (!input_tensor_a.memory_config().is_sharded()) {
                if (input_tensor_a.memory_config().buffer_type() == BufferType::DRAM) {
                    mem_config_actual = operations::binary_ng::compute_mem_config_actual(
                        input_tensor_b, input_tensor_a.logical_shape());
                    log_debug(
                        tt::LogOp,
                        "BinaryNgDeviceOperation: Using memory config from input tensor B since it is sharded");
                }
            } else if (input_tensor_b.shard_spec()->grid.size() > input_tensor_a.shard_spec()->grid.size()) {
                mem_config_actual =
                    operations::binary_ng::compute_mem_config_actual(input_tensor_b, input_tensor_a.logical_shape());
                log_debug(
                    tt::LogOp,
                    "BinaryNgDeviceOperation: Using memory config from input tensor B since it has a larger shard "
                    "grid");
            }
        }
    } else if (memory_config.has_value()) {
        mem_config_actual = *memory_config;
        // If the provided memory config is sharded but doesn't have a shard spec, inherit from input
        if (mem_config_actual.is_sharded() && !mem_config_actual.shard_spec().has_value()) {
            if (input_tensor_a.is_sharded()) {
                mem_config_actual =
                    operations::binary_ng::compute_mem_config_actual(input_tensor_a, input_tensor_b.logical_shape());
                log_debug(tt::LogOp, "BinaryNgDeviceOperation: Inheriting shard spec from input tensor A");
            } else if (input_tensor_b.is_sharded()) {
                mem_config_actual =
                    operations::binary_ng::compute_mem_config_actual(input_tensor_b, input_tensor_a.logical_shape());
                log_debug(tt::LogOp, "BinaryNgDeviceOperation: Inheriting shard spec from input tensor B");
            } else {
                TT_THROW("Output memory config is sharded but has no shard spec, and no input tensors are sharded");
            }
        } else {
            log_debug(tt::LogOp, "BinaryNgDeviceOperation: Using provided memory config from function argument");
        }
    } else {
        mem_config_actual = output_tensor->memory_config();
        log_debug(tt::LogOp, "BinaryNgDeviceOperation: Using memory config from output tensor since it is provided");
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        binary_op_type,
        {lhs_activations.begin(), lhs_activations.end()},
        {rhs_activations.begin(), rhs_activations.end()},
        {post_activations.begin(), post_activations.end()},
        scalar_value,
        mem_config_actual,
        is_where_op ? dtype_b : dtype_a,  // TODO: For mixed dtypes we need to set this value to the appropriate
                                          // dtype depending on which LLK is meant to be used.
        output_dtype,
        ttnn::operations::binary_ng::get_worker_grid(
            input_tensor_a, &input_tensor_b, output_tensor, memory_config, sub_core_grids, mem_config_actual),
        std::nullopt,
        sub_core_grids,
        subtile_broadcast_type,
        is_sfpu_op,
        is_quant_op,
        is_where_op};

    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b, output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a,
    float scalar,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<bool>& fast_and_approximate_mode,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    std::optional<ttnn::operations::unary::ScalarVariant> /*scalar_value*/,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::binary_ng::BinaryNgDeviceOperation;
    DataType dtype_a = input_tensor_a.dtype();
    bool is_sfpu_op = (ttnn::operations::binary_ng::utils::is_binary_sfpu_op(
        binary_op_type, dtype_a, dtype_a, fast_and_approximate_mode.value_or(false)));
    bool is_quant_op = ttnn::operations::binary_ng::utils::is_quant_op(binary_op_type);
    MemoryConfig mem_config_actual = memory_config.value_or(
        output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config());

    auto operation_attributes = OperationType::operation_attributes_t{
        binary_op_type,
        {lhs_activations.begin(), lhs_activations.end()},
        {rhs_activations.begin(), rhs_activations.end()},
        {post_activations.begin(), post_activations.end()},
        scalar,
        mem_config_actual,
        input_tensor_a.dtype(),
        output_dtype,
        ttnn::operations::binary_ng::get_worker_grid(
            input_tensor_a, nullptr, output_tensor, memory_config, sub_core_grids, mem_config_actual),
        std::nullopt,
        sub_core_grids,
        ttnn::operations::binary_ng::SubtileBroadcastType::NONE,
        is_sfpu_op,
        is_quant_op,
        false};
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, std::nullopt, output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
