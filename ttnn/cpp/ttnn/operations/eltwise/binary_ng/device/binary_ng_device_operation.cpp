// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_device_operation.hpp"
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
        case SQUARED_DIFFERENCE: return a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16);
        case LOGADDEXP:
        case LOGADDEXP2:
        case LDEXP:
        case BIAS_GELU:
        case HYPOT: return (a == FLOAT32 && b == FLOAT32);
        case RSUB:
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
    const Tensor& input_tensor_a, const Tensor* input_tensor_b, const std::optional<Tensor>& output_tensor) {
    auto get_tensor_grid = [](const Tensor& tensor) -> CoreRangeSet {
        // Return the actual tensor's shard grid for worker grid selection
        return tensor.shard_spec()->grid;
    };

    auto get_full_device_grid = [](const Tensor& tensor) -> CoreRangeSet {
        auto device = tensor.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    };

    auto get_max_grid = [&get_tensor_grid, &get_full_device_grid](const Tensor& t1, const Tensor& t2) -> CoreRangeSet {
        if (!t1.is_sharded() && !t2.is_sharded()) {
            // Both interleaved - use full device grid
            return get_full_device_grid(t1);
        }
        if (!t1.is_sharded()) {
            // Only t2 is sharded - use t2's grid
            return get_tensor_grid(t2);
        }
        if (!t2.is_sharded()) {
            // Only t1 is sharded - use t1's grid
            return get_tensor_grid(t1);
        }

        // Both sharded - pick the one with more cores
        auto grid1 = t1.shard_spec()->grid;
        auto grid2 = t2.shard_spec()->grid;
        return grid1.num_cores() >= grid2.num_cores() ? get_tensor_grid(t1) : get_tensor_grid(t2);
    };

    auto get_min_grid = [&get_tensor_grid, &get_full_device_grid](const Tensor& t1, const Tensor& t2) -> CoreRangeSet {
        if (!t1.is_sharded() && !t2.is_sharded()) {
            // Both interleaved - use full device grid
            return get_full_device_grid(t1);
        }
        if (!t1.is_sharded()) {
            // Only t2 is sharded - use t2's grid
            return get_tensor_grid(t2);
        }
        if (!t2.is_sharded()) {
            // Only t1 is sharded - use t1's grid
            return get_tensor_grid(t1);
        }

        // Both sharded - pick the one with fewer cores
        auto grid1 = t1.shard_spec()->grid;
        auto grid2 = t2.shard_spec()->grid;
        return grid1.num_cores() <= grid2.num_cores() ? get_tensor_grid(t1) : get_tensor_grid(t2);
    };

    auto get_elementwise_max_grid = [&get_tensor_grid, &get_full_device_grid](
                                        const Tensor& t1, const Tensor& t2) -> CoreRangeSet {
        if (!t1.is_sharded() && !t2.is_sharded()) {
            // Both interleaved - use full device grid
            return get_full_device_grid(t1);
        }
        if (!t1.is_sharded()) {
            // Only t2 is sharded - use t2's grid
            return get_tensor_grid(t2);
        }
        if (!t2.is_sharded()) {
            // Only t1 is sharded - use t1's grid
            return get_tensor_grid(t1);
        }

        // Both sharded - compute element-wise max grid: (max(a.x, b.x), max(a.y, b.y))
        auto grid1 = t1.shard_spec()->grid;
        auto grid2 = t2.shard_spec()->grid;
        auto bbox1 = grid1.bounding_box();
        auto bbox2 = grid2.bounding_box();

        // Get grid sizes (end coordinates + 1 since coordinates are 0-based)
        uint32_t max_x = std::max(bbox1.end_coord.x, bbox2.end_coord.x);
        uint32_t max_y = std::max(bbox1.end_coord.y, bbox2.end_coord.y);

        // Create new grid from (0,0) to (max_x, max_y)
        return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(max_x, max_y))});
    };

    // Check environment variable for grid selection strategy
    static const char* grid_strategy_env = std::getenv("TT_METAL_BINARY_NG_GRID_STRATEGY");
    std::string_view strategy = grid_strategy_env ? grid_strategy_env : "current";

    if (strategy == "a_first") {
        // Prefer A, then B, then C
        if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        } else if (input_tensor_b && input_tensor_b->is_sharded()) {
            return get_tensor_grid(*input_tensor_b);
        } else if (output_tensor.has_value() && output_tensor->is_sharded()) {
            return get_tensor_grid(*output_tensor);
        }
    } else if (strategy == "b_first") {
        // Prefer B, then A, then C
        if (input_tensor_b && input_tensor_b->is_sharded()) {
            return get_tensor_grid(*input_tensor_b);
        } else if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        } else if (output_tensor.has_value() && output_tensor->is_sharded()) {
            return get_tensor_grid(*output_tensor);
        }
    } else if (strategy == "full_grid") {
        // Always use full device grid
        return get_full_device_grid(input_tensor_a);
    } else if (strategy == "half_grid") {
        // Use half of full device grid (32 cores in 4x8 layout)
        return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(3, 7))});  // 4 columns × 8 rows = 32 cores
    } else if (strategy == "max_ab") {
        // Use max of A and B grids
        if (input_tensor_b) {
            return get_max_grid(input_tensor_a, *input_tensor_b);
        } else if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        }
        // Fallback: use full device grid
        auto device = input_tensor_a.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    } else if (strategy == "min_ab") {
        // Use min of A and B grids
        if (input_tensor_b) {
            return get_min_grid(input_tensor_a, *input_tensor_b);
        } else if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        }
        // Fallback: use full device grid
        auto device = input_tensor_a.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    } else if (strategy == "max_abc") {
        // Choose compute grid based on max_ab (max of A and B)
        // Then C's shardspec core grid will be adjusted to match the compute grid
        if (input_tensor_b) {
            return get_max_grid(input_tensor_a, *input_tensor_b);
        } else if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        }
        // Fallback: use full device grid
        auto device = input_tensor_a.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    } else if (strategy == "new_grid") {
        // Element-wise max grid: (max(a.x, b.x), max(a.y, b.y))
        // Example: if a's grid is (4,8) and b's grid is (8,4), result is (8,8)
        if (input_tensor_b) {
            // if (input_tensor_a.is_sharded() && input_tensor_b->is_sharded() &&
            //     input_tensor_a.memory_config().memory_layout() == input_tensor_b->memory_config().memory_layout()) {
            return get_elementwise_max_grid(input_tensor_a, *input_tensor_b);
            //} else {
            //    return get_max_grid(input_tensor_a, *input_tensor_b);
            //}
        } else if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        }
        // Fallback: use full device grid
        auto device = input_tensor_a.device();
        return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    } else {
        // Default "current" strategy: prefer C, then A, then B
        if (output_tensor.has_value() && output_tensor->is_sharded()) {
            return get_tensor_grid(*output_tensor);
        }
        if (input_tensor_a.is_sharded()) {
            return get_tensor_grid(input_tensor_a);
        } else if (input_tensor_b && input_tensor_b->is_sharded()) {
            return get_tensor_grid(*input_tensor_b);
        }
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
    // TODO: a more generalized way to skip the hashing of an EltwiseUnaryWithParam?
    // Don't hash the quantization scale, otherwise we build the kernel for each different scale
    return tt::stl::hash::hash_objects_with_default_seed(
        binary_op_type,
        lhs_activations,
        rhs_activations,
        is_quant_op ? ttnn::SmallVector<unary::EltwiseUnaryWithParam>{} : post_activations,
        memory_config,
        get_dtype(),
        compute_kernel_config,
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

        ShardSpec output_shard_spec{CoreRangeSet(), {0, 0}};
        // Check if memory config was inherited from an input (needs adjustment)
        // or explicitly provided by user (use as-is)
        bool inherited_from_input_a =
            input_a_shard_spec.has_value() && shard_spec.has_value() && *shard_spec == *input_a_shard_spec;
        bool inherited_from_input_b =
            input_b_shard_spec.has_value() && shard_spec.has_value() && *shard_spec == *input_b_shard_spec;

        if (shard_spec.has_value() && !inherited_from_input_a && !inherited_from_input_b) {
            // User explicitly provided a shard spec that differs from both inputs - use as-is
            output_shard_spec = *shard_spec;
        } else if (input_a_shard_spec.has_value() && !inherited_from_input_b) {
            // A has a spec AND we're not using B's spec → adjust from A
            auto padded_output_shape = input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(output_shape);
            output_shard_spec =
                adjust_to_shape(*input_a_shard_spec, input_tensor_a.padded_shape(), padded_output_shape);
        } else if (input_b_shard_spec.has_value()) {
            // B has a spec (either inherited from B or fallback to B) → adjust from B
            TT_FATAL(tensor_b.has_value(), "Cannot adjust from input_b when tensor_b is not present");
            auto padded_output_shape = tensor_b->tensor_spec().tensor_layout().compute_padded_shape(output_shape);
            output_shard_spec = adjust_to_shape(*input_b_shard_spec, tensor_b->padded_shape(), padded_output_shape);
        } else {
            TT_FATAL(shard_spec.has_value(), "Sharded memory config specified but no shard spec available");
            output_shard_spec = *shard_spec;
        }

        // For max_abc strategy: replace the grid with the compute grid from max_ab
        const char* grid_strategy_env = std::getenv("TT_METAL_BINARY_NG_GRID_STRATEGY");
        std::string_view strategy = grid_strategy_env ? grid_strategy_env : "current";
        if (strategy == "max_abc") {
            // Replace the output shard spec's grid with the worker grid (compute grid from max_ab)
            output_shard_spec.grid = attributes.worker_grid;
        }

        return TensorSpec(
            output_shape,
            TensorLayout(
                attributes.get_dtype(),
                PageConfig(Layout::TILE),
                MemoryConfig(memory_layout, buffer_type, output_shard_spec)));
    }

    // If not sharded, use the memory config from input a that is interleaved
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
    const std::optional<bool>& fast_and_approximate_mode,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    std::optional<unary::ScalarVariant> scalar_value) {
    // Validate storage type for input tensors
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a.storage_type());

    TT_FATAL(
        input_tensor_b.storage_type() == StorageType::DEVICE,
        "Input tensor B must be on device, got storage type: {}",
        input_tensor_b.storage_type());

    auto subtile_broadcast_type = get_subtile_broadcast_type(
        input_tensor_a.logical_shape()[-2],
        input_tensor_a.logical_shape()[-1],
        input_tensor_b.logical_shape()[-2],
        input_tensor_b.logical_shape()[-1]);

    DataType dtype_a = input_tensor_a.dtype();
    DataType dtype_b = input_tensor_b.dtype();
    bool is_sfpu_op =
        (utils::is_binary_sfpu_op(binary_op_type, dtype_a, dtype_b, fast_and_approximate_mode.value_or(false)));
    bool is_quant_op = utils::is_quant_op(binary_op_type);
    bool is_where_op = (binary_op_type == BinaryOpType::WHERE_TTS || binary_op_type == BinaryOpType::WHERE_TST);
    return {
        operation_attributes_t{
            binary_op_type,
            {lhs_activations.begin(), lhs_activations.end()},
            {rhs_activations.begin(), rhs_activations.end()},
            {post_activations.begin(), post_activations.end()},
            scalar_value,
            memory_config.value_or(
                output_tensor.has_value()                     ? output_tensor->memory_config()
                : input_tensor_a.memory_config().is_sharded() ? input_tensor_a.memory_config()
                                                              : input_tensor_b.memory_config()),
            is_where_op ? dtype_b : dtype_a,  // TODO: For mixed dtypes we need to set this value to the appropriate
                                              // dtype depending on which LLK is meant to be used.
            output_dtype,
            [&]() {
                auto worker_grid = get_worker_grid(input_tensor_a, &input_tensor_b, output_tensor);
                // Simple logging for profiling analysis
                const char* strategy = std::getenv("TT_METAL_BINARY_NG_GRID_STRATEGY");
                if (strategy) {
                    auto grid_bbox = worker_grid.bounding_box();
                    auto grid_size = grid_bbox.grid_size();
                    fprintf(
                        stderr,
                        "WORKER_GRID: strategy=%s cores=%lu grid=%lux%lu\n",
                        strategy,
                        static_cast<unsigned long>(worker_grid.num_cores()),
                        static_cast<unsigned long>(grid_size.x),
                        static_cast<unsigned long>(grid_size.y));
                }
                return worker_grid;
            }(),
            std::nullopt,
            subtile_broadcast_type,
            is_sfpu_op,
            is_quant_op,
            is_where_op},
        tensor_args_t{input_tensor_a, input_tensor_b, output_tensor}};
}

std::tuple<BinaryNgDeviceOperation::operation_attributes_t, BinaryNgDeviceOperation::tensor_args_t>
BinaryNgDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<bool>& fast_and_approximate_mode,
    tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    std::optional<unary::ScalarVariant> scalar_value) {
    DataType dtype_a = input_tensor_a.dtype();
    bool is_sfpu_op =
        (utils::is_binary_sfpu_op(binary_op_type, dtype_a, dtype_a, fast_and_approximate_mode.value_or(false)));
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
            [&]() {
                auto worker_grid = get_worker_grid(input_tensor_a, nullptr, output_tensor);
                // Simple logging for profiling analysis
                const char* strategy = std::getenv("TT_METAL_BINARY_NG_GRID_STRATEGY");
                if (strategy) {
                    auto grid_bbox = worker_grid.bounding_box();
                    auto grid_size = grid_bbox.grid_size();
                    fprintf(
                        stderr,
                        "WORKER_GRID: strategy=%s cores=%lu grid=%lux%lu\n",
                        strategy,
                        static_cast<unsigned long>(worker_grid.num_cores()),
                        static_cast<unsigned long>(grid_size.x),
                        static_cast<unsigned long>(grid_size.y));
                }
                return worker_grid;
            }(),
            std::nullopt,
            SubtileBroadcastType::NONE,
            is_sfpu_op,
            is_quant_op,
            false},
        tensor_args_t{input_tensor_a, std::nullopt, output_tensor}};
}

}  // namespace ttnn::operations::binary_ng
