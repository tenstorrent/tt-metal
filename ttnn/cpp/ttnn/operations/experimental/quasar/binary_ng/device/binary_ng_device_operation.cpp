// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_device_operation.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "binary_ng_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <cmath>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::quasar::binary_ng {

namespace utils {
bool is_binary_sfpu_op(BinaryOpType val, DataType a, DataType b, bool fast_and_approximate_mode = false) {
    using enum BinaryOpType;
    using enum DataType;
    switch (val) {
        case ADD:
        case SUB:
        case LOGICAL_AND:
        case LOGICAL_OR:
        case LOGICAL_XOR:
        case SQUARED_DIFFERENCE:
        case RSUB: return a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16);
        case MUL:
            return !fast_and_approximate_mode || (a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16));
        case DIV: return !fast_and_approximate_mode || (a == FLOAT32 && b == FLOAT32) || (a == INT32 && b == INT32);
        case LOGADDEXP:
        case LOGADDEXP2:
        case LDEXP:
        case BIAS_GELU:
        case HYPOT: return (a == FLOAT32 && b == FLOAT32);
        case EQ:
        case NE:
        case GT:
        case LT:
        case GE:
        case LE: return a == b && (a == FLOAT32 || a == BFLOAT16 || a == INT32 || a == UINT16 || a == UINT32);
        case LCM:
        case GCD: return (a == INT32 && b == INT32);
        case LEFT_SHIFT:
        case RIGHT_SHIFT:
            return ((a == INT32 || a == UINT32 || a == UINT16) && (b == INT32 || b == UINT32 || b == UINT16));
        case LOGICAL_RIGHT_SHIFT: return ((a == INT32 || a == UINT32) && (b == INT32 || b == UINT32));
        case BITWISE_XOR:
        case BITWISE_OR:
        case BITWISE_AND: return a == b && (a == INT32 || a == UINT32 || a == UINT16);
        case DIV_FLOOR:
        case DIV_TRUNC:
        case REMAINDER:
        case FMOD:
        case QUANT:
        case REQUANT:
        case DEQUANT:
        case MAXIMUM:
        case MINIMUM:
        case XLOGY:
        case ATAN2:
        case POWER:
        case WHERE_TST:
        case WHERE_TTS:
        case ISCLOSE: return true;
        default: return false;
    }
    return false;
}

bool is_quant_op(const BinaryOpType val) {
    return (val == BinaryOpType::QUANT) || (val == BinaryOpType::DEQUANT) || (val == BinaryOpType::REQUANT);
}

ShardSpec generate_shard_spec_all_cores(
    const Tensor& input_tensor_a, const Shape& padded_out_shape, const TensorMemoryLayout& memory_layout) {
    // Generate shard spec using all worker cores
    auto* device = input_tensor_a.device();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    auto all_cores = CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    uint32_t num_cores = all_cores.num_cores();

    // Calculate squeezed tensor height (all dims except last) and width (last dim)
    uint32_t tensor_height = 1;
    for (int i = 0; i < static_cast<int>(padded_out_shape.rank()) - 1; ++i) {
        tensor_height *= padded_out_shape[i];
    }
    uint32_t tensor_width = padded_out_shape[-1];

    // Calculate shard shape based on memory layout (must be tile-aligned for TILE layout)
    std::array<uint32_t, 2> shard_shape = {0, 0};
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto height_padded = tt::round_up(tensor_height, num_cores * tt::constants::TILE_HEIGHT);
        auto shard_height = tt::round_up(tt::div_up(height_padded, num_cores), tt::constants::TILE_HEIGHT);
        shard_shape = {shard_height, tensor_width};
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        auto shard_width = tt::round_up(tt::div_up(tensor_width, num_cores), tt::constants::TILE_WIDTH);
        shard_shape = {tensor_height, shard_width};
    } else {
        // BLOCK_SHARDED
        CoreCoord grid_size = all_cores.bounding_box().grid_size();
        auto height_padded = tt::round_up(tensor_height, grid_size.y * tt::constants::TILE_HEIGHT);
        auto shard_height = tt::round_up(tt::div_up(height_padded, grid_size.y), tt::constants::TILE_HEIGHT);
        auto shard_width = tt::round_up(tt::div_up(tensor_width, grid_size.x), tt::constants::TILE_WIDTH);
        shard_shape = {shard_height, shard_width};
    }
    log_debug(tt::LogOp, "BinaryNgDeviceOperation: Generated shard spec using all {} worker cores", num_cores);
    return ShardSpec(all_cores, shard_shape, ShardOrientation::ROW_MAJOR);
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

    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR,
        "First operand to eltwise binary must have TILE or ROW_MAJOR layout, got {}",
        input_tensor_a.layout());

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
        TT_FATAL(
            input_tensor_b->layout() == Layout::TILE || input_tensor_b->layout() == Layout::ROW_MAJOR,
            "Second operand to eltwise binary must have TILE or ROW_MAJOR layout, got {}",
            input_tensor_b->layout());

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
            "Broadcasting rule violation at dimension index {} (output rank {}), dim a: {}, dim b: {}",
            i,
            larger_rank,
            a_dim,
            b_dim);

        if (i <= -7) {
            TT_FATAL(
                a_dim == b_dim,
                "Broadcasting rule violation for rank >= 7 at dimension index {} (output rank {}). "
                "Broadcast is supported up to rank 6. dim a: {}, dim b: {}",
                i,
                larger_rank,
                a_dim,
                b_dim);
        }
    }
    if (attributes.binary_op_type == BinaryOpType::ISCLOSE) {
        TT_FATAL(
            std::isfinite(attributes.rtol) && attributes.rtol >= 0.0f,
            "isclose: rtol must be a finite, non-negative value, got {}",
            attributes.rtol);
        TT_FATAL(
            std::isfinite(attributes.atol) && attributes.atol >= 0.0f,
            "isclose: atol must be a finite, non-negative value, got {}",
            attributes.atol);
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
            const auto& padded_out_shape =
                input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(output_shape);
            if (input_tensor_a.is_sharded()) {
                // Adjust shard spec from input A to match output shape
                const auto& padded_a_shape = input_tensor_a.padded_shape();

                shard_spec_opt = ttnn::operations::experimental::quasar::binary_ng::adjust_to_shape(
                    *input_tensor_a.memory_config().shard_spec(), padded_a_shape, padded_out_shape);
            } else if (tensor_b.has_value() && tensor_b->is_sharded()) {
                // Adjust shard spec from input B to match output shape
                const auto& padded_b_shape = tensor_b->padded_shape();
                shard_spec_opt = ttnn::operations::experimental::quasar::binary_ng::adjust_to_shape(
                    *tensor_b->memory_config().shard_spec(), padded_b_shape, padded_out_shape);
            } else {
                shard_spec_opt = utils::generate_shard_spec_all_cores(input_tensor_a, padded_out_shape, memory_layout);
            }
        }

        return TensorSpec(
            output_shape,
            TensorLayout(
                output_dtype,
                PageConfig(attributes.output_layout),
                MemoryConfig(memory_layout, buffer_type, shard_spec_opt)));
    }

    // If not sharded, use the memory config from input a that is interleaved
    return TensorSpec(
        output_shape, TensorLayout(output_dtype, PageConfig(attributes.output_layout), attributes.memory_config));
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

bool BinaryNgDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

bool BinaryNgDeviceOperation::matches_metal_v2_slice(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // The Metal 2.0 / DataflowBuffer factory (binary_ng_metal_v2_factory.cpp) is a generic port of the
    // descriptor factory's no-broadcast tiled path. It covers the whole SubtileBroadcastType::NONE,
    // tensor-tensor, TILE slice — FPU and SFPU, any dtype (lhs/rhs must match), full lhs/rhs/post
    // activations. Each of input_a / input_b / output may INDEPENDENTLY be interleaved (DRAM or L1,
    // read/written over the NoC) OR L1-sharded (height/block/width). The DFB kernels are per-operand
    // capable (reader: SRC_SHARDED / SRC_SHARDED_B; writer: DST_SHARDED). The factory borrows
    // all-or-nothing: only when all three operands are L1-sharded on one matching grid are their resident
    // L1 shards borrowed to back the DFBs (no NoC work); otherwise — any interleaved operand, mixed
    // strategies, or a divergent grid — NONE are borrowed and every operand is read/written over the NoC
    // via a sharding-aware TensorAccessor.
    // Everything else falls through to the descriptor path; this predicate is the correctness boundary.
    // Deferred to the descriptor (not handled by the factory): tensor-scalar, where-op, quantization,
    // row-major (non-TILE), and mixed lhs/rhs dtype.
    using tt::tt_metal::BufferType;
    using tt::tt_metal::TensorMemoryLayout;

    // Tensor-tensor only (tensor-scalar routes to the descriptor).
    if (!tensor_args.input_tensor_b.has_value() || attributes.scalar.has_value()) {
        return false;
    }
    // where-op / quantization route to the descriptor.
    if (attributes.is_where_op || attributes.is_quant_op) {
        return false;
    }
    // Zero-volume (degenerate) tensors fall to the descriptor path; the v2 factory would derive an
    // empty core set, and the descriptor handles the empty case as the main op does.
    if (tensor_args.input_tensor_a.logical_shape().volume() == 0 ||
        tensor_args.input_tensor_b->logical_shape().volume() == 0) {
        return false;
    }
    // Equal shapes (no subtile broadcast).
    if (attributes.subtile_broadcast_type != SubtileBroadcastType::NONE) {
        return false;
    }
    // TILE layout (32x32) in and out (row-major routes to the descriptor).
    if (attributes.input_layout_a != Layout::TILE || attributes.input_layout_b != Layout::TILE ||
        attributes.output_layout != Layout::TILE) {
        return false;
    }
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = *tensor_args.input_tensor_b;
    const auto& a_tile = a.tensor_spec().tile();
    const auto& b_tile = b.tensor_spec().tile();
    if (a_tile.get_height() != tt::constants::TILE_HEIGHT || a_tile.get_width() != tt::constants::TILE_WIDTH ||
        b_tile.get_height() != tt::constants::TILE_HEIGHT || b_tile.get_width() != tt::constants::TILE_WIDTH) {
        return false;
    }

    // lhs and rhs must share a data format. The compute kernel switches the unpacker between the two
    // operands without a per-operand data-format reconfig (on Quasar copy_tile_to_dst_init_short_with_dt
    // is a no-op, so the WH/BH format reconfig it performs is absent), so a differing rhs format would be
    // unpacked using the lhs format. Mixed-dtype lhs/rhs therefore falls to the descriptor path.
    if (a.dtype() != b.dtype()) {
        return false;
    }

    // Per-operand layout: each of a, b, and the output may INDEPENDENTLY be interleaved OR L1-sharded.
    // The decisive per-operand property the factory acts on is BORROWED (resident L1 shard backs the
    // DFB) vs NoC-READ (sharding-aware TensorAccessor) — not "sharded" — so mixed combinations are
    // admitted. The only restriction on a sharded operand is that it be L1-sharded with a tiled layout
    // (height/block/width); a sharded DRAM tensor or a non-tiled sharded layout is not handled here.
    auto is_l1_sharded_tiled = [](const tt::tt_metal::MemoryConfig& mc) {
        return mc.is_sharded() && mc.buffer_type() == BufferType::L1 &&
               (mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
                mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);
    };
    // A sharded operand must be valid L1-sharded-tiled AND actually carry a shard spec (a sharded
    // memory_config without a shard spec is rejected; the factory needs the spec to borrow / place).
    auto sharded_operand_ok = [&](const tt::tt_metal::MemoryConfig& mc, bool has_shard_spec) {
        if (!mc.is_sharded()) {
            return true;  // interleaved is always fine (NoC-read path)
        }
        return is_l1_sharded_tiled(mc) && has_shard_spec;
    };
    if (!sharded_operand_ok(a.memory_config(), a.shard_spec().has_value()) ||
        !sharded_operand_ok(b.memory_config(), b.shard_spec().has_value()) ||
        !sharded_operand_ok(attributes.memory_config, attributes.memory_config.shard_spec().has_value())) {
        return false;
    }
    // When a pre-allocated output tensor is supplied, validate its memory_config the same per-operand
    // way (its shard spec, if any, must be L1-sharded-tiled and present).
    if (tensor_args.output_tensor.has_value()) {
        const auto& out_mc = tensor_args.output_tensor->memory_config();
        if (!sharded_operand_ok(out_mc, tensor_args.output_tensor->shard_spec().has_value())) {
            return false;
        }
    }

    return true;
}

BinaryNgDeviceOperation::program_factory_t BinaryNgDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // DFB / Metal 2.0 is the default for the matching slice (arch-portable: CB-backed on WH/BH,
    // overlay-backed on Quasar). Everything else uses the descriptor path.
    if (matches_metal_v2_slice(attributes, tensor_args)) {
        return ProgramFactoryMetalV2{};
    }
    return ProgramFactory{};
}

}  // namespace ttnn::operations::experimental::quasar::binary_ng

namespace ttnn::prim::qsr {

ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::operations::experimental::quasar::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<bool>& fast_and_approximate_mode,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    float rtol,
    float atol,
    bool equal_nan) {
    using OperationType = ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation;

    // Validate storage type for input tensors
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a.storage_type());

    TT_FATAL(
        input_tensor_b.storage_type() == StorageType::DEVICE,
        "Input tensor B must be on device, got storage type: {}",
        input_tensor_b.storage_type());

    // Resolve sub_device_id to sub_core_grids if provided (after device validation)
    auto resolved_sub_core_grids = sub_core_grids;
    if (sub_device_id.has_value()) {
        TT_FATAL(!sub_core_grids.has_value(), "Cannot specify both sub_core_grids and sub_device_id");
        auto* device = input_tensor_a.device();
        resolved_sub_core_grids =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    }

    auto subtile_broadcast_type = ttnn::operations::experimental::quasar::binary_ng::get_subtile_broadcast_type(
        input_tensor_a.logical_shape()[-2],
        input_tensor_a.logical_shape()[-1],
        input_tensor_b.logical_shape()[-2],
        input_tensor_b.logical_shape()[-1]);

    DataType dtype_a = input_tensor_a.dtype();
    DataType dtype_b = input_tensor_b.dtype();
    bool is_sfpu_op = (ttnn::operations::experimental::quasar::binary_ng::utils::is_binary_sfpu_op(
        binary_op_type, dtype_a, dtype_b, fast_and_approximate_mode.value_or(false)));
    bool is_quant_op = ttnn::operations::experimental::quasar::binary_ng::utils::is_quant_op(binary_op_type);
    bool is_where_op =
        (binary_op_type == ttnn::operations::experimental::quasar::binary_ng::BinaryOpType::WHERE_TTS ||
         binary_op_type == ttnn::operations::experimental::quasar::binary_ng::BinaryOpType::WHERE_TST);

    auto output_layout = output_tensor ? output_tensor->layout() : Layout::TILE;
    if (!output_tensor.has_value() && input_tensor_a.layout() == Layout::ROW_MAJOR &&
        input_tensor_b.layout() == Layout::ROW_MAJOR) {
        output_layout = Layout::ROW_MAJOR;
    }

    MemoryConfig mem_config_actual = input_tensor_a.memory_config();
    if (input_tensor_a.is_sharded()) {
        mem_config_actual = operations::experimental::quasar::binary_ng::compute_mem_config_actual(
            input_tensor_a, input_tensor_b.logical_shape());
    }
    if (!memory_config.has_value() && !output_tensor.has_value()) {
        if (input_tensor_b.memory_config().is_sharded()) {
            // if a is interleaved but in L1 (not DRAM), still use a's memory config
            if (!input_tensor_a.memory_config().is_sharded()) {
                if (input_tensor_a.memory_config().buffer_type() == BufferType::DRAM) {
                    mem_config_actual = operations::experimental::quasar::binary_ng::compute_mem_config_actual(
                        input_tensor_b, input_tensor_a.logical_shape());
                    log_debug(
                        tt::LogOp,
                        "BinaryNgDeviceOperation: Using memory config from input tensor B since it is sharded");
                }
            } else if (input_tensor_b.shard_spec()->num_cores() > input_tensor_a.shard_spec()->num_cores()) {
                mem_config_actual = operations::experimental::quasar::binary_ng::compute_mem_config_actual(
                    input_tensor_b, input_tensor_a.logical_shape());
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
                mem_config_actual = operations::experimental::quasar::binary_ng::compute_mem_config_actual(
                    input_tensor_a, input_tensor_b.logical_shape());
                log_debug(tt::LogOp, "BinaryNgDeviceOperation: Inheriting shard spec from input tensor A");
            } else if (input_tensor_b.is_sharded()) {
                mem_config_actual = operations::experimental::quasar::binary_ng::compute_mem_config_actual(
                    input_tensor_b, input_tensor_a.logical_shape());
                log_debug(tt::LogOp, "BinaryNgDeviceOperation: Inheriting shard spec from input tensor B");
            } else {
                auto memory_layout = mem_config_actual.memory_layout();
                auto output_shape = operations::experimental::quasar::binary_ng::compute_broadcasted_output(
                    input_tensor_a.logical_shape(), input_tensor_b.logical_shape());
                const auto& padded_out_shape =
                    input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(output_shape);
                mem_config_actual = MemoryConfig(
                    memory_layout,
                    mem_config_actual.buffer_type(),
                    operations::experimental::quasar::binary_ng::utils::generate_shard_spec_all_cores(
                        input_tensor_a, padded_out_shape, memory_layout));
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
        ttnn::operations::experimental::quasar::binary_ng::get_worker_grid(
            input_tensor_a, &input_tensor_b, output_tensor, memory_config, resolved_sub_core_grids, mem_config_actual),
        std::nullopt,
        resolved_sub_core_grids,
        sub_device_id,
        subtile_broadcast_type,
        is_sfpu_op,
        is_quant_op,
        is_where_op,
        rtol,
        atol,
        equal_nan,
        input_tensor_a.layout(),
        input_tensor_b.layout(),
        output_layout};

    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b, output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a,
    ttnn::operations::unary::ScalarVariant scalar,
    ttnn::operations::experimental::quasar::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<bool>& fast_and_approximate_mode,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    std::optional<ttnn::operations::unary::ScalarVariant> /*scalar_value*/,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    using OperationType = ttnn::operations::experimental::quasar::binary_ng::BinaryNgDeviceOperation;

    // Validate storage type
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE,
        "Input tensor A must be on device, got storage type: {}",
        input_tensor_a.storage_type());

    // Resolve sub_device_id to sub_core_grids if provided (after device validation)
    auto resolved_sub_core_grids = sub_core_grids;
    if (sub_device_id.has_value()) {
        TT_FATAL(!sub_core_grids.has_value(), "Cannot specify both sub_core_grids and sub_device_id");
        auto* device = input_tensor_a.device();
        resolved_sub_core_grids =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    }

    DataType dtype_a = input_tensor_a.dtype();
    bool is_sfpu_op = (ttnn::operations::experimental::quasar::binary_ng::utils::is_binary_sfpu_op(
        binary_op_type, dtype_a, dtype_a, fast_and_approximate_mode.value_or(false)));
    bool is_quant_op = ttnn::operations::experimental::quasar::binary_ng::utils::is_quant_op(binary_op_type);
    MemoryConfig mem_config_actual = memory_config.value_or(
        output_tensor.has_value() ? output_tensor->memory_config() : input_tensor_a.memory_config());

    auto output_layout = output_tensor ? output_tensor->layout() : Layout::TILE;
    if (!output_tensor.has_value() && input_tensor_a.layout() == Layout::ROW_MAJOR) {
        output_layout = Layout::ROW_MAJOR;
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        binary_op_type,
        {lhs_activations.begin(), lhs_activations.end()},
        {rhs_activations.begin(), rhs_activations.end()},
        {post_activations.begin(), post_activations.end()},
        scalar,
        mem_config_actual,
        input_tensor_a.dtype(),
        output_dtype,
        ttnn::operations::experimental::quasar::binary_ng::get_worker_grid(
            input_tensor_a, nullptr, output_tensor, memory_config, resolved_sub_core_grids, mem_config_actual),
        std::nullopt,
        resolved_sub_core_grids,
        sub_device_id,
        ttnn::operations::experimental::quasar::binary_ng::SubtileBroadcastType::NONE,
        is_sfpu_op,
        is_quant_op,
        false,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*equal_nan=*/false,
        input_tensor_a.layout(),
        Layout::INVALID,
        output_layout};

    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, std::nullopt, output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim::qsr
