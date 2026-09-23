// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary {
namespace {

// is_integer_dtype doesn't include INT8 as of now because no unary op has native INT8 implementation yet.
bool is_integer_dtype(DataType dtype) {
    return dtype == DataType::INT32 || dtype == DataType::UINT32 || dtype == DataType::UINT16 ||
           dtype == DataType::UINT8;
}

bool is_int32(DataType dtype) { return dtype == DataType::INT32; }
bool is_uint32(DataType dtype) { return dtype == DataType::UINT32; }
bool is_unsigned_int(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT16 || dtype == DataType::UINT8;
}
bool is_int32_uint32(DataType dtype) { return is_int32(dtype) || is_uint32(dtype); }
bool is_int32_uint32_uint16(DataType dtype) { return is_int32_uint32(dtype) || dtype == DataType::UINT16; }
bool is_relu_family_int(DataType dtype) { return is_int32(dtype) || is_unsigned_int(dtype); }

// Integer dtypes that unary_op_utils.cpp maps to a distinct init/LLK (or a dtype-agnostic kernel).
bool unary_op_supports_integer_dtype(UnaryOpType op_type, DataType dtype) {
    switch (op_type) {
        case UnaryOpType::ABS: return is_unsigned_int(dtype);
        case UnaryOpType::ABS_INT32:
        case UnaryOpType::CLAMP_TSS:
        case UnaryOpType::GEZ:
        case UnaryOpType::GTZ:
        case UnaryOpType::LEZ:
        case UnaryOpType::LTZ:
        case UnaryOpType::NEG:
        case UnaryOpType::SIGNBIT: return is_int32(dtype);

        case UnaryOpType::REMAINDER: return is_uint32(dtype);

        case UnaryOpType::LEAKY_RELU: return is_unsigned_int(dtype);

        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::MAXIMUM:
        case UnaryOpType::MINIMUM:
        case UnaryOpType::RSUB:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::UNARY_EQ:
        case UnaryOpType::UNARY_GE:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LE:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::WHERE_TSS: return is_int32_uint32(dtype);

        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::EQZ:
        case UnaryOpType::FILL:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::LOGICAL_NOT_UNARY:
        case UnaryOpType::NEZ:
        case UnaryOpType::RIGHT_SHIFT:
        case UnaryOpType::SQUARE: return is_int32_uint32_uint16(dtype);

        case UnaryOpType::BITWISE_NOT: return is_int32_uint32(dtype);

        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN: return is_relu_family_int(dtype);

        // Copy / typecast kernels; valid on integer tiles without a separate integer LLK.
        case UnaryOpType::BITCAST:
        case UnaryOpType::IDENTITY:
        case UnaryOpType::TYPECAST: return true;

        default: return false;
    }
}

void validate_integer_input_dtype(const std::vector<EltwiseUnaryWithParam>& op_chain, DataType input_dtype) {
    if (!is_integer_dtype(input_dtype)) {
        return;
    }
    // A TYPECAST in a multi-op chain changes the dtype for later ops; this checker only
    // sees the original tensor dtype, so skip rather than reject a valid post-cast float op.
    if (op_chain.size() > 1) {
        for (const auto& op : op_chain) {
            if (op.type() == UnaryOpType::TYPECAST) {
                return;
            }
        }
    }
    for (const auto& op : op_chain) {
        TT_FATAL(
            unary_op_supports_integer_dtype(op.type(), input_dtype),
            "Unary: {} does not support integer input dtype {}",
            op.type(),
            input_dtype);
    }
}

}  // namespace

ttsl::hash::hash_t UnaryDeviceOperation::operation_attributes_t::to_hash() const {
    return ttsl::hash::hash_objects_with_default_seed(
        op_chain,
        output_dtype,
        memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        sub_core_grids,
        worker_grid);
}

void UnaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_args.output_tensor;

    auto out_memory_config = args.memory_config;
    if (output_tensor.has_value()) {
        out_memory_config = output_tensor->memory_config();
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Unary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Unary: Operands need to be allocated in buffers on the device. Buffer is null.");

    validate_integer_input_dtype(args.op_chain, input_tensor.dtype());

    for (const auto& op : args.op_chain) {
        if (op.type() == operations::unary::UnaryOpType::LGAMMA) {
            TT_FATAL(
                input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32,
                "Unary: LGAMMA requires BFLOAT16 or FLOAT32 input, got dtype {}",
                static_cast<int>(input_tensor.dtype()));
            const DataType effective_out = output_tensor.has_value() ? output_tensor->dtype() : args.output_dtype;
            TT_FATAL(
                effective_out == DataType::BFLOAT16 || effective_out == DataType::FLOAT32,
                "Unary: LGAMMA requires BFLOAT16 or FLOAT32 output, got dtype {}",
                static_cast<int>(effective_out));
            break;
        }
    }

    // No early exit: beta is per-op, so every SOFTCAP entry in the chain has to be checked.
    for (const auto& op : args.op_chain) {
        if (op.type() == operations::unary::UnaryOpType::SOFTCAP) {
            // ckernel_sfpu_softcap.h exists only
            // under hw/ckernels/blackhole. Without this the kernel reaches JIT and
            // dies on a missing header, which points nowhere useful.
            TT_FATAL(
                input_tensor.device()->arch() == tt::ARCH::BLACKHOLE,
                "Unary: SOFTCAP is implemented for Blackhole only, got arch {}",
                input_tensor.device()->arch());
            // The op always runs the Sollya polynomial tanh, whose ~2.3e-3 relative error is
            // below half a bf16 ULP but far coarser than fp32 would imply. Refuse fp32 rather
            // than hand back a wide fp32 tensor; tanh_tile is the fp32-grade path.
            TT_FATAL(
                input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B,
                "Unary: SOFTCAP supports BFLOAT16 and BFLOAT8_B inputs, got dtype {}",
                input_tensor.dtype());
            // beta reaches the kernel as (beta, 1/beta), so zero would hand the SFPU inf and
            // return something that is not beta * tanh(x / beta).
            const auto beta = op.get_param_if<float>(0);
            TT_FATAL(beta.has_value(), "Unary: SOFTCAP requires a float beta parameter");
            TT_FATAL(*beta != 0.0f, "Unary: SOFTCAP requires a non-zero beta");
        }
    }

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Unary: Non-sharded input must be interleaved. Input memory layout: {}",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    }

    if (!out_memory_config.is_sharded()) {
        TT_FATAL(
            out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Unary: Non-sharded output must be interleaved. Output memory layout: {}",
            static_cast<int>(out_memory_config.memory_layout()));
    }

    if (output_tensor.has_value()) {
        // The preallocated output's shape is checked inside compute_output_specs (binary_ng
        // pattern), which also covers the program-cache-hit path. Call it here so the check still
        // runs when the program cache is disabled and compute_program_hash is never reached.
        compute_output_specs(args, tensor_args);

        TT_FATAL(
            output_tensor->layout() == input_tensor.layout(),
            "Unary: Output format (tile/row-major) must match input when preallocated. Output: {}, Input: {}",
            static_cast<int>(output_tensor->layout()),
            static_cast<int>(input_tensor.layout()));
    }
}

tt::tt_metal::TensorSpec UnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Unary is elementwise, so the shape the op produces is the input's logical shape.
    const auto output_shape = tensor_args.input.logical_shape();

    if (tensor_args.output_tensor.has_value()) {
        // Check before returning the preallocated spec: taking the expected shape from the return
        // value instead would compare the preallocated tensor against itself.
        const auto preallocated_output_shape = tensor_args.output_tensor->logical_shape();
        TT_FATAL(
            preallocated_output_shape == output_shape,
            "Unary: Preallocated output shape must match computed shape. Computed: {}, Preallocated: {}",
            output_shape,
            preallocated_output_shape);
        return tensor_args.output_tensor->tensor_spec();
    }

    if (args.memory_config.is_sharded()) {
        const auto output_layout = tensor_args.input.layout();
        const auto& memory_layout = args.memory_config.memory_layout();
        const auto& buffer_type = args.memory_config.buffer_type();

        // ND_SHARDED does not carry a 2D shard_spec to reconstruct from. Reusing it is safe and the input's
        // ND distribution still describes the output.
        if (!args.memory_config.shard_spec().has_value() && args.memory_config.nd_shard_spec().has_value()) {
            return tt::tt_metal::TensorSpec(
                output_shape, TensorLayout(args.output_dtype, PageConfig(output_layout), args.memory_config));
        }

        auto shard_spec_opt = args.memory_config.shard_spec();

        if (!shard_spec_opt.has_value()) {
            const auto& padded_out_shape = tensor_args.input.padded_shape();
            if (tensor_args.input.memory_config().shard_spec().has_value()) {
                shard_spec_opt = adjust_to_shape(
                    *tensor_args.input.memory_config().shard_spec(),
                    tensor_args.input.padded_shape(),
                    padded_out_shape);
            } else {
                shard_spec_opt = generate_output_shard_spec(tensor_args.input, padded_out_shape, memory_layout);
            }
        }

        return tt::tt_metal::TensorSpec(
            output_shape,
            TensorLayout(
                args.output_dtype,
                PageConfig(output_layout),
                MemoryConfig(memory_layout, buffer_type, shard_spec_opt)));
    }

    const auto output_layout = tensor_args.input.layout();
    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            args.output_dtype,
            PageConfig(output_layout),
            args.memory_config,
            output_shape,
            tensor_args.input.padded_shape()));
}

Tensor UnaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return *tensor_args.output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t UnaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(ttnn::is_device_tensor(input_tensor), "Unary: Unexpected tensor type {}", input_tensor.storage_type());

    const auto output_spec = compute_output_specs(attributes, tensor_args);
    const auto shard_specs = get_shard_specs(input_tensor.tensor_spec(), output_spec);
    std::optional<uint32_t> src_shard_vol = std::nullopt;
    std::optional<uint32_t> dst_shard_vol = std::nullopt;
    if (shard_specs.has_value()) {
        const auto tile_hw = input_tensor.tensor_spec().tile().get_tile_hw();
        if (input_tensor.is_sharded()) {
            src_shard_vol = shard_specs->input_shard_spec.numel() / tile_hw;
        }
        const auto out_tile_hw = output_spec.tile().get_tile_hw();
        dst_shard_vol = shard_specs->output_shard_spec.numel() / out_tile_hw;
    }

    // On cache hit, the descriptor is not rebuilt and no relaxation is applied. The dispatched
    // tensor_layout must be the same as the one built for the cached program.  Anything omitted
    // from this key can give different config and fail validation (wrong data now and a hard TT_FATAL
    // once the Metal 2.0 port declares TensorParameter relaxations). The output layout needs its
    // own term because compute_output_specs can hand back a caller-supplied preallocated spec and
    // validation only compares its Layout enum against the input's.
    //
    // Hashing tensor_layout does not ignore shape. Alignment is part of tensor_layout and since
    // legacyShapeToAlignment returns {padded_h, padded_w} for an overpadded TILE tensor instead of tile
    // dims, differently padded H/W values produce different keys. Tile-aligned tensors are unaffected.
    //
    // Sharded distribution needs its own term. Since shape and shard squeeze together, one shard spec resolves
    // per shape ({64,64} over two cores: [64,64] -> [4], [64,128] -> [2,2]) and GRID_2D trims the bank list
    // from the unsqueezed shape ([64,128] and [64,192] over two and three banks). The accessor passes both as
    // compile-time args. Use the Buffer's stored sharding_args since they describe the actual buffer layout
    // used by the factory. A reshaped view keeps its parent tensor's sharding_args. A null buffer means the
    // output has not been allocated yet and its buffer will come from output_spec.
    //
    // TODO(port): When TensorParameter replaces TensorAccessorArgs, TensorSpec becomes the authoritative source.
    // Swap the Buffer branch for the spec on both sides since Metal 2.0 validation reads
    // spec.compute_buffer_sharding_args().
    const auto distribution_key = [](const tt::tt_metal::TensorSpec& spec,
                                     const Tensor* tensor) -> std::optional<std::pair<Shape, std::vector<CoreCoord>>> {
        if (!spec.memory_config().is_sharded()) {
            return std::nullopt;
        }
        const auto* buffer = tensor != nullptr && tensor->device() != nullptr ? tensor->buffer() : nullptr;
        const auto computed = buffer == nullptr ? std::optional{spec.compute_buffer_sharding_args()} : std::nullopt;
        const auto& distribution =
            buffer != nullptr ? buffer->buffer_distribution_spec() : computed->buffer_distribution_spec();
        if (!distribution.has_value()) {
            return std::nullopt;
        }
        return std::pair{distribution->shard_shape_in_pages(), distribution->cores()};
    };

    return operation::hash_operation<UnaryDeviceOperation>(
        attributes,
        input_tensor.tensor_spec().tensor_layout(),
        output_spec.tensor_layout(),
        // TODO: For ROW_MAJOR, page size depends on width. Hashing padded_shape ensures
        // different widths get separate cache entries. Consider hashing only the last
        // dimension to allow cache reuse when only height differs
        input_tensor.layout() == Layout::ROW_MAJOR ? std::optional{input_tensor.padded_shape()} : std::nullopt,
        distribution_key(input_tensor.tensor_spec(), &input_tensor),
        distribution_key(output_spec, tensor_args.output_tensor.has_value() ? &*tensor_args.output_tensor : nullptr),
        src_shard_vol,
        dst_shard_vol);
}

bool UnaryDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::unary

namespace ttnn::prim {

Tensor unary(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::unary::UnaryDeviceOperation;

    auto mem_config_actual =
        optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : (output_memory_config);

    auto worker_grid = ttnn::operations::unary::get_worker_grid(
        input, optional_output_tensor, std::optional<MemoryConfig>(output_memory_config), sub_core_grids);

    auto operation_attributes = OperationType::operation_attributes_t{
        .op_chain = op_chain,
        .output_dtype = output_dtype,
        .memory_config = mem_config_actual,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .preserve_fp32_precision = preserve_fp32_precision,
        .bfp8_pack_precise = bfp8_pack_precise,
        .worker_grid = worker_grid,
        .sub_core_grids = sub_core_grids,
    };

    auto tensor_args = OperationType::tensor_args_t{.input = input, .output_tensor = optional_output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
