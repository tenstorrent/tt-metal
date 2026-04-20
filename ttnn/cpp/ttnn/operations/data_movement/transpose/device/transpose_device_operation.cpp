// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_device_operation.hpp"
#include "transpose_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape;
using ttnn::operations::data_movement::transpose::generate_transpose_shard_spec;
using ttnn::operations::data_movement::transpose::is_native_transpose_sharding;

namespace ttnn::prim {

namespace {

// Compute the output padded shape for a given transpose dim. Mirrors the switch in
// compute_output_specs so derivation stays in one place.
ttnn::Shape transposed_padded_shape(const Tensor& input_tensor, TransposeOpDim dim) {
    auto out_padded = input_tensor.padded_shape();
    switch (dim) {
        case TransposeOpDim::CN: std::swap(out_padded[0], out_padded[1]); break;
        case TransposeOpDim::HC:
            if (input_tensor.layout() == Layout::ROW_MAJOR) {
                std::swap(out_padded[1], out_padded[2]);
            } else {
                const uint32_t C = input_tensor.logical_shape()[1];
                const uint32_t C_p = tt::round_up(C, input_tensor.tensor_spec().tile().get_height());
                out_padded[1] = out_padded[2];
                out_padded[2] = C_p;
            }
            break;
        case TransposeOpDim::WH: std::swap(out_padded[2], out_padded[3]); break;
        default: TT_THROW("Unsupported transpose dim"); break;
    }
    return out_padded;
}

// Derive the effective output MemoryConfig. When the user asks for a sharded output but omits
// shard_spec, we synthesize one here so the rest of the op (select_program_factory,
// compute_output_specs) can reason about a fully-specified config.
MemoryConfig derive_effective_output_memory_config(
    const TransposeDeviceOperation::operation_attributes_t& operation_attributes,
    const TransposeDeviceOperation::tensor_args_t& tensor_args) {
    auto output_mem_config = operation_attributes.output_mem_config;
    if (!output_mem_config.is_sharded() || output_mem_config.shard_spec().has_value()) {
        return output_mem_config;
    }
    const auto& input_tensor = tensor_args.input;
    const auto output_padded_shape = transposed_padded_shape(input_tensor, operation_attributes.dim);
    if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value()) {
        auto shard_spec = adjust_shard_spec_to_shape(
            input_tensor.shard_spec().value(), input_tensor.padded_shape(), output_padded_shape);
        return output_mem_config.with_shard_spec(shard_spec);
    }
    auto shard_spec =
        generate_transpose_shard_spec(input_tensor, output_padded_shape, output_mem_config.memory_layout());
    return output_mem_config.with_shard_spec(shard_spec);
}

}  // namespace

TransposeDeviceOperation::program_factory_t TransposeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto output_memory_config = derive_effective_output_memory_config(operation_attributes, tensor_args);
    const auto& dim = operation_attributes.dim;
    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    // When `native` is false, the specialized sharded WH/HC factories are skipped and we fall
    // through to the interleaved factories, which use TensorAccessorArgs for transparent NOC-based
    // access against either interleaved or sharded buffers.
    bool native = is_native_transpose_sharding(input_tensor.tensor_spec(), output_memory_config);

    uint32_t N = input_tensor.logical_shape()[0], C = input_tensor.logical_shape()[1];
    uint32_t output_width =
        (dim == TransposeOpDim::WH) ? input_tensor.logical_shape()[-2] : input_tensor.logical_shape()[-1];
    uint32_t output_height =
        (dim == TransposeOpDim::WH)
            ? input_tensor.logical_shape()[-1]
            : ((dim == TransposeOpDim::HC) ? input_tensor.logical_shape()[-3] : input_tensor.logical_shape()[-2]);

    bool input_height_sharded = native && input_tensor.is_sharded() && input_tensor.shard_spec().has_value() &&
                                input_tensor.shard_spec()->shape[1] == input_tensor.logical_shape()[-1];
    bool input_width_and_height_fully_in_shard =
        input_height_sharded && input_tensor.shard_spec()->shape[0] % input_tensor.logical_shape()[-2] == 0;
    bool output_height_sharded = native && output_memory_config.is_sharded() &&
                                 output_memory_config.shard_spec().has_value() &&
                                 output_memory_config.shard_spec()->shape[1] == output_width;
    bool output_width_sharded = native && output_memory_config.is_sharded() &&
                                output_memory_config.shard_spec().has_value() &&
                                output_memory_config.shard_spec()->shape[0] == output_height;
    bool output_width_and_height_fully_in_shard =
        output_height_sharded && output_memory_config.shard_spec()->shape[0] % output_height == 0;
    bool use_sharded_wh =
        native && ((input_width_and_height_fully_in_shard && output_width_and_height_fully_in_shard) ||
                   (N == 1 && C == 1 && input_height_sharded && output_width_sharded));
    bool use_sharded_hc = native && input_height_sharded && output_height_sharded && is_row_major;

    auto parallelization_strategy = get_parallelization_strategy(operation_attributes, tensor_args);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (use_sharded_wh) {
                if (is_row_major) {
                    return TransposeWHShardedRMProgramFactory{};
                }
                return TransposeWHShardedProgramFactory{};
            }
            return TransposeWHProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            if (use_sharded_hc) {
                return TransposeHCShardedProgramFactory{};
            }
            if (is_row_major) {
                return TransposeHCRMProgramFactory{};
            }
            return TransposeHCTiledInterleavedProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_CN: return TransposeCNProgramFactory{};

        default: TT_THROW("Unsupported parallelization strategy");
    }
}

TransposeOpParallelizationStrategy TransposeDeviceOperation::get_parallelization_strategy(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    switch (operation_attributes.dim) {
        case TransposeOpDim::WH: return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
        case TransposeOpDim::HC: return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
        case TransposeOpDim::CN: return TransposeOpParallelizationStrategy::MULTI_CORE_CN;
        default: TT_THROW("Unsupported transpose dim for parallelization strategy");
    }
}

void TransposeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& dim = operation_attributes.dim;
    const float pad_value = operation_attributes.pad_value;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to transpose need to be allocated in buffers on device!");
    TT_FATAL(
        !(dim != TransposeOpDim::HC && pad_value != 0.0f),
        "Non-zero padding {} is not supported for any transpose other than HC.",
        pad_value);
    TT_FATAL(
        dim == TransposeOpDim::HC || dim == TransposeOpDim::WH || dim == TransposeOpDim::CN,
        "Transpose HC, WH, CN are the only supported transpose operations. Transpose {} is not supported.",
        static_cast<int>(dim));

    const auto& shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    uint32_t W = shape[3], H = shape[2];

    if (!row_major) {
        TT_FATAL(
            W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0,
            "Tiled tensor H {} W {} must be a multiple of TILE HEIGHT {} and TILE WIDTH",
            H,
            W,
            TILE_HEIGHT,
            TILE_WIDTH);
        TT_FATAL(
            input_tensor.physical_volume() % TILE_HW == 0,
            "Tiled tensor volume {} must be a multiple of TILE HEIGHT * TILE WIDTH",
            input_tensor.physical_volume(),
            TILE_HW);
    }
}

TensorSpec TransposeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& dim = operation_attributes.dim;
    const auto output_mem_config = derive_effective_output_memory_config(operation_attributes, tensor_args);

    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    switch (dim) {
        case TransposeOpDim::CN:
            std::swap(output_shape[0], output_shape[1]);
            std::swap(output_padded_shape[0], output_padded_shape[1]);
            break;
        case TransposeOpDim::HC:
            if (input_tensor.layout() == Layout::ROW_MAJOR) {
                std::swap(output_shape[1], output_shape[2]);
                std::swap(output_padded_shape[1], output_padded_shape[2]);
            } else {
                uint32_t C = output_shape[1];
                uint32_t C_p = tt::round_up(C, input_tensor.tensor_spec().tile().get_height());
                uint32_t H = output_shape[2];
                output_shape[1] = H;
                output_shape[2] = C;
                output_padded_shape[1] = H;
                output_padded_shape[2] = C_p;
            }
            break;
        case TransposeOpDim::WH:
            std::swap(output_shape[2], output_shape[3]);
            std::swap(output_padded_shape[2], output_padded_shape[3]);
            break;
        default: TT_THROW("Unsupported transpose dim"); break;
    }

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            output_mem_config,
            output_shape,
            output_padded_shape));
}

Tensor TransposeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> TransposeDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> result({input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::Tensor transpose(
    const Tensor& input_tensor,
    ttnn::prim::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    float pad_value) {
    using OperationType = ttnn::prim::TransposeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = dim,
            .output_mem_config = output_mem_config,
            .pad_value = pad_value,
        },
        TransposeInputs{
            .input = input_tensor,
        });
}
}  // namespace ttnn::prim
