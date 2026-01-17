// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::transpose {

TransposeDeviceOperation::program_factory_t TransposeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    bool is_l1 = input_tensor.is_sharded() && input_tensor.buffer()->is_l1();
    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    auto parallelization_strategy = get_parallelization_strategy(operation_attributes, tensor_args);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (is_l1) {
                if (is_row_major) {
                    return program::TransposeWHShardedRMProgramFactory{};
                }
                return program::TransposeWHShardedProgramFactory{};
            }
            return program::TransposeWHProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            if (is_l1) {
                return program::TransposeHCShardedProgramFactory{};
            }
            if (is_row_major) {
                return program::TransposeHCRMProgramFactory{};
            }
            // Tiled interleaved (non-sharded TILE layout)
            return program::TransposeHCTiledInterleavedProgramFactory{};

        case TransposeOpParallelizationStrategy::MULTI_CORE_CN: return program::TransposeCNProgramFactory{};

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

void TransposeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void TransposeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& dim = operation_attributes.dim;
    const float pad_value = operation_attributes.pad_value;
    const auto& output_mem_config = operation_attributes.output_mem_config;

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
    uint32_t W = shape[3], H = shape[2], C = shape[1], N = shape[0];

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

    uint32_t ROW_MAJOR_STICK_WIDTH = 16;
    if (dim == TransposeOpDim::WH) {
        if (row_major) {
            TT_FATAL(
                (W * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0 &&
                    (H * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0,
                "Row major tensor W {} H {} must be a multiple of ROW_MAJOR_STICK_WIDTH for transpose wh",
                W,
                H,
                ROW_MAJOR_STICK_WIDTH);
        }
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Only height and block sharding is supported for transpose wh");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(
                (shard_spec.shape[0] % H == 0) || (H % shard_spec.shape[0] == 0),
                "Only a multiple of H {} or a factor of H is allows for the shard height {} for transpose WH",
                H,
                shard_spec.shape[0]);
            TT_FATAL(shard_spec.shape[1] == W, "Only height sharding is supported");
            if (H > shard_spec.shape[0]) {
                TT_FATAL(
                    N == 1,
                    "Transpose WH does not support sharded inputs when shard height {} is less than H {} and N {} > 1",
                    shard_spec.shape[0],
                    H,
                    N);
                TT_FATAL(
                    C == 1,
                    "Transpose WH does not support sharded inputs when  shard height {} is less than H {} and C {} > 1",
                    shard_spec.shape[0],
                    H,
                    N);
            }
            TT_FATAL(output_mem_config.is_sharded(), "Output must be sharded for transpose WH");
            TT_FATAL(
                output_mem_config.memory_layout() != TensorMemoryLayout::BLOCK_SHARDED,
                "Only height and width sharding output is supported for transpose wh");
        } else {
            TT_FATAL(!output_mem_config.is_sharded(), "Interleaved input tensors cannot output sharded outputs");
        }
    } else {
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only height sharding is supported for transpose hc");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(shard_spec.shape[1] == W, "Block/Width sharding is not supported");
            TT_FATAL(output_mem_config.is_sharded(), "Sharded input can only output sharded tensors for transpose hc");
            TT_FATAL(
                output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only height sharding is supported for the ouput of sharded transpose hc");
        } else {
            TT_FATAL(!output_mem_config.is_sharded(), "Interleaved inputs cannot output sharded outputs");
        }
    }

    if (dim == TransposeOpDim::HC) {
        if (row_major) {
            auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                        ? hal::get_dram_alignment()
                                        : hal::get_l1_alignment();
            TT_FATAL(
                (W * input_tensor.element_size()) % BUFFER_ALIGNMENT == 0,
                "Buffer is not aligned for this implementation row_size_bytes {} buffer_alignment {}",
                W * input_tensor.element_size(),
                BUFFER_ALIGNMENT);
        }
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32 ||
                input_tensor.dtype() == DataType::INT32,
            "Unsupported data type for input tensor");
        TT_FATAL(
            !(input_tensor.is_sharded() && input_tensor.layout() == Layout::TILE),
            "HC transpose does not support sharded+tilized inputs");
        TT_FATAL(
            !(input_tensor.is_sharded() && pad_value != 0.0f),
            "Sharded HC transpose does not support non-zero padding {}",
            pad_value);
    }
}

TransposeDeviceOperation::spec_return_value_t TransposeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& dim = operation_attributes.dim;
    const auto& output_mem_config = operation_attributes.output_mem_config;

    // TODO: Remove usage of input/output padded shape
    // - Get output alignment from input alignment and output dtype, layout, mem_config
    // - Get shard spec from output strides (logical shape + alignment)?
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    switch (dim) {
        case TransposeOpDim::CN:
            std::swap(output_shape[0], output_shape[1]);
            std::swap(output_padded_shape[0], output_padded_shape[1]);
            break;
        case TransposeOpDim::HC:
            if (input_tensor.is_sharded() || input_tensor.layout() != Layout::TILE) {
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
        case TransposeOpDim::NH:
            std::swap(output_shape[0], output_shape[2]);
            std::swap(output_padded_shape[0], output_padded_shape[2]);
            break;
        case TransposeOpDim::NW:
            std::swap(output_shape[0], output_shape[3]);
            std::swap(output_padded_shape[0], output_padded_shape[3]);
            break;
        case TransposeOpDim::CW:
            std::swap(output_shape[1], output_shape[3]);
            std::swap(output_padded_shape[1], output_padded_shape[3]);
            break;
    }

    auto result_output_mem_config = output_mem_config;
    if (output_mem_config.is_sharded()) {
        TT_FATAL(input_tensor.is_sharded(), "Sharded output tensor must have a sharded input tensor");
        if (dim == TransposeOpDim::WH) {
            const auto& input_padded_shape = input_tensor.padded_shape();
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            if (shard_spec.shape[0] >= input_padded_shape[-2]) {
                shard_spec.shape[0] = shard_spec.shape[0] / input_padded_shape[-2] * input_padded_shape[-1];
                shard_spec.shape[1] = input_padded_shape[-2];
                result_output_mem_config = result_output_mem_config.with_shard_spec(shard_spec);
            } else {
                std::swap(shard_spec.shape[0], shard_spec.shape[1]);
                result_output_mem_config =
                    MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, result_output_mem_config.buffer_type(), shard_spec);
            }
        } else if (dim == TransposeOpDim::HC) {
            result_output_mem_config = result_output_mem_config.with_shard_spec(input_tensor.shard_spec());
        } else {
            TT_ASSERT(false, "Unsupported sharding");
        }
    }

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            result_output_mem_config,
            output_shape,
            output_padded_shape));
}

TransposeDeviceOperation::tensor_return_value_t TransposeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<TransposeDeviceOperation::tensor_return_value_t>
TransposeDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::transpose

namespace ttnn::prim {
ttnn::Tensor transpose(
    const Tensor& input_tensor,
    ttnn::operations::data_movement::transpose::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    float pad_value) {
    using OperationType = ttnn::operations::data_movement::transpose::TransposeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = dim,
            .output_mem_config = output_mem_config,
            .pad_value = pad_value,
        },
        OperationType::tensor_args_t{
            .input = input_tensor,
        });
}
}  // namespace ttnn::prim
