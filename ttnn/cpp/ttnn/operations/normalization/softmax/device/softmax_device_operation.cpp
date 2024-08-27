// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"
#include "tt_metal/common/assert.hpp"
#include "common/base_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>
#include <type_traits>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return std::visit(
        [&](const auto& program_config) -> SoftmaxDeviceOperation::program_factory_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                return SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory{};
            } else {
                return SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory{};
            }
        },
        attributes.program_config
    );
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);

    if (attributes.scale.has_value() || attributes.is_scale_causal_mask_hw_dims_softmax) {
        TT_FATAL(attributes.scale.has_value(), "Scale must be provided when is_scale_causal_mask_hw_dims_softmax is true");
    }

    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, SoftmaxDefaultProgramConfig>) {
                TT_FATAL(!attributes.is_scale_causal_mask_hw_dims_softmax);
            } else if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                const auto shape = input_tensor.get_legacy_shape();
                uint32_t M = input_tensor.volume() / shape[-1];
                uint32_t K = shape[-1];

                TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
                TT_FATAL(program_config.block_w * TILE_WIDTH == shape[3], "shard width must equal to input tensor shape[3]!");
                TT_FATAL(attributes.inplace);

                if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
                    auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                    auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                    TT_FATAL(M * K / ((program_config.block_w * program_config.block_h) * TILE_HW) == num_cores_r * num_cores_c,
                             "number of shards must equal to number of cores. M = {}, K = {}, block_w = {}, block_h = {}, num_cores = {}",
                             M, K, program_config.block_w, program_config.block_h, num_cores_r * num_cores_c);
                } else {
                    TT_FATAL(attributes.is_causal_mask);
                    TT_FATAL(input_tensor.get_layout() == Layout::TILE);
                    TT_FATAL(input_tensor.is_sharded());
                    TT_FATAL(input_tensor.shard_spec()->orientation == ShardOrientation::ROW_MAJOR);
                    TT_FATAL(attributes.scale.has_value());
                }
            }
        },
        attributes.program_config
    );
}

void SoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

SoftmaxDeviceOperation::shape_return_value_t SoftmaxDeviceOperation::compute_output_shapes(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor.get_legacy_shape();
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (attributes.inplace) {
        return tensor_args.input_tensor;
    } else {
        auto output_shape = compute_output_shapes(attributes, tensor_args);
        const auto& input_tensor = tensor_args.input_tensor;
        return create_device_tensor(
            output_shape,
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            attributes.memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
    }
}

tt::stl::hash::hash_t SoftmaxDeviceOperation::compute_program_hash(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    std::cout << "compute_program_hash" << std::endl;
    auto out = operation::hash_operation<SoftmaxDeviceOperation>(
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.dtype(),
        tensor_args.mask.has_value() ? std::make_optional(tensor_args.mask.value().memory_config()) : std::nullopt,
        tensor_args.mask.has_value() ? std::make_optional(tensor_args.mask.value().dtype()) : std::nullopt,
        attributes.memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG)
    );
    std::cout << "compute_program_hash" << std::endl;
    return out;
}

std::tuple<SoftmaxDeviceOperation::operation_attributes_t, SoftmaxDeviceOperation::tensor_args_t>
SoftmaxDeviceOperation::invoke(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return std::make_tuple(operation_attributes, tensor_args);
}


}  // namespace ttnn::operations::normalization
