// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_new_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// ---------------------------------------------------------------
// select_program_factory: dispatch to the correct descriptor factory
// ---------------------------------------------------------------
MatmulNewDeviceOperation::program_factory_t MatmulNewDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    const auto& config = operation_attributes.program_config.value();

    return std::visit(
        [](const auto& c) -> program_factory_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return matmul_new_detail::MultiCoreDescriptorFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return matmul_new_detail::ReuseOptimizedDescriptorFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return matmul_new_detail::ReuseMcast2DDescriptorFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return matmul_new_detail::ReuseMcast1DDescriptorFactory{};
            } else if constexpr (std::is_same_v<
                                     T,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return matmul_new_detail::DRAMShardedDescriptorFactory{};
            } else if constexpr (
                std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                return matmul_new_detail::BatchedHSDRAMShardedDescriptorFactory{};
            } else {
                static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported program config type");
            }
        },
        config);
}

// Delegate to the original matmul's implementations -- they are identical
void MatmulNewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    MatmulDeviceOperation::validate_on_program_cache_miss(attributes, args);
}

MatmulNewDeviceOperation::spec_return_value_t MatmulNewDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    return MatmulDeviceOperation::compute_output_specs(attributes, args);
}

MatmulNewDeviceOperation::tensor_return_value_t MatmulNewDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    return MatmulDeviceOperation::create_output_tensors(attributes, args);
}

tt::stl::hash::hash_t MatmulNewDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    const auto& input_tensor_a = args.input_tensors.at(0);
    const auto& input_tensor_b = args.input_tensors.at(1);

    // Compute factory index matching the variant order in program_factory_t.
    const auto& config = attributes.program_config.value();
    size_t factory_index = std::visit(
        [](const auto& c) -> size_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return 0;
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return 1;
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return 2;
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return 3;
            } else if constexpr (std::is_same_v<
                                     T,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return 4;
            } else {
                return 5;
            }
        },
        config);

    auto hash = tt::tt_metal::operation::hash_operation<MatmulNewDeviceOperation>(
        attributes, factory_index, input_tensor_a, input_tensor_b);

    for (const auto& optional_input_tensor : args.optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            hash = tt::stl::hash::hash_objects(hash, optional_input_tensor.value());
        }
    }

    for (const auto& optional_output_tensor : args.optional_output_tensors) {
        if (optional_output_tensor.has_value()) {
            hash = tt::stl::hash::hash_objects(hash, optional_output_tensor.value());
        }
    }
    return hash;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<MatmulNewDeviceOperation::tensor_return_value_t>
MatmulNewDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    return MatmulDeviceOperation::create_op_performance_model(attributes, tensor_args, output_tensors);
}

// ---------------------------------------------------------------
// prim::matmul_new entry point
// ---------------------------------------------------------------
MatmulNewDeviceOperation::tensor_return_value_t matmul_new(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& optional_output_tensor,
    const MatmulParams& attributes) {
    if (!attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;
        if (bias.has_value()) {
            auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.value().dtype());
            bias_single_tile_size = tt::tile_size(bias_data_format);
        }

        MatmulParams attributes_with_program_config = attributes;
        attributes_with_program_config.program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            attributes.transpose_a,
            attributes.transpose_b,
            bias_single_tile_size,
            attributes);

        return ttnn::device_operation::launch<MatmulNewDeviceOperation>(
            attributes_with_program_config, {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
    }
    return ttnn::device_operation::launch<MatmulNewDeviceOperation>(
        attributes, {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
}

}  // namespace ttnn::prim
