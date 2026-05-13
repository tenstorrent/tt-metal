// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"

#include "reduce_op_device_operation_types.hpp"
#include "reduce_op_multi_core_w_program_factory.hpp"
#include "reduce_op_single_core_hw_program_factory.hpp"
#include "tt_stl/reflection.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct ReduceDeviceOperation {
    using operation_attributes_t = ReduceParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // The single-core HW factory and the multi-core W factory are migrated to the
    // Metal 2.0 host API (they use ProgramFactoryConcept: create() +
    // override_runtime_arguments()). The multi-core H factory remains on the legacy
    // ProgramDescriptor API because the width-sharded path uses the dynamic
    // CircularBuffer pattern (CBs built on a tensor's borrowed memory) which
    // Metal 2.0 has not yet implemented as a borrowed-memory DataflowBuffer.
    // The framework's variant adapter dispatches each alternative based on which
    // concept it satisfies.
    using ReduceSingleCoreHwProgramFactory = ::ttnn::prim::ReduceSingleCoreHwProgramFactory;
    using ReduceMultiCoreWProgramFactory = ::ttnn::prim::ReduceMultiCoreWProgramFactory;

    struct ReduceMultiCoreHProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t =
        std::variant<ReduceSingleCoreHwProgramFactory, ReduceMultiCoreHProgramFactory, ReduceMultiCoreWProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

ttnn::Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool negate = false,
    float post_mul_scaler = 1.0f);

}  // namespace ttnn::prim
