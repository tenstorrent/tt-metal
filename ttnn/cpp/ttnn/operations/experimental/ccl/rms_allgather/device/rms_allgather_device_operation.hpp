// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rms_allgather_device_operation_types.hpp"
#include "rms_allgather_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::fused::normalization {

namespace layernorm = ttnn::operations::normalization;

struct RMSAllGatherDeviceOperation {
    using operation_attributes_t = fused::normalization::operation_attributes_t;
    using tensor_args_t = fused::normalization::tensor_args_t;
    using spec_return_value_t = fused::normalization::spec_return_value_t;
    using tensor_return_value_t = fused::normalization::tensor_return_value_t;
    using program_factory_t = std::variant<program::RMSAllGatherMeshWorkloadFactory>;
    using shared_variables_t = program::RMSAllGatherMeshWorkloadFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::fused::normalization
