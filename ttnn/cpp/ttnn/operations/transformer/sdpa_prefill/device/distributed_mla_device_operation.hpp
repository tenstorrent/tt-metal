// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "distributed_mla_program_factory.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

struct DistributedMLADeviceOperation {
    using operation_attributes_t = DistributedMlaMeshWorkloadFactory::operation_attributes_t;
    using tensor_args_t = DistributedMlaMeshWorkloadFactory::tensor_args_t;
    using spec_return_value_t = DistributedMlaMeshWorkloadFactory::spec_return_value_t;
    using tensor_return_value_t = DistributedMlaMeshWorkloadFactory::tensor_return_value_t;

    using program_factory_t = std::variant<DistributedMlaMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::transformer::sdpa_prefill
