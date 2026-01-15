// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "groupnorm_sharded_program_factory.hpp"
#include "groupnorm_no_mcast_program_factory.hpp"
#include "groupnorm_mcast_program_factory.hpp"
#include "groupnorm_device_operation_types.hpp"

namespace ttnn::prim {

struct GroupNormDeviceOperation {
    using operation_attributes_t = GroupNormParams;
    using tensor_args_t = GroupNormInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<GroupNormShardedProgramFactory, GroupNormNoMcastProgramFactory, GroupNormMcastProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

Tensor group_norm(
    const Tensor& input,
    float eps,
    uint32_t num_groups,
    const MemoryConfig& output_mem_config,
    const GroupNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool use_welford,
    std::optional<Tensor> gamma,
    std::optional<Tensor> beta,
    std::optional<Tensor> input_mask,
    std::optional<Tensor> negative_mask,
    std::optional<Tensor> reciprocals);

}  // namespace ttnn::prim
