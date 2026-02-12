// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce {

struct DeepseekMinimalAllReduceDeviceOperation {
    using operation_attributes_t = deepseek_minimal_all_reduce::operation_attributes_t;
    using tensor_args_t = deepseek_minimal_all_reduce::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::DeepseekMinimalAllReduceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce

namespace ttnn::prim {

ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::DeepseekMinimalAllReduceDeviceOperation::
    tensor_return_value_t
    deepseek_minimal_all_reduce(
        const ttnn::Tensor& input_tensor,
        uint32_t num_links,
        tt::tt_fabric::Topology topology,
        std::optional<uint32_t> cluster_axis,
        const std::optional<ttnn::Tensor>& intermediate_tensor,
        const std::optional<ttnn::Tensor>& residual_tensor = std::nullopt,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt);

}  // namespace ttnn::prim
