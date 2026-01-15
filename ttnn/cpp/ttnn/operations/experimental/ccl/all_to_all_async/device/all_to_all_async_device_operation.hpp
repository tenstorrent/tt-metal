// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device.hpp>
#include "all_to_all_async_device_operation_types.hpp"
#include "all_to_all_async_program_factory.hpp"

namespace ttnn::experimental::prim {

struct AllToAllAsyncDeviceOperation {
    using operation_attributes_t = AllToAllAsyncParams;
    using tensor_args_t = AllToAllAsyncInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<AllToAllAsyncProgram>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor all_to_all_async(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    const ttnn::GlobalSemaphore& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id);

}  // namespace ttnn::experimental::prim
