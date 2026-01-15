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
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "all_to_all_async_generic_device_operation_types.hpp"
#include "all_to_all_async_generic_program_factory.hpp"

namespace ttnn::operations::experimental::ccl {

struct AllToAllAsyncGenericDeviceOperation {
    using operation_attributes_t = all_to_all_async_generic::operation_attributes_t;
    using tensor_args_t = all_to_all_async_generic::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using AllToAllAsyncGenericProgram = all_to_all_async_generic::AllToAllAsyncGenericProgram;
    using program_factory_t = std::variant<AllToAllAsyncGenericProgram>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::AllToAllAsyncGenericDeviceOperation::tensor_return_value_t
all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
