// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "topk_router_gpt_device_operation_types.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

struct TopkRouterGptDeviceOperation {
    using operation_attributes_t = topk_router_gpt::operation_attributes_t;
    using tensor_args_t = topk_router_gpt::tensor_args_t;
    using tensor_return_value_t = topk_router_gpt::tensor_return_value_t;
    using spec_return_value_t = topk_router_gpt::spec_return_value_t;

    // The five tensor addresses are the only per-dispatch state; they are declared as runtime-arg
    // bindings in create_descriptor, so the framework patches them on a cache hit. The rest of the
    // per-core block (roles, k-tile split, ring ordering, vchannels) derives from the tensor specs
    // and the device's DRAM bank assignment, all covered by the program hash — hence no override.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const Tensor& bias_tensor,
        uint32_t k,
        uint32_t num_experts);
};

}  // namespace ttnn::operations::experimental::topk_router_gpt

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> topk_router_gpt(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& bias_tensor,
    uint32_t k,
    uint32_t num_experts);

}  // namespace ttnn::experimental
