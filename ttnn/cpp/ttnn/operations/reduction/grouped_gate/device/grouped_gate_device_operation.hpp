// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction {

struct GroupedGateDeviceOperation {
    struct operation_attributes_t {
        float route_scale;
        float epsilon;
        uint32_t n_groups;
        uint32_t topk;
        uint32_t topk_groups;
        uint32_t n_activated_experts;
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& scores;
        const Tensor& bias;
    };

    using spec_return_value_t = std::tuple<Tensor, Tensor>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& scores,
        const Tensor& bias,
        const float route_scale,
        const float epsilon,
        const uint32_t n_groups,
        const uint32_t topk,
        const uint32_t topk_groups,
        const uint32_t n_activated_experts,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace ttnn::operations::reduction

namespace ttnn::prim {
constexpr auto grouped_gate =
    ttnn::register_operation<"ttnn::prim::grouped_gate", ttnn::operations::reduction::GroupedGateDeviceOperation>();
}  // namespace ttnn::prim
