// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

struct IterativeTopkDeviceOperation {
    struct operation_attributes_t {
        uint32_t k;
        tt::tt_metal::MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = std::array<TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            std::vector<tt::tt_metal::CoreCoord> cores;
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

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::iterative_topk::IterativeTopkDeviceOperation::tensor_return_value_t
iterative_topk(
    const Tensor& input, uint32_t k, const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::prim
