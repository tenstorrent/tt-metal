// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_combine {

struct PrefillCombineDeviceOperation {
    struct operation_attributes_t {
        const uint32_t num_chips;
        const uint32_t experts_per_chip;
        const uint32_t num_experts_per_tok;
        const uint32_t seq_len_per_chip;
        const MemoryConfig output_mem_config;
        const CoreRangeSet worker_core_range_set;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "num_chips",
            "experts_per_chip",
            "num_experts_per_tok",
            "seq_len_per_chip",
            "output_mem_config",
            "worker_core_range_set");

        auto attribute_values() const {
            return std::forward_as_tuple(
                num_chips,
                experts_per_chip,
                num_experts_per_tok,
                seq_len_per_chip,
                output_mem_config,
                worker_core_range_set);
        };
    };

    struct tensor_args_t {
        const ttnn::Tensor dispatched_tensor;
        const ttnn::Tensor metadata_tensor;
        const ttnn::Tensor experts_counter_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct PrefillCombineProgramFactory {
        struct shared_variables_t {
            // Placeholder for shared variables between create and override_runtime_arguments
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        // Stub declarations
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

    using program_factory_t = std::variant<PrefillCombineProgramFactory>;

    // Mandatory methods
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine

namespace ttnn::prim {
ttnn::Tensor prefill_combine(
    const ttnn::Tensor& dispatched_tensor,
    const ttnn::Tensor& metadata_tensor,
    const ttnn::Tensor& experts_counter_tensor,
    uint32_t num_chips,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set);
}  // namespace ttnn::prim
