// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::forward_substitution {
struct ForwardSubstitutionOperation {
    struct operation_attributes_t {
        const MemoryConfig memory_config;
    };
    struct tensor_args_t {
        const Tensor& input;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            std::vector<tt::tt_metal::CoreCoord> cores;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };
    using program_factory_t = std::variant<MultiCore>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::forward_substitution

namespace ttnn::prim {
ttnn::Tensor forward_substitution(const Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);
}
