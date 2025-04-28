// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::index_fill {
struct IndexFillOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const std::variant<float, int> value;
        const MemoryConfig memory_config;
    };
    struct tensor_args_t {
        const Tensor& input;
        const Tensor& index;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const uint32_t dim,
        const Tensor& index,
        const std::variant<float, int> value,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::index_fill
namespace ttnn::prim {
constexpr auto index_fill =
    ttnn::register_operation<"ttnn::prim::index_fill", ttnn::operations::index_fill::IndexFillOperation>();
}
