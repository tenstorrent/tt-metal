// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::full {

struct FullOperation {
    struct operation_attributes_t {
        const ttnn::SmallVector<uint32_t> shape;
        const std::variant<float, int> fill_value;
        const DataType dtype;
        const Layout layout;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& any;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle writer_id{};
            std::optional<tt::tt_metal::KernelHandle> reader_id = std::nullopt;
            std::vector<tt::tt_metal::CoreCoord> cores{};
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

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        ttnn::SmallVector<uint32_t> shape,
        std::variant<float, int> fill_value,
        const Tensor& any,
        const std::optional<DataType>& dtype,
        const std::optional<Layout>& layout,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace ttnn::operations::full

namespace ttnn::prim {
constexpr auto full = ttnn::register_operation<"ttnn::prim::full", ttnn::operations::full::FullOperation>();
}  // namespace ttnn::prim
