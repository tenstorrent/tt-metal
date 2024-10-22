// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::full_like {

struct FullLikeOperation {
    struct operation_attributes_t {
        const std::variant<float, int> fill_value;
        const DataType dtype;
        const Layout layout;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using shape_return_value_t = SimpleShape;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            KernelHandle writer_kernel_id;
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

    using program_factory_t = std::variant<ProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const std::variant<float, int> fill_value,
        const std::optional<DataType>& dtype,
        const std::optional<Layout>& layout,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace ttnn::operations::full_like

namespace ttnn::prim {
constexpr auto moreh_full_like =
    ttnn::register_operation<"ttnn::prim::moreh_full_like", ttnn::operations::full_like::FullLikeOperation>();
}  // namespace ttnn::prim
