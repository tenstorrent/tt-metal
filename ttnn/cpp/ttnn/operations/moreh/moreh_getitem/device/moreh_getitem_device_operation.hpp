// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
struct MorehGetItemOperation {
    struct operation_attributes_t {
        const std::vector<uint32_t> index_dims;
        // const CoreRange core_range;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::vector<Tensor>& index_tensors;
        const std::optional<Tensor>& output;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct MorehGetItemRmFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            uint32_t core_h;
            std::vector<uint32_t> index_dims;
            uint32_t input_dim_offset;
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

    struct MorehGetItemTilizedFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            uint32_t core_h;
            std::vector<uint32_t> index_dims;
            uint32_t input_dim_offset;
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

    using program_factory_t = std::variant<MorehGetItemRmFactory, MorehGetItemTilizedFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const std::vector<Tensor>& index_tensors,
        const std::vector<uint32_t> index_dims,
        const std::optional<Tensor>& output,
        // const CoreRange core_range,
        const std::optional<MemoryConfig> memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_getitem

namespace ttnn::prim {
constexpr auto moreh_getitem = ttnn::
    register_operation<"ttnn::prim::moreh_getitem", ttnn::operations::moreh::moreh_getitem::MorehGetItemOperation>();
}
