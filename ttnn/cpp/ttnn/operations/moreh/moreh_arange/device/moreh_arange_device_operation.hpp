// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"

namespace ttnn::operations::moreh::moreh_arange {

struct MorehArangeOperation {
    struct operation_attributes_t {
        float start;
        float end;
        float step;
        bool untilize_out;
        const DataType dtype;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& any;
        const std::optional<Tensor>& output;
    };

    using shape_return_value_t = Shape;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            KernelHandle kernel_id;
            std::size_t num_cores;
            std::size_t core_h;
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
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        float start,
        float end,
        float step,
        const Tensor& any,
        const std::optional<Tensor>& output,
        bool untilize_out,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace ttnn::operations::moreh::moreh_arange

namespace ttnn::prim {
constexpr auto moreh_arange =
    ttnn::register_operation<"ttnn::prim::moreh_arange", ttnn::operations::moreh::moreh_arange::MorehArangeOperation>();
}
