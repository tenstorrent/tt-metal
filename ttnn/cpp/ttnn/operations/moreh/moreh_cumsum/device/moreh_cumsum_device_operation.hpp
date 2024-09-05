// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {
struct MorehCumsumDeviceOperation {
    struct operation_attributes_t {
        int64_t dim;
        bool flip;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& output;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input, const Tensor& output, const int64_t dim, const bool flip);
};
}  // namespace ttnn::operations::moreh::moreh_cumsum

namespace ttnn::prim {
constexpr auto moreh_cumsum =
    ttnn::register_operation<"ttnn::prim::moreh_cumsum", ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation>();
}
