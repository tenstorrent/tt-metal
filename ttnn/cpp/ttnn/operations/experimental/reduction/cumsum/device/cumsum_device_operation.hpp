// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reduction {

struct CumSumDeviceOperation {
    struct operation_attributes_t {
        const int64_t dim;  // axis to perform cumsum on (must be `-tensor.dim <= dim < tensor.dim`)
        const tt::tt_metal::DataType dtype = tt::tt_metal::DataType::INVALID;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> preallocated_output;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {};
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

    using program_factory_t = std::variant<SingleCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        int64_t dim,
        std::optional<ttnn::DataType> dtype,
        std::optional<Tensor> preallocated_output);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::prim {

constexpr auto cumsum =
    ttnn::register_operation<"ttnn::prim::cumsum", ttnn::operations::experimental::reduction::CumSumDeviceOperation>();

}  // namespace ttnn::prim
