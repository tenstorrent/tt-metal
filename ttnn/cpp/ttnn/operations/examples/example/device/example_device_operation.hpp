// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

struct ExampleDeviceOperation {
    struct operation_attributes_t {
        bool attribute;
        int some_other_attribute;
    };
    struct tensor_args_t {
        // An example of the tensor that can only be used as an input
        const Tensor& input_tensor;

        // An example of the tensor that can be used for input/output or just for pre-allocated output
        // Tensor& io_tensor;

        // An example of an optional tensor
        // std::optional<Tensor> optional_output_tensor;

        // An example of a vector of tensors
        // std::vector<Tensor> vector_of_tensors;

        // An example of a tuple of tensors
        // std::tuple<Tensor, ...> tuple_of_tensors;

        // An example of a vector of optional tensors
        // std::vector<std::optional<Tensor>> vector_of_optional_tensors;

        // An example of a tuple of tensors
        // std::tuple<std::vector<std::optional<Tensor>>, std::optional<Tensor>> some_crazy_tuple_of_tensors;
    };

    // Can be a single ttnn::Shape, std::optional<ttnn::Shape>, std::vector<ttnn::Shape>, std::tuple<ttnn::Shape> etc.
    using shape_return_value_t = ttnn::Shape;

    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {
            int some_variable_from_create_to_use_in_override_runtime_arguments;
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

    struct MultiCore {
        struct shared_variables_t {
            int some_variable_from_create_to_use_in_override_runtime_arguments;
            int some_other_variable_from_create_to_use_in_override_runtime_arguments;
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

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::example
