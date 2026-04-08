// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::examples {

struct ExampleDeviceOperation {
    // Define the operation attributes. This is used to store all variables needed by operations that aren't tensors.
    struct operation_attributes_t {
        bool attribute;
        int some_other_attribute;
    };

    // Define the tensor arguments. This is used to store all tensors passed in and/or out of the operation.
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        // This example will use a tensor that can only be used as an input
        const Tensor& input_tensor;

        // However, the following examples show what else can be done with tensor_args_t

        // An example of the tensor that can be used for input/output or just for pre-allocated output
        // Tensor& io_tensor;

        // An example of an optional tensor
        // std::optional<Tensor> optional_output_tensor;

        // An example of a vector of tensors
        // std::vector<Tensor> vector_of_tensors;

        // An example of a vector of optional tensors
        // std::vector<std::optional<Tensor>> vector_of_optional_tensors;
    };

    // Define the return types for the spec(s) of the operation.
    // Can be a single ttnn::TensorSpec, std::optional<ttnn::TensorSpec>, std::vector<ttnn::TensorSpec>,
    // std::tuple<ttnn::TensorSpec, ...> etc.
    using spec_return_value_t = ttnn::TensorSpec;

    // Define the return types for the tensor(s) of the operation.
    // Can be a single Tensor, std::optional<Tensor>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    // Note: spec_return_value_t and tensor_return_value_t should follow the same pattern.
    // i.e. if spec_return_value_t is a std::vector<std::optional<ttnn::TensorSpec>> then tensor_return_value_t should
    // be std::vector<std::optional<Tensor>>

    // Returns a declarative ProgramDescriptor. The framework handles program
    // construction, caching, and runtime argument patching automatically.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Validate the operation when it creates a program. Also called on cache hit by default.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Optional: override to use lighter validation on cache hit.
    // If not provided, the framework calls validate_on_program_cache_miss.
    // static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output specs based on the operation attributes and tensor args.
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args.
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // -------------------------------------------------------------------------
    // Multi-variant programs (advanced)
    //
    // When an operation needs different program strategies (e.g. work
    // distribution that depends on input size), define named structs with
    // create_descriptor and put them in a variant:
    //
    //   struct SmallInput {
    //       static tt::tt_metal::ProgramDescriptor create_descriptor(
    //           const operation_attributes_t&,
    //           const tensor_args_t&,
    //           tensor_return_value_t&);
    //   };
    //   struct LargeInput {
    //       static tt::tt_metal::ProgramDescriptor create_descriptor(
    //           const operation_attributes_t&,
    //           const tensor_args_t&,
    //           tensor_return_value_t&);
    //   };
    //   using program_factory_t = std::variant<SmallInput, LargeInput>;
    //
    //   static program_factory_t select_program_factory(
    //       const operation_attributes_t&, const tensor_args_t&);
    // -------------------------------------------------------------------------
};

}  // namespace ttnn::operations::examples

namespace ttnn::prim {
ttnn::operations::examples::ExampleDeviceOperation::tensor_return_value_t example(const Tensor& input_tensor);
}  // namespace ttnn::prim
