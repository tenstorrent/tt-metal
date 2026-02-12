// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Define the device operation for matmul + add

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::onboarding::exercise {

struct MatmulAddOperation {
    // TODO: Define operation_attributes_t, tensor_args_t, ProgramFactory, etc.
    // See solution for reference
    struct operation_attributes_t {};
    struct tensor_args_t {
        const Tensor& a;
        const Tensor& b;
        const Tensor& c;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_id;
            tt::tt_metal::KernelHandle writer_id;
            tt::tt_metal::KernelHandle compute_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<ProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::onboarding::exercise

namespace ttnn::prim {
ttnn::Tensor exercise_matmul_add(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c);
}  // namespace ttnn::prim
