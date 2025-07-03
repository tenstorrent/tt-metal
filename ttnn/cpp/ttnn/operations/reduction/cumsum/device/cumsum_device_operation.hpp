// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction {

using namespace tt::tt_metal;
using namespace tt::stl;

struct CumSumDeviceOperation {
    struct operation_attributes_t {
        const int64_t dim;  // axis to perform cumsum on (must be `-tensor.dim <= dim < tensor.dim`)
        const tt::tt_metal::DataType dtype = tt::tt_metal::DataType::INVALID;
        const bool flip;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> preallocated_output;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle cumsum_reader_kernel_id;
            tt::tt_metal::KernelHandle cumsum_writer_kernel_id;

            // For multicore, define list of cores with `num_cores` and grid height (`num_cores_y`)
            // For i=0 to num_cores, Core coordinates {core.x, core.y} = {i / num_cores_y, i % num_cores_y}
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        int64_t dim,
        std::optional<ttnn::DataType> dtype,
        std::optional<Tensor> preallocated_output,
        const bool& flip);
};

}  // namespace ttnn::operations::reduction

namespace ttnn::prim {

constexpr auto cumsum =
    ttnn::register_operation<"ttnn::prim::cumsum", ttnn::operations::reduction::CumSumDeviceOperation>();

}  // namespace ttnn::prim
