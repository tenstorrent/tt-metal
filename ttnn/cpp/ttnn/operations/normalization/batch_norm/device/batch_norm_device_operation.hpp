// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization {
struct BatchNormOperation {
    struct operation_attributes_t {
        const float eps;
        const float momentum;
        const bool training;
        const MemoryConfig memory_config;

        DataType input_dtype;
        std::optional<DataType> dtype;
        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& batch_mean;
        const Tensor& batch_var;
        std::optional<Tensor> weight;
        std::optional<Tensor> bias;
        std::optional<Tensor> output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct BatchNormFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreCoord compute_with_storage_grid_size;
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

    using program_factory_t = std::variant<BatchNormFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const Tensor& batch_mean,
        const Tensor& batch_var,
        const float eps,
        const float momentum,
        const bool training,
        std::optional<Tensor> weight,
        std::optional<Tensor> bias,
        std::optional<Tensor> output,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
constexpr auto batch_norm =
    ttnn::register_operation<"ttnn::prim::batch_norm", ttnn::operations::normalization::BatchNormOperation>();
}
