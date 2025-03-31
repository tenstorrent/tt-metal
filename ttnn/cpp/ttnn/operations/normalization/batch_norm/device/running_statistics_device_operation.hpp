// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization {
struct RunningStatistics {
    struct operation_attributes_t {
        const float momentum;
        const MemoryConfig memory_config;

        DataType input_dtype;
        std::optional<DataType> dtype;
        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& batch_mean;
        const Tensor& batch_var;
        std::optional<Tensor> running_mean;
        std::optional<Tensor> running_var;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct RunningStatisticsProgramFactory {
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

    using program_factory_t = std::variant<RunningStatisticsProgramFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& batch_mean,
        const Tensor& batch_var,
        const float momentum,
        std::optional<Tensor> running_mean,
        std::optional<Tensor> running_var,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
constexpr auto running_statistics =
    ttnn::register_operation<"ttnn::prim::running_statistics", ttnn::operations::normalization::RunningStatistics>();
}
