// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

struct MorehAdamWDeviceOperation {
    struct operation_attributes_t {
        float lr = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weight_decay = 1e-2f;
        uint32_t step = 0;
        bool amsgrad = false;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& param_in;
        const Tensor& grad;
        const Tensor& exp_avg_in;
        const Tensor& exp_avg_sq_in;
        const std::optional<Tensor>& max_exp_avg_sq_in;

        const std::optional<Tensor>& param_out;
        const std::optional<Tensor>& exp_avg_out;
        const std::optional<Tensor>& exp_avg_sq_out;
        const std::optional<Tensor>& max_exp_avg_sq_out;
    };

    using spec_return_value_t = std::vector<std::optional<ttnn::TensorSpec>>;

    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_group1_id;
            tt::tt_metal::KernelHandle compute_kernel_group2_id;
            CoreRangeSet core_group_1;
            CoreRangeSet core_group_2;
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

    using program_factory_t = std::variant<MultiCore>;

    // Mandatory methods
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& param_in,
        const Tensor& grad,
        const Tensor& exp_avg_in,
        const Tensor& exp_avg_sq_in,

        std::optional<float> lr,
        std::optional<float> beta1,
        std::optional<float> beta2,
        std::optional<float> eps,
        std::optional<float> weight_decay,
        std::optional<uint32_t> step,
        std::optional<bool> amsgrad,

        const std::optional<Tensor>& max_exp_avg_sq_in,
        const std::optional<Tensor>& param_out,
        const std::optional<Tensor>& exp_avg_out,
        const std::optional<Tensor>& exp_avg_sq_out,
        const std::optional<Tensor>& max_exp_avg_sq_out,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::moreh::moreh_adamw

// Register the operation with the ttnn::register_operation API to make it available to the user as
// ttnn::prim::adamw
namespace ttnn::prim {
constexpr auto moreh_adamw = ttnn::
    register_operation<"ttnn::prim::moreh_adamw", ttnn::operations::moreh::moreh_adamw::MorehAdamWDeviceOperation>();
}  // namespace ttnn::prim
