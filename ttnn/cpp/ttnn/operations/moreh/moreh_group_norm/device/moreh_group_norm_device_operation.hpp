// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
struct MorehGroupNormOperation {
    struct operation_attributes_t {
        const uint32_t num_groups;
        const float eps;
        const std::vector<bool> are_required_outputs;
        const MemoryConfig memory_config;
        const MemoryConfig mean_memory_config;
        const MemoryConfig rstd_memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<const Tensor> gamma;
        const std::optional<const Tensor> beta;
        const std::optional<const Tensor> output;
        const std::optional<const Tensor> mean;
        const std::optional<const Tensor> rstd;
    };

    using shape_return_value_t = std::vector<std::optional<SimpleShape>>;
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct MorehGroupNormFactory {
        struct shared_variables_t {
            KernelHandle reader_kernels_id;
            KernelHandle writer_kernels_id;
            uint32_t num_cores_to_be_used;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& outputs);

        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& outputs);
    };

    using program_factory_t = std::variant<MorehGroupNormFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const uint32_t num_groups,
        const float eps,
        const std::optional<const Tensor> gamma,
        const std::optional<const Tensor> beta,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor> output,
        const std::optional<const Tensor> mean,
        const std::optional<const Tensor> rstd,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<MemoryConfig>& mean_memory_config,
        const std::optional<MemoryConfig>& rstd_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_group_norm

namespace ttnn::prim {
constexpr auto moreh_group_norm =
    ttnn::register_operation<"ttnn::prim::moreh_group_norm",
                             ttnn::operations::moreh::moreh_group_norm::MorehGroupNormOperation>();
}
