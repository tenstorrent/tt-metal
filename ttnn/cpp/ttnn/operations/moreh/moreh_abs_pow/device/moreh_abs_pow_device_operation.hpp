// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

#define MOREH_ABS_POW_FACTORY_H(name)                                                       \
    struct name {                                                                           \
        struct shared_variables_t {                                                         \
            tt::tt_metal::KernelHandle reader_kernels_id;                                   \
            tt::tt_metal::KernelHandle writer_kernels_id;                                   \
            std::size_t num_cores_to_be_used;                                               \
            std::size_t num_cores_y;                                                        \
        };                                                                                  \
                                                                                            \
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>; \
                                                                                            \
        static cached_program_t create(                                                     \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output_tensor);                                          \
                                                                                            \
        static void override_runtime_arguments(                                             \
            cached_program_t& cached_program,                                               \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output_tensor);                                          \
    };

namespace ttnn::operations::moreh::moreh_abs_pow {

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p);

struct MorehAbsPowOperation {
    struct operation_attributes_t {
        const float p;

        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    MOREH_ABS_POW_FACTORY_H(MorehAbsPowFactory)

    using program_factory_t = std::variant<MorehAbsPowFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        float p,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_abs_pow

namespace ttnn::prim {
constexpr auto moreh_abs_pow = ttnn::
    register_operation<"ttnn::prim::moreh_abs_pow", ttnn::operations::moreh::moreh_abs_pow::MorehAbsPowOperation>();
}  // namespace ttnn::prim
