// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1 {

struct MorehClipGradNormStep1Operation {
    struct operation_attributes_t {
        const float norm_type;
        const uint32_t tile_offset_of_tmp_pow_sum;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const std::vector<Tensor>& inputs;
        const Tensor& tmp_pow_sum;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            uint32_t num_cores_to_be_used;
            size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tmp_pow_sum);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tmp_pow_sum);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor>& inputs,
        float norm_type,
        uint32_t tile_offset_of_tmp_pow_sum,
        const Tensor& tmp_pow_sum,
        const std::optional<MemoryConfig>& memory_config,
        const DeviceComputeKernelConfig& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1

namespace ttnn::prim {
constexpr auto moreh_clip_grad_norm_step1 = ttnn::register_operation<
    "ttnn::prim::moreh_clip_grad_norm_step1",
    ttnn::operations::moreh::moreh_clip_grad_norm_step1::MorehClipGradNormStep1Operation>();
}  // namespace ttnn::prim
