// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetRecurrenceOperation {
    // Operation attributes: DeltaNet architecture params
    struct operation_attributes_t {
        uint32_t num_heads;    // 48
        uint32_t head_k_dim;   // 128
        uint32_t head_v_dim;   // 128
        uint32_t num_k_heads;  // 16
        uint32_t gqa_ratio;    // 3
        float scale;           // 1/sqrt(128)
        float norm_eps;        // 1e-6
    };

    // Tensor arguments
    struct tensor_args_t {
        const Tensor& conv_out;     // (1,1,B_pad,conv_dim=10240) after conv1d+silu
        const Tensor& b_proj;       // (1,1,B_pad,num_v_heads=48) beta gate input
        const Tensor& a_proj;       // (1,1,B_pad,num_v_heads=48) alpha gate input
        const Tensor& z_proj;       // (1,1,B_pad,value_dim=6144) gating input
        const Tensor& dt_bias;      // (1,1,1,num_v_heads) dt bias
        const Tensor& A_exp;        // (1,1,1,num_v_heads) A exponential
        const Tensor& norm_weight;  // (1,1,1,head_v_dim) norm weight
        const Tensor& state;        // (1,num_v_heads,head_k_dim,head_v_dim) persistent state (buffer mutated in-place)
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
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

    using program_factory_t = std::variant<SingleCore>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {
ttnn::operations::experimental::deltanet::DeltaNetRecurrenceOperation::tensor_return_value_t deltanet_recurrence(
    const Tensor& conv_out,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& z_proj,
    const Tensor& dt_bias,
    const Tensor& A_exp,
    const Tensor& norm_weight,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t head_k_dim,
    uint32_t head_v_dim,
    uint32_t num_k_heads,
    uint32_t gqa_ratio,
    float scale,
    float norm_eps);
}  // namespace ttnn::prim
