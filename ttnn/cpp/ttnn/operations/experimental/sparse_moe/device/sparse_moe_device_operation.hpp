// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sparse MoE Expert Matmul: reads only active expert weights from DRAM.
// Fuses: gate_up matmul → SiLU → scale by routing weight → down matmul → accumulate
// Saves ~50% DRAM bandwidth by skipping inactive expert weights.

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_moe {

struct SparseMoeExpertOperation {
    struct operation_attributes_t {
        uint32_t num_experts;       // 64 per device
        uint32_t expert_inter_dim;  // 512 (moe_intermediate_size)
        uint32_t hidden_dim;        // 2048
        uint32_t batch_size;        // 32
    };

    struct tensor_args_t {
        const Tensor& input;        // (1, 1, batch, hidden) bf16
        const Tensor& expert_gu;    // (1, 1, hidden, num_experts*2*inter) bfp4 — packed gate+up
        const Tensor& expert_dw;    // (1, 1, num_experts*inter, hidden) bfp4 — packed down
        const Tensor& expert_mask;  // (1, 1, batch, num_experts) bf16 — routing weights (0=inactive)
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
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<SingleCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::sparse_moe
