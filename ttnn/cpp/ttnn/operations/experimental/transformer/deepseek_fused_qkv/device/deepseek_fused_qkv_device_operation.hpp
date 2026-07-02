// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv {

// -----------------------------------------------------------------------------
// DeepseekFusedQkvDeviceOperation
//
// Fuses the deepseek_v4_flash decode `_qkv` calc (attention.py lines 680-696)
// into one monolithic device op with DRAM-sharded weights:
//
//   Q  path:  q_a = rmsnorm_w(hidden @ Wqa)           [1, q_lora]
//             q   = q_a @ Wqb                          [1, H*Dh]
//             q   = reshape [1, 1, H, Dh]
//             q   = rmsnorm_unweighted(q, over Dh)     (per head)
//             q   = partial_rope(q, cos, sin, Rd)      [1, 1, H, Dh]
//
//   KV path (independent, runs on a disjoint parallel core partition):
//             kv  = rmsnorm_w(hidden @ Wkv)            [1, Dh]
//             kv  = partial_rope(kv, cos, sin, Rd)     [1, 1, 1, Dh]
//
// Weights (Wqa/Wqb/Wkv) are DRAM WIDTH_SHARDED across the device's DRAM banks,
// read with the "saturating DRAM bandwidth" reader-per-bank pattern. The two
// paths execute concurrently in a single program; Q's q_a->q_b transition does
// an in-program cross-core reduce (RMSNorm over q_lora) + re-broadcast.
// -----------------------------------------------------------------------------
struct DeepseekFusedQkvDeviceOperation {
    struct operation_attributes_t {
        float eps;
        uint32_t rope_dim;
        uint32_t num_heads;
        MemoryConfig q_mem_config;
        MemoryConfig kv_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& hidden;
        const Tensor& wqa;
        const Tensor& wqb;
        const Tensor& wkv;
        const Tensor& qa_norm_w;
        const Tensor& kv_norm_w;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
    };

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct MultiCoreProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_fused_qkv(
    const ttnn::Tensor& hidden,
    const ttnn::Tensor& wqa,
    const ttnn::Tensor& wqb,
    const ttnn::Tensor& wkv,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kv_norm_w,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    float eps,
    uint32_t rope_dim,
    uint32_t num_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& q_mem_config,
    const std::optional<tt::tt_metal::MemoryConfig>& kv_mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
