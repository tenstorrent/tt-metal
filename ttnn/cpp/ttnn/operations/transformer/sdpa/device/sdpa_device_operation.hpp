// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct SDPAOperation {
    using operation_attributes_t = SDPAParams;
    using tensor_args_t = SDPAInputs;
    // Multi-output (mirrors JointSDPA under the same device_operation::launch framework):
    // [out] when return_lse is false, or [out, lse] when true. Existing callers go through the
    // free function ttnn::prim::sdpa which returns element [0], so they are untouched.
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct SDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SDPAProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attrs, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

// Single-output façade: returns element [0] of the launch result. Every existing internal caller
// (flash_mla, ring_distributed, chunked, decode) is untouched and always gets exactly one output
// tensor from a program with return_lse=false.
Tensor sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<Tensor>& chunk_start_idx_tensor,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& cu_window_seqlens = std::nullopt);

// T6 multi-output entry: sets return_lse=true and returns [out, lse]. Used only by the new
// return_lse public API path; leaves the single-output free function above byte-identical.
std::vector<Tensor> sdpa_with_lse(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<Tensor>& chunk_start_idx_tensor,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& cu_window_seqlens = std::nullopt);

}  // namespace ttnn::prim
