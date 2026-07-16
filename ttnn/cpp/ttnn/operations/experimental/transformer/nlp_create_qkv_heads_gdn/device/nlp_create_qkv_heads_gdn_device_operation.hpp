// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

// GDN fork of nlp_create_qkv_heads: splits a fused token-major [B, 1, S, (Nq+Nk+Nv)*head_dim]
// input into head-major Q [B, Nq, S, head_dim], K [B, Nk, S, head_dim], V [B, Nv, S, head_dim].
// Unlike the base op, Q/K/V may each have an INDEPENDENT head count (GDN uses Nq==Nk!=Nv), the K
// head-transpose is dropped (always false), and only the INTERLEAVED path is supported. head_dim
// is shared across Q/K/V (caller must guarantee Dk==Dv).

namespace ttnn::operations::experimental::transformer {

struct NlpCreateHeadsGdnDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_q_heads;
        uint32_t num_k_heads;
        uint32_t num_v_heads;
        uint32_t head_dim;
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::vector<std::optional<Tensor>> optional_output_tensors;
    };

    using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec, ttnn::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

    struct Interleaved {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<Interleaved>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_gdn(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim
