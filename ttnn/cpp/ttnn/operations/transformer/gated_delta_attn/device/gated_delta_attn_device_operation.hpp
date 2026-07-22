// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_device_operation_types.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_program_factory.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Device operation returning two tensors: [output_y, final_state].
struct GatedDeltaAttnSeqDeviceOperation {
    using operation_attributes_t = GatedDeltaAttnSeqParams;
    using tensor_args_t = GatedDeltaAttnSeqInputs;

    // Return two Tensors: output [BH, num_chunks, C, Dv] and final_state [BH, Dk, Dv].
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GatedDeltaAttnSeqProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Low-level dispatch function (used by public API).
std::vector<Tensor> gated_delta_attn_seq(
    const Tensor& L_unit,
    const Tensor& v_beta_sc,
    const Tensor& k_bd_sc,
    const Tensor& intra_attn,
    const Tensor& q_decay,
    const Tensor& k_decay_t,
    const Tensor& dl_exp,
    const Tensor& L_inv,
    const std::optional<Tensor>& initial_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
