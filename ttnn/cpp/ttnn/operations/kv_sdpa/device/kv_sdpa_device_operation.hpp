// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::kv_sdpa {

// Specialized fused-flash scaled-dot-product attention for the small-query MQA case.
//
//   Q is [1, NQH, Sq, DH] with Sq == one tile (32); K/V are [1, NKH, KV, DH] with NKH dividing NQH
//   (GQA/MQA); non-causal full attention. Output is [1, NQH, Sq, DH].
//
// One core per Q head runs the production transformer-SDPA online-softmax routine (sdpa_standard),
// specialized to Sq == 1 tile / single-KV-head reuse, so the attention math matches the production
// fused-flash kernel without its general-case overhead.
struct KvSdpaDeviceOperation {
    struct operation_attributes_t {
        uint32_t scale_bits;  // fp32 bit-pattern of the softmax scale
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    };

    struct tensor_args_t {
        const Tensor& q;
        const Tensor& k;  // new/suffix K  [1, NKH, Skv, DH]
        const Tensor& v;  // new/suffix V
        // Optional attention mask: additive bf16 mask [1, 1, Sq, KV] over the full folded KV
        // ([prefix ; suffix]; column-tile g aligned to KV-tile g). Applied when provided
        // (use_provided_mask compile path); omit (None) for the fast unmasked non-causal path.
        std::optional<Tensor> mask;
        // Optional resident prefix K/V [1, NKH, prefix, DH]. When provided, attention is over the
        // concatenation [past_k ; k] / [past_v ; v] -- read as two ranges in the reader, so the caller
        // does NOT pre-concatenate (saves the ttnn.concat ops + the [prefix+suffix] materialization).
        std::optional<Tensor> past_k;
        std::optional<Tensor> past_v;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    // One core per Q head; the compute kernel calls the transformer-SDPA sdpa_standard() flash loop.
    struct FlashFused {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<FlashFused>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::kv_sdpa

namespace ttnn::prim {
ttnn::operations::kv_sdpa::KvSdpaDeviceOperation::tensor_return_value_t kv_sdpa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::optional<Tensor> mask,
    uint32_t scale_bits,
    std::optional<Tensor> past_k,
    std::optional<Tensor> past_v,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);
}  // namespace ttnn::prim
