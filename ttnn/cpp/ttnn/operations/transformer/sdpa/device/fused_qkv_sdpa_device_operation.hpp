// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/fused_qkv_sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Prefill SDPA that reads Q, K and V out of one fused projection output instead of three tensors,
// so the head split never runs as its own op. head_dim must be a whole number of tiles, which makes
// the split pure address arithmetic in the reader -- see kernels/dataflow/fused_qkv_reader.cpp.
//
// Deliberately narrow next to SDPAOperation: non-causal only, no paging, chunking, sliding window,
// attention sink or MLA. Everything it does not support is rejected in validation rather than
// silently ignored.
struct FusedQKVSDPAOperation {
    using operation_attributes_t = FusedQKVSDPAParams;
    using tensor_args_t = FusedQKVSDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct FusedQKVSDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<FusedQKVSDPAProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attrs, const tensor_args_t&);
};

Tensor fused_qkv_sdpa(
    const Tensor& input_tensor_qkv,
    const std::optional<Tensor>& attn_mask,
    uint32_t num_heads,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim
