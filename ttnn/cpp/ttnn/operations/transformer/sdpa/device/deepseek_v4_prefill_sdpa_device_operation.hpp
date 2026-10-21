// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct DeepseekV4PrefillSDPAOperation {
    using operation_attributes_t = DeepseekV4PrefillSDPAParams;
    using tensor_args_t = DeepseekV4PrefillSDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& output);

        // Cache hit: refresh the buffer addresses only.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor deepseek_v4_prefill_sdpa(
    const Tensor& q,
    const Tensor& window_kv,
    const Tensor& start,
    const std::optional<Tensor>& cache_kv,
    const std::optional<Tensor>& cache_idx,
    float scale,
    uint32_t v_dim,
    uint32_t window_size,
    uint32_t k_chunk_size,
    uint32_t compress_ratio,
    bool empty_tail,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& attention_sink);

}  // namespace ttnn::prim
