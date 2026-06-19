// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct SparseSDPAOperation {
    using operation_attributes_t = SparseSDPAParams;
    using tensor_args_t = SparseSDPAInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SparseSDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SparseSDPAProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-validates only the inputs the program hash does not key on (so they can vary on a cache hit): the kv
    // shape/layout (its length T rides on runtime args) and cache_batch_idx (a dynamic runtime arg). All else
    // is pinned by the hash, so a hit already passed full validation at miss time.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // cache_batch_idx is excluded from the program hash, so it must be re-applied to the cached program on
    // every dispatch (the non-Buffer analog of buffer-address patching). Returns the per-core gather page
    // offset (cache_batch_idx * T) for the reader/writer when indexed; empty otherwise.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

Tensor sparse_sdpa(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& indices,
    float scale,
    uint32_t v_dim,
    uint32_t k_chunk_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx = std::nullopt);

}  // namespace ttnn::prim
