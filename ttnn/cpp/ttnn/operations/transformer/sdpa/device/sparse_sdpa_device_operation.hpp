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

// Reader/writer runtime-arg layout — single source of truth shared by the program factory (which EMITS the
// args in create_descriptor) and get_dynamic_runtime_args (which RE-APPLIES the indexed-cache page offset to
// the cached program). kv_batch_page_offset sits at the fixed index below and is the only re-applied arg. If
// the order before kv_batch_page_offset changes in the factory, update these indices here, or the dynamic
// re-apply silently writes the wrong slot. Block-cyclic remap configuration is compile-time.
namespace sparse_sdpa_rt {
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterKernelIdx = 1;
inline constexpr uint32_t kReaderBatchOffsetArg = 5;  // {q, kv, idx, tok_start, tok_count, [kv_batch_page_offset]}
inline constexpr uint32_t kWriterBatchOffsetArg = 4;  // {out, tok_start, tok_count, kv, [kv_batch_page_offset]}
}  // namespace sparse_sdpa_rt

struct SparseSDPAOperation {
    using operation_attributes_t = SparseSDPAParams;
    using tensor_args_t = SparseSDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SparseSDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& output);
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
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
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
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt);

}  // namespace ttnn::prim
