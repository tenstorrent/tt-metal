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
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SparseSDPAProgramFactory {
        // The split-K and L1 choices depend on the free L1, so compute_program_hash hashes them.
        struct Plan {
            bool split_k = false;
            uint32_t num_used = 0;   // cores that get work
            uint32_t passes = 1;     // head passes per token (the Q tile rows that fit L1 at once)
            bool fused = true;       // fused recipe chunks (their CBs fit L1)
            uint32_t group = 1;      // gathered chunks per recipe step (sparse_step_group)
            uint32_t q_rm_rows = 2;  // Q tile rows the reader stages (2: one ahead of compute's tilize; 1: L1 floor)
            uint32_t k_rm_rows = 0;  // cb_k_rm rows (0: one chunk / scaled two slabs; < one chunk: 32-row gathers)
        };
        static Plan select_plan(const operation_attributes_t& attrs, const tensor_args_t& t);

        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& output);

        // Cache-hit re-apply of the per-dispatch state the hash excludes: buffer addresses and
        // kv_batch_page_offset (cache_batch_idx * kv length T). Patched in place; no rebuild.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
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
};

Tensor sparse_sdpa(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& indices,
    float scale,
    uint32_t v_dim,
    transformer::SparseKVFormat kv_format,
    uint32_t k_chunk_size,
    transformer::SDPAPrecision precision,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt,
    const std::optional<Tensor>& attention_sink = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim
