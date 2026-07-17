// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation_types.hpp"
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

struct SparseSDPAMsaOperation {
    using operation_attributes_t = SparseSDPAMsaParams;
    using tensor_args_t = SparseSDPAMsaInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SparseSDPAMsaProgramFactory {
        // The MeshCoordinate overload opts this op into per-coordinate program creation, so each device bakes
        // its own causal chunk_start (see compute_chunk_start_local). Without it the mesh adapter builds one
        // program for the whole device range and every rank shares rank 0's offset.
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& t,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<SparseSDPAMsaProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks invariants excluded from the program hash, such as interleaved K/V length and cache_batch_idx.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Per-device causal start: chunk_start_idx + rank*S along cluster_axis (rank from the coordinate; 0 on a
    // single device or when non-causal). Shared by create_descriptor at miss and hit time so the two cannot drift.
    static uint32_t compute_chunk_start_local(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Cache-hit re-apply of all per-dispatch state (per-core K/V offsets, group strides, causal chunk_start,
    // buffer addresses), since the hash excludes interleaved K/V T and cache_batch_idx. See the .cpp.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

Tensor sparse_sdpa_msa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    float scale,
    uint32_t block_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::prim
