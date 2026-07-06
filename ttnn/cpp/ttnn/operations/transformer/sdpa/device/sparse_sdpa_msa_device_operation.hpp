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
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"

namespace ttnn::prim {

// Runtime-arg slots patched on cache hits for indexed K/V caches and runtime K/V group strides.
// Keep these in sync with create_descriptor().
namespace sparse_sdpa_msa_rt {
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterKernelIdx = 1;
// reader args: {q, k, v, idx, work_start, work_count, k_batch_tile_offset, v_batch_tile_offset,
//               k_group_tile_stride, v_group_tile_stride}
inline constexpr uint32_t kReaderKBatchOffsetArg = 6;
inline constexpr uint32_t kReaderVBatchOffsetArg = 7;
inline constexpr uint32_t kReaderKGroupStrideArg = 8;
inline constexpr uint32_t kReaderVGroupStrideArg = 9;
// Per-device causal chunk_start (chunk_start_idx + rank*S); patched at dispatch when causal masking is on.
inline constexpr uint32_t kReaderChunkStartArg = 10;
// writer args: {out, work_start, work_count, k, v, k_batch_tile_offset, v_batch_tile_offset,
//               k_group_tile_stride, v_group_tile_stride}
inline constexpr uint32_t kWriterKBatchOffsetArg = 5;
inline constexpr uint32_t kWriterVBatchOffsetArg = 6;
inline constexpr uint32_t kWriterKGroupStrideArg = 7;
inline constexpr uint32_t kWriterVGroupStrideArg = 8;
}  // namespace sparse_sdpa_msa_rt

struct SparseSDPAMsaOperation {
    using operation_attributes_t = SparseSDPAMsaParams;
    using tensor_args_t = SparseSDPAMsaInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // ProgramDescriptor factory, invoked PER mesh coordinate by the adapter so each device bakes its own
    // per-device causal chunk_start (chunk_start_idx + device_index*S) from its coordinate -- mirrors the
    // indexer_score (MSA sibling) create_at, and matches the SDPA sibling ring_joint_sdpa's adapter wiring.
    struct SparseSDPAMsaProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& t,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    // Minimal operation-shaped helper so the descriptor factory can be adapted into a mesh workload.
    struct DescriptorAdapterOperation {
        using operation_attributes_t = SparseSDPAMsaOperation::operation_attributes_t;
        using tensor_args_t = SparseSDPAMsaOperation::tensor_args_t;
        using spec_return_value_t = SparseSDPAMsaOperation::spec_return_value_t;
        using tensor_return_value_t = SparseSDPAMsaOperation::tensor_return_value_t;
    };

    // Wraps the ProgramDescriptor factory into a per-coordinate mesh workload: create runs once per device
    // (so chunk_start is baked per-device), and override_runtime_arguments re-applies buffer bindings plus the
    // hash-excluded scalars (cache-slot offsets, GQA strides, and per-device chunk_start) on cache hits.
    struct MeshWorkloadFactory {
        using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
            DescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<SparseSDPAMsaProgramFactory>;
        using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& args,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<MeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks invariants excluded from the program hash, such as interleaved K/V length and cache_batch_idx.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
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
