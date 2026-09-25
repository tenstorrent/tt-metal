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
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/block_cyclic_causal_geometry.hpp"

namespace ttnn::prim {

struct SparseSDPAMsaOperation {
    using operation_attributes_t = SparseSDPAMsaParams;
    using tensor_args_t = SparseSDPAMsaInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
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

        // Cache-hit re-apply of all per-dispatch state (buffer addresses, per-core K/V offsets, group strides,
        // causal chunk_start), since the hash excludes interleaved K/V T and cache_batch_idx. See the .cpp.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& attrs,
            const tensor_args_t& t,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<SparseSDPAMsaProgramFactory>;

    // Runtime-arg slots, named so create_descriptor's emplace order and override_runtime_arguments'
    // in-place writes reference the same symbols instead of agreeing on bare positions.
    enum ReaderArg : uint32_t {
        kReaderQAddr,
        kReaderKAddr,
        kReaderVAddr,
        kReaderIdxAddr,
        kReaderWorkStart,
        kReaderWorkCount,
        kReaderKBatchOffset,
        kReaderVBatchOffset,
        kReaderKGroupStride,
        kReaderVGroupStride,
        kReaderChunkStart,
        kReaderStraddleQ,  // host-path geometry: rows >= this jump by kReaderStraddleJump (mid-slab start)
        kReaderStraddleJump,
        kReaderChunkStartMeta,  // chunk_start_idx_tensor address (0 = host path)
        kReaderDeviceIndex,     // SP rank for the in-kernel geometry on the metadata path
        kReaderSlotMeta,        // cache_batch_idx_tensor address (0 = host path)
        kReaderKSlotStride,     // K/V tiles per cache slot, for the on-device slot offset
        kReaderVSlotStride,
        kReaderNumLayers,
        kReaderLayerIdx,
        kReaderCacheSlots,  // K/V batch extent, bounds the recomposed slot
        kReaderArgCount,
    };
    enum WriterArg : uint32_t {
        kWriterOutAddr,
        kWriterWorkStart,
        kWriterWorkCount,
        kWriterKAddr,
        kWriterVAddr,
        kWriterKBatchOffset,
        kWriterVBatchOffset,
        kWriterKGroupStride,
        kWriterVGroupStride,
        kWriterSlotMeta,  // the writer co-gathers the lower K/V halves, so it selects the slot too
        kWriterKSlotStride,
        kWriterVSlotStride,
        kWriterNumLayers,
        kWriterLayerIdx,
        kWriterCacheSlots,
        kWriterArgCount,
    };
    enum ComputeArg : uint32_t { kComputeWorkStart, kComputeWorkCount, kComputeArgCount };

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks invariants excluded from the program hash, such as interleaved K/V length and cache_batch_idx.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // This device's rank for the causal geometry: along cluster_axis, or linear over the mesh when unset (0 on a
    // single device).
    static uint32_t compute_device_index(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    // Per-device causal geometry from the host chunk_start_idx, via the same closed form the kernel applies to
    // the metadata tensor (and the indexer applies in tiles): linear chunk_start_idx + rank*S for contiguous
    // K/V; rotation-exact, with the boundary chip's slab straddle, for a block-cyclic cache. Zero when
    // non-causal or on the metadata path (the kernel derives it).
    static tt::block_cyclic::CausalGeometry compute_causal_geometry(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Work split plus every scalar compute_program_hash excludes: interleaved K/V T and batch slots, n_kv,
    // cache_batch_idx, chunk_start_idx/cluster_axis, the slot-fold layer terms, and the metadata tensor addresses.
    // Single-sourced so create_descriptor (miss-bake) and override_runtime_arguments (hit-patch) write the same values
    // and cannot drift.
    struct DispatchArgs {
        tt::tt_metal::CoreCoord grid;  // per-core arg order: core i == {i % grid.x, i / grid.x}
        uint32_t num_cores = 0;
        uint32_t base_work = 0;  // total_work = S * n_kv over num_cores; the first `extra` cores take one more
        uint32_t extra = 0;
        uint32_t k_batch_tile_offset = 0;
        uint32_t v_batch_tile_offset = 0;
        uint32_t k_group_tile_stride = 0;
        uint32_t v_group_tile_stride = 0;
        tt::block_cyclic::CausalGeometry geometry{};
        uint32_t device_index = 0;
        uint32_t k_slot_tile_stride = 0;  // n_kv * k_group_tile_stride
        uint32_t v_slot_tile_stride = 0;
        uint32_t cache_slots = 0;  // K/V batch extent
        uint32_t chunk_start_meta_addr = 0;
        uint32_t slot_meta_addr = 0;
    };
    static DispatchArgs compute_dispatch_args(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
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
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt,
    const std::optional<Tensor>& chunk_start_idx_tensor = std::nullopt,
    const std::optional<Tensor>& cache_batch_idx_tensor = std::nullopt,
    uint32_t index_cache_num_layers = 1,
    uint32_t index_cache_layer_idx = 0);

}  // namespace ttnn::prim
