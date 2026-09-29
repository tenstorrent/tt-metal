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
        // its own causal geometry (see compute_causal_geometry). Without it the mesh adapter builds one
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
        kReaderStraddleRow,  // host-path geometry: rows >= this jump by kReaderStraddleJump (mid-slab start)
        kReaderStraddleJump,
        kReaderChunkStartMeta,  // chunk_start_idx_tensor address (0 = host path)
        kReaderDeviceIndex,     // SP rank for the in-kernel geometry on the metadata path
        kReaderSlotMeta,        // cache_batch_idx_tensor address (0 = host path)
        kReaderKSlotStride,     // K/V tiles per cache slot, for the on-device slot offset
        kReaderVSlotStride,
        kReaderNumLayers,
        kReaderLayerIdx,
        kReaderCacheSlots,  // K/V batch extent, bounds the recomposed slot
        kReaderTpIndex,     // TP rank within the SP slab for the in-kernel geometry (TP-sub-sharded q; else 0)
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

    // Per-device causal geometry, in rows: the global position of this device's query row 0, plus the
    // straddle (query rows >= straddle_q sit straddle_jump positions further along; jump 0 = none). The type the
    // kernels use, so the host bakes exactly what the metadata path derives on device.
    using CausalGeometry = tt::block_cyclic::CausalGeometry;
    // This device's rank for the causal geometry: along cluster_axis, or linear over the mesh when unset (0 on a
    // single device).
    static uint32_t compute_device_index(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    // This device's TP rank within its SP rank's rows when q is also seq-sharded over TP (chunk_local == tp*S),
    // from the mesh axis other than cluster_axis. 0 unless the geometry is rotation-exact. Either chunk-start form:
    // the host geometry uses it, and the metadata path hands it to the kernel.
    static uint32_t compute_tp_index(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    // Block-cyclic cache + cluster_axis (the SP-sharded chunked-prefill cache read): the rotation-exact
    // geometry of the update_padded_kv_cache writer, shared with indexer_score (block_cyclic_causal_geometry.hpp)
    // so the diagonal-block mask and the indexer's selection agree on every query's position, including mid-slab
    // (non-chunk-aligned) starts. Otherwise the linear chunk_start_idx + rank*S along cluster_axis (rank from the
    // coordinate; 0 on a single device). All zeros when non-causal or on the metadata path (the kernel derives it).
    static CausalGeometry compute_causal_geometry(
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
        CausalGeometry causal{};
        uint32_t device_index = 0;
        uint32_t tp_index = 0;
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
