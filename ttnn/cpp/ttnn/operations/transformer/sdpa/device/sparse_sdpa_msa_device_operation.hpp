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
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

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
        kReaderStraddleRow,
        kReaderStraddleJump,
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
        kWriterArgCount,
    };
    enum ComputeArg : uint32_t { kComputeWorkStart, kComputeWorkCount, kComputeArgCount };

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks invariants excluded from the program hash, such as interleaved K/V length and cache_batch_idx.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    // Rejects an explicit block-cache slot request that L1 cannot honour at all; runs on misses and hits.
    static void validate_kv_cache_request(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Per-device causal geometry, in rows: the global position of this device's query row 0, plus the
    // straddle (query rows >= straddle_row sit straddle_jump positions further along; jump 0 = none).
    struct CausalGeometry {
        uint32_t chunk_start = 0;
        uint32_t straddle_row = 0;
        uint32_t straddle_jump = 0;
    };
    // Block-cyclic cache + cluster_axis (the SP-sharded chunked-prefill cache read): the rotation-exact
    // geometry of the update_padded_kv_cache writer, shared with indexer_score so the diagonal-block mask and
    // the indexer's selection agree on every query's position, including mid-slab (non-chunk-aligned) starts.
    // Otherwise the linear chunk_start_idx + rank*S along cluster_axis (rank from the coordinate; 0 on a single
    // device). All zeros when non-causal.
    static CausalGeometry compute_causal_geometry(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Work split plus every scalar compute_program_hash excludes: interleaved K/V T and batch slots, n_kv,
    // cache_batch_idx, chunk_start_idx/cluster_axis. Single-sourced so create_descriptor (miss-bake) and
    // override_runtime_arguments (hit-patch) write the same values and cannot drift.
    struct DispatchArgs {
        tt::tt_metal::CoreCoord grid;  // per-core arg order: core i == {i % grid.x, i / grid.x}
        uint32_t num_cores = 0;
        uint32_t base_work = 0;  // total_work = S * n_kv over num_cores; the first `extra` cores take one more
        uint32_t extra = 0;
        uint32_t k_batch_tile_offset = 0;
        uint32_t v_batch_tile_offset = 0;
        uint32_t k_group_tile_stride = 0;
        uint32_t v_group_tile_stride = 0;
        CausalGeometry causal;
    };
    static DispatchArgs compute_dispatch_args(
        const operation_attributes_t& attrs,
        const tensor_args_t& t,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Circular-buffer ids, shared with the kernels as compile-time args. Fixed ids (not allocation order), so a
    // conditional buffer -- the causal mask tiles, the block cache -- is skipped without renumbering the rest.
    enum Cb : uint32_t {
        cb_q_rm = 0,  // Q rows (row-major, reader -> compute tilize)
        cb_q_in,      // Q tiled [Sqt, DHt]
        cb_k_in,      // streamed path only: one K block [Skt, DHt] (reader upper half, writer lower half)
        cb_v_in,      // streamed path only: one V block [Skt, vDHt]
        cb_scale,     // reduce identity scaler (1 tile)
        cb_qk_im,     // scores [Sqt, Skt]
        cb_max_a,     // running max ping-pong [Sqt, 1]
        cb_max_b,
        cb_sum_a,  // running sum ping-pong [Sqt, 1]
        cb_sum_b,
        cb_out_a,  // running out ping-pong [Sqt, vDHt] (single-buffered for L1 accumulation)
        cb_out_b,
        cb_corr,           // exp(prev_max - cur_max) correction [Sqt, 1]
        cb_out_im,         // fixed pre-untilize copy of the final out [Sqt, vDHt]
        cb_out_rm,         // untilized row-major out (compute -> writer)
        cb_idx,            // reader-internal: one token's block-id row (uint32)
        cb_ctrl,           // reader -> compute per token: active block count + causal geometry (ctrl:: words)
        cb_col_identity,   // ones-in-col0 (writer-built): finalizes the partial row-sum via matmul_reduce
        cb_recip_scratch,  // 1-tile reciprocal scratch for normalize_row_streaming
        cb_kreq,           // reader -> writer gather request (kernels/sparse_sdpa_msa_common.hpp kreq)
        cb_kack,           // writer -> reader: its lower tile halves landed
        cb_neginf,         // causal only: persistent all -inf tile (writer-built) for fully-future key tiles
        cb_vmask,          // causal only: per-token partial-column tile (reader-built) for the boundary key tile
        cb_k_cache,        // block cache only: K blocks by slot, reader-owned, compute reads in place
        cb_v_cache,        // block cache only: V blocks by slot
        cb_slot,           // block cache only: per-block slot id reader -> compute (depth = reader run-ahead)
        cb_count
    };

    // Kernel geometry every circular-buffer size and compile-time argument derives from; computed once per call
    // and passed to base_cbs / resolve_kv_cache so the hash and the factory see the same values.
    struct Geometry {
        uint32_t H_logical = 0, H = 0, S = 0, topk = 0, n_kv = 0, d = 0, v_dim = 0;
        uint32_t DHt = 0, vDHt = 0, Skt = 0, Sqt = 0, k_tiles_per_block = 0, v_tiles_per_block = 0;
        uint32_t k_tile_bytes = 0, v_tile_bytes = 0, q_row_bytes = 0, idx_row_bytes = 0;
        tt::DataFormat k_df = tt::DataFormat::Invalid, v_df = tt::DataFormat::Invalid;
        tt::DataFormat q_rm_df = tt::DataFormat::Invalid, q_in_df = tt::DataFormat::Invalid;
        tt::DataFormat out_df = tt::DataFormat::Invalid;
        bool q_is_fp8 = false;
    };
    static Geometry geometry(const operation_attributes_t& attrs, const tensor_args_t& t);

    struct CbSpec {
        uint32_t id;
        uint32_t page_size;
        uint32_t num_pages;
        tt::DataFormat df;
    };
    // Every circular buffer except the block cache. The streamed K/V block buffers exist only when the block
    // cache does not serve K/V. The cache sizing budgets against this list.
    static std::vector<CbSpec> base_cbs(const Geometry& g, bool causal, bool block_cache_serves_kv);

    // The block cache as resolved for THIS call. slots == 0 selects the streamed kernels. auto depends on the free
    // L1 at call time, so slots is a program-hash input. slot_depth = cb_slot depth = blocks the reader may run
    // ahead of compute.
    struct KvCachePlan {
        uint32_t slots = 0;
        uint32_t slot_depth = 1;
        uint32_t block_bytes = 0;
        uint64_t free_l1 = 0;
    };
    static KvCachePlan resolve_kv_cache(const Geometry& g, const operation_attributes_t& attrs, const tensor_args_t& t);
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
    std::optional<uint32_t> kv_cache_blocks = std::nullopt);

}  // namespace ttnn::prim
