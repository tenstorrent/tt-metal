// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <array>
#include <bit>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ttnn::prim {

namespace {

// Compile-time args filled by index name (sparse_sdpa_msa_common.hpp) so the factory and the kernel share one
// layout; every index must be set exactly once before the accessor blocks are appended.
template <uint32_t N>
class CtArgs {
public:
    void set(uint32_t index, uint32_t value) {
        TT_FATAL(index < N && !set_[index], "compile-time arg {} set twice or out of range", index);
        args_[index] = value;
        set_[index] = true;
    }
    std::vector<uint32_t> take() const {
        for (uint32_t i = 0; i < N; ++i) {
            TT_FATAL(set_[i], "compile-time arg {} never set", i);
        }
        return {args_.begin(), args_.end()};
    }

private:
    std::array<uint32_t, N> args_{};
    std::array<bool, N> set_{};
};

// Block-cyclic ("slab") remap in BLOCK units; the defaults are the identity remap.
struct BlockCyclicCt {
    uint32_t enabled = 0;
    uint32_t chunk_local = 1;
    uint32_t sp = 1;
    uint32_t shard_stride_gap = 0;
    uint32_t slab_stride_gap = 0;
};
// emplace_runtime_args' vector overload registers each Buffer* as an address binding at its slot,
// so the args can be filled by enum index instead of positionally.
using RtArgs = std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>;
}  // namespace

tt::tt_metal::ProgramDescriptor SparseSDPAMsaOperation::SparseSDPAMsaProgramFactory::create_descriptor(
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using enum SparseSDPAMsaOperation::Cb;

    tt::tt_metal::ProgramDescriptor desc;

    // K/V are separate pre-tiled caches, gathered one block at a time by the reader and writer together.
    const Geometry g = derive_kernel_geometry(attrs, t);
    const uint32_t H_logical = g.H_logical, H = g.H, S = g.S, topk = g.topk, n_kv = g.n_kv;
    const uint32_t DHt = g.DHt, vDHt = g.vDHt, Skt = g.Skt, Sqt = g.Sqt;
    const uint32_t k_tiles_per_block = g.k_tiles_per_block, v_tiles_per_block = g.v_tiles_per_block;
    const uint32_t k_half = k_tiles_per_block >> 1;  // the writer gathers tiles [0, half), the reader the rest
    const uint32_t v_half = v_tiles_per_block >> 1;
    const uint32_t k_tile_bytes = g.k_tile_bytes, v_tile_bytes = g.v_tile_bytes;
    const uint32_t block_size = attrs.block_size;  // tokens per block == one chunk
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);
    const uint32_t q_row_bytes = g.q_row_bytes, idx_row_bytes = g.idx_row_bytes;
    const uint32_t out_elem_bytes = output.element_size();
    const bool q_is_fp8 = g.q_is_fp8;
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;

    // Work split and the hash-excluded per-dispatch scalars (K/V slot offsets, group strides, per-coordinate
    // causal geometry) come from the helper override_runtime_arguments also uses, so the values baked here
    // and the ones patched on a cache hit cannot drift.
    const auto dyn = SparseSDPAMsaOperation::compute_dispatch_args(attrs, t, mesh_dispatch_coordinate);
    const tt::tt_metal::CoreCoord grid = dyn.grid;
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = dyn.num_cores;

    // ---- CBs (fixed ids = Cb enum; descriptor insertion order is irrelevant) ----
    const auto cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page_size}}},
        });
    };
    // Per-core K/V block cache: the reader fills a slot on a miss and compute reads it in place, replacing the
    // streamed K/V block buffers. The plan is hashed, so this layout is fixed for the program's lifetime.
    const KvCachePlan& kv = attrs.kv_cache_plan;
    const uint32_t kv_cache_slots = kv.slots;
    const uint32_t kv_cache_slot_depth = kv.slot_depth;
    for (const CbSpec& s : base_cbs(g, attrs.causal_enabled(), /*block_cache_serves_kv=*/kv_cache_slots > 0)) {
        cb(s.id, s.page_size, s.num_pages, s.df);
    }
    if (kv_cache_slots > 0) {
        cb(cb_k_cache, k_tile_bytes, kv_cache_slots * k_tiles_per_block, g.k_df);
        cb(cb_v_cache, v_tile_bytes, kv_cache_slots * v_tiles_per_block, g.v_df);
        // Depth = blocks the reader may run ahead of compute (a miss's DRAM read overlaps the previous block's
        // math); at depth 2 the reader keeps the previous block's slot off the victim list.
        cb(cb_slot, message_page_bytes(1), kv_cache_slot_depth, bf);
    }

    // Block-cyclic ("slab") cache: the invP remap is baked as compile-time args, so a natural-order cache folds
    // to identity. Units are BLOCKS here (sparse_sdpa works in rows, indexer_score in tiles). Reader and writer
    // take the same block {enable, chunk_local, sp, shard_stride_gap, slab_stride_gap}.
    const auto block_cyclic_ct = [&attrs, &t, block_size]() {
        BlockCyclicCt args;
        if (!attrs.has_block_cyclic()) {
            return args;
        }
        const auto& bc = attrs.block_cyclic.value();
        const uint32_t chunk_local_blk = bc.chunk_local / block_size;
        const uint32_t shard_len_blk = (t.k.logical_shape()[2] / bc.sp) / block_size;
        args.enabled = 1;
        args.chunk_local = chunk_local_blk;
        args.sp = bc.sp;
        args.shard_stride_gap = shard_len_blk - chunk_local_blk;
        args.slab_stride_gap = chunk_local_blk * (bc.sp - 1);
        return args;
    }();

    // ---- compile-time args ----
    // Reader args: scalars, derived geometry, CB ids, element sizes, then q/k/v/indices accessors.
    // K/V use RuntimeTensorShape.
    namespace rct = sparse_sdpa_msa::reader_ct;
    CtArgs<rct::COUNT> reader_args;
    reader_args.set(rct::H_LOGICAL, H_logical);
    reader_args.set(rct::H, H);
    reader_args.set(rct::S, S);
    reader_args.set(rct::TOPK, topk);
    reader_args.set(rct::N_KV, n_kv);
    reader_args.set(rct::Q_ROW_BYTES, q_row_bytes);
    reader_args.set(rct::IDX_ROW_BYTES, idx_row_bytes);
    reader_args.set(rct::K_TILES_PER_BLOCK, k_tiles_per_block);
    reader_args.set(rct::V_TILES_PER_BLOCK, v_tiles_per_block);
    reader_args.set(rct::K_HALF, k_half);
    reader_args.set(rct::V_HALF, v_half);
    reader_args.set(rct::CB_Q_RM, cb_q_rm);
    reader_args.set(rct::CB_K_IN, cb_k_in);
    reader_args.set(rct::CB_V_IN, cb_v_in);
    reader_args.set(rct::CB_IDX, cb_idx);
    reader_args.set(rct::CB_CTRL, cb_ctrl);
    reader_args.set(rct::CB_KREQ, cb_kreq);
    reader_args.set(rct::CB_KACK, cb_kack);
    reader_args.set(rct::K_TILE_BYTES, k_tile_bytes);  // K is tiled: per-tile read size
    reader_args.set(rct::V_TILE_BYTES, v_tile_bytes);  // V is tiled: per-tile read size
    reader_args.set(rct::CAUSAL_MASK_ENABLED, attrs.causal_enabled() ? 1u : 0u);
    reader_args.set(rct::BLOCK_SIZE, block_size);  // for diag_block = p/bs, offset = p%bs
    reader_args.set(rct::CB_VMASK, cb_vmask);      // reader builds the per-token partial-column tile
    reader_args.set(rct::BLOCK_CYCLIC, block_cyclic_ct.enabled);
    reader_args.set(rct::BC_CHUNK_LOCAL, block_cyclic_ct.chunk_local);
    reader_args.set(rct::BC_SP, block_cyclic_ct.sp);
    reader_args.set(rct::BC_SHARD_STRIDE_GAP, block_cyclic_ct.shard_stride_gap);
    reader_args.set(rct::BC_SLAB_STRIDE_GAP, block_cyclic_ct.slab_stride_gap);
    reader_args.set(rct::KV_CACHE_SLOTS, kv_cache_slots);  // 0 = streamed path
    reader_args.set(rct::CB_K_CACHE, cb_k_cache);
    reader_args.set(rct::CB_V_CACHE, cb_v_cache);
    reader_args.set(rct::CB_SLOT, cb_slot);
    reader_args.set(rct::KV_CACHE_SLOT_DEPTH, kv_cache_slot_depth);
    std::vector<uint32_t> reader_ct = reader_args.take();
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);

    // Writer builds persistent compute tiles, co-gathers K/V halves, and drains row-major output.
    const uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;
    const uint32_t block_tiles = Sqt * vDHt;
    namespace wct = sparse_sdpa_msa::writer_ct;
    CtArgs<wct::COUNT> writer_args;
    writer_args.set(wct::H_LOGICAL, H_logical);
    writer_args.set(wct::S, S);
    writer_args.set(wct::N_KV, n_kv);
    writer_args.set(wct::ROW_BYTES, row_bytes);
    writer_args.set(wct::BLOCK_TILES, block_tiles);
    writer_args.set(wct::K_TILES_PER_BLOCK, k_tiles_per_block);
    writer_args.set(wct::V_TILES_PER_BLOCK, v_tiles_per_block);
    writer_args.set(wct::K_HALF, k_half);
    writer_args.set(wct::V_HALF, v_half);
    writer_args.set(wct::CB_OUT_RM, cb_out_rm);
    writer_args.set(wct::CB_SCALE, cb_scale);
    writer_args.set(wct::CB_COL_IDENTITY, cb_col_identity);
    writer_args.set(wct::CB_K_IN, cb_k_in);
    writer_args.set(wct::CB_V_IN, cb_v_in);
    writer_args.set(wct::CB_KREQ, cb_kreq);
    writer_args.set(wct::CB_KACK, cb_kack);
    writer_args.set(wct::K_TILE_BYTES, k_tile_bytes);
    writer_args.set(wct::V_TILE_BYTES, v_tile_bytes);
    writer_args.set(wct::CAUSAL_MASK_ENABLED, attrs.causal_enabled() ? 1u : 0u);
    writer_args.set(wct::CB_NEGINF, cb_neginf);  // writer builds the persistent -inf mask tile
    writer_args.set(wct::BLOCK_CYCLIC, block_cyclic_ct.enabled);
    writer_args.set(wct::BC_CHUNK_LOCAL, block_cyclic_ct.chunk_local);
    writer_args.set(wct::BC_SP, block_cyclic_ct.sp);
    writer_args.set(wct::BC_SHARD_STRIDE_GAP, block_cyclic_ct.shard_stride_gap);
    writer_args.set(wct::BC_SLAB_STRIDE_GAP, block_cyclic_ct.slab_stride_gap);
    writer_args.set(wct::KV_CACHE_SLOTS, kv_cache_slots);  // 0 = streamed path
    writer_args.set(wct::CB_K_CACHE, cb_k_cache);
    writer_args.set(wct::CB_V_CACHE, cb_v_cache);
    std::vector<uint32_t> writer_ct = writer_args.take();
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);

    namespace cct = sparse_sdpa_msa::compute_ct;
    CtArgs<cct::COUNT> compute_args;
    compute_args.set(cct::H, H);
    compute_args.set(cct::DHT, DHt);
    compute_args.set(cct::VDHT, vDHt);
    compute_args.set(cct::SKT, Skt);
    compute_args.set(cct::SCALE_FP32, scale_packed);
    compute_args.set(cct::CB_Q_RM, cb_q_rm);
    compute_args.set(cct::CB_Q_IN, cb_q_in);
    compute_args.set(cct::CB_K_IN, cb_k_in);
    compute_args.set(cct::CB_V_IN, cb_v_in);
    compute_args.set(cct::CB_SCALE, cb_scale);
    compute_args.set(cct::CB_QK_IM, cb_qk_im);
    compute_args.set(cct::CB_MAX_A, cb_max_a);
    compute_args.set(cct::CB_MAX_B, cb_max_b);
    compute_args.set(cct::CB_SUM_A, cb_sum_a);
    compute_args.set(cct::CB_SUM_B, cb_sum_b);
    compute_args.set(cct::CB_OUT_A, cb_out_a);
    compute_args.set(cct::CB_OUT_B, cb_out_b);
    compute_args.set(cct::CB_CORR, cb_corr);
    compute_args.set(cct::CB_OUT_IM, cb_out_im);
    compute_args.set(cct::CB_OUT_RM, cb_out_rm);
    compute_args.set(cct::CB_CTRL, cb_ctrl);
    compute_args.set(cct::CB_COL_IDENTITY, cb_col_identity);
    compute_args.set(cct::CB_RECIP_SCRATCH, cb_recip_scratch);

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/sparse_sdpa_msa_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/sparse_sdpa_msa_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;

    // Query sub-blocking: qsb tile rows must fit in DEST.
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    uint32_t qsb = 1;
    for (uint32_t dd = std::min(Sqt, dst_size); dd >= 1; --dd) {
        if (Sqt % dd == 0) {
            qsb = dd;
            break;
        }
    }
    compute_args.set(cct::QSB, qsb);
    compute_args.set(cct::CAUSAL_MASK_ENABLED, attrs.causal_enabled() ? 1u : 0u);
    compute_args.set(cct::CB_NEGINF, cb_neginf);            // full -inf mask tile (future key-tiles)
    compute_args.set(cct::CB_VMASK, cb_vmask);              // partial-column mask tile (boundary key-tile)
    compute_args.set(cct::KV_CACHE_SLOTS, kv_cache_slots);  // 0 = compute streams from cb_k_in/cb_v_in
    compute_args.set(cct::CB_K_CACHE, cb_k_cache);
    compute_args.set(cct::CB_V_CACHE, cb_v_cache);
    compute_args.set(cct::CB_SLOT, cb_slot);
    const std::vector<uint32_t> compute_ct = compute_args.take();

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_msa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    // fp8 Q must unpack into 32-bit DEST before tilize packs to bfp8.
    if (q_is_fp8) {
        unpack_to_dest_mode[cb_q_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx};
    std::map<std::string, std::string> cdefs{
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    compute_desc.defines = tt::tt_metal::KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* k_buf = t.k.buffer();
    auto* v_buf = t.v.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t work_start = i * dyn.base_work + std::min(i, dyn.extra);
        uint32_t work_count = dyn.base_work + (i < dyn.extra ? 1u : 0u);
        // Both sides index the same slot enums, so a reorder here cannot silently desync the
        // cache-hit patch in override_runtime_arguments; Buffer* slots stay address bindings.
        using RArg = SparseSDPAMsaOperation::ReaderArg;
        RtArgs reader_rt(RArg::kReaderArgCount);
        reader_rt[RArg::kReaderQAddr] = q_buf;
        reader_rt[RArg::kReaderKAddr] = k_buf;
        reader_rt[RArg::kReaderVAddr] = v_buf;
        reader_rt[RArg::kReaderIdxAddr] = idx_buf;
        reader_rt[RArg::kReaderWorkStart] = work_start;
        reader_rt[RArg::kReaderWorkCount] = work_count;
        reader_rt[RArg::kReaderKBatchOffset] = dyn.k_batch_tile_offset;
        reader_rt[RArg::kReaderVBatchOffset] = dyn.v_batch_tile_offset;
        reader_rt[RArg::kReaderKGroupStride] = dyn.k_group_tile_stride;
        reader_rt[RArg::kReaderVGroupStride] = dyn.v_group_tile_stride;
        // Baked per-coordinate (one program per device, so each rank masks against its own global
        // position) and re-applied on cache hits.
        reader_rt[RArg::kReaderChunkStart] = dyn.causal.chunk_start;
        reader_rt[RArg::kReaderStraddleRow] = dyn.causal.straddle_row;
        reader_rt[RArg::kReaderStraddleJump] = dyn.causal.straddle_jump;
        reader_desc.emplace_runtime_args(core, reader_rt);

        using WArg = SparseSDPAMsaOperation::WriterArg;
        RtArgs writer_rt(WArg::kWriterArgCount);
        writer_rt[WArg::kWriterOutAddr] = out_buf;
        writer_rt[WArg::kWriterWorkStart] = work_start;
        writer_rt[WArg::kWriterWorkCount] = work_count;
        writer_rt[WArg::kWriterKAddr] = k_buf;
        writer_rt[WArg::kWriterVAddr] = v_buf;
        writer_rt[WArg::kWriterKBatchOffset] = dyn.k_batch_tile_offset;
        writer_rt[WArg::kWriterVBatchOffset] = dyn.v_batch_tile_offset;
        writer_rt[WArg::kWriterKGroupStride] = dyn.k_group_tile_stride;
        writer_rt[WArg::kWriterVGroupStride] = dyn.v_group_tile_stride;
        writer_desc.emplace_runtime_args(core, writer_rt);

        using CArg = SparseSDPAMsaOperation::ComputeArg;
        RtArgs compute_rt(CArg::kComputeArgCount);
        compute_rt[CArg::kComputeWorkStart] = work_start;
        compute_rt[CArg::kComputeWorkCount] = work_count;
        compute_desc.emplace_runtime_args(core, compute_rt);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
