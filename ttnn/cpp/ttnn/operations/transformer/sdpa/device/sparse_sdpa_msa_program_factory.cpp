// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_program.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
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
namespace recipes = ttnn::operations::transformer::sdpa::detail;
namespace msa_cb = ::sparse_sdpa_msa::recipe_cb;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;

recipes::PrecisionPolicy msa_policy(const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    return recipes::resolve_precision_policy(recipes::select_recipe(attrs.precision, t.k.dtype()));
}

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

tt::tt_metal::ProgramDescriptor SparseSDPAMsaOperation::recipe_layout(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t, const Geometry& g, uint32_t kv_cache_slots) {
    const auto policy = msa_policy(attrs, t);
    const uint32_t q_tiles = recipes::recipe_compute_q_tiles(policy, g.Sqt, g.Skt);
    const auto grid = t.q.device()->compute_with_storage_grid_size();
    const auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto v_storage = recipes::recipe_kv_storage(t.v.dtype());
    auto program = recipes::recipe_compute_program(
        policy,
        core_grid,
        1,
        q_tiles,
        g.Skt,
        g.DHt,
        attrs.scale,
        g.vDHt,
        v_storage != policy.selection.kv_storage ? std::optional<recipes::KVStorage>(v_storage) : std::nullopt,
        g.q_in_df);
    // Compute tilizes Q itself (one entry); K / V hold one block (the reader and writer fill the block at the CBs'
    // bases) or the block cache's slots; the output CB holds a token's rows (untilized after the step that
    // normalizes them), BF16 whatever Q's type.
    const uint32_t kv_blocks = kv_cache_slots > 0 ? kv_cache_slots : 1;
    for (auto& c : program.cbs) {
        auto& format = c.format_descriptors.front();
        if (format.buffer_index == recipe_ring_cb::q) {
            c.total_size = q_tiles * g.DHt * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::k) {
            c.total_size = kv_blocks * g.k_tiles_per_block * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::v) {
            c.total_size = kv_blocks * g.v_tiles_per_block * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::out) {
            format.data_format = tt::DataFormat::Float16_b;
            format.page_size = tt::tile_size(tt::DataFormat::Float16_b);
            c.total_size = q_tiles * g.vDHt * format.page_size;
        }
    }
    const auto add_cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        program.cbs.push_back(CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page_size}}},
        });
    };
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    add_cb(msa_cb::q_rm, g.q_row_bytes, q_tiles * tt::constants::TILE_HEIGHT, g.q_rm_df);
    add_cb(msa_cb::idx, g.idx_row_bytes, 1, bf);
    add_cb(msa_cb::ctrl, message_page_bytes(sparse_sdpa_msa::ctrl::WORD_COUNT), 2, bf);
    add_cb(msa_cb::kreq, message_page_bytes(sparse_sdpa_msa::kreq::WORD_COUNT), 2, bf);
    add_cb(msa_cb::kack, message_page_bytes(1), 2, bf);
    add_cb(msa_cb::out_rm, tt::tile_size(g.out_df), g.Sqt * g.vDHt, g.out_df);
    if (kv_cache_slots > 0) {
        // The reader runs one block ahead of compute (a miss's DRAM read overlaps the previous block's math) and
        // keeps the previous block's slot off the victim list.
        add_cb(msa_cb::slot, message_page_bytes(1), sparse_sdpa_msa::KV_CACHE_SLOT_DEPTH, tt::DataFormat::UInt32);
    }
    // fp8 Q tilizes through the FP32 dest.
    auto& compute = program.kernels.front();
    auto& config = std::get<tt::tt_metal::ComputeConfigDescriptor>(compute.config);
    config.unpack_to_dest_mode.resize(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (g.q_is_fp8) {
        config.unpack_to_dest_mode[msa_cb::q_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    return program;
}

tt::tt_metal::ProgramDescriptor SparseSDPAMsaOperation::SparseSDPAMsaProgramFactory::create_descriptor(
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
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
    // Work split and the hash-excluded per-dispatch scalars (K/V slot offsets, group strides, per-coordinate
    // causal geometry) come from the helper override_runtime_arguments also uses, so the values baked here
    // and the ones patched on a cache hit cannot drift.
    const auto dyn = SparseSDPAMsaOperation::compute_dispatch_args(attrs, t, mesh_dispatch_coordinate);
    const tt::tt_metal::CoreCoord grid = dyn.grid;
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = dyn.num_cores;

    // ---- CBs: the recipe layout (recipe_layout) ----
    const uint32_t kv_cache_slots = attrs.kv_cache_slots;
    auto layout = recipe_layout(attrs, t, g, kv_cache_slots);
    desc.cbs = std::move(layout.cbs);
    auto& recipe_compute = layout.kernels.front();
    const uint32_t q_tiles = recipes::named_arg(recipe_compute.named_compile_time_args, "q_tiles");
    {
        const uint64_t cb_align = tt::tt_metal::hal::get_dram_alignment();
        uint64_t cb_bytes = 0;
        for (const auto& c : desc.cbs) {
            cb_bytes += tt::align(static_cast<uint64_t>(c.total_size), cb_align);
        }
        const uint64_t l1_available = recipes::recipe_free_l1(*t.q.device());
        TT_FATAL(
            cb_bytes <= l1_available,
            "sparse_sdpa_msa recipe needs {} B of L1 per core ({} heads, d {}, block {}), but only {} B are available",
            cb_bytes,
            H,
            g.d,
            block_size,
            l1_available);
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
    reader_args.set(rct::K_TILE_BYTES, k_tile_bytes);  // K is tiled: per-tile read size
    reader_args.set(rct::V_TILE_BYTES, v_tile_bytes);  // V is tiled: per-tile read size
    reader_args.set(rct::CAUSAL_MASK_ENABLED, attrs.causal_enabled() ? 1u : 0u);
    reader_args.set(rct::BLOCK_SIZE, block_size);  // for diag_block = p/bs, offset = p%bs
    reader_args.set(rct::BLOCK_CYCLIC, block_cyclic_ct.enabled);
    reader_args.set(rct::BC_CHUNK_LOCAL, block_cyclic_ct.chunk_local);
    reader_args.set(rct::BC_SP, block_cyclic_ct.sp);
    reader_args.set(rct::BC_SHARD_STRIDE_GAP, block_cyclic_ct.shard_stride_gap);
    reader_args.set(rct::BC_SLAB_STRIDE_GAP, block_cyclic_ct.slab_stride_gap);
    reader_args.set(rct::KV_CACHE_SLOTS, kv_cache_slots);  // 0 = streamed path
    reader_args.set(rct::D_TILES, DHt);
    reader_args.set(rct::Q_TILES, q_tiles);
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
    writer_args.set(wct::K_TILE_BYTES, k_tile_bytes);
    writer_args.set(wct::V_TILE_BYTES, v_tile_bytes);
    writer_args.set(wct::BLOCK_CYCLIC, block_cyclic_ct.enabled);
    writer_args.set(wct::BC_CHUNK_LOCAL, block_cyclic_ct.chunk_local);
    writer_args.set(wct::BC_SP, block_cyclic_ct.sp);
    writer_args.set(wct::BC_SHARD_STRIDE_GAP, block_cyclic_ct.shard_stride_gap);
    writer_args.set(wct::BC_SLAB_STRIDE_GAP, block_cyclic_ct.slab_stride_gap);
    writer_args.set(wct::KV_CACHE_SLOTS, kv_cache_slots);  // 0 = streamed path
    writer_args.set(wct::D_TILES, DHt);
    std::vector<uint32_t> writer_ct = writer_args.take();
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);

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

    // The recipe compute: its named arguments, compute config (fidelity, FP32 DEST and unpack-to-dest modes) and
    // defines, with the MSA kernel's own. MSA masks only the causal diagonal block's future keys (recipe_ring_mask 0)
    // and keeps each token's state resident.
    const auto policy = msa_policy(attrs, t);
    auto compute_named = recipes::recipe_named_args(recipe_compute.named_compile_time_args);
    recipes::set_named_arg(compute_named, "recipe_ring", 1);
    compute_named.emplace_back("q_tiles", q_tiles);
    compute_named.emplace_back("head_tiles", Sqt);
    compute_named.emplace_back("k_tiles", Skt);
    compute_named.emplace_back("d_tiles", DHt);
    compute_named.emplace_back("vd_tiles", vDHt);
    compute_named.emplace_back("scale", scale_packed);
    compute_named.emplace_back("causal", attrs.causal_enabled() ? 1u : 0u);
    compute_named.emplace_back("kv_cache_slots", kv_cache_slots);
    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_msa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.named_compile_time_args = std::move(compute_named);
    compute_desc.defines = recipe_compute.defines;
    compute_desc.config = recipe_compute.config;
    // Math builds at -O2 (sparse_sdpa_msa_compute.cpp); with fused chunks or ACCURATE, unpack and pack too (their code
    // at -O3 overflows the kernel config buffer, as on the rings).
    const bool recipe_fused = recipes::named_arg(compute_desc.named_compile_time_args, "recipe_fused") != 0;
    if (recipe_fused || policy.selection.recipe == recipes::Recipe::D) {
        compute_desc.opt_level = recipes::recipe_opt_level(tt::tt_metal::KernelBuildOptLevel::O2);
    }

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
