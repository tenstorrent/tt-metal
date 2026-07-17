// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <map>
#include <string>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SparseSDPAMsaOperation::SparseSDPAMsaProgramFactory::create_descriptor(
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Fixed CB ids shared with the kernels. Function scope avoids unity-build collisions with sparse_sdpa.
    // K/V are separate pre-tiled caches. Reader and writer co-gather each block into shared K/V CBs.
    enum SparseCB : uint32_t {
        cb_q_rm = 0,  // Q rows (row-major, reader -> compute tilize)
        cb_q_in,      // Q tiled [Sqt, DHt]
        cb_k_in,      // K tiled [Skt, DHt] (reader-filled from the tiled cache; QK within-tile transpose)
        cb_v_in,      // V tiled [Skt, vDHt] (reader-filled; separate tensor)
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
        cb_ctrl,           // reader -> compute: active block count per token
        cb_col_identity,   // ones-in-col0 (writer-built): finalizes the partial row-sum via matmul_reduce
        cb_recip_scratch,  // 1-tile reciprocal scratch for normalize_row_streaming
        cb_kreq,           // reader->writer dual-NoC handoff {block_id, is_last} (writer co-gathers the lower half)
        cb_kack,           // writer->reader ack that its half of the block landed in cb_k_in/cb_v_in
        cb_neginf,         // causal mask: persistent all -inf tile (writer-built); masks full future key-tiles
        cb_vmask,          // causal mask: per-token partial-column "vertical" tile (reader-built) for the boundary
        cb_count
    };

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H_total = t.q.logical_shape()[1];  // total query heads
    const uint32_t n_kv = t.k.logical_shape()[1];     // KV groups
    const uint32_t H_logical = H_total / n_kv;        // query heads per KV group
    const uint32_t H =
        ((H_logical + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t topk = t.indices.logical_shape()[3];  // max selected blocks per (group, query)
    const uint32_t d = t.q.logical_shape()[3];           // head dim (e.g. 128)
    const uint32_t v_dim = t.v.logical_shape()[3];       // V width (output width)
    const uint32_t block_size = attrs.block_size;        // tokens per block == k_chunk (one block per chunk)

    const uint32_t DHt = d / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_dim / tt::constants::TILE_WIDTH;
    const uint32_t k_chunk = block_size;                       // a chunk is exactly one block
    const uint32_t Skt = k_chunk / tt::constants::TILE_WIDTH;  // tiles per chunk along keys (block_size/32)
    const uint32_t Sqt = H / tt::constants::TILE_HEIGHT;       // query tile-rows (32 heads each)
    const uint32_t k_tiles_per_block = Skt * DHt;
    const uint32_t v_tiles_per_block = Skt * vDHt;
    const uint32_t k_half = k_tiles_per_block >> 1;
    const uint32_t v_half = v_tiles_per_block >> 1;
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);

    // Q is row-major; K/V are tiled and addressed per tile.
    const uint32_t q_elem_bytes = t.q.element_size();          // 2 (bf16)
    const uint32_t idx_elem_bytes = t.indices.element_size();  // 4
    const uint32_t out_elem_bytes = output.element_size();     // 2
    const uint32_t q_row_bytes = d * q_elem_bytes;
    const uint32_t idx_row_bytes = topk * idx_elem_bytes;
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);  // 2048 (intermediate/Q/out tiles are bf16)
    // K/V cache formats drive CB format and tile size.
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    const uint32_t k_tile_bytes = tt::tile_size(k_df);
    const uint32_t v_tile_bytes = tt::tile_size(v_df);
    // Q is read row-major and tiled on chip. fp8 Q uses bfp8_b for cb_q_in and requires fp32 DEST.
    const tt::DataFormat q_rm_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const bool q_is_fp8 = (t.q.dtype() == DataType::FP8_E4M3);
    const tt::DataFormat q_in_df = q_is_fp8 ? tt::DataFormat::Bfp8_b : q_rm_df;
    const uint32_t q_in_tile_bytes = tt::tile_size(q_in_df);
    // Output dtype matches Q.
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    auto* device = t.q.device();
    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t total_work = S * n_kv;
    const uint32_t base_work = total_work / num_cores;
    const uint32_t extra = total_work % num_cores;

    // ---- CBs (fixed order = SparseCB enum) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(q_row_bytes, H, q_rm_df);              // cb_q_rm : H row-sticks (native Q dtype: bf16/fp8)
    cb(q_in_tile_bytes, Sqt * DHt, q_in_df);  // cb_q_in : [Sqt,DHt] (bfp8 when Q is fp8, else Q's float format)
    // Single-buffered: reader reserves one block and writer fills its half into the same L1 region.
    cb(k_tile_bytes, Skt * DHt, k_df);       // cb_k_in : [Skt,DHt] tiled cache (reader-filled, no tilize)
    cb(v_tile_bytes, Skt * vDHt, v_df);      // cb_v_in : [Skt,vDHt] tiled cache (reader-filled, no tilize)
    cb(tile_bytes, 1, bf);                   // cb_scale
    cb(tile_bytes, Sqt * Skt, bf);           // cb_qk_im : [Sqt,Skt]
    cb(tile_bytes, Sqt, bf);                 // cb_max_a
    cb(tile_bytes, Sqt, bf);                 // cb_max_b
    cb(tile_bytes, Sqt, bf);                 // cb_sum_a
    cb(tile_bytes, Sqt, bf);                 // cb_sum_b
    cb(tile_bytes, Sqt * vDHt, bf);          // cb_out_a
    cb(tile_bytes, Sqt * vDHt, bf);          // cb_out_b
    cb(tile_bytes, Sqt, bf);                 // cb_corr
    cb(tile_bytes, Sqt * vDHt, bf);          // cb_out_im (bf16 accumulator, full precision)
    cb(out_tile_bytes, Sqt * vDHt, out_df);  // cb_out_rm : untilized output in Q's dtype
    cb(topk * idx_elem_bytes, 1, bf);        // cb_idx : one block-id row
    cb(16, 2, bf);                           // cb_ctrl : active block count (double-buffered)
    cb(tile_bytes, 1, bf);                   // cb_col_identity
    cb(tile_bytes, 1, bf);                   // cb_recip_scratch
    cb(16, 2, bf);                           // cb_kreq : {block_id, is_last} reader->writer (double-buffered)
    cb(16, 2, bf);                           // cb_kack : writer->reader ack (double-buffered)
    // Mask tiles are touched only under CAUSAL_MASK_ENABLED in the kernels, so skip their L1 when causal
    // masking is off. Safe to gate: these are the trailing CBs, so omitting them shifts no other buffer index.
    if (attrs.causal_enabled()) {
        cb(tile_bytes, 1, bf);  // cb_neginf : persistent all -inf mask tile
        cb(tile_bytes, 2, bf);  // cb_vmask : per-token partial-column mask tile
    }

    // ---- compile-time args ----
    // Reader args: scalars, derived geometry, CB ids, element sizes, then q/k/v/indices accessors.
    // K/V use RuntimeTensorShape.
    std::vector<uint32_t> reader_ct = {
        H_logical, H, S, topk, n_kv, q_row_bytes, idx_row_bytes, k_tiles_per_block, v_tiles_per_block, k_half, v_half};
    for (uint32_t id : {cb_q_rm, cb_k_in, cb_v_in, cb_idx, cb_ctrl, cb_kreq, cb_kack}) {
        reader_ct.push_back(id);
    }
    reader_ct.push_back(k_tile_bytes);                      // K is tiled: per-tile read size
    reader_ct.push_back(v_tile_bytes);                      // V is tiled: per-tile read size
    reader_ct.push_back(attrs.causal_enabled() ? 1u : 0u);  // CAUSAL_MASK_ENABLED
    reader_ct.push_back(block_size);                        // block_size: for diag_block = p/bs, offset = p%bs
    reader_ct.push_back(cb_vmask);                          // reader builds the per-token partial-column tile
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
    std::vector<uint32_t> writer_ct = {
        H_logical,
        S,
        n_kv,
        row_bytes,
        block_tiles,
        k_tiles_per_block,
        v_tiles_per_block,
        k_half,
        v_half,
        cb_out_rm,
        cb_scale,
        cb_col_identity};
    for (uint32_t id : {cb_k_in, cb_v_in, cb_kreq, cb_kack}) {
        writer_ct.push_back(id);
    }
    writer_ct.push_back(k_tile_bytes);
    writer_ct.push_back(v_tile_bytes);
    writer_ct.push_back(attrs.causal_enabled() ? 1u : 0u);  // CAUSAL_MASK_ENABLED
    writer_ct.push_back(cb_neginf);                         // writer builds the persistent -inf mask tile
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);

    std::vector<uint32_t> compute_ct = {
        H,        DHt,      vDHt,      Skt,       scale_packed, cb_q_rm,         cb_q_in,         cb_k_in,
        cb_v_in,  cb_scale, cb_qk_im,  cb_max_a,  cb_max_b,     cb_sum_a,        cb_sum_b,        cb_out_a,
        cb_out_b, cb_corr,  cb_out_im, cb_out_rm, cb_ctrl,      cb_col_identity, cb_recip_scratch};

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
    compute_ct.push_back(qsb);
    compute_ct.push_back(attrs.causal_enabled() ? 1u : 0u);  // CAUSAL_MASK_ENABLED
    compute_ct.push_back(cb_neginf);                         // full -inf mask tile (future key-tiles)
    compute_ct.push_back(cb_vmask);                          // partial-column mask tile (boundary key-tile)

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
    const uint32_t cache_batch_idx = attrs.cache_batch_idx.value_or(0);
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t tiles_per_row = T / tt::constants::TILE_HEIGHT;
    const uint32_t k_group_tile_stride = tiles_per_row * DHt;
    const uint32_t v_group_tile_stride = tiles_per_row * vDHt;
    const uint32_t k_batch_tile_offset = cache_batch_idx * n_kv * k_group_tile_stride;
    const uint32_t v_batch_tile_offset = cache_batch_idx * n_kv * v_group_tile_stride;
    // Baked per coordinate (one program per device) so each rank masks against its own global position.
    const uint32_t chunk_start_local =
        SparseSDPAMsaOperation::compute_chunk_start_local(attrs, t, mesh_dispatch_coordinate);
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t work_start = i * base_work + std::min(i, extra);
        uint32_t work_count = base_work + (i < extra ? 1u : 0u);
        // Cache slot offsets are patched on hits because cache_batch_idx is not hashed.
        reader_desc.emplace_runtime_args(
            core,
            {q_buf,
             k_buf,
             v_buf,
             idx_buf,
             work_start,
             work_count,
             k_batch_tile_offset,
             v_batch_tile_offset,
             k_group_tile_stride,
             v_group_tile_stride,
             chunk_start_local});  // arg 10: baked per-coordinate; re-applied on cache hits
                                   // (override_runtime_arguments)
        // Writer args 5/6 are the K/V cache-slot offsets patched on cache hits.
        writer_desc.emplace_runtime_args(
            core,
            {out_buf,
             work_start,
             work_count,
             k_buf,
             v_buf,
             k_batch_tile_offset,
             v_batch_tile_offset,
             k_group_tile_stride,
             v_group_tile_stride});
        compute_desc.emplace_runtime_args(core, {work_start, work_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
