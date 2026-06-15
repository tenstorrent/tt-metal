// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <map>
#include <string>

namespace ttnn::prim {

// Fixed CB id convention shared with the kernels.
// Flash/online-softmax keeps ping-pong (prev/cur) buffers for the running max, sum, and output.
enum SparseCB : uint32_t {
    cb_q_rm = 0,   // Q rows (row-major, reader -> compute tilize)
    cb_q_in,       // Q tiled [Sqt, DHt]
    cb_k_rm,       // K chunk rows (row-major)
    cb_k_in,       // K tiled [Skt, DHt] (read directly by QK via within-tile transpose and by PV)
    cb_neginf,     // all-(-inf) tile (reader-built once): bcast-added to fully-masked key tiles
    cb_mask_part,  // per-token partial-boundary mask tile (row 0: -inf for cols >= valid%32)
    cb_scale,      // reduce identity scaler (1 tile)
    cb_qk_im,      // scores [Sqt, Skt]
    cb_max_a,      // running max ping-pong [Sqt, 1]
    cb_max_b,
    cb_sum_a,  // running sum ping-pong [Sqt, 1]
    cb_sum_b,
    cb_out_a,  // running out ping-pong [Sqt, vDHt] (single-buffered for L1 accumulation)
    cb_out_b,
    cb_corr,           // exp(prev_max - cur_max) correction [Sqt, 1]
    cb_out_im,         // fixed pre-untilize copy of the final out [Sqt, vDHt]
    cb_out_rm,         // untilized row-major out (compute -> writer)
    cb_idx,            // reader-internal: one token's index row (uint32)
    cb_ctrl,           // reader -> compute: active chunk count (= ceil(valid_keys / k_chunk)) per token
    cb_col_identity,   // ones-in-col0 (reader-built): finalizes the partial row-sum via matmul_reduce
    cb_recip_scratch,  // 1-tile reciprocal scratch for normalize_row_streaming
    cb_count
};

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, Tensor& output) {
    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];  // 32
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t topk = t.indices.logical_shape()[3];
    const uint32_t k_dim = t.q.logical_shape()[3];  // head dim, from the tensor (e.g. 576)
    const uint32_t v_dim = attrs.v_dim;             // V width (op arg; e.g. 512)

    const uint32_t DHt = k_dim / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_dim / tt::constants::TILE_WIDTH;
    const uint32_t k_chunk = attrs.k_chunk_size;
    const uint32_t Skt = k_chunk / tt::constants::TILE_WIDTH;  // tiles per chunk along keys
    const uint32_t n_chunks = topk / k_chunk;
    const uint32_t Sqt = H / tt::constants::TILE_HEIGHT;  // query tile-rows (32 heads each)
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);

    const uint32_t bf16 = 2;
    const uint32_t q_row_bytes = k_dim * bf16;  // 1152
    const tt::DataFormat bf = tt::DataFormat::Float16_b;
    const uint32_t tile_bytes = tt::tile_size(bf);  // 2048

    auto* device = t.q.device();
    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;

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
    cb(q_row_bytes, H, bf);          // cb_q_rm : H row-sticks
    cb(tile_bytes, Sqt * DHt, bf);   // cb_q_in : [Sqt,DHt]
    cb(q_row_bytes, k_chunk, bf);    // cb_k_rm : k_chunk row-sticks
    cb(tile_bytes, Skt * DHt, bf);   // cb_k_in : [Skt,DHt] (consumed by both QK and PV; no Kᵀ/V copies)
    cb(tile_bytes, 1, bf);           // cb_neginf : all-(-inf) tile (persistent)
    cb(tile_bytes, 1, bf);           // cb_mask_part : per-token partial-boundary mask tile
    cb(tile_bytes, 1, bf);           // cb_scale
    cb(tile_bytes, Sqt * Skt, bf);   // cb_qk_im : [Sqt,Skt]
    cb(tile_bytes, Sqt, bf);         // cb_max_a
    cb(tile_bytes, Sqt, bf);         // cb_max_b
    cb(tile_bytes, Sqt, bf);         // cb_sum_a
    cb(tile_bytes, Sqt, bf);         // cb_sum_b
    cb(tile_bytes, Sqt * vDHt, bf);  // cb_out_a : [Sqt,vDHt] (single-buffered for L1 acc)
    cb(tile_bytes, Sqt * vDHt, bf);  // cb_out_b
    cb(tile_bytes, Sqt, bf);         // cb_corr : [Sqt,1]
    cb(tile_bytes, Sqt * vDHt, bf);  // cb_out_im : [Sqt,vDHt] fixed pre-untilize copy
    cb(tile_bytes, Sqt * vDHt, bf);  // cb_out_rm : untilized [H,V_DIM] (Sqt*vDHt tile-sized pages)
    cb(topk * 4, 1, bf);             // cb_idx : one index row (uint32 bytes)
    cb(16, 2, bf);                   // cb_ctrl : n_active per token (uint32; 16B aligned, double-buffered)
    cb(tile_bytes, 1, bf);           // cb_col_identity : ones-in-col0 (reader-built)
    cb(tile_bytes, 1, bf);           // cb_recip_scratch : 1-tile reciprocal scratch

    // ---- compile-time args ----
    // Depth-8 NoC trid-ring: bounds outstanding K reads/core to fight DRAM congestion. Depth 8 is gentle
    // enough to be ~neutral on sparse tokens while recovering ~9% on dense (swept empirically). 0 = disabled.
    const uint32_t k_trids = 8;
    // Ring only for tokens with n_active >= ring_min_active (>= half the chunks). Sparse tokens desync
    // naturally (no congestion to recover) so they skip the ring's overhead; only dense-ish tokens keep
    // all cores synced and saturate DRAM. n_chunks/2 was the sweet spot across the sparsity sweep.
    const uint32_t ring_min_active = n_chunks / 2 < 1 ? 1 : n_chunks / 2;

    // CB ids are passed to each kernel as compile-time args (the SparseCB enum is the single source).
    // Keep each block's order in sync with the kernel's CB-id reads. Reader CBs precede the
    // TensorAccessorArgs (which must chain last).
    std::vector<uint32_t> reader_ct = {H, S, topk, k_chunk, k_trids, ring_min_active, k_dim};
    for (uint32_t id : {cb_q_rm, cb_k_rm, cb_mask_part, cb_idx, cb_ctrl}) {
        reader_ct.push_back(id);
    }
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct);

    // The writer is the lighter dataflow kernel, so it builds the three persistent compute-input tiles.
    std::vector<uint32_t> writer_ct = {H, S, vDHt, cb_out_rm, cb_scale, cb_col_identity, cb_neginf};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = {H,
                                        DHt,
                                        vDHt,
                                        Skt,
                                        scale_packed,
                                        cb_q_rm,
                                        cb_q_in,
                                        cb_k_rm,
                                        cb_k_in,
                                        cb_neginf,
                                        cb_mask_part,
                                        cb_scale,
                                        cb_qk_im,
                                        cb_max_a,
                                        cb_max_b,
                                        cb_sum_a,
                                        cb_sum_b,
                                        cb_out_a,
                                        cb_out_b,
                                        cb_corr,
                                        cb_out_im,
                                        cb_out_rm,
                                        cb_ctrl,
                                        cb_col_identity,
                                        cb_recip_scratch};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/sparse_sdpa_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/sparse_sdpa_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, dfs, pl1] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config =
        tt::tt_metal::ComputeConfigDescriptor{.math_fidelity = math_fidelity, .fp32_dest_acc_en = false};
    // EXP_APPROX_MODE selects the approximate exp LLK in the compute kernel.
    std::map<std::string, std::string> cdefs{
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    compute_desc.defines = tt::tt_metal::KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* kv_buf = t.kv.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t tok_start = i * base + std::min(i, extra);
        uint32_t tok_count = base + (i < extra ? 1u : 0u);
        reader_desc.emplace_runtime_args(core, {q_buf, kv_buf, idx_buf, tok_start, tok_count});
        writer_desc.emplace_runtime_args(core, {out_buf, tok_start, tok_count});
        compute_desc.emplace_runtime_args(core, {tok_start, tok_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
