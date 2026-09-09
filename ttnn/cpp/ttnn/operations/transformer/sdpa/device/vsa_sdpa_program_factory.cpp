// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ttnn::prim {

namespace {
// emplace_runtime_args' vector overload registers each Buffer* as an address binding at its slot,
// so the args can be filled by enum index instead of positionally.
using RtArgs = std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>;

constexpr uint32_t round_up_16(uint32_t bytes) { return (bytes + 15u) & ~15u; }
}  // namespace

tt::tt_metal::ProgramDescriptor VsaSdpaOperation::VsaSdpaProgramFactory::create_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output) {
    // Fixed CB ids shared with the kernels. Function scope avoids unity-build collisions with the sibling
    // sparse SDPA ops. K/V are gathered per chunk of up to m blocks; reader and writer co-gather tile halves.
    enum VsaCB : uint32_t {
        cb_q_in = 0,       // Q tiled [Sqt, DHt] (reader-filled directly from the TILE tensor; no tilize)
        cb_k_in,           // K tiled [m*Skt, DHt] chunk (reader/writer co-gathered; single-buffered, fixed batch)
        cb_v_in,           // V tiled [m*Skt, vDHt] chunk
        cb_scale,          // reduce identity scaler (1 tile, writer-built)
        cb_qk_im,          // scores [Sqt, m*Skt]; a partial last chunk leaves its tail columns unread
        cb_max_a,          // running max ping-pong [Sqt, 1]
        cb_max_b,
        cb_sum_a,  // running sum ping-pong [Sqt, 1]
        cb_sum_b,
        cb_out_a,  // running out ping-pong [Sqt, vDHt] (single-buffered for L1 accumulation)
        cb_out_b,
        cb_corr,           // exp(prev_max - cur_max) correction [Sqt, 1]
        cb_out_im,         // normalized final out tiles [Sqt, vDHt] (compute -> writer; output stays TILE)
        cb_idx,            // reader-internal: one row's block-id list (uint32, W entries)
        cb_counts,         // reader-internal: the block_counts row (uint32, W entries; read once, persistent)
        cb_ctrl,           // reader -> compute per chunk: {n_valid, is_last, counts[m]}
        cb_col_identity,   // ones-in-col0 (writer-built): finalizes the partial row-sum via matmul_reduce
        cb_recip_scratch,  // 1-tile reciprocal scratch for normalize_row_streaming
        cb_kreq,           // reader -> writer per chunk: {n_valid, is_last, block_ids[m]} (writer co-gathers)
        cb_kack,           // writer -> reader ack that its halves of the chunk landed in cb_k_in/cb_v_in
        cb_neginf,         // persistent all -inf tile (writer-built); stamps fully-padded key tiles
        cb_vmask,          // per-ragged-block partial-column mask tiles (reader-built)
        cb_count
    };

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t W = t.indices.logical_shape()[3];  // index row width (>= T/block_size, sentinel-padded)
    const uint32_t block_size = attrs.block_size;
    const uint32_t m = attrs.k_chunk_blocks;

    constexpr uint32_t q_chunk_tokens = 64;  // one query tile = one index row (fixed by the VSA contract)
    const uint32_t DHt = d / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;  // v_dim == d
    const uint32_t Sqt = q_chunk_tokens / tt::constants::TILE_HEIGHT;    // query tile-rows per work item (2)
    const uint32_t Skt = block_size / tt::constants::TILE_WIDTH;         // key tile-columns per block (2)
    const uint32_t KT_stride = m * Skt;                                  // physical row width of cb_qk_im
    const uint32_t n_q_tiles = S / q_chunk_tokens;
    const uint32_t n_kv_blocks = T / block_size;
    const uint32_t k_tiles_per_block = Skt * DHt;
    const uint32_t v_tiles_per_block = Skt * vDHt;
    const uint32_t k_half = k_tiles_per_block >> 1;  // writer gathers [0, half), reader [half, per_block)
    const uint32_t v_half = v_tiles_per_block >> 1;
    const uint32_t q_tiles_per_work = Sqt * DHt;
    const uint32_t out_tiles_per_work = Sqt * vDHt;
    const uint32_t k_head_stride = (T / tt::constants::TILE_HEIGHT) * DHt;
    const uint32_t v_head_stride = (T / tt::constants::TILE_HEIGHT) * vDHt;
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);

    const uint32_t idx_elem_bytes = t.indices.element_size();  // 4
    const uint32_t idx_row_bytes = W * idx_elem_bytes;
    const uint32_t counts_row_bytes = W * t.block_counts.element_size();
    const uint32_t ctrl_page_bytes = round_up_16((2 + m) * 4);  // {n_valid, is_last, counts[m]}
    const uint32_t kreq_page_bytes = round_up_16((2 + m) * 4);  // {n_valid, is_last, block_ids[m]}
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);  // 2048 (intermediates are bf16)
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t q_tile_bytes = tt::tile_size(q_df);
    const uint32_t k_tile_bytes = tt::tile_size(k_df);
    const uint32_t v_tile_bytes = tt::tile_size(v_df);
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    const auto dyn = VsaSdpaOperation::compute_dispatch_args(attrs, t);
    const tt::tt_metal::CoreCoord grid = dyn.grid;
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = dyn.num_cores;

    // ---- CBs (fixed order = VsaCB enum) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(q_tile_bytes, Sqt * DHt, q_df);  // cb_q_in
    // Single-buffered, one fixed-size chunk each: reader reserves the whole chunk and the writer fills its
    // halves into the same L1 region, so per-chunk offsets are stable at the buffer base. A partial last chunk
    // still reserves/pushes the full batch; its tail tiles are never gathered or read.
    cb(k_tile_bytes, m * k_tiles_per_block, k_df);  // cb_k_in
    cb(v_tile_bytes, m * v_tiles_per_block, v_df);  // cb_v_in
    cb(tile_bytes, 1, bf);                          // cb_scale
    cb(tile_bytes, Sqt * KT_stride, bf);            // cb_qk_im
    cb(tile_bytes, Sqt, bf);                        // cb_max_a
    cb(tile_bytes, Sqt, bf);                        // cb_max_b
    cb(tile_bytes, Sqt, bf);                        // cb_sum_a
    cb(tile_bytes, Sqt, bf);                        // cb_sum_b
    cb(tile_bytes, Sqt * vDHt, bf);                 // cb_out_a
    cb(tile_bytes, Sqt * vDHt, bf);                 // cb_out_b
    cb(tile_bytes, Sqt, bf);                        // cb_corr
    cb(out_tile_bytes, 2 * out_tiles_per_work, out_df);  // cb_out_im (double-buffered toward the writer)
    cb(idx_row_bytes, 1, bf);                       // cb_idx
    cb(counts_row_bytes, 1, bf);                    // cb_counts
    cb(ctrl_page_bytes, 4, bf);                     // cb_ctrl
    cb(tile_bytes, 1, bf);                          // cb_col_identity
    cb(tile_bytes, 1, bf);                          // cb_recip_scratch
    cb(kreq_page_bytes, 2, bf);                     // cb_kreq
    cb(16, 2, bf);                                  // cb_kack
    cb(tile_bytes, 1, bf);                          // cb_neginf
    cb(tile_bytes, 2 * m, bf);                      // cb_vmask (up to m ragged blocks per chunk, double-buffered)

    // ---- compile-time args ----
    std::vector<uint32_t> reader_ct = {
        W,
        n_kv_blocks,
        m,
        n_q_tiles,
        block_size,
        q_tiles_per_work,
        k_tiles_per_block,
        v_tiles_per_block,
        k_half,
        v_half,
        k_head_stride,
        v_head_stride,
        idx_row_bytes,
        counts_row_bytes,
        q_tile_bytes,
        k_tile_bytes,
        v_tile_bytes};
    for (uint32_t id : {cb_q_in, cb_k_in, cb_v_in, cb_idx, cb_counts, cb_ctrl, cb_kreq, cb_kack, cb_vmask}) {
        reader_ct.push_back(id);
    }
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.block_counts.buffer()).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {
        m,
        n_q_tiles,
        out_tiles_per_work,
        k_tiles_per_block,
        v_tiles_per_block,
        k_half,
        v_half,
        k_head_stride,
        v_head_stride,
        k_tile_bytes,
        v_tile_bytes,
        out_tile_bytes};
    for (uint32_t id : {cb_out_im, cb_scale, cb_col_identity, cb_k_in, cb_v_in, cb_kreq, cb_kack, cb_neginf}) {
        writer_ct.push_back(id);
    }
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.v.buffer()).append_to(writer_ct, writer_crt);

    std::vector<uint32_t> compute_ct = {
        DHt,      vDHt,     Skt,      m,        block_size,      scale_packed,     cb_q_in,   cb_k_in,
        cb_v_in,  cb_scale, cb_qk_im, cb_max_a, cb_max_b,        cb_sum_a,         cb_sum_b,  cb_out_a,
        cb_out_b, cb_corr,  cb_out_im, cb_ctrl, cb_col_identity, cb_recip_scratch, cb_neginf, cb_vmask};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/vsa_sdpa_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/vsa_sdpa_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;

    // Query sub-blocking: Sqt (= 2) tile rows always fit in DEST (>= 4 tiles under fp32 acc).
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    TT_FATAL(Sqt <= dst_size, "vsa_sdpa: {} query tile-rows must fit in DEST ({})", Sqt, dst_size);
    compute_ct.push_back(Sqt);  // qsb == Sqt: a single query band

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/vsa_sdpa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .math_approx_mode = math_approx};
    std::map<std::string, std::string> cdefs{
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    compute_desc.defines = tt::tt_metal::KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* k_buf = t.k.buffer();
    auto* v_buf = t.v.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* counts_buf = t.block_counts.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t work_start = i * dyn.base_work + std::min(i, dyn.extra);
        uint32_t work_count = dyn.base_work + (i < dyn.extra ? 1u : 0u);
        using RArg = VsaSdpaOperation::ReaderArg;
        RtArgs reader_rt(RArg::kReaderArgCount);
        reader_rt[RArg::kReaderQAddr] = q_buf;
        reader_rt[RArg::kReaderKAddr] = k_buf;
        reader_rt[RArg::kReaderVAddr] = v_buf;
        reader_rt[RArg::kReaderIdxAddr] = idx_buf;
        reader_rt[RArg::kReaderCountsAddr] = counts_buf;
        reader_rt[RArg::kReaderWorkStart] = work_start;
        reader_rt[RArg::kReaderWorkCount] = work_count;
        reader_desc.emplace_runtime_args(core, reader_rt);

        using WArg = VsaSdpaOperation::WriterArg;
        RtArgs writer_rt(WArg::kWriterArgCount);
        writer_rt[WArg::kWriterOutAddr] = out_buf;
        writer_rt[WArg::kWriterKAddr] = k_buf;
        writer_rt[WArg::kWriterVAddr] = v_buf;
        writer_rt[WArg::kWriterWorkStart] = work_start;
        writer_rt[WArg::kWriterWorkCount] = work_count;
        writer_desc.emplace_runtime_args(core, writer_rt);

        using CArg = VsaSdpaOperation::ComputeArg;
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
