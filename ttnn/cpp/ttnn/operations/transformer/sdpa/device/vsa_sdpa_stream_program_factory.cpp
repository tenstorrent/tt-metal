// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v2) program factory. Head-grouped resident-row scheduling: head h owns the
// contiguous core range [floor(h*C/H), floor((h+1)*C/H)); within it, core j takes rows
// {j, j+stride, ...} (stride = the range size), which spreads the long fully-dense exempt-query
// rows evenly instead of piling them onto one core. Each core streams the ascending-order union of
// its resident rows' listed blocks exactly once per pass of at most R_MAX rows.

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
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
using RtArgs = std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>;

constexpr uint32_t kRMax = 20;         // resident rows per pass (<= 32: the reader tracks rows in one word)
constexpr uint32_t kStreamDepth = 6;   // KV blocks in flight

struct StreamSchedule {
    uint32_t head = 0;
    uint32_t row_start = 0;
    uint32_t row_stride = 1;
    uint32_t row_count = 0;
};

StreamSchedule core_schedule(uint32_t core, uint32_t num_cores, uint32_t heads, uint32_t n_q_tiles) {
    // head h owns cores [floor(h*C/H), floor((h+1)*C/H)); binary-search-free inversion
    uint32_t head = static_cast<uint32_t>((static_cast<uint64_t>(core) * heads) / num_cores);
    while ((static_cast<uint64_t>(head + 1) * num_cores) / heads <= core) {
        ++head;
    }
    while ((static_cast<uint64_t>(head) * num_cores) / heads > core) {
        --head;
    }
    const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(head) * num_cores) / heads);
    const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(head + 1) * num_cores) / heads);
    const uint32_t stride = c1 - c0;
    const uint32_t j = core - c0;
    const uint32_t count = (n_q_tiles > j) ? (n_q_tiles - j + stride - 1) / stride : 0;
    return StreamSchedule{head, j, stride, count};
}
}  // namespace

VsaSdpaOperation::program_factory_t VsaSdpaOperation::select_program_factory(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& /*t*/) {
    if (attrs.streaming) {
        return VsaSdpaStreamProgramFactory{};
    }
    return VsaSdpaProgramFactory{};
}

tt::tt_metal::ProgramDescriptor VsaSdpaOperation::VsaSdpaStreamProgramFactory::create_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output) {
    // Function-scoped to avoid unity-build collisions with the v1 factory's enum.
    enum StreamCB : uint32_t {
        cb_q_res = 0,      // resident Q rows [R_MAX, Sqt*DHt] (RAM-mode)
        cb_k_stream,       // stream slots [depth, Skt*DHt] (RAM-mode)
        cb_v_stream,       // stream slots [depth, Skt*vDHt] (RAM-mode)
        cb_o_res,          // resident O accumulators [R_MAX, Sqt*vDHt] (RAM-mode, bf16)
        cb_max_res,        // resident running max [R_MAX, Sqt] (RAM-mode)
        cb_sum_res,        // resident running sum [R_MAX, Sqt] (RAM-mode)
        cb_maxtmp,         // per-visit new max [Sqt]
        cb_psum,           // per-visit partial sum [Sqt]
        cb_corr,           // per-visit rescale factor [Sqt]
        cb_qk,             // per-visit scores/probs [Sqt, Skt]
        cb_scale,          // reduce identity scaler
        cb_col_identity,   // ones-in-col0 for the sum reduction
        cb_recip_scratch,  // reciprocal scratch
        cb_neginf,         // persistent -inf tile
        cb_vmask,          // ragged partial-column mask tiles (FIFO, one per ragged block in flight)
        cb_ctrl,           // reader -> compute visit/flush messages
        cb_kreq,           // reader -> writer K fetch requests
        cb_kack,           // writer -> reader K completion
        cb_free,           // compute -> reader stream-slot credits
        cb_qdone,          // writer -> compute pass-Q-resident token
        cb_out,            // normalized row outputs (compute -> writer)
        cb_idxrow,         // reader scratch: one index row
        cb_counts,         // reader scratch: the counts row
        cb_bitmap,         // reader scratch: R_MAX x ceil(KVB/32) membership words
        cb_count
    };

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t W = t.indices.logical_shape()[3];
    const uint32_t block_size = attrs.block_size;

    constexpr uint32_t q_chunk_tokens = 64;
    const uint32_t DHt = d / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t Sqt = q_chunk_tokens / tt::constants::TILE_HEIGHT;
    const uint32_t Skt = block_size / tt::constants::TILE_WIDTH;
    const uint32_t n_q_tiles = S / q_chunk_tokens;
    const uint32_t n_kv_blocks = T / block_size;
    const uint32_t k_tiles_per_block = Skt * DHt;
    const uint32_t v_tiles_per_block = Skt * vDHt;
    const uint32_t q_tiles_per_row = Sqt * DHt;
    const uint32_t out_tiles_per_row = Sqt * vDHt;
    const uint32_t k_head_stride = (T / tt::constants::TILE_HEIGHT) * DHt;
    const uint32_t v_head_stride = (T / tt::constants::TILE_HEIGHT) * vDHt;
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);
    const uint32_t idx_row_bytes = W * t.indices.element_size();
    const uint32_t counts_row_bytes = W * t.block_counts.element_size();
    const uint32_t bitmap_bytes = ((kRMax * ((n_kv_blocks + 31) / 32) * 4 + 15) / 16) * 16;

    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t q_tile_bytes = tt::tile_size(q_df);
    const uint32_t k_tile_bytes = tt::tile_size(k_df);
    const uint32_t v_tile_bytes = tt::tile_size(v_df);
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(H <= num_cores, "vsa_sdpa streaming needs at least one core per head (H {}, cores {})", H, num_cores);
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    // ---- CBs (fixed order = StreamCB enum) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(q_tile_bytes, kRMax * q_tiles_per_row, q_df);           // cb_q_res
    cb(k_tile_bytes, kStreamDepth * k_tiles_per_block, k_df);  // cb_k_stream
    cb(v_tile_bytes, kStreamDepth * v_tiles_per_block, v_df);  // cb_v_stream
    cb(tile_bytes, kRMax * out_tiles_per_row, bf);             // cb_o_res
    cb(tile_bytes, kRMax * Sqt, bf);                           // cb_max_res
    cb(tile_bytes, kRMax * Sqt, bf);                           // cb_sum_res
    cb(tile_bytes, Sqt, bf);                                   // cb_maxtmp
    cb(tile_bytes, Sqt, bf);                                   // cb_psum
    cb(tile_bytes, Sqt, bf);                                   // cb_corr
    cb(tile_bytes, Sqt * Skt, bf);                             // cb_qk
    cb(tile_bytes, 1, bf);                                     // cb_scale
    cb(tile_bytes, 1, bf);                                     // cb_col_identity
    cb(tile_bytes, 1, bf);                                     // cb_recip_scratch
    cb(tile_bytes, 1, bf);                                     // cb_neginf
    cb(tile_bytes, kStreamDepth, bf);                          // cb_vmask
    cb(16, 64, bf);                                            // cb_ctrl
    cb(16, kStreamDepth, bf);                                  // cb_kreq
    cb(16, kStreamDepth, bf);                                  // cb_kack
    cb(16, kStreamDepth + 2, bf);                              // cb_free
    cb(16, 2, bf);                                             // cb_qdone
    cb(out_tile_bytes, 2 * out_tiles_per_row, out_df);         // cb_out
    cb(idx_row_bytes, 1, bf);                                  // cb_idxrow
    cb(counts_row_bytes, 1, bf);                               // cb_counts
    cb(bitmap_bytes, 1, bf);                                   // cb_bitmap

    // ---- compile-time args ----
    std::vector<uint32_t> reader_ct = {
        W,
        n_kv_blocks,
        n_q_tiles,
        block_size,
        kRMax,
        kStreamDepth,
        v_tiles_per_block,
        v_head_stride,
        idx_row_bytes,
        counts_row_bytes,
        v_tile_bytes};
    for (uint32_t id : {cb_v_stream, cb_idxrow, cb_counts, cb_bitmap, cb_ctrl, cb_kreq, cb_kack, cb_free, cb_vmask}) {
        reader_ct.push_back(id);
    }
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.v.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.block_counts.buffer()).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {
        n_q_tiles,
        kRMax,
        q_tiles_per_row,
        out_tiles_per_row,
        k_tiles_per_block,
        k_head_stride,
        q_tile_bytes,
        k_tile_bytes,
        out_tile_bytes};
    for (uint32_t id : {cb_q_res, cb_k_stream, cb_scale, cb_col_identity, cb_neginf, cb_kreq, cb_kack, cb_qdone, cb_out}) {
        writer_ct.push_back(id);
    }
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(writer_ct, writer_crt);

    std::vector<uint32_t> compute_ct = {
        DHt, vDHt, Skt, Sqt, kRMax, block_size, scale_packed,
        cb_q_res, cb_k_stream, cb_v_stream, cb_o_res, cb_max_res, cb_sum_res, cb_maxtmp,
        cb_psum, cb_corr, cb_qk, cb_scale, cb_col_identity, cb_recip_scratch, cb_neginf,
        cb_vmask, cb_ctrl, cb_free, cb_qdone, cb_out, kStreamDepth};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/vsa_sdpa_stream_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/vsa_sdpa_stream_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    TT_FATAL(Sqt * vDHt <= dst_size, "vsa_sdpa streaming needs the PV subblock ({}) to fit DEST ({})",
             Sqt * vDHt, dst_size);

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/vsa_sdpa_stream_compute.cpp";
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
        const auto sched = core_schedule(i, num_cores, H, n_q_tiles);

        RtArgs reader_rt(7);
        reader_rt[0] = v_buf;
        reader_rt[1] = idx_buf;
        reader_rt[2] = counts_buf;
        reader_rt[3] = sched.head;
        reader_rt[4] = sched.row_start;
        reader_rt[5] = sched.row_stride;
        reader_rt[6] = sched.row_count;
        reader_desc.emplace_runtime_args(core, reader_rt);

        RtArgs writer_rt(7);
        writer_rt[0] = out_buf;
        writer_rt[1] = k_buf;
        writer_rt[2] = q_buf;
        writer_rt[3] = sched.head;
        writer_rt[4] = sched.row_start;
        writer_rt[5] = sched.row_stride;
        writer_rt[6] = sched.row_count;
        writer_desc.emplace_runtime_args(core, writer_rt);

        RtArgs compute_rt(1);
        compute_rt[0] = sched.row_count;
        compute_desc.emplace_runtime_args(core, compute_rt);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void VsaSdpaOperation::VsaSdpaStreamProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value) {
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    (void)attrs;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        auto& reader = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader[0] = t.v.buffer()->address();
        reader[1] = t.indices.buffer()->address();
        reader[2] = t.block_counts.buffer()->address();
        auto& writer = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer[0] = tensor_return_value.buffer()->address();
        writer[1] = t.k.buffer()->address();
        writer[2] = t.q.buffer()->address();
    }
}

}  // namespace ttnn::prim
