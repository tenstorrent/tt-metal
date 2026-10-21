// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_sdpa_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <string>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor DeepseekV4PrefillSDPAOperation::ProgramFactory::create_descriptor(
    const DeepseekV4PrefillSDPAParams& attrs, const DeepseekV4PrefillSDPAInputs& t, Tensor& output) {
    namespace ds = ::deepseek_v4_prefill_sdpa;
    // Fixed CB ids shared with the kernels (local: unity builds put sparse_sdpa's same-named ids in this TU). The
    // compute kernel is sparse_sdpa's (bf16 path), so the flash-state buffers follow its contract; its scaled-FP8
    // CB ids are passed but never allocated or touched.
    enum : uint32_t {
        cb_q_rm = 0,        // Q rows (row-major, reader -> compute tilize)
        cb_q_in,            // Q tiled [Sqt, DHt]
        cb_k_rm,            // K chunk rows (row-major), gathered from window_kv and cache_kv
        cb_k_in,            // K tiled [Skt, DHt]
        cb_neginf,          // all-(-inf) tile for fully masked key tiles
        cb_mask_part,       // per-token partial-boundary mask tile
        cb_scale,           // reduce identity scaler
        cb_qk_im,           // scores [Sqt, Skt]
        cb_max_a,           // running max ping-pong [Sqt, 1]
        cb_max_b,           //
        cb_sum_a,           // running sum ping-pong [Sqt, 1]
        cb_sum_b,           //
        cb_out_a,           // running out ping-pong [Sqt, vDHt]
        cb_out_b,           //
        cb_corr,            // exp(prev_max - cur_max) [Sqt, 1]
        cb_out_im,          // final out before untilize [Sqt, vDHt]
        cb_out_rm,          // untilized out (compute -> writer)
        cb_idx,             // one token's key list (uint32), shared by reader and writer
        cb_ctrl,            // reader -> compute: active chunk count, valid key count
        cb_col_identity,    // ones in column 0: finalizes the row-sum
        cb_recip_scratch,   // reciprocal scratch for normalize_row_streaming
        cb_kreq,            // reader -> writer gather request
        cb_kack,            // writer -> reader gather done
        cb_k_rope_rm,       // unused (scaled FP8 only)
        cb_k_scale_bcast,   // unused (scaled FP8 only)
        cb_k_latent_tile,   // unused (scaled FP8 only)
        cb_k_rope_tile,     // unused (scaled FP8 only)
        cb_attention_sink,  // one sink scalar per head, column 0
        cb_sink_scratch,    // writer scratch for the sink row
        cb_start,           // reader scratch for the start position
    };
    tt::tt_metal::ProgramDescriptor desc;

    const bool has_cache = t.cache_kv.has_value();
    const bool use_attention_sink = t.attention_sink.has_value();
    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t k_dim = t.q.logical_shape()[3];
    const uint32_t v_dim = attrs.v_dim;
    const uint32_t topk = has_cache ? t.cache_idx->logical_shape()[3] : 0;
    const uint32_t window = attrs.window_size;
    const uint32_t k_chunk = attrs.k_chunk_size;
    // A token's list holds at most topk cache rows plus window rows; whole chunks are gathered.
    const uint32_t list_capacity = tt::round_up(topk + window, k_chunk);

    const uint32_t DHt = k_dim / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_dim / tt::constants::TILE_WIDTH;
    const uint32_t Skt = k_chunk / tt::constants::TILE_WIDTH;
    // Compute works on whole tiles of heads: the reader zero-fills Q rows [H, H_pad) and the writer drops them.
    const uint32_t H_pad = tt::round_up(H, tt::constants::TILE_HEIGHT);
    const uint32_t Sqt = H_pad / tt::constants::TILE_HEIGHT;
    const uint32_t row_bytes = k_dim * sizeof(uint16_t);
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);

    auto* device = t.q.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    const auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;

    const auto cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = bf, .page_size = page_size}}},
        });
    };
    cb(cb_q_rm, row_bytes, H_pad);
    cb(cb_q_in, tile_bytes, Sqt * DHt);
    cb(cb_k_rm, row_bytes, k_chunk);
    cb(cb_k_in, tile_bytes, Skt * DHt);
    cb(cb_neginf, tile_bytes, 1);
    cb(cb_mask_part, tile_bytes, 1);
    cb(cb_scale, tile_bytes, 1);
    cb(cb_qk_im, tile_bytes, Sqt * Skt);
    cb(cb_max_a, tile_bytes, Sqt);
    cb(cb_max_b, tile_bytes, Sqt);
    cb(cb_sum_a, tile_bytes, Sqt);
    cb(cb_sum_b, tile_bytes, Sqt);
    cb(cb_out_a, tile_bytes, Sqt * vDHt);
    cb(cb_out_b, tile_bytes, Sqt * vDHt);
    cb(cb_corr, tile_bytes, Sqt);
    cb(cb_out_im, tile_bytes, Sqt * vDHt);
    cb(cb_out_rm, tile_bytes, Sqt * vDHt);
    cb(cb_idx, list_capacity * sizeof(uint32_t), 1);
    cb(cb_ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH);
    cb(cb_col_identity, tile_bytes, 1);
    cb(cb_recip_scratch, tile_bytes, 1);
    cb(cb_kreq, ::sparse_sdpa::gather_request::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH);
    cb(cb_kack, ::sparse_sdpa::ACK_PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH);
    cb(cb_start, ds::START_PAGE_BYTES, 1);
    if (use_attention_sink) {
        cb(cb_attention_sink, tile_bytes, Sqt);
        cb(cb_sink_scratch,
           tt::round_up(H * static_cast<uint32_t>(sizeof(uint16_t)), tt::tt_metal::hal::get_dram_alignment()),
           1);
    }

    // Without a cache the cache and index accessors are never dereferenced; window_kv stands in for them.
    auto* q_buf = t.q.buffer();
    auto* window_buf = t.window_kv.buffer();
    auto* cache_buf = has_cache ? t.cache_kv->buffer() : window_buf;
    auto* idx_buf = has_cache ? t.cache_idx->buffer() : window_buf;
    auto* start_buf = t.start.buffer();
    auto* out_buf = output.buffer();
    auto* sink_buf = use_attention_sink ? t.attention_sink->buffer() : nullptr;

    std::vector<uint32_t> reader_ct = {
        H,
        S,
        topk,
        list_capacity,
        k_chunk,
        k_dim,
        window,
        cb_q_rm,
        cb_k_rm,
        cb_mask_part,
        cb_idx,
        cb_ctrl,
        cb_kreq,
        cb_kack,
        cb_start,
        attrs.compress_ratio,
        static_cast<uint32_t>(attrs.empty_tail)};
    TT_FATAL(
        reader_ct.size() == ds::reader_ct_arg::END,
        "deepseek_v4_prefill_sdpa reader compile-time argument layout is out of sync");
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(q_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(window_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(cache_buf, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(start_buf).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {
        H,
        S,
        vDHt,
        k_dim,
        cb_out_rm,
        cb_scale,
        cb_col_identity,
        cb_neginf,
        cb_idx,
        cb_kreq,
        cb_kack,
        static_cast<uint32_t>(use_attention_sink),
        cb_attention_sink,
        cb_sink_scratch};
    TT_FATAL(
        writer_ct.size() == ds::writer_ct_arg::END,
        "deepseek_v4_prefill_sdpa writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(window_buf).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(cache_buf, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(sink_buf).append_to(writer_ct, writer_crt);

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;
    // The DST-bound primitives hold qsb query tile-rows at once: the largest divisor of Sqt that fits DEST.
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    uint32_t qsb = 1;
    for (uint32_t d = std::min(Sqt, dst_size); d >= 1; --d) {
        if (Sqt % d == 0) {
            qsb = d;
            break;
        }
    }
    const std::vector<uint32_t> compute_ct = {
        H_pad,
        DHt,
        vDHt,
        Skt,
        std::bit_cast<uint32_t>(attrs.scale),
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
        cb_recip_scratch,
        /*scaled_kv=*/0,
        cb_k_rope_rm,
        cb_k_scale_bcast,
        cb_k_latent_tile,
        cb_k_rope_tile,
        static_cast<uint32_t>(math_approx),
        qsb,
        /*packed_row_bytes=*/0,
        static_cast<uint32_t>(use_attention_sink),
        cb_attention_sink};
    TT_FATAL(
        compute_ct.size() == ::sparse_sdpa::compute_ct_arg::END,
        "deepseek_v4_prefill_sdpa compute compile-time argument layout is out of sync");

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_sdpa_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_sdpa_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .math_approx_mode = math_approx};

    reader_desc.emplace_common_runtime_args({q_buf, window_buf, cache_buf, idx_buf, start_buf});
    writer_desc.emplace_common_runtime_args({out_buf, window_buf, cache_buf, sink_buf});
    TT_FATAL(
        reader_desc.common_runtime_args.size() == ds::reader_common_arg::END &&
            writer_desc.common_runtime_args.size() == ds::writer_common_arg::END,
        "deepseek_v4_prefill_sdpa common runtime argument layout is out of sync");
    reader_desc.common_runtime_args.insert(reader_desc.common_runtime_args.end(), reader_crt.begin(), reader_crt.end());
    writer_desc.common_runtime_args.insert(writer_desc.common_runtime_args.end(), writer_crt.begin(), writer_crt.end());

    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        const uint32_t tok_start = i * base + std::min(i, extra);
        const uint32_t tok_count = base + (i < extra ? 1u : 0u);
        reader_desc.emplace_runtime_args(core, {tok_start, tok_count});
        writer_desc.emplace_runtime_args(core, {tok_start, tok_count});
        compute_desc.emplace_runtime_args(core, {tok_start, tok_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void DeepseekV4PrefillSDPAOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DeepseekV4PrefillSDPAParams& /*operation_attributes*/,
    const DeepseekV4PrefillSDPAInputs& t,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    namespace ds = ::deepseek_v4_prefill_sdpa;
    const uint32_t window = t.window_kv.buffer()->address();
    const uint32_t cache = t.cache_kv.has_value() ? t.cache_kv->buffer()->address() : window;
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, 1);
    reader_args[ds::reader_common_arg::Q_ADDRESS] = t.q.buffer()->address();
    reader_args[ds::reader_common_arg::WINDOW_ADDRESS] = window;
    reader_args[ds::reader_common_arg::CACHE_ADDRESS] = cache;
    reader_args[ds::reader_common_arg::INDICES_ADDRESS] =
        t.cache_idx.has_value() ? t.cache_idx->buffer()->address() : window;
    reader_args[ds::reader_common_arg::START_ADDRESS] = t.start.buffer()->address();
    writer_args[ds::writer_common_arg::OUTPUT_ADDRESS] = tensor_return_value.buffer()->address();
    writer_args[ds::writer_common_arg::WINDOW_ADDRESS] = window;
    writer_args[ds::writer_common_arg::CACHE_ADDRESS] = cache;
    writer_args[ds::writer_common_arg::ATTENTION_SINK_ADDRESS] =
        t.attention_sink.has_value() ? t.attention_sink->buffer()->address() : 0u;
}

}  // namespace ttnn::prim
