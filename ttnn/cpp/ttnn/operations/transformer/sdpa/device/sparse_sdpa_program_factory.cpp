// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>  // GetCommonRuntimeArgs (cache-hit in-place patch)
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>  // corerange_to_cores
#include <array>
#include <bit>
#include <string>
#include <vector>

namespace ttnn::prim {

// Fixed CB id convention shared with the kernels.
// Flash/online-softmax keeps ping-pong (prev/cur) buffers for the running max, sum, and output.
enum SparseCB : uint32_t {
    cb_q_rm = 0,   // Q rows (row-major, reader -> compute tilize)
    cb_q_in,       // Q tiled [Sqt, DHt]
    cb_k_rm,       // K chunk rows (row-major)
    cb_k_in,       // K tiled [Skt,DHt], or scaled-cache latent [Skt,vDHt] in BFP8
    cb_neginf,     // all-(-inf) tile (writer-built once): bcast-added to fully-masked key tiles
    cb_mask_part,  // per-token partial-boundary mask tile (row 0: -inf for cols >= valid%32)
    cb_scale,      // reduce identity scaler (1 tile)
    cb_qk_im,      // scores [Sqt, Skt]
    cb_max_a,      // running max ping-pong [Sqt, 1]
    cb_max_b,
    cb_sum_a,  // running sum ping-pong [Sqt, 1]
    cb_sum_b,
    cb_out_a,  // running out ping-pong [Sqt, vDHt] (single-buffered for L1 accumulation)
    cb_out_b,
    cb_corr,            // exp(prev_max - cur_max) correction [Sqt, 1]
    cb_out_im,          // fixed pre-untilize copy of the final out [Sqt, vDHt]
    cb_out_rm,          // untilized row-major out (compute -> writer)
    cb_idx,             // reader-internal: one token's index row (uint32)
    cb_ctrl,            // reader -> compute: per-segment control_message (active chunk count is 0 for an empty HEAD)
    cb_col_identity,    // ones-in-col0 (writer-built): finalizes the partial row-sum via matmul_reduce
    cb_recip_scratch,   // 1-tile reciprocal scratch for normalize_row_streaming
    cb_kreq,            // reader->writer K-gather handoff (dual-NoC split)
    cb_kack,            // writer->reader ack that its half of the chunk landed in cb_k_rm
    cb_k_rope_rm,       // scaled FP8 only: format-only BF16 view of one packed row slab
    cb_k_scale_bcast,   // scaled FP8 only: per-row broadcast scale tiles (FP32, or FP16 with fused dequant)
    cb_k_latent_tile,   // scaled FP8 without fused dequant: one TILE_HEIGHT-row BFP8 latent slab
    cb_k_rope_tile,     // scaled FP8 only: one K chunk's BF16 RoPE tiles
    cb_attention_sink,  // persistent first-column vector: one sink scalar per head
    cb_sink_scratch,    // writer scratch for gathering sink scalars
    // split-K (sparse_sdpa_common.hpp seg_kind): this core's HEAD flash state out, the neighbour's HEAD state in
    cb_head_ready,  // compute -> writer: the HEAD state is in cb_out_b / cb_max_b / cb_sum_b
    cb_head_ack,    // writer -> compute: the HEAD state was sent, compute can pop it
    cb_recv_out,    // core i+1's HEAD state, NoC-written straight into these CBs' base
    cb_recv_max,
    cb_recv_sum,
    cb_merge_ctrl,  // writer -> compute: the neighbour's semaphore value (SPLIT_PARTIAL_*)
    cb_corr2,       // exp(recv_max - merged_max)
    cb_count
};

namespace {

// Shapes and formats shared by make_cbs and create_descriptor.
struct OpDims {
    bool use_attention_sink;
    bool scaled_kv;
    // Fused FP8 dequant (one FPU pass for tilize x scale, see kernels/compute/sparse_sdpa_fused_dequant.hpp). The
    // attention-sink term is added to the row-sum unscaled, so the sink path keeps the two-pass dequant.
    bool fused_dequant;
    uint32_t H;      // head count, from the tensor (any multiple of TILE_HEIGHT)
    uint32_t S;      // query rows
    uint32_t topk;   // indices per query row
    uint32_t k_dim;  // head dim, from the tensor (e.g. 576)
    uint32_t v_dim;  // V width (op arg; e.g. 512)
    uint32_t k_chunk;
    uint32_t DHt;
    uint32_t vDHt;
    uint32_t Skt;                // tiles per chunk along keys
    uint32_t Sqt;                // query tile-rows (32 heads each)
    uint32_t packed_page_bytes;  // scaled FP8 packed row stride (0 otherwise)
    tt::DataFormat q_df;         // native (row-major) Q format
    tt::DataFormat kv_df;        // native (row-major) KV format
};

OpDims op_dims(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    const bool use_attention_sink = t.attention_sink.has_value();
    const bool scaled_kv = attrs.has_scaled_kv();
    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t k_dim = t.q.logical_shape()[3];
    return OpDims{
        .use_attention_sink = use_attention_sink,
        .scaled_kv = scaled_kv,
        .fused_dequant = scaled_kv && !use_attention_sink,
        .H = H,
        .S = t.q.logical_shape()[2],
        .topk = t.indices.logical_shape()[3],
        .k_dim = k_dim,
        .v_dim = attrs.v_dim,
        .k_chunk = attrs.k_chunk_size,
        .DHt = k_dim / TILE_WIDTH,
        .vDHt = attrs.v_dim / TILE_WIDTH,
        .Skt = attrs.k_chunk_size / TILE_WIDTH,
        .Sqt = H / TILE_HEIGHT,
        .packed_page_bytes = scaled_kv ? static_cast<uint32_t>(t.kv.mesh_buffer().aligned_page_size()) : 0u,
        .q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype()),
        .kv_df = tt::tt_metal::datatype_to_dataformat_converter(t.kv.dtype()),
    };
}

// The CB list (fixed ids = SparseCB enum; descriptor insertion order is irrelevant). select_plan sizes the split-K L1
// footprint from this same list.
tt::tt_metal::ProgramDescriptor::CBDescriptors make_cbs(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, tt::tt_metal::DataType out_dtype, bool split_k) {
    tt::tt_metal::ProgramDescriptor::CBDescriptors cbs;
    const OpDims d = op_dims(attrs, t);
    const uint32_t Sqt = d.Sqt, Skt = d.Skt, DHt = d.DHt, vDHt = d.vDHt;

    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat fp32 = tt::DataFormat::Float32;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);  // 2048
    const uint32_t scale_blocks = ::sparse_sdpa::scale_block_count(d.v_dim);
    // K and Q are read row-major in their native dtype (cb_k_rm / cb_q_rm) and tilized by compute. FP8 (and the
    // scaled cache's FP8 latent) tilizes into BFP8 (near-lossless, PCC 0.99998): it halves cb_k_in / cb_q_in L1 and
    // cheapens both matmuls (the HW within-face transpose works on block-float). BF16 stays BF16.
    const bool kv_is_fp8 = d.kv_df == tt::DataFormat::Fp8_e4m3;
    const tt::DataFormat k_in_df = (d.scaled_kv || kv_is_fp8) ? tt::DataFormat::Bfp8_b : bf;
    const uint32_t k_in_tile_bytes = tt::tile_size(k_in_df);
    const uint32_t k_in_width_tiles = d.scaled_kv ? vDHt : DHt;
    const tt::DataFormat q_in_df = d.q_df == tt::DataFormat::Fp8_e4m3 ? tt::DataFormat::Bfp8_b : bf;
    // Output dtype matches q (compute_output_specs). The final untilize packs the bf16 accumulator (cb_out_im) to
    // this format in cb_out_rm (fp8_e4m3 is a regular float8, not block-float, so it untilizes fine).
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(out_dtype);

    const auto& core_grid = attrs.worker_core_grid;
    const auto cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(cb_q_rm, d.k_dim * t.q.element_size(), d.H, d.q_df);
    cb(cb_q_in, tt::tile_size(q_in_df), Sqt * DHt, q_in_df);
    if (d.scaled_kv) {
        // One double-buffered allocation with an owning FP8 FIFO and a format-only BF16 RoPE alias. Both use
        // the physical packed-page stride. Only cb_k_rm advances FIFO state; compute derives the alias read
        // pointer from the owner before reading RoPE. This lets the reader gather slab N+1 while compute
        // reconstructs slab N without independent alias pointers drifting at the ring wrap.
        cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = d.packed_page_bytes * tt::constants::TILE_HEIGHT * ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH,
            .core_ranges = core_grid,
            .format_descriptors = {{
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_k_rm),
                    .data_format = d.kv_df,
                    .page_size = d.packed_page_bytes},
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_k_rope_rm),
                    .data_format = bf,
                    .page_size = d.packed_page_bytes},
            }},
        });
    } else {
        cb(cb_k_rm, d.k_dim * t.kv.element_size(), d.k_chunk, d.kv_df);
    }
    cb(cb_k_in, k_in_tile_bytes, Skt * k_in_width_tiles, k_in_df);
    cb(cb_neginf, tile_bytes, 1, bf);
    cb(cb_mask_part, tile_bytes, 1, bf);
    cb(cb_scale, tile_bytes, 1, bf);
    cb(cb_qk_im, tile_bytes, Sqt * Skt, bf);
    cb(cb_max_a, tile_bytes, Sqt, bf);
    cb(cb_max_b, tile_bytes, Sqt, bf);
    cb(cb_sum_a, tile_bytes, Sqt, bf);
    cb(cb_sum_b, tile_bytes, Sqt, bf);
    cb(cb_out_a, tile_bytes, Sqt * vDHt, bf);
    cb(cb_out_b, tile_bytes, Sqt * vDHt, bf);
    cb(cb_corr, tile_bytes, Sqt, bf);
    cb(cb_out_im, tile_bytes, Sqt * vDHt, bf);
    cb(cb_out_rm, tt::tile_size(out_df), Sqt * vDHt, out_df);
    cb(cb_idx, d.topk * t.indices.element_size(), 1, bf);
    cb(cb_ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    cb(cb_col_identity, tile_bytes, 1, bf);
    cb(cb_recip_scratch, tile_bytes, 1, bf);
    cb(cb_kreq, ::sparse_sdpa::gather_request::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    cb(cb_kack, ::sparse_sdpa::ACK_PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    if (d.scaled_kv) {
        const uint32_t scale_pages = scale_blocks * ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH;
        if (d.fused_dequant) {
            // FP16 scale tiles. Declared UInt16: the JIT rejects an FP16 (A-exponent) CB next to the BF16 ones, and
            // compute reprograms srcB to Float16 for the fused multiply.
            cb(cb_k_scale_bcast, tt::tile_size(tt::DataFormat::UInt16), scale_pages, tt::DataFormat::UInt16);
        } else {
            cb(cb_k_scale_bcast, tt::tile_size(fp32), scale_pages, fp32);
            cb(cb_k_latent_tile, k_in_tile_bytes, vDHt, tt::DataFormat::Bfp8_b);
        }
        cb(cb_k_rope_tile, tile_bytes, Skt * (DHt - vDHt), bf);
    }
    if (d.use_attention_sink) {
        cb(cb_attention_sink, tile_bytes, Sqt, bf);
        cb(cb_sink_scratch, d.H * sizeof(uint16_t), 1, bf);
    }
    if (split_k) {
        // Split-K merge state: core i+1's HEAD state in, the merge scratch, and the HEAD send handshake.
        cb(cb_recv_out, tile_bytes, Sqt * vDHt, bf);
        for (uint32_t id : {cb_recv_max, cb_recv_sum, cb_corr2}) {
            cb(id, tile_bytes, Sqt, bf);
        }
        cb(cb_merge_ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, 1, bf);
        for (uint32_t id : {cb_head_ready, cb_head_ack}) {
            cb(id, ::sparse_sdpa::ACK_PAGE_BYTES, 1, bf);
        }
    }
    return cbs;
}

// Relative split-K gather speed (%) of Blackhole NoC0 rows y = FIRST_ROW_Y.. when every core gathers K at once: the
// top rows are the slowest and the bottom row the fastest, the same on every chip measured. Tuned on a p150 12x10
// grid for the GLM-5.3 (64 heads) and Kimi K3.1 (96 heads) shapes. Rows outside the table weigh 100.
constexpr std::array<uint32_t, 10> ROW_SPEED_PCT = {85, 84, 88, 97, 102, 107, 110, 112, 114, 115};
constexpr uint32_t FIRST_ROW_Y = 2;

// Split-K unit boundaries: core i owns units [bounds[i], bounds[i + 1]) of the token-major (token, chunk) order.
// Equal split, or (row_weighted) in proportion to ROW_SPEED_PCT so the slow top rows do not finish last. The
// weighted split falls back to the equal one when any core would get less than one token (the S > cores pairwise
// merge needs a whole token per core).
std::vector<uint64_t> split_k_unit_bounds(
    const tt::tt_metal::IDevice& device,
    const std::vector<CoreCoord>& cores,
    uint32_t num_used,
    uint64_t total_units,
    uint32_t n_chunks,
    bool row_weighted) {
    std::vector<uint64_t> cum_w(num_used + 1, 0);  // cum_w[i] = weight of cores [0, i)
    for (uint32_t i = 0; i < num_used; ++i) {
        uint64_t w = 1;
        if (row_weighted) {
            const uint32_t r = device.worker_core_from_logical_core(cores[i]).y - FIRST_ROW_Y;  // wraps for y < 2
            w = r < ROW_SPEED_PCT.size() ? ROW_SPEED_PCT[r] : 100u;
        }
        cum_w[i + 1] = cum_w[i] + w;
    }
    std::vector<uint64_t> bounds(num_used + 1);
    for (uint32_t i = 0; i <= num_used; ++i) {
        bounds[i] = total_units * cum_w[i] / cum_w[num_used];
    }
    if (row_weighted) {
        for (uint32_t i = 0; i < num_used; ++i) {
            if (bounds[i + 1] - bounds[i] < n_chunks) {
                return split_k_unit_bounds(device, cores, num_used, total_units, n_chunks, /*row_weighted=*/false);
            }
        }
    }
    return bounds;
}

}  // namespace

SparseSDPAOperation::SparseSDPAProgramFactory::Plan SparseSDPAOperation::SparseSDPAProgramFactory::select_plan(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    // Split-K: deal (token, chunk) units contiguously over the cores when whole tokens do not divide evenly. GLM per
    // chip has 160 query rows: on 96 cores the whole-token split leaves 64 cores with 2 tokens and 32 with 1
    // (makespan 2 tokens); split-K gives every core 26-27 of the 2560 chunk units (1.67 tokens). Merges are pairwise
    // between neighbouring cores, at most one per core (needs Sqt <= 4 so a merge fits DST once).
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t Sqt = t.q.logical_shape()[1] / tt::constants::TILE_HEIGHT;
    const uint32_t n_chunks = t.indices.logical_shape()[3] / attrs.k_chunk_size;
    const uint32_t num_cores = attrs.worker_core_grid.num_cores();
    // BF16 and plain FP8 KV: split-K only when every token gets two cores (2S <= cores); with more rows it was
    // slower in most measured shapes.
    const bool split_k_ok =
        !t.attention_sink.has_value() && Sqt <= 4 && n_chunks > 1 && (attrs.has_scaled_kv() || 2 * S <= num_cores);
    Plan plan{.split_k = split_k_ok && S > num_cores && (S % num_cores) != 0, .num_used = num_cores};
    // Small S (fewer query rows than cores, e.g. GLM chunk 1024/2048: 32/64 rows per chip on 96 cores): the
    // whole-token split leaves most cores idle at one token-time (~330 us, compute-bound). Deal the (token, chunk)
    // units over `num_used` <= min(num_cores, 2S) cores instead, so every token straddles at most two neighbouring
    // cores and the existing pairwise HEAD -> TAIL merge applies unchanged: every core range must start at a token
    // boundary or run to one (no range strictly inside a token). num_used is the largest such count; S=32 -> 64
    // cores (half a token each), S=64 on 96 cores -> 96 (2/3).
    if (split_k_ok && S < num_cores) {
        const uint64_t units = static_cast<uint64_t>(S) * n_chunks;
        const auto valid = [&](uint32_t m) {
            for (uint32_t i = 0; i < m; ++i) {
                const uint64_t b0 = units * i / m, b1 = units * (i + 1) / m;
                if (b1 == b0 || (b0 % n_chunks != 0 && b1 < (b0 / n_chunks + 1) * n_chunks)) {
                    return false;
                }
            }
            return true;
        };
        for (uint32_t m = std::min<uint32_t>(num_cores, 2 * S); m > S; --m) {
            if (valid(m)) {
                plan = Plan{.split_k = true, .num_used = m};
                break;
            }
        }
    }
    if (!plan.split_k) {
        return plan;
    }
    // Keep the whole-token split when the merge CBs do not fit in the free L1 (e.g. large H with large k_chunk).
    // ProgramImpl::allocate_circular_buffers places each CB at the DRAM alignment, not the L1 one.
    const uint64_t cb_align = t.q.device()->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
    uint64_t cb_bytes = 0;
    for (const auto& c : make_cbs(attrs, t, SparseSDPAOperation::compute_output_specs(attrs, t).data_type(), true)) {
        cb_bytes += (c.total_size + cb_align - 1) / cb_align * cb_align;
    }
    if (cb_bytes > ttnn::operations::data_movement::get_max_l1_space(t.q)) {
        return Plan{.split_k = false, .num_used = num_cores};
    }
    return plan;
}

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, Tensor& output) {
    tt::tt_metal::ProgramDescriptor desc;

    const OpDims d = op_dims(attrs, t);
    const uint32_t H = d.H, S = d.S, topk = d.topk, k_dim = d.k_dim, v_dim = d.v_dim, k_chunk = d.k_chunk;
    const uint32_t DHt = d.DHt, vDHt = d.vDHt, Skt = d.Skt, Sqt = d.Sqt, packed_page_bytes = d.packed_page_bytes;
    const auto use_attention_sink = static_cast<uint32_t>(d.use_attention_sink);
    const auto scaled_kv = static_cast<uint32_t>(d.scaled_kv);
    const auto fused_dequant = static_cast<uint32_t>(d.fused_dequant);
    // Element sizes come from the tensors (no hardcoded byte counts); passed to the kernels as compile args.
    const uint32_t q_elem_bytes = t.q.element_size();
    const uint32_t kv_elem_bytes = t.kv.element_size();
    const uint32_t idx_elem_bytes = t.indices.element_size();  // uint32 -> 4
    const uint32_t out_elem_bytes = output.element_size();

    auto* device = t.q.device();
    const auto& core_grid = attrs.worker_core_grid;
    const auto cores = corerange_to_cores(core_grid, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores = cores.size();

    const uint32_t n_chunks = topk / k_chunk;
    const Plan plan = select_plan(attrs, t);
    const auto split_k = static_cast<uint32_t>(plan.split_k);
    const uint32_t num_used = plan.num_used;
    desc.cbs = make_cbs(attrs, t, output.dtype(), plan.split_k);
    if (plan.split_k) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = 0, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});
    }

    // ---- compile-time args ----
    // CB ids are passed to each kernel as compile-time args (the SparseCB enum is the single source). Each block's
    // order is the kernel's *_ct_arg enum in sparse_sdpa_common.hpp; the TensorAccessorArgs chain last.
    // The cache layout is compile-time so the remap is fully optimized out for natural-order KV.
    std::array<uint32_t, 5> block_cyclic_ct{0, 1, 1, 0, 0};
    if (attrs.has_block_cyclic()) {
        const auto& bc = attrs.block_cyclic.value();
        const uint32_t seq_len_local = t.kv.logical_shape()[2] / bc.sp;
        block_cyclic_ct = {1, bc.chunk_local, bc.sp, seq_len_local - bc.chunk_local, bc.chunk_local * (bc.sp - 1)};
    }

    std::vector<uint32_t> reader_ct = {H, S, topk, k_chunk, k_dim};
    reader_ct.insert(reader_ct.end(), {cb_q_rm, cb_k_rm, cb_mask_part, cb_idx, cb_ctrl});
    reader_ct.insert(reader_ct.end(), {q_elem_bytes, kv_elem_bytes, idx_elem_bytes});
    reader_ct.insert(reader_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    reader_ct.insert(
        reader_ct.end(),
        {scaled_kv, v_dim, cb_k_scale_bcast, packed_page_bytes, cb_kreq, cb_kack, split_k, fused_dequant});
    TT_FATAL(
        reader_ct.size() == ::sparse_sdpa::reader_ct_arg::END,
        "sparse_sdpa reader compile-time argument layout is out of sync");
    // kv (the K cache) uses a RUNTIME tensor shape: its T dimension (the cache length) is passed as common
    // runtime args, NOT compile-time args, so changing T reuses the same program (no recompile). q/indices
    // stay compile-time — their dims define the program. The accessor's runtime metadata is the same on
    // every core, so it rides on the kernel's common (not per-core) runtime args.
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);

    // The writer is the lighter dataflow kernel, so it builds the three persistent compute-input tiles.
    std::vector<uint32_t> writer_ct = {H, S, vDHt, cb_out_rm, cb_scale, cb_col_identity, cb_neginf, out_elem_bytes};
    writer_ct.insert(writer_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    writer_ct.insert(writer_ct.end(), {scaled_kv, k_dim, kv_elem_bytes, cb_idx, cb_kreq, cb_kack, packed_page_bytes});
    writer_ct.insert(writer_ct.end(), {use_attention_sink, cb_attention_sink, cb_sink_scratch});
    writer_ct.insert(
        writer_ct.end(),
        {split_k,
         cb_out_b,
         cb_max_b,
         cb_sum_b,
         cb_head_ready,
         cb_head_ack,
         cb_recv_out,
         cb_recv_max,
         cb_recv_sum,
         cb_merge_ctrl,
         fused_dequant});
    TT_FATAL(
        writer_ct.size() == ::sparse_sdpa::writer_ct_arg::END,
        "sparse_sdpa writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(d.use_attention_sink ? t.attention_sink->buffer() : nullptr)
        .append_to(writer_ct, writer_crt);

    // packer_l1_acc has no ComputeConfigDescriptor field (this op does no L1 packer accumulation).
    [[maybe_unused]] auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);
    // Query sub-blocking: the DST-bound primitives (matmuls, reduce, sub_exp, salad) hold `qsb` query
    // tile-rows live in DEST, so qsb must be <= dst_size (8 without fp32 acc, 4 with, half-sync). Process
    // Sqt rows in q_groups = Sqt/qsb passes. Pick the largest divisor of Sqt that fits so groups are
    // equal-height (salad's sbh is a compile-time template param). qsb==Sqt (one group) for small H.
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    uint32_t qsb = 1;
    for (uint32_t c = std::min(Sqt, dst_size); c >= 1; --c) {
        if (Sqt % c == 0) {
            qsb = c;
            break;
        }
    }
    const std::vector<uint32_t> compute_ct = {
        H,
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
        scaled_kv,
        cb_k_rope_rm,
        cb_k_scale_bcast,
        cb_k_latent_tile,
        cb_k_rope_tile,
        static_cast<uint32_t>(math_approx),
        qsb,
        packed_page_bytes,
        use_attention_sink,
        cb_attention_sink,
        split_k,
        cb_head_ready,
        cb_head_ack,
        cb_recv_out,
        cb_recv_max,
        cb_recv_sum,
        cb_merge_ctrl,
        cb_corr2,
        fused_dequant};
    TT_FATAL(
        compute_ct.size() == ::sparse_sdpa::compute_ct_arg::END,
        "sparse_sdpa compute compile-time argument layout is out of sync");

    // ---- kernels ----
    const auto make_kernel =
        [&](const char* source, std::vector<uint32_t> ct, tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
            tt::tt_metal::KernelDescriptor k;
            k.kernel_source = std::string("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/") + source;
            k.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            k.core_ranges = core_grid;
            k.compile_time_args = std::move(ct);
            k.config = std::move(config);
            return k;
        };
    // fp8 inputs must unpack into a 32-bit dest (fp8 -> fp32 in DEST, then packed to bfp8 cb_*_in).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (d.kv_df == tt::DataFormat::Fp8_e4m3) {
        unpack_to_dest_mode[cb_k_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (d.q_df == tt::DataFormat::Fp8_e4m3) {
        unpack_to_dest_mode[cb_q_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    auto reader_desc =
        make_kernel("dataflow/sparse_sdpa_reader.cpp", std::move(reader_ct), tt::tt_metal::ReaderConfigDescriptor{});
    auto writer_desc =
        make_kernel("dataflow/sparse_sdpa_writer.cpp", std::move(writer_ct), tt::tt_metal::WriterConfigDescriptor{});
    auto compute_desc = make_kernel(
        "compute/sparse_sdpa_compute.cpp",
        compute_ct,
        tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_acc,
            .dst_full_sync_en = dst_full_sync,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .math_approx_mode = math_approx});

    // Indexed KV cache: the gather page ids are offset by cache_batch_idx * T to select the cache's batch
    // slot (0 when not indexed). The cache-hit override re-derives this from the current attrs/tensor T.
    const uint32_t kv_batch_page_offset = attrs.cache_batch_idx.value_or(0) * t.kv.logical_shape()[2];
    reader_desc.emplace_common_runtime_args({t.q.buffer(), t.kv.buffer(), t.indices.buffer(), kv_batch_page_offset});
    writer_desc.emplace_common_runtime_args(
        {output.buffer(),
         t.kv.buffer(),
         kv_batch_page_offset,
         d.use_attention_sink ? t.attention_sink->buffer() : nullptr});
    TT_FATAL(
        reader_desc.common_runtime_args.size() == ::sparse_sdpa::reader_common_arg::END &&
            writer_desc.common_runtime_args.size() == ::sparse_sdpa::writer_common_arg::END,
        "sparse_sdpa common runtime argument layout is out of sync");
    reader_desc.common_runtime_args.insert(reader_desc.common_runtime_args.end(), reader_crt.begin(), reader_crt.end());
    writer_desc.common_runtime_args.insert(writer_desc.common_runtime_args.end(), writer_crt.begin(), writer_crt.end());

    // ---- per-core work: whole tokens, or split-K (token, chunk) unit ranges ----
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;
    std::vector<uint64_t> unit_bounds;
    if (plan.split_k) {
        const bool row_weighted = S > num_cores && device->arch() == tt::ARCH::BLACKHOLE;
        unit_bounds =
            split_k_unit_bounds(*device, cores, num_used, static_cast<uint64_t>(S) * n_chunks, n_chunks, row_weighted);
    }
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t tok_start = i * base + std::min(i, extra);
        uint32_t tok_count = base + (i < extra ? 1u : 0u);
        uint32_t head_c0 = 0, tail_c1 = 0, prev_x = 0, prev_y = 0;
        if (plan.split_k && i >= num_used) {
            tok_start = 0;  // small-S split-K: cores past num_used stay idle (no segments)
            tok_count = 0;
        } else if (plan.split_k) {
            // Core i owns units [b0, b1) of the token-major (token, chunk) order.
            const uint64_t b0 = unit_bounds[i];
            const uint64_t b1 = unit_bounds[i + 1];
            const uint32_t t0 = b0 / n_chunks, c0 = b0 % n_chunks, t1 = b1 / n_chunks, c1 = b1 % n_chunks;
            // S > cores: >= one token of units per core. S < cores: a range may be shorter than a token but must
            // touch a token boundary, so every token straddles at most two cores (pairwise merge).
            TT_FATAL(
                S < num_cores ? (c0 == 0 || b1 >= (static_cast<uint64_t>(t0) + 1) * n_chunks) : b1 - b0 >= n_chunks,
                "sparse_sdpa split-K: a core range lies strictly inside one token (would need a multi-way merge)");
            head_c0 = c0;                       // HEAD = token t0, chunks [c0, n_chunks)
            tok_start = c0 != 0 ? t0 + 1 : t0;  // FULL tokens [tok_start, t1)
            tok_count = t1 - tok_start;
            tail_c1 = c1;  // TAIL = token t1, chunks [0, c1)
            if (head_c0 != 0) {
                TT_FATAL(i > 0, "sparse_sdpa split-K: core 0 cannot have a HEAD segment");
                const auto prev = device->worker_core_from_logical_core(cores[i - 1]);
                prev_x = prev.x;
                prev_y = prev.y;
            }
        }
        for (auto* kernel : {&reader_desc, &writer_desc, &compute_desc}) {
            kernel->emplace_runtime_args(core, {tok_start, tok_count, head_c0, tail_c1, prev_x, prev_y});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void SparseSDPAOperation::SparseSDPAProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseSDPAParams& operation_attributes,
    const SparseSDPAInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Token assignments are fixed by the hashed q shape. Refresh only the uniform dispatch values;
    // tensor-accessor metadata follows these slots and remains unchanged under the cache key.
    const auto& t = tensor_args;
    const uint32_t offset = operation_attributes.cache_batch_idx.value_or(0) * t.kv.logical_shape()[2];
    const uint32_t kv_addr = t.kv.mesh_buffer().address();
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, 1);
    reader_args[::sparse_sdpa::reader_common_arg::Q_ADDRESS] = t.q.mesh_buffer().address();
    reader_args[::sparse_sdpa::reader_common_arg::KV_ADDRESS] = kv_addr;
    reader_args[::sparse_sdpa::reader_common_arg::INDICES_ADDRESS] = t.indices.mesh_buffer().address();
    reader_args[::sparse_sdpa::reader_common_arg::KV_BATCH_PAGE_OFFSET] = offset;
    writer_args[::sparse_sdpa::writer_common_arg::OUTPUT_ADDRESS] = tensor_return_value.mesh_buffer().address();
    writer_args[::sparse_sdpa::writer_common_arg::KV_ADDRESS] = kv_addr;
    writer_args[::sparse_sdpa::writer_common_arg::KV_BATCH_PAGE_OFFSET] = offset;
    writer_args[::sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS] =
        t.attention_sink.has_value() ? t.attention_sink->mesh_buffer().address() : 0u;
}

}  // namespace ttnn::prim
