// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <array>
#include <bit>
#include <string>

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
    cb_corr,           // exp(prev_max - cur_max) correction [Sqt, 1]
    cb_out_im,         // fixed pre-untilize copy of the final out [Sqt, vDHt]
    cb_out_rm,         // untilized row-major out (compute -> writer)
    cb_idx,            // reader-internal: one token's index row (uint32)
    cb_ctrl,           // reader -> compute: [active chunk count (=ceil(valid_keys/k_chunk)), valid_keys] per token
    cb_col_identity,   // ones-in-col0 (writer-built): finalizes the partial row-sum via matmul_reduce
    cb_recip_scratch,  // 1-tile reciprocal scratch for normalize_row_streaming
    cb_kreq,           // reader->writer K-gather handoff (dual-NoC split)
    cb_kack,           // writer->reader ack that its half of the chunk landed in cb_k_rm
    cb_k_rope_rm,      // scaled FP8 only: format-only BF16 view of one packed row slab
    cb_k_scale_bcast,  // scaled FP8 only: one FP32 per-row broadcast tile per scale block
    cb_k_latent_tile,  // scaled FP8 only: one TILE_HEIGHT-row BFP8 latent slab
    cb_k_rope_tile,    // scaled FP8 only: one K chunk's BF16 RoPE tiles
    cb_count
};

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, Tensor& output) {
    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];  // head count, from the tensor (any multiple of TILE_HEIGHT)
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t topk = t.indices.logical_shape()[3];
    const uint32_t k_dim = t.q.logical_shape()[3];  // head dim, from the tensor (e.g. 576)
    const uint32_t v_dim = attrs.v_dim;             // V width (op arg; e.g. 512)

    const uint32_t DHt = k_dim / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_dim / tt::constants::TILE_WIDTH;
    const uint32_t k_chunk = attrs.k_chunk_size;
    const uint32_t Skt = k_chunk / tt::constants::TILE_WIDTH;  // tiles per chunk along keys
    const uint32_t Sqt = H / tt::constants::TILE_HEIGHT;       // query tile-rows (32 heads each)
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);

    // Element sizes come from the tensors (no hardcoded byte counts); passed to the kernels as compile args.
    const uint32_t q_elem_bytes = t.q.element_size();
    const bool scaled_kv = t.has_scaled_kv();
    const uint32_t kv_elem_bytes = t.kv.element_size();
    const uint32_t idx_elem_bytes = t.indices.element_size();  // uint32 -> 4
    const uint32_t out_elem_bytes = output.element_size();
    const uint32_t q_row_bytes = k_dim * q_elem_bytes;
    const uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    const uint32_t packed_page_bytes = scaled_kv ? t.kv.buffer()->aligned_page_size() : 0;
    const uint32_t scale_blocks = ::sparse_sdpa::scale_block_count(v_dim);
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat fp32 = tt::DataFormat::Float32;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);  // 2048
    constexpr uint32_t fp32_tile_bytes = tt::tile_size(fp32);
    // cb_k_rm holds the row-major K in its native dtype (fp8 or bf16); compute tilizes it into cb_k_in.
    // For fp8 K, cb_k_in is bfp8_b (fp8->bfp8 is near-lossless): it halves cb_k_in's L1 footprint vs bf16
    // and both matmuls read it directly (the HW within-face transpose works on block-float). The compute
    // kernel restores the bf16 unpack srcA / pack formats after each per-chunk K tilize (see comments there).
    const tt::DataFormat native_kv_df = tt::tt_metal::datatype_to_dataformat_converter(t.kv.dtype());
    const bool kv_storage_is_fp8 = native_kv_df == tt::DataFormat::Fp8_e4m3;
    const bool plain_kv_is_fp8 = !scaled_kv && kv_storage_is_fp8;
    // When K arrives as fp8, tilize it into bfp8_b cb_k_in (fp8->bfp8 is near-lossless, PCC 0.99998) to
    // halve cb_k_in's L1 footprint and cheapen both matmuls. bf16 K keeps bf16 cb_k_in (no precision loss).
    const tt::DataFormat k_in_df = (scaled_kv || plain_kv_is_fp8) ? tt::DataFormat::Bfp8_b : bf;
    const uint32_t k_in_tile_bytes = tt::tile_size(k_in_df);
    const uint32_t k_in_width_tiles = scaled_kv ? vDHt : DHt;
    // Q is handled symmetrically: fp8 Q -> bfp8_b cb_q_in (halves cb_q_in L1, which scales with H), bf16
    // Q -> bf16 cb_q_in. Q is read row-major in its native dtype (cb_q_rm) and tilized into cb_q_in.
    const tt::DataFormat q_rm_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const bool q_is_fp8 = (q_rm_df == tt::DataFormat::Fp8_e4m3);
    const tt::DataFormat q_in_df = q_is_fp8 ? tt::DataFormat::Bfp8_b : bf;
    const uint32_t q_in_tile_bytes = tt::tile_size(q_in_df);
    // Output dtype matches q (compute_output_specs). The final untilize packs the bf16 accumulator
    // (cb_out_im) to this format in cb_out_rm — fp8_e4m3 is a regular float8, not block-float, so it
    // untilizes fine. cb_out_rm's tile-sized pages use the output format's tile size (fp8 -> 1024, bf16 -> 2048).
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    auto* device = t.q.device();
    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;

    // ---- CBs (fixed ids = SparseCB enum; descriptor insertion order is irrelevant) ----
    const auto cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(cb_q_rm, q_row_bytes, H, q_rm_df);              // cb_q_rm : H row-sticks (native Q dtype: fp8 or bf16)
    cb(cb_q_in, q_in_tile_bytes, Sqt * DHt, q_in_df);  // cb_q_in : [Sqt,DHt] (bfp8 when Q is fp8)
    if (scaled_kv) {
        // One double-buffered allocation with an owning FP8 FIFO and a format-only BF16 RoPE alias. Both use
        // the physical packed-page stride. Only cb_k_rm advances FIFO state; compute derives the alias read
        // pointer from the owner before reading RoPE. This lets the reader gather slab N+1 while compute
        // reconstructs slab N without independent alias pointers drifting at the ring wrap.
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = packed_page_bytes * tt::constants::TILE_HEIGHT * ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH,
            .core_ranges = core_grid,
            .format_descriptors = {{
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_k_rm),
                    .data_format = native_kv_df,
                    .page_size = packed_page_bytes},
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_k_rope_rm),
                    .data_format = bf,
                    .page_size = packed_page_bytes},
            }},
        });
    } else {
        cb(cb_k_rm, k_row_bytes, k_chunk, native_kv_df);
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
    cb(cb_out_rm, out_tile_bytes, Sqt * vDHt, out_df);
    cb(cb_idx, topk * idx_elem_bytes, 1, bf);
    cb(cb_ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    cb(cb_col_identity, tile_bytes, 1, bf);
    cb(cb_recip_scratch, tile_bytes, 1, bf);
    cb(cb_kreq, ::sparse_sdpa::gather_request::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    cb(cb_kack, ::sparse_sdpa::ACK_PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    if (scaled_kv) {
        cb(cb_k_scale_bcast, fp32_tile_bytes, scale_blocks * ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, fp32);
        cb(cb_k_latent_tile, k_in_tile_bytes, vDHt, tt::DataFormat::Bfp8_b);
        cb(cb_k_rope_tile, tile_bytes, Skt * (DHt - vDHt), bf);
    }

    // ---- compile-time args ----
    // CB ids are passed to each kernel as compile-time args (the SparseCB enum is the single source).
    // Keep each block's order in sync with the kernel's reads. Layout: scalars, then CB ids, then the
    // element-size args, then the TensorAccessorArgs (which must chain last). The element sizes sit after
    // the CB ids so adding them leaves every CB-id index unchanged (only the accessor offset shifts).
    std::vector<uint32_t> reader_ct = {H, S, topk, k_chunk, k_dim};
    for (uint32_t id : {cb_q_rm, cb_k_rm, cb_mask_part, cb_idx, cb_ctrl}) {
        reader_ct.push_back(id);
    }
    reader_ct.push_back(q_elem_bytes);
    reader_ct.push_back(kv_elem_bytes);
    reader_ct.push_back(idx_elem_bytes);
    // The cache layout is compile-time so the remap is fully optimized out for natural-order KV.
    const auto block_cyclic_ct = [&attrs, &t]() {
        std::array<uint32_t, 5> args{0, 1, 1, 0, 0};
        if (!attrs.has_block_cyclic()) {
            return args;
        }
        const auto& bc = attrs.block_cyclic.value();
        const uint32_t seq_len_local = t.kv.logical_shape()[2] / bc.sp;
        args = {
            1,
            bc.chunk_local,
            bc.sp,
            seq_len_local - bc.chunk_local,
            bc.chunk_local * (bc.sp - 1),
        };
        return args;
    }();
    reader_ct.insert(reader_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    reader_ct.insert(
        reader_ct.end(),
        {static_cast<uint32_t>(scaled_kv), v_dim, cb_k_scale_bcast, packed_page_bytes, cb_kreq, cb_kack});
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
    writer_ct.insert(
        writer_ct.end(),
        {static_cast<uint32_t>(scaled_kv), k_dim, kv_elem_bytes, cb_k_rm, cb_idx, cb_kreq, cb_kack, packed_page_bytes});
    TT_FATAL(
        writer_ct.size() == ::sparse_sdpa::writer_ct_arg::END,
        "sparse_sdpa writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);

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
    compute_ct.insert(
        compute_ct.end(),
        {static_cast<uint32_t>(scaled_kv), cb_k_rope_rm, cb_k_scale_bcast, cb_k_latent_tile, cb_k_rope_tile});

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/sparse_sdpa_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;  // kv runtime tensor-shape metadata (same on every core)
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/sparse_sdpa_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;  // kv runtime tensor-shape metadata (same on every core)
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // Order matches get_compute_kernel_config_args: (fidelity, approx_mode, fp32_dest_acc, packer_l1_acc,
    // dst_full_sync). packer_l1_acc has no ComputeConfigDescriptor field for this op (no L1 packer accum), so
    // it is unused here.
    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;

    // Query sub-blocking: the DST-bound primitives (matmuls, reduce, sub_exp, salad) hold `qsb` query
    // tile-rows live in DEST, so qsb must be <= dst_size (8 without fp32 acc, 4 with, half-sync). Process
    // Sqt rows in q_groups = Sqt/qsb passes. Pick the largest divisor of Sqt that fits so groups are
    // equal-height (salad's sbh is a compile-time template param). qsb==Sqt (one group) for small H.
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    uint32_t qsb = 1;
    for (uint32_t d = std::min(Sqt, dst_size); d >= 1; --d) {
        if (Sqt % d == 0) {
            qsb = d;
            break;
        }
    }
    compute_ct.push_back(static_cast<uint32_t>(math_approx));
    compute_ct.push_back(qsb);
    compute_ct.push_back(packed_page_bytes);
    TT_FATAL(
        compute_ct.size() == ::sparse_sdpa::compute_ct_arg::END,
        "sparse_sdpa compute compile-time argument layout is out of sync");

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    // fp8 inputs must unpack into a 32-bit dest (fp8 -> fp32 in DEST, then packed to bfp8 cb_*_in).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (kv_storage_is_fp8) {
        unpack_to_dest_mode[cb_k_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (q_is_fp8) {
        unpack_to_dest_mode[cb_q_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx};
    auto* q_buf = t.q.buffer();
    auto* kv_buf = t.kv.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* out_buf = output.buffer();
    // Indexed KV cache: the gather page ids are offset by cache_batch_idx * T to select the cache's batch
    // slot. Baked here for the cache-miss build; re-applied on every dispatch by get_dynamic_runtime_args
    // (so changing the slot doesn't recompile). 0 when not indexed (kv is a single [1,1,T,K_DIM] cache).
    const uint32_t kv_T = t.kv.logical_shape()[2];
    const uint32_t kv_batch_page_offset = attrs.cache_batch_idx.value_or(0) * kv_T;
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t tok_start = i * base + std::min(i, extra);
        uint32_t tok_count = base + (i < extra ? 1u : 0u);
        // kv_batch_page_offset sits at a fixed index (sparse_sdpa_rt::k{Reader,Writer}BatchOffsetArg), re-applied
        // on a cache hit by get_dynamic_runtime_args (the slot changes per dispatch). If you reorder the args
        // before it, update those constants or the re-apply targets the wrong slot.
        reader_desc.emplace_runtime_args(core, {q_buf, kv_buf, idx_buf, tok_start, tok_count, kv_batch_page_offset});
        writer_desc.emplace_runtime_args(core, {out_buf, tok_start, tok_count, kv_buf, kv_batch_page_offset});
        compute_desc.emplace_runtime_args(core, {tok_start, tok_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
