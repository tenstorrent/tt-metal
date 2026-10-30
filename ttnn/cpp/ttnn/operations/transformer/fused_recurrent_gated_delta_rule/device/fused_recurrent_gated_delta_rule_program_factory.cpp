// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for fused_recurrent_gated_delta_rule.
// Parallelism: one Tensix core per (B*HV) head; each core walks the T token axis sequentially,
// holding the recurrent state S [K,V] on-core. All math is in the compute kernel (the two-pass
// recurrence of kernels/compute/fused_recurrent_gdn_math.hpp), derived from flash-linear-attention
// `naive_recurrent_gated_delta_rule`.
//
// The circular-buffer plan (ids, tile dims, which CBs unpack straight into fp32 DEST) is the shared
// header kernels/fused_recurrent_gdn_cb.hpp; the reader builds the tiny 16x32 / 32x16 token tiles
// from the host-prepared [1, D] tiles, the writer writes the 2 KB o tiles into 4 KB output pages.

#include "fused_recurrent_gated_delta_rule_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "kernels/fused_recurrent_gdn_cb.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

namespace cb = gdn_decode_cb;

tt::tt_metal::ProgramDescriptor FusedRecurrentGatedDeltaRuleProgramFactory::create_descriptor(
    const FusedRecurrentGatedDeltaRuleParams& attrs,
    const FusedRecurrentGatedDeltaRuleInputs& in,
    std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t T = attrs.T;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t per_token = attrs.output_per_token_state ? 1u : 0u;
    const bool ring = in.initial_state_block_idx.has_value();
    const uint32_t use_blk_idx = ring ? 1u : 0u;
    Buffer* blk_buf = ring ? in.initial_state_block_idx->buffer() : nullptr;
    // The [BH] index vector is one ROW_MAJOR page; the aligned page size keeps the DRAM read aligned.
    const uint32_t blk_page_bytes = ring ? blk_buf->aligned_page_size() : 0u;

    const uint32_t kv = Kt * Vt;

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    TT_FATAL(BH <= grid.x * grid_y, "num_heads {} exceeds compute cores {}", BH, grid.x * grid_y);

    std::vector<CoreCoord> head_cores(BH);
    std::set<CoreRange> core_set;
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
        core_set.insert(CoreRange{head_cores[h], head_cores[h]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    const tt::DataFormat fmt = tt::DataFormat::Float32;
    // One fp32 tile CB: `n_tiles * nbuf` tiles of the dims the shared plan assigns to the id. The tile dims are
    // declared on the CB so matmul_tiles / pack_tile / get_tile_size read them from there.
    auto fmt_desc = [&](uint32_t idx) {
        const auto dims = cb::tile_dims[idx];
        return CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(idx),
            .data_format = fmt,
            .page_size = cb::tile_bytes(dims),
            .tile = TileDescriptor{dims[0], dims[1], /*transpose=*/false}};
    };
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * cb::tile_bytes(cb::tile_dims[idx]),
            .core_ranges = cores,
            .format_descriptors = {{fmt_desc(idx)}}});
    };
    // A state CB: two buffer indices over one allocation, `idx_utd` (UnpackToDestFp32, read by copy_tile) and `idx_mm`
    // (default unpack, the read-pass matmul in1); see fused_recurrent_gdn_cb.hpp. Producers and consumers move both
    // ids in lockstep, so the two read/write pointers always address the same tiles.
    auto add_state_cb = [&](uint32_t idx_utd, uint32_t idx_mm, uint32_t n_tiles) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * cb::tile_bytes(cb::tile_dims[idx_utd]),
            .core_ranges = cores,
            .format_descriptors = {{fmt_desc(idx_utd), fmt_desc(idx_mm)}}});
    };

    // reader -> compute (the per-token tiles hold token_cb_nbuf tokens so the reader runs a token ahead)
    add_cb(cb::qk16, Kt, cb::token_cb_nbuf);
    add_cb(cb::kT16, Kt, cb::token_cb_nbuf);
    add_cb(cb::qcol32, Kt, cb::token_cb_nbuf);
    add_cb(cb::v16, Vt, cb::token_cb_nbuf);
    add_cb(cb::gcol32, 1, cb::token_cb_nbuf);
    add_cb(cb::bcol16, 1, cb::token_cb_nbuf);
    add_state_cb(cb::S, cb::S_mm, kv);
    add_cb(cb::ones32, 1);
    add_cb(cb::zero32, 1);
    // compute -> compute (c_10 r16 and c_12 ccol32 are not allocated: the fused read+vector pass keeps r in DEST
    // and c goes from DEST straight into ebc16)
    add_cb(cb::ecol32, 1);
    add_cb(cb::craw16, 1);
    add_cb(cb::ebc16, 1);
    add_cb(cb::u16, Vt);
    add_state_cb(cb::s2, cb::s2_mm, kv);
    add_state_cb(cb::s3, cb::s3_mm, kv);
    // compute -> writer
    add_cb(cb::out16, Vt, 2);
    add_cb(cb::state_out, kv, 2);
    if (ring) {
        const tt::DataFormat idx_fmt = datatype_to_dataformat_converter(in.initial_state_block_idx->dtype());
        desc.cbs.push_back(CBDescriptor{
            .total_size = blk_page_bytes,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb::blkidx),
                .data_format = idx_fmt,
                .page_size = blk_page_bytes}}}});
    }

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> compute_ct = {Kt, Vt, per_token};

    std::vector<uint32_t> reader_ct = {Kt, Vt, use_blk_idx};
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);  // the log decay; the kernel takes exp(g)
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);
    TensorAccessorArgs(blk_buf).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt, per_token};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_fused_recurrent_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_fused_recurrent_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/fused_recurrent_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    // Honor the caller's fidelity / fp32 DEST / sync settings (the public wrapper defaults to HiFi4, fp32 acc).
    // math_approx_mode is pinned to false: the kernel computes exp(g) on the SFPU and the recurrence's bits are a
    // compatibility contract (the identity and digest tests), so the approximate exp is never used.
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;
    // Every CB that is only ever read by copy_tile into DEST unpacks straight to fp32 DEST (an exact copy; the
    // default route truncates fp32 to tf32 through SrcA).
    std::vector<UnpackToDestMode> utd(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t i = 0; i < cb::num_cbs; i++) {
        if (cb::unpack_to_dest_fp32[i]) {
            utd[i] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = std::move(utd),
        .math_approx_mode = false};
    compute.runtime_args.reserve(BH);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* st_buf = outputs[1].buffer();

    for (uint32_t h = 0; h < BH; h++) {
        const auto& core = head_cores[h];
        reader.emplace_runtime_args(
            core, {h, T, q_buf, k_buf, v_buf, g_buf, beta_buf, s0_buf, blk_buf, blk_page_bytes, BH});
        // BH is needed by the writer to place per-token state token-major (page t*BH + h).
        writer.emplace_runtime_args(core, {h, T, o_buf, st_buf, BH});
        compute.emplace_runtime_args(core, {T});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
