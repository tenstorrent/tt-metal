// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the experimental gated-delta prefill-then-query op.
//
// This is the multi-core skeleton of the real op. The recurrence itself is NOT implemented
// yet; this step establishes the multi-core work distribution and a correct K read path:
//
//   * Work is distributed exactly one V-head per core: Nv cores, no intra-head splitting.
//     A core owns its V-head's ENTIRE sequence and sweeps it in one pass, so the recurrence
//     is sequential within a core and needs no cross-core reduction. Each K-head is replicated
//     across its gva_ratio (= Nv/Nk) V-heads, so a core for v_head reads K-head
//     (v_head / gva_ratio).
//   * The reader streams its core's whole K head into cb_k (hidden-major, one tile at a time),
//     then its V head into cb_v the same way — both kept resident for later stages.
//   * The compute kernel does the first real math: K @ K^T, one gram_block x gram_block output
//     block at a time. Per block it reads gram_block x 1 chunks and accumulates the partial
//     product over the hidden dim (kt = 1 per step), then packs the block into cb_kkt. The rest
//     of the recurrence and the outputs (O, state') come next — O/state' NOT yet written.

#include "gated_delta_prefill_query_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor GatedDeltaPrefillQueryProgramFactory::create_descriptor(
    const GatedDeltaPrefillQueryParams& attrs,
    const GatedDeltaPrefillQueryInputs& in,
    std::vector<Tensor>& /*outputs*/) {
    const auto& k = in.k;  // [1, Nk, S, d]  TILE  bf16
    const auto& v = in.v;  // [1, Nv, S, d]  TILE  bf16

    const uint32_t Nk = attrs.num_k_heads;
    const uint32_t Nv = attrs.num_v_heads;
    const uint32_t d = attrs.head_dim;
    const uint32_t gva_ratio = Nv / Nk;  // V-heads sharing one (replicated) K-head

    const uint32_t d_tiles = d / TILE_WIDTH;                                               // hidden-dim width, in tiles
    const uint32_t seq_tiles = static_cast<uint32_t>(k.padded_shape()[-2]) / TILE_HEIGHT;  // K/V sequence, in tiles

    IDevice* device = k.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_avail = grid.x * grid.y;

    // One V-head's recurrence lands on exactly one core, so we need at least Nv cores.
    TT_FATAL(
        num_cores_avail >= Nv,
        "gated_delta_prefill_query needs at least num_v_heads ({}) cores; the compute grid has {}",
        Nv,
        num_cores_avail);

    // ---- Core work distribution: one V-head per core, whole sequence, no intra-head split ----
    // Core i owns V-head i for all seq_tiles tokens, and reads K-head i / gva_ratio (blocked GVA,
    // matching repeat_interleave(gva_ratio) in the torch reference). Cores beyond Nv are unused:
    // the recurrence is sequential along the sequence, so splitting a head across cores would
    // require a cross-core scan/reduction — the whole-sweep-per-head shape avoids that entirely.
    const uint32_t num_cores = Nv;
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    // cb_k / cb_v each hold one V-head's whole [S, d] section, resident for the full sweep.
    const uint32_t kv_section_tiles = seq_tiles * d_tiles;

    const tt::DataFormat k_df = datatype_to_dataformat_converter(k.dtype());  // bf16
    const uint32_t k_tile_bytes = tt::tile_size(k_df);

    // K @ K^T is computed one gram_block x gram_block output block at a time: read a
    // gram_block x 1 chunk of K (gram_block seq-tiles at one hidden tile), accumulate its
    // partial gram_block x gram_block product over the hidden dim (kt = 1 per step). gram_block
    // = 4 so the 4x4 = 16-tile output block fits DST (bf16, full-sync). Result is bf16.
    const tt::DataFormat kkt_df = tt::DataFormat::Float16_b;
    const uint32_t kkt_tile_bytes = tt::tile_size(kkt_df);
    constexpr uint32_t gram_block = 4;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);

    constexpr uint8_t cb_k = static_cast<uint8_t>(tt::CBIndex::c_0);      // K section (bf16, resident)
    constexpr uint8_t cb_kkt = static_cast<uint8_t>(tt::CBIndex::c_1);    // unit lower-tri result (bf16)
    constexpr uint8_t cb_mask = static_cast<uint8_t>(tt::CBIndex::c_2);   // strict-lower mask (bf16)
    constexpr uint8_t cb_ident = static_cast<uint8_t>(tt::CBIndex::c_3);  // identity matrix (bf16)
    constexpr uint8_t cb_gram = static_cast<uint8_t>(tt::CBIndex::c_4);   // raw K @ K^T (bf16)
    constexpr uint8_t cb_solve = static_cast<uint8_t>(tt::CBIndex::c_5);  // triangle-solve scaffold out (bf16)
    constexpr uint8_t cb_v = static_cast<uint8_t>(tt::CBIndex::c_6);      // V section (bf16, resident)

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/transformer/gated_delta_prefill_query/device/kernels/";

    ProgramDescriptor program;

    // ---- Reader: streams this core's whole K head into cb_k (hidden-major, one tile at a
    //      time, each hidden column's seq-tiles contiguous so a gram_block x 1 chunk is
    //      contiguous), then its V head into cb_v the same way. CT args:
    //      [d_tiles, seq_tiles, <k TensorAccessorArgs...>, <v TensorAccessorArgs...>]. ----
    std::vector<uint32_t> reader_ct_args{d_tiles, seq_tiles};
    TensorAccessorArgs(k.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(v.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kdir + "dataflow/reader_gated_delta_prefill_query.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_ct_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    // ---- Writer: hand-builds the constant strict-lower mask (cb_mask) and identity (cb_ident)
    //      tiles in L1, then DPRINTs the first unit-lower-triangular result for verification. ----
    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kdir + "dataflow/writer_gated_delta_prefill_query.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.config = WriterConfigDescriptor{};

    // ---- Compute: all matmuls (K @ K^T, gram_block x gram_block blocks, kt=1 accumulated over
    //      the hidden dim) into cb_gram, then masking per tile: (*) strict-lower mask, then
    //      += identity via dest reuse (no pack/unpack round trip) -> unit lower-triangular tiles
    //      in cb_kkt. Doing masking after the matmuls keeps the matmul<->eltwise switch to once
    //      (all bf16, no format reformats). cb_k is NOT popped (kept resident). CT args:
    //      [d_tiles, gram_block, seq_tiles] — the height is compile-time now that a core always
    //      sweeps its head's whole sequence. ----
    KernelDescriptor compute_kernel;
    compute_kernel.kernel_source = kdir + "compute/gated_delta_prefill_query.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = all_cores;
    compute_kernel.compile_time_args = {d_tiles, gram_block, seq_tiles};
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = false,  // bf16 DST so a full 4x4 = 16-tile output block fits
        .dst_full_sync_en = true,   // use all 16 DST tiles for the output block
        .math_approx_mode = math_approx_mode};

    // ---- Per-core runtime args ----
    // Core i <-> V-head i; the seq range is always the whole sequence, so only the head ids are
    // per-core (seq_tiles is a compile-time arg for both kernels).
    reader_kernel.runtime_args.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i % grid.x, i / grid.x};
        // Reader: k_addr, k_head_id, v_head_id, v_addr.
        reader_kernel.emplace_runtime_args(core, {k.buffer(), i / gva_ratio, i, v.buffer()});
    }

    program.kernels.push_back(std::move(reader_kernel));
    program.kernels.push_back(std::move(writer_kernel));
    program.kernels.push_back(std::move(compute_kernel));

    // ---- cb_k / cb_v: this core's whole K and V head sections, resident for the full sweep. ----
    program.cbs.push_back(CBDescriptor{
        .total_size = kv_section_tiles * k_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_k, .data_format = k_df, .page_size = k_tile_bytes}}}});
    program.cbs.push_back(CBDescriptor{
        .total_size = kv_section_tiles * k_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_v, .data_format = k_df, .page_size = k_tile_bytes}}}});

    // ---- cb_kkt: K @ K^T result, in gram_block x gram_block output blocks. Sized to hold the
    //      whole per-core result since there is no consumer yet: ceil(S/gram_block) blocks of up
    //      to gram_block^2 tiles. NOTE: now that a core owns the whole sequence, these scaffold
    //      CBs (cb_kkt/cb_gram/cb_solve) scale with seq_tiles and will overflow L1 for long
    //      prefills — they become chunk-sized/streaming when the recurrence lands. ----
    const uint32_t gram_blocks = (seq_tiles + gram_block - 1) / gram_block;
    const uint32_t kkt_capacity_tiles = gram_blocks * gram_block * gram_block;
    program.cbs.push_back(CBDescriptor{
        .total_size = kkt_capacity_tiles * kkt_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_kkt, .data_format = kkt_df, .page_size = kkt_tile_bytes}}}});

    // ---- cb_gram (raw K@K^T) and cb_solve (triangle-solve scaffold output): per-core capacity. ----
    program.cbs.push_back(CBDescriptor{
        .total_size = kkt_capacity_tiles * kkt_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_gram, .data_format = kkt_df, .page_size = kkt_tile_bytes}}}});
    // cb_solve holds one output tile per (masked tile, hidden tile) step.
    program.cbs.push_back(CBDescriptor{
        .total_size = kkt_capacity_tiles * d_tiles * kkt_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_solve, .data_format = kkt_df, .page_size = kkt_tile_bytes}}}});

    // ---- cb_mask / cb_ident: one constant tile each, hand-built by the writer kernel. ----
    program.cbs.push_back(CBDescriptor{
        .total_size = kkt_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_mask, .data_format = kkt_df, .page_size = kkt_tile_bytes}}}});
    program.cbs.push_back(CBDescriptor{
        .total_size = kkt_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_ident, .data_format = kkt_df, .page_size = kkt_tile_bytes}}}});

    return program;
}

}  // namespace ttnn::experimental::prim
