// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the experimental gated-delta prefill-then-query op.
//
// This is the multi-core skeleton of the real op. The recurrence itself is NOT implemented
// yet; this step establishes the multi-core work distribution and a correct K read path:
//
//   * Work is distributed one V-head per core. Each K-head is replicated across its
//     gva_ratio (= Nv/Nk) V-heads, so a core for v_head reads K-head (v_head / gva_ratio).
//   * All available cores are used: the Nv V-heads are spread balanced-greedily over the
//     whole compute grid, and any cores beyond Nv split a V-head's sequence into more
//     sections (split is always along sequence, never the hidden dim). Cores sharing a
//     V-head form the group that a later step will tree-reduce.
//   * The reader streams its core's whole K section into cb_k (hidden-major, one tile at a time),
//     then its V section into cb_v the same way — both kept resident for later stages.
//   * The compute kernel does the first real math: K @ K^T, one gram_block x gram_block output
//     block at a time. Per block it reads gram_block x 1 chunks and accumulates the partial
//     product over the hidden dim (kt = 1 per step), then packs the block into cb_kkt. The rest
//     of the recurrence and the outputs (O, state') come next — O/state' NOT yet written.

#include "gated_delta_prefill_query_device_operation.hpp"

#include <algorithm>

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

    // One V-head's recurrence lands on at least one core, so we need at least Nv cores.
    TT_FATAL(
        num_cores_avail >= Nv,
        "gated_delta_prefill_query needs at least num_v_heads ({}) cores; the compute grid has {}",
        Nv,
        num_cores_avail);

    // ---- Prefill K-buffer sizing hyperparameter ----
    // out_block_size is the cb_k capacity, in TILES — how many K tiles can be buffered before
    // the reader blocks on the compute consumer. It is seeded from num_out_blocks (a rough
    // target of how many chunks to split the sequence into) and then rounded DOWN to a multiple
    // of d_tiles so the buffer holds a whole number of full-hidden-dim seq-rows (the reader
    // pushes one d_tiles-wide row at a time). out_block_size / d_tiles is the buffering depth in
    // seq-rows.
    constexpr uint32_t num_out_blocks = 1;
    const uint32_t target_block_tiles = (seq_tiles * d_tiles + num_out_blocks - 1) / num_out_blocks;
    uint32_t out_block_size = (target_block_tiles / d_tiles) * d_tiles;  // round down to hidden-dim alignment
    if (out_block_size == 0) {
        out_block_size = d_tiles;  // never smaller than one full hidden-dim row
    }
    TT_FATAL(
        out_block_size % d_tiles == 0,
        "out_block_size ({}) must be a multiple of d_tiles ({})",
        out_block_size,
        d_tiles);

    // ---- Core work distribution (maximize utilization) ----
    // Spread all available cores over the Nv V-heads balanced-greedily: each V-head gets
    // base (= C/Nv) sections, the first (C % Nv) V-heads get one extra, so no core is idle.
    // A V-head's seq_tiles are split into contiguous, balanced seq ranges across its sections.
    // Sections are capped at seq_tiles (a section needs >= 1 seq-tile); only cores beyond
    // Nv * seq_tiles stay unused (short sequences only).
    struct CoreWork {
        uint32_t v_head;
        uint32_t k_head;
        uint32_t section_id;
        uint32_t num_sections;
        uint32_t seq_tile_start;
        uint32_t seq_tile_count;
    };
    std::vector<CoreWork> work;
    work.reserve(num_cores_avail);

    uint32_t max_seq_tile_count = 0;  // longest section, for cb_kkt sizing
    const uint32_t base_sections = num_cores_avail / Nv;
    const uint32_t extra_sections = num_cores_avail % Nv;
    for (uint32_t v = 0; v < Nv; ++v) {
        uint32_t sections = base_sections + (v < extra_sections ? 1u : 0u);
        sections = std::min(sections, seq_tiles);  // cannot have more sections than seq-tiles
        if (sections == 0) {
            sections = 1;
        }
        const uint32_t seq_base = seq_tiles / sections;
        const uint32_t seq_rem = seq_tiles % sections;
        uint32_t start = 0;
        for (uint32_t s = 0; s < sections; ++s) {
            const uint32_t count = seq_base + (s < seq_rem ? 1u : 0u);
            work.push_back({v, v / gva_ratio, s, sections, start, count});
            start += count;
            max_seq_tile_count = std::max(max_seq_tile_count, count);
        }
    }
    const uint32_t num_cores = static_cast<uint32_t>(work.size());
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

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

    // ---- Reader: streams this core's whole K section into cb_k (hidden-major, one tile at a
    //      time, each hidden column's seq-tiles contiguous so a gram_block x 1 chunk is
    //      contiguous), then its V section into cb_v the same way. CT args:
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
    //      [d_tiles, gram_block]. ----
    KernelDescriptor compute_kernel;
    compute_kernel.kernel_source = kdir + "compute/gated_delta_prefill_query.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = all_cores;
    compute_kernel.compile_time_args = {d_tiles, gram_block};
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = false,  // bf16 DST so a full 4x4 = 16-tile output block fits
        .dst_full_sync_en = true,   // use all 16 DST tiles for the output block
        .math_approx_mode = math_approx_mode};

    // ---- Per-core runtime args ----
    reader_kernel.runtime_args.reserve(num_cores);
    compute_kernel.runtime_args.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i % grid.x, i / grid.x};
        const CoreWork& w = work[i];
        // Reader: k_addr, k_head_id, seq_tile_start, seq_tile_count, v_head_id, section_id,
        // num_sections, v_addr. (section_id/num_sections reserved for the future tree reduction.)
        reader_kernel.emplace_runtime_args(
            core,
            {k.buffer(),
             w.k_head,
             w.seq_tile_start,
             w.seq_tile_count,
             w.v_head,
             w.section_id,
             w.num_sections,
             v.buffer()});
        // Compute: number of seq-tiles this core owns (= result strip length).
        compute_kernel.emplace_runtime_args(core, {w.seq_tile_count});
    }

    program.kernels.push_back(std::move(reader_kernel));
    program.kernels.push_back(std::move(writer_kernel));
    program.kernels.push_back(std::move(compute_kernel));

    // ---- cb_k / cb_v: K and V inputs, each sized to one hidden-dim-aligned block. ----
    program.cbs.push_back(CBDescriptor{
        .total_size = out_block_size * k_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_k, .data_format = k_df, .page_size = k_tile_bytes}}}});
    program.cbs.push_back(CBDescriptor{
        .total_size = out_block_size * k_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_v, .data_format = k_df, .page_size = k_tile_bytes}}}});

    // ---- cb_kkt: K @ K^T result, in gram_block x gram_block output blocks. Sized to hold the
    //      whole per-core result since there is no consumer yet: ceil(H/gram_block) blocks of up
    //      to gram_block^2 tiles (H = longest section). ----
    const uint32_t gram_blocks = (max_seq_tile_count + gram_block - 1) / gram_block;
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
