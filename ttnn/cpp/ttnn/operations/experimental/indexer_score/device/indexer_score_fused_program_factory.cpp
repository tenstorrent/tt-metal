// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_fused_program_factory.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>    // tt::tile_size
#include <tt-metalium/work_split.hpp>  // corerange_to_cores
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "hostdevcommon/kernel_structs.h"  // tt::CBIndex

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"        // RingSDPAFusedOpSignaler
#include "ttnn/operations/transformer/sdpa/device/ring_id_sequencer.hpp"  // host replay for band arrival order
#include "ttnn/operations/ccl/ccl_common.hpp"     // linearized index / neighbor / fwd-bwd config
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"  // AllGatherFusedOpSignaler
// the fused AG helper (the only Linear+fuse-capable all-gather):
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

#include "kernels/indexer_score_cb.hpp"
#include "kernels/indexer_score_work_split.hpp"

namespace ttnn::operations::experimental::indexer_score::program {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

namespace {

// --- duplicated verbatim from the classic factory (device-index + per-device causal geometry) so the fused
// factory reuses the exact same chunk_start derivation without touching the working classic .cpp. -----------
struct DeviceCausalGeometry {
    uint32_t chunk_start_tiles;
    uint32_t straddle_q_tile;
    uint32_t straddle_jump_tiles;
};
DeviceCausalGeometry device_causal_geometry(const operation_attributes_t& args, uint32_t device_index, uint32_t Sq) {
    const uint32_t TW = tt::constants::TILE_WIDTH;
    if (!args.block_cyclic.has_value()) {
        return {(args.chunk_start_idx + device_index * Sq) / TW, 0u, 0u};
    }
    const uint32_t sp = args.block_cyclic->sp;
    const uint32_t chunk_local = args.block_cyclic->chunk_local;
    const uint32_t chunk_global = sp * chunk_local;
    if (args.cluster_axis.has_value()) {
        TT_FATAL(device_index < sp, "indexer_score fused: device_index {} out of range for sp={}", device_index, sp);
        const uint32_t boundary_slab = args.chunk_start_idx / chunk_global;
        const uint32_t boundary_chip = (args.chunk_start_idx / chunk_local) % sp;
        const uint32_t offset = args.chunk_start_idx % chunk_local;
        const uint32_t update_idxt = device_index < boundary_chip    ? (boundary_slab + 1) * chunk_local
                                     : device_index == boundary_chip ? boundary_slab * chunk_local + offset
                                                                     : boundary_slab * chunk_local;
        const uint32_t logical_start =
            (update_idxt / chunk_local) * chunk_global + device_index * chunk_local + (update_idxt % chunk_local);
        uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
        if (device_index == boundary_chip && offset != 0 && offset + Sq > chunk_local) {
            straddle_q_tile = (chunk_local - offset) / TW;
            straddle_jump_tiles = (chunk_global - chunk_local) / TW;
        }
        return {logical_start / TW, straddle_q_tile, straddle_jump_tiles};
    }
    const uint32_t chunk_start = args.chunk_start_idx + device_index * Sq;
    const uint32_t offset = chunk_start % chunk_local;
    uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
    if (offset != 0 && offset + Sq > chunk_local) {
        straddle_q_tile = (chunk_local - offset) / TW;
        straddle_jump_tiles = (chunk_global - chunk_local) / TW;
    }
    return {chunk_start / TW, straddle_q_tile, straddle_jump_tiles};
}

uint32_t device_index_for(const operation_attributes_t& args, const ttnn::MeshCoordinate& coord, const Tensor& q) {
    if (q.device_storage().get_coords().size() <= 1) {
        return 0;
    }
    return ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.cluster_axis);
}

// forward/backward all-gather writes expected for this device on the given topology (mirrors ring_joint's
// build_ring_write_plan: Linear swaps num_targets_{fwd,bwd} into the plan).
struct RingWrites {
    uint32_t forward_writes_expected;
    uint32_t backward_writes_expected;
};
RingWrites ring_writes_for(uint32_t ring_size, uint32_t ring_index, ttnn::ccl::Topology topology) {
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    (void)dynamic_alternate;
    if (topology == ttnn::ccl::Topology::Ring && (ring_index % 2 == 0)) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    if (topology == ttnn::ccl::Topology::Linear) {
        return {static_cast<uint32_t>(num_targets_backward), static_cast<uint32_t>(num_targets_forward)};
    }
    return {static_cast<uint32_t>(num_targets_forward), static_cast<uint32_t>(num_targets_backward)};
}

// One device's fused program: indexer compute (banded schedule, DSA path) + co-scheduled ring_attention AG.
ProgramDescriptor build_fused_program_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    const Tensor& out,
    const ttnn::MeshCoordinate& coord) {
    ProgramDescriptor desc;

    const auto& q = tensors.q;
    const auto& k = tensors.k;  // the gathered [B,1,T,D] persistent AG output buffer (read side)
    const auto& w = tensors.weights;
    const auto& fused = *args.fused_ring;
    TT_FATAL(tensors.k_local.has_value(), "indexer_score fused: k_local (all-gather input) is required");
    const auto& k_local = *tensors.k_local;

    // Fused path is DSA-only: relu, single head-summed plane, no block-pool, learned weights (no synth gate).
    TT_FATAL(args.apply_relu, "indexer_score fused: DSA path only (apply_relu=true)");
    TT_FATAL(args.num_groups == 1, "indexer_score fused: num_groups must be 1 (DSA)");
    TT_FATAL(args.block_size == 0, "indexer_score fused: block-max-pool not supported on the fused path");
    TT_FATAL(!args.synthesize_gate, "indexer_score fused: synthesize_gate (MSA) not supported on the fused path");
    // Step-E band reorder assumes no head streaming (all heads resident): stream_heads pads the band loop with
    // phantom q-mcast bands that the reorder would perturb. The deployed DSA config uses head_group_size 0/Hi.
    TT_FATAL(
        resolve_head_group(args.program_config, q.logical_shape()[1]) == q.logical_shape()[1],
        "indexer_score fused: head_group_size must be 0 or Hi (no head streaming) on the fused path");

    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    const uint32_t device_index = device_index_for(args, coord, q);
    const auto geom = device_causal_geometry(args, device_index, Sq);
    const uint32_t chunk_t = geom.chunk_start_tiles;

    const uint32_t Sqt = Sq / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = T / tt::constants::TILE_WIDTH;
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;

    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    const uint32_t G = 1;
    const uint32_t subblock_basis = HB;

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(q.device()->arch(), args.compute_kernel_config);
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_subblock_h =
        ttnn::prim::detail::determine_largest_subblock_size(subblock_basis, 1, dst_size).first;
    TT_FATAL(
        subblock_basis % qk_subblock_h == 0,
        "head count {} not divisible by qk_subblock_h {}",
        subblock_basis,
        qk_subblock_h);

    // --- banded-product schedule (DSA: group_count = Sqt/QC), on a grid REDUCED to leave AG worker rows. -----
    const auto phys_grid = q.device()->compute_with_storage_grid_size();
    const uint32_t grid_x = phys_grid.x;
    // Reserve enough top rows for the AG workers (num_links * 2 senders per link), so the compute rectangle
    // and the AG worker cores are disjoint (ccl_core_grid_offset points at the first reserved row).
    const uint32_t ag_worker_cores = std::max<uint32_t>(1u, fused.num_links * 2u);
    const uint32_t reserved_rows = std::max<uint32_t>(1u, (ag_worker_cores + grid_x - 1) / grid_x);
    TT_FATAL(phys_grid.y > reserved_rows, "indexer_score fused: grid too small to reserve {} AG rows", reserved_rows);
    const uint32_t grid_y = phys_grid.y - reserved_rows;  // effective compute rows
    const CoreCoord ccl_core_grid_offset{0, grid_y};      // AG workers start on the first reserved row

    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);
    const uint32_t group_rows = rows_for_groups(group_count, grid_y);
    const uint32_t cols_used = cols_for_bands(band_count, grid_x);
    const uint32_t num_blocks = band_row_blocks(group_count, band_count, grid_x, grid_y);
    const uint32_t rows_used = group_rows * num_blocks;
    const uint32_t num_groups = group_count / group_rows;

    std::vector<std::vector<uint32_t>> band_start(num_blocks, std::vector<uint32_t>(cols_used));
    std::vector<std::vector<uint32_t>> band_size(num_blocks, std::vector<uint32_t>(cols_used));
    {
        const uint32_t bands_per_block = band_count / num_blocks, blk_extra = band_count % num_blocks;
        uint32_t blk_off = 0;
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t blk_bands = bands_per_block + (blk < blk_extra ? 1u : 0u);
            const uint32_t bands_per_col = blk_bands / cols_used, extra = blk_bands % cols_used;
            uint32_t off = blk_off;
            for (uint32_t col = 0; col < cols_used; ++col) {
                band_size[blk][col] = bands_per_col + (col < extra ? 1u : 0u);
                band_start[blk][col] = off;
                off += band_size[blk][col];
            }
            blk_off += blk_bands;
        }
    }
    const uint32_t bands_in_widest_block = (band_count + num_blocks - 1) / num_blocks;
    const uint32_t max_bands = (bands_in_widest_block + cols_used - 1) / cols_used;

    const CoreRange core_rect(CoreCoord{0, 0}, CoreCoord{cols_used - 1, rows_used - 1});
    const CoreRangeSet core_ranges(core_rect);

    std::vector<std::vector<CoreCoord>> phys(rows_used, std::vector<CoreCoord>(cols_used));
    for (uint32_t row = 0; row < rows_used; ++row) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            phys[row][col] = q.device()->worker_core_from_logical_core(CoreCoord{col, row});
        }
    }

    const uint32_t k_mcast_on = (group_rows > 1) ? 1u : 0u;
    const uint32_t q_mcast_on = (cols_used > 1) ? 1u : 0u;

    // Semaphores: mcast handshake (3 per active direction), pushed as descriptors with sequential ids.
    auto push_sem = [&](uint32_t init) -> uint32_t {
        const uint32_t id = static_cast<uint32_t>(desc.semaphores.size());
        // core_type defaults to CoreType::WORKER in SemaphoreDescriptor.
        desc.semaphores.push_back(SemaphoreDescriptor{.id = id, .core_ranges = core_ranges, .initial_value = init});
        return id;
    };
    const uint32_t k_send_sem = k_mcast_on ? push_sem(0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? push_sem(0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? push_sem(1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? push_sem(0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? push_sem(0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? push_sem(1) : 0;

    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);
    const bool q_is_bfp8 = q.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const bool k_is_bfp8 = k.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const tt::DataFormat q_fmt = q_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const tt::DataFormat k_fmt = k_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const uint32_t q_tile = tt::tile_size(q_fmt);
    const uint32_t k_tile = tt::tile_size(k_fmt);
    const tt::DataFormat acc_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t acc_tile = fp32_dest_acc_en ? fp32_tile : bf16_tile;
    const bool stream_heads = HB < Hi;

    // CB allocation (DSA: no fuse_single, no block-pool). Slot order == CbArg; index continuous from c_0.
    std::array<uint32_t, num_cb_args> cb_id{};
    uint32_t next_cb_index = tt::CBIndex::c_0;
    auto make_cb = [&](uint32_t slot, uint32_t ntiles, tt::DataFormat fmt, uint32_t tile_bytes) {
        const uint32_t idx = next_cb_index++;
        cb_id[slot] = idx;
        desc.cbs.push_back(CBDescriptor{
            .total_size = ntiles * tile_bytes,
            .core_ranges = core_ranges,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = tile_bytes}}}});
    };
    const uint32_t qk_batch_cap = (QC == 1) ? subblock_basis : tt::constants::TILE_HEIGHT;
    const uint32_t qk_batch_heads = std::min<uint32_t>(subblock_basis, qk_batch_cap);
    TT_FATAL(
        subblock_basis % qk_batch_heads == 0,
        "head count {} not divisible by qk_batch_heads {}",
        subblock_basis,
        qk_batch_heads);
    const bool single_chunk = (qk_batch_heads == subblock_basis) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk) ? KC : 1u;

    make_cb(cb_q_arg, (stream_heads ? 2 : 1) * HB * QC * Dt, q_fmt, q_tile);
    make_cb(cb_k_arg, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w_arg, Hi * QC, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask_arg, num_mask_tiles, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_qk_arg, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    make_cb(cb_out_strip_arg, 2 * KC, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_acc_strip_arg, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // Fused-op signal semaphores + consumer signaler (inlined init_fused_op against desc; MULTI).
    const uint32_t ring_size = args.cluster_axis.has_value()
                                   ? static_cast<uint32_t>(q.device()->get_view().shape()[*args.cluster_axis])
                                   : static_cast<uint32_t>(q.device()->get_view().shape().mesh_size());
    const auto rw = ring_writes_for(ring_size, device_index, fused.topology);

    ttnn::prim::RingSDPAFusedOpSignaler sdpa_sig;
    sdpa_sig.init_all_gather(ring_size, device_index, rw.forward_writes_expected, rw.backward_writes_expected);
    sdpa_sig.fused_op_signaler_mode = ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;
    sdpa_sig.fused_op_receiver_cores_noc.clear();
    for (const auto& core : corerange_to_cores(core_ranges, std::nullopt, /*row_wise=*/true)) {
        sdpa_sig.fused_op_receiver_cores_noc.push_back(q.device()->worker_core_from_logical_core(core));
    }
    const uint32_t fused_sem0 = push_sem(0);
    const uint32_t fused_sem1 = push_sem(0);
    sdpa_sig.fused_op_receiver_signal_semaphores = {fused_sem0, fused_sem1};
    sdpa_sig.num_fused_op_cores_to_signal = static_cast<uint32_t>(sdpa_sig.fused_op_receiver_cores_noc.size());
    sdpa_sig.initialized_fused_op = true;

    // Compile-time args (common dims + CB indices).
    std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, QC, KC, HB, G, /*block_tiles=*/0u};
    common_ct.insert(common_ct.end(), cb_id.begin(), cb_id.end());

    std::vector<uint32_t> reader_ct = common_ct;
    tt::tt_metal::TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*w.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k_local.buffer()).append_to(reader_ct);  // fused: local SP shard (AG input)
    reader_ct.push_back(k_mcast_on);
    reader_ct.push_back(q_mcast_on);
    reader_ct.push_back(k_send_sem);
    reader_ct.push_back(k_recv_sem);
    reader_ct.push_back(k_valid_sem);
    reader_ct.push_back(q_send_sem);
    reader_ct.push_back(q_recv_sem);
    reader_ct.push_back(q_valid_sem);
    reader_ct.push_back(0u);  // fuse_single off (DSA)
    reader_ct.push_back(0u);  // fused_stream_k off
    reader_ct.push_back(0u);  // synthesize_gate off (reads real weights)
    reader_ct.push_back(0u);  // gate_scale_bits (unused)

    KernelDescriptor::Defines reader_defines;
    reader_defines.emplace_back("FUSED_RING", "1");  // enables the reader's coarse all-gather barrier
    if (args.has_block_cyclic()) {
        const uint32_t sp = args.block_cyclic->sp;
        const uint32_t cl_t = args.block_cyclic->chunk_local / tt::constants::TILE_WIDTH;
        const uint32_t chunk_tiles = sp * cl_t;
        reader_defines.emplace_back("BC_ENABLE", "1");
        reader_defines.emplace_back("BC_SP", std::to_string(sp));
        reader_defines.emplace_back("BC_CHUNK_LOCAL_T", std::to_string(cl_t));
        reader_defines.emplace_back("BC_SLAB_STRIDE_GAP", std::to_string(cl_t * (sp - 1)));
        reader_defines.emplace_back("BC_SHARD_STRIDE_GAP", std::to_string((Tt - chunk_tiles) / sp));
    }

    std::vector<uint32_t> writer_ct = common_ct;
    const uint32_t out_elem_bytes = out.element_size();
    writer_ct.push_back(T * out_elem_bytes);  // row-major page = one output row (no pooling)
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);
    compute_ct.push_back(qk_col_batch);
    compute_ct.push_back(1u);  // apply_relu (DSA)
    compute_ct.push_back(0u);  // fuse_single off
    compute_ct.push_back(0u);  // fused_stream_k off

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source = kdir + "reader_indexer_score.cpp";
    reader_kernel.core_ranges = core_ranges;
    reader_kernel.compile_time_args = reader_ct;
    reader_kernel.defines = reader_defines;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source = kdir + "writer_indexer_score.cpp";
    writer_kernel.core_ranges = core_ranges;
    writer_kernel.compile_time_args = writer_ct;
    writer_kernel.defines = {{"FUSED_RING", "1"}};  // read the reordered band-visit perm from rt args
    writer_kernel.config = WriterConfigDescriptor{};

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source = kdir + "compute_indexer_score.cpp";
    compute_kernel.core_ranges = core_ranges;
    compute_kernel.compile_time_args = compute_ct;
    compute_kernel.defines = {{"FUSED_RING", "1"}};  // read the reordered band-visit perm from rt args
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};

    const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
    const uint32_t k_batch_page_offset = args.cache_batch_idx.value_or(0) * Tt * Dt;
    const uint32_t kv_len_tiles = args.kv_len.value_or(T) / tt::constants::TILE_WIDTH;

    std::vector<uint32_t> fused_rt;
    sdpa_sig.push_ring_sdpa_fused_op_rt_args(fused_rt);  // {ring_size, ring_index, fwd, bwd, sem0, sem1}

    // ---- Step E: band-visit reorder (local-first, then remote by ring arrival) --------------------------
    // Replay RingIdSequencer on the HOST (same seed as the reader) so shard_order[c] = the ring iteration that
    // delivers SP shard c (local shard -> 0). A band's readiness = max arrival-iter over the shards its tiles
    // land in; sorting a core's bands by readiness makes it score its local + already-arrived bands first and
    // hide the farther slabs' transport behind that compute. The SAME permutation is fed to reader/compute/
    // writer (band identity preserved) so the cb_k / cb_out FIFOs stay in lockstep.
    std::vector<uint32_t> shard_order(ring_size, 0);
    {
        RingIdSequencer seq(device_index, ring_size, rw.backward_writes_expected, rw.forward_writes_expected);
        for (uint32_t i = 0; i < ring_size; ++i) {
            const uint32_t rid = seq.get_next_ring_id([](uint32_t, uint32_t) {});
            shard_order[rid] = i;
        }
    }
    const uint32_t sll_t = Tt / ring_size;  // tiles per SP shard in the gathered buffer
    const uint32_t cl_t = args.has_block_cyclic() ? args.block_cyclic->chunk_local / tt::constants::TILE_WIDTH : 0;
    const auto band_readiness = [&](uint32_t band_abs) -> uint32_t {
        const uint32_t start = band_abs * KC;
        const uint32_t end = std::min(start + KC, Tt);
        uint32_t r = 0;
        for (uint32_t L = start; L < end; ++L) {
            const uint32_t shard = args.has_block_cyclic() ? (L / cl_t) % ring_size : (L / sll_t);
            r = std::max(r, shard_order[shard]);
        }
        return r;
    };

    for (uint32_t row = 0; row < rows_used; ++row) {
        uint32_t q_xs = u32(phys[row][0].x), q_xe = u32(phys[row][0].x);
        for (uint32_t bbox_col = 0; bbox_col < cols_used; ++bbox_col) {
            q_xs = std::min<uint32_t>(q_xs, u32(phys[row][bbox_col].x));
            q_xe = std::max<uint32_t>(q_xe, u32(phys[row][bbox_col].x));
        }
        const uint32_t q_py = u32(phys[row][0].y);
        const uint32_t q_diag = std::min<uint32_t>(row, cols_used - 1);
        const CoreCoord q_sender = phys[row][q_diag];
        const uint32_t block = row / group_rows;
        const uint32_t block_base = block * group_rows;
        for (uint32_t col = 0; col < cols_used; ++col) {
            uint32_t k_ys = u32(phys[block_base][col].y), k_ye = u32(phys[block_base][col].y);
            for (uint32_t bbox_row = block_base; bbox_row < block_base + group_rows; ++bbox_row) {
                k_ys = std::min<uint32_t>(k_ys, u32(phys[bbox_row][col].y));
                k_ye = std::max<uint32_t>(k_ye, u32(phys[bbox_row][col].y));
            }
            const uint32_t k_px = u32(phys[block_base][col].x);
            const CoreCoord k_sender = phys[block_base][col];
            const CoreCoord core{col, row};
            const std::array<uint32_t, 6> sched = {
                row % group_rows, group_rows, num_groups, band_start[block][col], band_size[block][col], max_bands};

            // This core's band-visit permutation (offsets into [0, num_bands), sorted by arrival readiness).
            // Identical for every row in a k-mcast column (same band range + ring schedule), so the k-mcast
            // stays in lockstep. Appended to all three kernels below.
            std::vector<uint32_t> band_perm(band_size[block][col]);
            std::iota(band_perm.begin(), band_perm.end(), 0u);
            std::stable_sort(band_perm.begin(), band_perm.end(), [&](uint32_t a, uint32_t b) {
                return band_readiness(band_start[block][col] + a) < band_readiness(band_start[block][col] + b);
            });

            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(q.buffer());
            reader_rt.push_back(k.buffer());
            reader_rt.push_back(w.buffer());
            for (uint32_t s : sched) {
                reader_rt.push_back(s);
            }
            const auto push_mcast_dir = [&](uint32_t role,
                                            uint32_t xs,
                                            uint32_t ys,
                                            uint32_t xe,
                                            uint32_t ye,
                                            const CoreCoord& s,
                                            uint32_t ndst) {
                reader_rt.push_back(role);
                reader_rt.push_back(xs);
                reader_rt.push_back(ys);
                reader_rt.push_back(xe);
                reader_rt.push_back(ye);
                reader_rt.push_back(u32(s.x));
                reader_rt.push_back(u32(s.y));
                reader_rt.push_back(ndst);
            };
            push_mcast_dir(
                k_mcast_on ? (row == block_base ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                k_px,
                k_ys,
                k_px,
                k_ye,
                k_sender,
                group_rows - 1);
            push_mcast_dir(
                q_mcast_on ? (col == q_diag ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                q_xs,
                q_py,
                q_xe,
                q_py,
                q_sender,
                cols_used - 1);
            reader_rt.push_back(k_batch_page_offset);  // slot 25
            reader_rt.push_back(kv_len_tiles);         // slot 26
            reader_rt.append(fused_rt);                // slots 27..32 (ring_size, ring_index, fwd, bwd, sem0, sem1)
            reader_rt.push_back(k_local.buffer());     // slot 33: local SP shard address (dual-source read)
            for (uint32_t off : band_perm) {
                reader_rt.push_back(off);  // slots 34..: band-visit permutation
            }
            reader_kernel.emplace_runtime_args(core, reader_rt);

            KernelDescriptor::RTArgList compute_rt;
            for (uint32_t s : sched) {
                compute_rt.push_back(s);
            }
            compute_rt.push_back(kv_len_tiles);
            compute_rt.push_back(chunk_t);
            compute_rt.push_back(geom.straddle_q_tile);
            compute_rt.push_back(geom.straddle_jump_tiles);
            for (uint32_t off : band_perm) {
                compute_rt.push_back(off);  // slots 10..: band-visit permutation
            }
            compute_kernel.emplace_runtime_args(core, compute_rt);

            KernelDescriptor::RTArgList writer_rt;
            writer_rt.push_back(out.buffer());
            for (uint32_t s : sched) {
                writer_rt.push_back(s);
            }
            writer_rt.push_back(kv_len_tiles);
            writer_rt.push_back(chunk_t);
            writer_rt.push_back(geom.straddle_q_tile);
            writer_rt.push_back(geom.straddle_jump_tiles);
            for (uint32_t off : band_perm) {
                writer_rt.push_back(off);  // slots 11..: band-visit permutation
            }
            writer_kernel.emplace_runtime_args(core, writer_rt);
        }
    }

    // Consumer kernels FIRST (indices 0/1/2), then the AG helper appends its workers (3..).
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    // Producer-side signaler copies the consumer's receiver cores + signal semaphores.
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> ag_sig =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    ag_sig->init_fused_op(
        sdpa_sig.fused_op_receiver_cores_noc,
        sdpa_sig.fused_op_receiver_signal_semaphores,
        sdpa_sig.fused_op_signaler_mode);

    const auto forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(q, coord, /*offset=*/1, fused.topology, args.cluster_axis);
    const auto backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        q, coord, /*offset=*/-1, fused.topology, args.cluster_axis);

    std::vector<Tensor> ag_in = {k_local};
    std::vector<Tensor> ag_out = {k};
    ttnn::ring_attention_all_gather_async_multi_core_with_workers_helper(
        desc,
        ag_in,
        coord,
        forward_coord,
        backward_coord,
        ag_out,
        fused.dim,
        fused.num_links,
        ring_size,
        device_index,
        fused.topology,
        fused.ag_semaphore,
        fused.ag_sub_device_id,
        ag_sig,
        ccl_core_grid_offset,
        ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR);

    log_debug(
        tt::LogOp,
        "indexer_score FUSED coord=({}) ring_size={} ring_index={} fwd_exp={} bwd_exp={} grid={}x{}(+{} ag) "
        "rows_used={} cols_used={} band_count={} k_mcast={} q_mcast={}",
        device_index,
        ring_size,
        device_index,
        rw.forward_writes_expected,
        rw.backward_writes_expected,
        grid_x,
        grid_y,
        reserved_rows,
        rows_used,
        cols_used,
        band_count,
        k_mcast_on,
        q_mcast_on);

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor IndexerScoreFusedProgramFactory::create_workload_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    tensor_return_value_t& out,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_fused_program_descriptor(args, tensors, out, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

IndexerScoreFusedMeshWorkloadFactory::cached_mesh_workload_t IndexerScoreFusedMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensors, out);
}

void IndexerScoreFusedMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    // Buffer addresses (q/k/w/out + the AG's gathered buffer) auto-patch via the descriptor bindings. The
    // per-device chunk_start is baked per-coord at build; a program-cache hit with a NEW chunk_start would be
    // stale (Step B tests dispatch cold -> always a miss -> correct). Scalar re-patch is a later follow-up.
    descriptor_adapter_t::apply_descriptor(cached, args, tensors, out);
}

}  // namespace ttnn::operations::experimental::indexer_score::program
