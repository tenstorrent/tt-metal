// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_indexer_score_dsa_program_factory.hpp"

#include <algorithm>
#include <array>
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
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"  // RingSDPAFusedOpSignaler
#include "ttnn/operations/ccl/ccl_common.hpp"                       // linearized index / neighbor / fwd-bwd config
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"                    // AllGatherFusedOpSignaler
// the fused AG helper (the only Linear+fuse-capable all-gather):
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

#include "indexer_score_host_common.hpp"  // shared causal geometry / device index / persistent-cache args
#include "ring_indexer_score_schedule.hpp"
#include "kernels/indexer_score_cb.hpp"
#include "kernels/indexer_score_work_split.hpp"

namespace ttnn::operations::experimental::indexer_score::program {

namespace ag_rt = ttnn::ring_attention_all_gather_async_detail;

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kReaderKernelIndex = 0;
constexpr uint32_t kWriterKernelIndex = 1;
constexpr uint32_t kComputeKernelIndex = 2;
constexpr uint32_t kFirstAllGatherKernelIndex = 3;
constexpr uint32_t kAllGatherReaderForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kReaderForwardKernelOffset;
constexpr uint32_t kAllGatherWriterForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kWriterForwardKernelOffset;
constexpr uint32_t kAllGatherReaderBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kReaderBackwardKernelOffset;
constexpr uint32_t kAllGatherWriterBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kWriterBackwardKernelOffset;

// Runtime-arg slots the kernels match POSITIONALLY. The reader shares the classic factory's layout for slots
// 0..26 (q/k/w addrs, schedule(6), 2 mcast dirs, persistent-cache), then appends a fused-ring tail. Derived
// (not hardcoded) from one source and locked to the kernel-side offsets by the static_assert below, so a drift
// fails the build instead of silently desyncing the kernels. File-local, mirroring the classic factory.
namespace rt_arg {
constexpr uint32_t reader_num_scalars = 3 + 6;  // q/k/w addrs + schedule {row_group0..max_bands}
constexpr uint32_t mcast_args_per_dir = 8;      // role, rect(xs,ys,xe,ye), sender(sx,sy), ndst
constexpr uint32_t reader_num_mcast_dirs = 2;   // K column, then Q/W row
// {ring_size, ring_index, fwd, bwd, sem0, sem1, split_enabled, split_shard_id, split_second_half_wait}
// (the kernel-side RingSDPAOpReceiver consumes all nine; this consumer runs with split forwarding off)
constexpr uint32_t fused_rt_width = 9;
constexpr uint32_t reader_k_batch_offset = reader_num_scalars + reader_num_mcast_dirs * mcast_args_per_dir;  // 25
constexpr uint32_t reader_kv_len_tiles = reader_k_batch_offset + 1;                                          // 26
constexpr uint32_t reader_fused_rt_base = reader_kv_len_tiles + 1;                                           // 27
constexpr uint32_t reader_k_local_addr = reader_fused_rt_base + fused_rt_width;                              // 36
constexpr uint32_t reader_k_local_batch_offset = reader_k_local_addr + 1;                                    // 37
constexpr uint32_t reader_band_perm_base = reader_k_local_batch_offset + 1;                                  // 38
// Compute RT: schedule(6), kv_len_tiles, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles, then perm.
constexpr uint32_t compute_kv_len_tiles = 6;
constexpr uint32_t compute_chunk_start_tiles = compute_kv_len_tiles + 1;
constexpr uint32_t compute_straddle_q_tile = compute_chunk_start_tiles + 1;
constexpr uint32_t compute_straddle_jump_tiles = compute_straddle_q_tile + 1;
constexpr uint32_t compute_band_perm_base = compute_straddle_jump_tiles + 1;  // 10
// Writer RT: out addr, schedule(6), kv_len_tiles, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles, perm.
constexpr uint32_t writer_kv_len_tiles = 1 + 6;
constexpr uint32_t writer_chunk_start_tiles = writer_kv_len_tiles + 1;
constexpr uint32_t writer_straddle_q_tile = writer_chunk_start_tiles + 1;
constexpr uint32_t writer_straddle_jump_tiles = writer_straddle_q_tile + 1;
constexpr uint32_t writer_band_perm_base = writer_straddle_jump_tiles + 1;  // 11
// Lock the derived offsets to the values the kernels hardcode (reader receiver reads the fused block at 27;
// compute/writer read their perm at 10/11). A drift here would silently desync the kernels -> this fails to build.
static_assert(
    reader_k_batch_offset == 25 && reader_kv_len_tiles == 26 && reader_fused_rt_base == 27 &&
        reader_k_local_addr == 36 && reader_k_local_batch_offset == 37 && reader_band_perm_base == 38 &&
        compute_band_perm_base == 10 && writer_band_perm_base == 11,
    "indexer_score fused rt_arg slot layout drifted from the kernel-side expectations");
}  // namespace rt_arg

// Block-cyclic caches are slab-major on every SP rank. A global valid prefix can end partway through a
// global slab, whose valid row count differs by rank, so gather complete touched slabs. This is the smallest
// uniform per-rank prefix that preserves the fixed-size ring protocol and contains every valid key.
std::optional<uint32_t> gather_valid_height_tiles(const operation_attributes_t& args, const Tensor& k_local) {
    if (!args.kv_len.has_value() || !args.block_cyclic.has_value()) {
        return std::nullopt;
    }
    const uint32_t chunk_local = args.block_cyclic->chunk_local;
    const uint32_t chunk_global = args.block_cyclic->sp * chunk_local;
    const uint32_t valid_slabs = (*args.kv_len + chunk_global - 1) / chunk_global;
    const uint32_t valid_local_rows = valid_slabs * chunk_local;
    return std::min<uint32_t>(valid_local_rows, k_local.logical_shape()[2]) / tt::constants::TILE_HEIGHT;
}

// One device's fused program: indexer compute (banded schedule, DSA path) + co-scheduled ring_attention AG.
ProgramDescriptor build_ring_program_descriptor(
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

    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    const uint32_t device_index = device_index_for(args, coord, q);
    // 2D SP×TP: the K cache is SP-sharded + TP-replicated and the ring AG still gathers along the SP axis
    // (cluster_axis), so the reader's K sourcing is unchanged; TP only sub-shards the QUERY rows. tp_index is
    // this device's rank along seq_subshard_axis (the TP axis) -- device_causal_geometry adds its tp_index*Sq
    // slab-local sub-offset (rotation-exact, incl. straddle). 0 (SP-only) when seq_subshard_axis is unset.
    const uint32_t tp_index = (args.tp_axis().has_value() && q.device_storage().get_coords().size() > 1)
                                  ? ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.tp_axis())
                                  : 0u;
    const auto geom = device_causal_geometry(args, device_index, tp_index, Sq);
    const uint32_t chunk_t = geom.chunk_start_tiles;

    const uint32_t Sqt = Sq / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = T / tt::constants::TILE_WIDTH;
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;
    // Per-SP-shard block-cyclic run width in tiles (0 for a contiguous cache).
    const uint32_t cl_t = args.has_block_cyclic() ? args.block_cyclic->chunk_local / tt::constants::TILE_WIDTH : 0;

    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    const uint32_t G = 1;
    const uint32_t subblock_basis = HB;
    // Step-E band reorder assumes no head streaming (all heads resident): stream_heads pads the band loop with
    // phantom q-mcast bands that the reorder would perturb. HB == Hi means head_group_size was 0 or Hi.
    TT_FATAL(HB == Hi, "indexer_score fused: head_group_size must be 0 or Hi (no head streaming) on the fused path");

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

    // --- banded-product schedule (DSA: group_count = Sqt/QC), on a grid REDUCED to leave AG worker columns. ---
    const auto phys_grid = q.device()->compute_with_storage_grid_size();
    // Keep ALL compute ROWS and reserve COLUMNS for the AG workers. rows_for_groups() picks the largest DIVISOR
    // of the group count <= grid_y, so shaving even one row can halve the schedule (10 groups: grid_y 10->9 drops
    // group_rows 10->5 -> 55 cores instead of 110). cols_for_bands() just distributes the bands over min(bands,
    // compute_cols_x) columns (uneven remainder handled by band_list), so a reserved column costs exactly one
    // column of compute -- no divisor cliff. The AG uses num_links*2 workers, so compute keeps
    // grid_y*compute_cols_x cores.
    const uint32_t grid_y = phys_grid.y;  // full compute-grid height (all rows kept for compute)
    TT_FATAL(fused.num_links >= 1, "indexer_score fused: num_links must be >= 1 (got {})", fused.num_links);
    const uint32_t ag_worker_cores = fused.num_links * 2u;                   // AG uses 2 workers (fwd/bwd) per link
    const uint32_t reserved_cols = (ag_worker_cores + grid_y - 1) / grid_y;  // columns needed to hold the workers
    TT_FATAL(phys_grid.x > reserved_cols, "indexer_score fused: grid too small to reserve {} AG cols", reserved_cols);
    // Compute columns after reserving the AG worker column(s); NOTE this differs from the classic factory's
    // grid_x (which is the FULL grid width) -- here the rightmost reserved_cols columns belong to the AG.
    const uint32_t compute_cols_x = phys_grid.x - reserved_cols;
    const CoreCoord ccl_core_grid_offset{compute_cols_x, 0};  // AG workers start in the first reserved column

    const uint32_t ring_size = ring_size_for(args, q);  // shared with validate/signaler (same ring extent)
    TT_FATAL(ring_size <= 32, "indexer_score fused: Ring size {} exceeds the consumer gate capacity 32", ring_size);
    const uint32_t sll_t = Tt / ring_size;  // physical tiles per SP shard in gathered K
    // The two-marker protocol requires both ring directions to carry slices. Linear topologies and a two-chip
    // ring leave one direction with no targets, whose writer intentionally discards its local packet stream.
    const bool partial_readiness_enabled =
        fused.topology == ttnn::ccl::Topology::Ring && ring_size > 2 &&
        ag_rt::uses_output_bank_owned_schedule({k_local}, {k}, /*dim=*/2, ring_size, args.cache_batch_idx);
    const uint32_t units_per_shard = units_in_group(KC, sll_t);
    const uint32_t work_unit_count = ring_size * units_per_shard;
    const uint32_t group_count = Sqt / QC;                             // q-groups (banded schedule rows)
    const uint32_t group_rows = rows_for_groups(group_count, grid_y);  // grid rows per group (k-mcast)
    // The shard-major deal assigns every arrival wave independently. Size the physical lane space from one
    // shard, not the ring-wide work total, so every lane receives at least its local-shard unit. Otherwise a
    // small capacity can create an empty compute lane whose reader still joins the row multicast rendezvous.
    const uint32_t cols_used = cols_for_bands(units_per_shard, compute_cols_x);
    const uint32_t num_blocks =
        band_row_blocks(group_count, units_per_shard, compute_cols_x, grid_y);  // row-block reps
    const uint32_t rows_used = group_rows * num_blocks;                         // grid rows used by compute
    const uint32_t num_groups = group_count / group_rows;  // phase-stack count (groups per row, round-robin)

    // Group physical SP shards by their ring-arrival wave (the local shard is wave 0). Work units never cross a
    // shard boundary, so each has exactly one readiness dependency. RingIdSequencer already emits each wave in
    // the desired paired-direction order; preserve it directly rather than reconstructing and sorting it later.
    const auto rw = ring_schedule::ring_writes_for(ring_size, device_index, fused.topology);
    const auto shards_by_wave = ring_schedule::arrival_waves(ring_size, device_index, rw);
    // Deal each shard's physical KC units over the block-column lane space. A true bidirectional Ring rotates each
    // arrival wave's column residues within both row blocks; Linear and Ring-2 retain the prior mapping. The local
    // wave's formulaic shift is zero, paired shards keep matching offsets, adjacent row blocks retain their DRAM
    // access pairing, and the last unit stays padded within its source shard.
    auto work_list = ring_schedule::make_work_list(
        shards_by_wave,
        units_per_shard,
        sll_t,
        KC,
        num_blocks,
        cols_used,
        ring_schedule::rotation_enabled(fused.topology, ring_size));
    uint32_t max_bands = 0;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            max_bands = std::max<uint32_t>(max_bands, static_cast<uint32_t>(work_list[blk][col].size()));
        }
    }

    const CoreRange core_rect(CoreCoord{0, 0}, CoreCoord{cols_used - 1, rows_used - 1});
    const CoreRangeSet core_ranges(core_rect);

    std::vector<std::vector<CoreCoord>> phys(rows_used, std::vector<CoreCoord>(cols_used));
    for (uint32_t row = 0; row < rows_used; ++row) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            phys[row][col] = q.device()->worker_core_from_logical_core(CoreCoord{col, row});
        }
    }

    // k-mcast shares a block's band-chunk down its group_rows rows; q/w-mcast needs >1 column along a row.
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
    // Provably false under the HB==Hi guard above; kept so the CB sizing below mirrors the classic factory.
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
    // cb_qk buffers a batch of relu(q.kT) tiles so compute runs the batch's matmuls then mul+accumulates,
    // hoisting the matmul<->eltwise reinit out of the per-head loop (shared with the classic factory).
    const auto [qk_batch_heads, qk_col_batch] = dsa_qk_batching(subblock_basis, QC, KC, stream_heads);

    make_cb(cb_q_arg, (stream_heads ? 2 : 1) * HB * QC * Dt, q_fmt, q_tile);
    make_cb(cb_k_arg, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w_arg, Hi * QC, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask_arg, num_mask_tiles, tt::DataFormat::Float16_b, bf16_tile);
    // cb_qk stages the batched relu(q.kT) strip for the gate-mul phase.
    make_cb(cb_qk_arg, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    // cb_out_strip holds the untilized output, double-buffered (2*KC; no block-pool on the DSA path).
    make_cb(cb_out_strip_arg, 2 * KC, tt::DataFormat::Float16_b, bf16_tile);
    // cb_acc_strip accumulates a whole unit's QC*KC strip, then untilizes under ONE pack_untilize bracket.
    // max(2*KC, .) keeps the QC<=2 double buffer and a whole multiple of QC*KC so a push never wraps mid-unit.
    make_cb(cb_acc_strip_arg, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // Fused-op signal semaphores + consumer signaler (inlined init_fused_op against desc; MULTI). ring_size / rw
    // are computed above (hoisted for the readiness-balanced band assignment).
    ttnn::prim::RingSDPAFusedOpSignaler sdpa_sig;
    sdpa_sig.init_all_gather(ring_size, device_index, rw.forward_writes_expected, rw.backward_writes_expected);
    sdpa_sig.fused_op_signaler_mode = ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;
    sdpa_sig.fused_op_receiver_cores_noc.clear();
    // Signal ONLY the cores that actually gate on the all-gather. The AG master worker's per-slab signal is a
    // UNICAST LOOP over the receiver cores (worker_sync_utils MULTI mode: one noc_semaphore_inc per core per
    // delivered slab), and it sits on the gather's critical path -- so signalling the whole 100-core compute
    // rectangle slowed the co-scheduled AG ~40-80us vs standalone. With k-mcast on (group_rows>1) only the
    // k-mcast SENDER of each column reads K from DRAM and gates (reader: k_dir.role==sender); the receiver rows
    // take the already-gated K over the column mcast and never wait on the AG semaphore. So the sender set --
    // row == block_base for each block, every column -- is the minimal correct receiver list (num_blocks*cols_used
    // cores, ~group_rows-fold fewer signals). With k-mcast off every core is its own k-reader, so signal them all.
    std::vector<CoreCoord> signal_cores;
    if (k_mcast_on) {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            for (uint32_t col = 0; col < cols_used; ++col) {
                signal_cores.push_back(CoreCoord{col, blk * group_rows});
            }
        }
    } else {
        signal_cores = corerange_to_cores(core_ranges, std::nullopt, /*row_wise=*/true);
    }
    for (const auto& core : signal_cores) {
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
    reader_ct.push_back(1u);  // fused_ring on
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
    // Block-cyclic (per-SP-shard) K layout, passed as 5 compile-time-constexpr args to match the non-fused
    // reader's logical_to_physical_page<> template: {block_cyclic, chunk_local_tiles, sp, shard_stride_gap,
    // slab_stride_gap}. Default {0,1,1,0,0} (identity) when not block-cyclic. Gaps match the regular factory:
    // shard_stride_gap = Tt/sp - chunk_local, slab_stride_gap = chunk_local*(sp-1).
    const auto block_cyclic_ct = [&args, Tt, cl_t]() {
        std::array<uint32_t, 5> ct{0, 1, 1, 0, 0};
        if (!args.has_block_cyclic()) {
            return ct;
        }
        const uint32_t sp = args.block_cyclic->sp;
        ct = {1, cl_t, sp, (Tt / sp) - cl_t, cl_t * (sp - 1)};
        return ct;
    }();
    reader_ct.insert(reader_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    reader_ct.push_back(static_cast<uint32_t>(partial_readiness_enabled));

    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.push_back(1u);  // fused_ring on
    const uint32_t out_elem_bytes = out.element_size();
    writer_ct.push_back(T * out_elem_bytes);  // row-major page = one output row (no pooling)
    writer_ct.push_back(block_cyclic_ct[0]);  // shard-major physical -> logical output mapping
    writer_ct.push_back(block_cyclic_ct[1]);
    writer_ct.push_back(ring_size);  // physical shard count (also block-cyclic SP when enabled)
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);
    compute_ct.push_back(qk_col_batch);
    compute_ct.push_back(1u);                  // apply_relu (DSA)
    compute_ct.push_back(0u);                  // fuse_single off
    compute_ct.push_back(0u);                  // fused_stream_k off
    compute_ct.push_back(1u);                  // fused_ring on
    compute_ct.push_back(block_cyclic_ct[0]);  // shard-major physical -> logical causal mapping
    compute_ct.push_back(block_cyclic_ct[1]);
    compute_ct.push_back(ring_size);  // physical shard count (also block-cyclic SP when enabled)

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source = kdir + "reader_indexer_score.cpp";
    reader_kernel.core_ranges = core_ranges;
    reader_kernel.compile_time_args = reader_ct;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source = kdir + "writer_indexer_score.cpp";
    writer_kernel.core_ranges = core_ranges;
    writer_kernel.compile_time_args = writer_ct;
    writer_kernel.config = WriterConfigDescriptor{};

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source = kdir + "compute_indexer_score.cpp";
    compute_kernel.core_ranges = core_ranges;
    compute_kernel.compile_time_args = compute_ct;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};

    const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
    // Append a scalar range (e.g. the 6-element sched array) to a kernel's RT list, preserving push order.
    // (band_perm is a std::vector, so it uses RTArgList::append directly.)
    const auto append_scalars = [](KernelDescriptor::RTArgList& rt, const auto& range) {
        for (uint32_t v : range) {
            rt.push_back(v);
        }
    };
    const auto pcache = persistent_cache_args(args, k);  // kv_len derivation shared with the classic factory
    // Indexed fused mode gathers the selected cache slot into slot 0 of batch-1 k. Keep the gathered and
    // local offsets independent: remote reads start at zero, own-shard reads select the original k_local slot.
    const uint32_t k_batch_page_offset = args.cache_batch_idx.has_value() ? 0u : pcache.k_batch_page_offset;
    const uint32_t local_slot_pages = (k_local.logical_shape()[2] / tt::constants::TILE_HEIGHT) *
                                      (k_local.logical_shape()[3] / tt::constants::TILE_WIDTH);
    const uint32_t k_local_batch_page_offset = args.cache_batch_idx.value_or(0) * local_slot_pages;
    const uint32_t kv_len_tiles = pcache.kv_len_tiles;

    std::vector<uint32_t> fused_rt;
    sdpa_sig.push_ring_sdpa_fused_op_rt_args(fused_rt);  // the fused_rt_width-wide block (see rt_arg above)

    // ---- Shard-major visit order: local physical shard first, then remote shards by ring arrival. --------
    for (uint32_t row = 0; row < rows_used; ++row) {
        // Q/W row mcast rect + diagonal sender (shared with the classic factory).
        const auto qb = q_mcast_bbox(phys, row, cols_used);
        const uint32_t q_xs = qb.xs, q_xe = qb.xe, q_py = qb.py, q_diag = qb.diag_col;
        const CoreCoord q_sender = qb.sender;
        const uint32_t block = row / group_rows;
        const uint32_t block_base = block * group_rows;
        for (uint32_t col = 0; col < cols_used; ++col) {
            // K column mcast rect + block-top sender (shared with the classic factory).
            const auto kb = k_mcast_bbox(phys, block_base, col, group_rows);
            const uint32_t k_ys = kb.ys, k_ye = kb.ye, k_px = kb.px;
            const CoreCoord k_sender = kb.sender;
            const CoreCoord core{col, row};
            // Physical K-tile starts, identical for every row in a K-mcast column.
            const std::vector<uint32_t>& physical_starts = work_list[block][col];
            const uint32_t col_num_bands = static_cast<uint32_t>(physical_starts.size());
            const std::array<uint32_t, 6> sched = {
                row % group_rows, group_rows, num_groups, /*band0=*/0u, col_num_bands, max_bands};

            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(q.buffer());
            reader_rt.push_back(k.buffer());
            reader_rt.push_back(w.buffer());
            append_scalars(reader_rt, sched);
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
            // Reader tail (sequential push; slots named in rt_arg, matched positionally by the kernel).
            reader_rt.push_back(k_batch_page_offset);  // rt_arg::reader_k_batch_offset (25)
            reader_rt.push_back(kv_len_tiles);         // rt_arg::reader_kv_len_tiles (26)
            reader_rt.append(fused_rt);                // rt_arg::reader_fused_rt_base (27..35): ring/dir/sems/split
            reader_rt.push_back(k_local.buffer());     // rt_arg::reader_k_local_addr (36): local SP shard address
            reader_rt.push_back(k_local_batch_page_offset);  // selected slot in the original local cache
            reader_rt.append(physical_starts);               // rt_arg::reader_band_perm_base (38..): physical K starts
            reader_kernel.emplace_runtime_args(core, reader_rt);

            KernelDescriptor::RTArgList compute_rt;
            append_scalars(compute_rt, sched);
            compute_rt.push_back(kv_len_tiles);
            compute_rt.push_back(chunk_t);
            compute_rt.push_back(geom.straddle_q_tile);
            compute_rt.push_back(geom.straddle_jump_tiles);
            compute_rt.append(physical_starts);  // rt_arg::compute_band_perm_base (10..): physical K starts
            compute_kernel.emplace_runtime_args(core, compute_rt);

            KernelDescriptor::RTArgList writer_rt;
            writer_rt.push_back(out.buffer());
            append_scalars(writer_rt, sched);
            writer_rt.push_back(kv_len_tiles);
            writer_rt.push_back(chunk_t);
            writer_rt.push_back(geom.straddle_q_tile);
            writer_rt.push_back(geom.straddle_jump_tiles);
            writer_rt.append(physical_starts);  // rt_arg::writer_band_perm_base (11..): physical K starts
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
        ttnn::ccl::get_physical_neighbor_from_physical_coord(q, coord, /*offset=*/1, fused.topology, args.sp_axis());
    const auto backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(q, coord, /*offset=*/-1, fused.topology, args.sp_axis());

    std::vector<Tensor> ag_in = {k_local};
    std::vector<Tensor> ag_out = {k};
    // The gather concatenates the SP shards along the seq axis (dim 2); the reader's block-cyclic permutation
    // assumes this, so it is a fixed constant, not a configurable knob.
    constexpr int32_t ag_seq_concat_dim = 2;
    ttnn::ring_attention_all_gather_async_multi_core_with_workers_helper(
        desc,
        ag_in,
        coord,
        forward_coord,
        backward_coord,
        ag_out,
        ag_seq_concat_dim,
        fused.num_links,
        ring_size,
        device_index,
        fused.topology,
        fused.ag_semaphore,
        fused.ag_sub_device_id,
        ag_sig,
        ccl_core_grid_offset,
        // COL_MAJOR so the reserved-column offset lays the workers DOWN the free column ((compute_cols_x,0),
        // (compute_cols_x,1), ...) instead of running off the right grid edge as row-major would.
        ttnn::ccl::CoreAllocationStrategy::COL_MAJOR,
        args.cache_batch_idx,
        gather_valid_height_tiles(args, k_local),
        /*slot_id=*/std::nullopt,
        /*kv_actual_isl=*/std::nullopt,
        /*chunk_local_tiles=*/0,
        /*kv_cache_num_layers=*/1,
        /*kv_cache_layer_idx=*/0,
        // This consumer uses midpoint/completion readiness rather than diametric split forwarding.
        /*split_forwarding_enabled=*/false,
        /*partial_readiness_enabled=*/partial_readiness_enabled);

    log_debug(
        tt::LogOp,
        "indexer_score FUSED coord=({}) ring_size={} ring_index={} fwd_exp={} bwd_exp={} grid={}x{}(+{} ag) "
        "rows_used={} cols_used={} work_unit_count={} k_mcast={} q_mcast={}",
        device_index,
        ring_size,
        device_index,
        rw.forward_writes_expected,
        rw.backward_writes_expected,
        compute_cols_x,
        grid_y,
        reserved_cols,
        rows_used,
        cols_used,
        work_unit_count,
        k_mcast_on,
        q_mcast_on);

    return desc;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::WorkloadDescriptor RingIndexerScoreDsaProgramFactory::create_workload_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    tensor_return_value_t& out,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = CMAKE_UNIQUE_NAMESPACE::build_ring_program_descriptor(args, tensors, out, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

RingIndexerScoreDsaMeshWorkloadFactory::cached_mesh_workload_t
RingIndexerScoreDsaMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensors, out);
}

void RingIndexerScoreDsaMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    // Buffer addresses (q/k/w/out/k_local + the AG's gathered buffer) auto-patch via the descriptor's
    // BufferBinding fast path.
    descriptor_adapter_t::apply_descriptor(cached, args, tensors, out);

    // The per-dispatch scalars chunk_start_idx / kv_len / cache_batch_idx are HASH-EXCLUDED (see
    // compute_program_hash: one cached program is reused across chunked-prefill chunks and decode steps that
    // differ only in these), so on a program-cache HIT the WorkloadDescriptor fast path above leaves them
    // frozen at the FIRST dispatch's values -- a stale causal offset (chunk_start_tiles/straddle) and valid
    // length (kv_len_tiles), which silently corrupts every chunk after the first. Patch the fields directly
    // into the cached programs, as the classic Program-model factory does. Rebuilding the descriptor here was
    // correct but expensive: it reconstructed geometry, schedules, descriptors and consumer runtime args for
    // every mesh coordinate even though only these scalar fields vary between cache hits.
    using tt::tt_metal::GetRuntimeArgs;

    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& k_local = *tensors.k_local;
    const auto pcache = persistent_cache_args(args, k);
    const uint32_t k_batch_page_offset = args.cache_batch_idx.has_value() ? 0u : pcache.k_batch_page_offset;
    const auto& k_local_shape = k_local.logical_shape();
    const uint32_t local_slot_pages =
        (k_local_shape[2] / tt::constants::TILE_HEIGHT) * (k_local_shape[3] / tt::constants::TILE_WIDTH);
    const uint32_t k_local_batch_page_offset = args.cache_batch_idx.value_or(0) * local_slot_pages;
    for (auto& [range, program] : cached.workload.get_programs()) {
        const uint32_t device_index = device_index_for(args, range.start_coord(), q);
        const uint32_t tp_index =
            (args.tp_axis().has_value() && q.device_storage().get_coords().size() > 1)
                ? ttnn::ccl::get_linearized_index_from_physical_coord(q, range.start_coord(), args.tp_axis())
                : 0u;
        const auto geom = device_causal_geometry(args, device_index, tp_index, q.logical_shape()[2]);

        const auto patch_field = [&](uint32_t kernel_idx, uint32_t slot, uint32_t value) {
            auto& grid_args = GetRuntimeArgs(program, kernel_idx);
            for (auto& col_args : grid_args) {
                for (auto& core_args : col_args) {
                    TT_FATAL(
                        slot < core_args.size(),
                        "indexer_score fused override: scalar slot {} out of range (size {}) for kernel {}",
                        slot,
                        core_args.size(),
                        kernel_idx);
                    core_args[slot] = value;
                }
            }
        };

        patch_field(kReaderKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::reader_k_batch_offset, k_batch_page_offset);
        patch_field(kReaderKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::reader_kv_len_tiles, pcache.kv_len_tiles);
        patch_field(
            kReaderKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::reader_k_local_batch_offset, k_local_batch_page_offset);
        patch_field(kComputeKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::compute_kv_len_tiles, pcache.kv_len_tiles);
        patch_field(
            kComputeKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::compute_chunk_start_tiles, geom.chunk_start_tiles);
        patch_field(kComputeKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::compute_straddle_q_tile, geom.straddle_q_tile);
        patch_field(
            kComputeKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::compute_straddle_jump_tiles, geom.straddle_jump_tiles);
        patch_field(kWriterKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::writer_kv_len_tiles, pcache.kv_len_tiles);
        patch_field(
            kWriterKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::writer_chunk_start_tiles, geom.chunk_start_tiles);
        patch_field(kWriterKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::writer_straddle_q_tile, geom.straddle_q_tile);
        patch_field(
            kWriterKernelIndex, CMAKE_UNIQUE_NAMESPACE::rt_arg::writer_straddle_jump_tiles, geom.straddle_jump_tiles);

        // The descriptor fast path patches buffer bindings but not scalar fields embedded in the fused AG
        // workers. cache_batch_idx and the exact kv_len are hash-excluded (only the structural schedule class is
        // hashed), so update the selected input slot and slab-rounded gather extent on every cache hit.
        const auto& shape = k_local.padded_shape();
        const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
        const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t input_batch_base =
            ag_rt::input_batch_base_pages(args.cache_batch_idx.value_or(0), shape[1], Ht, Wt);
        const auto valid_Ht = gather_valid_height_tiles(args, k_local);
        const uint32_t valid_pages = std::min(valid_Ht.value_or(Ht), Ht) * Wt;

        const auto patch_ag_field = [&](uint32_t kernel_idx, uint32_t slot, uint32_t value) {
            auto& grid_args = GetRuntimeArgs(program, kernel_idx);
            bool patched_any_core = false;
            for (auto& col_args : grid_args) {
                for (auto& core_args : col_args) {
                    if (core_args.size() == 0) {
                        continue;
                    }
                    TT_FATAL(
                        core_args.size() > slot,
                        "indexer_score fused override: AG scalar slot {} out of range (size {}) for kernel {}",
                        slot,
                        core_args.size(),
                        kernel_idx);
                    core_args[slot] = value;
                    patched_any_core = true;
                }
            }
            TT_FATAL(
                patched_any_core, "indexer_score fused override: AG kernel {} has no runtime arguments", kernel_idx);
        };

        // Semaphore identity is hash-excluded. Rebind both ring directions on every cache hit so callers
        // can alternate double-buffered semaphore pairs without compiling a second otherwise-identical
        // program. Runtime-arg layouts are declared in ring_attention_all_gather_async_detail.
        const auto& ag_semaphores = args.fused_ring->ag_semaphore;
        TT_FATAL(
            ag_semaphores.size() >= 2,
            "indexer_score fused override: expected at least 2 AG semaphores, got {}",
            ag_semaphores.size());
        const uint32_t backward_semaphore = static_cast<uint32_t>(ag_semaphores[0].address());
        const uint32_t forward_semaphore = static_cast<uint32_t>(ag_semaphores[1].address());
        patch_ag_field(kAllGatherReaderForwardKernelIndex, ag_rt::kReaderReadySemaphoreFieldOffset, forward_semaphore);
        patch_ag_field(kAllGatherWriterForwardKernelIndex, ag_rt::kWriterReadySemaphoreFieldOffset, forward_semaphore);
        patch_ag_field(
            kAllGatherReaderBackwardKernelIndex, ag_rt::kReaderReadySemaphoreFieldOffset, backward_semaphore);
        patch_ag_field(
            kAllGatherWriterBackwardKernelIndex, ag_rt::kWriterReadySemaphoreFieldOffset, backward_semaphore);

        constexpr uint32_t ag_reader_input_base =
            ag_rt::kReaderRuntimeArgHeaderCount + ag_rt::kInputBatchBaseFieldOffset;
        constexpr uint32_t ag_reader_valid_pages = ag_rt::kReaderRuntimeArgHeaderCount + ag_rt::kValidPagesFieldOffset;
        constexpr uint32_t ag_writer_valid_pages = ag_rt::kWriterRuntimeArgHeaderCount + ag_rt::kValidPagesFieldOffset;
        patch_ag_field(kAllGatherReaderForwardKernelIndex, ag_reader_input_base, input_batch_base);
        patch_ag_field(kAllGatherReaderBackwardKernelIndex, ag_reader_input_base, input_batch_base);
        patch_ag_field(kAllGatherReaderForwardKernelIndex, ag_reader_valid_pages, valid_pages);
        patch_ag_field(kAllGatherReaderBackwardKernelIndex, ag_reader_valid_pages, valid_pages);
        patch_ag_field(kAllGatherWriterForwardKernelIndex, ag_writer_valid_pages, valid_pages);
        patch_ag_field(kAllGatherWriterBackwardKernelIndex, ag_writer_valid_pages, valid_pages);
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program
