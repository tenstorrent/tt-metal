// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>  // log_info: per-program schedule/mcast summary
#include <tt-metalium/mesh_workload.hpp>
#include "hostdevcommon/kernel_structs.h"  // tt::CBIndex

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"  // get_linearized_index_from_physical_coord
#include "kernels/indexer_score_runtime_args.hpp"
#include "indexer_score_host_common.hpp"         // shared causal geometry / device index / persistent-cache args
#include "kernels/indexer_score_cb.hpp"          // shared host/device CB-index argument layout (CbArg)
#include "kernels/indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace ttnn::operations::experimental::indexer_score::program {

namespace {
// The reader's common block (indexer_common::reader order). Single source for create_at (miss) and
// override_runtime_arguments (hit), so the cache-hit patch writes exactly what the miss built. Every buffer
// address here -- q/k/w and the trace metadata tensors -- is re-read from the CURRENT tensors on every dispatch,
// so a cache hit with freshly allocated metadata tensors re-points the reader instead of reading the tensors
// of the dispatch that built the program. Fused-only slots stay zero. A zero metadata address is the reader's
// "absent" signal (a live buffer is never at address 0).
std::array<uint32_t, indexer_common::reader::Count> classic_reader_values(
    const operation_attributes_t& args, const tensor_args_t& tensors, uint32_t device_index, uint32_t tp_index) {
    const auto [k_batch_page_offset, kv_len_tiles] = persistent_cache_args(args, tensors.k);
    const auto address_or_zero = [](const std::optional<Tensor>& t) {
        return t.has_value() ? t->buffer()->address() : 0u;
    };
    std::array<uint32_t, indexer_common::reader::Count> values{};
    values[indexer_common::reader::Q] = tensors.q.buffer()->address();
    values[indexer_common::reader::K] = tensors.k.buffer()->address();
    values[indexer_common::reader::W] = tensors.weights.buffer()->address();
    values[indexer_common::reader::BatchOffset] = k_batch_page_offset;
    values[indexer_common::reader::KvLength] = kv_len_tiles;
    values[indexer_common::reader::ChunkMetadata] = address_or_zero(tensors.chunk_start_idx_tensor);
    values[indexer_common::reader::DeviceIndex] = device_index;
    values[indexer_common::reader::TpIndex] = tp_index;
    values[indexer_common::reader::SlotMetadata] = address_or_zero(tensors.cache_batch_idx_tensor);
    values[indexer_common::reader::NumLayers] = args.index_cache_num_layers;
    values[indexer_common::reader::LayerIndex] = args.index_cache_layer_idx;
    values[indexer_common::reader::ValidEnd] = address_or_zero(tensors.valid_end_tensor);
    return values;
}
}  // namespace

// Banded-product schedule: the work space (group_count q-row-groups x band_count k-bands) tiles onto a
// rows_used x cols_used core rectangle -- groups -> rows (q/w mcast along a row), bands -> columns (k
// mcast down a column). One cell = one QC x up-to-KC work unit. See indexer_score_work_split.hpp.
IndexerScoreProgramFactory::cached_program_t IndexerScoreProgramFactory::create_at(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinate& coord,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& w = tensors.weights;

    // Inputs and knobs are validated in IndexerScoreDeviceOperation::validate_on_program_cache_miss;
    // here we only derive tile dims and the one build-specific (subblock) constraint.
    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    // This device's SP-ring index and chunk_start (tiles), from the coordinate. chunk_t is a compute RUNTIME
    // arg, so the binary is identical across coords and steps. tp_index = its rank along the TP axis
    // (seq_shard_axes[1], the 2D SP×TP query sub-shard); 0 when not sub-sharded or single-device.
    const uint32_t device_index = device_index_for(args, coord, q);
    const uint32_t tp_index = (args.tp_axis().has_value() && q.device_storage().get_coords().size() > 1)
                                  ? ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.tp_axis())
                                  : 0u;
    const auto geom = device_causal_geometry(args, device_index, tp_index, Sq);
    const uint32_t chunk_t = geom.chunk_start_tiles;

    const uint32_t Sqt = Sq / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = T / tt::constants::TILE_WIDTH;
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;

    // Work-unit knobs, converted from elements to tiles / heads.
    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);

    // num_groups: G==1 sums all heads (DSA/GLM); G>1 sums Hi/G heads per group into G planes (M3). The
    // subblock height and cb_qk batch key off the per-plane width: HB when G==1, plane_heads when G>1.
    const uint32_t G = args.num_groups;
    const uint32_t plane_heads = Hi / G;
    const uint32_t subblock_basis = (G > 1) ? plane_heads : HB;

    // block-max-pool: 0 = off; >0 = max over each block_size-key block -> [.,.,Sq,T/block_size] (M3).
    // block_tiles k-tiles per block; a unit's KC tiles pool to blocks_per_unit block scores.
    const uint32_t block_tiles = args.block_size ? args.block_size / tt::constants::TILE_WIDTH : 0;
    const bool block_pool = block_tiles != 0;
    const uint32_t blocks_per_unit = block_pool ? (KC / block_tiles) : KC;
    const uint32_t nblocks = block_pool ? (Tt / block_tiles) : 0;  // total block columns per output row (pool only)

    // Compute knobs from the resolved config (validate guarantees fp32_dest_acc_en / dst_full_sync_en false).
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(q.device()->arch(), args.compute_kernel_config);

    // qk matmul subblock: heads are output rows, k column is 1 tile wide, so only the subblock height matters.
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;  // half-sync, as in sdpa_program_factory
    const uint32_t qk_subblock_h =
        ttnn::prim::detail::determine_largest_subblock_size(subblock_basis, 1, dst_size).first;
    TT_FATAL(
        subblock_basis % qk_subblock_h == 0,
        "per-plane head count {} must be divisible by qk_subblock_h={}",
        subblock_basis,
        qk_subblock_h);

    // QC/KC/HB are verbatim from the config (no auto-tune; caller owns the trade-off, oversized fails at
    // CB allocation).

    // ---- banded-product schedule -------------------------------------------------------------
    // groups -> rows (phase-stack when group_count > grid_y), bands -> columns (each owns a contiguous chunk).
    // When the group dimension leaves grid rows idle (short sequences: group_count < grid_y), replicate each
    // group across num_blocks row-blocks and split its band range across them (a band-chunk per block). Cells
    // in different blocks write disjoint output columns -- no cross-core reduce -- and each block's k-mcast
    // stays a contiguous per-column rectangle (a block's group_rows rows share that block's band-chunk).
    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);  // ceil(Tt/KC)
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t grid_x = grid.x, grid_y = grid.y;

    const uint32_t group_rows = rows_for_groups(group_count, grid_y);
    const uint32_t cols_used = cols_for_bands(band_count, grid_x);
    // Row-block replication factor (shared with the perf model so their core counts can't drift): fill the
    // idle rows (grid_y / group_rows), but never finer than one band per (block, column) cell. num_blocks==1
    // is the original single-band-row schedule -- the deployed long-sequence cases, where group_rows fills
    // grid_y.
    const uint32_t num_blocks = band_row_blocks(group_count, band_count, grid_x, grid_y);
    const uint32_t rows_used = group_rows * num_blocks;
    // Phase-stack count: groups dealt round-robin onto the group_rows rows (1 when group_rows == group_count).
    const uint32_t num_groups = group_count / group_rows;

    // Widest cell's band count: the streaming q-mcast pad target (the kernels pad each row to max_bands with
    // q-only phantom bands so the rendezvous stays uniform). Widest block has ceil(band_count/num_blocks)
    // bands; its widest column ceil(that/cols_used).
    const uint32_t bands_in_widest_block = (band_count + num_blocks - 1) / num_blocks;
    const uint32_t max_bands = (bands_in_widest_block + cols_used - 1) / cols_used;

    const CoreRange core_rect(CoreCoord{0, 0}, CoreCoord{cols_used - 1, rows_used - 1});
    const CoreRangeSet core_ranges(core_rect);

    // Physical coords of the used rectangle, indexed [row][col], for the mcast bounding boxes below.
    std::vector<std::vector<CoreCoord>> phys(rows_used, std::vector<CoreCoord>(cols_used));
    for (uint32_t row = 0; row < rows_used; ++row) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            phys[row][col] = q.device()->worker_core_from_logical_core(CoreCoord{col, row});
        }
    }

    // k-mcast shares a block's band-chunk down its group_rows rows; q/w-mcast needs >1 column along a row
    // (both HB-independent).
    const uint32_t k_mcast_on = (group_rows > 1) ? 1u : 0u;
    const uint32_t q_mcast_on = (cols_used > 1) ? 1u : 0u;

    // 3 semaphores per active direction: send (receivers ready), recv (sender relays valid in), valid
    // (constant 1). Mirrors SDPA chain_link's handshake.
    const uint32_t k_send_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;

    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);

    // q (srcB) and k (srcA) are matmul inputs; either may be bfp8_b to halve its footprint (w stays bf16).
    const bool q_is_bfp8 = q.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const bool k_is_bfp8 = k.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    // Fused single-head fast path: one head/plane, no ReLU, bf16 q -> the matmul writes the gated score
    // straight to the accumulator (cb_qk sized to 1, cb_out_strip deepened). Composes with the multicast.
    // bfp8 q is EXCLUDED: the in-place gate-fold (scale_q_by_w_inplace) re-packs gated q to bfp8 and the
    // matmul then misreads it (PCC ~0, garbage magnitudes) -- bf16 q has no shared-exponent section so it is
    // unaffected. bfp8 q therefore stays on the non-fused path (gate applied on the fp32 matmul output).
    const bool fuse_single = (plane_heads == 1) && !args.apply_relu && !q_is_bfp8;
    // Fused + mcast: K is one block -> wait whole chunk. Fused + no mcast: stream K in column sub-chunks.
    const bool fused_stream_k = fuse_single && (k_mcast_on == 0) && (q_mcast_on == 0);
    const tt::DataFormat q_fmt = q_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const tt::DataFormat k_fmt = k_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const uint32_t q_tile = tt::tile_size(q_fmt);
    const uint32_t k_tile = tt::tile_size(k_fmt);

    // Continuous CB indices: each make_cb claims the next free index and records it under its CbArg slot.
    // The whole array is forwarded to the kernels (CbArg = shared slot order), so indices can't drift.
    std::array<uint32_t, num_cb_args> cb_id{};
    uint32_t next_cb_index = tt::CBIndex::c_0;
    auto make_cb = [&](uint32_t slot, uint32_t ntiles, tt::DataFormat fmt, uint32_t tile_bytes) {
        const uint32_t idx = next_cb_index++;
        cb_id[slot] = idx;
        tt::tt_metal::CreateCircularBuffer(
            program,
            core_ranges,
            tt::tt_metal::CircularBufferConfig(ntiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };

    const bool stream_heads = HB < Hi;

    // One-line schedule/mcast summary (per cache miss) for profiling.
    log_debug(
        tt::LogOp,
        "indexer_score schedule: G={} U={} grid={}x{} group_rows={} num_blocks={} rows_used={} cols_used={} "
        "num_groups={} max_bands={} stream_heads={} k_mcast={} q_mcast={}",
        group_count,
        band_count,
        grid_x,
        grid_y,
        group_rows,
        num_blocks,
        rows_used,
        cols_used,
        num_groups,
        max_bands,
        stream_heads ? 1 : 0,
        k_mcast_on,
        q_mcast_on);

    // Allocate each CB by its CbArg slot; make_cb assigns the next continuous index.
    make_cb(cb_q_arg, (stream_heads ? 2 : 1) * HB * QC * Dt, q_fmt, q_tile);
    make_cb(cb_k_arg, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w_arg, Hi * QC, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask_arg, num_mask_tiles, tt::DataFormat::Float16_b, bf16_tile);
    const tt::DataFormat acc_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t acc_tile = fp32_dest_acc_en ? fp32_tile : bf16_tile;
    // cb_qk buffers a batch of relu(q.kT) tiles so compute runs the batch's matmuls then mul+accumulates,
    // hoisting the matmul<->eltwise reinit out of the per-head loop (shared with the fused factory).
    const auto [qk_batch_heads, qk_col_batch] = dsa_qk_batching(subblock_basis, QC, KC, stream_heads);
    // G>1 reuses the full-strip path (the fallback is not wired for groups); only fires if plane_heads
    // exceeds the batch cap (QC>1 with plane_heads>32).
    TT_FATAL(
        G == 1 || qk_col_batch > 1,
        "num_groups {}>1 requires the full-strip path (got qk_col_batch=1; plane_heads {} likely > batch cap)",
        G,
        plane_heads);
    // cb_qk stages the matmul output for the gate-mul phase; the fused path writes straight to the
    // accumulator, so cb_qk is unused there (1 tile, L1 spent on a deeper cb_out_strip).
    make_cb(cb_qk_arg, fuse_single ? 1u : (qk_col_batch * qk_batch_heads), acc_fmt, acc_tile);
    // cb_out_strip holds the pooled/untilized output, normally double-buffered. The fused block-pool path
    // deepens it to the whole unit's blocks so the pool and writer decouple (no 2-row-ring mutual stall).
    const uint32_t out_strip_tiles =
        (fuse_single && block_pool) ? (QC * blocks_per_unit) : 2 * (block_pool ? blocks_per_unit : KC);
    make_cb(cb_out_strip_arg, out_strip_tiles, tt::DataFormat::Float16_b, bf16_tile);
    // Block-max-pool scratch CBs (only when pooling): cb_scaler = one 1.0 reduce-MAX tile; cb_pool_scratch
    // = the writer's one-tile row-assembly buffer.
    if (block_pool) {
        make_cb(cb_scaler_arg, 1, tt::DataFormat::Float16_b, bf16_tile);
        make_cb(cb_pool_scratch_arg, 1, tt::DataFormat::Float16_b, bf16_tile);
    }
    // cb_acc_strip accumulates a whole unit's QC*KC strip, then untilizes under ONE pack_untilize bracket.
    // max(2*KC, .) keeps the QC<=2 double buffer and a whole multiple of QC*KC so a push never wraps mid-unit.
    make_cb(cb_acc_strip_arg, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // Trace-safe metadata CBs, after the shared CbArg slots so their indices never shift. Chunk start: the
    // reader lands the scalar and publishes the derived causal bounds, one mailbox per consumer (compute,
    // writer). Slot: a dedicated landing page for the user id.
    const bool has_meta = tensors.has_chunk_start_metadata();
    const bool has_slot_meta = tensors.has_cache_slot_metadata();
    uint32_t cb_meta_derived = 0, cb_meta_writer = 0, cb_meta_slot = 0;
    const auto make_meta_cb = [&](uint32_t& slot) {
        constexpr uint32_t meta_page_bytes = 64;
        slot = next_cb_index++;
        tt::tt_metal::CreateCircularBuffer(
            program,
            core_ranges,
            tt::tt_metal::CircularBufferConfig(meta_page_bytes, {{slot, tt::DataFormat::UInt32}})
                .set_page_size(slot, meta_page_bytes));
    };
    if (has_meta) {
        make_meta_cb(cb_meta_derived);
        make_meta_cb(cb_meta_writer);
    }
    if (has_slot_meta) {
        make_meta_cb(cb_meta_slot);
    }

    // Common args: 9 dims then the CB indices in CbArg order. chunk_t is NOT here (per-device runtime arg).
    std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, QC, KC, HB, G, block_tiles};
    common_ct.insert(common_ct.end(), cb_id.begin(), cb_id.end());

    // MSA synthesizes the constant gate in-kernel (no weights tensor / no fill op): the reader fills cb_w
    // with `gate_scale` instead of reading DRAM, and the weights accessor below is the unused q placeholder.
    // Pack the scale into a bf16 pair (two values per word) for the reader's word-wise fill.
    const uint16_t gate_scale_bf16 = static_cast<uint16_t>(__builtin_bit_cast(uint32_t, args.gate_scale) >> 16);
    const uint32_t gate_scale_bits = (static_cast<uint32_t>(gate_scale_bf16) << 16) | gate_scale_bf16;

    std::vector<uint32_t> reader_ct = common_ct;
    reader_ct.push_back(0u);  // fused_ring off
    tt::tt_metal::TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*w.buffer()).append_to(reader_ct);  // q placeholder when synthesize_gate
    // Keep the reader CT layout identical to the fused factory. This accessor is unused when fused_ring is off.
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    // multicast: on/off per direction (q_mcast_on covers q and w) then the 6 semaphore ids.
    reader_ct.push_back(k_mcast_on);
    reader_ct.push_back(q_mcast_on);
    reader_ct.push_back(k_send_sem);
    reader_ct.push_back(k_recv_sem);
    reader_ct.push_back(k_valid_sem);
    reader_ct.push_back(q_send_sem);
    reader_ct.push_back(q_recv_sem);
    reader_ct.push_back(q_valid_sem);
    // Fused single-head: reader reads q+w FIRST (the matmul gate needs them), then streams k (when no mcast).
    reader_ct.push_back(fuse_single ? 1u : 0u);
    reader_ct.push_back(fused_stream_k ? 1u : 0u);        // fused: stream k (no mcast) vs whole mcast block
    reader_ct.push_back(args.synthesize_gate ? 1u : 0u);  // fill cb_w with gate_scale in L1 vs read DRAM
    reader_ct.push_back(gate_scale_bits);                 // bf16 pair, the in-kernel gate fill value
    // invP KEY remap: keyed on key_stripes()/key_stripe_chunk(), NOT block_cyclic->{sp, chunk_local} -- under KV
    // dedup the keys are striped finer than the queries, and the causal geometry above stays on the query pair.
    const auto block_cyclic_ct = [&args, Tt]() {
        std::array<uint32_t, 5> ct{0, 1, 1, 0, 0};
        if (!args.has_block_cyclic()) {
            return ct;
        }
        const uint32_t sp = args.key_stripes();
        const uint32_t chunk_local = args.key_stripe_chunk() / tt::constants::TILE_WIDTH;
        ct = {
            1,
            chunk_local,
            sp,
            (Tt / sp) - chunk_local,
            chunk_local * (sp - 1),
        };
        return ct;
    }();
    reader_ct.insert(reader_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    // Keep the shared reader's full-mesh rank-mapping CT tail canonical for the classic path.
    reader_ct.insert(reader_ct.end(), {0u, 0u, 0u, 0u});
    reader_ct.push_back(0u);  // partial all-gather readiness off (non-fused path)
    reader_ct.push_back(1u);  // unused physical SP size
    // Trace-safe metadata blocks. The SAME reader binary serves both factories and every block is fixed
    // width (a placeholder q accessor when absent), pushed AFTER partial readiness and #55617's physical SP
    // size, matching the reader's meta_ct_base. Chunk start: flag, rt base, two mailbox CBs, Sq,
    // rotation-exact flag, key-stripe split, chunk extent, accessor.
    reader_ct.push_back(has_meta ? 1u : 0u);
    reader_ct.push_back(indexer_common::reader::ChunkMetadata);
    reader_ct.push_back(cb_meta_derived);
    reader_ct.push_back(cb_meta_writer);
    reader_ct.push_back(has_meta ? Sq : 0u);
    // Same predicate device_causal_geometry() uses, so the reader picks the host's causal branch.
    reader_ct.push_back(has_meta && rotation_exact_sp_geometry(args) ? 1u : 0u);
    reader_ct.push_back(has_meta ? args.key_stripe_split : 1u);
    reader_ct.push_back(has_meta ? chunk_extent_for(args, q) / tt::constants::TILE_WIDTH : 0u);
    tt::tt_metal::TensorAccessorArgs(has_meta ? *tensors.chunk_start_idx_tensor->buffer() : *q.buffer())
        .append_to(reader_ct);
    // Cache-slot select: rt base, k pages per slot, landing CB, slot count, accessor. Presence is the runtime
    // address (0 = scalar path), so no flag. On this path k itself is the multi-slot cache.
    reader_ct.push_back(indexer_common::reader::SlotMetadata);
    reader_ct.push_back(has_slot_meta ? Tt * Dt : 0u);
    reader_ct.push_back(cb_meta_slot);
    reader_ct.push_back(has_slot_meta ? static_cast<uint32_t>(k.logical_shape()[0]) : 0u);
    tt::tt_metal::TensorAccessorArgs(has_slot_meta ? *tensors.cache_batch_idx_tensor->buffer() : *q.buffer())
        .append_to(reader_ct);
    // Real-token end: rt base + accessor; presence is the runtime address (0 = uncapped).
    const bool has_valid_end = tensors.has_valid_end_metadata();
    reader_ct.push_back(indexer_common::reader::ValidEnd);
    tt::tt_metal::TensorAccessorArgs(has_valid_end ? *tensors.valid_end_tensor->buffer() : *q.buffer())
        .append_to(reader_ct);

    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.push_back(0u);                             // fused_ring off
    const uint32_t out_elem_bytes = out.element_size();  // bf16 today
    // row-major page = one output row: T scores, or nblocks block-scores when pooling.
    const uint32_t out_row_elems = block_pool ? nblocks : T;
    writer_ct.push_back(out_row_elems * out_elem_bytes);
    writer_ct.push_back(0u);  // shard-major mapping off
    writer_ct.push_back(1u);  // unused block-cyclic run width
    writer_ct.push_back(1u);  // unused logical key stripe count
    writer_ct.push_back(1u);  // unused physical SP size
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);
    writer_ct.push_back(has_meta ? 1u : 0u);  // causal bounds from the reader's mailbox
    writer_ct.push_back(cb_meta_writer);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);             // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);               // k-columns batched per mode switch in the full-strip path
    compute_ct.push_back(args.apply_relu ? 1u : 0u);  // 1 = relu(q.kT) (DSA/GLM), 0 = raw q.kT (M3)
    // Fused single-head fast path: plane_heads==1, no ReLU, bf16 q (the gate is applied to q in place, so
    // bfp8 q falls back). GLM/DSv32 and bfp8-q MSA fall back, byte-identical.
    compute_ct.push_back(fuse_single ? 1u : 0u);
    compute_ct.push_back(fused_stream_k ? 1u : 0u);  // fused: incremental k wait (stream) vs whole-chunk
    compute_ct.push_back(0u);                        // fused_ring off
    compute_ct.push_back(0u);                        // shard-major block-cyclic mapping off
    compute_ct.push_back(1u);                        // unused block-cyclic run width
    compute_ct.push_back(1u);                        // unused logical key stripe count
    compute_ct.push_back(1u);                        // unused physical SP size
    compute_ct.push_back(has_meta ? 1u : 0u);        // causal bounds from the reader's mailbox
    compute_ct.push_back(cb_meta_derived);

    const std::unordered_map<std::string, uint32_t> schedule_args{
        {"schedule_blocks", num_blocks},
        {"schedule_group_rows", group_rows},
        {"schedule_groups", num_groups},
        {"schedule_max_bands", max_bands},
        {"schedule_ring_size", 1u},
        {"schedule_units", band_count},
        {"schedule_cols", cols_used},
        {"schedule_rotate", 0u}};
    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        kdir + "reader_indexer_score.cpp",
        core_ranges,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct, {}, schedule_args));
    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        kdir + "writer_indexer_score.cpp",
        core_ranges,
        tt::tt_metal::WriterDataMovementConfig(writer_ct, {}, schedule_args));
    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        kdir + "compute_indexer_score.cpp",
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct,
            .named_compile_args = schedule_args});

    // One core identity per kernel; geometry and multicast axes are shared.
    // Indexed-cache k page offset + valid kv_len, baked at miss and re-applied each dispatch (both hash-excluded).
    // On the metadata path the causal bounds below are placeholders (chunk_start_idx is inert); compute and
    // the writer take the reader's derivation from their mailboxes instead.
    const uint32_t kv_len_tiles = persistent_cache_args(args, k).kv_len_tiles;
    const auto reader_common = classic_reader_values(args, tensors, device_index, tp_index);
    std::vector<uint32_t> reader_values(reader_common.begin(), reader_common.end());
    append_multicast_axes(reader_values, phys);
    tt::tt_metal::SetCommonRuntimeArgs(program, reader_id, reader_values);
    tt::tt_metal::SetCommonRuntimeArgs(
        program, compute_id, {kv_len_tiles, chunk_t, geom.straddle_q_tile, geom.straddle_jump_tiles});
    tt::tt_metal::SetCommonRuntimeArgs(
        program,
        writer_id,
        {out.buffer()->address(), kv_len_tiles, chunk_t, geom.straddle_q_tile, geom.straddle_jump_tiles});
    for (uint32_t row = 0; row < rows_used; ++row) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            const CoreCoord core{col, row};
            const std::vector<uint32_t> runtime{row * cols_used + col};
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, runtime);
            tt::tt_metal::SetRuntimeArgs(program, compute_id, core, runtime);
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, runtime);
        }
    }

    return {
        std::move(program),
        IndexerScoreSharedVariables{
            .reader_kernel = reader_id,
            .compute_kernel = compute_id,
            .writer_kernel = writer_id,
            .device_index = device_index,
            .tp_index = tp_index}};
}

IndexerScoreProgramFactory::cached_mesh_workload_t IndexerScoreProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    // One program per coordinate (each device derives its own chunk_start); a single device loops once at index 0.
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            const ttnn::MeshCoordinateRange single{coord, coord};
            auto cached = create_at(args, coord, tensors, out);
            shared_variables[single] = cached.shared_variables;
            mesh_workload.add_program(single, std::move(cached.program));
        }
    }
    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void IndexerScoreProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const operation_attributes_t& args,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    // Re-apply all hash-excluded runtime values on a hit: buffer addresses (including the trace metadata
    // tensors), cache_batch_idx / kv_len / the slot-recomposition layer terms, and chunk_start (per-coordinate,
    // from the stored device_index).
    const uint32_t Sq = tensors.q.logical_shape()[2];
    const uint32_t kv_len_tiles = persistent_cache_args(args, tensors.k).kv_len_tiles;
    const uint32_t out_address = out.buffer()->address();
    for (auto& [range, shared] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.reader_kernel);
        auto& compute_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.compute_kernel);
        auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.writer_kernel);
        const auto reader_values = classic_reader_values(args, tensors, shared.device_index, shared.tp_index);
        const auto geom = device_causal_geometry(args, shared.device_index, shared.tp_index, Sq);
        const std::array<uint32_t, indexer_common::compute::Count> causal_values = {
            kv_len_tiles, geom.chunk_start_tiles, geom.straddle_q_tile, geom.straddle_jump_tiles};
        std::copy(reader_values.begin(), reader_values.end(), reader_args.data());
        std::copy(causal_values.begin(), causal_values.end(), compute_args.data());
        writer_args[indexer_common::writer::Output] = out_address;
        std::copy(causal_values.begin(), causal_values.end(), writer_args.data() + indexer_common::writer::KvLength);
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program
