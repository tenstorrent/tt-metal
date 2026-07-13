// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <algorithm>
#include <array>
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
#include "ttnn/operations/ccl/ccl_common.hpp"    // get_linearized_index_from_physical_coord
#include "indexer_score_host_common.hpp"         // shared causal geometry / device index / persistent-cache args
#include "kernels/indexer_score_cb.hpp"          // shared host/device CB-index argument layout (CbArg)
#include "kernels/indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace ttnn::operations::experimental::indexer_score::program {

// Runtime-arg slots, shared by create_at()/override_runtime_arguments() and matched positionally by the
// kernels. Reader: q,k,w addrs then schedule(6) + mcast(2x8) + persistent-cache(2); compute: schedule(6)
// then kv_len_tiles[6] + chunk_start_tiles[7]; writer: out addr then schedule(6) + kv_len_tiles.
namespace rt_arg {
constexpr uint32_t reader_q_addr = 0;
constexpr uint32_t reader_k_addr = 1;
constexpr uint32_t reader_w_addr = 2;
constexpr uint32_t compute_chunk_start_tiles = 7;  // after 6 sched scalars + kv_len[6]
constexpr uint32_t compute_straddle_q_tile = 8;    // mid-slab boundary-chip diagonal jump (q-tile-row)
constexpr uint32_t compute_straddle_jump_tiles = 9;
constexpr uint32_t writer_out_addr = 0;
// Persistent-cache args, appended after each kernel's schedule/mcast args (hash-excluded, re-patched on a hit).
constexpr uint32_t reader_num_scalars = 3 + 6;  // q/k/w addrs + schedule {row_group0..max_bands}
constexpr uint32_t mcast_args_per_dir = 8;      // role, rect (xs,ys,xe,ye), sender (sx,sy), ndst
constexpr uint32_t reader_num_mcast_dirs = 2;   // K column, then Q/W row
constexpr uint32_t reader_k_batch_offset = reader_num_scalars + reader_num_mcast_dirs * mcast_args_per_dir;  // 25
constexpr uint32_t reader_kv_len_tiles = reader_k_batch_offset + 1;                                          // 26
constexpr uint32_t compute_kv_len_tiles = 6;          // after the 6 schedule scalars {row_group0..max_bands}
constexpr uint32_t writer_kv_len_tiles = 1 + 6;       // out_addr + the 6 schedule scalars {row_group0..max_bands}
constexpr uint32_t writer_chunk_start_tiles = 1 + 7;  // after out_addr + 6 sched scalars + kv_len[7]; match writer
constexpr uint32_t writer_straddle_q_tile = 1 + 8;    // mid-slab forced-local block jump (block-pool only)
constexpr uint32_t writer_straddle_jump_tiles = 1 + 9;
}  // namespace rt_arg

// Patch one runtime-arg slot on a program-cache hit, asserting the slot exists.
inline void patch_arg(tt::tt_metal::RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(index < args.size(), "indexer_score override: {} index {} >= args size {}", name, index, args.size());
    args[index] = value;
}

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
    // arg, so the binary is identical across coords and steps. tp_index = its rank along seq_subshard_axis
    // (the 2D SP×TP query sub-shard); 0 when not sub-sharded or single-device.
    const uint32_t device_index = device_index_for(args, coord, q);
    const uint32_t tp_index =
        (args.seq_subshard_axis.has_value() && q.device_storage().get_coords().size() > 1)
            ? ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.seq_subshard_axis)
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
    const uint32_t num_cores = rows_used * cols_used;
    // Phase-stack count: groups dealt round-robin onto the group_rows rows (1 when group_rows == group_count).
    const uint32_t num_groups = group_count / group_rows;

    // 2-D band deal: bands split into num_blocks contiguous blocks (front blocks get the remainder), each
    // block split across cols_used columns (front columns get the remainder). Indexed [block][col]; with
    // num_blocks==1 this is exactly the original per-column split over the whole band range.
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

    // Common args: 9 dims then the CB indices in CbArg order. chunk_t is NOT here (per-device runtime arg).
    std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, QC, KC, HB, G, block_tiles};
    common_ct.insert(common_ct.end(), cb_id.begin(), cb_id.end());

    // MSA synthesizes the constant gate in-kernel (no weights tensor / no fill op): the reader fills cb_w
    // with `gate_scale` instead of reading DRAM, and the weights accessor below is the unused q placeholder.
    // Pack the scale into a bf16 pair (two values per word) for the reader's word-wise fill.
    const uint16_t gate_scale_bf16 = static_cast<uint16_t>(__builtin_bit_cast(uint32_t, args.gate_scale) >> 16);
    const uint32_t gate_scale_bits = (static_cast<uint32_t>(gate_scale_bf16) << 16) | gate_scale_bf16;

    std::vector<uint32_t> reader_ct = common_ct;
    tt::tt_metal::TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*w.buffer()).append_to(reader_ct);  // q placeholder when synthesize_gate
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
    const auto block_cyclic_ct = [&args, Tt]() {
        std::array<uint32_t, 5> ct{0, 1, 1, 0, 0};
        if (!args.has_block_cyclic()) {
            return ct;
        }
        const uint32_t sp = args.block_cyclic->sp;
        const uint32_t chunk_local = args.block_cyclic->chunk_local / tt::constants::TILE_WIDTH;
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

    std::vector<uint32_t> writer_ct = common_ct;
    const uint32_t out_elem_bytes = out.element_size();  // bf16 today
    // row-major page = one output row: T scores, or nblocks block-scores when pooling.
    const uint32_t out_row_elems = block_pool ? nblocks : T;
    writer_ct.push_back(out_row_elems * out_elem_bytes);
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);             // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);               // k-columns batched per mode switch in the full-strip path
    compute_ct.push_back(args.apply_relu ? 1u : 0u);  // 1 = relu(q.kT) (DSA/GLM), 0 = raw q.kT (M3)
    // Fused single-head fast path: plane_heads==1, no ReLU, bf16 q (the gate is applied to q in place, so
    // bfp8 q falls back). GLM/DSv32 and bfp8-q MSA fall back, byte-identical.
    compute_ct.push_back(fuse_single ? 1u : 0u);
    compute_ct.push_back(fused_stream_k ? 1u : 0u);  // fused: incremental k wait (stream) vs whole-chunk

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
    auto reader_id = tt::tt_metal::CreateKernel(
        program, kdir + "reader_indexer_score.cpp", core_ranges, tt::tt_metal::ReaderDataMovementConfig(reader_ct));
    auto writer_id = tt::tt_metal::CreateKernel(
        program, kdir + "writer_indexer_score.cpp", core_ranges, tt::tt_metal::WriterDataMovementConfig(writer_ct));
    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        kdir + "compute_indexer_score.cpp",
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct});

    // Per-core args: schedule {row_group0, group_stride, num_groups, band0, num_bands, max_bands} then
    // (reader only) the K-column + Q/W-row mcast 8-tuples (role, rect xs/ys/xe/ye, sender sx/sy, ndst). The
    // mcast rects are fixed per core; only the data changes per phase.
    const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
    // Indexed-cache k page offset + valid kv_len, baked at miss and re-applied each dispatch (both hash-excluded).
    const auto [k_batch_page_offset, kv_len_tiles] = persistent_cache_args(args, k);
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t row = 0; row < rows_used; ++row) {
        // Q/W row mcast rect + diagonal sender (shared with the fused factory).
        const auto qb = q_mcast_bbox(phys, row, cols_used);
        const uint32_t q_xs = qb.xs, q_xe = qb.xe, q_py = qb.py, q_diag = qb.diag_col;
        const CoreCoord q_sender = qb.sender;
        // This row's band-chunk block and its row base within the grid. The k-mcast spans only the block's
        // group_rows rows; row % group_rows is the group this row computes (same in every block).
        const uint32_t block = row / group_rows;
        const uint32_t block_base = block * group_rows;
        for (uint32_t col = 0; col < cols_used; ++col) {
            // K column mcast rect + block-top sender (shared with the fused factory).
            const auto kb = k_mcast_bbox(phys, block_base, col, group_rows);
            const uint32_t k_ys = kb.ys, k_ye = kb.ye, k_px = kb.px;
            const CoreCoord k_sender = kb.sender;

            const CoreCoord core{col, row};
            cores.push_back(core);
            // max_bands is uniform (global widest cell); streaming pads its band loop to it.
            const std::array<uint32_t, 6> sched = {
                row % group_rows, group_rows, num_groups, band_start[block][col], band_size[block][col], max_bands};

            std::vector<uint32_t> reader_rt = {q.buffer()->address(), k.buffer()->address(), w.buffer()->address()};
            reader_rt.insert(reader_rt.end(), sched.begin(), sched.end());
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
            // K column: per row-block, sender is the block's top row, receivers the rest of the block;
            // vertical rect spanning only the block's group_rows rows.
            push_mcast_dir(
                k_mcast_on ? (row == block_base ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                k_px,
                k_ys,
                k_px,
                k_ye,
                k_sender,
                group_rows - 1);
            // Q/W row: sender on the diagonal column, receivers the rest of the row; horizontal rect.
            push_mcast_dir(
                q_mcast_on ? (col == q_diag ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                q_xs,
                q_py,
                q_xe,
                q_py,
                q_sender,
                cols_used - 1);
            // Persistent-cache args last (slots reader[25,26]).
            reader_rt.push_back(k_batch_page_offset);
            reader_rt.push_back(kv_len_tiles);
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, reader_rt);
            // compute: schedule[0-5], kv_len_tiles[6], chunk_start_tiles[7], straddle[8,9] (hash-excluded runtime).
            std::vector<uint32_t> compute_rt(sched.begin(), sched.end());
            compute_rt.push_back(kv_len_tiles);              // slot [6]
            compute_rt.push_back(chunk_t);                   // slot [7]
            compute_rt.push_back(geom.straddle_q_tile);      // slot [8], mid-slab boundary-chip diagonal jump
            compute_rt.push_back(geom.straddle_jump_tiles);  // slot [9]
            tt::tt_metal::SetRuntimeArgs(program, compute_id, core, compute_rt);
            std::vector<uint32_t> writer_rt = {out.buffer()->address()};
            writer_rt.insert(writer_rt.end(), sched.begin(), sched.end());
            writer_rt.push_back(kv_len_tiles);              // slot [7], after out_addr + the 6 schedule scalars
            writer_rt.push_back(chunk_t);                   // slot [8], per-device chunk-start (tiles); forced-local
            writer_rt.push_back(geom.straddle_q_tile);      // slot [9], mid-slab forced-local block jump (pool only)
            writer_rt.push_back(geom.straddle_jump_tiles);  // slot [10]
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, writer_rt);
        }
    }

    return {
        std::move(program),
        IndexerScoreSharedVariables{
            .reader_kernel = reader_id,
            .compute_kernel = compute_id,
            .writer_kernel = writer_id,
            .worker_cores = cores,
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
    // Re-apply all hash-excluded runtime values on a hit: buffer addresses, cache_batch_idx / kv_len, and
    // chunk_start (per-coordinate, from the stored device_index).
    const uint32_t Sq = tensors.q.logical_shape()[2];
    const auto [k_batch_page_offset, kv_len_tiles] = persistent_cache_args(args, tensors.k);
    for (auto& [range, shared] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel);
        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, shared.compute_kernel);
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel);
        const auto geom = device_causal_geometry(args, shared.device_index, shared.tp_index, Sq);
        const uint32_t chunk_t = geom.chunk_start_tiles;
        for (const auto& core : shared.worker_cores) {
            auto& reader_rt = reader_args[core.x][core.y];
            patch_arg(reader_rt, rt_arg::reader_q_addr, tensors.q.buffer()->address(), "reader.q_addr");
            patch_arg(reader_rt, rt_arg::reader_k_addr, tensors.k.buffer()->address(), "reader.k_addr");
            patch_arg(reader_rt, rt_arg::reader_w_addr, tensors.weights.buffer()->address(), "reader.w_addr");
            patch_arg(reader_rt, rt_arg::reader_k_batch_offset, k_batch_page_offset, "reader.k_batch_offset");
            patch_arg(reader_rt, rt_arg::reader_kv_len_tiles, kv_len_tiles, "reader.kv_len_tiles");
            patch_arg(compute_args[core.x][core.y], rt_arg::compute_kv_len_tiles, kv_len_tiles, "compute.kv_len_tiles");
            patch_arg(compute_args[core.x][core.y], rt_arg::compute_chunk_start_tiles, chunk_t, "compute.chunk_start");
            patch_arg(
                compute_args[core.x][core.y],
                rt_arg::compute_straddle_q_tile,
                geom.straddle_q_tile,
                "compute.straddle_q_tile");
            patch_arg(
                compute_args[core.x][core.y],
                rt_arg::compute_straddle_jump_tiles,
                geom.straddle_jump_tiles,
                "compute.straddle_jump_tiles");
            patch_arg(writer_args[core.x][core.y], rt_arg::writer_out_addr, out.buffer()->address(), "writer.out_addr");
            patch_arg(writer_args[core.x][core.y], rt_arg::writer_kv_len_tiles, kv_len_tiles, "writer.kv_len_tiles");
            patch_arg(writer_args[core.x][core.y], rt_arg::writer_chunk_start_tiles, chunk_t, "writer.chunk_start");
            patch_arg(
                writer_args[core.x][core.y],
                rt_arg::writer_straddle_q_tile,
                geom.straddle_q_tile,
                "writer.straddle_q_tile");
            patch_arg(
                writer_args[core.x][core.y],
                rt_arg::writer_straddle_jump_tiles,
                geom.straddle_jump_tiles,
                "writer.straddle_jump_tiles");
        }
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program
