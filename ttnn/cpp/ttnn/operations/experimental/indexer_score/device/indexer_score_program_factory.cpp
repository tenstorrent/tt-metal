// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>         // log_info: per-program schedule/mcast summary
#include <tt-metalium/mesh_workload.hpp>
#include "hostdevcommon/kernel_structs.h"  // tt::CBIndex

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"    // get_linearized_index_from_physical_coord
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
constexpr uint32_t compute_chunk_start_tiles = 7;  // after the 6 sched scalars + kv_len[6]; match the compute kernel
constexpr uint32_t writer_out_addr = 0;
// Persistent-cache args, appended after the schedule/mcast args of each kernel (excluded from the hash,
// re-patched on a cache hit). Reader: 3 addrs + 6 schedule + 2 mcast dirs * 8 = 25, then the two below.
constexpr uint32_t reader_num_scalars = 3 + 6;  // q/k/w addrs + schedule {row_group0..max_bands}
constexpr uint32_t mcast_args_per_dir = 8;      // role, rect (xs,ys,xe,ye), sender (sx,sy), ndst
constexpr uint32_t reader_num_mcast_dirs = 2;   // K column, then Q/W row
constexpr uint32_t reader_k_batch_offset = reader_num_scalars + reader_num_mcast_dirs * mcast_args_per_dir;  // 25
constexpr uint32_t reader_kv_len_tiles = reader_k_batch_offset + 1;                                          // 26
constexpr uint32_t compute_kv_len_tiles = 6;     // after the 6 schedule scalars {row_group0..max_bands}
constexpr uint32_t writer_kv_len_tiles = 1 + 6;  // out_addr + the 6 schedule scalars {row_group0..max_bands}
}  // namespace rt_arg

// Per-device chunk_start (in tiles): base + rank*Sq, /TILE_WIDTH (the per-rank stride is exactly the
// per-device query count Sq). Shared by create_at (derives device_index from the coordinate) and
// override (reuses the stored device_index).
inline uint32_t chunk_start_tiles_for(const operation_attributes_t& args, uint32_t device_index, uint32_t Sq) {
    return (args.chunk_start_idx + device_index * Sq) / tt::constants::TILE_WIDTH;
}

// This device's linearized SP-ring index; 0 on a single device (no coordinate lookup needed).
inline uint32_t device_index_for(
    const operation_attributes_t& args, const ttnn::MeshCoordinate& coord, const Tensor& q) {
    if (q.device_storage().get_coords().size() <= 1) {
        return 0;
    }
    return ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.cluster_axis);
}

// Patch one runtime-arg slot on a program-cache hit, asserting the slot exists.
inline void patch_arg(tt::tt_metal::RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(index < args.size(), "indexer_score override: {} index {} >= args size {}", name, index, args.size());
    args[index] = value;
}

// The two non-hashed runtime args derived from k's (hashed) shape + the optionals. Single source for both
// create() (bakes them at miss) and override_runtime_arguments() (re-patches them on a hit) -- a divergence
// would silently mis-patch the slot/kv_len on a cache hit.
struct PersistentCacheArgs {
    uint32_t k_batch_page_offset;  // cache_batch_idx * Tt * Dt; 0 when not indexed
    uint32_t kv_len_tiles;         // valid key prefix in tiles; full Tt when kv_len unset
};
inline PersistentCacheArgs persistent_cache_args(const operation_attributes_t& attrs, const Tensor& k) {
    const auto& shape = k.logical_shape();
    const uint32_t Tt = shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t Dt = shape[3] / tt::constants::TILE_WIDTH;
    return {
        .k_batch_page_offset = attrs.cache_batch_idx.value_or(0) * Tt * Dt,
        .kv_len_tiles = attrs.kv_len.value_or(shape[2]) / tt::constants::TILE_WIDTH};
}

// Banded-product schedule: the work space (group_count q-row-groups x band_count k-bands) tiles onto
// a rows_used x cols_used core rectangle -- groups -> rows (q/w multicast along a row), bands ->
// columns (k multicast down a column). One (group, band) cell = one work unit of QC q-tile-rows x
// up-to-KC k-tiles. Heads stream in HB-head groups; the causal masked suffix is stamped to -inf by
// compute (zeros unsafe: gates can be negative). See indexer_score_work_split.hpp for the grid map.
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

    // This device's SP-ring index and chunk_start (tiles), derived from the mesh coordinate (stride = Sq).
    // chunk_t is a compute RUNTIME arg (not compile-time), so the binary is identical across coords and steps.
    const uint32_t device_index = device_index_for(args, coord, q);
    const uint32_t chunk_t = chunk_start_tiles_for(args, device_index, Sq);

    const uint32_t Sqt = Sq / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = T / tt::constants::TILE_WIDTH;
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;

    // Work-unit knobs, converted from elements to tiles / heads.
    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);

    // Compute knobs from the resolved compute config. math_fidelity defaults to the dtype-derived
    // choice (bf16 -> HiFi2, both bfp8 -> LoFi) but a caller can override it; validate guarantees
    // fp32_dest_acc_en==false / dst_full_sync_en==false (the bf16-DEST half-sync layout the custom
    // blocked bcast-col MUL LLK + packer path are validated for).
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(q.device()->arch(), args.compute_kernel_config);

    // qk matmul subblock: heads are output rows, k column is 1 tile wide (SDPA-style), so only the
    // subblock height (head rows) is needed.
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;  // half-sync, as in sdpa_program_factory
    const uint32_t qk_subblock_h = ttnn::prim::detail::determine_largest_subblock_size(HB, 1, dst_size).first;
    TT_FATAL(HB % qk_subblock_h == 0, "head group {} must be divisible by qk_subblock_h={}", HB, qk_subblock_h);

    // QC/KC/HB are verbatim from the config -- no auto-tune; the caller owns the perf trade-off (see
    // glx_config() in the test). An oversized config is not clamped; it fails at CB allocation.

    // ---- banded-product schedule -------------------------------------------------------------
    // Work space = group_count q-row-groups x band_count k-bands, tiled onto a rows_used x cols_used
    // rectangle: groups -> rows (q/w mcast along a row), bands -> columns (k mcast down a column). Rows
    // phase-stack groups when group_count > grid_y; each column owns a contiguous chunk of bands.
    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);  // ceil(Tt/KC)
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t grid_x = grid.x, grid_y = grid.y;

    const uint32_t rows_used = rows_for_groups(group_count, grid_y);
    const uint32_t cols_used = cols_for_bands(band_count, grid_x);
    const uint32_t num_cores = rows_used * cols_used;

    // Even band split across the used columns: the first (band_count % cols_used) columns get one extra.
    std::vector<uint32_t> col_band_start(cols_used), col_band_size(cols_used);
    {
        const uint32_t bands_per_col = band_count / cols_used, extra = band_count % cols_used;
        uint32_t off = 0;
        for (uint32_t col = 0; col < cols_used; ++col) {
            col_band_size[col] = bands_per_col + (col < extra ? 1u : 0u);
            col_band_start[col] = off;
            off += col_band_size[col];
        }
    }
    // rows_used divides group_count, so every row runs the same num_groups (group = y + p*rows_used).
    // Uniformity keeps every column's k-mcast in lockstep.
    const uint32_t num_groups = group_count / rows_used;
    // Widest column's band count: the streaming q-mcast pad target. Streaming re-reads q per output tile,
    // so the uneven per-column band counts would desync the row's q-mcast; the kernels pad each row to
    // max_bands with dummy q-only "phantom" bands (band-independent data) to keep the rendezvous uniform.
    const uint32_t max_bands = (band_count + cols_used - 1) / cols_used;

    const CoreRange core_rect(CoreCoord{0, 0}, CoreCoord{cols_used - 1, rows_used - 1});
    const CoreRangeSet core_ranges(core_rect);

    // Physical coords of the used rectangle, indexed [row][col], for the mcast bounding boxes below.
    std::vector<std::vector<CoreCoord>> phys(rows_used, std::vector<CoreCoord>(cols_used));
    for (uint32_t row = 0; row < rows_used; ++row) {
        for (uint32_t col = 0; col < cols_used; ++col) {
            phys[row][col] = q.device()->worker_core_from_logical_core(CoreCoord{col, row});
        }
    }

    // k-mcast needs >1 row down a column; q/w-mcast needs >1 column along a row. Both are HB-independent:
    // streaming keeps q-mcast via the phantom-band pad (see max_bands above), so neither gates on HB.
    const uint32_t k_mcast_on = (rows_used > 1) ? 1u : 0u;
    const uint32_t q_mcast_on = (cols_used > 1) ? 1u : 0u;

    // 3 semaphores per active direction: send (receivers ready), recv (sender relays valid in), valid
    // (constant 1, relay source). Mirrors SDPA chain_link's handshake.
    const uint32_t k_send_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;

    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);

    // q (srcB) and k (srcA) are matmul inputs; either may be bfp8_b to halve its DRAM/L1 footprint
    // (w stays bf16). The format flows into the CB; the compute kernel needs no change.
    const bool q_is_bfp8 = q.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const bool k_is_bfp8 = k.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const tt::DataFormat q_fmt = q_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const tt::DataFormat k_fmt = k_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const uint32_t q_tile = tt::tile_size(q_fmt);
    const uint32_t k_tile = tt::tile_size(k_fmt);

    // Continuous CB indices, allocated on demand: each make_cb claims the next free index (c_0, c_1, ...)
    // and records it under its CbArg slot. The whole array is forwarded to the kernels as compile-time
    // args (CbArg is the shared slot order; see indexer_score_cb.hpp), so every buffer is resolved through
    // this one allocation and indices can never drift host<->device.
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

    // One-line schedule/mcast summary (per program-cache miss) so profiling can read the active
    // multicast directions without decoding compile-time args.
    log_debug(
        tt::LogOp,
        "indexer_score schedule: G={} U={} grid={}x{} rows_used={} cols_used={} num_groups={} "
        "max_bands={} stream_heads={} k_mcast={} q_mcast={}",
        group_count,
        band_count,
        grid_x,
        grid_y,
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
    // cb_qk buffers a batch of the group's relu(q.kT) tiles so compute runs that batch's matmuls then
    // its mul+accumulates, hoisting the matmul<->eltwise reinit out of the per-head-pass loop.
    // QC==1 has spare L1 -> batch the whole group; QC>1 doubles cb_q/cb_w, so cap at 32.
    const uint32_t qk_batch_cap = (QC == 1) ? HB : 32u;
    const uint32_t qk_batch_heads = std::min<uint32_t>(HB, qk_batch_cap);  // multiple of qk_subblock_h
    // The compute kernel walks HB in qk_batch_heads-sized chunks (chunk += qk_batch_heads), so HB must
    // be a whole multiple or the last chunk over-reads past the resident head group. Only reachable when
    // the 32-cap engages (HB > 32 && QC > 1); the deployed cases never hit it, but guard loudly.
    TT_FATAL(
        HB % qk_batch_heads == 0,
        "head_group {} not divisible by qk_batch_heads {} (QC>1 with HB>32); reduce head_group_size or q_chunk_size",
        HB,
        qk_batch_heads);
    // Full-strip path batches the whole k chunk's columns per matmul<->mul mode switch (one switch per
    // batch, not per output tile): w is column-independent and the whole k chunk is resident, so a row's
    // columns share one switch. Only when the group is a single chunk (qk_batch_heads == Hi) and the fast
    // strip is used (KC >= 2); cb_qk then holds KC * qk_batch_heads tiles and fails at allocation if the
    // caller's k_chunk_size is too large for L1 (the caller owns the knob trade-off).
    const bool single_chunk = (qk_batch_heads == Hi) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk) ? KC : 1u;
    make_cb(cb_qk_arg, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    // cb_out_strip holds a unit's untilized output. Uniform KC push/pop keeps the packer's KC-tile
    // reads contiguous (a non-uniform push would wrap the ring mid-strip).
    make_cb(cb_out_strip_arg, 2 * KC, tt::DataFormat::Float16_b, bf16_tile);
    // cb_acc_strip accumulates a whole unit's QC*KC strip, then all QC strips untilize under ONE
    // pack_untilize bracket (per-strip cost amortizes over QC*KC, not KC). max(2*KC, .) keeps the
    // QC<=2 double buffer and a whole multiple of the QC*KC batch so a uniform push never wraps mid-unit.
    make_cb(cb_acc_strip_arg, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // No up-front L1-fit guard: an oversized QC/KC/head_group config fails at CB allocation. The caller
    // owns the knob trade-off (see glx_config() in the test).

    // Common args: 7 dims then the CB indices in CbArg order (kernels read both from this shared base).
    // chunk_t is NOT here -- it is a per-device compute runtime arg (derived from the coordinate above).
    std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, QC, KC, HB};
    common_ct.insert(common_ct.end(), cb_id.begin(), cb_id.end());

    std::vector<uint32_t> reader_ct = common_ct;
    tt::tt_metal::TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*w.buffer()).append_to(reader_ct);
    // multicast: on/off per direction (q_mcast_on covers q and w) then the 6 semaphore ids, at fixed
    // offsets right after the three TensorAccessors.
    reader_ct.push_back(k_mcast_on);
    reader_ct.push_back(q_mcast_on);
    reader_ct.push_back(k_send_sem);
    reader_ct.push_back(k_recv_sem);
    reader_ct.push_back(k_valid_sem);
    reader_ct.push_back(q_send_sem);
    reader_ct.push_back(q_recv_sem);
    reader_ct.push_back(q_valid_sem);

    std::vector<uint32_t> writer_ct = common_ct;
    const uint32_t out_elem_bytes = out.element_size();  // from the output tensor's dtype (bf16 today)
    writer_ct.push_back(T * out_elem_bytes);             // row-major page = one full row of T scores
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);  // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);    // k-columns batched per mode switch in the full-strip path

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
    // (reader only) the K-column + Q/W-row mcast 8-tuples. role is a McastRole (none = per-core DRAM read); rect
    // (xs,ys,xe,ye) + sender (sx,sy) are physical NoC; ndst = #receivers. mcast rects are fixed per core
    // (one column for k, one row for q); only the data changes per group/band phase.
    const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
    // Indexed-cache k page offset and valid kv_len, baked here for the cache-miss build and re-applied each
    // dispatch in override_runtime_arguments (both excluded from the hash). The grid/work-split above stay
    // keyed on the hashed Tt; kv_len_tiles only narrows the per-cell columns the kernels touch.
    const auto [k_batch_page_offset, kv_len_tiles] = persistent_cache_args(args, k);
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t row = 0; row < rows_used; ++row) {
        // physical bbox of this row across the used columns (q/w mcast rect); py constant along the row.
        uint32_t q_xs = u32(phys[row][0].x), q_xe = u32(phys[row][0].x);
        for (uint32_t bbox_col = 0; bbox_col < cols_used; ++bbox_col) {
            q_xs = std::min<uint32_t>(q_xs, u32(phys[row][bbox_col].x));
            q_xe = std::max<uint32_t>(q_xe, u32(phys[row][bbox_col].x));
        }
        const uint32_t q_py = u32(phys[row][0].y);
        const uint32_t q_diag = std::min<uint32_t>(row, cols_used - 1);  // diagonal sender column
        const CoreCoord q_sender = phys[row][q_diag];
        for (uint32_t col = 0; col < cols_used; ++col) {
            // physical bbox of this column down the used rows (k mcast rect); px constant down the column.
            uint32_t k_ys = u32(phys[0][col].y), k_ye = u32(phys[0][col].y);
            for (uint32_t bbox_row = 0; bbox_row < rows_used; ++bbox_row) {
                k_ys = std::min<uint32_t>(k_ys, u32(phys[bbox_row][col].y));
                k_ye = std::max<uint32_t>(k_ye, u32(phys[bbox_row][col].y));
            }
            const uint32_t k_px = u32(phys[0][col].x);
            const CoreCoord k_sender = phys[0][col];

            const CoreCoord core{col, row};
            cores.push_back(core);
            // {row_group0, group_stride, num_groups, band0, num_bands, max_bands}. max_bands is uniform
            // (the row's widest column); the streaming reader/compute pad their band loop to it.
            const std::array<uint32_t, 6> sched = {
                row, rows_used, num_groups, col_band_start[col], col_band_size[col], max_bands};

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
            // K column: sender row 0, receivers rows [1, rows_used); vertical rect spanning the column.
            push_mcast_dir(
                k_mcast_on ? (row == 0 ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                k_px,
                k_ys,
                k_px,
                k_ye,
                k_sender,
                rows_used - 1);
            // Q/W row: sender on the diagonal column, receivers the rest of the row; horizontal rect.
            push_mcast_dir(
                q_mcast_on ? (col == q_diag ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                q_xs,
                q_py,
                q_xe,
                q_py,
                q_sender,
                cols_used - 1);
            // Persistent-cache args last (slots reader[25,26]): after the mcast tuples in both branches.
            reader_rt.push_back(k_batch_page_offset);
            reader_rt.push_back(kv_len_tiles);
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, reader_rt);
            // compute: schedule scalars[0-5], then kv_len_tiles[6] and chunk_start_tiles[7]. Both are
            // hash-excluded runtime values -- kv_len is per-dispatch, chunk_t is per-device (from the
            // coordinate) -- so distinct values reuse one compiled program.
            std::vector<uint32_t> compute_rt(sched.begin(), sched.end());
            compute_rt.push_back(kv_len_tiles);  // slot [6]
            compute_rt.push_back(chunk_t);       // slot [7]
            tt::tt_metal::SetRuntimeArgs(program, compute_id, core, compute_rt);
            std::vector<uint32_t> writer_rt = {out.buffer()->address()};
            writer_rt.insert(writer_rt.end(), sched.begin(), sched.end());
            writer_rt.push_back(kv_len_tiles);  // slot [7], after out_addr + the 6 schedule scalars
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
            .device_index = device_index}};
}

IndexerScoreProgramFactory::cached_mesh_workload_t IndexerScoreProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensors,
    tensor_return_value_t& out) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    // One program per coordinate: each device derives its own chunk_start from its coordinate. A single
    // device is a 1x1 mesh, so this loops once with index 0 (the single-chip / scalar path).
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
    // All hash-excluded runtime values re-applied on a cache hit: buffer addresses (uniform across the mesh),
    // cache_batch_idx / kv_len (per-dispatch), and chunk_start (per-coordinate, recomputed from the stored
    // device_index) -- so a hit patches all of them without recompiling.
    const uint32_t Sq = tensors.q.logical_shape()[2];
    const auto [k_batch_page_offset, kv_len_tiles] = persistent_cache_args(args, tensors.k);
    for (auto& [range, shared] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel);
        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, shared.compute_kernel);
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel);
        const uint32_t chunk_t = chunk_start_tiles_for(args, shared.device_index, Sq);
        for (const auto& core : shared.worker_cores) {
            auto& reader_rt = reader_args[core.x][core.y];
            patch_arg(reader_rt, rt_arg::reader_q_addr, tensors.q.buffer()->address(), "reader.q_addr");
            patch_arg(reader_rt, rt_arg::reader_k_addr, tensors.k.buffer()->address(), "reader.k_addr");
            patch_arg(reader_rt, rt_arg::reader_w_addr, tensors.weights.buffer()->address(), "reader.w_addr");
            patch_arg(reader_rt, rt_arg::reader_k_batch_offset, k_batch_page_offset, "reader.k_batch_offset");
            patch_arg(reader_rt, rt_arg::reader_kv_len_tiles, kv_len_tiles, "reader.kv_len_tiles");
            patch_arg(compute_args[core.x][core.y], rt_arg::compute_kv_len_tiles, kv_len_tiles, "compute.kv_len_tiles");
            patch_arg(compute_args[core.x][core.y], rt_arg::compute_chunk_start_tiles, chunk_t, "compute.chunk_start");
            patch_arg(writer_args[core.x][core.y], rt_arg::writer_out_addr, out.buffer()->address(), "writer.out_addr");
            patch_arg(writer_args[core.x][core.y], rt_arg::writer_kv_len_tiles, kv_len_tiles, "writer.kv_len_tiles");
        }
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program
