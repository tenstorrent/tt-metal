// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

namespace {
constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// CB index allocation (kept stable across kernels via named compile-time args).
constexpr uint32_t CB_IN0_X = tt::CBIndex::c_0;
constexpr uint32_t CB_IN1_GATE = tt::CBIndex::c_1;
constexpr uint32_t CB_IN1_UP = tt::CBIndex::c_2;
constexpr uint32_t CB_IN1_DOWN = tt::CBIndex::c_3;
constexpr uint32_t CB_GATE_INT = tt::CBIndex::c_4;
constexpr uint32_t CB_UP_INT = tt::CBIndex::c_5;
constexpr uint32_t CB_ACTIVATED = tt::CBIndex::c_6;
constexpr uint32_t CB_PARTIALS_GU = tt::CBIndex::c_7;
constexpr uint32_t CB_PARTIALS_D = tt::CBIndex::c_8;
constexpr uint32_t CB_OUT = tt::CBIndex::c_9;
constexpr uint32_t CB_COUNTS_SCRATCH = tt::CBIndex::c_10;
constexpr uint32_t CB_IDX_SCRATCH = tt::CBIndex::c_11;
constexpr uint32_t CB_IN0_DOWN_FULL = tt::CBIndex::c_12;
// Second gate/up matmul partials CB. With the fused gate+up phase, each
// K-block accumulates simultaneously into partials_gu (gate matmul) and
// partials_up (up matmul) using the SAME shared x K-block — one x read
// per K-block feeds both matmuls instead of two.
constexpr uint32_t CB_PARTIALS_UP = tt::CBIndex::c_13;
// Writer-only scratch for the device-side `start` (expert_region_offsets)
// page in direct-write mode. Allocated unconditionally (negligible L1) so
// the CB-index layout is stable across both write modes.
constexpr uint32_t CB_START_SCRATCH = tt::CBIndex::c_14;
}  // namespace

UnifiedRoutedExpertFfnProgramFactory::cached_program_t UnifiedRoutedExpertFfnProgramFactory::create(
    const UnifiedRoutedExpertFfnParams& op,
    const UnifiedRoutedExpertFfnInputs& t,
    Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_proj.padded_shape();
    const auto& down_shape = t.down_proj.padded_shape();

    const uint32_t M_tiles_full = x_shape[-2] / TILE;
    const uint32_t K_gate_tiles = x_shape[-1] / TILE;            // = N_gate K = emb / TILE
    const uint32_t N_gate_tiles_full = gate_shape[-1] / TILE;    // = hidden / TILE
    const uint32_t K_down_tiles = down_shape[-2] / TILE;         // = hidden / TILE
    const uint32_t N_down_tiles_full = down_shape[-1] / TILE;    // = emb / TILE

    // Blackhole compute grid is 13x10 worker cores; we use the bottom-left
    // 11x8 = 88 to leave headroom for dispatch and to give per_core_M /
    // per_core_N clean divisors of common chunk sizes (chunk_M_tiles values
    // {16, 24, ..., 64} all divide by GRID_Y=8). N-axis is rounded UP to a
    // multiple of GRID_X via ceil_div per_core_N. Phantom tiles past the
    // actual tensor dims (col 64-65 of hidden, col 224-230 of emb) are
    // zero-padded in the reader (zero-fill L1 instead of DRAM read).
    // Compute runs uniform per_core_N; writer skips DRAM writes past
    // actual_N. K dim of the down matmul is also padded to N_gate_padded so
    // the activated L1 mcast (one sender per K-block, sender = gx == kb)
    // covers exactly per_core_N_gu cols per step; activated cols past
    // actual_hidden are 0 (gate/up weight OOB zero-fill propagates through
    // silu and multiply).
    constexpr uint32_t GRID_X = 11;
    constexpr uint32_t GRID_Y = 8;
    const auto grid_size = t.x.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x >= GRID_X && grid_size.y >= GRID_Y,
        "unified_routed_expert_ffn: expected at least {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid_size.x,
        grid_size.y);
    const uint32_t chunk_M_tiles = op.chunk_M_tiles;
    const uint32_t per_core_M = chunk_M_tiles / GRID_Y;
    TT_FATAL(
        per_core_M * GRID_Y == chunk_M_tiles,
        "chunk_M_tiles ({}) must be divisible by GRID_Y ({})",
        chunk_M_tiles,
        GRID_Y);
    // M_tiles_full is NOT required to divide chunk_M_tiles. The kernel runs
    // ceil(M_tiles_full / chunk_M_tiles) chunks; the reader zero-fills L1
    // rows past min(count_tiles, M_tiles_full) in the last chunk; the writer
    // skips OOB writes for output rows >= M_tiles_full. Avoids the host-side
    // pad/slice round-trip in the composite for non-aligned M.
    const uint32_t num_chunks = (M_tiles_full + chunk_M_tiles - 1) / chunk_M_tiles;

    const uint32_t per_core_N_gu = (N_gate_tiles_full + GRID_X - 1) / GRID_X;
    const uint32_t per_core_N_d = (N_down_tiles_full + GRID_X - 1) / GRID_X;
    const uint32_t N_gate_tiles_padded = per_core_N_gu * GRID_X;
    const uint32_t K_down_tiles_padded = N_gate_tiles_padded;  // down K = gate N

    // With 11x8 = 88 cores, per_core_N_gu (= 6) and per_core_N_d (= 21) leave
    // ~250KB of headroom per core vs an 8x8 layout. Use it to double
    // in0_block_w_gu, halving the gate / up K-loop iteration count.
    const uint32_t in0_block_w_gu = 16;
    const uint32_t in0_block_w_d = per_core_N_gu;
    TT_FATAL(
        K_gate_tiles % in0_block_w_gu == 0,
        "K_gate_tiles ({}) must be divisible by in0_block_w_gu ({})",
        K_gate_tiles,
        in0_block_w_gu);
    TT_FATAL(
        K_down_tiles_padded % in0_block_w_d == 0,
        "K_down_tiles_padded ({}) must be divisible by in0_block_w_d ({})",
        K_down_tiles_padded,
        in0_block_w_d);
    (void)K_down_tiles;  // actual K_down; used by reader for OOB; suppress unused warning here

    // Subblock dims. DST tile-register file is 16 tiles wide; fp32_dest_acc_en
    // halves usable capacity (fp32 accumulator occupies two tile slots). With
    // fp32_dest_acc_en=false (bf16 dst), per-thread DST capacity is 8 tiles.
    constexpr uint32_t DST_CAPACITY = 8;
    const uint32_t gu_out_subblock_h = 1;
    uint32_t gu_sub_w = 1;
    for (uint32_t cand = DST_CAPACITY; cand >= 1; --cand) {
        if (per_core_N_gu % cand == 0) {
            gu_sub_w = cand;
            break;
        }
    }
    const uint32_t gu_out_subblock_w = gu_sub_w;
    TT_FATAL(
        gu_out_subblock_h * gu_out_subblock_w <= DST_CAPACITY,
        "gu subblock h*w ({}) exceeds dst capacity",
        gu_out_subblock_h * gu_out_subblock_w);
    const uint32_t d_out_subblock_h = 1;
    uint32_t d_sub_w = 1;
    for (uint32_t cand = DST_CAPACITY; cand >= 1; --cand) {
        if (per_core_N_d % cand == 0) {
            d_sub_w = cand;
            break;
        }
    }
    const uint32_t d_out_subblock_w = d_sub_w;

    // Phase-level numbers.
    const uint32_t gu_in0_num_subblocks = per_core_M / gu_out_subblock_h;
    const uint32_t gu_in1_num_subblocks = per_core_N_gu / gu_out_subblock_w;
    const uint32_t gu_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    const uint32_t gu_in0_subblock_num_tiles = gu_out_subblock_h * in0_block_w_gu;
    const uint32_t gu_in1_block_num_tiles = in0_block_w_gu * per_core_N_gu;
    const uint32_t gu_in1_block_w = per_core_N_gu;
    const uint32_t gu_num_blocks = K_gate_tiles / in0_block_w_gu;
    const uint32_t gu_out_block_num_tiles = per_core_M * per_core_N_gu;

    const uint32_t d_in0_num_subblocks = per_core_M / d_out_subblock_h;
    const uint32_t d_in1_num_subblocks = per_core_N_d / d_out_subblock_w;
    const uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    const uint32_t d_in0_subblock_num_tiles = d_out_subblock_h * in0_block_w_d;
    const uint32_t d_in1_block_num_tiles = in0_block_w_d * per_core_N_d;
    const uint32_t d_in1_block_w = per_core_N_d;
    const uint32_t d_num_blocks = K_down_tiles_padded / in0_block_w_d;
    const uint32_t d_out_block_num_tiles = per_core_M * per_core_N_d;

    // -------------------------- data formats / tile sizes -----------------
    const tt::DataFormat x_df = tt::tt_metal::datatype_to_dataformat_converter(t.x.dtype());
    const tt::DataFormat gate_df = tt::tt_metal::datatype_to_dataformat_converter(t.gate_proj.dtype());
    const tt::DataFormat up_df = tt::tt_metal::datatype_to_dataformat_converter(t.up_proj.dtype());
    const tt::DataFormat down_df = tt::tt_metal::datatype_to_dataformat_converter(t.down_proj.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    // Intermediate and partials share the same format — required by the
    // compute kernel's mm_init pattern (mm_init's 3rd arg drives the packer's
    // data-format config; mismatched formats need explicit pack reconfig that
    // the kernel doesn't do). Use bfp8_b for both: 1KB/tile is half the bf16
    // cost so we fit in L1 with both intermediates and partials sized to the
    // full per-core block.
    const tt::DataFormat intermed_df = tt::DataFormat::Bfp8_b;
    const tt::DataFormat partials_gu_df = tt::DataFormat::Float16_b;
    const tt::DataFormat partials_d_df = tt::DataFormat::Float16_b;

    const uint32_t x_tile_size = tt::tile_size(x_df);
    const uint32_t gate_tile_size = tt::tile_size(gate_df);
    const uint32_t up_tile_size = tt::tile_size(up_df);
    const uint32_t down_tile_size = tt::tile_size(down_df);
    const uint32_t out_tile_size = tt::tile_size(out_df);
    const uint32_t intermed_tile_size = tt::tile_size(intermed_df);
    const uint32_t partials_gu_tile_size = tt::tile_size(partials_gu_df);
    const uint32_t partials_d_tile_size = tt::tile_size(partials_d_df);

    // -------------------------- compute grid ------------------------------
    const CoreRange core_range({0, 0}, {GRID_X - 1, GRID_Y - 1});
    const CoreRangeSet core_range_set{core_range};

    auto* x_buffer = t.x.buffer();
    auto* gate_buffer = t.gate_proj.buffer();
    auto* up_buffer = t.up_proj.buffer();
    auto* down_buffer = t.down_proj.buffer();
    auto* counts_buffer = t.counts.buffer();
    auto* idx_buffer = t.global_expert_idx_table.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    // Direct-write mode: when expert_region_offsets is supplied, the writer
    // writes this expert's output straight into the shared output buffer at
    // start[global_id]/TILE tile-rows (fusing ttnn::insert). Otherwise the
    // writer targets tile row 0 of a per-expert buffer. The `start` accessor
    // is appended unconditionally so the writer's CT-arg layout is stable;
    // when not in direct-write mode it points at out_buffer and is never read.
    const bool direct_write = t.expert_region_offsets.has_value();
    auto* start_buffer = direct_write ? t.expert_region_offsets->buffer() : out_buffer;
    // dst_M_tiles bounds destination writes. Equals M_tiles_full when the
    // output matches x's shape; is the shared buffer's tile-row count in
    // direct-write mode.
    const uint32_t dst_M_tiles = tensor_return_value.padded_shape()[-2] / TILE;

    // -------------------------- semaphores --------------------------------
    // Weight-multicast semaphores for in1 (gate/up/down). Pattern: per
    // N-col group (gx fixed, gy=0..GRID_Y-1), one sender at gy=0 reads the
    // weight slice from DRAM and mcasts it to the other GRID_Y-1 cores in
    // the same column. Receivers atomic-inc `ready` on the sender to signal
    // "I'm ready"; sender waits for ready==GRID_Y-1, mcasts the block, then
    // mcast-sets `valid` to 1 on all receivers; receivers wait for valid==1.
    // Same sem pair is reused across gate/up/down phases (phases are
    // sequential, sem values reset between K-blocks).
    const uint32_t in1_ready_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t in1_valid_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // in0 (x) multicast within M-row groups: sender at (gx=0, gy) reads x
    // for that M-row, mcasts to (gx=1..GRID_X-1, gy). Used for phases 1 and
    // 2 (gate and up matmul) where every core in a row needs the same x
    // slice. Phase 4 uses cb_in0_down_full (sourced from DRAM scratch) so
    // doesn't use this pair.
    const uint32_t in0_ready_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t in0_valid_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // Activated multicast sems (phase 4): replace the DRAM scratch round-trip
    // with an L1 NoC mcast. For phase-4 K-block kb, sender = core at
    // (gx=kb, my_mt). Sender's reader mcasts its cb_activated block to all
    // 8 M-row cores' cb_in0_down_full (loopback included). Receivers wait on
    // act_valid_sem; sender waits on act_ready_sem reaching GRID_X-1 incs from
    // the 7 receivers. Sender position rotates per K-block so each core takes
    // a turn as sender exactly once per chunk.
    const uint32_t act_ready_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t act_valid_sem_id = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);

    // -------------------------- circular buffers --------------------------
    // Double-buffered DRAM-streamed inputs.
    auto make_cb = [&](uint32_t cb_idx, tt::DataFormat fmt, uint32_t num_tiles, uint32_t tile_bytes) {
        tt::tt_metal::CircularBufferConfig cfg =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_bytes, {{cb_idx, fmt}})
                .set_page_size(cb_idx, tile_bytes);
        return tt::tt_metal::CreateCircularBuffer(program, core_range_set, cfg);
    };

    // Single-buffered DRAM-streamed inputs (no double-buffer) to fit L1.
    // Double-buffered input CBs so the reader (NCRISC) can fetch K-block N+1
    // while compute consumes K-block N. PM FPU util = 0 today says we're
    // memory-bound; bigger input CBs let the kernel pipeline DRAM I/O with
    // compute instead of serialising.
    make_cb(CB_IN0_X, x_df, /*tiles=*/gu_in0_block_num_tiles * 2, x_tile_size);
    make_cb(CB_IN1_GATE, gate_df, /*tiles=*/gu_in1_block_num_tiles * 2, gate_tile_size);
    make_cb(CB_IN1_UP, up_df, /*tiles=*/gu_in1_block_num_tiles * 2, up_tile_size);
    make_cb(CB_IN1_DOWN, down_df, /*tiles=*/d_in1_block_num_tiles * 2, down_tile_size);
    // Intermediate L1 buffers hold one full per-core block each.
    make_cb(CB_GATE_INT, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    // cb_up_intermed removed — multiply reads cb_partials_up directly.
    make_cb(CB_ACTIVATED, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    // Partials CBs: sized to the full per-core output block. Within ONE
    // K-block iteration the kernel pushes (in0_num_subblocks *
    // in1_num_subblocks) subblocks to partials before any pops happen
    // (the pops happen on the NEXT K-block iteration's reload). So the CB
    // must hold all those subblocks = the full block.
    make_cb(
        CB_PARTIALS_GU,
        partials_gu_df,
        /*tiles=*/gu_out_block_num_tiles,
        partials_gu_tile_size);
    // Second gate/up matmul accumulator (cb_partials_up), used by the fused
    // gate+up phase to share x reads across both matmuls.
    make_cb(CB_PARTIALS_UP, partials_gu_df, /*tiles=*/gu_out_block_num_tiles, partials_gu_tile_size);
    make_cb(
        CB_PARTIALS_D,
        partials_d_df,
        /*tiles=*/d_out_block_num_tiles,
        partials_d_tile_size);
    // Output CB: writer drains one subblock at a time. 2-subblock staging
    // pipelines compute/writer one-ahead and is safe under the tightest L1
    // budget (the 256-expert / 32-per-chip case the unfused path is run on).
    constexpr uint32_t cb_out_stage_count = 2u;
    make_cb(
        CB_OUT,
        out_df,
        /*tiles=*/d_out_subblock_h * d_out_subblock_w * cb_out_stage_count,
        out_tile_size);
    // cb_in0_down_full: reader pushes per_core_M × in0_block_w_d tiles of activated
    // once per down K-block. Single-buffered to save L1.
    // cb_in0_down_full double-buffered — fits because we eliminated
    // cb_up_intermed (multiply reads cb_partials_up directly).
    make_cb(
        CB_IN0_DOWN_FULL,
        intermed_df,
        /*tiles=*/d_in0_block_num_tiles * 2,
        intermed_tile_size);

    // Scratch CBs for the device-side count lookup. The reader does a single
    // noc_async_read_page(page=0, ...) of each tensor and then indexes
    // counts[global_expert_id] / idx[local_expert_id]. Both indices stay within
    // the tensor's own length (counts: [0, num_global_experts); idx:
    // [0, idx_len)), and the reader's page-0 read contract requires the entire
    // vector to live in page 0 — i.e. aligned_page_size already covers every
    // index the reader can produce. So aligned_page_size is the exact capacity
    // the scratch needs; we keep a num_entries*4B floor purely as a defensive
    // guard in case a future layout reports a sub-row page.
    //
    // Previously this floored to MAX_GLOBAL_EXPERTS * 4B (a fixed 4 KB), which
    // over-allocated ~3 KB/core for the 256-expert DS-V3 path and ~2.5 KB for
    // Kimi (384). That slack is what tipped the mesh-4x2 perf-256 program over
    // the L1 ceiling once MEM_MAILBOX_SIZE grew 256 B (#46526). Sizing to the
    // real per-call requirement reclaims it (~6 KB/core across both scratches)
    // — far more than the 192 B overlap — and still scales correctly up to the
    // host-side-validated MAX_GLOBAL_EXPERTS limit.
    const uint32_t counts_num_entries = static_cast<uint32_t>(t.counts.logical_shape()[-1]);
    const uint32_t idx_num_entries = static_cast<uint32_t>(t.global_expert_idx_table.logical_shape()[-1]);
    const uint32_t counts_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(counts_buffer->aligned_page_size()),
        counts_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    const uint32_t idx_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(idx_buffer->aligned_page_size()),
        idx_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    tt::tt_metal::CircularBufferConfig counts_cb_cfg =
        tt::tt_metal::CircularBufferConfig(counts_scratch_bytes, {{CB_COUNTS_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_COUNTS_SCRATCH, counts_scratch_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, counts_cb_cfg);
    // CB_IDX_SCRATCH holds the device-side global_expert_idx_table page so
    // reader/compute/writer can resolve `global_expert_id =
    // idx_table[local_expert_id]` without re-reading DRAM. Sized the same way:
    // a single-chip deployment can place all experts locally, so the idx table
    // may itself be up to MAX_GLOBAL_EXPERTS entries.
    tt::tt_metal::CircularBufferConfig idx_cb_cfg =
        tt::tt_metal::CircularBufferConfig(idx_scratch_bytes, {{CB_IDX_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_IDX_SCRATCH, idx_scratch_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, idx_cb_cfg);

    // CB_START_SCRATCH holds the device-side `start` (expert_region_offsets)
    // page for the writer in direct-write mode. Same sizing rationale as the
    // counts scratch: expert_region_offsets is validated to have the same
    // length as counts, so size it to the real per-call requirement (lands the
    // tensor's page and holds every region-offset entry) rather than the old
    // fixed MAX_GLOBAL_EXPERTS floor.
    const uint32_t start_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(start_buffer->aligned_page_size()),
        counts_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    tt::tt_metal::CircularBufferConfig start_cb_cfg =
        tt::tt_metal::CircularBufferConfig(start_scratch_bytes, {{CB_START_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_START_SCRATCH, start_scratch_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, start_cb_cfg);

    // -------------------------- kernel build ------------------------------
    // Reader compile-time args. Order must exactly match the layout the reader
    // kernel reads via get_compile_time_arg_val(idx) and the TensorAccessor
    // offsets it computes after the named-arg block.
    std::vector<uint32_t> reader_ct_args = {
        CB_IN0_X,
        CB_IN1_GATE,
        CB_IN1_UP,
        CB_IN1_DOWN,
        CB_IN0_DOWN_FULL,
        CB_COUNTS_SCRATCH,
        CB_IDX_SCRATCH,
        op.local_expert_id,
        per_core_M,
        per_core_N_gu,
        per_core_N_d,
        K_gate_tiles,
        K_down_tiles,
        in0_block_w_gu,
        in0_block_w_d,
        N_gate_tiles_full,
        N_down_tiles_full,
        M_tiles_full,
        num_chunks,
        chunk_M_tiles,
        // CB_ACTIVATED — consumed by the reader during phase 4 L1 mcast.
        CB_ACTIVATED,
        // GRID_X — M-row mcast group size, used for both num_dests and the
        // NoC-table endpoint index.
        GRID_X,
        // K_down_tiles_padded — phase-4 K-loop bound. K dim of down is
        // padded to N_gate_padded so per-K-block sender = gx == kb holds.
        K_down_tiles_padded,
    };
    tt::tt_metal::TensorAccessorArgs(x_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(up_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(down_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(idx_buffer).append_to(reader_ct_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/"
        "unified_routed_expert_ffn_reader.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // Writer compile-time args (must match writer's get_compile_time_arg_val order).
    std::vector<uint32_t> writer_ct_args = {
        CB_ACTIVATED,       // 0
        CB_OUT,             // 1
        per_core_M,         // 2
        per_core_N_gu,      // 3
        per_core_N_d,       // 4
        gu_out_subblock_h,  // 5
        gu_out_subblock_w,  // 6
        d_out_subblock_h,   // 7
        d_out_subblock_w,   // 8
        N_gate_tiles_full,  // 9
        N_down_tiles_full,  // 10
        num_chunks,         // 11
        chunk_M_tiles,      // 12
        // device-side count read: writer also waits on the reader's push
        // and bounds its cb_out drain loop by effective_chunks so it does
        // not wait forever on chunks compute never pushes.
        CB_COUNTS_SCRATCH,   // 13
        CB_IDX_SCRATCH,      // 14
        op.local_expert_id,  // 15
        // M_tiles_full: needed for the writer to skip OOB output writes when
        // M_tiles_full doesn't divide chunk_M_tiles. The last chunk runs
        // chunk_M_tiles rows per core, of which only those < M_tiles_full
        // correspond to real output rows in the tensor.
        M_tiles_full,  // 16
        // direct_write: 1 -> writer adds start[global_id]/TILE tile-rows and
        // targets the shared output buffer (fuses ttnn::insert).
        static_cast<uint32_t>(direct_write),  // 17
        // dst_M_tiles: tile-row count of the destination buffer (= M_tiles_full
        // for the per-expert output; the shared buffer's rows in direct mode).
        dst_M_tiles,       // 18
        CB_START_SCRATCH,  // 19
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(start_buffer).append_to(writer_ct_args);

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/"
        "unified_routed_expert_ffn_writer.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    // Compute kernel compile-time args: positional + named CB ids.
    std::vector<uint32_t> compute_ct_args = {
        // gate
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        // up
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        // down
        in0_block_w_d,
        d_in0_num_subblocks,
        d_in0_block_num_tiles,
        d_in0_subblock_num_tiles,
        d_in1_num_subblocks,
        d_in1_block_num_tiles,
        d_in1_block_w,
        d_num_blocks,
        // gate/up out subblock
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_block_num_tiles,
        // down out subblock
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_block_num_tiles,
        // chunk loop control
        num_chunks,
        // device-side count read: local_expert_id + chunk_M_tiles (in tiles)
        // let compute convert count -> effective_chunks and bound the loop.
        op.local_expert_id,
        chunk_M_tiles,
    };
    std::unordered_map<std::string, uint32_t> compute_named_args = {
        {"cb_in0_x", CB_IN0_X},
        {"cb_in1_gate", CB_IN1_GATE},
        {"cb_in1_up", CB_IN1_UP},
        {"cb_in1_down", CB_IN1_DOWN},
        {"cb_gate_intermed", CB_GATE_INT},
        {"cb_up_intermed", CB_UP_INT},
        {"cb_activated", CB_ACTIVATED},
        {"cb_in0_down_full", CB_IN0_DOWN_FULL},
        {"cb_mm_partials_gu", CB_PARTIALS_GU},
        {"cb_mm_partials_up", CB_PARTIALS_UP},
        {"cb_mm_partials_d", CB_PARTIALS_D},
        {"cb_out", CB_OUT},
        // For device-side count read: compute waits on the reader's push and
        // bounds its chunk loop by effective_chunks = ceil(count/chunk_M_tiles).
        {"cb_counts_scratch", CB_COUNTS_SCRATCH},
        {"cb_idx_scratch", CB_IDX_SCRATCH},
    };

    // PACKER_L1_ACC controls cross-K-block accumulation via packer L1 RMW.
    std::map<std::string, std::string> compute_defines{};
    compute_defines["PACKER_L1_ACC"] = "1";

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/compute/"
        "fused_swiglu.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
            .defines = compute_defines,
            .named_compile_args = compute_named_args,
        });

    // -------------------------- per-core runtime args ---------------------
    // Cross-core synchronization is now done entirely via L1-mcast: weights
    // mcast within N-col groups using in1_{ready,valid}_sem, x mcast within
    // M-row groups using in0_{ready,valid}_sem, and activated mcast within
    // M-row groups (rotating sender per phase-4 K-block) using
    // act_{ready,valid}_sem. No global cross-grid barrier is needed.
    std::vector<CoreCoord> cores;
    cores.reserve(GRID_X * GRID_Y);
    for (uint32_t gy = 0; gy < GRID_Y; ++gy) {
        for (uint32_t gx = 0; gx < GRID_X; ++gx) {
            cores.push_back(CoreCoord{gx, gy});
        }
    }

    auto* device = t.x.device();

    for (uint32_t idx = 0; idx < cores.size(); ++idx) {
        const auto& core = cores[idx];
        const uint32_t gy = idx / GRID_X;
        const uint32_t gx = idx % GRID_X;
        const uint32_t my_mt = gy;
        const uint32_t my_nt_gu = gx;
        const uint32_t my_nt_d = gx;

        // Weight-multicast topology for in1 (gate/up/down). For each N-col
        // group (fixed gx), the sender is the gy=0 core. Receivers are the
        // GRID_Y-1 cores at gy=1..GRID_Y-1 sharing the same gx. NoC
        // multicast destination rectangle is a single NoC column spanning
        // those receiver rows.
        const bool is_in1_sender = (gy == 0);
        const auto sender_noc = device->worker_core_from_logical_core(CoreCoord{gx, 0});
        const auto first_recv_noc = device->worker_core_from_logical_core(CoreCoord{gx, 1});
        const auto last_recv_noc = device->worker_core_from_logical_core(CoreCoord{gx, GRID_Y - 1});
        const uint32_t in1_num_receivers = GRID_Y - 1;
        const uint32_t in1_mcast_nx_start = first_recv_noc.x;
        const uint32_t in1_mcast_ny_start = first_recv_noc.y;
        const uint32_t in1_mcast_nx_end = last_recv_noc.x;
        const uint32_t in1_mcast_ny_end = last_recv_noc.y;
        const uint32_t in1_sender_nx = sender_noc.x;
        const uint32_t in1_sender_ny = sender_noc.y;

        // x (in0) multicast topology: per M-row, sender at gx=0, receivers
        // at gx=1..GRID_X-1.
        const bool is_in0_sender = (gx == 0);
        const auto in0_sender_noc = device->worker_core_from_logical_core(CoreCoord{0, gy});
        const auto in0_first_recv_noc = device->worker_core_from_logical_core(CoreCoord{1, gy});
        const auto in0_last_recv_noc = device->worker_core_from_logical_core(CoreCoord{GRID_X - 1, gy});
        const uint32_t in0_num_receivers = GRID_X - 1;
        const uint32_t in0_mcast_nx_start = in0_first_recv_noc.x;
        const uint32_t in0_mcast_ny_start = in0_first_recv_noc.y;
        const uint32_t in0_mcast_nx_end = in0_last_recv_noc.x;
        const uint32_t in0_mcast_ny_end = in0_last_recv_noc.y;
        const uint32_t in0_sender_nx = in0_sender_noc.x;
        const uint32_t in0_sender_ny = in0_sender_noc.y;

        // Reader runtime arg layout (must match unified_routed_expert_ffn_reader.cpp):
        //   0..5: tensor addrs (x, gate, up, down, counts, idx)
        //   6: my_mt
        //   7: my_nt_gu
        //   8: my_nt_d
        //   9..18: in1 multicast args
        //  19..28: in0 multicast args
        //  29: act_ready_sem_id
        //  30: act_valid_sem_id
        //  31+: M-row NoC coord table (GRID_X pairs of x, y)
        std::vector<uint32_t> reader_args = {
            x_buffer->address(),
            gate_buffer->address(),
            up_buffer->address(),
            down_buffer->address(),
            counts_buffer->address(),
            idx_buffer->address(),
            my_mt,
            my_nt_gu,
            my_nt_d,
            static_cast<uint32_t>(is_in1_sender),
            in1_ready_sem_id,
            in1_valid_sem_id,
            in1_num_receivers,
            in1_mcast_nx_start,
            in1_mcast_ny_start,
            in1_mcast_nx_end,
            in1_mcast_ny_end,
            in1_sender_nx,
            in1_sender_ny,
            static_cast<uint32_t>(is_in0_sender),
            in0_ready_sem_id,
            in0_valid_sem_id,
            in0_num_receivers,
            in0_mcast_nx_start,
            in0_mcast_ny_start,
            in0_mcast_nx_end,
            in0_mcast_ny_end,
            in0_sender_nx,
            in0_sender_ny,
            act_ready_sem_id,
            act_valid_sem_id,
        };
        // M-row NoC coord table: for our M-row (gy=my_mt), the NoC (x, y) of
        // each of the GRID_X cores (gx=0..GRID_X-1). Reader uses this per
        // phase-4 K-block (kb=0..K_down_tiles_padded-1) to find the sender's
        // NoC addr and to build the M-row mcast rectangle.
        for (uint32_t gxi = 0; gxi < GRID_X; ++gxi) {
            const auto noc = device->worker_core_from_logical_core(CoreCoord{gxi, gy});
            reader_args.push_back(static_cast<uint32_t>(noc.x));
            reader_args.push_back(static_cast<uint32_t>(noc.y));
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_args);

        // Writer runtime arg layout (must match unified_routed_expert_ffn_writer.cpp):
        //   0: output_addr
        //   1: my_mt
        //   2: my_nt_d
        //   3: start_addr (expert_region_offsets in direct-write mode; else
        //      out_buffer, unused by the kernel)
        std::vector<uint32_t> writer_args = {
            out_buffer->address(),
            my_mt,
            my_nt_d,
            start_buffer->address(),
        };
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_args);
    }

    return cached_program_t{
        std::move(program),
        UnifiedRoutedExpertFfnSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .cores = std::move(cores)}};
}

void UnifiedRoutedExpertFfnProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UnifiedRoutedExpertFfnParams& /*op*/,
    const UnifiedRoutedExpertFfnInputs& t,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto reader_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    const uint32_t x_addr = t.x.buffer()->address();
    const uint32_t gate_addr = t.gate_proj.buffer()->address();
    const uint32_t up_addr = t.up_proj.buffer()->address();
    const uint32_t down_addr = t.down_proj.buffer()->address();
    const uint32_t counts_addr = t.counts.buffer()->address();
    const uint32_t idx_addr = t.global_expert_idx_table.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();
    const uint32_t start_addr =
        t.expert_region_offsets.has_value() ? t.expert_region_offsets->buffer()->address() : out_addr;

    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_id, core);
        reader_args[0] = x_addr;
        reader_args[1] = gate_addr;
        reader_args[2] = up_addr;
        reader_args[3] = down_addr;
        reader_args[4] = counts_addr;
        reader_args[5] = idx_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_id, core);
        writer_args[0] = out_addr;
        writer_args[3] = start_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
