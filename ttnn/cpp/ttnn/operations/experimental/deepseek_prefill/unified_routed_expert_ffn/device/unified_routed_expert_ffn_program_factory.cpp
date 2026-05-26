// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_program_factory.hpp"

#include <cstdint>
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
// partials_up (up matmul) using the SAME shared x K-block, halving x DRAM
// reads vs the v1 sequential-phases design.
constexpr uint32_t CB_PARTIALS_UP = tt::CBIndex::c_13;
// Region offsets share CB_IDX_SCRATCH's L1 region. The idx CB is made
// two-page: page 0 = idx table, page 1 = expert_region_offsets. Both are
// UInt32 vectors of length num_routed_experts so the page sizes match
// exactly — no separate CB allocation (which on Blackhole rounded up to
// a 4KB region per core and pushed cb_in0_x past the L1 budget for the
// 256-expert / 32-per-chip case).
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

    // v2 layout: 11x8 = 88 compute cores. N-axis is rounded UP to a multiple
    // of GRID_X via ceil_div per_core_N. Phantom tiles past the actual tensor
    // dims (col 64-65 of hidden, col 224-230 of emb) are zero-padded in the
    // reader (zero-fill L1 instead of DRAM read). Compute runs uniform
    // per_core_N; writer skips DRAM writes past actual_N. K dim of the down
    // matmul is also padded to N_gate_padded so the activated L1 mcast (one
    // sender per K-block, sender = gx == kb) covers exactly per_core_N_gu
    // cols per step; activated cols past actual_hidden are 0 (gate/up weight
    // OOB zero-fill propagates through silu and multiply).
    constexpr uint32_t GRID_X = 11;
    constexpr uint32_t GRID_Y = 8;
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

    // With 11x8 = 88 cores, per_core_N_gu (= 6) and per_core_N_d (= 21) are
    // smaller than in v1 (8x8), freeing ~250KB of L1 per core. Use it to
    // double in0_block_w_gu, halving the gate / up K-loop iteration count.
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

    // Subblock dims. With bf16 dst (fp32_dest_acc_en=false), capacity is 8.
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
    // v2 compute kernel's mm_init pattern (mm_init's 3rd arg drives the
    // packer's data-format config; mismatched formats need explicit pack
    // reconfig that the v2 kernel doesn't do). Use bfp8_b for both: 1KB/tile
    // is half the bf16 cost so we fit in L1 with both intermediates and
    // partials sized to the full per-core block.
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
    auto* region_offsets_buffer = t.expert_region_offsets.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    // -------------------------- scratch buffer + semaphore ----------------
    // LEGACY DRAM scratch — used by the previous DRAM-round-trip activated
    // path. The current L1-mcast activated path doesn't touch this buffer;
    // both reader and writer kernels declare the accessor but
    // `(void)scratch_acc` it. The buffer is kept as a 1-tile placeholder
    // (~1KB DRAM) so the TensorAccessorArgs CT-arg layout in both kernels
    // stays compatible with the existing slot order. Without shrinking,
    // each cached program (one per local_expert_id × layer) would hold a
    // ~1.15MB DRAM buffer it never reads, multiplying across the full
    // DS-V3 prefill to ~184MB of leaked DRAM.
    const uint32_t scratch_num_tiles = 1;
    const uint32_t scratch_bytes = scratch_num_tiles * tt::tile_size(intermed_df);
    tt::tt_metal::InterleavedBufferConfig scratch_cfg{
        .device = t.x.device(),
        .size = scratch_bytes,
        .page_size = tt::tile_size(intermed_df),
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto activated_scratch = tt::tt_metal::CreateBuffer(scratch_cfg);
    auto* scratch_buffer = activated_scratch.get();

    // Global semaphore reserved on every compute core (CreateSemaphore puts
    // it at the same L1 offset on every core in the range). The writer kernel
    // NoC-increments the OWNER core's slot; the reader on every core waits on
    // ITS OWN slot for value == total_cores — that requires each writer to
    // increment all cores' slots, OR we designate one core as the master and
    // have only its reader wait. Simpler approach: every writer increments
    // every reader's slot via NoC mcast.
    const uint32_t total_cores = GRID_X * GRID_Y;
    // ready_sem lives on EVERY core's L1 at the same offset, but is meaningful
    // only on the controller's L1 — non-controllers atomic-inc the controller's
    // slot per barrier. Reused for both barrier A and barrier B per chunk with
    // chunk-indexed targets.
    const uint32_t semaphore_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // done_sem lives on every core's L1. Controller writes the new "done" value
    // to every core's slot (including its own) once it sees ready_sem reach the
    // per-barrier target. Reader spins on it with target 2*chunk+1; the next-
    // chunk phase-3-drain (in writer) spins on it with target 2*chunk+2.
    const uint32_t done_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // Weight-multicast semaphores for in1 (gate/up/down). Pattern: per
    // N-col group (gx fixed, gy=0..GRID_Y-1), one sender at gy=0 reads the
    // weight slice from DRAM and mcasts it to the other GRID_Y-1 cores in
    // the same column. Receivers atomic-inc `ready` on the sender to signal
    // "I'm ready"; sender waits for ready==GRID_Y-1, mcasts the block, then
    // mcast-sets `valid` to 1 on all receivers; receivers wait for valid==1.
    // Same sem pair is reused across gate/up/down phases (phases are
    // sequential, sem values reset between K-blocks).
    constexpr uint32_t INVALID_SEM = 0;
    constexpr uint32_t VALID_SEM = 1;
    (void)INVALID_SEM;
    (void)VALID_SEM;
    const uint32_t in1_ready_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t in1_valid_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // in0 (x) multicast within M-row groups: sender at (gx=0, gy) reads x
    // for that M-row, mcasts to (gx=1..GRID_X-1, gy). Used for phases 1 and
    // 2 (gate and up matmul) where every core in a row needs the same x
    // slice. Phase 4 uses cb_in0_down_full (sourced from DRAM scratch) so
    // doesn't use this pair.
    const uint32_t in0_ready_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t in0_valid_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    // Activated multicast sems (phase 4): replace the DRAM scratch round-trip
    // with an L1 NoC mcast. For phase-4 K-block kb, sender = core at
    // (gx=kb, my_mt). Sender's reader mcasts its cb_activated block to all
    // 8 M-row cores' cb_in0_down_full (loopback included). Receivers wait on
    // act_valid_sem; sender waits on act_ready_sem reaching GRID_X-1 incs from
    // the 7 receivers. Sender position rotates per K-block so each core takes
    // a turn as sender exactly once per chunk.
    const uint32_t act_ready_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);
    const uint32_t act_valid_sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);

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
    // Output CB: writer drains one subblock at a time. In the unfused path
    // (use_region_offsets=false, 256-expert / 32-per-chip case) the L1
    // budget is tight enough that the prior 4-subblock staging pushed the
    // static CB region into already-allocated L1 buffers. Drop to 2
    // subblocks (still pipelines compute/writer one-ahead) only on that
    // path; keep 4 for the fused path where L1 has headroom.
    const uint32_t cb_out_stage_count = op.use_region_offsets ? 4u : 2u;
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

    // Scratch CBs for the device-side count lookup. One page each, sized to
    // the corresponding tensor's aligned page size so noc_async_read_page
    // can land them.
    const uint32_t counts_page_size = counts_buffer->aligned_page_size();
    const uint32_t idx_page_size = idx_buffer->aligned_page_size();
    tt::tt_metal::CircularBufferConfig counts_cb_cfg =
        tt::tt_metal::CircularBufferConfig(counts_page_size, {{CB_COUNTS_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_COUNTS_SCRATCH, counts_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, counts_cb_cfg);
    // CB_IDX_SCRATCH:
    //   * use_region_offsets=true (fused path): holds BOTH idx and
    //     region_offsets in one page:
    //       bytes [0, idx_page_size)                     -> idx table
    //       bytes [idx_page_size, idx_page_size+region)  -> region_offsets
    //     Reader reads both into the single L1 page (idx first,
    //     region_offsets appended), pushes 1. Writer/compute cb_wait_front
    //     and read at their respective offset.
    //   * use_region_offsets=false (unfused path): holds ONLY idx. Saves
    //     one region_offsets page per core; combined with the smaller
    //     CB_OUT staging this drops the static CB region below the L1
    //     buffer placed near the top of L1 (the placement the prior
    //     offset_cumsum-output-to-DRAM mitigation already shifted around).
    const uint32_t region_offsets_page_size = region_offsets_buffer->aligned_page_size();
    const uint32_t idx_cb_page_size =
        op.use_region_offsets ? (idx_page_size + region_offsets_page_size) : idx_page_size;
    tt::tt_metal::CircularBufferConfig idx_cb_cfg =
        tt::tt_metal::CircularBufferConfig(idx_cb_page_size, {{CB_IDX_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_IDX_SCRATCH, idx_cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, idx_cb_cfg);

    // -------------------------- kernel build ------------------------------
    // Reader compile-time args. Order must exactly match the layout the reader
    // kernel reads via get_compile_time_arg_val(idx) and the TensorAccessor
    // offsets it computes from offset 19 onwards.
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
        total_cores,
        num_chunks,
        chunk_M_tiles,
        // CB_ACTIVATED — consumed by the reader during phase 4 L1 mcast.
        CB_ACTIVATED,
        // GRID_X — replaces the hard-coded 8 in reader's L1 mcast. With
        // 11-core M-rows we need the mcast num_dests and the NoC-table
        // endpoint index to track.
        GRID_X,
        // K_down_tiles_padded — phase-4 K-loop bound. K dim of down is
        // padded to N_gate_padded so per-K-block sender = gx == kb holds.
        K_down_tiles_padded,
        // region_offsets share CB_IDX_SCRATCH's single page — reader writes
        // idx at offset 0 and region_offsets at offset idx_page_bytes
        // within the same L1 page. Pass idx_page_bytes so the kernels know
        // where to split. CT arg slot here is the byte offset of
        // region_offsets within the shared idx-scratch L1 page. Only read
        // when use_region_offsets=true; otherwise unused.
        idx_page_size,
        // use_region_offsets: when false, the kernel skips the
        // region_offsets DRAM read and uses start_tile_row=0 (correct for
        // the unfused extract->FFN path where `x` is already the per-expert
        // slice).
        static_cast<uint32_t>(op.use_region_offsets),
    };
    tt::tt_metal::TensorAccessorArgs(x_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(up_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(down_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(idx_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(scratch_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(region_offsets_buffer).append_to(reader_ct_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/"
        "reader_unified_re.cpp",
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
        // Byte offset of region_offsets within CB_IDX_SCRATCH's L1 page.
        // Writer reads idx at offset 0 and region_offsets at this offset.
        // Only used when use_region_offsets=true.
        idx_page_size,                                 // 17
        static_cast<uint32_t>(op.use_region_offsets),  // 18
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(scratch_buffer).append_to(writer_ct_args);

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/"
        "writer_unified_re.cpp",
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
        "fused_swiglu_v3.cpp",
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
    // Every writer NoC-increments ALL cores' semaphore slots so every reader
    // sees value == total_cores once every core has hit its sync point.
    // Writer therefore needs the NoC (x, y) of every core in the grid.
    // For v1 we hard-code "increment self only" by using each core's own
    // coords as the target — this makes the increment effectively local and
    // we instead spin a per-core counter on the reader. The cross-core gather
    // is achieved by having the reader poll the counter as it ramps from 0
    // up to total_cores: each writer increments ITS OWN sem slot, and every
    // reader on every core polls THEIR OWN slot until it reaches 1, which
    // means THAT core's writer finished its phase-3 drain. But that only
    // gives us a per-core barrier, not a global one.
    //
    // The correct minimum-overhead pattern for "wait until every core has
    // finished phase 3" is: each writer NoC-increments a single global
    // semaphore on a designated owner core; every reader on every core
    // polls THAT same semaphore via remote NoC reads through
    // `noc_semaphore_wait_remote` (or equivalent). For simplicity in v1 we
    // implement that via every writer NoC-incrementing every core's slot
    // (64 increments per writer = 4096 NoC sem-incs per program; cheap at
    // ~10ns each).
    std::vector<CoreCoord> cores;
    cores.reserve(GRID_X * GRID_Y);
    for (uint32_t gy = 0; gy < GRID_Y; ++gy) {
        for (uint32_t gx = 0; gx < GRID_X; ++gx) {
            cores.push_back(CoreCoord{gx, gy});
        }
    }

    // Resolve the multicast rectangle covering the whole compute grid in
    // NoC (virtual) coords. The writer kernel uses noc_semaphore_inc_multicast
    // to bump every core's local sem slot in a single NoC transaction.
    auto* device = t.x.device();
    const auto top_left_noc = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto bot_right_noc = device->worker_core_from_logical_core(CoreCoord{GRID_X - 1, GRID_Y - 1});
    const uint32_t mcast_x_start = top_left_noc.x;
    const uint32_t mcast_y_start = top_left_noc.y;
    const uint32_t mcast_x_end = bot_right_noc.x;
    const uint32_t mcast_y_end = bot_right_noc.y;

    for (uint32_t idx = 0; idx < cores.size(); ++idx) {
        const auto& core = cores[idx];
        const uint32_t gy = idx / GRID_X;
        const uint32_t gx = idx % GRID_X;
        const uint32_t my_mt = gy;
        const uint32_t my_nt_gu = gx;
        const uint32_t my_nt_d = gx;
        const uint32_t chunk_start_tile_row = 0;  // single chunk for v1

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

        std::vector<uint32_t> reader_args = {
            x_buffer->address(),
            gate_buffer->address(),
            up_buffer->address(),
            down_buffer->address(),
            counts_buffer->address(),
            idx_buffer->address(),
            scratch_buffer->address(),
            semaphore_addr,
            my_mt,
            my_nt_gu,
            my_nt_d,
            chunk_start_tile_row,
            // in1 multicast args (indices 12..20):
            static_cast<uint32_t>(is_in1_sender),  // 12
            in1_ready_sem_addr,                    // 13
            in1_valid_sem_addr,                    // 14
            in1_num_receivers,                     // 15
            in1_mcast_nx_start,                    // 16
            in1_mcast_ny_start,                    // 17
            in1_mcast_nx_end,                      // 18
            in1_mcast_ny_end,                      // 19
            in1_sender_nx,                         // 20
            in1_sender_ny,                         // 21
            // in0 multicast args (indices 22..31):
            static_cast<uint32_t>(is_in0_sender),  // 22
            in0_ready_sem_addr,                    // 23
            in0_valid_sem_addr,                    // 24
            in0_num_receivers,                     // 25
            in0_mcast_nx_start,                    // 26
            in0_mcast_ny_start,                    // 27
            in0_mcast_nx_end,                      // 28
            in0_mcast_ny_end,                      // 29
            in0_sender_nx,                         // 30
            in0_sender_ny,                         // 31
            // done_sem id (32) — kept for now but unused after L1 mcast switch.
            done_sem_addr,  // 32
            // Activated L1 mcast sems (33, 34) — replace the cross-core
            // phase-3/4 barrier with per-K-block sender/receiver handshake.
            act_ready_sem_addr,  // 33
            act_valid_sem_addr,  // 34
            // expert_region_offsets DRAM buffer address (35). Reader pulls
            // the page into cb_region_offsets_scratch L1 at kernel start;
            // writer cb_wait_fronts the same CB to read start_tile_row.
            region_offsets_buffer->address(),  // 35
        };
        // M-row NoC coord table: for our M-row (gy=my_mt), the NoC (x, y) of
        // each of the 8 cores (gx=0..GRID_X-1). Reader uses this per phase-4
        // K-block (kb=0..7) to find the sender's NoC addr and to build the
        // M-row mcast rectangle. Starts at index 36 now (was 35 before
        // region_offsets_addr was added).
        for (uint32_t gxi = 0; gxi < GRID_X; ++gxi) {
            const auto noc = device->worker_core_from_logical_core(CoreCoord{gxi, gy});
            reader_args.push_back(static_cast<uint32_t>(noc.x));
            reader_args.push_back(static_cast<uint32_t>(noc.y));
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_args);

        // Writer runtime arg layout matches writer_unified_re.cpp:
        //   0: output_addr
        //   1: scratch_addr
        //   2: sem_addr (local L1 offset)
        //   3: num_sync_cores (total cores covered by the multicast)
        //   4: mcast_x_start  (NoC virtual coords of the rectangle's top-left)
        //   5: mcast_y_start
        //   6: mcast_x_end    (NoC virtual coords of the bottom-right)
        //   7: mcast_y_end
        //   8: my_mt
        //   9: my_nt_gu
        //  10: my_nt_d
        //  11: chunk_start_tile_row (legacy / unused)
        //  12: done_sem_addr (controller writes per-barrier; readers/writers wait)
        //  13+ : per-core NoC coords (interleaved x, y)
        std::vector<uint32_t> writer_args = {
            out_buffer->address(),
            scratch_buffer->address(),
            semaphore_addr,
            static_cast<uint32_t>(cores.size()),
            mcast_x_start,
            mcast_y_start,
            mcast_x_end,
            mcast_y_end,
            my_mt,
            my_nt_gu,
            my_nt_d,
            chunk_start_tile_row,
            done_sem_addr,
        };
        // Append the NoC virtual coords of every compute core (interleaved
        // x, y). The controller writer uses this to unicast-set each other
        // core's sem slot directly.
        for (const auto& c : cores) {
            const auto noc_coord = device->worker_core_from_logical_core(c);
            writer_args.push_back(static_cast<uint32_t>(noc_coord.x));
            writer_args.push_back(static_cast<uint32_t>(noc_coord.y));
        }
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_args);
    }

    return cached_program_t{
        std::move(program),
        UnifiedRoutedExpertFfnSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .cores = std::move(cores),
            .activated_scratch = std::move(activated_scratch),
            .semaphore_addr = semaphore_addr}};
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
    const auto* scratch_buf = cached_program.shared_variables.activated_scratch.get();

    const uint32_t x_addr = t.x.buffer()->address();
    const uint32_t gate_addr = t.gate_proj.buffer()->address();
    const uint32_t up_addr = t.up_proj.buffer()->address();
    const uint32_t down_addr = t.down_proj.buffer()->address();
    const uint32_t counts_addr = t.counts.buffer()->address();
    const uint32_t idx_addr = t.global_expert_idx_table.buffer()->address();
    const uint32_t scratch_addr = scratch_buf->address();
    const uint32_t region_offsets_addr = t.expert_region_offsets.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_id, core);
        reader_args[0] = x_addr;
        reader_args[1] = gate_addr;
        reader_args[2] = up_addr;
        reader_args[3] = down_addr;
        reader_args[4] = counts_addr;
        reader_args[5] = idx_addr;
        reader_args[6] = scratch_addr;
        // index 35 = region_offsets_addr (must stay in sync with the layout
        // built in create() and consumed by reader_unified_re.cpp).
        reader_args[35] = region_offsets_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_id, core);
        writer_args[0] = out_addr;
        writer_args[1] = scratch_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
