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

    // v1 layout: 8x8 compute grid, per_core_N choices that divide hidden/emb
    // exactly so we don't need phantom-column padding.
    constexpr uint32_t GRID_X = 8;
    constexpr uint32_t GRID_Y = 8;
    const uint32_t chunk_M_tiles = op.chunk_M_tiles;
    const uint32_t per_core_M = chunk_M_tiles / GRID_Y;
    TT_FATAL(per_core_M * GRID_Y == chunk_M_tiles, "chunk_M_tiles ({}) must be divisible by GRID_Y ({})", chunk_M_tiles, GRID_Y);
    TT_FATAL(N_gate_tiles_full % GRID_X == 0, "hidden_tiles ({}) must be divisible by GRID_X ({})", N_gate_tiles_full, GRID_X);
    TT_FATAL(N_down_tiles_full % GRID_X == 0, "emb_tiles ({}) must be divisible by GRID_X ({})", N_down_tiles_full, GRID_X);

    const uint32_t per_core_N_gu = N_gate_tiles_full / GRID_X;
    const uint32_t per_core_N_d = N_down_tiles_full / GRID_X;

    // K-block sizes. Pick the largest in0_block_w that divides K and keeps
    // per-CB L1 in budget (we hardcode safe choices for v1).
    const uint32_t in0_block_w_gu = 16;
    const uint32_t in0_block_w_d = 8;
    TT_FATAL(
        K_gate_tiles % in0_block_w_gu == 0,
        "K_gate_tiles ({}) must be divisible by in0_block_w_gu ({})",
        K_gate_tiles,
        in0_block_w_gu);
    TT_FATAL(
        K_down_tiles % in0_block_w_d == 0,
        "K_down_tiles ({}) must be divisible by in0_block_w_d ({})",
        K_down_tiles,
        in0_block_w_d);

    // Subblock dims (out_subblock_h * out_subblock_w <= 8 for safe dst alloc).
    const uint32_t gu_out_subblock_h = 1;
    const uint32_t gu_out_subblock_w = per_core_N_gu;  // per_core_N_gu * 1 <= 8 for v1 (per_core_N_gu=8)
    TT_FATAL(
        gu_out_subblock_h * gu_out_subblock_w <= 8,
        "gu subblock h*w ({}) exceeds dst capacity",
        gu_out_subblock_h * gu_out_subblock_w);
    const uint32_t d_out_subblock_h = 1;
    // Largest divisor of per_core_N_d that is <= 8.
    uint32_t d_sub_w = 1;
    for (uint32_t cand = 8; cand >= 1; --cand) {
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
    const uint32_t d_num_blocks = K_down_tiles / in0_block_w_d;
    const uint32_t d_out_block_num_tiles = per_core_M * per_core_N_d;

    // -------------------------- data formats / tile sizes -----------------
    const tt::DataFormat x_df = tt::tt_metal::datatype_to_dataformat_converter(t.x.dtype());
    const tt::DataFormat gate_df = tt::tt_metal::datatype_to_dataformat_converter(t.gate_proj.dtype());
    const tt::DataFormat up_df = tt::tt_metal::datatype_to_dataformat_converter(t.up_proj.dtype());
    const tt::DataFormat down_df = tt::tt_metal::datatype_to_dataformat_converter(t.down_proj.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    // Intermediate (gate/up/activated) format: use input activation format
    // (bf8) for tightness; partials use bfloat16 for accumulation precision.
    const tt::DataFormat intermed_df = x_df;
    const tt::DataFormat partials_df = tt::DataFormat::Float16_b;

    const uint32_t x_tile_size = tt::tile_size(x_df);
    const uint32_t gate_tile_size = tt::tile_size(gate_df);
    const uint32_t up_tile_size = tt::tile_size(up_df);
    const uint32_t down_tile_size = tt::tile_size(down_df);
    const uint32_t out_tile_size = tt::tile_size(out_df);
    const uint32_t intermed_tile_size = tt::tile_size(intermed_df);
    const uint32_t partials_tile_size = tt::tile_size(partials_df);

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

    // -------------------------- scratch buffer + semaphore ----------------
    // DRAM scratch holds activated between phases 3 and 4, sized for the full
    // chunk M × hidden N tile region (same shape as gate output).
    const uint32_t scratch_num_tiles = chunk_M_tiles * N_gate_tiles_full;
    const uint32_t scratch_bytes = scratch_num_tiles * tt::tile_size(x_df);
    tt::tt_metal::InterleavedBufferConfig scratch_cfg{
        .device = t.x.device(),
        .size = scratch_bytes,
        .page_size = tt::tile_size(x_df),
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
    const uint32_t semaphore_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);

    // -------------------------- circular buffers --------------------------
    // Double-buffered DRAM-streamed inputs.
    auto make_cb = [&](uint32_t cb_idx, tt::DataFormat fmt, uint32_t num_tiles, uint32_t tile_bytes) {
        tt::tt_metal::CircularBufferConfig cfg =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_bytes, {{cb_idx, fmt}})
                .set_page_size(cb_idx, tile_bytes);
        return tt::tt_metal::CreateCircularBuffer(program, core_range_set, cfg);
    };

    make_cb(CB_IN0_X, x_df, /*tiles=*/gu_in0_block_num_tiles * 2, x_tile_size);
    make_cb(CB_IN1_GATE, gate_df, /*tiles=*/gu_in1_block_num_tiles * 2, gate_tile_size);
    make_cb(CB_IN1_UP, up_df, /*tiles=*/gu_in1_block_num_tiles * 2, up_tile_size);
    make_cb(CB_IN1_DOWN, down_df, /*tiles=*/d_in1_block_num_tiles * 2, down_tile_size);
    // Intermediate L1 buffers hold one full per-core block each.
    make_cb(CB_GATE_INT, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    make_cb(CB_UP_INT, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    make_cb(CB_ACTIVATED, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    // Partials CBs: subblock-sized, used for spill-and-reload between K-blocks.
    make_cb(
        CB_PARTIALS_GU,
        partials_df,
        /*tiles=*/gu_out_subblock_h * gu_out_subblock_w * 2,
        partials_tile_size);
    make_cb(
        CB_PARTIALS_D,
        partials_df,
        /*tiles=*/d_out_subblock_h * d_out_subblock_w * 2,
        partials_tile_size);
    // Output CB: writer drains one subblock at a time.
    make_cb(CB_OUT, out_df, /*tiles=*/d_out_subblock_h * d_out_subblock_w * 4, out_tile_size);
    // cb_in0_down_full: reader pushes per_core_M × in0_block_w_d tiles of activated
    // (read back from DRAM scratch) once per down K-block. Double-buffered so the
    // matmul kernel can overlap K-block iterations with reader fetches.
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
    tt::tt_metal::CircularBufferConfig idx_cb_cfg =
        tt::tt_metal::CircularBufferConfig(idx_page_size, {{CB_IDX_SCRATCH, tt::DataFormat::UInt32}})
            .set_page_size(CB_IDX_SCRATCH, idx_page_size);
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
    };
    tt::tt_metal::TensorAccessorArgs(x_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(up_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(down_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(idx_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(scratch_buffer).append_to(reader_ct_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/"
        "reader_unified_re.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // Writer compile-time args (must match writer's get_compile_time_arg_val order).
    std::vector<uint32_t> writer_ct_args = {
        CB_ACTIVATED,
        CB_OUT,
        per_core_M,
        per_core_N_gu,
        per_core_N_d,
        gu_out_subblock_h,
        gu_out_subblock_w,
        d_out_subblock_h,
        d_out_subblock_w,
        N_gate_tiles_full,
        N_down_tiles_full,
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
        {"cb_mm_partials_d", CB_PARTIALS_D},
        {"cb_out", CB_OUT},
    };

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

    // Resolve each core's NoC coords (BH virtual coords) for the writer's
    // semaphore-target list.
    auto* device = t.x.device();
    std::vector<uint32_t> core_noc_coords_flat;
    core_noc_coords_flat.reserve(cores.size() * 2);
    for (const auto& core : cores) {
        const auto virtual_core = device->worker_core_from_logical_core(core);
        core_noc_coords_flat.push_back(virtual_core.x);
        core_noc_coords_flat.push_back(virtual_core.y);
    }

    for (uint32_t idx = 0; idx < cores.size(); ++idx) {
        const auto& core = cores[idx];
        const uint32_t gy = idx / GRID_X;
        const uint32_t gx = idx % GRID_X;
        const uint32_t my_mt = gy;
        const uint32_t my_nt_gu = gx;
        const uint32_t my_nt_d = gx;
        const uint32_t chunk_start_tile_row = 0;  // single chunk for v1

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
            chunk_start_tile_row};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_args);

        // Writer runtime arg layout matches writer_unified_re.cpp:
        //   0: output_addr
        //   1: scratch_addr
        //   2: sem_addr (local L1 offset)
        //   3..(3+2*total_cores-1): pairs of (sem_core_x, sem_core_y) for every
        //                           core whose semaphore slot we'll increment.
        //   then per-core (mt, nt_gu, nt_d, chunk_start_tile_row).
        std::vector<uint32_t> writer_args = {
            out_buffer->address(),
            scratch_buffer->address(),
            semaphore_addr,
            static_cast<uint32_t>(cores.size()),
        };
        writer_args.insert(writer_args.end(), core_noc_coords_flat.begin(), core_noc_coords_flat.end());
        writer_args.push_back(my_mt);
        writer_args.push_back(my_nt_gu);
        writer_args.push_back(my_nt_d);
        writer_args.push_back(chunk_start_tile_row);
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

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_id, core);
        writer_args[0] = out_addr;
        writer_args[1] = scratch_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
