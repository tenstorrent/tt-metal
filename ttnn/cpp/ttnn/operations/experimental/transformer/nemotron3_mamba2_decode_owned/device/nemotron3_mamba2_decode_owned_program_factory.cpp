// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "nemotron3_mamba2_decode_owned_program_factory.hpp"

#include <tuple>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// CB index assignments — MUST match compute kernel's get_compile_time_arg_val(N)
// in nemotron3_mamba2_decode_owned/device/kernels/compute/nemotron3_mamba2_decode_owned.cpp.
constexpr uint32_t CB_X = tt::CBIndex::c_0;
constexpr uint32_t CB_Z = tt::CBIndex::c_1;
constexpr uint32_t CB_DT = tt::CBIndex::c_2;
constexpr uint32_t CB_DT_BIAS = tt::CBIndex::c_3;
constexpr uint32_t CB_A_LOG = tt::CBIndex::c_4;
constexpr uint32_t CB_D = tt::CBIndex::c_5;
constexpr uint32_t CB_B = tt::CBIndex::c_6;
constexpr uint32_t CB_C = tt::CBIndex::c_7;
constexpr uint32_t CB_STATE_IN = tt::CBIndex::c_8;
constexpr uint32_t CB_DECAY = tt::CBIndex::c_9;
constexpr uint32_t CB_DT_B = tt::CBIndex::c_10;
constexpr uint32_t CB_STATE_SCALED = tt::CBIndex::c_11;
constexpr uint32_t CB_Y_PARTIAL = tt::CBIndex::c_12;
constexpr uint32_t CB_STATE_OUT = tt::CBIndex::c_13;
constexpr uint32_t CB_Y = tt::CBIndex::c_14;
// G1 day-4 (debug_mode=3): scratch CBs for the outer-product update.
// cb_x_col holds the transposed [head_dim] vector (col-vector tile) so that
// matmul_tiles(cb_x_col, cb_dt_B) computes the outer product x[d] ⊗ dt_B[s]
// directly into a full 32×32 tile (pattern forked from GDN's mul_outer /
// transpose_k_indexed at qwen36_gdn_decode_owned.cpp:328 + :312).
// cb_outer holds that outer-product tile per (d, s) inner-loop iteration
// before it gets added to state_scaled.
constexpr uint32_t CB_X_COL = tt::CBIndex::c_15;
constexpr uint32_t CB_OUTER = tt::CBIndex::c_16;
// G1 day-4.6 (mode=5): post-update state buffer for the C·state_out^T reduce.
// add_state_scaled_outer_two packs to BOTH cb_state_out (writer) AND this CB
// (compute reads it back in Phase 4 to compute y_partial_full).
// Same fp32 format as cb_state_scaled / cb_state_in. 8 tiles per (B, head).
constexpr uint32_t CB_STATE_POST_UPDATE = tt::CBIndex::c_17;

CBHandle create_circular_buffer(
    Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    uint32_t num_tiles,
    uint32_t tile_size,
    const tt::DataFormat& format) {
    const CircularBufferConfig config =
        CircularBufferConfig(num_tiles * tile_size, {{cb_id, format}}).set_page_size(cb_id, tile_size);
    return CreateCircularBuffer(program, cores, config);
}

}  // namespace

Nemotron3Mamba2DecodeOwnedProgramFactory::cached_program_t Nemotron3Mamba2DecodeOwnedProgramFactory::create(
    const Nemotron3Mamba2DecodeOwnedParams& operation_attributes,
    const Nemotron3Mamba2DecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& x = tensor_args.x;
    const auto& z = tensor_args.z;
    const auto& dt = tensor_args.dt;
    const auto& dt_bias = tensor_args.dt_bias;
    const auto& A_log = tensor_args.A_log;
    const auto& D = tensor_args.D;
    const auto& B_in = tensor_args.B_in;
    const auto& C_in = tensor_args.C_in;
    const auto& ssm_state = tensor_args.ssm_state;
    auto& ssm_state_out = std::get<0>(output_tensors);
    auto& y_out = std::get<1>(output_tensors);

    auto* x_buf = x.buffer();
    auto* z_buf = z.buffer();
    auto* dt_buf = dt.buffer();
    auto* dt_bias_buf = dt_bias.buffer();
    auto* A_log_buf = A_log.buffer();
    auto* D_buf = D.buffer();
    auto* B_buf = B_in.buffer();
    auto* C_buf = C_in.buffer();
    auto* ssm_state_buf = ssm_state.buffer();
    auto* ssm_state_out_buf = ssm_state_out.buffer();
    auto* y_out_buf = y_out.buffer();

    // Mamba2 SSD per-block shape constants:
    //   head_dim_tiles = head_dim / TILE_W  (Nemotron-3 Nano: 64 / 32 = 2)
    //   ssm_state_tiles = ssm_state / TILE_W (Nemotron-3 Nano: 128 / 32 = 4)
    const uint32_t head_dim_tiles = x.padded_shape()[-1] / TILE;
    const uint32_t ssm_state_tiles = B_in.padded_shape()[-1] / TILE;
    // SPMD work unit (decision D1): block = (batch, head). Total blocks = B * num_heads.
    const uint32_t batch = x.padded_shape()[0];
    const uint32_t num_heads = x.padded_shape()[1];
    const uint32_t total_blocks = batch * num_heads;

    const bool row_major = true;
    auto grid_size = x.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            split_work_to_cores(grid_size, total_blocks, row_major);

    // Tile size: assume bf16 for non-state tensors, fp32 for ssm_state (decision D4).
    const tt::DataFormat bf16_format = datatype_to_dataformat_converter(x.dtype());
    const uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    const tt::DataFormat fp32_format = datatype_to_dataformat_converter(ssm_state.dtype());
    const uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    // CB allocations — sizes per decision D2 (one block at a time, double-buffered).
    // Per-(batch, head) tile counts:
    //   x, z          : head_dim_tiles tiles  (Nemotron-3: 2)
    //   dt, dt_bias, A_log, D : 1 tile each  (scalars broadcast across 32×32 positions)
    //   B, C          : ssm_state_tiles tiles  (Nemotron-3: 4)
    //   ssm_state, ssm_state_out : head_dim_tiles * ssm_state_tiles tiles  (Nemotron-3: 8)
    //   y             : head_dim_tiles tiles
    // All scaled ×2 for double-buffering. Intermediates sized per their use pattern.
    create_circular_buffer(program, all_cores, CB_X, head_dim_tiles * 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_Z, head_dim_tiles * 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_DT, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_DT_BIAS, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_A_LOG, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_D, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_B, ssm_state_tiles * 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_C, ssm_state_tiles * 2, bf16_tile_size, bf16_format);
    create_circular_buffer(
        program, all_cores, CB_STATE_IN, head_dim_tiles * ssm_state_tiles * 2, fp32_tile_size, fp32_format);
    create_circular_buffer(program, all_cores, CB_DECAY, 2, bf16_tile_size, bf16_format);
    // cb_dt_B double-duty: holds scalar dt_eff at debug_mode=2 (1 tile),
    // holds dt_eff*B vector at debug_mode≥3 (ssm_state_tiles tiles). Size for the larger case.
    create_circular_buffer(program, all_cores, CB_DT_B, ssm_state_tiles * 2, bf16_tile_size, bf16_format);
    create_circular_buffer(
        program, all_cores, CB_STATE_SCALED, head_dim_tiles * ssm_state_tiles * 2, fp32_tile_size, fp32_format);
    create_circular_buffer(program, all_cores, CB_Y_PARTIAL, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(
        program, all_cores, CB_STATE_OUT, head_dim_tiles * ssm_state_tiles * 2, fp32_tile_size, fp32_format);
    create_circular_buffer(program, all_cores, CB_Y, head_dim_tiles * 2, bf16_tile_size, bf16_format);
    // G1 day-4: outer-product scratch (decision D11, see kernel header).
    // cb_x_col is one transposed head-dim tile per d-iter (bf16).
    // cb_outer is one outer-product tile per (d, s) inner-iter.
    //
    // FORMAT FIX (day-5 multi-step): cb_outer is FP32 to match
    // cb_state_scaled in add_state_scaled_outer / _two. When add_tiles
    // mixes fp32 + bf16 sources on Blackhole, the bf16 contribution can
    // be silently dropped — exposed only when state_in=0 makes outer
    // the sole non-zero term. With both fp32, the math is exact.
    // (The single-step smoke missed this because random state_in
    // ~0.3 dominated; outer ~0.0075 looked like roundoff.)
    create_circular_buffer(program, all_cores, CB_X_COL, 2, bf16_tile_size, bf16_format);
    create_circular_buffer(program, all_cores, CB_OUTER, 2, fp32_tile_size, fp32_format);
    // G1 day-4.6: post-update state buffer for mode=5's C·state_out^T reduce.
    // Same size + format as cb_state_in / cb_state_scaled.
    create_circular_buffer(
        program, all_cores, CB_STATE_POST_UPDATE, head_dim_tiles * ssm_state_tiles * 2, fp32_tile_size, fp32_format);

    // Reader compile-time args: 9 CB indices + 9 TensorAccessorArgs.
    std::vector<uint32_t> reader_compile_time_args = {
        CB_X, CB_Z, CB_DT, CB_DT_BIAS, CB_A_LOG, CB_D, CB_B, CB_C, CB_STATE_IN};
    TensorAccessorArgs(x_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(z_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(dt_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(dt_bias_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(A_log_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(D_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(B_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(C_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(ssm_state_buf).append_to(reader_compile_time_args);

    // Compute compile-time args: 15 CB indices in the exact order the kernel reads them
    // via get_compile_time_arg_val(0..14).
    std::vector<uint32_t> compute_compile_time_args = {
        CB_X,
        CB_Z,
        CB_DT,
        CB_DT_BIAS,
        CB_A_LOG,
        CB_D,
        CB_B,
        CB_C,
        CB_STATE_IN,
        CB_DECAY,
        CB_DT_B,
        CB_STATE_SCALED,
        CB_Y_PARTIAL,
        CB_STATE_OUT,
        CB_Y,
        // G1 day-4 additions:
        CB_X_COL,
        CB_OUTER,
        // G1 day-4.6 (mode=5):
        CB_STATE_POST_UPDATE};

    // Writer compile-time args: 2 CB indices + 2 TensorAccessorArgs.
    std::vector<uint32_t> writer_compile_time_args = {CB_STATE_OUT, CB_Y};
    TensorAccessorArgs(ssm_state_out_buf).append_to(writer_compile_time_args);
    TensorAccessorArgs(y_out_buf).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nemotron3_mamba2_decode_owned/device/kernels/dataflow/"
        "reader_nemotron3_mamba2_decode_owned.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nemotron3_mamba2_decode_owned/device/kernels/dataflow/"
        "writer_nemotron3_mamba2_decode_owned.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nemotron3_mamba2_decode_owned/device/kernels/compute/"
        "nemotron3_mamba2_decode_owned.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,  // decision D4
            .compile_args = compute_compile_time_args});

    auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, row_major);
    Nemotron3Mamba2DecodeOwnedSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .num_cores = num_cores,
        .g1_numcores = core_group_1.num_cores(),
        .g2_numcores = core_group_2.num_cores(),
        .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
        .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
        .head_dim_tiles = head_dim_tiles,
        .ssm_state_tiles = ssm_state_tiles,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensors);
    return cached_program;
}

void Nemotron3Mamba2DecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Nemotron3Mamba2DecodeOwnedParams& operation_attributes,
    const Nemotron3Mamba2DecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    auto& ssm_state_out = std::get<0>(output_tensors);
    auto& y_out = std::get<1>(output_tensors);

    auto* x_buf = tensor_args.x.buffer();
    auto* z_buf = tensor_args.z.buffer();
    auto* dt_buf = tensor_args.dt.buffer();
    auto* dt_bias_buf = tensor_args.dt_bias.buffer();
    auto* A_log_buf = tensor_args.A_log.buffer();
    auto* D_buf = tensor_args.D.buffer();
    auto* B_buf = tensor_args.B_in.buffer();
    auto* C_buf = tensor_args.C_in.buffer();
    auto* ssm_state_buf = tensor_args.ssm_state.buffer();
    auto* ssm_state_out_buf = ssm_state_out.buffer();
    auto* y_out_buf = y_out.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t debug_mode = operation_attributes.debug_mode != 0 ? operation_attributes.debug_mode
                                                                     : (operation_attributes.debug_fill ? 1 : 0);

    // Reader runtime args (per core): 9 DRAM addrs + start_block + num_blocks_per_core
    //                                + head_dim_tiles + ssm_state_tiles
    std::vector<std::vector<uint32_t>> reader_runtime_args(shared.cores.size(), std::vector<uint32_t>(13, 0));
    // Compute runtime args (per core): block_count + head_dim_tiles + ssm_state_tiles + debug_mode
    std::vector<std::vector<uint32_t>> compute_runtime_args(shared.cores.size(), std::vector<uint32_t>(4, 0));
    // Writer runtime args (per core): 2 DRAM addrs + start_block + num_blocks_per_core
    //                                + head_dim_tiles + ssm_state_tiles
    std::vector<std::vector<uint32_t>> writer_runtime_args(shared.cores.size(), std::vector<uint32_t>(6, 0));

    uint32_t blocks_written = 0;
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const uint32_t blocks_per_core =
            i < shared.g1_numcores ? shared.num_blocks_per_core_group_1 : shared.num_blocks_per_core_group_2;

        // Reader
        reader_runtime_args[i][0] = x_buf->address();
        reader_runtime_args[i][1] = z_buf->address();
        reader_runtime_args[i][2] = dt_buf->address();
        reader_runtime_args[i][3] = dt_bias_buf->address();
        reader_runtime_args[i][4] = A_log_buf->address();
        reader_runtime_args[i][5] = D_buf->address();
        reader_runtime_args[i][6] = B_buf->address();
        reader_runtime_args[i][7] = C_buf->address();
        reader_runtime_args[i][8] = ssm_state_buf->address();
        reader_runtime_args[i][9] = blocks_written;
        reader_runtime_args[i][10] = blocks_per_core;
        reader_runtime_args[i][11] = shared.head_dim_tiles;
        reader_runtime_args[i][12] = shared.ssm_state_tiles;

        // Compute
        compute_runtime_args[i][0] = blocks_per_core;
        compute_runtime_args[i][1] = shared.head_dim_tiles;
        compute_runtime_args[i][2] = shared.ssm_state_tiles;
        compute_runtime_args[i][3] = debug_mode;

        // Writer
        writer_runtime_args[i][0] = ssm_state_out_buf->address();
        writer_runtime_args[i][1] = y_out_buf->address();
        writer_runtime_args[i][2] = blocks_written;
        writer_runtime_args[i][3] = blocks_per_core;
        writer_runtime_args[i][4] = shared.head_dim_tiles;
        writer_runtime_args[i][5] = shared.ssm_state_tiles;

        blocks_written += blocks_per_core;
    }

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores, reader_runtime_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores, compute_runtime_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores, writer_runtime_args);
}

}  // namespace ttnn::experimental::prim
