// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the gated-delta-attention sequential scan kernel (Path A).
//
// Parallelism: one Tensix core per head.
// Each core: forward substitution (using pre-computed L_inv) + inter-chunk state scan.

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_program_factory.hpp"

#include <optional>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

GatedDeltaAttnSeqProgramFactory::cached_program_t GatedDeltaAttnSeqProgramFactory::create(
    const GatedDeltaAttnSeqParams& attrs, const GatedDeltaAttnSeqInputs& in, std::vector<Tensor>& outputs) {
    Program program{};

    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;

    const uint32_t Ct = C / TILE_HEIGHT;  // e.g. 4
    const uint32_t Kt = Dk / TILE_WIDTH;  // e.g. 4
    const uint32_t Vt = Dv / TILE_WIDTH;  // e.g. 4

    const uint32_t state_tiles = Kt * Vt;
    const uint32_t out_tiles = Ct * Vt;
    const uint32_t in_kv_tiles = Ct * Kt;
    const uint32_t attn_tiles = Ct * Ct;
    const uint32_t kdt_tiles = Kt * Ct;

    // Build per-head core coordinates (column-major to avoid harvested NOC columns).
    auto* device = in.L_unit.device();
    CoreCoord compute_grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = compute_grid.y;

    TT_FATAL(
        BH <= compute_grid.x * grid_y,
        "num_heads {} exceeds total compute cores {}×{}={}",
        BH,
        compute_grid.x,
        grid_y,
        compute_grid.x * grid_y);

    std::vector<CoreCoord> head_cores(BH);
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
    }

    std::set<CoreRange> core_set;
    for (auto& c : head_cores) {
        core_set.insert(CoreRange{c, c});
    }
    CoreRangeSet cores{core_set};

    tt::DataFormat df_f32 = tt::DataFormat::Float32;

    // -----------------------------------------------------------------------
    // Circular buffers — all fp32
    // -----------------------------------------------------------------------
    auto make_cb = [&](uint32_t idx, tt::DataFormat fmt, uint32_t n_tiles, uint32_t n_bufs = 1) {
        uint32_t sz = n_tiles * n_bufs * tt::tile_size(fmt);
        CircularBufferConfig cfg(sz, {{idx, fmt}});
        cfg.set_page_size(idx, tt::tile_size(fmt));
        CreateCircularBuffer(program, cores, cfg);
    };

    // Per-chunk inputs — single-buffered to stay within L1 budget (1.5 MB).
    make_cb(0, df_f32, attn_tiles, 1);   // L_unit [C,C]
    make_cb(1, df_f32, out_tiles, 1);    // v_beta_sc [C,Dv]
    make_cb(2, df_f32, in_kv_tiles, 1);  // k_bd_sc [C,Dk]
    make_cb(3, df_f32, attn_tiles, 1);   // intra_attn [C,C]
    make_cb(4, df_f32, in_kv_tiles, 1);  // q_decay [C,Dk]
    make_cb(5, df_f32, kdt_tiles, 1);    // k_decay_t [Dk,C]
    make_cb(6, df_f32, 1, 1);            // dl_exp (fp32 scalar)
    // CB7: unused (was identity_32)
    make_cb(8, df_f32, state_tiles, 1);  // S — persistent state

    // Forward-substitution scratch
    // CBs 9-11 hold max(Vt, Kt) tiles for fwd_sub rows.
    const uint32_t scratch_Xt = std::max(Vt, Kt);
    make_cb(9, df_f32, scratch_Xt, 1);   // fwd_rhs
    make_cb(10, df_f32, scratch_Xt, 1);  // corr_mm
    make_cb(11, df_f32, scratch_Xt, 1);  // temp_rhs
    // CB12, CB13: unused (were Neumann-only)

    // Diagonal block inverses — 1 tile each, loaded per chunk by reader
    make_cb(14, df_f32, 1, 1);  // L_inv0
    make_cb(15, df_f32, 1, 1);  // L_inv1
    make_cb(16, df_f32, 1, 1);  // L_inv2
    make_cb(17, df_f32, 1, 1);  // L_inv3

    // Forward-substitution outputs
    make_cb(18, df_f32, out_tiles, 1);    // v_cor = L^{-1} @ v_beta_sc
    make_cb(19, df_f32, in_kv_tiles, 1);  // k_cum = L^{-1} @ k_bd_sc

    // 7-step scratch
    make_cb(20, df_f32, out_tiles, 1);    // v_prime
    make_cb(21, df_f32, out_tiles, 1);    // v_new
    make_cb(22, df_f32, out_tiles, 1);    // o_inter
    make_cb(23, df_f32, out_tiles, 1);    // intra_v
    make_cb(24, df_f32, out_tiles, 2);    // out (writer pops, double-buffered)
    make_cb(25, df_f32, state_tiles, 1);  // s_upd
    make_cb(26, df_f32, state_tiles, 1);  // S_tmp
    make_cb(27, df_f32, state_tiles, 1);  // final_state (writer reads once)

    // -----------------------------------------------------------------------
    // Kernel paths
    // -----------------------------------------------------------------------
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gated_delta_attn/device/kernels/";

    const std::vector<uint32_t> ct_args = {Ct, Kt, Vt};

    // Reader/writer also carry per-tensor TensorAccessorArgs compile-time blocks,
    // appended right after {Ct, Kt, Vt} in the SAME order the kernels consume them
    // (reader: L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp,
    // L_inv, initial_state; writer: out, final_state). Each interleaved-DRAM tensor
    // appends two args, so the device-side TensorAccessorArgs<3> chain stays aligned.
    // initial_state is optional: a null buffer still appends two (zeroed) args so the
    // s0 accessor's compile-time offset is unconditionally present.
    std::vector<uint32_t> reader_ct_args = ct_args;
    TensorAccessorArgs(in.L_unit.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.v_beta_sc.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.k_bd_sc.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.intra_attn.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.q_decay.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.k_decay_t.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.dl_exp.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.L_inv.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = ct_args;
    TensorAccessorArgs(outputs[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(outputs[1].buffer()).append_to(writer_ct_args);

    auto reader_id = CreateKernel(
        program,
        kdir + "dataflow/reader_gated_delta_attn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    auto writer_id = CreateKernel(
        program,
        kdir + "dataflow/writer_gated_delta_attn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    auto compute_id = CreateKernel(
        program,
        kdir + "compute/gated_delta_attn.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = ct_args});

    // -----------------------------------------------------------------------
    // Per-core runtime arguments
    // -----------------------------------------------------------------------
    uint32_t lu_addr = in.L_unit.buffer()->address();
    uint32_t vbs_addr = in.v_beta_sc.buffer()->address();
    uint32_t kbs_addr = in.k_bd_sc.buffer()->address();
    uint32_t att_addr = in.intra_attn.buffer()->address();
    uint32_t qdec_addr = in.q_decay.buffer()->address();
    uint32_t kdt_addr = in.k_decay_t.buffer()->address();
    uint32_t dle_addr = in.dl_exp.buffer()->address();
    uint32_t linv_addr = in.L_inv.buffer()->address();
    uint32_t s0_addr = in.initial_state.has_value() ? in.initial_state->buffer()->address() : 0u;

    uint32_t out_addr = outputs[0].buffer()->address();
    uint32_t state_addr = outputs[1].buffer()->address();

    for (uint32_t h = 0; h < BH; h++) {
        auto& core = head_cores[h];
        SetRuntimeArgs(
            program,
            reader_id,
            core,
            {h, NC, lu_addr, vbs_addr, kbs_addr, att_addr, qdec_addr, kdt_addr, dle_addr, linv_addr, s0_addr});
        SetRuntimeArgs(program, writer_id, core, {h, NC, out_addr, state_addr});
        SetRuntimeArgs(program, compute_id, core, {NC});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_id,
            .writer_kernel_id = writer_id,
            .compute_kernel_id = compute_id,
            .grid_y = grid_y,
            .num_cores = BH,
        }};
}

void GatedDeltaAttnSeqProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const GatedDeltaAttnSeqParams& attrs,
    const GatedDeltaAttnSeqInputs& in,
    std::vector<Tensor>& outputs) {
    auto& program = cached.program;
    auto& sv = cached.shared_variables;
    const uint32_t BH = attrs.num_heads;
    const uint32_t grid_y = sv.grid_y;

    uint32_t lu_addr = in.L_unit.buffer()->address();
    uint32_t vbs_addr = in.v_beta_sc.buffer()->address();
    uint32_t kbs_addr = in.k_bd_sc.buffer()->address();
    uint32_t att_addr = in.intra_attn.buffer()->address();
    uint32_t qdec_addr = in.q_decay.buffer()->address();
    uint32_t kdt_addr = in.k_decay_t.buffer()->address();
    uint32_t dle_addr = in.dl_exp.buffer()->address();
    uint32_t linv_addr = in.L_inv.buffer()->address();
    uint32_t s0_addr = in.initial_state.has_value() ? in.initial_state->buffer()->address() : 0u;
    uint32_t out_addr = outputs[0].buffer()->address();
    uint32_t state_addr = outputs[1].buffer()->address();

    for (uint32_t h = 0; h < BH; h++) {
        CoreCoord core{h / grid_y, h % grid_y};
        auto& ra = GetRuntimeArgs(program, sv.reader_kernel_id, core);
        ra[2] = lu_addr;
        ra[3] = vbs_addr;
        ra[4] = kbs_addr;
        ra[5] = att_addr;
        ra[6] = qdec_addr;
        ra[7] = kdt_addr;
        ra[8] = dle_addr;
        ra[9] = linv_addr;
        ra[10] = s0_addr;

        auto& wa = GetRuntimeArgs(program, sv.writer_kernel_id, core);
        wa[2] = out_addr;
        wa[3] = state_addr;
    }
}

}  // namespace ttnn::prim
