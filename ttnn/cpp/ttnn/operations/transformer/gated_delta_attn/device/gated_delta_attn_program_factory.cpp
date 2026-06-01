// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the gated-delta-attention sequential scan kernel (Path A).
//
// Parallelism: one Tensix core per head.
// Each core: forward substitution (using pre-computed L_inv) + inter-chunk state scan.

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_program_factory.hpp"

#include <cstdlib>
#include <optional>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
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

    const uint32_t Ct = C / TILE_HEIGHT;         // e.g. 4
    const uint32_t Kt = Dk / TILE_WIDTH;         // e.g. 4
    const uint32_t Vt_global = Dv / TILE_WIDTH;  // full value-tile count, e.g. 4

    // Multi-core-per-head value split (env-tunable). Each head's value dim is sharded across
    // split_v cores so the per-core CB footprint shrinks (fits below the TP=32 persistent L1
    // buffers -> no clash) and the work parallelizes. split_v=1 == original one-core-per-head.
    const char* _sph = std::getenv("QWEN36_SEQ_CORES_PER_HEAD");
    uint32_t split_v = _sph ? static_cast<uint32_t>(std::atoi(_sph)) : 1u;
    if (split_v < 1) {
        split_v = 1;
    }
    TT_FATAL(
        Vt_global % split_v == 0, "value tiles {} not divisible by QWEN36_SEQ_CORES_PER_HEAD {}", Vt_global, split_v);
    const uint32_t Vt = Vt_global / split_v;  // LOCAL value-tile count (drives CB sizes + ct_args)
    const uint32_t num_cores = BH * split_v;

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
        num_cores <= compute_grid.x * grid_y,
        "cores {} (heads {} x split_v {}) exceed total compute cores {}×{}={}",
        num_cores,
        BH,
        split_v,
        compute_grid.x,
        grid_y,
        compute_grid.x * grid_y);

    // Column offset (env-tunable): the default column 0 collides with the galaxy's
    // TP=32 persistent CCL L1 buffers (see llama_ccl.py "static-CB clash on core (0,0)").
    const char* _col_env = std::getenv("QWEN36_SEQ_CORE_COL");
    const uint32_t col_off = _col_env ? static_cast<uint32_t>(std::atoi(_col_env)) : 0;

    // One core per (head, value-slice). Core i handles head=i/split_v, v_slice=i%split_v.
    std::vector<CoreCoord> head_cores(num_cores);
    for (uint32_t i = 0; i < num_cores; i++) {
        head_cores[i] = CoreCoord{col_off + i / grid_y, i % grid_y};
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
    make_cb(24, df_f32, out_tiles, 2);    // out (writer pops, DOUBLE-buffered — bisect: testing if
                                          // single-buffering CB24 was the coherence bug. At split_v=4
                                          // out_tiles is tiny so double-buffer still fits L1.)
    make_cb(25, df_f32, state_tiles, 1);  // s_upd
    make_cb(26, df_f32, state_tiles, 1);  // S_tmp
    make_cb(27, df_f32, state_tiles, 1);  // final_state (writer reads once)

    // -----------------------------------------------------------------------
    // Kernel paths
    // -----------------------------------------------------------------------
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gated_delta_attn/device/kernels/";

    std::vector<uint32_t> ct_args = {Ct, Kt, Vt};

    auto reader_id = CreateKernel(
        program,
        kdir + "dataflow/reader_gated_delta_attn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ct_args});

    auto writer_id = CreateKernel(
        program,
        kdir + "dataflow/writer_gated_delta_attn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

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

    for (uint32_t i = 0; i < num_cores; i++) {
        auto& core = head_cores[i];
        uint32_t head = i / split_v;
        uint32_t v_off = (i % split_v) * Vt;  // Vt = local value-tile count
        SetRuntimeArgs(
            program,
            reader_id,
            core,
            {head,
             NC,
             lu_addr,
             vbs_addr,
             kbs_addr,
             att_addr,
             qdec_addr,
             kdt_addr,
             dle_addr,
             linv_addr,
             s0_addr,
             v_off,
             Vt_global});
        SetRuntimeArgs(program, writer_id, core, {head, NC, out_addr, state_addr, v_off, Vt_global});
        SetRuntimeArgs(program, compute_id, core, {NC});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_id,
            .writer_kernel_id = writer_id,
            .compute_kernel_id = compute_id,
            .grid_y = grid_y,
            .num_cores = num_cores,
            .split_v = split_v,
            .Vt_global = Vt_global,
        }};
}

void GatedDeltaAttnSeqProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const GatedDeltaAttnSeqParams& attrs,
    const GatedDeltaAttnSeqInputs& in,
    std::vector<Tensor>& outputs) {
    (void)attrs;  // shapes unchanged across invocations; only buffer addresses are refreshed
    auto& program = cached.program;
    auto& sv = cached.shared_variables;
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

    const char* _col_env2 = std::getenv("QWEN36_SEQ_CORE_COL");
    const uint32_t col_off2 = _col_env2 ? static_cast<uint32_t>(std::atoi(_col_env2)) : 0;
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        CoreCoord core{col_off2 + i / grid_y, i % grid_y};
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
