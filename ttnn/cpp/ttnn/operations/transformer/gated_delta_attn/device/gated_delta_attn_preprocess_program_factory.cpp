// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_preprocess_program_factory.hpp"

#include <cstring>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

GatedDeltaAttnPreprocessProgramFactory::cached_program_t GatedDeltaAttnPreprocessProgramFactory::create(
    const GatedDeltaAttnPreprocessParams& attrs,
    const GatedDeltaAttnPreprocessInputs& in,
    std::vector<Tensor>& outputs) {
    Program program{};

    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;
    const uint32_t Ct = C / TILE_HEIGHT;
    const uint32_t Kt = Dk / TILE_WIDTH;
    const uint32_t Vt = Dv / TILE_WIDTH;

    auto* device = in.q.device();
    CoreCoord compute_grid = device->compute_with_storage_grid_size();
    const uint32_t total_cores = compute_grid.x * compute_grid.y;
    const uint32_t num_work = BH * NC;
    const uint32_t num_cores = std::min(total_cores, num_work);
    const uint32_t grid_y = compute_grid.y;

    std::vector<CoreCoord> cores_vec(num_cores);
    std::set<CoreRange> core_set;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores_vec[i] = CoreCoord{i / grid_y, i % grid_y};
        core_set.insert(CoreRange{cores_vec[i], cores_vec[i]});
    }
    CoreRangeSet cores{core_set};

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gated_delta_attn/device/kernels/";

    const tt::DataFormat df_f32 = tt::DataFormat::Float32;
    const uint32_t f32_tile = tt::tile_size(df_f32);
    auto make_cb = [&](uint32_t idx, uint32_t n_tiles) {
        CircularBufferConfig cfg(n_tiles * f32_tile, {{idx, df_f32}});
        cfg.set_page_size(idx, f32_tile);
        CreateCircularBuffer(program, cores, cfg);
    };

    const uint32_t attn_tiles = Ct * Ct;
    const uint32_t out_tiles = Ct * Vt;
    const uint32_t in_kv_tiles = Ct * Kt;
    const uint32_t kdt_tiles = Kt * Ct;
    const uint32_t beta_tiles = Ct;
    const uint32_t g_tiles = Ct;

    // Input CBs
    make_cb(0, in_kv_tiles);  // q
    make_cb(1, in_kv_tiles);  // k
    make_cb(2, out_tiles);    // v
    make_cb(3, beta_tiles);   // beta
    make_cb(4, g_tiles);      // g
    make_cb(5, attn_tiles);   // triu
    make_cb(6, attn_tiles);   // tril
    make_cb(7, attn_tiles);   // eye
    make_cb(8, attn_tiles);   // lower causal
    make_cb(9, 1);            // eye_32

    // Output CBs
    make_cb(10, attn_tiles);   // L_unit
    make_cb(11, out_tiles);    // v_beta_sc
    make_cb(12, in_kv_tiles);  // k_bd_sc
    make_cb(13, attn_tiles);   // intra_attn
    make_cb(14, in_kv_tiles);  // q_decay
    make_cb(15, kdt_tiles);    // k_decay_t
    make_cb(16, 1);            // dl_exp
    make_cb(17, Ct);           // L_inv
    make_cb(18, attn_tiles);   // qk scratch
    make_cb(19, kdt_tiles);    // k^T scratch
    make_cb(20, attn_tiles);   // diagonal/strict scratch

    // Decay-path scratch/state CBs (see compute kernel).
    make_cb(21, Ct);           // ones (Ct all-ones tiles; filled once)
    make_cb(22, Ct);           // decay_col   [C,1] prefix-sum of g
    make_cb(23, Ct);           // decay_row   [1,C] transpose(decay_col)
    make_cb(24, Ct);           // decay_exp   [C,1] exp(clamp(decay_col))
    make_cb(25, Ct);           // D_inv / dl_raw temp [C,1]
    make_cb(26, attn_tiles);   // L_mask      [C,C] (kept for L_mat + intra_attn)
    make_cb(27, in_kv_tiles);  // k^T raw     [K,C] transpose(k) (kept for kk + intra_attn)
    make_cb(28, Ct);           // decay_diff_exp [C,1]

    std::vector<uint32_t> reader_ct_args = {Ct, Kt, Vt};
    TensorAccessorArgs(in.q.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.k.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.v.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.beta.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.g.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.triu_ones.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.tril_mask.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.eye.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.lower_causal.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.eye_32.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {Ct, Kt, Vt};
    for (auto& output : outputs) {
        TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    }

    auto reader_id = CreateKernel(
        program,
        kdir + "dataflow/reader_gated_delta_attn_preprocess.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    auto writer_id = CreateKernel(
        program,
        kdir + "dataflow/writer_gated_delta_attn_preprocess.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // diag_alpha as float bits: the compute kernel forms D_diag = 1 + alpha*diag(kk) for the
    // partial-damping regularization (alpha=0 -> unit diagonal; alpha=1 -> full damping).
    const float alpha = attrs.diag_alpha;
    uint32_t alpha_bits = 0;
    std::memcpy(&alpha_bits, &alpha, sizeof(uint32_t));

    auto compute_id = CreateKernel(
        program,
        kdir + "compute/gated_delta_attn_preprocess.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {Ct, Kt, Vt, alpha_bits}});

    std::vector<uint32_t> output_addrs;
    output_addrs.reserve(outputs.size());
    for (auto& output : outputs) {
        output_addrs.push_back(output.buffer()->address());
    }

    std::vector<uint32_t> input_addrs = {
        in.q.buffer()->address(),
        in.k.buffer()->address(),
        in.v.buffer()->address(),
        in.beta.buffer()->address(),
        in.g.buffer()->address(),
        in.triu_ones.buffer()->address(),
        in.tril_mask.buffer()->address(),
        in.eye.buffer()->address(),
        in.lower_causal.buffer()->address(),
        in.eye_32.buffer()->address(),
    };

    for (uint32_t i = 0; i < num_cores; i++) {
        std::vector<uint32_t> rargs = {i, num_work, NC, num_cores};
        rargs.insert(rargs.end(), input_addrs.begin(), input_addrs.end());
        SetRuntimeArgs(program, reader_id, cores_vec[i], rargs);

        std::vector<uint32_t> args = {i, num_work, NC};
        args.insert(args.end(), output_addrs.begin(), output_addrs.end());
        args.push_back(num_cores);
        SetRuntimeArgs(program, writer_id, cores_vec[i], args);
        SetRuntimeArgs(program, compute_id, cores_vec[i], {i, num_work, num_cores});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_id,
            .writer_kernel_id = writer_id,
            .compute_kernel_id = compute_id,
            .grid_y = grid_y,
            .num_cores = num_cores,
        }};
}

void GatedDeltaAttnPreprocessProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const GatedDeltaAttnPreprocessParams& attrs,
    const GatedDeltaAttnPreprocessInputs& in,
    std::vector<Tensor>& outputs) {
    auto& program = cached.program;
    auto& sv = cached.shared_variables;

    std::vector<uint32_t> output_addrs;
    output_addrs.reserve(outputs.size());
    for (auto& output : outputs) {
        output_addrs.push_back(output.buffer()->address());
    }

    const uint32_t num_work = attrs.num_heads * attrs.num_chunks;
    std::vector<uint32_t> input_addrs = {
        in.q.buffer()->address(),
        in.k.buffer()->address(),
        in.v.buffer()->address(),
        in.beta.buffer()->address(),
        in.g.buffer()->address(),
        in.triu_ones.buffer()->address(),
        in.tril_mask.buffer()->address(),
        in.eye.buffer()->address(),
        in.lower_causal.buffer()->address(),
        in.eye_32.buffer()->address(),
    };
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        CoreCoord core{i / sv.grid_y, i % sv.grid_y};
        auto& rargs = GetRuntimeArgs(program, sv.reader_kernel_id, core);
        rargs[0] = i;
        rargs[1] = num_work;
        rargs[2] = attrs.num_chunks;
        rargs[3] = sv.num_cores;
        for (uint32_t j = 0; j < input_addrs.size(); j++) {
            rargs[4 + j] = input_addrs[j];
        }

        auto& args = GetRuntimeArgs(program, sv.writer_kernel_id, core);
        args[0] = i;
        args[1] = num_work;
        args[2] = attrs.num_chunks;
        for (uint32_t j = 0; j < output_addrs.size(); j++) {
            args[3 + j] = output_addrs[j];
        }
        args[11] = sv.num_cores;

        auto& cargs = GetRuntimeArgs(program, sv.compute_kernel_id, core);
        cargs[0] = i;
        cargs[1] = num_work;
        cargs[2] = sv.num_cores;
    }
}

}  // namespace ttnn::prim
