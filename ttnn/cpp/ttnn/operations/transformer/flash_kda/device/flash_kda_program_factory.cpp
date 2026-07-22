// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the Flash KDA recurrent state update kernel.
//
// Parallelism: one Tensix core per item (mirrors gated_delta_attn's one-core-per-head
// convention). Each core performs the full single-step recurrence once (no chunk loop).

#include "ttnn/operations/transformer/flash_kda/device/flash_kda_program_factory.hpp"

#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

FlashKdaProgramFactory::cached_program_t FlashKdaProgramFactory::create(
    const FlashKdaParams& attrs, const FlashKdaInputs& in, std::vector<Tensor>& outputs) {
    Program program{};

    const std::uint32_t N = attrs.num_items;
    const std::uint32_t Dk = in.k.logical_shape()[2];
    const std::uint32_t Dv = in.v.logical_shape()[2];

    const std::uint32_t Kt = Dk / TILE_WIDTH;
    const std::uint32_t Vt = Dv / TILE_WIDTH;
    const std::uint32_t state_tiles = Kt * Vt;

    // Build per-item core coordinates (column-major to avoid harvested NOC columns).
    auto* device = in.S_prev.device();
    CoreCoord compute_grid = device->compute_with_storage_grid_size();
    const std::uint32_t grid_y = compute_grid.y;

    TT_FATAL(
        N <= compute_grid.x * grid_y,
        "num_items {} exceeds total compute cores {}×{}={}",
        N,
        compute_grid.x,
        grid_y,
        compute_grid.x * grid_y);

    std::vector<CoreCoord> item_cores(N);
    for (std::uint32_t i = 0; i < N; i++) {
        item_cores[i] = CoreCoord{i / grid_y, i % grid_y};
    }

    std::set<CoreRange> core_set;
    for (auto& c : item_cores) {
        core_set.insert(CoreRange{c, c});
    }
    CoreRangeSet cores{core_set};

    tt::DataFormat df_f32 = tt::DataFormat::Float32;

    // -----------------------------------------------------------------------
    // Circular buffers — all fp32, single-buffered (one-shot per core, no chunk loop).
    // -----------------------------------------------------------------------
    auto make_cb = [&](std::uint32_t idx, std::uint32_t n_tiles) {
        std::uint32_t sz = n_tiles * tt::tile_size(df_f32);
        CircularBufferConfig cfg(sz, {{idx, df_f32}});
        cfg.set_page_size(idx, tt::tile_size(df_f32));
        CreateCircularBuffer(program, cores, cfg);
    };

    make_cb(0, state_tiles);  // S_prev [Dk,Dv]
    make_cb(1, Kt);           // g [Dk,1]
    make_cb(2, Kt);           // k [1,Dk]
    make_cb(3, Vt);           // v [1,Dv]
    make_cb(4, 1);            // beta [1,1]
    make_cb(5, Kt);           // q [1,Dk]

    make_cb(6, Kt);            // k_col (transpose of k)
    make_cb(7, state_tiles);   // S_tilde = S_prev * g
    make_cb(8, Vt);            // pred = k @ S_tilde
    make_cb(9, Vt);            // err = v - pred
    make_cb(10, Vt);           // delta = beta * err
    make_cb(11, Vt);           // delta_bcast
    make_cb(14, state_tiles);  // outer = k outer delta

    make_cb(12, state_tiles);  // output 0: S_new
    make_cb(13, Vt);           // output 1: out

    // -----------------------------------------------------------------------
    // Kernel paths
    // -----------------------------------------------------------------------
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/flash_kda/device/kernels/";

    const std::vector<std::uint32_t> ct_args = {Kt, Vt};

    // Reader/writer also carry per-tensor TensorAccessorArgs compile-time blocks, appended
    // right after {Kt, Vt} in the SAME order the kernels consume them (reader: S_prev, g, k,
    // v, beta, q; writer: S_new, out).
    std::vector<std::uint32_t> reader_ct_args = ct_args;
    TensorAccessorArgs(in.S_prev.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.g.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.k.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.v.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.beta.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(in.q.buffer()).append_to(reader_ct_args);

    std::vector<std::uint32_t> writer_ct_args = ct_args;
    TensorAccessorArgs(outputs[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(outputs[1].buffer()).append_to(writer_ct_args);

    auto reader_id = CreateKernel(
        program,
        kdir + "dataflow/reader_flash_kda.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    auto writer_id = CreateKernel(
        program,
        kdir + "dataflow/writer_flash_kda.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    auto compute_id = CreateKernel(
        program,
        kdir + "compute/flash_kda.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = ct_args});

    // -----------------------------------------------------------------------
    // Per-core runtime arguments
    // -----------------------------------------------------------------------
    std::uint32_t s_prev_addr = in.S_prev.buffer()->address();
    std::uint32_t g_addr = in.g.buffer()->address();
    std::uint32_t k_addr = in.k.buffer()->address();
    std::uint32_t v_addr = in.v.buffer()->address();
    std::uint32_t beta_addr = in.beta.buffer()->address();
    std::uint32_t q_addr = in.q.buffer()->address();

    std::uint32_t s_new_addr = outputs[0].buffer()->address();
    std::uint32_t out_addr = outputs[1].buffer()->address();

    for (std::uint32_t i = 0; i < N; i++) {
        auto& core = item_cores[i];
        SetRuntimeArgs(program, reader_id, core, {i, s_prev_addr, g_addr, k_addr, v_addr, beta_addr, q_addr});
        SetRuntimeArgs(program, writer_id, core, {i, s_new_addr, out_addr});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_id,
            .writer_kernel_id = writer_id,
            .compute_kernel_id = compute_id,
            .grid_y = grid_y,
            .num_cores = N,
        }};
}

void FlashKdaProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const FlashKdaParams& attrs, const FlashKdaInputs& in, std::vector<Tensor>& outputs) {
    auto& program = cached.program;
    auto& sv = cached.shared_variables;
    const std::uint32_t N = attrs.num_items;
    const std::uint32_t grid_y = sv.grid_y;

    std::uint32_t s_prev_addr = in.S_prev.buffer()->address();
    std::uint32_t g_addr = in.g.buffer()->address();
    std::uint32_t k_addr = in.k.buffer()->address();
    std::uint32_t v_addr = in.v.buffer()->address();
    std::uint32_t beta_addr = in.beta.buffer()->address();
    std::uint32_t q_addr = in.q.buffer()->address();
    std::uint32_t s_new_addr = outputs[0].buffer()->address();
    std::uint32_t out_addr = outputs[1].buffer()->address();

    for (std::uint32_t i = 0; i < N; i++) {
        CoreCoord core{i / grid_y, i % grid_y};

        auto& ra = GetRuntimeArgs(program, sv.reader_kernel_id, core);
        ra[1] = s_prev_addr;
        ra[2] = g_addr;
        ra[3] = k_addr;
        ra[4] = v_addr;
        ra[5] = beta_addr;
        ra[6] = q_addr;

        auto& wa = GetRuntimeArgs(program, sv.writer_kernel_id, core);
        wa[1] = s_new_addr;
        wa[2] = out_addr;
    }
}

}  // namespace ttnn::prim
