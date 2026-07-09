// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lnbw_program_factory.hpp"
#include "lnbw_device_operation_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static const char* kLnBwKernelDir = "ttnn/cpp/ttnn/operations/experimental/fused_rotate/device/kernels/";

LnBwProgramFactory::cached_program_t LnBwProgramFactory::create(
    const LnBwParams& attrs, const LnBwInputs& inputs, Tensor& output) {
    Program program{};

    const auto& gy = inputs.gy;  // g_out (matmul, pre-silu-bw)
    const auto& x = inputs.x;
    const auto& red = inputs.red;
    const auto& n = inputs.n;
    const auto& gamma = inputs.gamma;

    const uint32_t Wt = attrs.W / TILE_WIDTH;
    const uint32_t Et = gy.padded_shape()[-2] / TILE_HEIGHT;

    tt::DataFormat data_format = datatype_to_dataformat_converter(gy.dtype());
    const uint32_t tile_bytes = tile_size(data_format);

    auto* device = gy.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, Et);

    auto make_cb = [&](uint32_t cb_index, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * tile_bytes, {{cb_index, data_format}}).set_page_size(cb_index, tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    constexpr uint32_t cb_gout = tt::CBIndex::c_0;   // g_out (matmul)
    constexpr uint32_t cb_x = tt::CBIndex::c_1;
    constexpr uint32_t cb_red = tt::CBIndex::c_2;
    constexpr uint32_t cb_xc = tt::CBIndex::c_3;
    constexpr uint32_t cb_xhat = tt::CBIndex::c_4;
    constexpr uint32_t cb_prod = tt::CBIndex::c_5;
    constexpr uint32_t cb_s = tt::CBIndex::c_6;
    constexpr uint32_t cb_rstd = tt::CBIndex::c_7;
    constexpr uint32_t cb_n = tt::CBIndex::c_8;       // pre-silu activation
    constexpr uint32_t cb_gamma = tt::CBIndex::c_9;   // LN affine scale [1,W], resident
    constexpr uint32_t cb_gy = tt::CBIndex::c_10;     // internal: g_out*silu'(n)*gamma
    constexpr uint32_t cb_g1 = tt::CBIndex::c_11;     // internal preamble scratch
    constexpr uint32_t cb_dx = tt::CBIndex::c_16;
    make_cb(cb_gout, 2 * Wt);
    make_cb(cb_x, 2 * Wt);
    make_cb(cb_red, 2);
    make_cb(cb_xc, 2 * Wt);
    make_cb(cb_xhat, 2 * Wt);
    make_cb(cb_prod, 2 * Wt);
    make_cb(cb_s, 4);
    make_cb(cb_rstd, 2);
    make_cb(cb_n, 2 * Wt);
    make_cb(cb_gamma, Wt);
    make_cb(cb_gy, 2 * Wt);
    make_cb(cb_g1, 2 * Wt);
    make_cb(cb_dx, 2 * Wt);

    // ---- reader ----
    std::vector<uint32_t> reader_ct = {cb_gout, cb_x, cb_red, Wt, tile_bytes, cb_n, cb_gamma};
    TensorAccessorArgs(*gy.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*red.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*n.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*gamma.buffer()).append_to(reader_ct);
    KernelHandle reader_id = CreateKernel(
        program, std::string(kLnBwKernelDir) + "lnbw_reader.cpp", all_cores, ReaderDataMovementConfig(reader_ct));

    // ---- writer ----
    std::vector<uint32_t> writer_ct = {cb_dx, Wt, tile_bytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    KernelHandle writer_id = CreateKernel(
        program, std::string(kLnBwKernelDir) + "lnbw_writer.cpp", all_cores, WriterDataMovementConfig(writer_ct));

    // ---- compute ----
    std::vector<uint32_t> compute_ct = {
        cb_gout, cb_x, cb_red, cb_xc, cb_xhat, cb_prod, cb_s, cb_rstd, cb_dx, Wt, attrs.eps_bits,
        cb_n, cb_gamma, cb_gy, cb_g1};
    KernelHandle compute_id = CreateKernel(
        program,
        std::string(kLnBwKernelDir) + "lnbw_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_ct});

    auto* gy_buf = gy.buffer();
    auto* x_buf = x.buffer();
    auto* red_buf = red.buffer();
    auto* n_buf = n.buffer();
    auto* gamma_buf = gamma.buffer();
    auto* out_buf = output.buffer();
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    uint32_t row_offset = 0;
    for (const auto& core : cores) {
        uint32_t rows = core_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        SetRuntimeArgs(
            program, reader_id, core,
            {gy_buf->address(), x_buf->address(), red_buf->address(), row_offset, rows,
             n_buf->address(), gamma_buf->address()});
        SetRuntimeArgs(program, writer_id, core, {out_buf->address(), row_offset, rows});
        SetRuntimeArgs(program, compute_id, core, {rows});
        row_offset += rows;
    }

    return cached_program_t{std::move(program), {reader_id, writer_id, compute_id, cores}};
}

void LnBwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const LnBwParams&, const LnBwInputs& inputs, Tensor& output) {
    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto reader_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_id = cached_program.shared_variables.writer_kernel_id;
    auto* gy_buf = inputs.gy.buffer();
    auto* x_buf = inputs.x.buffer();
    auto* red_buf = inputs.red.buffer();
    auto* n_buf = inputs.n.buffer();
    auto* gamma_buf = inputs.gamma.buffer();
    auto* out_buf = output.buffer();
    for (const auto& core : cores) {
        auto& ra = GetRuntimeArgs(program, reader_id, core);
        ra[0] = gy_buf->address();
        ra[1] = x_buf->address();
        ra[2] = red_buf->address();
        ra[5] = n_buf->address();
        ra[6] = gamma_buf->address();
        auto& wa = GetRuntimeArgs(program, writer_id, core);
        wa[0] = out_buf->address();
    }
}

}  // namespace ttnn::experimental::prim
