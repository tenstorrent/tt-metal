// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_program_factory.hpp"
#include "gate_device_operation_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static const char* kGateKernelDir = "ttnn/cpp/ttnn/operations/experimental/fused_rotate/device/kernels/";

GateProgramFactory::cached_program_t GateProgramFactory::create(
    const GateParams& attrs, const GateInputs& inputs, Tensor& output) {
    Program program{};

    const auto& a = inputs.a;
    const auto& gate = inputs.gate;
    const auto& b = inputs.b;

    const uint32_t Wt = attrs.Wt;
    const uint32_t Gt = attrs.Gt;
    const uint32_t Ht = attrs.Ht;
    const uint32_t Et = a.padded_shape()[-2] / TILE_HEIGHT;

    tt::DataFormat data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t tile_bytes = tile_size(data_format);

    auto* device = a.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, Et);

    auto make_cb = [&](uint32_t cb_index, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * tile_bytes, {{cb_index, data_format}}).set_page_size(cb_index, tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_gate = tt::CBIndex::c_1;
    constexpr uint32_t cb_b = tt::CBIndex::c_2;
    constexpr uint32_t cb_sp = tt::CBIndex::c_3;    // silu'(b), Ht tiles (bw only)
    constexpr uint32_t cb_s = tt::CBIndex::c_4;     // scratch sigmoid(b) (bw only)
    constexpr uint32_t cb_p = tt::CBIndex::c_5;     // scratch silu(b)=b*s (bw only)
    constexpr uint32_t cb_r = tt::CBIndex::c_6;     // scratch p*s (bw only)
    constexpr uint32_t cb_tmp = tt::CBIndex::c_7;   // scratch s+p (bw only)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    make_cb(cb_a, 2 * Wt);
    make_cb(cb_gate, 2 * Gt);
    make_cb(cb_b, 2 * Ht);
    make_cb(cb_sp, 2 * Ht);
    make_cb(cb_s, 2 * Ht);
    make_cb(cb_p, 2 * Ht);
    make_cb(cb_r, 2 * Ht);
    make_cb(cb_tmp, 2 * Ht);
    make_cb(cb_out, 2 * Wt);

    // ---- reader ----
    std::vector<uint32_t> reader_ct = {cb_a, cb_gate, cb_b, Wt, Gt, Ht, tile_bytes, attrs.mode};
    TensorAccessorArgs(*a.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*gate.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*b.buffer()).append_to(reader_ct);
    KernelHandle reader_id = CreateKernel(
        program, std::string(kGateKernelDir) + "gate_reader.cpp", all_cores, ReaderDataMovementConfig(reader_ct));

    // ---- writer ----
    std::vector<uint32_t> writer_ct = {cb_out, Wt, tile_bytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    KernelHandle writer_id = CreateKernel(
        program, std::string(kGateKernelDir) + "gate_writer.cpp", all_cores, WriterDataMovementConfig(writer_ct));

    // ---- compute ----
    std::vector<uint32_t> compute_ct = {
        cb_a, cb_gate, cb_b, cb_sp, cb_out, Wt, Gt, Ht, attrs.mode, cb_s, cb_p, cb_r, cb_tmp};
    KernelHandle compute_id = CreateKernel(
        program,
        std::string(kGateKernelDir) + "gate_compute.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_ct});

    auto* a_buf = a.buffer();
    auto* gate_buf = gate.buffer();
    auto* b_buf = b.buffer();
    auto* out_buf = output.buffer();
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    uint32_t row_offset = 0;
    for (const auto& core : cores) {
        uint32_t rows = core_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        SetRuntimeArgs(
            program, reader_id, core,
            {a_buf->address(), gate_buf->address(), b_buf->address(), row_offset, rows});
        SetRuntimeArgs(program, writer_id, core, {out_buf->address(), row_offset, rows});
        SetRuntimeArgs(program, compute_id, core, {rows});
        row_offset += rows;
    }

    return cached_program_t{std::move(program), {reader_id, writer_id, compute_id, cores}};
}

void GateProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const GateParams&, const GateInputs& inputs, Tensor& output) {
    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto reader_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_id = cached_program.shared_variables.writer_kernel_id;
    auto* a_buf = inputs.a.buffer();
    auto* gate_buf = inputs.gate.buffer();
    auto* b_buf = inputs.b.buffer();
    auto* out_buf = output.buffer();
    for (const auto& core : cores) {
        auto& ra = GetRuntimeArgs(program, reader_id, core);
        ra[0] = a_buf->address();
        ra[1] = gate_buf->address();
        ra[2] = b_buf->address();
        auto& wa = GetRuntimeArgs(program, writer_id, core);
        wa[0] = out_buf->address();
    }
}

}  // namespace ttnn::experimental::prim
