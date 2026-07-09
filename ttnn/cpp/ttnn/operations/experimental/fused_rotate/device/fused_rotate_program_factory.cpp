// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rotate_program_factory.hpp"
#include "fused_rotate_device_operation_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/fused_rotate/device/kernels/";

FusedRotateProgramFactory::cached_program_t FusedRotateProgramFactory::create(
    const FusedRotateParams& attrs, const FusedRotateInputs& inputs, Tensor& output) {
    Program program{};

    const auto& x = inputs.x_flat;
    const auto& coef = inputs.coef_exp;

    const uint32_t Wt = attrs.W / TILE_WIDTH;
    const uint32_t n_in_tiles = attrs.n_in * Wt;
    const uint32_t n_out_tiles = attrs.n_out * Wt;
    const uint32_t coef_tiles = attrs.nnz;
    const uint32_t Et = x.padded_shape()[-2] / TILE_HEIGHT;  // number of tile-rows (edges/32)

    tt::DataFormat data_format = datatype_to_dataformat_converter(x.dtype());
    const uint32_t tile_bytes = tile_size(data_format);

    auto* device = x.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, Et);

    // Circular buffers. Keep a whole tile-row resident (compute indexes randomly into the
    // n_in input blocks), double-buffered across rows to overlap DRAM with compute.
    auto make_cb = [&](uint32_t cb_index, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * tile_bytes, {{cb_index, data_format}}).set_page_size(cb_index, tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_coef = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    make_cb(cb_x, 2 * n_in_tiles);
    make_cb(cb_coef, 2 * coef_tiles);
    make_cb(cb_out, 2 * n_out_tiles);

    // ---- reader ----
    std::vector<uint32_t> reader_ct = {cb_x, cb_coef, n_in_tiles, coef_tiles, tile_bytes};
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*coef.buffer()).append_to(reader_ct);
    KernelHandle reader_id = CreateKernel(
        program,
        std::string(kKernelDir) + "reader.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct));

    // ---- writer ----
    std::vector<uint32_t> writer_ct = {cb_out, n_out_tiles, tile_bytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    KernelHandle writer_id = CreateKernel(
        program,
        std::string(kKernelDir) + "writer.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct));

    // ---- compute ----
    std::vector<uint32_t> compute_ct = {cb_x, cb_coef, cb_out, n_in_tiles, coef_tiles, n_out_tiles, attrs.n_out, Wt};
    KernelHandle compute_id = CreateKernel(
        program,
        std::string(kKernelDir) + "compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,  // 8 fp32 dest slots so the fan-in (d<=5) fits in dst[0..d-1]
            .compile_args = compute_ct});

    // Runtime args. The sparsity pattern (deg/ks/js) is identical on every core.
    auto* x_buf = x.buffer();
    auto* coef_buf = coef.buffer();
    auto* out_buf = output.buffer();
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    uint32_t row_offset = 0;
    for (const auto& core : cores) {
        uint32_t rows;
        if (core_group_1.contains(core)) {
            rows = rows_per_core_1;
        } else {
            rows = rows_per_core_2;
        }
        SetRuntimeArgs(program, reader_id, core, {x_buf->address(), coef_buf->address(), row_offset, rows});
        SetRuntimeArgs(program, writer_id, core, {out_buf->address(), row_offset, rows});

        std::vector<uint32_t> compute_rt = {rows};
        compute_rt.insert(compute_rt.end(), attrs.deg.begin(), attrs.deg.end());
        compute_rt.insert(compute_rt.end(), attrs.ks.begin(), attrs.ks.end());
        compute_rt.insert(compute_rt.end(), attrs.js.begin(), attrs.js.end());
        SetRuntimeArgs(program, compute_id, core, compute_rt);

        row_offset += rows;
    }

    return cached_program_t{
        std::move(program), {reader_id, writer_id, compute_id, cores}};
}

void FusedRotateProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const FusedRotateParams&, const FusedRotateInputs& inputs, Tensor& output) {
    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto reader_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_id = cached_program.shared_variables.writer_kernel_id;
    auto* x_buf = inputs.x_flat.buffer();
    auto* coef_buf = inputs.coef_exp.buffer();
    auto* out_buf = output.buffer();
    for (const auto& core : cores) {
        auto& ra = GetRuntimeArgs(program, reader_id, core);
        ra[0] = x_buf->address();
        ra[1] = coef_buf->address();
        auto& wa = GetRuntimeArgs(program, writer_id, core);
        wa[0] = out_buf->address();
    }
}

}  // namespace ttnn::experimental::prim
