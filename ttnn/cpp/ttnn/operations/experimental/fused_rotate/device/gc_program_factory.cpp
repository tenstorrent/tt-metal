// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gc_program_factory.hpp"
#include "gc_device_operation_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static const char* kGcKernelDir = "ttnn/cpp/ttnn/operations/experimental/fused_rotate/device/kernels/";

FusedGcProgramFactory::cached_program_t FusedGcProgramFactory::create(
    const FusedGcParams& attrs, const FusedGcInputs& inputs, Tensor& output) {
    Program program{};

    const auto& gout = inputs.gout;
    const auto& xin = inputs.xin;
    const auto& sel = inputs.sel;

    const uint32_t Wt = attrs.W / TILE_WIDTH;
    const uint32_t n_out_tiles = attrs.n_out * Wt;
    const uint32_t n_in_tiles = attrs.n_in * Wt;
    const uint32_t out_tiles = (attrs.nnz + TILE_WIDTH - 1) / TILE_WIDTH;  // ceil(nnz/32)
    const uint32_t Et = gout.padded_shape()[-2] / TILE_HEIGHT;

    tt::DataFormat data_format = datatype_to_dataformat_converter(gout.dtype());
    const uint32_t tile_bytes = tile_size(data_format);

    auto* device = gout.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, Et);

    auto make_cb = [&](uint32_t cb_index, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * tile_bytes, {{cb_index, data_format}}).set_page_size(cb_index, tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    constexpr uint32_t cb_gout = tt::CBIndex::c_0;
    constexpr uint32_t cb_xin = tt::CBIndex::c_1;
    constexpr uint32_t cb_sel = tt::CBIndex::c_2;
    constexpr uint32_t cb_prod = tt::CBIndex::c_24;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    make_cb(cb_gout, 2 * n_out_tiles);
    make_cb(cb_xin, 2 * n_in_tiles);
    make_cb(cb_sel, 32);
    make_cb(cb_prod, 32 * Wt);   // one output-tile worth of products (d<=32 nonzeros x Wt)
    make_cb(cb_out, 2 * out_tiles);

    // ---- reader ----
    std::vector<uint32_t> reader_ct = {cb_gout, cb_xin, cb_sel, n_out_tiles, n_in_tiles, tile_bytes};
    TensorAccessorArgs(*gout.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*xin.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*sel.buffer()).append_to(reader_ct);
    KernelHandle reader_id =
        CreateKernel(program, std::string(kGcKernelDir) + "gc_reader.cpp", all_cores, ReaderDataMovementConfig(reader_ct));

    // ---- writer (reuse the generic writer: cb_out, out_tiles, tile_bytes) ----
    std::vector<uint32_t> writer_ct = {cb_out, out_tiles, tile_bytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    KernelHandle writer_id =
        CreateKernel(program, std::string(kGcKernelDir) + "writer.cpp", all_cores, WriterDataMovementConfig(writer_ct));

    // ---- compute ----
    std::vector<uint32_t> compute_ct = {cb_gout, cb_xin, cb_sel,      cb_prod,  cb_out,
                                        n_out_tiles, n_in_tiles, Wt, attrs.nnz, out_tiles};
    KernelHandle compute_id = CreateKernel(
        program,
        std::string(kGcKernelDir) + "gc_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_ct});

    auto* gout_buf = gout.buffer();
    auto* xin_buf = xin.buffer();
    auto* sel_buf = sel.buffer();
    auto* out_buf = output.buffer();
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    uint32_t row_offset = 0;
    for (const auto& core : cores) {
        uint32_t rows = core_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        SetRuntimeArgs(
            program, reader_id, core,
            {gout_buf->address(), xin_buf->address(), sel_buf->address(), row_offset, rows});
        SetRuntimeArgs(program, writer_id, core, {out_buf->address(), row_offset, rows});

        std::vector<uint32_t> compute_rt = {rows};
        compute_rt.insert(compute_rt.end(), attrs.is_.begin(), attrs.is_.end());
        compute_rt.insert(compute_rt.end(), attrs.js.begin(), attrs.js.end());
        SetRuntimeArgs(program, compute_id, core, compute_rt);

        row_offset += rows;
    }

    return cached_program_t{std::move(program), {reader_id, writer_id, compute_id, cores}};
}

void FusedGcProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const FusedGcParams&, const FusedGcInputs& inputs, Tensor& output) {
    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto reader_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_id = cached_program.shared_variables.writer_kernel_id;
    auto* gout_buf = inputs.gout.buffer();
    auto* xin_buf = inputs.xin.buffer();
    auto* sel_buf = inputs.sel.buffer();
    auto* out_buf = output.buffer();
    for (const auto& core : cores) {
        auto& ra = GetRuntimeArgs(program, reader_id, core);
        ra[0] = gout_buf->address();
        ra[1] = xin_buf->address();
        ra[2] = sel_buf->address();
        auto& wa = GetRuntimeArgs(program, writer_id, core);
        wa[0] = out_buf->address();
    }
}

}  // namespace ttnn::experimental::prim
