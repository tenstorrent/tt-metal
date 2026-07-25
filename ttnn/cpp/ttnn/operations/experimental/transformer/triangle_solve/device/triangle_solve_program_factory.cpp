// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the experimental per-tile triangle-solve op.
//
// Single tile, single core. The reader streams one L tile into cb_l and one RHS tile into cb_rhs;
// the compute kernel runs the SFPU forward-substitution solve (triangle_solve_tile) and packs the
// solution into cb_x; the writer streams cb_x back to the output buffer.

#include "triangle_solve_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor TriangleSolveProgramFactory::create_descriptor(
    const TriangleSolveParams& attrs, const TriangleSolveInputs& in, std::vector<Tensor>& outputs) {
    const auto& l_neg = in.l_neg;  // [1, 1, 32, 32]  TILE  bf16
    const auto& rhs = in.rhs;      // [1, 1, 32, 32]  TILE  bf16
    const auto& output = outputs[0];

    IDevice* device = rhs.device();

    const CoreCoord core = {0, 0};
    const CoreRangeSet all_cores(CoreRange(core, core));

    const tt::DataFormat df = datatype_to_dataformat_converter(rhs.dtype());  // bf16
    const uint32_t tile_bytes = tt::tile_size(df);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);

    constexpr uint8_t cb_l = static_cast<uint8_t>(tt::CBIndex::c_0);    // negated unit-lower-tri L
    constexpr uint8_t cb_rhs = static_cast<uint8_t>(tt::CBIndex::c_1);  // RHS
    constexpr uint8_t cb_x = static_cast<uint8_t>(tt::CBIndex::c_2);    // solution X

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/transformer/triangle_solve/device/kernels/";

    ProgramDescriptor program;

    // ---- Reader: one L tile -> cb_l, one RHS tile -> cb_rhs. ----
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(l_neg.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(rhs.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kdir + "dataflow/reader_triangle_solve.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_ct_args);
    reader_kernel.config = ReaderConfigDescriptor{};
    reader_kernel.emplace_runtime_args(core, {l_neg.buffer(), rhs.buffer()});

    // ---- Writer: one cb_x tile -> output buffer. ----
    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kdir + "dataflow/writer_triangle_solve.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_ct_args);
    writer_kernel.config = WriterConfigDescriptor{};
    writer_kernel.emplace_runtime_args(core, {output.buffer()});

    // ---- Compute: SFPU forward-substitution solve of L X = RHS for one tile. ----
    KernelDescriptor compute_kernel;
    compute_kernel.kernel_source = kdir + "compute/triangle_solve.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = all_cores;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = math_approx_mode};

    program.kernels.push_back(std::move(reader_kernel));
    program.kernels.push_back(std::move(writer_kernel));
    program.kernels.push_back(std::move(compute_kernel));

    // ---- CBs: one tile each (bf16). ----
    program.cbs.push_back(CBDescriptor{
        .total_size = tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_l, .data_format = df, .page_size = tile_bytes}}}});
    program.cbs.push_back(CBDescriptor{
        .total_size = tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_rhs, .data_format = df, .page_size = tile_bytes}}}});
    program.cbs.push_back(CBDescriptor{
        .total_size = tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_x, .data_format = df, .page_size = tile_bytes}}}});

    return program;
}

}  // namespace ttnn::experimental::prim
