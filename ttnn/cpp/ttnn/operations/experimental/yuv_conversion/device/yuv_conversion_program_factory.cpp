// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <cstring>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static constexpr uint32_t TILE_HW = 32 * 32;

// Convert a float to bf16 packed into the upper 16 bits of a uint32 (for generate_bcast_unary_scalar).
static uint32_t f32_to_bf16_packed(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return bits & 0xFFFF0000u;
}

YUVConversionProgramFactory::cached_program_t YUVConversionProgramFactory::create(
    const YUVConversionParams& op_attrs,
    const YUVConversionInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor>& output) {
    Program program{};

    const auto& input = tensor_args.input;
    auto& [y_out, u_out, v_out] = output;

    Buffer* src_buf = input.buffer();
    Buffer* y_buf = y_out.buffer();
    Buffer* u_buf = u_out.buffer();
    Buffer* v_buf = v_out.buffer();

    const auto* device = input.device();

    const auto& shape = input.logical_shape();
    uint32_t H = shape[1];
    uint32_t W = shape[2];
    uint32_t T = shape[3];
    uint32_t H2 = H / 2, W2 = W / 2;
    uint32_t HW = H * W;

    uint32_t num_t_tiles = (T + 31) / 32;

    // --- Multicore work split ------------------------------------------------
    uint32_t num_row_groups = H2;
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, groups_per_core_g1, groups_per_core_g2] =
        split_work_to_cores(grid_size, num_row_groups, /*row_wise=*/true);

    // --- Data formats --------------------------------------------------------
    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat u8_fmt = tt::DataFormat::UInt8;

    uint32_t bf16_tile_size = tt::tile_size(bf16_fmt);  // 2048 for 32×32 bf16
    uint32_t bf16_rm_page = 32 * 32 * 2;                // row-major 32×32 bf16 = 2048

    // --- CB indices ----------------------------------------------------------
    constexpr uint32_t cb_R_rm = 0;
    constexpr uint32_t cb_G_rm = 1;
    constexpr uint32_t cb_B_rm = 2;
    constexpr uint32_t cb_tilized = 3;
    constexpr uint32_t cb_partial = 4;
    constexpr uint32_t cb_temp = 5;
    constexpr uint32_t cb_wr = 6;
    constexpr uint32_t cb_wg = 7;
    constexpr uint32_t cb_wb = 8;
    constexpr uint32_t cb_off = 9;
    constexpr uint32_t cb_sum = 10;
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_17;

    // --- Circular buffers ----------------------------------------------------
    // Row-major channel input CBs (reader → compute): 4 pages for UV corners
    for (uint32_t id : {cb_R_rm, cb_G_rm, cb_B_rm}) {
        auto cfg = CircularBufferConfig(4 * bf16_rm_page, {{id, bf16_fmt}}).set_page_size(id, bf16_rm_page);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Tile-format scratch CBs for compute
    for (uint32_t id : {cb_tilized, cb_partial, cb_temp, cb_sum}) {
        auto cfg = CircularBufferConfig(bf16_tile_size, {{id, bf16_fmt}}).set_page_size(id, bf16_tile_size);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Scalar tile CBs (persistent, reader generates once per pass)
    for (uint32_t id : {cb_wr, cb_wg, cb_wb, cb_off}) {
        auto cfg = CircularBufferConfig(bf16_tile_size, {{id, bf16_fmt}}).set_page_size(id, bf16_tile_size);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Output CB: bf16 row-major tiles from compute → writer (double-buffered)
    {
        auto cfg =
            CircularBufferConfig(2 * bf16_rm_page, {{cb_out_rm, bf16_fmt}}).set_page_size(cb_out_rm, bf16_rm_page);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Writer scratch CB: uint8, 32 bytes (one stick for bf16→uint8 conversion)
    {
        uint32_t scratch_size = tt::align(32u, y_buf->alignment());
        auto cfg = CircularBufferConfig(scratch_size, {{cb_scratch, u8_fmt}}).set_page_size(cb_scratch, scratch_size);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // --- Compile-time args ---------------------------------------------------
    std::vector<uint32_t> reader_ct_args = {
        cb_R_rm,
        cb_G_rm,
        cb_B_rm,
        cb_wr,
        cb_wg,
        cb_wb,
        cb_off,
        cb_out_rm,
        num_t_tiles,
        T,
        H,
        W,
        H2,
        W2,
        HW,
    };
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        cb_out_rm,
        cb_scratch,
        num_t_tiles,
        T,
    };
    TensorAccessorArgs(*y_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*u_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*v_buf).append_to(writer_ct_args);

    // Compute compile-time args don't depend on per-core work; num_y_tiles and
    // num_uv_tiles are set per-core below via a helper, but since all cores in
    // group 1 share the same value (and group 2 likewise), we create two compute
    // kernels for the two groups, or pass them as runtime args.
    // For simplicity, we use compile-time args for the CB indices and tile counts
    // that are the same across all cores.  The tile counts differ per group, so
    // we use two compute kernels (one per group).

    auto make_compute_ct = [&](uint32_t y_count, uint32_t uv_count) {
        uint32_t y_batches = (y_count + 31) / 32;
        uint32_t uv_batches = (uv_count + 31) / 32;
        uint32_t num_y_tiles = y_batches * num_t_tiles;
        uint32_t num_uv_tiles_per_plane = uv_batches * num_t_tiles;
        return std::vector<uint32_t>{
            cb_R_rm,
            cb_G_rm,
            cb_B_rm,
            cb_tilized,
            cb_partial,
            cb_temp,
            cb_wr,
            cb_wg,
            cb_wb,
            cb_off,
            cb_sum,
            cb_out_rm,
            num_y_tiles,
            num_uv_tiles_per_plane,
            num_t_tiles,
        };
    };

    uint32_t y_count_g1 = 2 * groups_per_core_g1 * W;
    uint32_t uv_count_g1 = groups_per_core_g1 * W2;
    uint32_t y_count_g2 = 2 * groups_per_core_g2 * W;
    uint32_t uv_count_g2 = groups_per_core_g2 * W2;

    // --- Kernel creation -----------------------------------------------------
    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/dataflow/reader_yuv_chwt.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/dataflow/writer_yuv_planes.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // Compute kernel(s) — one per core group with different tile counts.
    KernelHandle compute_id_g1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/compute/yuv_compute.cpp",
        core_group_1,
        ComputeConfig{.compile_args = make_compute_ct(y_count_g1, uv_count_g1)});

    KernelHandle compute_id_g2{};
    if (groups_per_core_g2 > 0) {
        compute_id_g2 = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/compute/yuv_compute.cpp",
            core_group_2,
            ComputeConfig{.compile_args = make_compute_ct(y_count_g2, uv_count_g2)});
    }

    // --- Per-core runtime args -----------------------------------------------
    const auto& coeff = op_attrs.coefficients;

    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    uint32_t group_idx = 0;

    for (const auto& core : cores_vec) {
        uint32_t num_groups = core_group_1.contains(core) ? groups_per_core_g1 : groups_per_core_g2;

        uint32_t y_start = 2 * group_idx * W;
        uint32_t y_count = 2 * num_groups * W;
        uint32_t uv_start = group_idx * W2;
        uint32_t uv_count = num_groups * W2;

        // Reader runtime args: src_addr, Y coeffs (bf16 packed), Cb coeffs (pre-scaled),
        // Cr coeffs (pre-scaled), y_start, y_count, uv_start, uv_count.
        // For UV, the weight coefficients are pre-multiplied by 0.25 so the compute
        // kernel can skip the separate average step: sum * (wr * 0.25) = avg * wr.
        SetRuntimeArgs(
            program,
            reader_id,
            core,
            {
                src_buf->address(),
                // Y coefficients as bf16 packed
                f32_to_bf16_packed(coeff.y[0]),
                f32_to_bf16_packed(coeff.y[1]),
                f32_to_bf16_packed(coeff.y[2]),
                f32_to_bf16_packed(coeff.y[3]),
                // Cb coefficients: wr*0.25, wg*0.25, wb*0.25, offset (not scaled)
                f32_to_bf16_packed(coeff.cb[0] * 0.25f),
                f32_to_bf16_packed(coeff.cb[1] * 0.25f),
                f32_to_bf16_packed(coeff.cb[2] * 0.25f),
                f32_to_bf16_packed(coeff.cb[3]),
                // Cr coefficients: same pre-scaling
                f32_to_bf16_packed(coeff.cr[0] * 0.25f),
                f32_to_bf16_packed(coeff.cr[1] * 0.25f),
                f32_to_bf16_packed(coeff.cr[2] * 0.25f),
                f32_to_bf16_packed(coeff.cr[3]),
                y_start,
                y_count,
                uv_start,
                uv_count,
            });

        SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                y_buf->address(),
                u_buf->address(),
                v_buf->address(),
                y_start,
                y_count,
                uv_start,
                uv_count,
            });

        group_idx += num_groups;
    }

    return cached_program_t{
        std::move(program),
        {reader_id,
         writer_id,
         compute_id_g1,
         compute_id_g2,
         num_cores,
         core_group_1,
         core_group_2,
         groups_per_core_g1,
         groups_per_core_g2,
         W,
         W2}};
}

void YUVConversionProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const YUVConversionParams& op_attrs,
    const YUVConversionInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor>& output) {
    auto& sv = cached_program.shared_variables;
    auto& program = cached_program.program;
    auto& [y_out, u_out, v_out] = output;

    const auto& coeff = op_attrs.coefficients;
    const uint32_t src_addr = tensor_args.input.buffer()->address();
    const uint32_t y_addr = y_out.buffer()->address();
    const uint32_t u_addr = u_out.buffer()->address();
    const uint32_t v_addr = v_out.buffer()->address();

    const CoreCoord grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores = num_cores_to_corerangeset(sv.num_cores, grid_size, /*row_wise=*/true);
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);

    uint32_t group_idx = 0;
    for (const auto& core : cores_vec) {
        uint32_t num_groups = sv.core_group_1.contains(core) ? sv.groups_per_core_g1 : sv.groups_per_core_g2;

        uint32_t y_start = 2 * group_idx * sv.W;
        uint32_t y_count = 2 * num_groups * sv.W;
        uint32_t uv_start = group_idx * sv.W2;
        uint32_t uv_count = num_groups * sv.W2;

        {
            auto& args = GetRuntimeArgs(program, sv.reader_kernel_id, core);
            args[0] = src_addr;
            args[1] = f32_to_bf16_packed(coeff.y[0]);
            args[2] = f32_to_bf16_packed(coeff.y[1]);
            args[3] = f32_to_bf16_packed(coeff.y[2]);
            args[4] = f32_to_bf16_packed(coeff.y[3]);
            args[5] = f32_to_bf16_packed(coeff.cb[0] * 0.25f);
            args[6] = f32_to_bf16_packed(coeff.cb[1] * 0.25f);
            args[7] = f32_to_bf16_packed(coeff.cb[2] * 0.25f);
            args[8] = f32_to_bf16_packed(coeff.cb[3]);
            args[9] = f32_to_bf16_packed(coeff.cr[0] * 0.25f);
            args[10] = f32_to_bf16_packed(coeff.cr[1] * 0.25f);
            args[11] = f32_to_bf16_packed(coeff.cr[2] * 0.25f);
            args[12] = f32_to_bf16_packed(coeff.cr[3]);
            args[13] = y_start;
            args[14] = y_count;
            args[15] = uv_start;
            args[16] = uv_count;
        }
        {
            auto& args = GetRuntimeArgs(program, sv.writer_kernel_id, core);
            args[0] = y_addr;
            args[1] = u_addr;
            args[2] = v_addr;
            args[3] = y_start;
            args[4] = y_count;
            args[5] = uv_start;
            args[6] = uv_count;
        }

        group_idx += num_groups;
    }
}

}  // namespace ttnn::experimental::prim
