// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <fmt/format.h>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// Pack a float as a bf16 broadcast scalar for the reader's generate_bcast_unary_scalar,
// which reads the value from the upper 16 bits. Duplicating into both halves means the
// upper half carries the value regardless of pack order; bfloat16 rounds tie-to-even.
static uint32_t f32_to_bf16_packed(float v) { return pack_two_bfloat16_into_uint32({bfloat16(v), bfloat16(v)}); }

// Pack the 12 reader coefficient runtime args (Y, then Cb, then Cr; the two
// chroma weight rows pre-scaled by 0.25 so the compute kernel's 4-corner sum
// equals the 2x2 average). Shared by create() and override_runtime_arguments().
static std::array<uint32_t, 12> packed_coeffs(const YUVCoefficients& c) {
    return {
        f32_to_bf16_packed(c.y[0]),
        f32_to_bf16_packed(c.y[1]),
        f32_to_bf16_packed(c.y[2]),
        f32_to_bf16_packed(c.y[3]),
        f32_to_bf16_packed(c.cb[0] * 0.25f),
        f32_to_bf16_packed(c.cb[1] * 0.25f),
        f32_to_bf16_packed(c.cb[2] * 0.25f),
        f32_to_bf16_packed(c.cb[3]),
        f32_to_bf16_packed(c.cr[0] * 0.25f),
        f32_to_bf16_packed(c.cr[1] * 0.25f),
        f32_to_bf16_packed(c.cr[2] * 0.25f),
        f32_to_bf16_packed(c.cr[3]),
    };
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

    // A row-group is 2 H-rows -> 1 UV row.  The unit of work is one
    // (row-group, T-tile): its 2 RGB rows are read from DRAM once into L1 and
    // reused for the Y, Cb and Cr outputs (no 3x DRAM re-read).  Splitting on
    // (row-group x T-tile) rather than whole row-groups gives finer, more
    // balanced work distribution.
    uint32_t num_units = H2 * num_t_tiles;

    // Per-unit tile counts (identical for every unit -> a single compute kernel).
    uint32_t y_sticks = 2 * W;  // 2 H-rows worth of spatial sticks
    uint32_t uv_sticks = W2;    // 1 UV row worth of spatial sticks
    uint32_t y_tiles = (y_sticks + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t uv_tiles = (uv_sticks + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // --- Multicore work split (over units) -----------------------------------
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2] =
        split_work_to_cores(grid_size, num_units, /*row_wise=*/true);

    // --- Data formats --------------------------------------------------------
    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat u8_fmt = tt::DataFormat::UInt8;

    uint32_t bf16_tile_size = tt::tile_size(bf16_fmt);  // 2048 for 32x32 bf16
    uint32_t bf16_rm_page = TILE_HEIGHT * TILE_WIDTH * 2;

    // --- CB indices ----------------------------------------------------------
    constexpr uint32_t cb_R_rm = 0;
    constexpr uint32_t cb_G_rm = 1;
    constexpr uint32_t cb_B_rm = 2;
    constexpr uint32_t cb_tilized = 3;
    constexpr uint32_t cb_partial = 4;
    constexpr uint32_t cb_temp = 5;
    constexpr uint32_t cb_sum = 6;
    constexpr uint32_t cb_out_bf16 = 7;
    constexpr uint32_t cb_out_rm = 8;
    constexpr uint32_t cb_scratch = 9;
    // 12 resident scalar CBs (Y, Cb, Cr) x (wr, wg, wb, off), generated once.
    constexpr uint32_t cb_scalar_base = 10;  // 10..21

    // --- Circular buffers ----------------------------------------------------
    // Row-major channel input CBs (reader -> compute): 4 pages for UV corners.
    for (uint32_t id : {cb_R_rm, cb_G_rm, cb_B_rm}) {
        auto cfg = CircularBufferConfig(4 * bf16_rm_page, {{id, bf16_fmt}}).set_page_size(id, bf16_rm_page);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Tile-format scratch CBs for compute.
    for (uint32_t id : {cb_tilized, cb_partial, cb_temp, cb_sum}) {
        auto cfg = CircularBufferConfig(bf16_tile_size, {{id, bf16_fmt}}).set_page_size(id, bf16_tile_size);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // 12 resident scalar tile CBs (1 tile each, populated once, never popped).
    for (uint32_t k = 0; k < 12; k++) {
        uint32_t id = cb_scalar_base + k;
        auto cfg = CircularBufferConfig(bf16_tile_size, {{id, bf16_fmt}}).set_page_size(id, bf16_tile_size);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Intermediate bf16 row-major CB: untilize writes here, then typecast reads from it.
    {
        auto cfg =
            CircularBufferConfig(2 * bf16_rm_page, {{cb_out_bf16, bf16_fmt}}).set_page_size(cb_out_bf16, bf16_rm_page);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Output CB: uint8 row-major tiles from compute -> writer (double-buffered).
    {
        uint32_t u8_rm_page = TILE_HEIGHT * TILE_WIDTH * 1;
        auto cfg = CircularBufferConfig(2 * u8_rm_page, {{cb_out_rm, u8_fmt}}).set_page_size(cb_out_rm, u8_rm_page);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // Per-unit RGB residency scratch: holds one unit's 2 RGB rows (3 channels,
    // one T-tile) so the reader reads DRAM once and re-arranges from L1 for the
    // Y / Cb / Cr sub-passes.  One page, addressed raw by the reader.
    uint32_t scratch_bytes = 3 * y_sticks * TILE_WIDTH * 2;  // 3ch * 2W sticks * 32 T * bf16
    {
        auto cfg =
            CircularBufferConfig(scratch_bytes, {{cb_scratch, bf16_fmt}}).set_page_size(cb_scratch, scratch_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // --- Compile-time args ---------------------------------------------------
    std::vector<uint32_t> reader_ct_args = {
        cb_R_rm,
        cb_G_rm,
        cb_B_rm,
        cb_scratch,
        cb_scalar_base,
        num_t_tiles,
        T,
        W,
        W2,
        HW,
        y_tiles,
        uv_tiles,
    };
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        cb_R_rm,
        cb_G_rm,
        cb_B_rm,
        cb_tilized,
        cb_partial,
        cb_temp,
        cb_sum,
        cb_out_bf16,
        cb_out_rm,
        cb_scalar_base,
        y_tiles,
        uv_tiles,
    };

    std::vector<uint32_t> writer_ct_args = {
        cb_out_rm,
        num_t_tiles,
        T,
        W,
        W2,
        y_tiles,
        uv_tiles,
    };
    TensorAccessorArgs(*y_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*u_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*v_buf).append_to(writer_ct_args);

    // --- Typecast defines for compute kernel ---------------------------------
    std::map<std::string, std::string> compute_defines;
    compute_defines["TYPECAST_LLK_INIT"] =
        fmt::format("typecast_tile_init<{}u, {}u>", static_cast<uint32_t>(bf16_fmt), static_cast<uint32_t>(u8_fmt));
    compute_defines["TYPECAST_LLK"] =
        fmt::format("typecast_tile<{}u, {}u>", static_cast<uint32_t>(bf16_fmt), static_cast<uint32_t>(u8_fmt));

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

    // Every unit has identical shape, so one compute kernel serves all cores;
    // the per-core unit count is a runtime arg.
    // fp32_dest_acc_en is required for the SFPU bf16->uint8 typecast.
    KernelHandle compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/compute/yuv_compute.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_ct_args, .defines = compute_defines});

    // --- Per-core runtime args -----------------------------------------------
    const auto coeffs = packed_coeffs(op_attrs.coefficients);

    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    uint32_t unit_idx = 0;
    for (const auto& core : cores_vec) {
        uint32_t units = core_group_1.contains(core) ? units_per_core_g1 : units_per_core_g2;

        std::vector<uint32_t> reader_rt = {src_buf->address()};
        reader_rt.insert(reader_rt.end(), coeffs.begin(), coeffs.end());
        reader_rt.push_back(unit_idx);
        reader_rt.push_back(units);
        SetRuntimeArgs(program, reader_id, core, reader_rt);

        SetRuntimeArgs(program, compute_id, core, {units});

        SetRuntimeArgs(
            program, writer_id, core, {y_buf->address(), u_buf->address(), v_buf->address(), unit_idx, units});

        unit_idx += units;
    }

    return cached_program_t{
        std::move(program),
        {reader_id,
         writer_id,
         compute_id,
         num_cores,
         core_group_1,
         core_group_2,
         units_per_core_g1,
         units_per_core_g2}};
}

void YUVConversionProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const YUVConversionParams& op_attrs,
    const YUVConversionInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor>& output) {
    auto& sv = cached_program.shared_variables;
    auto& program = cached_program.program;
    auto& [y_out, u_out, v_out] = output;

    const auto coeffs = packed_coeffs(op_attrs.coefficients);
    const uint32_t src_addr = tensor_args.input.buffer()->address();
    const uint32_t y_addr = y_out.buffer()->address();
    const uint32_t u_addr = u_out.buffer()->address();
    const uint32_t v_addr = v_out.buffer()->address();

    const CoreCoord grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores = num_cores_to_corerangeset(sv.num_cores, grid_size, /*row_wise=*/true);
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);

    uint32_t unit_idx = 0;
    for (const auto& core : cores_vec) {
        uint32_t units = sv.core_group_1.contains(core) ? sv.units_per_core_g1 : sv.units_per_core_g2;

        {
            auto& args = GetRuntimeArgs(program, sv.reader_kernel_id, core);
            args[0] = src_addr;
            for (uint32_t k = 0; k < 12; k++) {
                args[1 + k] = coeffs[k];
            }
            args[13] = unit_idx;
            args[14] = units;
        }
        {
            auto& args = GetRuntimeArgs(program, sv.writer_kernel_id, core);
            args[0] = y_addr;
            args[1] = u_addr;
            args[2] = v_addr;
            args[3] = unit_idx;
            args[4] = units;
        }

        unit_idx += units;
    }
}

}  // namespace ttnn::experimental::prim
