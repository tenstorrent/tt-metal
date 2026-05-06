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

static constexpr uint32_t CHUNK_ELEMS = 32;
static constexpr uint32_t BF16_CHUNK_BYTES = CHUNK_ELEMS * 2;  // 64
static constexpr uint32_t U8_CHUNK_BYTES = CHUNK_ELEMS * 1;    // 32

static uint32_t f32_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
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

    uint32_t partial_elems = T % CHUNK_ELEMS;
    uint32_t num_full_chunks = T / CHUNK_ELEMS;
    uint32_t partial_bytes = partial_elems * 2;

    // --- Multicore work split --------------------------------------------
    // Work unit = one "row group" (2 Y rows + 1 UV row). Total = H/2 groups.
    uint32_t num_row_groups = H2;

    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, groups_per_core_g1, groups_per_core_g2] =
        split_work_to_cores(grid_size, num_row_groups, /*row_wise=*/true);

    // --- Circular buffers ------------------------------------------------
    constexpr uint32_t c_R = 0;
    constexpr uint32_t c_G = 1;
    constexpr uint32_t c_B = 2;
    constexpr uint32_t c_out = tt::CBIndex::c_16;

    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat u8_fmt = tt::DataFormat::UInt8;

    const uint32_t staging_page_bytes = tt::align(BF16_CHUNK_BYTES, src_buf->alignment());
    const uint32_t out_page_bytes = tt::align(U8_CHUNK_BYTES, y_buf->alignment());

    for (uint32_t id : {c_R, c_G, c_B}) {
        auto cfg = CircularBufferConfig(staging_page_bytes, {{id, bf16_fmt}}).set_page_size(id, staging_page_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    }
    {
        auto cfg = CircularBufferConfig(2 * out_page_bytes, {{c_out, u8_fmt}}).set_page_size(c_out, out_page_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    }

    // --- Compile-time args (same for all cores) --------------------------
    std::vector<uint32_t> reader_ct_args = {
        c_R,
        c_G,
        c_B,
        c_out,
        num_full_chunks,
        (partial_elems > 0) ? 1u : 0u,
        BF16_CHUNK_BYTES,
        partial_bytes,
        H,
        W,
        T,
        H2,
        W2,
    };
    TensorAccessorArgs(*src_buf).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        c_out,
        num_full_chunks,
        (partial_elems > 0) ? 1u : 0u,
        CHUNK_ELEMS,
        partial_elems,
    };
    TensorAccessorArgs(*y_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*u_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*v_buf).append_to(writer_ct_args);

    // --- Kernel creation -------------------------------------------------
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

    // --- Per-core runtime args -------------------------------------------
    const auto& coeff = op_attrs.coefficients;

    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    uint32_t group_idx = 0;

    for (const auto& core : cores_vec) {
        uint32_t num_groups = core_group_1.contains(core) ? groups_per_core_g1 : groups_per_core_g2;

        uint32_t y_start = 2 * group_idx * W;
        uint32_t y_count = 2 * num_groups * W;
        uint32_t uv_start = group_idx * W2;
        uint32_t uv_count = num_groups * W2;

        SetRuntimeArgs(
            program,
            reader_id,
            core,
            {
                src_buf->address(),
                f32_bits(coeff.y[0]),
                f32_bits(coeff.y[1]),
                f32_bits(coeff.y[2]),
                f32_bits(coeff.y[3]),
                f32_bits(coeff.cb[0]),
                f32_bits(coeff.cb[1]),
                f32_bits(coeff.cb[2]),
                f32_bits(coeff.cb[3]),
                f32_bits(coeff.cr[0]),
                f32_bits(coeff.cr[1]),
                f32_bits(coeff.cr[2]),
                f32_bits(coeff.cr[3]),
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
        {reader_id, writer_id, num_cores, core_group_1, core_group_2, groups_per_core_g1, groups_per_core_g2, W, W2}};
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
            args[1] = f32_bits(coeff.y[0]);
            args[2] = f32_bits(coeff.y[1]);
            args[3] = f32_bits(coeff.y[2]);
            args[4] = f32_bits(coeff.y[3]);
            args[5] = f32_bits(coeff.cb[0]);
            args[6] = f32_bits(coeff.cb[1]);
            args[7] = f32_bits(coeff.cb[2]);
            args[8] = f32_bits(coeff.cb[3]);
            args[9] = f32_bits(coeff.cr[0]);
            args[10] = f32_bits(coeff.cr[1]);
            args[11] = f32_bits(coeff.cr[2]);
            args[12] = f32_bits(coeff.cr[3]);
            // y_start, y_count, uv_start, uv_count don't change across invocations
            // with the same shape, but set them for completeness.
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
