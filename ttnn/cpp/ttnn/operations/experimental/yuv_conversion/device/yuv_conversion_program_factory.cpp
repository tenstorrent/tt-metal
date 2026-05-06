// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstring>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// Degenerate-tile approach (mirrors typecast_rm_chunked):
//   CB page = one T-chunk (32 bf16 values = 64 bytes for input/scratch;
//                          32 uint8 values = 32 bytes for output).
//   Three passes: Y, Cb, Cr.
//   UV spatial averaging is done in the reader (RISC-V float arithmetic).

static constexpr uint32_t CHUNK_ELEMS = 32;                    // T values per chunk
static constexpr uint32_t BF16_CHUNK_BYTES = CHUNK_ELEMS * 2;  // 64
static constexpr uint32_t U8_CHUNK_BYTES = CHUNK_ELEMS * 1;    // 32

YUVConversionProgramFactory::cached_program_t YUVConversionProgramFactory::create(
    const YUVConversionParams& op_attrs,
    const YUVConversionInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor>& output) {
    Program program{};
    CoreCoord core{0, 0};

    const auto& input = tensor_args.input;
    auto& [y_out, u_out, v_out] = output;

    Buffer* src_buf = input.buffer();
    Buffer* y_buf = y_out.buffer();
    Buffer* u_buf = u_out.buffer();
    Buffer* v_buf = v_out.buffer();

    const auto& shape = input.logical_shape();
    uint32_t H = shape[1];
    uint32_t W = shape[2];
    uint32_t T = shape[3];

    uint32_t H2 = H / 2, W2 = W / 2;
    uint32_t spatial_Y = H * W;
    uint32_t spatial_UV = H2 * W2;

    // T-chunk parameters
    uint32_t partial_elems = T % CHUNK_ELEMS;    // e.g. 17 for T=81
    uint32_t num_full_chunks = T / CHUNK_ELEMS;  // e.g. 2
    uint32_t partial_bytes = partial_elems * 2;  // bf16 bytes
    uint32_t num_chunks = num_full_chunks + (partial_elems > 0 ? 1 : 0);

    // --- Circular buffers ------------------------------------------------
    // c_0..c_2: one bf16 chunk per page (input channels R, G, B)
    // c_3, c_4: scratch (same size, bf16)
    // c_16:     one uint8 chunk per page (output)
    constexpr uint32_t c_R = 0;
    constexpr uint32_t c_G = 1;
    constexpr uint32_t c_B = 2;
    constexpr uint32_t c_s0 = 3;
    constexpr uint32_t c_s1 = 4;
    constexpr uint32_t c_out = tt::CBIndex::c_16;

    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat u8_fmt = tt::DataFormat::UInt8;

    // 2-page double buffering for throughput
    for (uint32_t id : {c_R, c_G, c_B, c_s0, c_s1}) {
        auto cfg = CircularBufferConfig(2 * BF16_CHUNK_BYTES, {{id, bf16_fmt}}).set_page_size(id, BF16_CHUNK_BYTES);
        CreateCircularBuffer(program, core, cfg);
    }
    {
        auto cfg = CircularBufferConfig(2 * U8_CHUNK_BYTES, {{c_out, u8_fmt}}).set_page_size(c_out, U8_CHUNK_BYTES);
        CreateCircularBuffer(program, core, cfg);
    }

    // --- Compile-time args -----------------------------------------------
    std::vector<uint32_t> reader_ct_args = {
        c_R,
        c_G,
        c_B,
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

    // One degenerate-tile triplet per (spatial_position × chunk)
    uint32_t y_triplets = spatial_Y * num_chunks;
    uint32_t uv_triplets = spatial_UV * num_chunks;
    std::vector<uint32_t> compute_ct_args = {
        c_R,
        c_G,
        c_B,
        c_s0,
        c_s1,
        c_out,
        y_triplets,
        uv_triplets,
    };

    std::vector<uint32_t> writer_ct_args = {
        c_out,
        num_full_chunks,
        (partial_elems > 0) ? 1u : 0u,
        CHUNK_ELEMS,
        partial_elems,
        H,
        W,
        T,
        H2,
        W2,
    };
    TensorAccessorArgs(*y_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*u_buf).append_to(writer_ct_args);
    TensorAccessorArgs(*v_buf).append_to(writer_ct_args);

    // --- Kernel creation -------------------------------------------------
    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/dataflow/reader_yuv_chwt.cpp",
        core,
        ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/dataflow/writer_yuv_planes.cpp",
        core,
        WriterDataMovementConfig(writer_ct_args));

    // Coefficients as full float32 bits (mul_unary_tile expects float32 bit pattern)
    auto f32_bits = [](float v) -> uint32_t {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        return u;
    };
    const auto& coeff = op_attrs.coefficients;
    std::vector<uint32_t> compute_rt_args = {
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
    };

    KernelHandle compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/yuv_conversion/device/kernels/compute/yuv_chwt.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .compile_args = compute_ct_args,
        });

    // --- Runtime args ----------------------------------------------------
    SetRuntimeArgs(program, reader_id, core, {src_buf->address()});
    SetRuntimeArgs(program, compute_id, core, compute_rt_args);
    SetRuntimeArgs(program, writer_id, core, {y_buf->address(), u_buf->address(), v_buf->address()});

    return cached_program_t{std::move(program), {reader_id, compute_id, writer_id, core}};
}

void YUVConversionProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const YUVConversionParams& op_attrs,
    const YUVConversionInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor>& output) {
    auto& [reader_id, compute_id, writer_id, core] = cached_program.shared_variables;
    auto& program = cached_program.program;
    auto& [y_out, u_out, v_out] = output;

    auto f32_bits = [](float v) -> uint32_t {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        return u;
    };

    {
        auto& args = GetRuntimeArgs(program, reader_id, core);
        args[0] = tensor_args.input.buffer()->address();
    }
    {
        const auto& coeff = op_attrs.coefficients;
        auto& args = GetRuntimeArgs(program, compute_id, core);
        args[0] = f32_bits(coeff.y[0]);
        args[1] = f32_bits(coeff.y[1]);
        args[2] = f32_bits(coeff.y[2]);
        args[3] = f32_bits(coeff.y[3]);
        args[4] = f32_bits(coeff.cb[0]);
        args[5] = f32_bits(coeff.cb[1]);
        args[6] = f32_bits(coeff.cb[2]);
        args[7] = f32_bits(coeff.cb[3]);
        args[8] = f32_bits(coeff.cr[0]);
        args[9] = f32_bits(coeff.cr[1]);
        args[10] = f32_bits(coeff.cr[2]);
        args[11] = f32_bits(coeff.cr[3]);
    }
    {
        auto& args = GetRuntimeArgs(program, writer_id, core);
        args[0] = y_out.buffer()->address();
        args[1] = u_out.buffer()->address();
        args[2] = v_out.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
