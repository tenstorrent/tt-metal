// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ComplexMulFactory implementation.  Same CB layout and writer/compute
// kernels as apply_twiddles_xl; the only difference is the reader
// kernel (complex_mul_reader.cpp) which reads BOTH input complex
// tensors A and B from DRAM (no on-the-fly twiddle generation).

#include "complex_mul_factory.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

namespace {

// `_cm` suffix avoids Unity-build ODR collision with anonymous-namespace
// symbols in apply_twiddles_factory.cpp / apply_twiddles_xl_factory.cpp.
constexpr uint32_t kTileHW_cm        = 32u;
constexpr uint32_t kTileElems_cm     = kTileHW_cm * kTileHW_cm;          // 1024
constexpr uint32_t kTileBytesFp32_cm = kTileElems_cm * sizeof(float);    // 4096
constexpr uint32_t kTileBytesBf16_cm = kTileElems_cm * sizeof(uint16_t); // 2048

}  // namespace

tt::tt_metal::ProgramDescriptor ComplexMulFactory::create_descriptor(
    const ComplexMulParams& /*attrs*/,
    const ComplexMulTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;

    const auto& a_r_tensor = tensor_args.a_real;
    const auto& a_i_tensor = tensor_args.a_imag;
    const auto& b_r_tensor = tensor_args.b_real;
    const auto& b_i_tensor = tensor_args.b_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    // M = total row count = product of all dims except the last.  P =
    // last dim = row length.  Both already validated in the device op.
    const auto& shape = a_r_tensor.padded_shape();
    const uint32_t P  = static_cast<uint32_t>(shape[-1]);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M >= 1u, "ComplexMulFactory: M must be >= 1 (got {}).", M);

    const DataType dtype = a_r_tensor.dtype();
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    auto* const a_r_buf   = a_r_tensor.buffer();
    auto* const a_i_buf   = a_i_tensor.buffer();
    auto* const b_r_buf   = b_r_tensor.buffer();
    auto* const b_i_buf   = b_i_tensor.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();
    TT_FATAL(a_r_buf && a_i_buf && b_r_buf && b_i_buf && out_r_buf && out_i_buf,
        "ComplexMulFactory: all input/output tensors must be on device.");

    auto* device_raw = a_r_tensor.device();

    // ── Pick core grid: pow-2 num_cores dividing M.
    //   Unlike apply_twiddles[_xl] which guarantees M is a pow-2 multiple
    //   of big_modulus, complex_mul accepts arbitrary M (the chirp
    //   pre-multiply in Bluestein has last-dim P = N which can be any
    //   length).  We must therefore (a) FLOOR num_cores to a pow-2 first
    //   (else for M=37 we'd hit `num_cores = 37`, a non-pow-2 → invalid
    //   batch grid → dispatch-core placement TT_FATAL), then (b) shrink
    //   that pow-2 until it divides M.  Worst case (M odd prime)
    //   collapses to num_cores=1, which is correct.
    const auto dev_grid = device_raw->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    const uint32_t cap = (M < max_cores) ? M : max_cores;
    uint32_t num_cores = 1u;
    while ((num_cores << 1) <= cap) {
        num_cores <<= 1;
    }
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u,
        "ComplexMulFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;
    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Circular Buffers — IDENTICAL to apply_twiddles_(xl_)factory so
    //   we can reuse apply_twiddles_compute.cpp + apply_twiddles_writer.cpp
    //   binaries verbatim.  CB IDs 0..7 are the fp32 compute pipeline
    //   (A=input1, T=input2, B=output, TMP=SFPU scratch); IDs 8..11 are
    //   bf16 staging tiles allocated only when input/output is bf16.
    constexpr uint32_t kNumFp32Cbs = 8;
    constexpr uint32_t kCbTilesFp32[kNumFp32Cbs] = {
        2, 2, 2, 2, 2, 2, 1, 1   // A_R, A_I, T_R, T_I, B_R, B_I, TMP_R, TMP_I
    };
    for (uint32_t id = 0; id < kNumFp32Cbs; ++id) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbTilesFp32[id] * kTileBytesFp32_cm,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32_cm,
                }
            },
        });
    }
    if (is_bf16) {
        // 4 bf16 staging tiles: IN_R_BF16(8), IN_I_BF16(9),
        // OUT_R_BF16(10), OUT_I_BF16(11).  IN_R/I_BF16 are reused for
        // BOTH A and B by the reader (read A → expand to A_R/A_I, then
        // read B → expand to T_R/T_I, with push/pop in between).
        constexpr uint32_t kBf16CbIds[4] = { 8u, 9u, 10u, 11u };
        for (uint32_t i = 0; i < 4; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesBf16_cm,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kBf16CbIds[i]),
                        .data_format  = tt::DataFormat::Float16_b,
                        .page_size    = kTileBytesBf16_cm,
                    }
                },
            });
        }
    }

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t input_bf16_flag  = is_bf16 ? 1u : 0u;
    const uint32_t output_bf16_flag = is_bf16 ? 1u : 0u;

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/complex_mul_reader.cpp",
        .core_ranges = crs,
        // {P, INPUT_BF16}
        .compile_time_args = {P, input_bf16_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };

    // Writer is the SAME binary as apply_twiddles_writer.cpp.
    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_writer.cpp",
        .core_ranges = crs,
        // {N1=P, OUTPUT_BF16}
        .compile_time_args = {P, output_bf16_flag},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumFp32Cbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute is the SAME binary as apply_twiddles_compute.cpp.
    KernelDescriptor compute{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/apply_twiddles_compute.cpp",
        .core_ranges = crs,
        .compile_time_args = {},
        .runtime_args = {},
        .config = ComputeConfigDescriptor{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
        },
    };

    // ── Per-core runtime args ──────────────────────────────────────────
    const uint32_t in_page_size_bytes  = static_cast<uint32_t>(a_r_buf->aligned_page_size());
    const uint32_t out_page_size_bytes = static_cast<uint32_t>(out_r_buf->aligned_page_size());
    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);
    compute.runtime_args.reserve(num_cores);

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * rows_per_core;

        reader.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                a_r_buf->address(),
                a_i_buf->address(),
                b_r_buf->address(),
                b_i_buf->address(),
                base,
                rows_per_core,
                /*in_page_size_override=*/in_page_size_bytes,
            });

        writer.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                out_r_buf->address(),
                out_i_buf->address(),
                base,
                rows_per_core,
                /*out_page_size_override=*/out_page_size_bytes,
            });

        compute.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{ rows_per_core });
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::experimental::prim
