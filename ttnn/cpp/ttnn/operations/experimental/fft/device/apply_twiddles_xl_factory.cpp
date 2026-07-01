// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesXlFactory implementation.  Same dispatch / CB layout as
// ApplyTwiddlesFactory (so apply_twiddles_compute + apply_twiddles_writer
// can be reused verbatim); the only difference is the reader kernel,
// which builds the twiddle row in L1 from a per-(device, big_modulus,
// full_N) cached delta table (apply_twiddles_xl_host).
//
// Why a separate factory (not a flag on ApplyTwiddlesFactory)?
//   - Different runtime args + different reader compile-time args.
//   - Different cached host buffer (delta vs full twiddle table).
//   - Different validation envelope (twiddle_N2 cap is the whole reason
//     this op exists).  Keeping them separate preserves the original op's
//     simplicity and program-cache identity.

#include "apply_twiddles_xl_factory.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>

#include "apply_twiddles_xl_host.hpp"
#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core, buf_addr

namespace ttnn::experimental::prim {

namespace {

// `_xl` suffix avoids Unity-build ODR collision with anonymous-namespace
// symbols in apply_twiddles_factory.cpp / batched_stockham_factory.cpp.
constexpr uint32_t kTileHW_xl        = 32u;
constexpr uint32_t kTileElems_xl     = kTileHW_xl * kTileHW_xl;          // 1024
constexpr uint32_t kTileBytesFp32_xl = kTileElems_xl * sizeof(float);    // 4096
constexpr uint32_t kTileBytesBf16_xl = kTileElems_xl * sizeof(uint16_t); // 2048

constexpr bool is_pow2_xl(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

}  // namespace

tt::tt_metal::ProgramDescriptor ApplyTwiddlesXlFactory::create_descriptor(
    const ApplyTwiddlesXlParams& attrs,
    const ApplyTwiddlesXlTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& in_r_tensor = tensor_args.input_real;
    const auto& in_i_tensor = tensor_args.input_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    const uint32_t P           = attrs.P;
    const uint32_t big_modulus = attrs.big_modulus;
    const uint32_t full_N      = attrs.full_N;

    TT_FATAL(is_pow2_xl(P) && P >= 2u && P <= kTileElems_xl,
        "ApplyTwiddlesXlFactory: P must be pow-2 in [2, 1024] (got {}).", P);
    TT_FATAL(is_pow2_xl(big_modulus) && big_modulus >= 1u,
        "ApplyTwiddlesXlFactory: big_modulus must be pow-2 and >= 1 (got {}).",
        big_modulus);
    TT_FATAL(is_pow2_xl(full_N) && full_N >= big_modulus,
        "ApplyTwiddlesXlFactory: full_N must be pow-2 and >= big_modulus "
        "(got full_N={} big_modulus={}).", full_N, big_modulus);

    // M = total rows.
    const auto& shape = in_r_tensor.padded_shape();
    TT_FATAL(static_cast<uint32_t>(shape[-1]) == P,
        "ApplyTwiddlesXlFactory: input last dim ({}) must equal P ({}).",
        static_cast<uint32_t>(shape[-1]), P);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M >= 1u && (M % big_modulus) == 0u,
        "ApplyTwiddlesXlFactory: row count M ({}) must be a multiple of "
        "big_modulus ({}).", M, big_modulus);

    const DataType dtype = in_r_tensor.dtype();
    TT_FATAL(dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "ApplyTwiddlesXlFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    TT_FATAL(in_i_tensor.dtype() == dtype,
        "ApplyTwiddlesXlFactory: input_real and input_imag dtypes must match.");
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    auto* const in_r_buf  = in_r_tensor.buffer();
    auto* const in_i_buf  = in_i_tensor.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();
    TT_FATAL(in_r_buf && in_i_buf && out_r_buf && out_i_buf,
        "ApplyTwiddlesXlFactory: all input/output tensors must be on device.");

    auto* device_raw = in_r_tensor.device();
    auto md = device_raw->get_mesh_device();

    auto delta_plan = apply_twiddles_xl_host::get_or_create(md, big_modulus, full_N);

    // ── Pick core grid: pow-2 num_cores dividing M (matches apply_twiddles).
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (M < max_cores) ? M : max_cores;
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u,
        "ApplyTwiddlesXlFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;
    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Circular Buffers ── Same as apply_twiddles_factory (so we can
    //    reuse apply_twiddles_compute.cpp + apply_twiddles_writer.cpp
    //    binaries verbatim).
    constexpr uint32_t kNumFp32Cbs = 8;
    constexpr uint32_t kCbTilesFp32[kNumFp32Cbs] = {
        2, 2, 2, 2, 2, 2, 1, 1   // A_R, A_I, T_R, T_I, B_R, B_I, TMP_R, TMP_I
    };
    for (uint32_t id = 0; id < kNumFp32Cbs; ++id) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbTilesFp32[id] * kTileBytesFp32_xl,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32_xl,
                }
            },
        });
    }
    if (is_bf16) {
        constexpr uint32_t kBf16CbIds[4] = { 8u, 9u, 10u, 11u };
        for (uint32_t i = 0; i < 4; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesBf16_xl,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kBf16CbIds[i]),
                        .data_format  = tt::DataFormat::Float16_b,
                        .page_size    = kTileBytesBf16_xl,
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
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_xl_reader.cpp",
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
    const uint32_t in_page_size_bytes  = static_cast<uint32_t>(in_r_buf->aligned_page_size());
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
                in_r_buf->address(),
                in_i_buf->address(),
                fft_stockham::buf_addr(delta_plan->dr_buf),
                fft_stockham::buf_addr(delta_plan->di_buf),
                base,
                rows_per_core,
                big_modulus,
                /*in_page_size_override=*/in_page_size_bytes,
                /*in_imag_page_size_override=*/in_page_size_bytes,
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
