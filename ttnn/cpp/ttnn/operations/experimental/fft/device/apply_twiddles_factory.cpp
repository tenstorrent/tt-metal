// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesFactory implementation — see header for op semantics.
//
// Dispatch model: M = product(input.shape[:-1]) rows total (each row =
// N1 elements).  We split rows across the same multi-core grid that
// BatchedStockhamFactory uses (pow-2 num_cores, picked so num_cores | M).
// Each core processes `rows_per_core` consecutive rows starting at
// `base_row`; the reader's inner loop computes `tw_row = row % N2` to
// broadcast the right twiddle row.
//
// Same ROW_MAJOR safety as BatchedStockhamFactory: input/output addrgens
// receive the ttnn buffer's aligned_page_size() as a runtime override,
// the reader/writer use InterleavedAddrGen<true> (NOT *Fast) to honour
// it; twiddle buffer pages are tile-sized so *Fast is safe there.

#include "apply_twiddles_factory.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>

#include "apply_twiddles_host.hpp"
#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kTileHW_at        = 32u;
constexpr uint32_t kTileElems_at_f   = kTileHW_at * kTileHW_at;          // 1024
constexpr uint32_t kTileBytesFp32_at = kTileElems_at_f * sizeof(float);  // 4096
constexpr uint32_t kTileBytesBf16_at = kTileElems_at_f * sizeof(uint16_t); // 2048

constexpr bool is_pow2_at(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

}  // namespace

tt::tt_metal::ProgramDescriptor ApplyTwiddlesFactory::create_descriptor(
    const ApplyTwiddlesParams& attrs,
    const ApplyTwiddlesTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& in_r_tensor = tensor_args.input_real;
    const auto& in_i_tensor = tensor_args.input_imag;
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    const uint32_t N1 = attrs.N1;
    const uint32_t N2 = attrs.N2;

    TT_FATAL(is_pow2_at(N1) && N1 >= 2u && N1 <= kTileElems_at_f,
        "ApplyTwiddlesFactory: N1 must be pow-2 in [2, 1024] (got {}).", N1);
    TT_FATAL(is_pow2_at(N2) && N2 >= 1u && N2 <= kTileElems_at_f,
        "ApplyTwiddlesFactory: N2 must be pow-2 in [1, 1024] (got {}).", N2);

    // ── Resolve M = total row count (product of leading dims) ──────────
    const auto& shape = in_r_tensor.padded_shape();
    TT_FATAL(static_cast<uint32_t>(shape[-1]) == N1,
        "ApplyTwiddlesFactory: last dim ({}) must equal N1 ({}).",
        static_cast<uint32_t>(shape[-1]), N1);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M % N2 == 0u,
        "ApplyTwiddlesFactory: row count M ({}) must be a multiple of N2 ({}).",
        M, N2);

    const DataType dtype = in_r_tensor.dtype();
    TT_FATAL(dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "ApplyTwiddlesFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    TT_FATAL(in_i_tensor.dtype() == dtype,
        "ApplyTwiddlesFactory: input_real and input_imag dtypes must match.");
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    auto* const in_r_buf  = in_r_tensor.buffer();
    auto* const in_i_buf  = in_i_tensor.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();
    TT_FATAL(in_r_buf && in_i_buf && out_r_buf && out_i_buf,
        "ApplyTwiddlesFactory: all input/output tensors must be on device.");

    // ── MeshDevice (no-op deleter — tensor owns lifetime) ──────────────
    auto* device_raw = in_r_tensor.device();
    auto md = device_raw->get_mesh_device();

    // ── Cached fp32 twiddle table for (N1, N2) ─────────────────────────
    auto tw_plan = apply_twiddles_host::get_or_create(md, N1, N2);

    // ── Pick core grid: pow-2 num_cores that divides M ─────────────────
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (M < max_cores) ? M : max_cores;
    while (num_cores > 1u && (M % num_cores) != 0u) {
        num_cores >>= 1;   // halve until it divides
    }
    TT_FATAL(num_cores >= 1u && (M % num_cores) == 0u,
        "ApplyTwiddlesFactory: failed to pick num_cores for M={}.", M);
    const uint32_t rows_per_core = M / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Circular Buffers ───────────────────────────────────────────────
    // fp32 compute CBs (IDs 0..7) — match apply_twiddles_common.h.
    constexpr uint32_t kNumFp32Cbs = 8;
    constexpr uint32_t kCbTilesFp32[kNumFp32Cbs] = {
        2, 2, 2, 2, 2, 2, 1, 1   // A_R, A_I, T_R, T_I, B_R, B_I, TMP_R, TMP_I
    };
    for (uint32_t id = 0; id < kNumFp32Cbs; ++id) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbTilesFp32[id] * kTileBytesFp32_at,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32_at,
                }
            },
        });
    }
    // bf16 staging CBs (IDs 8..11) — only allocated when needed.
    if (is_bf16) {
        constexpr uint32_t kBf16CbIds[4] = { 8u, 9u, 10u, 11u };
        for (uint32_t i = 0; i < 4; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesBf16_at,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kBf16CbIds[i]),
                        .data_format  = tt::DataFormat::Float16_b,
                        .page_size    = kTileBytesBf16_at,
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
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_reader.cpp",
        .core_ranges = crs,
        // {N1, INPUT_BF16}
        .compile_time_args = {N1, input_bf16_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_writer.cpp",
        .core_ranges = crs,
        // {N1, OUTPUT_BF16}
        .compile_time_args = {N1, output_bf16_flag},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumFp32Cbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

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
    // ROW_MAJOR tensors have page_size = N1*elem_size; reader/writer use
    // InterleavedAddrGen<true> with that override so the per-bank stride
    // matches the allocator (see apply_twiddles_reader.cpp for the full
    // rationale — same bug we hit in batch_fft).  Twiddle buffer pages
    // are tile-sized so its addrgen needs no override.
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
                fft_stockham::buf_addr(tw_plan->tw_r_buf),
                fft_stockham::buf_addr(tw_plan->tw_i_buf),
                base,
                rows_per_core,
                N2,
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
