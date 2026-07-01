// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RebankRmFactory — pure-copy page-size conversion for ROW_MAJOR tensors.
//
// Converts (B_total, N) with page_size = N*elem_bytes
//       to (B_total * N/chunk, chunk) with page_size = chunk*elem_bytes.
//
// No compute kernel — only reader + writer sharing one double-buffered CB.
// CB size = 2 * chunk * elem_bytes (one "slot" per double-buffer side).
//
// Work distribution: num_units = B_total * N / chunk.  Each core gets
// units_per_core consecutive units so that the sequential scan of the
// source tensor is bank-locality friendly.

#include "rebank_rm_factory.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>

#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t CB_REBANK = 0u;

constexpr uint32_t log2u_rb(uint32_t n) {
    uint32_t r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

}  // namespace

tt::tt_metal::ProgramDescriptor RebankRmFactory::create_descriptor(
    const RebankRmParams&     operation_attributes,
    const RebankRmTensorArgs& tensor_args,
    ttnn::Tensor&             tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& x      = tensor_args.input;
    const auto& s_x    = x.padded_shape();
    const uint32_t chunk = operation_attributes.chunk_size;

    TT_FATAL(s_x.size() >= 2u,
        "rebank_rm: input must be ≥ 2D (got {}D).", s_x.size());

    const uint32_t N = static_cast<uint32_t>(s_x[-1]);
    uint32_t B_total = 1u;
    for (int d = 0; d < static_cast<int>(s_x.size()) - 1; ++d)
        B_total *= static_cast<uint32_t>(s_x[d]);

    TT_FATAL(rebank_is_pow2(chunk) && chunk >= 1u && chunk <= N,
        "rebank_rm: chunk_size must be pow-2 in [1, N={}] (got {}).", N, chunk);
    TT_FATAL(N % chunk == 0u,
        "rebank_rm: N={} must be divisible by chunk_size={}", N, chunk);

    const uint32_t chunks_per_row = N / chunk;
    const uint32_t num_units      = B_total * chunks_per_row;

    const DataType dtype   = x.dtype();
    const bool     is_bf16 = (dtype == DataType::BFLOAT16);
    const uint32_t elem_bytes  = is_bf16 ? 2u : 4u;
    const uint32_t chunk_bytes = chunk * elem_bytes;

    auto* const src_buf = x.buffer();
    auto* const dst_buf = tensor_return_value.buffer();
    TT_FATAL(src_buf && dst_buf,
        "rebank_rm: input/output tensors must be on device.");

    auto* device_raw = x.device();
    auto md = device_raw->get_mesh_device();

    // ── Core grid ─────────────────────────────────────────────────────
    const auto dev_grid   = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (num_units < max_cores) ? num_units : max_cores;
    while (num_cores > 1u && (num_units % num_cores) != 0u)
        num_cores >>= 1u;
    TT_FATAL(num_cores >= 1u && (num_units % num_cores) == 0u,
        "rebank_rm: failed to pick num_cores for num_units={}.", num_units);

    // Validate that pick_batch_grid(num_cores) stays within the physical grid.
    // When num_cores is non-pow2 (e.g. 63 units → pick_batch_grid returns {7,9}
    // which exceeds dev_grid.y=8), the CoreRange would include dispatch cores and
    // the kernel placement fails.  Decrement num_cores until the grid fits.
    {
        auto [gc, gr] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);
        while (num_cores > 1u && gr > dev_grid.y) {
            --num_cores;
            while (num_cores > 1u && (num_units % num_cores) != 0u)
                --num_cores;
            std::tie(gc, gr) = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);
        }
    }

    const uint32_t units_per_core = num_units / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    // ── CB: double-buffered, chunk-sized ─────────────────────────────
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2u * chunk_bytes,
        .core_ranges = crs,
        .format_descriptors = {
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CB_REBANK),
                .data_format  = is_bf16 ? tt::DataFormat::Float16_b
                                        : tt::DataFormat::Float32,
                .page_size    = chunk_bytes,
            }
        },
    });

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t is_bf16_flag    = is_bf16 ? 1u : 0u;
    const uint32_t log2_chunks_per_row = log2u_rb(chunks_per_row);

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_reader.cpp",
        .core_ranges = crs,
        // {CHUNK, CHUNKS_PER_ROW, IS_BF16}
        .compile_time_args = {chunk, chunks_per_row, is_bf16_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };
    (void)log2_chunks_per_row;  // encoded via compile-time arg

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_writer.cpp",
        .core_ranges = crs,
        // {CHUNK, IS_BF16}
        .compile_time_args = {chunk, is_bf16_flag},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    const uint32_t src_page_size_bytes =
        static_cast<uint32_t>(src_buf->aligned_page_size());
    const uint32_t dst_page_size_bytes =
        static_cast<uint32_t>(dst_buf->aligned_page_size());

    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);

    for (uint32_t c = 0u; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * units_per_core;

        reader.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                src_buf->address(),
                base,
                units_per_core,
                src_page_size_bytes,
            });

        writer.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                dst_buf->address(),
                base,
                units_per_core,
                dst_page_size_bytes,
            });
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));

    return desc;
}

}  // namespace ttnn::experimental::prim
