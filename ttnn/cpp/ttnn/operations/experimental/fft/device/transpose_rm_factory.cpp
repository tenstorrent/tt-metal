// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TransposeRmFactory implementation — see header for op semantics.
//
// Dispatch model: total work units = B * (A/32) * (C/32) where B is the
// product of leading dims.  We split units evenly across a pow-2-sized
// multi-core grid (same grid-picking logic as the FFT factories).  Each
// core processes `units_per_core` consecutive units.  Linear-to-3D
// decode is in the kernel.
//
// No twiddle, no compute kernel — only reader + writer.  Two CB slots
// double-buffer the 32×32 block staging area so the reader can stay one
// block ahead of the writer.

#include "transpose_rm_factory.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>

#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t T_BLOCK = 32u;

}  // namespace

tt::tt_metal::ProgramDescriptor TransposeRmFactory::create_descriptor(
    const TransposeRmParams&,
    const TransposeRmTensorArgs& tensor_args,
    ttnn::Tensor& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& x   = tensor_args.input;
    const auto& y   = tensor_return_value;
    const auto& s_x = x.padded_shape();

    const uint32_t A = static_cast<uint32_t>(s_x[-2]);
    const uint32_t C = static_cast<uint32_t>(s_x[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(s_x.size()) - 2; ++d) {
        B *= static_cast<uint32_t>(s_x[d]);
    }

    const uint32_t A_tiles = A / T_BLOCK;
    const uint32_t C_tiles = C / T_BLOCK;
    const uint32_t num_units = B * A_tiles * C_tiles;
    TT_FATAL(num_units > 0u,
        "transpose_rm: zero work units (B={}, A_tiles={}, C_tiles={}).", B, A_tiles, C_tiles);

    const DataType dtype = x.dtype();
    const bool is_bf16   = (dtype == DataType::BFLOAT16);
    const uint32_t elem_bytes = is_bf16 ? 2u : 4u;
    const uint32_t block_bytes = T_BLOCK * T_BLOCK * elem_bytes;

    auto* const src_buf = x.buffer();
    auto* const dst_buf = y.buffer();
    TT_FATAL(src_buf && dst_buf,
        "transpose_rm: input/output tensors must be on device.");

    auto* device_raw = x.device();
    auto md = device_raw->get_mesh_device();

    // ── Pick a pow-2 core count that divides num_units cleanly ─────────
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (num_units < max_cores) ? num_units : max_cores;
    while (num_cores > 1u && (num_units % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(num_cores >= 1u && (num_units % num_cores) == 0u,
        "transpose_rm: failed to pick num_cores for num_units={}.", num_units);
    const uint32_t units_per_core = num_units / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Single CB, double-buffered ─────────────────────────────────────
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2u * block_bytes,   // 2 slots for read/write overlap
        .core_ranges = crs,
        .format_descriptors = {
            CBFormatDescriptor{
                .buffer_index = 0u,
                .data_format  = is_bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32,
                .page_size    = block_bytes,
            }
        },
    });

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t is_bf16_flag = is_bf16 ? 1u : 0u;

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/transpose_rm_reader.cpp",
        .core_ranges = crs,
        .compile_time_args = {A_tiles, C_tiles, is_bf16_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/transpose_rm_writer.cpp",
        .core_ranges = crs,
        .compile_time_args = {A_tiles, C_tiles, is_bf16_flag},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    const uint32_t src_page_size_bytes = static_cast<uint32_t>(src_buf->aligned_page_size());
    const uint32_t dst_page_size_bytes = static_cast<uint32_t>(dst_buf->aligned_page_size());
    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);

    for (uint32_t c = 0; c < num_cores; ++c) {
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
