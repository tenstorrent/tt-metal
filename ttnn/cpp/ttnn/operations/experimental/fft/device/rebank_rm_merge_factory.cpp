// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RebankRmMergeFactory — inverse page-size conversion for ROW_MAJOR tensors.
//
// Converts (B_total * N2, N1) with page_size = N1*elem_bytes
//       to (B_total, N1*N2)   with page_size = N1*N2*elem_bytes.
//
// No compute kernel — only reader + writer sharing one double-buffered CB.
// CB size = 2 * N1 * elem_bytes (one "slot" per double-buffer side, tiny).
//
// Work unit = one source row of N1 elements.
// num_units = B_total * N2.
// Reader: sequential full-page reads from source.
// Writer: writes at byte offset (unit % N2) * N1 * elem_bytes within dest page
//         (unit / N2).

#include "rebank_rm_merge_factory.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>

#include "stockham_host.hpp"   // pick_batch_grid, max_cores_for_grid, batch_logical_core
#include "rebank_rm_device_operation_types.hpp"  // rebank_is_pow2

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t CB_MERGE = 0u;

constexpr uint32_t log2u_rm(uint32_t n) {
    uint32_t r = 0u;
    while ((1u << r) < n) ++r;
    return r;
}

}  // namespace

tt::tt_metal::ProgramDescriptor RebankRmMergeFactory::create_descriptor(
    const RebankRmMergeParams&     operation_attributes,
    const RebankRmMergeTensorArgs& tensor_args,
    ttnn::Tensor&                  tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& x             = tensor_args.input;
    const auto& s_x           = x.padded_shape();
    const uint32_t chunks_per_merge = operation_attributes.chunks_per_merge;

    TT_FATAL(s_x.size() == 2u,
        "rebank_rm_merge: input must be 2D (got {}D).", s_x.size());

    const uint32_t N1      = static_cast<uint32_t>(s_x[-1]);  // source last-dim
    const uint32_t B_total_in = static_cast<uint32_t>(s_x[0]);  // B * chunks_per_merge
    TT_FATAL(B_total_in % chunks_per_merge == 0u,
        "rebank_rm_merge: input rows {} not divisible by chunks_per_merge {}.",
        B_total_in, chunks_per_merge);

    const uint32_t num_units = B_total_in;  // one unit per source row

    const DataType dtype   = x.dtype();
    const bool     is_bf16 = (dtype == DataType::BFLOAT16);
    const uint32_t elem_bytes  = is_bf16 ? 2u : 4u;
    const uint32_t chunk_bytes = N1 * elem_bytes;

    auto* const src_buf = x.buffer();
    auto* const dst_buf = tensor_return_value.buffer();
    TT_FATAL(src_buf && dst_buf,
        "rebank_rm_merge: input/output tensors must be on device.");

    auto* device_raw = x.device();
    auto md = device_raw->get_mesh_device();

    // ── Core grid ─────────────────────────────────────────────────────
    const auto dev_grid   = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (num_units < max_cores) ? num_units : max_cores;
    while (num_cores > 1u && (num_units % num_cores) != 0u)
        num_cores >>= 1u;
    TT_FATAL(num_cores >= 1u && (num_units % num_cores) == 0u,
        "rebank_rm_merge: failed to pick num_cores for num_units={}.", num_units);

    // Validate that the resulting core grid fits within the physical device.
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

    // ── CB: double-buffered, N1-element-sized (tiny) ──────────────────
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2u * chunk_bytes,
        .core_ranges = crs,
        .format_descriptors = {
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CB_MERGE),
                .data_format  = is_bf16 ? tt::DataFormat::Float16_b
                                        : tt::DataFormat::Float32,
                .page_size    = chunk_bytes,
            }
        },
    });

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t is_bf16_flag = is_bf16 ? 1u : 0u;
    (void)log2u_rm(chunks_per_merge);  // kept for debugging; not passed to kernel

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_merge_reader.cpp",
        .core_ranges = crs,
        // {CHUNK, IS_BF16}
        .compile_time_args = {N1, is_bf16_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_merge_writer.cpp",
        .core_ranges = crs,
        // {CHUNK, CHUNKS_PER_MERGE, IS_BF16}
        .compile_time_args = {N1, chunks_per_merge, is_bf16_flag},
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
