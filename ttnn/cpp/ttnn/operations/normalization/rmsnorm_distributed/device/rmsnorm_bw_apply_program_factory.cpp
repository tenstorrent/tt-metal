// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_apply_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <set>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::normalization::rmsnorm_distributed_bw {
namespace {

constexpr uint8_t kCbDy = 0;
constexpr uint8_t kCbX = 1;
constexpr uint8_t kCbGamma = 2;
constexpr uint8_t kCbInvRms = 3;
constexpr uint8_t kCbD = 4;
constexpr uint8_t kCbOut = 16;
constexpr uint8_t kCbAcc = 17;
constexpr uint8_t kCbPart = 18;
constexpr uint8_t kCbZeroDone = 19;
constexpr uint8_t kCbGo = 20;
constexpr uint32_t kSemReady1 = 0;
constexpr uint32_t kSemArrive1 = 1;
constexpr uint32_t kSemReady2 = 2;
constexpr uint32_t kSemArrive2 = 3;

CBDescriptor make_cb(uint8_t cb_id, uint32_t tiles, uint32_t page, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = tiles * page,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = page,
        }}},
    };
}

}  // namespace

ProgramDescriptor create_rmsnorm_bw_apply_program_descriptor(
    const Tensor& x,
    const Tensor& dy,
    const Tensor& gamma,
    const Tensor& inv_rms,
    const Tensor& d,
    Tensor& dx_out,
    const std::optional<Tensor>& dgamma_out) {
    auto* device = x.device();
    const auto& padded = x.padded_shape();
    TT_FATAL(padded.rank() == 4, "rmsnorm_bw_apply: x must be rank-4");
    const uint32_t tile_h = x.tensor_spec().tile().get_height();
    const uint32_t tile_w = x.tensor_spec().tile().get_width();
    TT_FATAL(tile_h == TILE_HEIGHT && tile_w == TILE_WIDTH, "rmsnorm_bw_apply: only 32x32 tiles are supported");

    const uint32_t Wt = padded[3] / tile_w;
    const uint32_t num_rows = padded[0] * padded[1] * (padded[2] / tile_h);
    TT_FATAL(Wt > 0 && num_rows > 0, "rmsnorm_bw_apply: empty tensor");
    const uint32_t page = tt::tile_size(tt::DataFormat::Float32);
    TT_FATAL(page == 4096, "rmsnorm_bw_apply: expected 4 KiB fp32 tiles, got {}", page);

    const bool with_dgamma = dgamma_out.has_value();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t max_cores = grid.x * grid.y;
    const uint32_t num_cores = std::min({max_cores, num_rows});
    TT_FATAL(
        grid.x <= TILE_HEIGHT, "rmsnorm_bw_apply: grid.x ({}) exceeds in-tile gather rows ({})", grid.x, TILE_HEIGHT);
    const uint32_t n_rows_used = (num_cores + grid.x - 1) / grid.x;
    TT_FATAL(
        n_rows_used <= TILE_HEIGHT,
        "rmsnorm_bw_apply: grid rows used ({}) exceed in-tile gather rows ({})",
        n_rows_used,
        TILE_HEIGHT);

    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    std::set<CoreRange> range_set;
    for (uint32_t k = 0; k < num_cores; ++k) {
        CoreCoord core{k % grid.x, k / grid.x};
        cores.push_back(core);
        range_set.insert(CoreRange(core, core));
    }
    CoreRangeSet core_ranges(range_set);

    const uint32_t base = num_rows / num_cores;
    const uint32_t rem = num_rows % num_cores;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbDy, 2 * Wt, page, core_ranges));
    desc.cbs.push_back(make_cb(kCbX, 2 * Wt, page, core_ranges));
    desc.cbs.push_back(make_cb(kCbGamma, Wt, page, core_ranges));
    desc.cbs.push_back(make_cb(kCbInvRms, 2, page, core_ranges));
    desc.cbs.push_back(make_cb(kCbD, 2, page, core_ranges));
    desc.cbs.push_back(make_cb(kCbOut, 2 * Wt, page, core_ranges));
    if (with_dgamma) {
        desc.cbs.push_back(make_cb(kCbAcc, Wt, page, core_ranges));
        desc.cbs.push_back(make_cb(kCbPart, Wt, page, core_ranges));
        for (uint8_t cb_id : {kCbZeroDone, kCbGo}) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = 64,
                .core_ranges = core_ranges,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = cb_id,
                    .data_format = tt::DataFormat::Float32,
                    .page_size = 64,
                }}},
            });
        }
        for (uint32_t sid : {kSemReady1, kSemArrive1, kSemReady2, kSemArrive2}) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = sid,
                .core_ranges = core_ranges,
                .initial_value = 0,
            });
        }
    }

    std::vector<uint32_t> reader_ct = {Wt, page, with_dgamma ? 1u : 0u};
    TensorAccessorArgs(*dy.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*x.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*gamma.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*inv_rms.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*d.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {
        Wt, page, with_dgamma ? 1u : 0u, kSemReady1, kSemArrive1, kSemReady2, kSemArrive2};
    TensorAccessorArgs(*dx_out.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*(with_dgamma ? dgamma_out->buffer() : dx_out.buffer())).append_to(writer_ct);

    const uint32_t neg_one_bits = std::bit_cast<uint32_t>(-1.0f);
    std::vector<uint32_t> compute_ct = {Wt, neg_one_bits, with_dgamma ? 1u : 0u};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/dataflow/"
        "reader_rmsnorm_bw_apply.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/dataflow/"
        "writer_rmsnorm_bw_apply.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint8_t cb : {kCbDy, kCbX, kCbGamma, kCbInvRms, kCbD}) {
        unpack_modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (with_dgamma) {
        unpack_modes[kCbAcc] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_bw_apply.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_ranges;
    compute_desc.compile_time_args = std::move(compute_ct);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .unpack_to_dest_mode = std::move(unpack_modes),
        .math_approx_mode = false,
    };

    auto* dy_buf = dy.buffer();
    auto* x_buf = x.buffer();
    auto* g_buf = gamma.buffer();
    auto* inv_buf = inv_rms.buffer();
    auto* d_buf = d.buffer();
    auto* out_buf = dx_out.buffer();
    Buffer* dg_buf = with_dgamma ? dgamma_out->buffer() : nullptr;

    const auto vcoord = [&](uint32_t cx, uint32_t cy) {
        return device->worker_core_from_logical_core(CoreCoord{cx, cy});
    };
    const auto root_v = vcoord(0, 0);

    uint32_t row_start = 0;
    for (uint32_t k = 0; k < num_cores; ++k) {
        const auto& core = cores[k];
        const uint32_t row_count = base + (k < rem ? 1u : 0u);
        reader_desc.emplace_runtime_args(core, {dy_buf, x_buf, g_buf, inv_buf, d_buf, row_start, row_count});

        const uint32_t role = (core.x == 0 && core.y == 0) ? 2u : (core.x == 0 ? 1u : 0u);
        const uint32_t row_cols = std::min(grid.x, num_cores - core.y * grid.x);
        const auto me = vcoord(core.x, core.y);
        const auto leader = vcoord(0, core.y);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(out_buf);
        writer_args.push_back(row_start);
        writer_args.push_back(row_count);
        writer_args.push_back(with_dgamma ? dg_buf : out_buf);
        writer_args.push_back(role);
        writer_args.push_back(static_cast<uint32_t>(core.x));
        writer_args.push_back(static_cast<uint32_t>(core.y));
        writer_args.push_back(static_cast<uint32_t>(me.x));
        writer_args.push_back(static_cast<uint32_t>(me.y));
        writer_args.push_back(static_cast<uint32_t>(leader.x));
        writer_args.push_back(static_cast<uint32_t>(leader.y));
        writer_args.push_back(static_cast<uint32_t>(root_v.x));
        writer_args.push_back(static_cast<uint32_t>(root_v.y));
        writer_args.push_back(row_cols);
        writer_args.push_back(n_rows_used);
        writer_args.push_back(row_cols + n_rows_used);
        for (uint32_t cx = 0; cx < row_cols; ++cx) {
            writer_args.push_back(static_cast<uint32_t>(vcoord(cx, core.y).x));
        }
        for (uint32_t cy = 0; cy < n_rows_used; ++cy) {
            writer_args.push_back(static_cast<uint32_t>(vcoord(0, cy).y));
        }
        writer_desc.emplace_runtime_args(core, writer_args);
        compute_desc.emplace_runtime_args(core, {row_count, role});
        row_start += row_count;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw
