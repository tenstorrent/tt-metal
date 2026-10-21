// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/kda_block_matmul.hpp"

#include <algorithm>
#include <functional>
#include <string>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/kernel_lib/mcast/host/mcast.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim::kda_factory_detail {

namespace {

namespace mcast = ttnn::kernel_lib::host;
using tt::tt_metal::CoreRangeSet;

constexpr uint32_t cb_a = 16;
constexpr uint32_t cb_w = 17;
constexpr uint32_t cb_acc = 18;
constexpr uint32_t cb_out = 19;
constexpr uint32_t cb_copy = 20;
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/";

}  // namespace

void check_block_matmul(
    const Tensor& a, const Tensor& w, const tt::tt_metal::CoreRange& cores, uint32_t k_block, std::string_view name) {
    const uint32_t kt = a.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(w.padded_shape()[-2] == a.padded_shape()[-1], "{}: A's and W's K differ", name);
    TT_FATAL(k_block > 0 && kt % k_block == 0, "{}: K={} tiles must split into {}-tile blocks", name, kt, k_block);
    const auto grid = a.device()->compute_with_storage_grid_size();
    TT_FATAL(cores.end_coord.x < grid.x && cores.end_coord.y < grid.y, "{}: cores {} leave the grid", name, cores);
}

tt::tt_metal::ProgramDescriptor block_matmul_program(
    const tt::tt_metal::distributed::MeshDevice& device,
    const Tensor& a,
    const Tensor& w,
    const Tensor& output,
    const tt::tt_metal::CoreRange& rect,
    uint32_t k_block,
    uint32_t output_column,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using tt::tt_metal::CoreCoord;
    const uint32_t mt = a.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t kt = a.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t w_row_tiles = w.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t nt = w_row_tiles;
    const uint32_t out_row_tiles = output.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t x0 = rect.start_coord.x;
    const uint32_t y0 = rect.start_coord.y;
    const uint32_t rows = rect.end_coord.y - y0 + 1;
    const uint32_t cols = rect.end_coord.x - x0 + 1;
    const uint32_t m_block = (mt + rows - 1) / rows;
    const uint32_t n_block = (nt + cols - 1) / cols;
    const uint32_t k_blocks = kt / k_block;
    // The largest output subblock that fits the four FP32 DST tiles, widest first.
    uint32_t sub_h = 1, sub_w = 1;
    for (uint32_t h : {1u, 2u, 4u}) {
        for (uint32_t sw : {4u, 3u, 2u, 1u}) {
            if (h * sw <= 4 && m_block % h == 0 && n_block % sw == 0 &&
                std::pair{h * sw, sw} > std::pair{sub_h * sub_w, sub_w}) {
                sub_h = h;
                sub_w = sw;
            }
        }
    }

    const CoreRangeSet rect_set(rect);
    const uint32_t a_tile = tile_bytes(a);
    const uint32_t w_tile = tile_bytes(w);
    const uint32_t out_tile = tile_bytes(output);
    const uint32_t acc_tile = tt::constants::TILE_HW * sizeof(float);
    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs = {
        cb_descriptor(
            cb_a, 2 * m_block * k_block, a_tile, tt::tt_metal::datatype_to_dataformat_converter(a.dtype()), rect_set),
        cb_descriptor(
            cb_w, 2 * k_block * n_block, w_tile, tt::tt_metal::datatype_to_dataformat_converter(w.dtype()), rect_set),
        cb_descriptor(cb_acc, m_block * n_block, acc_tile, tt::DataFormat::Float32, rect_set),
        cb_descriptor(
            cb_out,
            m_block * n_block,
            out_tile,
            tt::tt_metal::datatype_to_dataformat_converter(output.dtype()),
            rect_set),
    };

    std::vector<uint32_t> a_ct_base = {cb_a, m_block, k_block, k_blocks, kt};
    std::vector<uint32_t> w_ct_base = {cb_w, cb_out, m_block, n_block, k_block, k_blocks, w_row_tiles, out_row_tiles};
    const auto a_kernel = [&](bool sends, const std::vector<CoreCoord>& cores) {
        auto ct = a_ct_base;
        ct.push_back(sends);
        tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(ct);
        tt::tt_metal::KernelDescriptor kernel{
            .kernel_source = std::string(kernel_dir) + "dataflow/block_matmul_reader_a.cpp",
            .core_ranges = merged_core_set(cores),
            .compile_time_args = std::move(ct),
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_1},
        };
        for (const auto& core : cores) {
            const uint32_t m0 = (core.y - y0) * m_block;
            const uint32_t m_valid = m0 < mt ? std::min(m_block, mt - m0) : 0;
            tt::tt_metal::KernelDescriptor::RTArgList args;
            args.push_back(a.buffer());
            args.push_back(m0);
            args.push_back(m_valid);
            kernel.emplace_runtime_args(core, args);
        }
        return kernel;
    };
    const auto w_kernel = [&](bool sends, const std::vector<CoreCoord>& cores) {
        auto ct = w_ct_base;
        ct.insert(ct.end(), {static_cast<uint32_t>(sends), sub_w, sub_h});
        tt::tt_metal::TensorAccessorArgs(w.buffer()).append_to(ct);
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(ct);
        tt::tt_metal::KernelDescriptor kernel{
            .kernel_source = std::string(kernel_dir) + "dataflow/block_matmul_reader_w_writer.cpp",
            .core_ranges = merged_core_set(cores),
            .compile_time_args = std::move(ct),
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0},
        };
        for (const auto& core : cores) {
            const uint32_t m0 = (core.y - y0) * m_block;
            const uint32_t n0 = (core.x - x0) * n_block;
            const uint32_t m_valid = m0 < mt ? std::min(m_block, mt - m0) : 0;
            const uint32_t n_valid = n0 < nt ? std::min(n_block, nt - n0) : 0;
            tt::tt_metal::KernelDescriptor::RTArgList args;
            args.push_back(w.buffer());
            args.push_back(output.buffer());
            for (uint32_t value : {m0, m_valid, n0, n_valid, output_column}) {
                args.push_back(value);
            }
            kernel.emplace_runtime_args(core, args);
        }
        return kernel;
    };
    std::vector<CoreCoord> a_senders, a_receivers, w_senders, w_receivers;
    for (const auto& core : tt::tt_metal::corerange_to_cores(rect_set, std::nullopt, true)) {
        (core.x == x0 ? a_senders : a_receivers).push_back(core);
        (core.y == y0 ? w_senders : w_receivers).push_back(core);
    }
    std::vector<tt::tt_metal::KernelDescriptor> a_kernels, w_kernels;
    a_kernels.push_back(a_kernel(true, a_senders));
    if (!a_receivers.empty()) {
        a_kernels.push_back(a_kernel(false, a_receivers));
    }
    w_kernels.push_back(w_kernel(true, w_senders));
    if (!w_receivers.empty()) {
        w_kernels.push_back(w_kernel(false, w_receivers));
    }

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(device.arch(), compute_kernel_config);
    (void)packer_l1_acc;
    // The last K block reloads the FP32 accumulator into DST; unpacking straight to DST keeps it exact.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_modes(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_modes[cb_acc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/block_matmul.cpp",
        .core_ranges = rect_set,
        .compile_time_args = {cb_a, cb_w, cb_acc, cb_out, m_block, n_block, k_block, k_blocks, sub_w, sub_h},
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .unpack_to_dest_mode = std::move(unpack_modes),
                .math_approx_mode = math_approx_mode},
    };

    // A along each row on NoC 1 and W down each column on NoC 0, the 2D multicast matmul's NoC split.
    const auto line = [&](bool along_row, tt::tt_metal::NOC noc) {
        return mcast::Mcast(
            device,
            mcast::McastConfig{.noc = noc, .handshake = true},
            rect_set,
            along_row ? cols : rows,
            mcast::McastFixedSenderConfig{.sender_index = 0, .placement = mcast::McastSenderPlacement::Uniform},
            along_row ? mcast::McastCoreOrder::RowMajor : mcast::McastCoreOrder::ColumnMajor);
    };
    const auto refs = [](std::vector<tt::tt_metal::KernelDescriptor>& kernels) {
        std::vector<std::reference_wrapper<tt::tt_metal::KernelDescriptor>> out;
        for (auto& kernel : kernels) {
            out.emplace_back(kernel);
        }
        return out;
    };
    const auto mc_a = line(true, tt::tt_metal::NOC::NOC_1);
    const auto mc_w = line(false, tt::tt_metal::NOC::NOC_0);
    mc_a.attach(desc, "a", refs(a_kernels), static_cast<uint32_t>(desc.semaphores.size()));
    mc_w.attach(desc, "w", refs(w_kernels), mc_a.next_semaphore_id());
    for (auto* kernels : {&a_kernels, &w_kernels}) {
        for (auto& kernel : *kernels) {
            desc.kernels.push_back(std::move(kernel));
        }
    }
    desc.kernels.push_back(std::move(compute));
    return desc;
}

tt::tt_metal::ProgramDescriptor copy_tile_columns_program(
    const Tensor& source,
    uint32_t source_column,
    const Tensor& destination,
    uint32_t columns,
    const std::vector<tt::tt_metal::CoreCoord>& cores) {
    TT_FATAL(!cores.empty(), "copy_tile_columns_program needs at least one core");
    const uint32_t rows = source.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const std::vector<tt::tt_metal::CoreCoord> used(
        cores.begin(), cores.begin() + std::min<size_t>(rows, cores.size()));
    const uint32_t per_core = (rows + used.size() - 1) / used.size();
    const uint32_t page = tile_bytes(source);
    const CoreRangeSet set = merged_core_set(used);
    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs = {
        cb_descriptor(cb_copy, columns, page, tt::tt_metal::datatype_to_dataformat_converter(source.dtype()), set)};
    std::vector<uint32_t> ct = {cb_copy, columns};
    tt::tt_metal::TensorAccessorArgs(source.buffer()).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(destination.buffer()).append_to(ct);
    tt::tt_metal::KernelDescriptor kernel{
        .kernel_source = std::string(kernel_dir) + "dataflow/copy_tile_columns.cpp",
        .core_ranges = set,
        .compile_time_args = std::move(ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    for (uint32_t index = 0; index < used.size(); ++index) {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.push_back(source.buffer());
        args.push_back(destination.buffer());
        for (uint32_t value :
             {source.padded_shape()[-1] / tt::constants::TILE_WIDTH,
              source_column,
              destination.padded_shape()[-1] / tt::constants::TILE_WIDTH,
              std::min(rows, index * per_core),
              std::min(rows, (index + 1) * per_core)}) {
            args.push_back(value);
        }
        kernel.emplace_runtime_args(used[index], args);
    }
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

}  // namespace ttnn::experimental::prim::kda_factory_detail
