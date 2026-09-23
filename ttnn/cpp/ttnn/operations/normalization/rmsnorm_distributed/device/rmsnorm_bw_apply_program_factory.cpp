// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_apply_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

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

constexpr uint32_t kReaderKernelIdx = 0;
constexpr uint32_t kWriterKernelIdx = 1;
constexpr uint32_t kComputeKernelIdx = 2;

CBDescriptor make_cb(uint8_t cb_id, uint32_t pages, uint32_t page, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = pages * page,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = page,
        }}},
    };
}

KernelDescriptor make_kernel(
    std::string source,
    const CoreRangeSet& cores,
    std::vector<uint32_t> compile_time_args,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor kernel;
    kernel.kernel_source = std::move(source);
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = std::move(compile_time_args);
    kernel.config = std::move(config);
    return kernel;
}

struct ApplyWorkSplit {
    uint32_t Wt = 0;
    uint32_t num_rows = 0;
    uint32_t num_cores = 0;
    uint32_t page = 0;
    uint32_t n_rows_used = 0;
    uint32_t base = 0;
    uint32_t rem = 0;
    CoreCoord grid{};
    CoreRangeSet core_ranges;
};

ApplyWorkSplit make_work_split(const Tensor& x) {
    const auto occ = compute_apply_occupancy(x);
    ApplyWorkSplit ws;
    ws.Wt = occ.Wt;
    ws.num_rows = occ.num_rows;
    ws.num_cores = occ.num_cores;
    ws.page = tt::tile_size(tt::DataFormat::Float32);
    ws.grid = CoreCoord{occ.grid_x, occ.grid_y};
    ws.n_rows_used = occ.n_rows_used;
    ws.core_ranges = num_cores_to_corerangeset(ws.num_cores, ws.grid, /*row_wise=*/true);
    ws.base = ws.num_rows / ws.num_cores;
    ws.rem = ws.num_rows % ws.num_cores;
    return ws;
}

struct ApplyCoreWork {
    CoreCoord core;
    uint32_t row_start = 0;
    uint32_t row_count = 0;
    uint32_t role = 0;
    uint32_t row_cols = 0;
};

std::vector<ApplyCoreWork> make_core_layout(const ApplyWorkSplit& ws) {
    std::vector<ApplyCoreWork> layout;
    layout.reserve(ws.num_cores);
    uint32_t row_start = 0;
    for (uint32_t k = 0; k < ws.num_cores; ++k) {
        const CoreCoord core{k % ws.grid.x, k / ws.grid.x};
        const uint32_t row_count = ws.base + (k < ws.rem ? 1u : 0u);
        const uint32_t role = (core.x == 0 && core.y == 0) ? 2u : (core.x == 0 ? 1u : 0u);
        const uint32_t row_cols = std::min(ws.grid.x, ws.num_cores - core.y * ws.grid.x);
        layout.push_back({core, row_start, row_count, role, row_cols});
        row_start += row_count;
    }
    return layout;
}

struct ApplyBuffers {
    Buffer* dy = nullptr;
    Buffer* x = nullptr;
    Buffer* gamma = nullptr;
    Buffer* inv_rms = nullptr;
    Buffer* d = nullptr;
    Buffer* dx = nullptr;
    Buffer* dgamma = nullptr;
};

ApplyBuffers make_buffers(
    const RMSNormBwApplyOperation::tensor_args_t& tensor_args,
    RMSNormBwApplyOperation::tensor_return_value_t& outputs) {
    TT_FATAL(outputs.size() == 2 && outputs[0].has_value(), "rmsnorm_bw_apply: dx output is required");
    const bool with_dgamma = tensor_args.gamma.has_value();
    TT_FATAL(with_dgamma == outputs[1].has_value(), "rmsnorm_bw_apply: dgamma output must match gamma");
    ApplyBuffers bufs;
    bufs.dy = tensor_args.dy.buffer();
    bufs.x = tensor_args.x.buffer();
    // TensorAccessorArgs needs a live buffer even when the kernel does not read gamma / dgamma.
    bufs.gamma = with_dgamma ? tensor_args.gamma->buffer() : tensor_args.x.buffer();
    bufs.inv_rms = tensor_args.inv_rms.buffer();
    bufs.d = tensor_args.d.buffer();
    bufs.dx = outputs[0]->buffer();
    bufs.dgamma = with_dgamma ? outputs[1]->buffer() : outputs[0]->buffer();
    return bufs;
}

// Buffer entries become base addresses; create_descriptor also registers them as buffer bindings.
using RTArg = std::variant<uint32_t, Buffer*>;

// The per-core runtime args, in the order the kernels read them. create_descriptor and
// override_runtime_arguments both build them here so the two layouts cannot drift.
struct ApplyCoreArgs {
    std::vector<RTArg> reader;
    std::vector<RTArg> writer;
    std::vector<RTArg> compute;
};

ApplyCoreArgs make_core_args(
    const ApplyWorkSplit& ws, const ApplyBuffers& bufs, const ApplyCoreWork& work, IDevice* device) {
    const auto vcoord = [&](uint32_t cx, uint32_t cy) {
        return device->worker_core_from_logical_core(CoreCoord{cx, cy});
    };
    const auto me = vcoord(work.core.x, work.core.y);
    const auto leader = vcoord(0, work.core.y);
    const auto root = vcoord(0, 0);

    ApplyCoreArgs args;
    args.reader = {bufs.dy, bufs.x, bufs.gamma, bufs.inv_rms, bufs.d, work.row_start, work.row_count};
    args.writer = {
        bufs.dx,
        work.row_start,
        work.row_count,
        bufs.dgamma,
        work.role,
        static_cast<uint32_t>(work.core.x),
        static_cast<uint32_t>(work.core.y),
        static_cast<uint32_t>(me.x),
        static_cast<uint32_t>(me.y),
        static_cast<uint32_t>(leader.x),
        static_cast<uint32_t>(leader.y),
        static_cast<uint32_t>(root.x),
        static_cast<uint32_t>(root.y),
        work.row_cols,
        ws.n_rows_used};
    // Virtual coordinates of this core's grid row, then of the row leaders, for the two gather stages.
    args.writer.reserve(args.writer.size() + work.row_cols + ws.n_rows_used);
    for (uint32_t cx = 0; cx < work.row_cols; ++cx) {
        args.writer.emplace_back(static_cast<uint32_t>(vcoord(cx, work.core.y).x));
    }
    for (uint32_t cy = 0; cy < ws.n_rows_used; ++cy) {
        args.writer.emplace_back(static_cast<uint32_t>(vcoord(0, cy).y));
    }
    args.compute = {work.row_count, work.role};
    return args;
}

}  // namespace

ProgramDescriptor RMSNormBwApplyOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const bool with_dgamma = tensor_args.gamma.has_value();
    const auto ws = make_work_split(tensor_args.x);
    auto* device = tensor_args.x.device();
    const auto bufs = make_buffers(tensor_args, outputs);
    const uint32_t tile_w = tensor_args.x.tensor_spec().tile().get_width();

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbDy, 2 * ws.Wt, ws.page, ws.core_ranges));
    desc.cbs.push_back(make_cb(kCbX, 2 * ws.Wt, ws.page, ws.core_ranges));
    if (with_dgamma) {
        desc.cbs.push_back(make_cb(kCbGamma, ws.Wt, ws.page, ws.core_ranges));
    }
    desc.cbs.push_back(make_cb(kCbInvRms, 2, ws.page, ws.core_ranges));
    desc.cbs.push_back(make_cb(kCbD, 2, ws.page, ws.core_ranges));
    desc.cbs.push_back(make_cb(kCbOut, 2 * ws.Wt, ws.page, ws.core_ranges));
    if (with_dgamma) {
        desc.cbs.push_back(make_cb(kCbAcc, ws.Wt, ws.page, ws.core_ranges));
        desc.cbs.push_back(make_cb(kCbPart, ws.Wt, ws.page, ws.core_ranges));
        for (uint8_t cb_id : {kCbZeroDone, kCbGo}) {
            desc.cbs.push_back(make_cb(cb_id, 1, /*page=*/64, ws.core_ranges));
        }
        for (uint32_t sid : {kSemReady1, kSemArrive1, kSemReady2, kSemArrive2}) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = sid,
                .core_ranges = ws.core_ranges,
                .initial_value = 0,
            });
        }
    }

    uint64_t cb_bytes = 0;
    for (const auto& cb : desc.cbs) {
        cb_bytes += cb.total_size;
    }
    const uint64_t l1_budget =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    TT_FATAL(
        cb_bytes <= l1_budget,
        "rmsnorm_bw_apply: a local width of {} needs {} B of circular buffers per core, but only {} B of L1 is "
        "available. Shard the hidden dim across more devices.",
        ws.Wt * tile_w,
        cb_bytes,
        l1_budget);

    std::vector<uint32_t> reader_ct = {ws.Wt, ws.page, with_dgamma ? 1u : 0u};
    TensorAccessorArgs(*bufs.dy).append_to(reader_ct);
    TensorAccessorArgs(*bufs.x).append_to(reader_ct);
    TensorAccessorArgs(*bufs.gamma).append_to(reader_ct);
    TensorAccessorArgs(*bufs.inv_rms).append_to(reader_ct);
    TensorAccessorArgs(*bufs.d).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {
        ws.Wt, ws.page, with_dgamma ? 1u : 0u, kSemReady1, kSemArrive1, kSemReady2, kSemArrive2};
    TensorAccessorArgs(*bufs.dx).append_to(writer_ct);
    TensorAccessorArgs(*bufs.dgamma).append_to(writer_ct);

    const uint32_t neg_one_bits = std::bit_cast<uint32_t>(-1.0f);
    std::vector<uint32_t> compute_ct = {ws.Wt, neg_one_bits, with_dgamma ? 1u : 0u};

    auto reader_desc = make_kernel(
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/dataflow/"
        "reader_rmsnorm_bw_apply.cpp",
        ws.core_ranges,
        std::move(reader_ct),
        ReaderConfigDescriptor{});

    auto writer_desc = make_kernel(
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/dataflow/"
        "writer_rmsnorm_bw_apply.cpp",
        ws.core_ranges,
        std::move(writer_ct),
        WriterConfigDescriptor{});

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint8_t cb : {kCbDy, kCbX, kCbGamma, kCbInvRms, kCbD}) {
        unpack_modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (with_dgamma) {
        unpack_modes[kCbAcc] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "rmsnorm_bw_apply: fp32_dest_acc_en cannot be disabled");
    (void)packer_l1_acc;

    auto compute_desc = make_kernel(
        "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_bw_apply.cpp",
        ws.core_ranges,
        std::move(compute_ct),
        ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = std::move(unpack_modes),
            .math_approx_mode = math_approx_mode,
        });

    for (const auto& work : make_core_layout(ws)) {
        const auto args = make_core_args(ws, bufs, work, device);
        reader_desc.emplace_runtime_args(work.core, args.reader);
        writer_desc.emplace_runtime_args(work.core, args.writer);
        compute_desc.emplace_runtime_args(work.core, args.compute);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void RMSNormBwApplyOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto ws = make_work_split(tensor_args.x);
    const auto bufs = make_buffers(tensor_args, outputs);
    auto* device = tensor_args.x.device();

    // Declaring override_runtime_arguments opts out of the buffer-binding fast path, so this has to
    // re-apply every arg, addresses included.
    const auto apply = [](RuntimeArgsData& dst, const std::vector<RTArg>& src) {
        TT_FATAL(
            dst.size() == src.size(),
            "rmsnorm_bw_apply: cached program has {} runtime args but {} were re-derived",
            dst.size(),
            src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] = std::holds_alternative<Buffer*>(src[i]) ? std::get<Buffer*>(src[i])->address()
                                                             : std::get<uint32_t>(src[i]);
        }
    };

    for (const auto& work : make_core_layout(ws)) {
        const auto args = make_core_args(ws, bufs, work, device);
        apply(GetRuntimeArgs(program, kReaderKernelIdx, work.core), args.reader);
        apply(GetRuntimeArgs(program, kWriterKernelIdx, work.core), args.writer);
        apply(GetRuntimeArgs(program, kComputeKernelIdx, work.core), args.compute);
    }
}

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw
