// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan_device_operation.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace plan_detail {

constexpr uint32_t kMaxCores = 120;
constexpr uint32_t kBlockAlign = 16;  // keeps every core's word / byte / count range 16 B aligned for L1 reads
constexpr uint32_t kSegments = 4;     // work units per index row
enum PlanCb : uint32_t {
    kImage = 0,
    kRow = 1,
    kPre = 2,
    kScratch = 3,
    kDirty = 4,
    kAllDirty = 5,
    kRow2 = 6,
    kSync = 7
};
constexpr uint32_t kChunkWords = 16;  // bitmap words per dirty chunk (see the kernel)

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

}  // namespace plan_detail

using namespace plan_detail;

void SparsePushPlanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& idx = args.indices;
    TT_FATAL(idx.dtype() == DataType::UINT32 && idx.layout() == Layout::ROW_MAJOR, "indices must be UINT32 ROW_MAJOR");
    TT_FATAL(idx.buffer() != nullptr && idx.buffer()->buffer_type() == BufferType::DRAM, "indices must be in DRAM");
    TT_FATAL(idx.padded_shape().rank() == 4, "indices must be [1, 1, rows, topk]");
    TT_FATAL(attrs.num_devices >= 1 && attrs.num_devices <= 32, "sparse push plan supports up to 32 devices");
    TT_FATAL(attrs.stripe_chunk % 32 == 0, "stripe_chunk {} must be a multiple of 32", attrs.stripe_chunk);
    TT_FATAL(
        attrs.max_rows % attrs.stripe_chunk == 0 && attrs.max_rows % 512 == 0,
        "max_rows {} must be a multiple of stripe_chunk {} and of 512",
        attrs.max_rows,
        attrs.stripe_chunk);
    TT_FATAL(attrs.pages_per_packet >= 1 && attrs.banks >= 1, "pages_per_packet and banks must be positive");
}

SparsePushPlanDeviceOperation::spec_return_value_t SparsePushPlanDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const tt::tt_metal::TensorLayout rm(
        DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM});
    return {
        tt::tt_metal::TensorSpec(Shape({1, 1, layout.pages, SparsePushPlanLayout::kPageWords}), rm),
        tt::tt_metal::TensorSpec(args.indices.logical_shape(), rm)};
}

SparsePushPlanDeviceOperation::tensor_return_value_t SparsePushPlanDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    return {
        create_device_tensor(specs[0], args.indices.device()), create_device_tensor(specs[1], args.indices.device())};
}

tt::tt_metal::ProgramDescriptor SparsePushPlanDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs) {
    const auto& indices = args.indices;
    auto* idx_buf = indices.buffer();
    auto* table_buf = outputs[0].buffer();
    auto* compact_buf = outputs[1].buffer();
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const uint32_t rows = indices.padded_shape()[2];
    const uint32_t topk = indices.padded_shape()[3];
    const uint32_t segs = topk % (kSegments * 16) == 0 ? kSegments : 1;  // segments stay 64 B aligned
    const uint32_t seg_len = topk / segs;
    const uint32_t units = rows * segs;

    auto* device = indices.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t core_rows = std::max<uint32_t>(1, std::min<uint32_t>(grid.y, kMaxCores / grid.x));
    const std::vector<CoreCoord> core_list = tt::tt_metal::grid_to_cores(grid.x * core_rows, grid.x, grid.y, true);
    const uint32_t cores = core_list.size();
    std::set<CoreRange> ranges;
    for (const auto& c : core_list) {
        ranges.insert(CoreRange(c, c));
    }
    const CoreRangeSet all_cores(ranges);
    const uint32_t blocks_per_core = round_up((layout.num_blocks + cores - 1) / cores, kBlockAlign);

    tt::tt_metal::ProgramDescriptor desc;
    auto add_cb = [&](uint32_t id, uint32_t bytes) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id),
                .data_format = tt::DataFormat::RawUInt32,
                .page_size = bytes}}},
        });
    };
    add_cb(kImage, layout.pages * SparsePushPlanLayout::kPageWords * sizeof(uint32_t));
    add_cb(kRow, round_up(seg_len * sizeof(uint32_t), 64));
    add_cb(kRow2, round_up(seg_len * sizeof(uint32_t), 64));
    add_cb(kPre, round_up((layout.num_blocks + attrs.num_devices) * sizeof(uint32_t), 64));  // + per-owner bases
    // Reduce scratch: every core's copy of each chunk of this core's widest possible block range.
    const uint32_t range_chunks = (blocks_per_core * (attrs.stripe_chunk / 32) + kChunkWords - 1) / kChunkWords;
    add_cb(kScratch, cores * range_chunks * kChunkWords * 4);
    const uint32_t dirty_words = round_up((layout.bitmap_words / kChunkWords + 31) / 32, 16);
    add_cb(kDirty, (16 + dirty_words) * 4);
    add_cb(kAllDirty, cores * (16 + dirty_words) * 4);
    add_cb(kSync, 64);

    auto add_semaphore = [&]() {
        const auto id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
        return id;
    };
    const uint32_t arrive_sem = add_semaphore();
    const uint32_t go_sem = add_semaphore();
    const uint32_t ready_sem = add_semaphore();
    const uint32_t or_go_sem = add_semaphore();
    const uint32_t or_done_sem = add_semaphore();

    std::vector<uint32_t> ct = {
        attrs.num_devices,
        attrs.stripe_chunk,
        SparsePushPlanLayout::kBitmapOff,
        layout.prefix_off,
        layout.counts_off,
        SparsePushPlanLayout::kOwnerOff,
        layout.pages,
        attrs.pages_per_packet,
        attrs.banks,
        topk,
        cores,
        kBlockAlign,
        arrive_sem,
        go_sem,
        SparsePushPlanLayout::kMagic,
        seg_len,
        dirty_words,
        ready_sem,
        layout.num_blocks,
        or_go_sem,
        or_done_sem,
        SparsePushPlanLayout::kOwnerPrefixOff,
        args.chunk_start.has_value() ? 1u : 0u,
    };
    tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(compact_buf).append_to(ct);
    auto* start_buf = args.chunk_start.has_value() ? args.chunk_start->buffer() : nullptr;
    if (start_buf != nullptr) {
        tt::tt_metal::TensorAccessorArgs(start_buf).append_to(ct);
    }

    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/sparse_push_plan.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.compile_time_args = std::move(ct);
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};

    std::vector<uint32_t> core_xy;
    for (const auto& c : core_list) {
        const auto phys = device->worker_core_from_logical_core(c);
        core_xy.push_back(phys.x);
        core_xy.push_back(phys.y);
    }
    const uint32_t units_per_core = (units + cores - 1) / cores;
    for (uint32_t i = 0; i < cores; ++i) {
        const uint32_t r0 = std::min(units, i * units_per_core);
        const uint32_t r1 = std::min(units, r0 + units_per_core);
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.push_back(idx_buf);
        rt.push_back(table_buf);
        rt.push_back(compact_buf);
        rt.push_back(i);
        rt.push_back(r0);
        rt.push_back(r1 - r0);
        rt.push_back((r1 - r0) / 2);
        if (start_buf != nullptr) {
            rt.push_back(start_buf);
        }
        kernel.emplace_runtime_args(core_list[i], rt);
    }
    kernel.common_runtime_args = {core_xy.begin(), core_xy.end()};
    // NCRISC takes the second half of each core's rewrite once BRISC has the table and bases ready.
    tt::tt_metal::KernelDescriptor helper = kernel;
    helper.defines.emplace_back("REWRITE_HELPER", "1");
    helper.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default};
    desc.kernels.push_back(std::move(kernel));
    desc.kernels.push_back(std::move(helper));
    return desc;
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    const std::optional<Tensor>& chunk_start) {
    using Op = ttnn::operations::experimental::high_bw_all_gather::SparsePushPlanDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_devices = num_devices,
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .pages_per_packet = pages_per_packet,
            .banks = banks},
        Op::tensor_args_t{.indices = indices, .chunk_start = chunk_start});
}

}  // namespace ttnn::prim
