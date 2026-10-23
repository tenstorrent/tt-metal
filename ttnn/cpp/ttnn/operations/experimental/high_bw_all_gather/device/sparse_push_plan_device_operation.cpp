// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace plan_detail {

constexpr uint32_t kMaxCores = 120;
// Circular buffers, semaphores and misc-buffer size: see kernels/sparse_push_plan_work.hpp.
enum PlanCb : uint32_t {
    kImage = 0,
    kIdx = 1,
    kDirtyBytes = 2,
    kAllDirty = 3,
    kScratch = 4,
    kRanges = 5,
    kMisc = 6,
    kBuckets = 7,
    kRank16 = 8
};
constexpr uint32_t kWorkers = 5;
constexpr uint32_t kNumSems = 9;
constexpr uint32_t kMiscCopies = 160;
constexpr uint32_t kChunkWords = 16;  // bitmap words per dirty chunk (512 rows)
constexpr uint32_t kIdxAlign = 16;    // a core's index slice starts 64 B aligned

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
    TT_FATAL(topk % kIdxAlign == 0, "sparse push plan needs top-k {} to be a multiple of {}", topk, kIdxAlign);

    // Whole rows of the worker grid, so the cores form one multicast rectangle.
    auto* device = indices.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t core_rows = std::max<uint32_t>(1, std::min<uint32_t>(grid.y, kMaxCores / grid.x));
    const std::vector<CoreCoord> core_list = tt::tt_metal::grid_to_cores(grid.x * core_rows, grid.x, grid.y, true);
    const uint32_t cores = core_list.size();
    const CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, core_rows - 1)));
    const uint32_t total_idx = rows * topk;
    const uint32_t per_core = round_up((total_idx + cores - 1) / cores, kIdxAlign);
    const uint32_t chunks = layout.bitmap_words / kChunkWords;
    const uint32_t max_chunks = (chunks + cores - 1) / cores;  // reduce chunks per core
    TT_FATAL(
        max_chunks <= 32, "sparse push plan: {} chunks over {} cores exceed 32 reduce chunks per core", chunks, cores);
    const uint32_t array_words = round_up(cores, 16);  // see dirty_array_words in kernels/sparse_push_plan_work.hpp
    const uint32_t bucket_cap = round_up((per_core + kWorkers - 1) / kWorkers + 1, 16);
    // Rank ranges: whole owner rounds (block b belongs to b % num_devices), 16-block aligned for the L1 transfers.
    uint32_t range_blocks = 4 * attrs.num_devices;
    while (range_blocks % 16 != 0) {
        range_blocks += attrs.num_devices;
    }
    const uint32_t max_ranges = (layout.num_blocks + range_blocks - 1) / range_blocks;

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
    add_cb(kIdx, per_core * sizeof(uint32_t));
    add_cb(kDirtyBytes, round_up(chunks, 64));
    add_cb(kAllDirty, 3 * array_words * 4);
    add_cb(kScratch, max_chunks * cores * kChunkWords * 4);  // every core's copy of every reduce chunk, worst case
    add_cb(kRanges, round_up(max_ranges * attrs.num_devices * sizeof(uint32_t), 64));
    add_cb(kMisc, (kMiscCopies + round_up(max_chunks, 16)) * sizeof(uint32_t));
    // The buckets, spent after the scatter, then hold the OR's copy lists.
    add_cb(
        kBuckets, std::max(kWorkers * kWorkers * bucket_cap, round_up(max_chunks * cores, 2) / 2) * sizeof(uint32_t));
    add_cb(kRank16, round_up(layout.bitmap_words * sizeof(uint16_t), 64));
    for (uint32_t i = 0; i < kNumSems; ++i) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = i, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // Shared by the data movement and compute kernels (the compute kernel reads the first 18).
    std::vector<uint32_t> ct = {
        attrs.num_devices,
        attrs.stripe_chunk,
        SparsePushPlanLayout::kBitmapOff,
        layout.prefix_off,
        layout.counts_off,
        SparsePushPlanLayout::kOwnerOff,
        SparsePushPlanLayout::kOwnerPrefixOff,
        attrs.pages_per_packet,
        attrs.banks,
        cores,
        SparsePushPlanLayout::kMagic,
        chunks,
        layout.num_blocks,
        max_chunks,
        args.chunk_start.has_value() ? 1u : 0u,
        layout.bitmap_words,
        bucket_cap,
        range_blocks,
        topk,
    };
    std::vector<uint32_t> compute_ct = ct;
    tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(compact_buf).append_to(ct);
    auto* start_buf = args.chunk_start.has_value() ? args.chunk_start->buffer() : nullptr;
    if (start_buf != nullptr) {
        tt::tt_metal::TensorAccessorArgs(start_buf).append_to(ct);
    }

    constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = std::string(kKernelDir) + "sparse_push_plan.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.compile_time_args = std::move(ct);
    kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};
    tt::tt_metal::KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "sparse_push_plan_compute.cpp";
    compute.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct);
    compute.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    compute.config = tt::tt_metal::ComputeConfigDescriptor{};

    std::vector<uint32_t> common;
    for (const auto& c : core_list) {
        const auto phys = device->worker_core_from_logical_core(c);
        common.push_back(phys.x);
        common.push_back(phys.y);
    }
    const auto mcast_start = device->worker_core_from_logical_core(core_list.front());
    const auto mcast_end = device->worker_core_from_logical_core(core_list.back());
    common.insert(
        common.end(),
        {static_cast<uint32_t>(mcast_start.x),
         static_cast<uint32_t>(mcast_start.y),
         static_cast<uint32_t>(mcast_end.x),
         static_cast<uint32_t>(mcast_end.y)});
    for (uint32_t i = 0; i < cores; ++i) {
        const uint32_t i0 = std::min(total_idx, i * per_core);
        const uint32_t n = std::min(total_idx, i0 + per_core) - i0;
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.push_back(idx_buf);
        rt.push_back(table_buf);
        rt.push_back(compact_buf);
        rt.push_back(i);
        rt.push_back(i0);
        rt.push_back(n);
        if (start_buf != nullptr) {
            rt.push_back(start_buf);
        }
        kernel.emplace_runtime_args(core_list[i], rt);
        tt::tt_metal::KernelDescriptor::RTArgList crt;
        crt.push_back(i);
        crt.push_back(n);
        compute.emplace_runtime_args(core_list[i], crt);
    }
    kernel.common_runtime_args = {common.begin(), common.end()};
    // NCRISC is worker 1: a share of every L1 loop and of the reduce reads.
    tt::tt_metal::KernelDescriptor helper = kernel;
    helper.defines.emplace_back("PLAN_HELPER", "1");
    helper.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default};
    desc.kernels.push_back(std::move(kernel));
    desc.kernels.push_back(std::move(helper));
    desc.kernels.push_back(std::move(compute));
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
