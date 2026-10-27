// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "sparse_push_program.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {

constexpr uint32_t kIdxAlign = 16;  // a core's index slice starts 64 B aligned
constexpr uint32_t kChunkRows = ::sparse_push_plan::kChunkWords * 32;

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

// The plan spans the whole worker grid.
uint32_t plan_cores(tt::tt_metal::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    return grid.x * grid.y;
}

}  // namespace

void SparsePushPlanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& idx = args.indices;
    TT_FATAL(idx.dtype() == DataType::UINT32 && idx.layout() == Layout::ROW_MAJOR, "indices must be UINT32 ROW_MAJOR");
    TT_FATAL(idx.buffer() != nullptr && idx.buffer()->buffer_type() == BufferType::DRAM, "indices must be in DRAM");
    TT_FATAL(idx.padded_shape().rank() == 4, "indices must be [1, 1, rows, topk]");
    const uint32_t topk = idx.padded_shape()[3];
    TT_FATAL(topk % kIdxAlign == 0, "sparse push plan needs top-k {} to be a multiple of {}", topk, kIdxAlign);
    TT_FATAL(
        attrs.num_devices >= 1 && attrs.num_devices <= ::sparse_push::plan_table::kMaxDevices,
        "the plan table holds 1 to {} owners, got {}",
        ::sparse_push::plan_table::kMaxDevices,
        attrs.num_devices);
    sparse_push::validate_stripe_chunk(attrs.stripe_chunk, attrs.max_rows);
    TT_FATAL(
        attrs.max_rows % kChunkRows == 0,
        "sparse push plan max_rows {} must be a multiple of {} (its reduce chunk)",
        attrs.max_rows,
        kChunkRows);
    const uint32_t chunks = attrs.max_rows / kChunkRows, cores = plan_cores(idx.device());
    TT_FATAL(
        (chunks + cores - 1) / cores <= ::sparse_push_plan::kMaxChunksPerCore,
        "sparse push plan: {} chunks over {} cores exceed {} reduce chunks per core",
        chunks,
        cores,
        ::sparse_push_plan::kMaxChunksPerCore);
    sparse_push::packet_shape(idx.device(), attrs.page_bytes);  // checks page_bytes against the fabric payload
    if (args.chunk_start.has_value()) {
        const auto& start = *args.chunk_start;
        TT_FATAL(
            start.storage_type() == StorageType::DEVICE && start.buffer() != nullptr && start.device() == idx.device(),
            "chunk_start must be allocated on the indices' device");
        TT_FATAL(
            start.dtype() == DataType::UINT32 && start.logical_volume() == 1 &&
                start.memory_config().buffer_type() == BufferType::DRAM,
            "chunk_start must be a 1-element UINT32 DRAM tensor");
    }
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
    using namespace ::sparse_push_plan;
    const uint32_t topk = indices.padded_shape()[3];
    auto* device = indices.device();
    const auto shape = sparse_push::packet_shape(device, attrs.page_bytes);

    // The whole worker grid, one multicast rectangle.
    const auto grid = device->compute_with_storage_grid_size();
    const std::vector<CoreCoord> core_list = tt::tt_metal::grid_to_cores(plan_cores(device), grid.x, grid.y, true);
    const uint32_t cores = core_list.size();
    const CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    const uint32_t total_idx = rows * topk;
    const uint32_t per_core = round_up((total_idx + cores - 1) / cores, kIdxAlign);
    const uint32_t chunks = layout.bitmap_words / kChunkWords;
    const uint32_t max_chunks = (chunks + cores - 1) / cores;  // reduce chunks per core
    const uint32_t array_words = dirty_array_words(cores);
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
    add_cb(kImage, round_up(layout.image_words * sizeof(uint32_t), 64));
    add_cb(kIdx, per_core * sizeof(uint32_t));
    add_cb(kDirtyBytes, round_up(chunks, 64));
    add_cb(kAllDirty, kNumDirtyArrays * array_words * sizeof(uint32_t));
    add_cb(kScratch, max_chunks * cores * kChunkWords * 4);  // every core's copy of every reduce chunk, worst case
    add_cb(kRanges, round_up(max_ranges * attrs.num_devices * sizeof(uint32_t), 64));
    add_cb(kMisc, misc_words(max_chunks) * sizeof(uint32_t));
    // The buckets, spent after the scatter, then hold the OR's copy lists.
    add_cb(
        kBuckets, std::max(kWorkers * kWorkers * bucket_cap, round_up(max_chunks * cores, 2) / 2) * sizeof(uint32_t));
    add_cb(kRank16, round_up(layout.bitmap_words * sizeof(uint16_t), 64));
    for (uint32_t i = 0; i < kNumSems; ++i) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = i, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // Shared by the data movement and compute kernels.
    std::vector<uint32_t> ct = {
        attrs.num_devices,
        attrs.stripe_chunk,
        layout.counts_off,
        shape.pages_per_packet,
        shape.banks,
        cores,
        chunks,
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

    constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/kernels/";
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
    // Same kernel on NCRISC as worker 1.
    tt::tt_metal::KernelDescriptor helper = kernel;
    helper.defines.emplace_back("PLAN_HELPER", "1");
    helper.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default};
    desc.kernels.push_back(std::move(kernel));
    desc.kernels.push_back(std::move(helper));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    const std::optional<Tensor>& chunk_start) {
    using Op = ttnn::operations::experimental::sparse_kv_push::SparsePushPlanDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_devices = num_devices, .stripe_chunk = stripe_chunk, .max_rows = max_rows, .page_bytes = page_bytes},
        Op::tensor_args_t{.indices = indices, .chunk_start = chunk_start});
}

}  // namespace ttnn::prim
