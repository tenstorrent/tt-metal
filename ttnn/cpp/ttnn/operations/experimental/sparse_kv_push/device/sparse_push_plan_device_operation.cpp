// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "sparse_kv_push_common.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {

constexpr uint32_t kIdxAlign = 16;      // indices per core are a multiple of this, so each slice starts 64 B aligned
constexpr uint32_t kCbAlignBytes = 64;  // CB sizes are rounded up to whole 64 B
using ::sparse_push_plan::kChunkRows;

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

tt::tt_metal::KernelDescriptor plan_kernel(
    const char* source,
    const CoreRangeSet& cores,
    std::vector<uint32_t> compile_args,
    tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
    constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/kernels/";
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = std::string(kKernelDir) + source;
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = std::move(compile_args);
    kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    kernel.config = std::move(config);
    return kernel;
}

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
    validate_stripe_chunk(attrs.stripe_chunk, attrs.max_rows);
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
    packet_shape(idx.device(), attrs.page_bytes);  // checks page_bytes against the fabric payload
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
    const auto layout = PlanTableLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const tt::tt_metal::TensorLayout rm(
        DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM});
    return {
        tt::tt_metal::TensorSpec(Shape({1, 1, layout.pages, PlanTableLayout::kPageWords}), rm),
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
    const auto layout = PlanTableLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const uint32_t rows = indices.padded_shape()[2];
    using namespace ::sparse_push_plan;
    const uint32_t topk = indices.padded_shape()[3];
    auto* device = indices.device();
    const auto shape = packet_shape(device, attrs.page_bytes);

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
    // A bucket holds one worker's share of a core's indices (+1 for rounding), padded to whole 64 B.
    const uint32_t bucket_cap = pad_words((per_core + kWorkers - 1) / kWorkers + 1);
    // Rank ranges: a whole number of owner rounds (block b belongs to b % num_devices), so each range holds the same
    // number of blocks of every owner; at least 4 rounds, and a multiple of 16 blocks so every range's word ranks and
    // bitmap words are whole 16 B L1 transfers.
    constexpr uint32_t kMinOwnerRounds = 4, kRangeBlockAlign = 16;
    uint32_t range_blocks = kMinOwnerRounds * attrs.num_devices;
    while (range_blocks % kRangeBlockAlign != 0) {
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
    add_cb(kTableImage, round_up(layout.image_words * sizeof(uint32_t), kCbAlignBytes));
    add_cb(kIdx, per_core * sizeof(uint32_t));
    add_cb(kDirtyChunks, round_up(chunks, kCbAlignBytes));  // one byte per chunk
    add_cb(kDirtyMaps, kNumDirtyMaps * array_words * sizeof(uint32_t));
    // Every core's copy of every reduce chunk this core owns, worst case.
    add_cb(kScratch, max_chunks * cores * kChunkWords * sizeof(uint32_t));
    add_cb(kPartials, round_up(max_ranges * owner_stride(attrs.num_devices) * sizeof(uint32_t), kCbAlignBytes));
    add_cb(kMisc, misc_words(max_chunks) * sizeof(uint32_t));
    // Step 1's buckets (kWorkers x kWorkers of bucket_cap words), then step 2's copy lists: uint16 core ids, one list
    // of `cores` per reduce chunk.
    const uint32_t bucket_words = kWorkers * kWorkers * bucket_cap;
    const uint32_t list_words = round_up(max_chunks * cores * sizeof(uint16_t), sizeof(uint32_t)) / sizeof(uint32_t);
    add_cb(kBuckets, std::max(bucket_words, list_words) * sizeof(uint32_t));
    add_cb(kWordRanks, round_up(layout.bitmap_words * sizeof(uint16_t), kCbAlignBytes));
    for (uint32_t i = 0; i < kNumSems; ++i) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = i, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // Shared by the data movement and compute kernels.
    std::vector<uint32_t> ct = {
        attrs.num_devices,
        attrs.stripe_chunk,
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

    auto kernel = plan_kernel(
        "sparse_push_plan.cpp",
        all_cores,
        std::move(ct),
        tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    auto compute = plan_kernel(
        "sparse_push_plan_compute.cpp", all_cores, std::move(compute_ct), tt::tt_metal::ComputeConfigDescriptor{});

    // Common runtime args: every core's NoC (x, y), core 0 first, then the multicast rectangle of all of them.
    std::vector<uint32_t> common;
    for (const auto& core : core_list) {
        const auto noc_core = device->worker_core_from_logical_core(core);
        common.push_back(noc_core.x);
        common.push_back(noc_core.y);
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
        const uint32_t first_index = std::min(total_idx, i * per_core);  // this core's indices, flat
        const uint32_t index_count = std::min(total_idx, first_index + per_core) - first_index;
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.push_back(idx_buf);
        rt.push_back(table_buf);
        rt.push_back(compact_buf);
        rt.push_back(i);
        rt.push_back(first_index);
        rt.push_back(index_count);
        if (start_buf != nullptr) {
            rt.push_back(start_buf);
        }
        kernel.emplace_runtime_args(core_list[i], rt);
        tt::tt_metal::KernelDescriptor::RTArgList crt;
        crt.push_back(i);
        crt.push_back(index_count);
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
