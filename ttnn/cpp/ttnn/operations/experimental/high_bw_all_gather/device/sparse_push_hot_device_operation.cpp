// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_hot_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace hot_detail {

constexpr uint32_t kMaxCores = 120;
// Circular buffers, semaphores and misc-buffer size: see kernels/sparse_push_hot_work.hpp.
enum HotCb : uint32_t {
    kIdx = 0,
    kCompact = 1,
    kTables = 2,
    kHot = 3,
    kRank16 = 4,
    kRanges = 5,
    kMisc = 6,
    kOut = 7,
    kCold = 8
};
constexpr uint32_t kWorkers = 5;
constexpr uint32_t kNumSems = 3 + kWorkers;
constexpr uint32_t kMiscWords = 368;
constexpr uint32_t kIdxAlign = 16;

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

// Rank ranges: whole owner rounds (block b belongs to b % num_devices), 16-block aligned for the L1 transfers.
uint32_t range_blocks(uint32_t num_devices) {
    uint32_t blocks = 2 * num_devices;
    while (blocks % 16 != 0) {
        blocks += num_devices;
    }
    return blocks;
}

}  // namespace hot_detail

void SparsePushHotDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    using namespace hot_detail;
    for (const Tensor* t : {&args.indices, &args.compact, &args.table, &args.tables, &args.chunk_start}) {
        TT_FATAL(t->dtype() == DataType::UINT32 && t->layout() == Layout::ROW_MAJOR, "inputs must be UINT32 ROW_MAJOR");
        TT_FATAL(t->buffer() != nullptr && t->buffer()->buffer_type() == BufferType::DRAM, "inputs must be in DRAM");
    }
    TT_FATAL(args.indices.padded_shape() == args.compact.padded_shape(), "indices and compact must match");
    TT_FATAL(attrs.num_devices >= 1 && attrs.num_devices <= 32, "sparse push hot supports up to 32 devices");
    TT_FATAL(attrs.threshold >= 1 && attrs.threshold <= 32, "threshold {} must be in [1, 32]", attrs.threshold);
    TT_FATAL(attrs.stripe_chunk % 32 == 0, "stripe_chunk {} must be a multiple of 32", attrs.stripe_chunk);
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    TT_FATAL(
        args.table.padded_shape()[2] == layout.pages, "table must be this chip's plan table ({} pages)", layout.pages);
    TT_FATAL(
        args.tables.padded_shape()[2] == layout.pages * attrs.num_devices,
        "tables must hold every chip's plan table ({} pages)",
        layout.pages * attrs.num_devices);
}

SparsePushHotDeviceOperation::spec_return_value_t SparsePushHotDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const tt::tt_metal::TensorLayout rm(
        DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM});
    return tt::tt_metal::TensorSpec(Shape({1, 1, layout.pages, SparsePushPlanLayout::kPageWords}), rm);
}

SparsePushHotDeviceOperation::tensor_return_value_t SparsePushHotDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.indices.device());
}

tt::tt_metal::ProgramDescriptor SparsePushHotDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    using namespace hot_detail;
    auto* idx_buf = args.indices.buffer();
    auto* compact_buf = args.compact.buffer();
    auto* table_buf = args.table.buffer();
    auto* tables_buf = args.tables.buffer();
    auto* start_buf = args.chunk_start.buffer();
    auto* hot_buf = output.buffer();
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const uint32_t rows = args.indices.padded_shape()[2];
    const uint32_t topk = args.indices.padded_shape()[3];
    TT_FATAL(topk % kIdxAlign == 0, "sparse push hot needs top-k {} to be a multiple of {}", topk, kIdxAlign);

    auto* device = args.indices.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t core_rows = std::max<uint32_t>(1, std::min<uint32_t>(grid.y, kMaxCores / grid.x));
    const std::vector<CoreCoord> core_list = tt::tt_metal::grid_to_cores(grid.x * core_rows, grid.x, grid.y, true);
    const uint32_t cores = core_list.size();
    const CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, core_rows - 1)));
    const uint32_t total_idx = rows * topk;
    const uint32_t per_core = round_up((total_idx + cores - 1) / cores, kIdxAlign);
    const uint32_t blocks_per_range = range_blocks(attrs.num_devices);
    const uint32_t max_ranges = (layout.num_blocks + blocks_per_range - 1) / blocks_per_range;
    TT_FATAL(max_ranges <= cores, "sparse push hot: {} rank ranges exceed {} cores", max_ranges, cores);
    TT_FATAL(cores > attrs.num_devices, "sparse push hot needs more than {} cores", attrs.num_devices);
    const uint32_t range_words = blocks_per_range * (attrs.stripe_chunk / 32);

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
    add_cb(kIdx, per_core * sizeof(uint32_t));
    add_cb(kCompact, per_core * sizeof(uint32_t));
    // Every requester's bitmap words of the core's range, then this chip's own.
    add_cb(kTables, (attrs.num_devices + 1) * range_words * sizeof(uint32_t));
    for (uint32_t id : {kHot, kCold}) {
        add_cb(id, round_up(layout.bitmap_words * sizeof(uint32_t), 64));
    }
    // The hot rows' ranks and prefixes, then the cold rows' (same layout, 64 B aligned).
    add_cb(kRank16, 2 * round_up(layout.bitmap_words * sizeof(uint16_t), 64));
    // Counts and transposed prefixes per (range, owner, part), hot then cold (kernels/sparse_push_hot_work.hpp).
    const uint32_t range_stride = round_up(max_ranges, 4);
    const uint32_t counts_words = round_up(max_ranges * attrs.num_devices * 2, 16);
    const uint32_t cold_words = counts_words + round_up(2 * attrs.num_devices * range_stride, 16);
    add_cb(kRanges, 2 * cold_words * sizeof(uint32_t));
    add_cb(kMisc, kMiscWords * sizeof(uint32_t));
    // A table header (cores 0..N: requester c's, then this chip's own with its owner counts), then the hot header.
    add_cb(
        kOut,
        std::max((SparsePushPlanLayout::kOwnerOff + attrs.num_devices) * 4, SparsePushPlanLayout::kBitmapOff * 4));
    for (uint32_t i = 0; i < kNumSems; ++i) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = i, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // Shared by the data movement and compute kernels (the compute kernel reads the first 13).
    std::vector<uint32_t> ct = {
        attrs.num_devices,
        attrs.stripe_chunk,
        SparsePushHotLayout::kBitmapOff,
        layout.pages,
        attrs.pages_per_packet,
        attrs.banks,
        cores,
        attrs.threshold,
        attrs.prefix_cap,
        attrs.current_cap,
        blocks_per_range,
        SparsePushHotLayout::kMagic,
        SparsePushHotLayout::kBasesOff,
        topk,
        SparsePushPlanLayout::kOwnerOff,
        layout.num_blocks,
    };
    std::vector<uint32_t> compute_ct = ct;
    tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(compact_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(tables_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(hot_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(start_buf).append_to(ct);

    constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = std::string(kKernelDir) + "sparse_push_hot.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.compile_time_args = std::move(ct);
    kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};
    tt::tt_metal::KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "sparse_push_hot_compute.cpp";
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
        rt.push_back(compact_buf);
        rt.push_back(table_buf);
        rt.push_back(tables_buf);
        rt.push_back(hot_buf);
        rt.push_back(start_buf);
        rt.push_back(i);
        rt.push_back(i0);
        rt.push_back(n);
        kernel.emplace_runtime_args(core_list[i], rt);
        tt::tt_metal::KernelDescriptor::RTArgList crt;
        crt.push_back(i);
        crt.push_back(n);
        compute.emplace_runtime_args(core_list[i], crt);
    }
    kernel.common_runtime_args = {common.begin(), common.end()};
    // NCRISC is worker 1.
    tt::tt_metal::KernelDescriptor helper = kernel;
    helper.defines.emplace_back("HOT_HELPER", "1");
    helper.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default};
    desc.kernels.push_back(std::move(kernel));
    desc.kernels.push_back(std::move(helper));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

Tensor sparse_push_hot(
    const Tensor& indices,
    const Tensor& compact,
    const Tensor& table,
    const Tensor& tables,
    const Tensor& chunk_start,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    uint32_t threshold,
    uint32_t prefix_cap,
    uint32_t current_cap) {
    using Op = ttnn::operations::experimental::high_bw_all_gather::SparsePushHotDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_devices = num_devices,
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .pages_per_packet = pages_per_packet,
            .banks = banks,
            .threshold = threshold,
            .prefix_cap = prefix_cap,
            .current_cap = current_cap},
        Op::tensor_args_t{
            .indices = indices, .compact = compact, .table = table, .tables = tables, .chunk_start = chunk_start});
}

}  // namespace ttnn::prim
