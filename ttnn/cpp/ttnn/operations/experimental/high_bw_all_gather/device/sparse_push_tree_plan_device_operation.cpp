// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_tree_plan_device_operation.hpp"

#include <algorithm>
#include <set>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace tree_plan_detail {

constexpr uint32_t kMaxCores = 120;
constexpr uint32_t kCntWords = 208;  // per-core group counts (194 used), 64 B multiple
constexpr uint32_t kRecordBytes = 128;
enum TreeCb : uint32_t {
    kBits = 0,
    kCnt = 1,
    kAllCnt = 2,
    kOpen = 3,
    kCarry = 4,
    kScratch = 5,
    kRw = 6,
    kIdx = 7,
    kVars = 8,
    kDens = 9,
    kCache = 10,
    kHandoff = 11,
    kBlk = 12
};
constexpr uint32_t kCacheRows = 24576;  // rows a planner core remembers between its passes
constexpr uint32_t kWin = 256;  // must match the kernel's bitmap window

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

uint32_t senders(const SparsePushTreePlanParams& p) { return 4 * p.links * p.senders_per_link; }

}  // namespace tree_plan_detail

namespace td = tree_plan_detail;

void SparsePushTreePlanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const uint32_t n_dev = attrs.mesh_rows * attrs.mesh_cols;
    TT_FATAL(n_dev == 32 && attrs.mesh_cols == 4, "tree plan (prototype) supports the 8x4 mesh");
    TT_FATAL(td::senders(attrs) == 16, "tree plan (prototype) needs 2 links x 2 senders per channel");
    TT_FATAL(attrs.levels == 6, "tree plan (prototype) uses 6 levels");
    TT_FATAL(8 + 2 * attrs.pages_per_packet <= 32, "packet rows must fit a 128 B record");
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    TT_FATAL(
        args.tables.padded_shape()[2] == layout.pages * n_dev,
        "tables must hold every requester's plan table ({} pages)",
        layout.pages * n_dev);
    TT_FATAL(attrs.slot_capacity % 16 == 0, "slot_capacity must be a multiple of 16");
    for (const auto* t : {&args.tables, &args.indices, &args.rank, &args.paths}) {
        TT_FATAL(t->dtype() == DataType::UINT32 && t->layout() == Layout::ROW_MAJOR, "inputs must be UINT32 ROW_MAJOR");
        TT_FATAL(t->buffer() != nullptr && t->buffer()->buffer_type() == BufferType::DRAM, "inputs must be in DRAM");
    }
}

SparsePushTreePlanDeviceOperation::spec_return_value_t SparsePushTreePlanDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const tt::tt_metal::TensorLayout rm(
        DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM});
    const uint32_t stream_words = 32 + attrs.stream_records * (td::kRecordBytes / 4);
    return {
        tt::tt_metal::TensorSpec(Shape({1, 1, td::senders(attrs), stream_words}), rm),
        tt::tt_metal::TensorSpec(args.indices.logical_shape(), rm),
        tt::tt_metal::TensorSpec(Shape({1, 1, 1, attrs.slot_capacity}), rm),
        tt::tt_metal::TensorSpec(Shape({1, 1, 1, attrs.max_rows / 32}), rm)};
}

SparsePushTreePlanDeviceOperation::tensor_return_value_t SparsePushTreePlanDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    auto* device = args.indices.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device),
        create_device_tensor(specs[3], device)};
}

tt::tt_metal::ProgramDescriptor SparsePushTreePlanDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs) {
    const uint32_t n_dev = attrs.mesh_rows * attrs.mesh_cols;
    const auto layout = SparsePushPlanLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const uint32_t rows = args.indices.padded_shape()[2];
    const uint32_t topk = args.indices.padded_shape()[3];
    const uint32_t W = attrs.max_rows / 32;
    const uint32_t P = attrs.pages_per_packet, B = attrs.banks;
    const uint32_t stage_pages = td::round_up(td::round_up(attrs.stage_rows, P) / P + attrs.levels, B) / B * P * B;
    const uint32_t max_blocks = td::round_up(layout.num_blocks, 16);

    auto* device = args.indices.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t core_rows = std::max<uint32_t>(1, std::min<uint32_t>(grid.y, td::kMaxCores / grid.x));
    const uint32_t cores = grid.x * core_rows;
    const auto core_list = tt::tt_metal::grid_to_cores(cores, grid.x, grid.y, true);
    std::set<CoreRange> ranges;
    for (const auto& c : core_list) {
        ranges.insert(CoreRange(c, c));
    }
    const CoreRangeSet all_cores(ranges);
    const uint32_t wpc = td::round_up((W + cores - 1) / cores, 16);

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
    const uint32_t kGroups = 4 * attrs.levels * 2 * 2 + 2;
    add_cb(td::kBits, 33 * td::kWin * 4);
    add_cb(td::kCnt, td::kCntWords * 4);
    add_cb(td::kAllCnt, cores * td::kCntWords * 4);
    add_cb(td::kOpen, kGroups * td::kRecordBytes);
    add_cb(td::kCarry, kGroups * td::kRecordBytes);
    add_cb(td::kScratch, 3072);
    add_cb(td::kRw, (max_blocks + td::round_up(W, 16)) * 4);
    add_cb(td::kIdx, 2 * 2048);
    add_cb(td::kVars, 8192);
    add_cb(td::kDens, max_blocks * 4);
    add_cb(td::kBlk, max_blocks * 4);
    add_cb(td::kCache, td::kCacheRows * 8);
    add_cb(td::kHandoff, 1024);

    auto add_semaphore = [&]() {
        const auto id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
        return id;
    };
    const uint32_t arrive_sem = add_semaphore();
    const uint32_t go_sem = add_semaphore();
    const uint32_t handoff_sem = add_semaphore();

    auto* streams_buf = outputs[0].buffer();
    std::vector<uint32_t> ct = {
        n_dev,
        attrs.mesh_rows,
        attrs.mesh_cols,
        attrs.stripe_chunk,
        P,
        B,
        attrs.links,
        attrs.senders_per_link,
        attrs.levels,
        stage_pages,
        layout.pages,
        SparsePushPlanLayout::kBitmapOff,
        layout.counts_off,
        SparsePushPlanLayout::kOwnerOff,
        W,
        wpc,
        cores,
        rows,
        topk,
        streams_buf->aligned_page_size(),
        arrive_sem,
        go_sem,
        attrs.slot_capacity,
        td::kCntWords,
        max_blocks,
        td::kCacheRows,
        handoff_sem,
    };
    for (const auto* buf :
         {args.tables.buffer(),
          args.indices.buffer(),
          args.rank.buffer(),
          streams_buf,
          outputs[1].buffer(),
          outputs[2].buffer(),
          outputs[3].buffer(),
          args.paths.buffer()}) {
        tt::tt_metal::TensorAccessorArgs(buf).append_to(ct);
    }

    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/sparse_push_tree_plan.cpp";
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
    for (uint32_t i = 0; i < cores; ++i) {
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.push_back(i);
        rt.push_back(args.tables.buffer());
        rt.push_back(args.indices.buffer());
        rt.push_back(args.rank.buffer());
        rt.push_back(streams_buf);
        rt.push_back(outputs[1].buffer());
        rt.push_back(outputs[2].buffer());
        rt.push_back(outputs[3].buffer());
        rt.push_back(args.paths.buffer());
        kernel.emplace_runtime_args(core_list[i], rt);
    }
    kernel.common_runtime_args = {core_xy.begin(), core_xy.end()};
    // NCRISC rewrites the indices as soon as BRISC has the group counts, while BRISC emits records.
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

std::vector<Tensor> sparse_push_tree_plan(
    const Tensor& tables,
    const Tensor& indices,
    const Tensor& rank,
    const Tensor& paths,
    const ttnn::operations::experimental::high_bw_all_gather::SparsePushTreePlanParams& params) {
    using Op = ttnn::operations::experimental::high_bw_all_gather::SparsePushTreePlanDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        params, Op::tensor_args_t{.tables = tables, .indices = indices, .rank = rank, .paths = paths});
}

}  // namespace ttnn::prim
