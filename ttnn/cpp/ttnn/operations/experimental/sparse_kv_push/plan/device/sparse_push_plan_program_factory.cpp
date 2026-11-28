// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <string>
#include "sparse_push_plan_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/utils/math.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kCbAlignBytes = 64;  // CB sizes are rounded up to whole 64 B
// Kernel handles: create_descriptor pushes the kernels in this order.
constexpr tt::tt_metal::KernelHandle kPlanKernel = 0;    // BRISC
constexpr tt::tt_metal::KernelHandle kHelperKernel = 1;  // NCRISC, the same kernel and runtime args
// The compile-time arg that tells the two data movement kernels apart.
constexpr uint32_t kCtRole = 11;
// Per-core runtime args of the data movement kernels.
constexpr uint32_t kArgIndices = 0, kArgTable = 1, kArgCompact = 2, kArgSplit = 6;

::sparse_push_plan::SplitSource split_source(const SparsePushPlanParams& attrs, const SparsePushPlanInputs& args) {
    if (args.chunk_start_tensor.has_value()) {
        return ::sparse_push_plan::kSplitTensor;
    }
    return attrs.chunk_start.has_value() ? ::sparse_push_plan::kSplitValue : ::sparse_push_plan::kNoSplit;
}

tt::tt_metal::KernelDescriptor plan_kernel(
    const char* source,
    const tt::tt_metal::CoreRangeSet& cores,
    std::vector<uint32_t> compile_args,
    tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
    constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/";
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = std::string(kKernelDir) + source;
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = std::move(compile_args);
    kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    kernel.config = std::move(config);
    return kernel;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

tt::tt_metal::ProgramDescriptor SparsePushPlanProgramFactory::create_descriptor(
    const SparsePushPlanParams& attrs, const SparsePushPlanInputs& args, std::vector<Tensor>& outputs) {
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
    const uint32_t num_devices = device->shape().mesh_size();  // every chip owns a share of the cache

    // The whole worker grid, one multicast rectangle.
    const auto grid = device->compute_with_storage_grid_size();
    const std::vector<tt::tt_metal::CoreCoord> core_list =
        tt::tt_metal::grid_to_cores(plan_cores(device), grid.x, grid.y, true);
    const uint32_t cores = core_list.size();
    const tt::tt_metal::CoreRangeSet all_cores(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
    const uint32_t total_idx = rows * topk;
    const uint32_t per_core = tt::round_up(tt::div_up(total_idx, cores), kPlanIdxAlign);
    const uint32_t chunks = layout.bitmap_words / kChunkWords;
    const uint32_t max_chunks = tt::div_up(chunks, cores);  // reduce chunks per core
    const uint32_t array_words = dirty_array_words(cores);
    // A bucket holds one worker's share of a core's indices (+1 for rounding), padded to whole 64 B.
    // Workers a core: BRISC + NCRISC, or with the TRISCs too (PLAN_WORKERS=5: the dispatch-only plan only; env
    // PLAN_WORKERS=2 keeps it on the two DM RISCs).
    bool dispatch_only = false;  // (exact token of PLAN_ABL's comma list, as the kernel defines below)
    if (const char* abl_env = std::getenv("PLAN_ABL")) {
        const std::string list(abl_env);
        for (size_t at = 0; at <= list.size();) {
            size_t end = list.find(',', at);
            end = end == std::string::npos ? list.size() : end;
            dispatch_only = dispatch_only || list.substr(at, end - at) == "DISPATCH_ONLY";
            at = end + 1;
        }
    }
    // DISPATCH_ONLY serves the owner-major plans sparse_kv_dispatch reads; every other plan (sparse_kv_push's) is built
    // in full, so the switch can stay set process-wide.
    const bool dispatch_only_applies = attrs.owner_major;
    // (the rank table needs the ranks the dispatch-only plan skips)
    dispatch_only = dispatch_only && dispatch_only_applies && !attrs.rank_table;
    const char* workers_env = std::getenv("PLAN_WORKERS");
    const uint32_t workers = dispatch_only && !(workers_env != nullptr && std::string(workers_env) == "2") ? 5u : 2u;
    const uint32_t bucket_cap = pad_words(tt::div_up(per_core, workers) + 1);
    // Rank ranges: a whole number of owner rounds (block b belongs to b % num_devices), so each range holds the same
    // number of blocks of every owner; at least 4 rounds, and a multiple of 16 blocks so every range's word ranks and
    // bitmap words are whole 16 B L1 transfers.
    constexpr uint32_t kMinOwnerRounds = 4, kRangeBlockAlign = 16;
    uint32_t range_blocks = kMinOwnerRounds * num_devices;
    while (range_blocks % kRangeBlockAlign != 0) {
        range_blocks += num_devices;
    }
    const uint32_t max_ranges = tt::div_up(layout.num_blocks, range_blocks);

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
    add_cb(kTableImage, tt::round_up(layout.image_words * sizeof(uint32_t), kCbAlignBytes));
    add_cb(kIdx, per_core * sizeof(uint32_t));
    add_cb(kDirtyChunks, tt::round_up(chunks, kCbAlignBytes));  // one byte per chunk
    add_cb(kDirtyMaps, kNumDirtyMaps * array_words * sizeof(uint32_t));
    // Every core's copy of every reduce chunk this core owns, worst case.
    // Owner-major tables are built one slot at a time in kScratch, so it holds at least a slot.
    add_cb(
        kScratch,
        std::max<uint32_t>(
            max_chunks * cores * kChunkWords * sizeof(uint32_t),
            attrs.owner_major ? inbox_slot_bytes(attrs.max_rows, attrs.stripe_chunk, num_devices) : 0u));
    add_cb(kPartials, tt::round_up(max_ranges * owner_stride(num_devices) * sizeof(uint32_t), kCbAlignBytes));
    add_cb(kMisc, misc_words(max_chunks, workers) * sizeof(uint32_t));
    // Step 1's buckets (kWorkers x kWorkers of bucket_cap words), then step 2's copy lists: uint16 core ids, one list
    // of `cores` per reduce chunk.
    const uint32_t bucket_words = workers * workers * bucket_cap;
    const uint32_t list_words = tt::div_up(max_chunks * cores * sizeof(uint16_t), sizeof(uint32_t));
    add_cb(kBuckets, std::max(bucket_words, list_words) * sizeof(uint32_t));
    add_cb(kWordRanks, tt::round_up(layout.bitmap_words * sizeof(uint16_t), kCbAlignBytes));
    for (uint32_t i = 0; i < kNumSems; ++i) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = i, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    std::vector<uint32_t> ct = {
        num_devices,
        attrs.stripe_chunk,
        shape.pages_per_packet,
        shape.banks,
        cores,
        chunks,
        static_cast<uint32_t>(split_source(attrs, args)),
        layout.bitmap_words,
        bucket_cap,
        range_blocks,
        topk,
        0u,  // kCtRole: the driver (BRISC); the helper (NCRISC) gets 1
    };
    tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(compact_buf).append_to(ct);
    auto* start_buf = args.chunk_start_tensor.has_value() ? args.chunk_start_tensor->buffer() : nullptr;
    if (start_buf != nullptr) {
        tt::tt_metal::TensorAccessorArgs(start_buf).append_to(ct);
    }
    auto* rank_buf = attrs.rank_table ? outputs.back().buffer() : nullptr;  // (the last output)
    if (rank_buf != nullptr) {
        tt::tt_metal::TensorAccessorArgs(rank_buf).append_to(ct);
    }

    auto kernel = plan_kernel(
        "sparse_push_plan.cpp",
        all_cores,
        std::move(ct),
        tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

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
        } else if (attrs.chunk_start.has_value()) {
            rt.push_back(*attrs.chunk_start);
        }
        if (rank_buf != nullptr) {
            rt.push_back(rank_buf);
        }
        kernel.emplace_runtime_args(core_list[i], rt);
    }
    kernel.common_runtime_args = {common.begin(), common.end()};
    kernel.defines = {
        {"PLAN_OWNER_MAJOR", attrs.owner_major ? "1" : "0"},
        {"PLAN_RANK_TABLE", attrs.rank_table ? "1" : "0"},
        {"PLAN_RANK_PAGE_WORDS", std::to_string(rank_table_page_words(attrs.stripe_chunk))},
        {"PLAN_SLOT_BYTES",
         std::to_string(attrs.owner_major ? inbox_slot_bytes(attrs.max_rows, attrs.stripe_chunk, num_devices) : 0u)},
    };
    // Perf experiments only: PLAN_STATS (per-step DPRINT timeline) and PLAN_ABL=NAME,... (PLAN_ABL_NAME defines; see
    // the kernel). PLAN_STATS, PLAN_ABL and PLAN_WORKERS are in the program hash.
    if (std::getenv("PLAN_STATS") != nullptr) {
        kernel.defines.emplace_back("PLAN_STATS", "1");
    }
    if (const char* abl = std::getenv("PLAN_ABL")) {
        std::string list(abl);
        for (size_t at = 0; at < list.size();) {
            size_t end = list.find(',', at);
            end = end == std::string::npos ? list.size() : end;
            const std::string name = list.substr(at, end - at);
            for (char ch : name) {
                TT_FATAL(
                    (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_', "PLAN_ABL: bad name '{}'", name);
            }
            if (!name.empty() && (name != "DISPATCH_ONLY" || dispatch_only_applies)) {
                kernel.defines.emplace_back("PLAN_ABL_" + name, "1");
            }
            at = end + 1;
        }
    }
    if (workers > 2) {
        kernel.defines.emplace_back("PLAN_WORKERS", std::to_string(workers));
    }
    // Same kernel on NCRISC as worker 1.
    tt::tt_metal::KernelDescriptor helper = kernel;
    helper.compile_time_args[kCtRole] = 1;
    helper.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default};
    desc.kernels.push_back(std::move(kernel));
    desc.kernels.push_back(std::move(helper));
    if (workers >
        2) {  // workers 2-4 on the TRISCs (after the two DM kernels: their indices stay kPlanKernel / kHelperKernel)
        auto trisc = plan_kernel(
            "sparse_push_plan_trisc.cpp",
            all_cores,
            {layout.bitmap_words, cores, chunks, bucket_cap, num_devices, attrs.stripe_chunk / 32},
            tt::tt_metal::ComputeConfigDescriptor{});
        trisc.defines = {{"PLAN_WORKERS", std::to_string(workers)}};
        for (uint32_t i = 0; i < cores; ++i) {
            const uint32_t first_index = std::min(total_idx, i * per_core);
            const uint32_t index_count = std::min(total_idx, first_index + per_core) - first_index;
            tt::tt_metal::KernelDescriptor::RTArgList rt;
            rt.push_back(index_count);
            rt.push_back(i);
            trisc.emplace_runtime_args(core_list[i], rt);
        }
        desc.kernels.push_back(std::move(trisc));
    }
    return desc;
}

void SparsePushPlanProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparsePushPlanParams& attrs,
    const SparsePushPlanInputs& args,
    std::vector<Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    auto* device = args.indices.device();
    const auto grid = device->compute_with_storage_grid_size();
    const auto core_list = tt::tt_metal::grid_to_cores(plan_cores(device), grid.x, grid.y, true);
    const auto source = split_source(attrs, args);
    const uint32_t split = source == ::sparse_push_plan::kSplitTensor ? args.chunk_start_tensor->buffer()->address()
                                                                      : attrs.chunk_start.value_or(0);
    for (const auto kernel : {kPlanKernel, kHelperKernel}) {
        for (const auto& core : core_list) {
            auto& rt = tt::tt_metal::GetRuntimeArgs(program, kernel, core);
            rt[kArgIndices] = args.indices.buffer()->address();
            rt[kArgTable] = outputs[0].buffer()->address();
            rt[kArgCompact] = outputs[1].buffer()->address();
            if (source != ::sparse_push_plan::kNoSplit) {
                rt[kArgSplit] = split;
            }
            if (attrs.rank_table) {  // (the last runtime arg)
                rt[rt.size() - 1] = outputs.back().buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
