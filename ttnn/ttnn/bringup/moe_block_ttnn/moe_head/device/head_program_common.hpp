// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's program builders' shared constants and helpers (head_model.cpp, head_*_program.cpp,
// head_row_blocks.cpp, head_workload.cpp, head_weights.cpp): kernel / CB / semaphore ids, descriptor helpers, a
// runtime arg with its source address (Arg), the activation's compile-time values. Internal to moe_head.
#pragma once

#include "head_program.hpp"
#include "../../common/kernels/tile_bytes.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <numeric>
#include <map>
#include <set>
#include <utility>
#include <variant>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/kernel_lib/mcast/host/mcast.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "../../common/kernels/combine_plan_words.hpp"
#include "kernels/head_shared_words.hpp"
#include "kernels/head_named_args.hpp"
#include "../../common/args_table.hpp"
#include "../../common/chip_program.hpp"

namespace ttnn::operations::bringup::moe_block::moe_head {

namespace head_program_detail {

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::distributed::MeshDevice;

constexpr const char* kKernelDirectory = "ttnn/ttnn/bringup/moe_block_ttnn/moe_head/device/kernels/";

// the matmul cores' (shared expert's) constants
constexpr uint32_t kWSlots = 3;
constexpr uint32_t kXReadySemaphore = 0, kWValidSemaphore = 1, kWFreeSemaphore = 1 + kWSlots;
constexpr uint32_t kXLoadedSemaphore = 1 + 2 * kWSlots;
constexpr uint32_t kNumOwnSemaphores = 2 + 2 * kWSlots;
// (the matmul cores' phase words, the "ewords" persistent words: byte offsets, a flag each, kFlagBytes apart; the
// kernels get them as the *_flag_id named args, router_go_flag_id on the router cores)
constexpr uint32_t kRouterXReadyFlag = 0, kRouterWReadyFlag = 16, kRouterXSentFlag = 32, kXReadLocalFlag = 96,
                   kXAckFlag = 128, kWAckFlag = 144;
// (router iterations: router core 0's go count; skip_padding_chunks: the BRISC's live row chunks - the named arg
// live_chunks_flag_id)
constexpr uint32_t kRouterGoFlag = 160, kLiveChunksFlag = 240;
// (the "ewords" region's bytes: up to its last flag - 256, 64 words)
constexpr uint32_t kEngineWordsBytes = kLiveChunksFlag + kFlagBytes;
static_assert(
    kRouterXReadyFlag < kLiveChunksFlag && kRouterWReadyFlag < kLiveChunksFlag && kRouterXSentFlag < kLiveChunksFlag &&
        kXReadLocalFlag < kLiveChunksFlag && kXAckFlag < kLiveChunksFlag && kWAckFlag < kLiveChunksFlag &&
        kRouterGoFlag < kLiveChunksFlag,
    "the ewords region ends at its last flag, kLiveChunksFlag");
static_assert(kEngineWordsBytes == 256, "the ewords region: 64 words");
// (a port's mailbox, the "mbox" persistent words - bytes from its base: the reduce-scatter hand-off words
// (handoff_bytes a matmul core), the latent's fence word at their end and the latent hand-off words kLatentHandoffsGap
// past it; then the logits words, the port words and the zone words, a flag each, kFlagBytes apart: zone free, zone
// credit, router done, the logits' fence - head_model.cpp make_head_model's offsets)
constexpr uint32_t kLatentHandoffsGap = 128;
constexpr uint32_t kZoneFreeWord = 0, kZoneCreditWord = 16, kRouterDoneWord = 32, kLogitsFenceWord = 48;
constexpr uint32_t kZoneWordsBytes = kLogitsFenceWord + kFlagBytes;  // 64
static_assert(kZoneWordsBytes == 64, "the zone words: 4 flags");
// (a router core's slot-free words, the "sfree" persistent words: a direction's kSlotFreeUnits flags - a unit's each,
// router_fabric_sends.hpp - X forward's, then X backward's)
constexpr uint32_t kSlotFreeUnits = 4;
enum SharedExpertCb : uint8_t {
    SHARED_EXPERT_CB_X = 0,
    SHARED_EXPERT_CB_W = 1,
    SHARED_EXPERT_CB_H_LOCAL = 2,
    SHARED_EXPERT_CB_H_GATHERED = 3,
    SHARED_EXPERT_CB_PARTIAL_SUMS = 5
};
constexpr uint8_t kReduceScatterHoldCb = 6, kReduceScatterFinalBlockCb = 7, kReduceScatterArrivalForwardCb = 8,
                  kReduceScatterArrivalBackwardCb = 9, kReduceScatterFinalCounterCb = 10,
                  kReduceScatterOutputForwardCb = 11, kReduceScatterPassSyncCb = 12, kReduceScatterFinalFreeCb = 20;
constexpr uint8_t kLatentCb = 13, kDispatchPrepUnitCb = 14, kDispatchPrepOutCb = 15, kLatentReadyCb = 16;
constexpr uint8_t kRouterXCb = 17, kRouterWCb = 18, kRouterOutCb = 19;
// the x gather's
constexpr uint32_t kArrivalIncrementEvery = 8;
constexpr uint8_t kXGatherCb = 0;
// the router cores' CB ids
enum RouterCb : uint8_t {
    ROUTER_CB_LOGITS = 2,
    ROUTER_CB_ARRIVALS = 3,
    ROUTER_CB_ARRIVAL_ORDER = 4,
    ROUTER_CB_BIAS = 5,
    ROUTER_CB_SCORES = 6,
    ROUTER_CB_PACK_TO_UNPACK = 8,
    ROUTER_CB_BEST = 9,
    ROUTER_CB_EXPERT_INDEX = 11,
    ROUTER_CB_READY = 12
};
constexpr uint32_t kPlanWords = moe_block_plan_words::kPlanWords;  // (the combine's device plan)

inline uint32_t u32(size_t value) { return static_cast<uint32_t>(value); }
inline uint32_t noc_xy_word(const CoreCoord& core) {  // (a core's NoC x | y << 16)
    return moe_head_port_words::kNocXyX.pack(u32(core.x)) | moe_head_port_words::kNocXyY.pack(u32(core.y));
}
inline uint32_t noc_word(const CoreCoord& core) {  // (a core's NoC x | y << 8)
    return moe_head_port_words::kNocX.pack(u32(core.x)) | moe_head_port_words::kNocY.pack(u32(core.y));
}
using head_math::align_to_16;
using head_math::align_to_64;
using head_math::ceil_div;

inline CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(ranges);
}

inline KernelDescriptor kernel(
    const std::string& file, const CoreRangeSet& cores, KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor descriptor;
    descriptor.kernel_source = std::string(kKernelDirectory) + file;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = cores;
    descriptor.config = std::move(config);
    return descriptor;
}

inline DataMovementConfigDescriptor data_movement(DataMovementProcessor risc, NOC noc) {
    return DataMovementConfigDescriptor{.processor = risc, .noc = noc};
}

// A define: only what no named arg can carry - the per-chip values (chip_values turns them into reads of the chip's
// DRAM table block), the list-valued ROUTE_WORDS_*_INIT, ROUTER_SQRTSOFTPLUS with the LLK's INP_FLOAT32,
// WATCHER_ROUTING_ARGS_OFFSET (also a per-chip value: it selects the fabric headers the watcher includes)
inline void define(KernelDescriptor& kernel_descriptor, const std::string& name, const std::string& value = "") {
    kernel_descriptor.defines.emplace_back(name, value);
}
// A named compile-time arg (kernels/head_named_args.hpp: the names the kernels read), set or replaced.
inline void named_arg(KernelDescriptor& kernel_descriptor, const char* name, int64_t value) {
    TT_FATAL(value >= 0 && value <= int64_t(UINT32_MAX), "moe_head: named compile-time arg {} = {}", name, value);
    set_named_compile_time_arg(kernel_descriptor, name, static_cast<uint32_t>(value));
}
inline std::string integer_literal(int64_t value) { return std::to_string(value); }

// a CB in the arena, offset bytes into it
inline CBDescriptor arena_cb(
    Buffer* arena,
    uint8_t cb_index,
    int64_t offset,
    uint32_t size,
    tt::DataFormat data_format,
    uint32_t page_size,
    const CoreRangeSet& cores) {
    CBDescriptor descriptor;
    descriptor.total_size = size;
    descriptor.core_ranges = cores;
    descriptor.format_descriptors = {
        CBFormatDescriptor{.buffer_index = cb_index, .data_format = data_format, .page_size = page_size}};
    descriptor.buffer = arena;
    descriptor.address_offset = u32(offset);
    return descriptor;
}

inline void append_accessor(std::vector<uint32_t>& compile_time_args, const Buffer* buffer) {
    tt::tt_metal::TensorAccessorArgs(*buffer).append_to(compile_time_args);
}
inline size_t accessor_length(const Buffer* buffer) {
    std::vector<uint32_t> args;
    append_accessor(args, buffer);
    return args.size();
}

enum class XDirection : uint8_t { Forward = 0, Backward = 1 };
inline XDirection opposite(XDirection direction) {
    return direction == XDirection::Forward ? XDirection::Backward : XDirection::Forward;
}
inline ReduceScatterDirection reduce_scatter_direction(XDirection direction) {
    return direction == XDirection::Forward ? ReduceScatterDirection::Forward : ReduceScatterDirection::Backward;
}

using Coord = std::pair<uint32_t, uint32_t>;  // (mesh row, column)

using head_math::positive_modulo;

// A runtime arg: a value, and for an address the source it points into (HeadSource) and the offset there.
// a fabric connection's args (TENSIX VC0: eth channel, teardown and buffer-index semaphores); the first carries
// kFabricConnectionSource as its source (not an address: ChipProgram::set_runtime_args records where they sit)
inline constexpr uint32_t kFabricConnectionArgs = 3;
inline constexpr int32_t kFabricConnectionSource = -2;
struct Arg {
    uint32_t value = 0;
    int32_t source = -1;
    uint32_t offset = 0;
    Arg() = default;
    Arg(uint32_t value) : value(value) {}  // NOLINT(google-explicit-constructor): plain values read as args
    Arg(uint32_t value, int32_t source, uint32_t offset) : value(value), source(source), offset(offset) {}
    bool operator==(const Arg&) const = default;
};
// an address arg further into its source
inline Arg operator+(const Arg& arg, uint32_t delta) {
    return Arg(arg.value + delta, arg.source, arg.source >= 0 ? arg.offset + delta : 0);
}
using Args = std::vector<Arg>;

inline void add(Args& args, uint32_t value) { args.emplace_back(value); }
inline void add(Args& args, const Arg& arg) { args.push_back(arg); }
inline void add(Args& args, const std::vector<uint32_t>& words) {
    for (uint32_t word : words) {
        args.emplace_back(word);
    }
}
inline void add(Args& args, const Args& more) { args.insert(args.end(), more.begin(), more.end()); }

inline std::vector<uint32_t> values(const Args& args) {
    std::vector<uint32_t> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
        out.push_back(arg.value);
    }
    return out;
}

inline tt::DataFormat weight_format(tt::tt_metal::DataType dtype) {
    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        default: return tt::DataFormat::Bfp8_b;
    }
}

}  // namespace head_program_detail

using namespace head_program_detail;

}  // namespace ttnn::operations::bringup::moe_block::moe_head
