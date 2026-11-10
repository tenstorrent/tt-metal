// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's program builders' shared constants and helpers (head_model.cpp, head_*_program.cpp,
// head_row_blocks.cpp, head_workload.cpp, head_weights.cpp): kernel / CB / semaphore ids, descriptor helpers, a
// runtime arg with its source address (Arg), the activation's compile-time values. Internal to moe_head.
#pragma once

#include "head_program.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
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
#include "ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"
#include "ttnn/operations/experimental/moe_block/common/args_table.hpp"
#include "ttnn/operations/experimental/moe_block/common/chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

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

constexpr const char* kKernelDirectory = "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/";

// the matmul cores' (shared expert's) constants
constexpr uint32_t kWSlots = 3;
constexpr uint32_t kXReadySemaphore = 0, kWValidSemaphore = 1, kWFreeSemaphore = 1 + kWSlots;
constexpr uint32_t kXLoadedSemaphore = 1 + 2 * kWSlots;
constexpr uint32_t kNumOwnSemaphores = 2 + 2 * kWSlots;
constexpr uint32_t kRouterXReadyFlag = 0, kRouterWReadyFlag = 16, kRouterXSentFlag = 32, kXReadLocalFlag = 96,
                   kXAckFlag = 128, kWAckFlag = 144;
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

inline void define(KernelDescriptor& kernel_descriptor, const std::string& name, const std::string& value = "") {
    kernel_descriptor.defines.emplace_back(name, value);
}
inline std::string unsigned_literal(int64_t value) { return std::to_string(value) + "u"; }
inline std::string integer_literal(int64_t value) { return std::to_string(value); }

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

}  // namespace ttnn::operations::experimental::moe_block::moe_head
