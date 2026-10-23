// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/synthesize_output_shard_spec.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <mutex>
#include <numeric>

namespace ttnn::operations::unary {

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const tt::tt_metal::TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

bool is_uneven(const tt::tt_metal::TensorSpec& t) {
    if (!t.memory_config().is_sharded()) {
        return false;
    }
    // is_sharded() is true for ND_SHARDED and leaves shard_spec empty
    if (!get_shard_spec(t).has_value()) {
        return false;
    }
    const auto& shape = t.padded_shape();
    const auto& shard = get_shard_spec(t)->shape;
    const auto rank = shape.rank();
    TT_FATAL(rank >= 2, "Rank must be at least 2");
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

bool is_native_L1_sharding(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::MemoryConfig& output_memory_config) {
    if (!output_memory_config.is_sharded()) {
        return false;
    }
    if (!input_spec.memory_config().is_sharded()) {
        return false;
    }
    // ND_SHARDED config carries nd_shard_spec and not shard_spec
    if (!input_spec.memory_config().shard_spec().has_value() || !output_memory_config.shard_spec().has_value()) {
        return false;
    }
    if (is_uneven(input_spec)) {
        return false;
    }
    if (input_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM ||
        output_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM) {
        return false;
    }
    // The aliased path computes input shard k into whichever output shard lives on the same core, with
    // no data movement in between, so the two specs must place the same tensor region on each core.
    const auto& in_shard = *input_spec.memory_config().shard_spec();
    const auto& out_shard = *output_memory_config.shard_spec();
    return in_shard.grid == out_shard.grid && in_shard.shape == out_shard.shape &&
           in_shard.orientation == out_shard.orientation &&
           input_spec.memory_config().memory_layout() == output_memory_config.memory_layout();
}

namespace {

// Compute time per tile, measured with data in L1, against a core's DRAM time per tile. An op that computes for longer
// is bound by its SFPU work, and deeper reads cannot speed it up. Ordered by cost: a chain takes its costliest op's
// class, so one unmeasured op makes the whole chain unmeasured.
enum class OpCost : uint8_t {
    Light,       // well below the DRAM time per tile
    Moderate,    // below it, but close to it on a slower SFPU
    Mid,         // close to it
    Heavy,       // above it
    VeryHeavy,   // far above it: the SFPU sets the time whatever the reads do
    Unmeasured,  // keeps the default path
};

// Whether parameter index of op selects its fast variant.
bool is_fast_variant(const EltwiseUnaryWithParam& op, size_t index) {
    return op.get_param_if<float>(index).value_or(0.0f) == 1.0f;
}

OpCost op_cost(const EltwiseUnaryWithParam& op) {
    switch (op.type()) {
        case UnaryOpType::IDENTITY:
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::LEAKY_RELU:
        case UnaryOpType::THRESHOLD:
        case UnaryOpType::HARDTANH:
        case UnaryOpType::HARDSHRINK:
        case UnaryOpType::SOFTSHRINK:
        case UnaryOpType::HARDSIGMOID:
        case UnaryOpType::HARDSWISH:
        case UnaryOpType::HARDMISH:
        case UnaryOpType::SOFTSIGN:
        case UnaryOpType::SOFTCAP:
        case UnaryOpType::TANH:
        case UnaryOpType::ABS:
        case UnaryOpType::NEG:
        case UnaryOpType::SIGN:
        case UnaryOpType::SQUARE:
        case UnaryOpType::RECIP:
        case UnaryOpType::EXP:
        case UnaryOpType::LOG:
        case UnaryOpType::SIN:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU: return OpCost::Light;
        case UnaryOpType::COS:
        case UnaryOpType::RSQRT: return OpCost::Moderate;
        case UnaryOpType::SILU:
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::TANHSHRINK: return OpCost::Mid;
        case UnaryOpType::ELU:
        case UnaryOpType::CELU:
        case UnaryOpType::SELU:
        case UnaryOpType::MISH:
        case UnaryOpType::GELU_TANH:
        case UnaryOpType::LOGSIGMOID: return OpCost::Heavy;
        case UnaryOpType::LOGIT:
        case UnaryOpType::XIELU: return OpCost::VeryHeavy;
        // Parameter 1 of SIGMOID and parameter 0 of GELU, ERF and SQRT select the fast variant.
        case UnaryOpType::SQRT: return is_fast_variant(op, 0) ? OpCost::Light : OpCost::Moderate;
        case UnaryOpType::SIGMOID: return is_fast_variant(op, 1) ? OpCost::Light : OpCost::Mid;
        case UnaryOpType::GELU: return is_fast_variant(op, 0) ? OpCost::Light : OpCost::VeryHeavy;
        case UnaryOpType::ERF: return is_fast_variant(op, 0) ? OpCost::Mid : OpCost::Heavy;
        default: return OpCost::Unmeasured;
    }
}

// A chain costs as much as its most expensive op, and at least as much as a moderate one: light ops add up.
OpCost chain_cost(const std::vector<EltwiseUnaryWithParam>& op_chain) {
    OpCost cost = op_chain.size() > 1 ? OpCost::Moderate : OpCost::Light;
    for (const auto& op : op_chain) {
        cost = std::max(cost, op_cost(op));
    }
    return cost;
}

// Ops whose 4-byte kernels compute for longer than a 4 KB page takes to move.
bool slow_on_large_pages(const EltwiseUnaryWithParam& op) {
    switch (op.type()) {
        case UnaryOpType::ERF:
        case UnaryOpType::TANHSHRINK:
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::LOGSIGMOID: return true;
        // Parameter 0 of MISH selects the fast variant.
        case UnaryOpType::MISH: return !is_fast_variant(op, 0);
        default: return false;
    }
}

// With 4 KB pages a tile takes twice as long to move: single ops up to heavy and chains of light ops count as light,
// except the slow ones, and the rest as very heavy.
OpCost large_page_cost(OpCost cost, const std::vector<EltwiseUnaryWithParam>& op_chain) {
    const bool fits = cost <= (op_chain.size() == 1 ? OpCost::Heavy : OpCost::Moderate) &&
                      std::none_of(op_chain.begin(), op_chain.end(), slow_on_large_pages);
    return fits ? OpCost::Light : OpCost::VeryHeavy;
}

// What the flow choice needs to know about the tensor.
struct TensorReads {
    bool interleaved;
    bool width_shards;
    bool large_pages;  // above 2 KB (fp32, int32): twice the DRAM time per tile
    uint32_t banks;    // banks the tensor uses: all of them if interleaved, else one per shard
    uint64_t pages_per_bank;
    bool all_banks;  // banks >= the chip's DRAM banks
    uint32_t shard_width;
};

// Ops that keep the default path on many cores per DRAM bank, whatever their cost class. With 4 KB pages xielu's
// kernel is bound by its SFPU work, and no flow was faster. With 2 KB pages the default path's time for softplus swings
// with the shape, and the flows were slower at some sizes.
bool keeps_default_path(const std::vector<EltwiseUnaryWithParam>& op_chain, bool large_pages) {
    const UnaryOpType slow_op = large_pages ? UnaryOpType::XIELU : UnaryOpType::SOFTPLUS;
    return std::any_of(op_chain.begin(), op_chain.end(), [&](const auto& op) { return op.type() == slow_op; });
}

// Banks the default path's cores start reading on; in step, they keep reading those banks together. Its split gives
// each core a contiguous range of pages, the first total % cores ranges one page more.
//  - Interleaved: where all ranges have the same length, or more than half are one page longer, they start
//    ceil(pages / cores) apart. Elsewhere the starts spread differently, which was not measured, so they count as every
//    bank.
//  - Width shards: the shards the ranges start in.
uint32_t default_start_banks(
    const TensorReads& t, const DramReadPlan& plan, uint64_t total_pages, uint32_t num_cores, uint32_t num_dram_banks) {
    if (t.interleaved) {
        const uint64_t longer_ranges = total_pages % num_cores;
        if (longer_ranges != 0 && 2 * longer_ranges <= num_cores) {
            return t.banks;
        }
        return static_cast<uint32_t>(
            num_dram_banks / std::gcd(tt::div_up(total_pages, uint64_t{num_cores}), uint64_t{num_dram_banks}));
    }
    if (!t.width_shards) {
        return t.banks;
    }
    std::vector<bool> started(plan.num_shards);
    const uint64_t pages_per_core = total_pages / num_cores;
    const uint64_t cores_with_more = total_pages % num_cores;
    uint64_t first_page = 0;
    for (uint32_t core = 0; core < num_cores && first_page < total_pages; ++core) {
        started[(first_page % plan.row_pages) / plan.shard_width] = true;
        first_page += pages_per_core + (core < cores_with_more ? 1 : 0);
    }
    return static_cast<uint32_t>(std::count(started.begin(), started.end(), true));
}

// Pages above 2 KB (fp32, int32) on many cores per DRAM bank. The smallest measured share was 65 pages per core.
DramReadFlow choose_flow_large_pages(OpCost cost, const TensorReads& t, uint64_t pages_per_core) {
    constexpr uint64_t kMinPagesPerCore = 64;
    // Below this the work queue has too little to balance: 128 pages is 8 chunks of the minimum size.
    constexpr uint64_t kMinPagesPerCoreForQueue = 128;
    // Below this, heavy ops read one page at a time, so their first tiles arrive early; above it the SFPU sets their
    // time.
    constexpr uint64_t kMaxPagesPerCoreForOnePage = 512;
    // With fewer pages the default path's time for light ops is its launch time, and bursts do not shorten it.
    constexpr uint64_t kMinPagesPerCoreForLightBursts = 96;
    const bool small = pages_per_core < kMinPagesPerCoreForQueue;
    // On small width shards the default order already crosses banks, and on small interleaved tensors it already
    // keeps up: deeper reads only delay the first tiles.
    if (pages_per_core < kMinPagesPerCore || (small && (t.width_shards || t.interleaved))) {
        return DramReadFlow::None;
    }
    const DramReadFlow deep = small ? DramReadFlow::StaticBurst : DramReadFlow::WorkQueue;
    if (cost <= OpCost::Moderate) {
        return pages_per_core >= kMinPagesPerCoreForLightBursts ? deep : DramReadFlow::None;
    }
    // With every bank serving the tensor, DRAM keeps up with mid and heavy ops, so the flows only add overhead.
    if (t.all_banks) {
        return DramReadFlow::None;
    }
    if (cost == OpCost::Mid) {
        return deep;
    }
    return pages_per_core < kMaxPagesPerCoreForOnePage ? DramReadFlow::StaticOnePage : deep;
}

// Flow for many cores per DRAM bank. The default split queues cores on one bank while others idle, so the flows are
// faster up to ops close to the DRAM time per tile. Heavy ops keep the default path: its own time for them swings with
// the shape, so no flow was faster at every size.
DramReadFlow choose_flow_many_cores_per_bank(OpCost cost, const TensorReads& t, uint64_t pages_per_core) {
    if (t.large_pages) {
        return choose_flow_large_pages(cost, t, pages_per_core);
    }
    if (cost >= OpCost::Heavy) {
        return DramReadFlow::None;
    }
    // Smallest share from which the flows were faster for every measured op of the group at every measured size from
    // 16 to 196 pages per core. With every bank serving the tensor DRAM keeps up with mid ops, so they keep the default
    // path there. Shards on every bank were measured only at 96 pages per core.
    struct MinPagesPerCore {
        uint64_t light_or_moderate;
        uint64_t mid;
    };
    constexpr uint64_t kNever = std::numeric_limits<uint64_t>::max();
    constexpr MinPagesPerCore kInterleaved{65, kNever};
    constexpr MinPagesPerCore kShardsOnAllBanks{96, kNever};
    constexpr MinPagesPerCore kHeightShards{40, 32};
    constexpr MinPagesPerCore kWidthShards{73, 32};
    const MinPagesPerCore& min = t.interleaved    ? kInterleaved
                                 : t.all_banks    ? kShardsOnAllBanks
                                 : t.width_shards ? kWidthShards
                                                  : kHeightShards;
    if (pages_per_core < (cost == OpCost::Mid ? min.mid : min.light_or_moderate)) {
        return DramReadFlow::None;
    }
    // On every bank, bursts until the queue has enough pages to balance: 128 pages is 8 chunks of the minimum size.
    constexpr uint64_t kMinPagesPerCoreForQueueOnAllBanks = 128;
    return t.all_banks && pages_per_core < kMinPagesPerCoreForQueueOnAllBanks ? DramReadFlow::StaticBurst
                                                                              : DramReadFlow::WorkQueue;
}

// Flow for few cores per DRAM bank. The default split leaves fewer cores idle there, so the flows are faster only where
// DRAM clearly sets the time. start_banks: see default_start_banks.
DramReadFlow choose_flow_few_cores_per_bank(
    OpCost cost, const TensorReads& t, uint64_t pages_per_core, uint32_t start_banks) {
    // Bursts get a height shard's first tiles in sooner. With smaller shares the first burst is most of a core's
    // share, and with fewer 2 KB pages per bank the default path's time is its launch time anyway.
    constexpr uint64_t kMinPagesPerCoreForBursts = 24;
    constexpr uint64_t kMinPagesPerCoreForBurstsLargePages = 32;
    constexpr uint64_t kMinPagesPerBankForBursts = 384;
    // From this on the queue saves more than its start-up and last chunk cost.
    constexpr uint64_t kMinPagesPerCoreForQueue = 96;
    // Where the default path's cores start on between these many banks and have at least this share, they pile onto
    // those banks, and mid ops on interleaved tensors are faster with the queue.
    constexpr uint32_t kMinStartBanks = 3;
    constexpr uint32_t kMaxStartBanks = 4;
    constexpr uint64_t kMinPagesPerCoreForMidQueue = 448;
    // Smallest share where the queue was faster on width shards whose cores start in few shards; with large pages the
    // slower ops were slower with it at 48.
    constexpr uint64_t kMinPagesPerCoreForPiledQueue = 48;
    constexpr uint64_t kMinPagesPerCoreForPiledQueueLargePages = 64;
    // Narrower shards move the default path's cores on to the next bank sooner, and heavier ops were slower there.
    constexpr uint32_t kMinShardWidthForPiledQueue = 32;
    // Width shards where the cores start in at most two thirds of the shards: in step, they pile onto those banks. The
    // fewer the banks, the heavier the ops the queue is faster for.
    if (t.width_shards && t.shard_width >= kMinShardWidthForPiledQueue &&
        pages_per_core >= (t.large_pages ? kMinPagesPerCoreForPiledQueueLargePages : kMinPagesPerCoreForPiledQueue)) {
        const OpCost max_cost = 3 * start_banks <= t.banks       ? OpCost::VeryHeavy
                                : 2 * start_banks <= t.banks     ? OpCost::Heavy
                                : 3 * start_banks <= 2 * t.banks ? OpCost::Mid
                                                                 : OpCost::Light;
        if (max_cost > OpCost::Light && cost <= max_cost) {
            return DramReadFlow::WorkQueue;
        }
    }
    // Interleaved pages already cross the banks: light ops are faster with the queue, mid ops only where the cores pile
    // onto a few banks.
    if (t.interleaved) {
        const bool few_start_banks = start_banks >= kMinStartBanks && start_banks <= kMaxStartBanks &&
                                     pages_per_core >= kMinPagesPerCoreForMidQueue;
        const bool queue = cost == OpCost::Light || (cost == OpCost::Mid && few_start_banks);
        return queue && pages_per_core >= kMinPagesPerCoreForQueue ? DramReadFlow::WorkQueue : DramReadFlow::None;
    }
    // No bursts on width shards: the default order already crosses the banks.
    const bool bursts_pay = !t.width_shards && (t.large_pages ? pages_per_core >= kMinPagesPerCoreForBurstsLargePages
                                                              : pages_per_core >= kMinPagesPerCoreForBursts &&
                                                                    t.pages_per_bank >= kMinPagesPerBankForBursts);
    const DramReadFlow deep = pages_per_core >= kMinPagesPerCoreForQueue ? DramReadFlow::WorkQueue
                              : bursts_pay                               ? DramReadFlow::StaticBurst
                                                                         : DramReadFlow::None;
    // On fewer shards than banks the default split queues more cores on each bank, so the flows are faster up to heavy
    // ops, and for every op with large pages.
    if (!t.all_banks) {
        return cost <= OpCost::Heavy || t.large_pages ? deep : DramReadFlow::None;
    }
    // With every bank serving, the default reads keep up with moderate ops.
    return cost == OpCost::Light ? deep : DramReadFlow::None;
}

// The flows' page order for the tensor, see dram_read::RotatedPages; nullopt for layouts they do not read.
std::optional<DramReadPlan> page_order(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::TensorSpec& output_spec, uint64_t total_pages) {
    const auto& tile = input_spec.tile();
    const uint32_t row_pages = input_spec.padded_shape()[-1] / tile.get_width();
    const bool interleaved = !input_spec.memory_config().is_sharded();
    uint32_t shard_height = 0;
    uint32_t shard_width = row_pages;
    if (!interleaved) {
        const auto& shard = input_spec.memory_config().shard_spec()->shape;
        if (shard != output_spec.memory_config().shard_spec()->shape || shard[0] == 0 ||
            shard[0] % tile.get_height() != 0 || shard[1] % tile.get_width() != 0) {
            return std::nullopt;
        }
        shard_height = shard[0] / tile.get_height();
        shard_width = shard[1] / tile.get_width();
    }
    if (shard_width == 0 || row_pages % shard_width != 0) {
        return std::nullopt;
    }
    DramReadPlan plan{.shard_width = shard_width, .row_pages = row_pages};
    if (interleaved) {
        // One shard of every page: the order is the page order.
        plan.shard_stride = static_cast<uint32_t>(total_pages);
        plan.num_shards = 1;
        plan.last_shard_pages = static_cast<uint32_t>(total_pages);
    } else if (shard_width == row_pages) {
        // Height shards: whole rows, the last may be short.
        const uint32_t shard_pages = shard_height * shard_width;
        plan.shard_stride = shard_pages;
        plan.num_shards = static_cast<uint32_t>(tt::div_up(total_pages, shard_pages));
        plan.last_shard_pages = static_cast<uint32_t>(total_pages - (uint64_t{plan.num_shards} - 1) * shard_pages);
    } else if (uint64_t{shard_height} * row_pages == total_pages) {
        // Width shards: a column band of every row.
        plan.shard_stride = shard_width;
        plan.num_shards = row_pages / shard_width;
        plan.last_shard_pages = shard_height * shard_width;
    } else {
        return std::nullopt;  // block shards
    }
    return plan;
}

// Pages per work-queue chunk: 1/150 of a core's share, as smaller chunks load the scheduler and larger ones leave a
// tail.
uint32_t work_queue_chunk_pages(uint64_t pages_per_core, uint32_t banks, bool many_cores_per_bank) {
    constexpr uint64_t kChunksPerCore = 150;
    // A chunk must take longer to read than a request round trip to the scheduler, or the reader waits for it.
    constexpr uint64_t kMinChunkPages = 16;
    constexpr uint64_t kMaxChunkPages = 256;
    // With few cores per bank, larger chunks were slower at the largest sizes.
    constexpr uint64_t kMaxChunkPagesFewCoresPerBank = 48;
    uint64_t chunk_pages = std::clamp(
        (pages_per_core + kChunksPerCore / 2) / kChunksPerCore,
        kMinChunkPages,
        many_cores_per_bank ? kMaxChunkPages : kMaxChunkPagesFewCoresPerBank);
    // Consecutive pages of the read order come from consecutive banks, so a chunk size that is a multiple of the banks
    // starts every chunk on the same one, and the workers, in step, pile onto it.
    if (!many_cores_per_bank && chunk_pages % banks == 0) {
        ++chunk_pages;
    }
    return static_cast<uint32_t>(chunk_pages);
}

}  // namespace

DramReadPlan get_dram_read_plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    uint32_t num_cores,
    uint32_t num_device_cores,
    uint32_t num_dram_banks) {
    // From this many cores per DRAM bank the default split queues cores on a bank while others idle; with fewer it is
    // closer to balanced and a stricter rule applies.
    constexpr uint32_t kManyCoresPerDramBank = 8;
    // Smallest shares where a flow was faster than the default path: 32 pages per core with many cores per bank (mid
    // ops on shards), and the stricter rule's shortest flow starts at 24.
    constexpr uint64_t kMinPagesPerCoreManyCoresPerBank = 32;
    constexpr uint64_t kMinPagesPerCoreFewCoresPerBank = 24;
    // Smaller pages (bfp8, bfp4) leave most ops compute-bound, and the flows were not faster for them.
    constexpr uint32_t kMinPageBytes = 2048;

    // This runs for every unary op on every call, so the cheapest checks come first.
    if (op_chain.empty() || num_cores == 0 || num_dram_banks == 0) {
        return {};
    }
    const bool many_cores_per_bank = num_cores >= kManyCoresPerDramBank * num_dram_banks;
    // The stricter rule was measured on the whole grid; on part of it each core has more DRAM bandwidth.
    if (!many_cores_per_bank && num_cores != num_device_cores) {
        return {};
    }
    auto is_dram_tiled = [](const tt::tt_metal::TensorSpec& s) {
        const auto& mc = s.memory_config();
        return s.layout() == tt::tt_metal::Layout::TILE && mc.is_dram() &&
               (mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED || mc.shard_spec().has_value());
    };
    // The page order comes from the input, and the kernels move the output's pages: the two must match page for page.
    if (!is_dram_tiled(input_spec) || !is_dram_tiled(output_spec) ||
        input_spec.memory_config().memory_layout() != output_spec.memory_config().memory_layout() ||
        input_spec.padded_shape() != output_spec.padded_shape() || input_spec.tile() != output_spec.tile()) {
        return {};
    }
    const uint64_t total_pages = input_spec.padded_shape().volume() / input_spec.tile().get_tile_hw();
    const uint64_t pages_per_core = total_pages / num_cores;
    if (pages_per_core < (many_cores_per_bank ? kMinPagesPerCoreManyCoresPerBank : kMinPagesPerCoreFewCoresPerBank) ||
        total_pages > std::numeric_limits<uint32_t>::max() ||  // the kernels take 32-bit page ids
        input_spec.compute_page_size_bytes() < kMinPageBytes || output_spec.compute_page_size_bytes() < kMinPageBytes) {
        return {};
    }
    // Ops with no measured cost keep the default path; the choose_flow_* functions see measured costs only.
    const OpCost cost = chain_cost(op_chain);
    if (cost == OpCost::Unmeasured) {
        return {};
    }
    const auto order = page_order(input_spec, output_spec, total_pages);
    if (!order.has_value()) {
        return {};
    }

    DramReadPlan plan = *order;
    const bool interleaved = !input_spec.memory_config().is_sharded();
    const uint32_t banks = interleaved ? num_dram_banks : plan.num_shards;
    const TensorReads reads{
        .interleaved = interleaved,
        .width_shards = !interleaved && plan.shard_width != plan.row_pages,
        .large_pages = input_spec.compute_page_size_bytes() > kMinPageBytes,
        .banks = banks,
        .pages_per_bank = total_pages / banks,
        .all_banks = banks >= num_dram_banks,
        .shard_width = plan.shard_width};
    if (many_cores_per_bank) {
        plan.flow = keeps_default_path(op_chain, reads.large_pages)
                        ? DramReadFlow::None
                        : choose_flow_many_cores_per_bank(cost, reads, pages_per_core);
    } else {
        plan.flow = choose_flow_few_cores_per_bank(
            reads.large_pages ? large_page_cost(cost, op_chain) : cost,
            reads,
            pages_per_core,
            default_start_banks(reads, plan, total_pages, num_cores, num_dram_banks));
    }
    // Some compute kernels take a fixed tile count and cannot run the queue. With many cores per bank bursts stand in
    // for it: fp32 logit and logsigmoid on shards over part of the banks took up to 34% less time with them than with
    // the default path. With few cores per bank bursts are slower than the default path where the queue is faster.
    if (plan.flow == DramReadFlow::WorkQueue &&
        !utils::compute_kernel_supports_work_queue(op_chain[0].type(), input_spec.data_type())) {
        plan.flow = many_cores_per_bank ? DramReadFlow::StaticBurst : DramReadFlow::None;
    }
    if (plan.flow == DramReadFlow::None) {
        return {};
    }
    if (plan.flow == DramReadFlow::WorkQueue) {
        plan.chunk_pages = work_queue_chunk_pages(pages_per_core, banks, many_cores_per_bank);
    }
    return plan;
}

DramReadPlan get_dram_read_plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const Tensor& input,
    const tt::tt_metal::TensorSpec& output_spec,
    const tt::tt_metal::CoreRangeSet& worker_grid) {
    const auto* device = input.device();
    // The work queue relies on how Wormhole and Blackhole dataflow buffers map to L1; the flows were measured there.
    if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
        return {};
    }
    const auto grid = device->compute_with_storage_grid_size();
    return get_dram_read_plan(
        op_chain,
        input.tensor_spec(),
        output_spec,
        worker_grid.num_cores(),
        grid.x * grid.y,
        device->num_dram_channels());
}

std::optional<UnaryShardSpecs> get_shard_specs(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::TensorSpec& output_spec) {
    const bool input_sharded = input_spec.memory_config().is_sharded();
    const bool output_sharded = output_spec.memory_config().is_sharded();

    if (!input_sharded && !output_sharded) {
        return std::nullopt;
    }

    if (!is_native_L1_sharding(input_spec, output_spec.memory_config()) || is_uneven(output_spec)) {
        return std::nullopt;
    }

    // For ROW_MAJOR layout, shard element count must be a multiple of tile_hw
    // for the sharded borrowed-DFB path to work (it requires whole-tile pages).
    // Fall back to the interleaved path otherwise.
    // Each shard row is its own page, padded up to the L1 alignment, so the dense tile pages the
    // borrowed DFB walks only line up with the buffer when that padding is zero.
    if (input_spec.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        auto fits_borrowed_dfb = [](const tt::tt_metal::TensorSpec& spec) {
            const auto& shard = *get_shard_spec(spec);
            const auto tile_hw = spec.tile().get_tile_hw();
            const uint64_t shard_elements = static_cast<uint64_t>(shard.shape[0]) * shard.shape[1];
            return shard_elements % tile_hw == 0 &&
                   spec.compute_page_size_bytes() % tt::tt_metal::hal::get_l1_alignment() == 0;
        };

        if ((input_sharded && !fits_borrowed_dfb(input_spec)) || (output_sharded && !fits_borrowed_dfb(output_spec))) {
            static std::once_flag warn_flag;
            std::call_once(warn_flag, [] {
                log_warning(
                    tt::LogOp,
                    "Unary: ROW_MAJOR shard element count is not tile-aligned or its rows are not L1-aligned; "
                    "falling back to interleaved path");
            });
            return std::nullopt;
        }
    }

    TT_FATAL(get_shard_spec(output_spec).has_value(), "Output must have shard spec when using native sharded path");

    const auto& out_shape = output_spec.padded_shape();
    const auto& in_shape = input_spec.padded_shape();

    if (input_sharded) {
        const auto& in_shard = *get_shard_spec(input_spec);
        const auto& out_shard = *get_shard_spec(output_spec);
        return UnaryShardSpecs{.input_shard_spec = in_shard, .output_shard_spec = out_shard};
    }

    const auto& out_shard = *get_shard_spec(output_spec);
    auto adjusted_in = adjust_to_shape(out_shard, out_shape, in_shape);
    return UnaryShardSpecs{.input_shard_spec = adjusted_in, .output_shard_spec = out_shard};
}

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    auto ret = shard_spec;
    uint32_t from_volume_except_width = 1;
    uint32_t to_volume_except_width = 1;
    const auto from_rank = static_cast<int>(from_shape.rank());
    const auto to_rank = static_cast<int>(to_shape.rank());
    for (int i = 0; i < from_rank - 1; ++i) {
        from_volume_except_width *= from_shape[i];
    }
    for (int i = 0; i < to_rank - 1; ++i) {
        to_volume_except_width *= to_shape[i];
    }
    uint32_t from_width = from_shape[-1];
    uint32_t to_width = to_shape[-1];
    TT_FATAL(from_volume_except_width > 0, "Invalid from_shape: volume is zero");
    TT_FATAL(from_width > 0, "Invalid from_shape: width dimension is zero");
    ret.shape[0] = std::max((ret.shape[0] * to_volume_except_width) / from_volume_except_width, 32u);
    ret.shape[1] = std::max((ret.shape[1] * to_width) / from_width, 32u);
    return ret;
}

namespace {

// Small DRAM tensors, interleaved or sharded: each core of the grid adds host and dispatch time to a launch, so a
// tensor with few pages runs on part of the grid. Per cost class, the pages per core below are the most at which part
// of the grid still matched the whole grid's device time in trace replay. Never fewer than 8 cores, as fewer were not
// faster, and a power of two, so few distinct grids and cached programs arise. L1 tensors keep their grid.
tt::tt_metal::CoreRangeSet small_tensor_grid(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const tt::tt_metal::CoreRangeSet& workers) {
    constexpr uint32_t kMinCores = 8;
    const bool dram = input_tensor.memory_config().is_dram() &&
                      (!memory_config.has_value() || memory_config->is_dram()) &&
                      (!output_tensor.has_value() || output_tensor->memory_config().is_dram());
    if (!dram || input_tensor.layout() != tt::tt_metal::Layout::TILE) {
        return workers;
    }
    // On more cores than this, fewer pages per core keep pace with the whole grid, except light ops on height shards.
    constexpr uint32_t kMaxCoresAtFullLoad = 32;
    const OpCost cost = chain_cost(op_chain);
    const bool height_sharded =
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    const uint64_t pages_per_core = cost == OpCost::Light ? 4 : cost <= OpCost::Mid ? 2 : 1;
    const uint64_t pages_per_core_above_32_cores = cost == OpCost::Light ? (height_sharded ? 4 : 3) : 1;
    const uint64_t pages = input_tensor.physical_volume() / input_tensor.tensor_spec().tile().get_tile_hw();
    if (pages >= pages_per_core * workers.num_cores()) {
        return workers;
    }
    const auto cores_for = [&](uint64_t per_core) {
        return std::max(kMinCores, std::bit_ceil(static_cast<uint32_t>(tt::div_up(pages, per_core))));
    };
    auto cores = cores_for(pages_per_core);
    if (cores > kMaxCoresAtFullLoad) {
        cores = cores_for(pages_per_core_above_32_cores);
    }
    if (cores >= workers.num_cores()) {
        return workers;
    }
    log_debug(tt::LogOp, "Unary: Using {} worker cores for {} pages", cores, pages);
    return tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        workers.ranges().front().start_coord, cores, workers, true);  // row-wise
}

}  // namespace

tt::tt_metal::CoreRangeSet get_worker_grid(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        log_debug(tt::LogOp, "Unary: Using provided sub_core_grids for worker grid {}", sub_core_grids->str());
        return sub_core_grids.value();
    }

    // A sharded tensor's grid is in shard_spec or in nd_shard_spec when the config is ND_SHARDED
    auto shard_grid_of = [](const tt::tt_metal::MemoryConfig& mem_config) -> std::optional<tt::tt_metal::CoreRangeSet> {
        if (!mem_config.is_sharded()) {
            return std::nullopt;
        }
        if (mem_config.shard_spec().has_value()) {
            return mem_config.shard_spec()->grid;
        }
        if (mem_config.nd_shard_spec().has_value()) {
            return mem_config.nd_shard_spec()->grid;
        }
        return std::nullopt;
    };

    // All worker cores of the sub-device holding this grid, or nullopt if no sub-device holds it. Callers
    // fall through to the all-workers default on nullopt; the previous __builtin_unreachable() here made
    // that case UB instead. Returns by value: a pointer into worker_cores() would be valid, but the
    // grid is a small vector and copying it keeps the lifetime obvious.
    auto sub_device_workers_for =
        [](const tt::tt_metal::CoreRangeSet& grid,
           tt::tt_metal::distributed::MeshDevice* device) -> std::optional<tt::tt_metal::CoreRangeSet> {
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
        return std::nullopt;
    };

    // Each log sits inside the return so it never claims a grid the function then falls through.
    if (output_tensor.has_value()) {
        if (const auto grid = shard_grid_of(output_tensor->memory_config())) {
            if (const auto workers = sub_device_workers_for(*grid, output_tensor->device())) {
                log_debug(tt::LogOp, "Unary: Using output tensor grid for worker grid {}", grid->str());
                return small_tensor_grid(op_chain, input_tensor, output_tensor, memory_config, *workers);
            }
        }
    }

    if (memory_config.has_value()) {
        if (const auto grid = shard_grid_of(*memory_config)) {
            if (const auto workers = sub_device_workers_for(*grid, input_tensor.device())) {
                log_debug(tt::LogOp, "Unary: Using memory config shard spec grid for worker grid {}", grid->str());
                return small_tensor_grid(op_chain, input_tensor, output_tensor, memory_config, *workers);
            }
        }
    }

    // No branch on the input's own grid: native L1 sharding needs both sides on one grid, so the
    // memory_config branch above already returned it.
    auto* device = input_tensor.device();
    log_debug(tt::LogOp, "Unary: Using the worker cores of the device");
    return small_tensor_grid(
        op_chain,
        input_tensor,
        output_tensor,
        memory_config,
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front()));
}

tt::tt_metal::ShardSpec generate_output_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    uint32_t output_element_size_bytes) {
    // Unary: forced ROW_MAJOR — no input-orientation inheritance path (unlike Transpose/Repeat/Fold).
    // Output layout mirrors input (unary is dtype-only), so is_tile is derived from input layout.
    auto* device = input_tensor.device();
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    const bool is_tile = (input_tensor.layout() == tt::tt_metal::Layout::TILE);
    auto spec = ttnn::operations::data_movement::common::synthesize_output_shard_spec(
        compute_grid_size,
        padded_out_shape,
        memory_layout,
        {.is_tile = is_tile, .orientation_hint = tt::tt_metal::ShardOrientation::ROW_MAJOR, .caller_tag = "Unary"});
    // Shrink only when we produced an RM synth; typecast means the OUTPUT element size drives the page.
    if (!is_tile) {
        spec = *ttnn::operations::data_movement::common::shrink_shard_for_rm_page_alignment(
            spec,
            input_tensor.layout(),
            output_element_size_bytes,
            static_cast<uint64_t>(padded_out_shape[-1]),
            compute_grid_size,
            memory_layout,
            ttnn::operations::data_movement::common::RmPageAlignmentMode::Lenient);
    }
    log_debug(tt::LogOp, "Unary: Generated shard spec over {} populated cores", spec.grid.num_cores());
    return spec;
}

void require_buffer_distribution_matches_spec(
    const Tensor& tensor,
    const std::optional<tt::tt_metal::BufferDistributionSpec>& spec_distribution,
    const char* slot) {
    const auto& buffer_distribution = tensor.buffer()->buffer_distribution_spec();
    if (buffer_distribution.has_value() && spec_distribution.has_value() &&
        buffer_distribution->shard_shape_in_pages() == spec_distribution->shard_shape_in_pages() &&
        buffer_distribution->cores() == spec_distribution->cores()) {
        return;
    }
    const auto describe = [](const std::optional<tt::tt_metal::BufferDistributionSpec>& distribution) {
        if (!distribution.has_value()) {
            return std::string("no distribution");
        }
        std::string banks;
        for (const auto& core : distribution->cores()) {
            banks += (banks.empty() ? "" : ", ") + fmt::format("{}", core);
        }
        return fmt::format("shard shape in pages {}, banks [{}]", distribution->shard_shape_in_pages(), banks);
    };
    TT_THROW(
        "Unary: the sharded {} tensor's buffer distribution ({}) differs from its TensorSpec distribution ({}); "
        "Use a tensor whose buffer was allocated for its own spec.",
        slot,
        describe(buffer_distribution),
        describe(spec_distribution));
}

void require_buffer_distribution_matches_spec(const Tensor& tensor, const char* slot) {
    const auto sharding_args = tensor.tensor_spec().compute_buffer_sharding_args();
    require_buffer_distribution_matches_spec(tensor, sharding_args.buffer_distribution_spec(), slot);
}

}  // namespace ttnn::operations::unary
