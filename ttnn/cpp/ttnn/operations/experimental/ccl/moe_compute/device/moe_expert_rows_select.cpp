// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>

#include "kernels/moe_ring_common.h"
#include "moe_compute_device_operation.hpp"
#include "moe_expert_rows_plan.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

namespace ttnn::experimental::prim {

using namespace expert_rows_host;

namespace {

// The cost model. Cycles are AI clock cycles, at the clock under load (tt-smi): the live clock idles lower (800 MHz on
// Blackhole) and would make the choice depend on what ran before.
constexpr double kAiClockHzBlackhole = 1.35e9;
constexpr double kAiClockHzWormhole = 1.0e9;
// Peak DRAM bandwidth, decimal GB / s: Blackhole 512 (32 GB GDDR6 at 16 GT/s), Wormhole 12 x 21.5
// (ttnn/core/operation.cpp)
constexpr double kDramPeakBytesPerSecondBlackhole = 512e9;
constexpr double kDramPeakBytesPerSecondWormhole = 258e9;
// One 32 x 32 x 32 tile matmul per fidelity phase: 32768 multiply-adds at 2048 per cycle (8x16 x 16x16,
// tech_reports/matrix_engine/matrix_engine.md).
constexpr double kMatmulCyclesPerFidelityPhase = 16.0;
// Fitted to Blackhole p150b device profiles: the ring's pass of one job per ring position; writing one FP32 partial
// tile to L1 and reading it back into DEST, and one NoC command of the writer RISC (address and issue), both at the
// fastest row tiles per job.
constexpr double kRingHopCycles = 350.0;
constexpr double kReloadTileCycles = 64.0;
constexpr double kWriterOpCycles = 60.0;
// rows_combine.cpp sends a device's routed rows at these rates (Blackhole p150b). FullCcl: the placement program's span
// on device 0 over its rows' bytes (1x8 ring, Qwen3.6-like T 32 and 64 and DeepSeek-V3 T 8 per device). SingleDevice,
// NoC writes only: its call time minus ComputeOnly's for the same cell, over the rows' bytes, 180-320 GB/s at hidden
// size 6144-7168 (device 0 of 8-32, T 4-64).
constexpr double kFabricCombineBytesPerSecond = 23e9;
constexpr double kLocalCombineBytesPerSecond = 200e9;
constexpr double kCostMargin = 0.10;           // the margin the expert rows path must predict over the ring
constexpr uint64_t kRingSmallCbBytes = 4096;   // the ring tilize cores' count / flag CBs
constexpr uint64_t kRingTokensPerChunk = 32;   // the ring tilize cores' input chunk
constexpr uint64_t kRingFitMargin = 32768;     // L1 the ring-fit estimate leaves for what it does not model
constexpr uint32_t kCombinePacketHeaders = 3;  // combine_fabric.hpp: row unicasts, the barrier's two directions

double ai_clock_hz(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE ? kAiClockHzBlackhole : kAiClockHzWormhole; }

double dram_peak_bytes_per_s(tt::ARCH arch) {
    return arch == tt::ARCH::BLACKHOLE ? kDramPeakBytesPerSecondBlackhole : kDramPeakBytesPerSecondWormhole;
}

double matmul_tile_cycles(tt::tt_metal::MathFidelity fidelity) {
    return kMatmulCyclesPerFidelityPhase * tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(fidelity);
}

// Expected jobs (at most job_rows rows of one expert) of one call: local_experts experts, each routed by each token
// with probability top_k / global_experts (uniform routing).
double expected_jobs(const MoEExpertRowsShape& shape, uint32_t job_rows) {
    const double route_probability = std::min(1.0, static_cast<double>(shape.top_k) / shape.global_experts);
    const uint32_t tokens = shape.tokens;
    if (route_probability >= 1.0) {
        return shape.local_experts * std::ceil(tokens / static_cast<double>(job_rows));
    }
    // sum over j >= 1 of P(rows > job_rows (j - 1)), rows ~ Binomial(tokens, route_probability)
    double pmf = std::pow(1.0 - route_probability, tokens), cdf = 0.0, jobs = 0.0;
    for (uint32_t row_count = 0; row_count <= tokens; ++row_count) {
        cdf += pmf;
        if (row_count % job_rows == 0) {
            jobs += std::max(0.0, 1.0 - cdf);  // P(rows > row_count)
        }
        pmf *= (static_cast<double>(tokens - row_count) / (row_count + 1)) *
               (route_probability / (1.0 - route_probability));
    }
    return shape.local_experts * jobs;
}

// The cost model's terms of one plan, per job: the stream time of its weights at the DRAM peak (the busiest bank's
// share of the group's weights x banks); per row tile, the busiest core's matmul tiles and the partial tiles it writes
// and reads back between chunks, for a job of several row tiles and for a one-row-tile job of the same program (x in
// chunks of single_chunk_tiles hidden tiles, W2 whole).
struct JobCost {
    double stream_s = 0, busiest = 0, reloads = 0, single_reloads = 0;
};

JobCost job_cost(const MoEExpertRowsShape& shape, const ExpertRowsPlan& plan, uint32_t num_shards, tt::ARCH arch) {
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const uint32_t w0_w1_k_rows = hidden_tiles + shape.bias_rows(), w2_k_rows = intermediate_tiles + shape.bias_rows();
    const auto layouts = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    JobCost per_job;
    // tiles each bank (shard) streams for one job: group 0's W0/W1 runs split at the bank pieces, W2 in its own bank
    std::vector<double> bank_tiles(num_shards, 0.0);
    for (const auto& member : plan.cores) {
        // weight tiles per K row of the core's column groups and of its output groups
        const double gate_up_width = member.column_run().tiles();
        const double down_width = member.output_run().tiles();
        const double tiles = gate_up_width * w0_w1_k_rows + down_width * w2_k_rows;
        if (tiles > per_job.busiest) {
            per_job.busiest = tiles;
            if (plan.row_tiles > 1) {
                const uint32_t single_chunk =
                    expert_rows::single_chunk_tiles(plan.chunk_tiles, plan.row_tiles, hidden_tiles);
                per_job.reloads = gate_up_width * (tt::div_up(hidden_tiles, plan.chunk_tiles) - 1) +
                                  down_width * (tt::div_up(intermediate_tiles, plan.w2_chunk_rows) - 1);
                per_job.single_reloads = gate_up_width * (tt::div_up(hidden_tiles, single_chunk) - 1);
            }
        }
        if (member.expert_group == 0) {
            const ShardLayout& shard = layouts[member.ring_position];
            for (uint32_t column_group = member.first_column_group;
                 column_group < member.first_column_group + member.column_groups;
                 ++column_group) {
                uint32_t stream_tile = shard.w0_w1_first_tile + column_group * prepared.w0_w1_group_tiles;
                for (uint32_t left = shard.column_groups().width(column_group) * w0_w1_k_rows; left > 0;) {
                    const uint32_t piece = stream_tile / prepared.w0_w1_piece_tiles;
                    const uint32_t take = std::min(left, (piece + 1) * prepared.w0_w1_piece_tiles - stream_tile);
                    bank_tiles[piece] += take;
                    stream_tile += take;
                    left -= take;
                }
            }
            bank_tiles[member.ring_position] += down_width * w2_k_rows;
        }
    }
    const double busiest_bank = *std::max_element(bank_tiles.begin(), bank_tiles.end());
    per_job.stream_s = busiest_bank * num_shards * tt::tile_size(tt::DataFormat::Bfp4_b) / dram_peak_bytes_per_s(arch);
    return per_job;
}

// The cost model's prediction of one plan: its job terms and the seconds of the call.
struct PlanCost {
    double stream_s = 0, busiest = 0, reloads = 0, call_s = 0;
};

// The cost model of one call: tile_s seconds per tile matmul at the AI clock.
struct CostModel {
    const MoEExpertRowsShape& shape;
    uint32_t num_shards = 0;
    tt::ARCH arch;
    double tile_s = 0, clock_hz = 0;

    PlanCost cost(const ExpertRowsPlan& plan) const {
        const JobCost per_job = job_cost(shape, plan, num_shards, arch);
        PlanCost predicted{.stream_s = per_job.stream_s, .busiest = per_job.busiest, .reloads = per_job.reloads};
        const double compute = predicted.busiest * tile_s + predicted.reloads * kReloadTileCycles / clock_hz;
        predicted.call_s = std::max(
            expected_jobs(shape, expert_rows::tile_rows * plan.row_tiles) * predicted.stream_s,
            std::ceil(expected_jobs(shape, expert_rows::tile_rows) / plan.expert_groups) * compute);
        return predicted;
    }
};

struct CostedPlan {
    ExpertRowsPlan plan;
    PlanCost cost;
};

// TT_METAL_MOE_COMPUTE_KERNEL, a test and benchmark hook: =ring forces the ring; =expert_rows forces the expert rows
// path (over the cost model and the Wormhole gate; a call that path cannot run is refused with the reason).
enum class ForcedKernel { None, Ring, ExpertRows };
constexpr const char* kKernelEnv = "TT_METAL_MOE_COMPUTE_KERNEL";
constexpr const char* kRowTilesEnv = "TT_METAL_MOE_COMPUTE_ROW_TILES";
constexpr size_t kMaxRememberedChoices = 1024;  // call shapes whose choice select_moe_compute_expert_rows keeps

ForcedKernel forced_kernel() {
    const char* value = std::getenv(kKernelEnv);
    if (value == nullptr) {
        return ForcedKernel::None;
    }
    const std::string_view kernel(value);
    if (kernel == "ring") {
        return ForcedKernel::Ring;
    }
    TT_FATAL(
        kernel == "expert_rows", "moe_compute: TT_METAL_MOE_COMPUTE_KERNEL={}, expected ring or expert_rows", kernel);
    return ForcedKernel::ExpertRows;
}

// TT_METAL_MOE_COMPUTE_ROW_TILES=<row tiles>, a test and benchmark hook: jobs of that many row tiles, from 1 to
// max_row_tiles.
std::optional<uint32_t> forced_row_tiles(uint32_t max_row_tiles) {
    const char* value = std::getenv(kRowTilesEnv);
    if (value == nullptr) {
        return std::nullopt;
    }
    const std::string_view text(value);
    uint32_t row_tiles = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), row_tiles);
    TT_FATAL(
        error == std::errc() && end == text.data() + text.size() && row_tiles >= 1 && row_tiles <= max_row_tiles,
        "moe_compute: TT_METAL_MOE_COMPUTE_ROW_TILES={} outside 1..{} (the row tiles of the call's routed rows)",
        text,
        max_row_tiles);
    return row_tiles;
}

// The ring program for this call, logged with the reason; a call forced onto the expert rows path fails with it.
std::optional<MoEExpertRowsParams> ring_with_reason(ForcedKernel forced, const std::string& why) {
    TT_FATAL(
        forced != ForcedKernel::ExpertRows,
        "moe_compute: TT_METAL_MOE_COMPUTE_KERNEL=expert_rows, but the expert rows path cannot run: {}",
        why);
    log_debug(tt::LogOp, "moe_compute: ring kernel ({})", why);
    return std::nullopt;
}

// The first input layout rule of the expert rows path that the call breaks, with the observed value, or nullopt.
std::optional<std::string> input_layout_refusal(const MoEComputeInputs& tensor_args) {
    using NamedTensor = std::pair<std::string_view, const ttnn::Tensor*>;
    const std::array<NamedTensor, 3> routing_inputs = {{
        {"expert indices", &tensor_args.tilize_expert_indices_tensor},
        {"expert scores", &tensor_args.tilize_expert_scores_tensor},
        {"expert mapping", &tensor_args.tilize_expert_mapping_tensor},
    }};
    for (const auto& [name, tensor] : routing_inputs) {
        if (tensor->layout() != Layout::ROW_MAJOR) {
            return fmt::format("the {} tensor has layout {}, expected ROW_MAJOR", name, tensor->layout());
        }
    }
    // rows_reader.cpp reads the scores and the mapping as 16-bit values
    if (tensor_args.tilize_expert_scores_tensor.dtype() != DataType::BFLOAT16) {
        return fmt::format(
            "the expert scores are {}, expected BFLOAT16", tensor_args.tilize_expert_scores_tensor.dtype());
    }
    if (tensor_args.tilize_expert_mapping_tensor.dtype() != DataType::UINT16) {
        return fmt::format(
            "the expert mapping is {}, expected UINT16", tensor_args.tilize_expert_mapping_tensor.dtype());
    }
    const std::array<NamedTensor, 2> weights = {{
        {"W0/W1", &tensor_args.matmul_w0_w1_tensor},
        {"W2", &tensor_args.matmul_w2_tensor},
    }};
    for (const auto& [name, tensor] : weights) {
        if (tensor->dtype() != DataType::BFLOAT4_B) {
            return fmt::format("the {} weights are {}, expected BFLOAT4_B", name, tensor->dtype());
        }
        if (tensor->memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
            return fmt::format(
                "the {} weights are {}, expected HEIGHT_SHARDED", name, tensor->memory_config().memory_layout());
        }
    }
    return std::nullopt;
}

// The shape of a moe_compute call.
MoEExpertRowsShape expert_rows_shape(
    ttnn::MeshDevice& mesh_device, const MoEComputeParams& args, const MoEComputeInputs& tensor_args) {
    const auto& input = tensor_args.tilize_input_tensor;
    MoEExpertRowsShape shape;
    shape.hidden_size = input.logical_shape()[-1];
    shape.intermediate_size = args.intermediate_size;
    shape.local_experts = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    shape.top_k = tensor_args.tilize_expert_indices_tensor.logical_shape()[-1];
    shape.tokens = input.logical_shape()[0] * input.logical_shape()[1];
    shape.global_experts = tensor_args.tilize_expert_mapping_tensor.logical_shape()[-1];
    const auto& view = mesh_device.get_view();
    shape.num_sources = args.cluster_axis().value_or(1) == 1 ? view.num_cols() : view.num_rows();
    shape.index_page_bytes = aligned_page(tensor_args.tilize_expert_indices_tensor);
    shape.mapping_page_bytes = aligned_page(tensor_args.tilize_expert_mapping_tensor);
    shape.has_bias = args.has_bias;
    return shape;
}

// The CB budget per core: circular buffers end below the lowest live L1 buffer (and below L1_SMALL); bucketed, it is a
// program-cache key.
uint32_t cb_budget_of(ttnn::MeshDevice& mesh_device) {
    const auto& alloc = mesh_device.allocator();
    const uint64_t base = alloc->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t top = std::min<uint64_t>(
        mesh_device.lowest_occupied_compute_l1_address().value_or(std::numeric_limits<uint64_t>::max()),
        ttnn::ccl::l1_small_floor_address(mesh_device));
    constexpr uint32_t kBudgetBucket = 16 * 1024;
    return top > base ? static_cast<uint32_t>((top - base) / kBudgetBucket * kBudgetBucket) : 0;
}

// moe_compute's L1 outputs per core (per-bank upper bound); the placement program runs after they are allocated, so its
// CBs end below them.
uint64_t l1_outputs_per_core(ttnn::MeshDevice& mesh_device, const std::vector<tt::tt_metal::TensorSpec>& specs) {
    uint64_t bytes = 0;
    const uint32_t l1_banks = mesh_device.allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    for (uint32_t i = 0; i < specs.size(); ++i) {
        if (i != kMatmulOutput && specs[i].memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
            bytes += specs[i].compute_consumed_memory_bytes_per_bank(tt::tt_metal::hal::get_l1_alignment(), l1_banks);
        }
    }
    return bytes;
}

// The ring program's token-sized CBs on its tilize cores: the e_t lists, activation rows, the second RISC's halves of
// both, the mapping and an input chunk. When they do not fit below moe_compute's outputs, the ring program refuses the
// call and the caller would have to split it, so the expert rows path runs it.
uint64_t ring_tilize_cb_bytes(
    ttnn::MeshDevice* mesh_device,
    const MoEComputeParams& args,
    const MoEExpertRowsShape& shape,
    const std::vector<tt::tt_metal::TensorSpec>& specs,
    const MoEComputeInputs& tensor_args) {
    const auto ring_cores = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        args.num_token_parallel_cores,
        args.num_data_parallel_cores,
        shape.hidden_size,
        args.combine_params.has_value() ? args.combine_params->mux_core_range_set : CoreRangeSet{},
        args.bh_ring_size);
    const auto& map = tensor_args.tilize_expert_mapping_tensor;
    const uint64_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    const uint64_t tokens = shape.tokens, local_experts = shape.local_experts;
    const uint64_t activation_row_bytes = tt::align((2 * local_experts + 1) * kWordBytes, l1_align);
    const uint64_t tilize_cores = std::max<size_t>(1, ring_cores.tilize_cores.size());
    const uint64_t x_page = tensor_args.tilize_input_tensor.buffer()->aligned_page_size();
    return local_experts * specs[kETPages].compute_page_size_bytes()                      // e_t lists
           + tokens * activation_row_bytes                                                // activation rows
           + tt::div_up(tokens, 2) * l1_align * local_experts                             // second RISC's e_t
           + tt::div_up(tokens, 2) * activation_row_bytes                                 // second RISC's rows
           + map.buffer()->num_pages() * map.buffer()->aligned_page_size()                // mapping
           + kRingTokensPerChunk * tt::align(tt::div_up(x_page, tilize_cores), l1_align)  // input
           + kRingSmallCbBytes;
}

// Weight-stationary jobs: the most row tiles per job that keep the weight stream the longest part of a job while the
// jobs in flight (the expert groups, fewer when the call has fewer expected jobs) share DRAM: the busiest core's
// matmuls and the writer's NoC commands for the job's rows (x reads per chunk, tile build, row writes) both within one
// job's stream. The compute bound is rounded when the call has at least two expected jobs per group (the deal in
// rows_reader.cpp then balances the groups' work) and rounded down otherwise (a longer job holds up its group while the
// others have nothing left to do). Rows per expert are not known on the host; uniform routing rarely gives an expert
// more than 32 rows of a call, hot experts and longer calls do, so several row tiles are kept on a tie (they cost about
// the same when they merge nothing). Returns the chosen plan, or why a forced row-tile count cannot run.
std::variant<CostedPlan, std::string> choose_row_tiles(
    ttnn::MeshDevice* mesh_device,
    const CostModel& model,
    const std::vector<uint32_t>& shard_banks,
    uint32_t budget,
    tt::DataFormat partials,
    CostedPlan best) {
    const MoEExpertRowsShape& shape = model.shape;
    const std::optional<uint32_t> forced =
        forced_row_tiles(tt::div_up(shape.tokens * shape.top_k, expert_rows::tile_rows));
    const double jobs = expected_jobs(shape, expert_rows::tile_rows);
    const double compute_bound =
        std::min<double>(best.plan.expert_groups, jobs) * best.cost.stream_s / (best.cost.busiest * model.tile_s);
    const double compute_cap =
        std::round(jobs) >= 2.0 * best.plan.expert_groups ? std::round(compute_bound) : std::floor(compute_bound);
    uint32_t most = std::min<uint32_t>(compute_cap, tt::div_up(shape.tokens, expert_rows::tile_rows));
    uint32_t least = 2;
    if (forced.has_value()) {
        most = least = *forced;
    }
    // Wormhole's kernel config buffer does not hold the kernels of jobs of several row tiles.
    if (mesh_device->arch() == tt::ARCH::WORMHOLE_B0 && most > 1) {
        if (forced.has_value()) {
            return fmt::format("TT_METAL_MOE_COMPUTE_ROW_TILES={}: Wormhole runs one row tile per job", *forced);
        }
        most = 1;
    }
    for (uint32_t row_tiles = most; row_tiles >= least && row_tiles >= 2; --row_tiles) {
        auto candidate = plan_expert_rows(mesh_device, shape, shard_banks, budget, row_tiles, partials);
        auto* plan = std::get_if<ExpertRowsPlan>(&candidate);
        if (plan == nullptr) {
            continue;
        }
        const PlanCost cost = model.cost(*plan);
        uint32_t most_output_groups = 0;
        for (const auto& member : plan->cores) {
            most_output_groups = std::max(most_output_groups, member.output_groups);
        }
        // the writer's NoC commands per row of a job: one x read per chunk, two face-row reads per tile of its x
        // piece, and per output group two face-row reads per tile and one row write
        const double x_reads = tt::div_up(shape.hidden_tiles(), plan->chunk_tiles);
        const double piece_reads = 2.0 * tt::div_up(shape.hidden_tiles(), plan->group_cores);
        const double output_commands = (2.0 * kGroupTiles + 1) * most_output_groups;
        const double row_ops = x_reads + piece_reads + output_commands;
        const double writer_bound =
            plan->expert_groups * cost.stream_s / (expert_rows::tile_rows * row_ops * kWriterOpCycles / model.clock_hz);
        if (!forced.has_value() && (row_tiles > std::lround(writer_bound) || cost.call_s > best.cost.call_s)) {
            continue;
        }
        best = {std::move(*plan), cost};
        break;
    }
    if (forced.has_value() && best.plan.row_tiles != most) {
        return fmt::format("TT_METAL_MOE_COMPUTE_ROW_TILES={}: no plan fits", *forced);
    }
    return best;
}

}  // namespace

std::optional<std::string> expert_rows_host::shape_refusal(const MoEExpertRowsShape& shape) {
    if (shape.tokens == 0 || shape.tokens % shape.num_sources != 0) {
        return fmt::format(
            "{} tokens, expected a positive multiple of the {} sources", shape.tokens, shape.num_sources);
    }
    if (shape.top_k < 1 || shape.top_k > kMaxTopK) {
        return fmt::format("top_k {} outside 1..{}", shape.top_k, kMaxTopK);
    }
    if (uint64_t{shape.tokens} * shape.top_k > kMaxRoutedRows) {
        return fmt::format(
            "{} tokens * top_k {} = {} routed rows, more than the routing table's u16 rows hold ({})",
            shape.tokens,
            shape.top_k,
            uint64_t{shape.tokens} * shape.top_k,
            kMaxRoutedRows);
    }
    return std::nullopt;
}

JobCycles expert_rows_host::job_cycles(
    const MoEExpertRowsShape& shape,
    const ExpertRowsPlan& plan,
    uint32_t num_shards,
    tt::ARCH arch,
    tt::tt_metal::MathFidelity fidelity) {
    const JobCost per_job = job_cost(shape, plan, num_shards, arch);
    const double tile_cycles = matmul_tile_cycles(fidelity);
    return {
        .stream = plan.expert_groups * per_job.stream_s * ai_clock_hz(arch),
        .row_tile = per_job.busiest * tile_cycles + per_job.reloads * kReloadTileCycles,
        .single = per_job.busiest * tile_cycles + per_job.single_reloads * kReloadTileCycles};
}

//-----------------------------------------------------------------------------------------------------------------
// Selection between the ring program and the expert rows programs
//-----------------------------------------------------------------------------------------------------------------
namespace {

std::optional<MoEExpertRowsParams> choose_path(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args, ForcedKernel forced, uint32_t budget) {
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    if (forced == ForcedKernel::Ring) {
        return ring_with_reason(forced, "TT_METAL_MOE_COMPUTE_KERNEL=ring");
    }
    if (args.path == MoEComputePath::SingleCluster) {
        return ring_with_reason(
            forced, "an extent-1 cluster axis has no combine; the ring's dm1 writes the output itself");
    }
    if (!expert_rows_arch(mesh_device->arch())) {
        return ring_with_reason(
            forced, fmt::format("the expert rows path runs on Blackhole and Wormhole, not {}", mesh_device->arch()));
    }
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE && forced != ForcedKernel::ExpertRows) {
        return ring_with_reason(
            forced, "Wormhole takes the expert rows path only when TT_METAL_MOE_COMPUTE_KERNEL=expert_rows forces it");
    }
    // moe_compute's rules before the choice reads the call's tensors and program 1 runs (the launch checks them again)
    MoEComputeDeviceOperation::validate_on_program_cache_miss(args, tensor_args);
    if (const auto refusal = input_layout_refusal(tensor_args)) {
        return ring_with_reason(forced, *refusal);
    }
    if (args.num_shared_experts_per_device.value_or(0) > 0 && args.combine_params.has_value()) {
        return ring_with_reason(
            forced,
            fmt::format(
                "{} shared experts per device with the combine (split over the mesh)",
                *args.num_shared_experts_per_device));
    }
    const auto specs = MoEComputeDeviceOperation::compute_output_specs(args, tensor_args);
    const MoEExpertRowsShape shape = expert_rows_shape(*mesh_device, args, tensor_args);
    if (const auto refusal = shape_refusal(shape)) {
        return ring_with_reason(forced, *refusal);
    }
    const uint32_t scores_page_bytes = aligned_page(tensor_args.tilize_expert_scores_tensor);
    if (scores_page_bytes != shape.index_page_bytes) {
        return ring_with_reason(
            forced,
            fmt::format(
                "the expert scores pages are {} B, the expert indices pages {} B",
                scores_page_bytes,
                shape.index_page_bytes));
    }
    const auto shard_banks = shard_banks_of(tensor_args.matmul_w0_w1_tensor);
    const tt::DataFormat partials = partials_format(args.fp32_dest_acc_en);
    auto first = plan_expert_rows(mesh_device, shape, shard_banks, budget, 1, partials);
    if (const auto* refusal = std::get_if<std::string>(&first)) {
        return ring_with_reason(forced, *refusal);
    }
    const uint64_t outputs_per_core = l1_outputs_per_core(*mesh_device, specs);
    // place.cpp and rows_combine.cpp run on the same combine cores
    const PlaceLayout place = place_layout(shape.tokens, shape.top_k, shape.local_experts);
    uint64_t placement_cb_bytes = place.scratch_bytes;
    if (args.path != MoEComputePath::ComputeOnly) {
        const uint32_t senders = args.num_token_parallel_cores * args.num_data_parallel_cores;
        placement_cb_bytes += combine_layout(shape.tokens, shape.top_k, shape.hidden_size, senders).bytes;
        if (args.path == MoEComputePath::FullCcl) {
            placement_cb_bytes += kCombinePacketHeaders * tt::tt_fabric::CLIENT_INTERFACE_SIZE;
        }
    }
    if (placement_cb_bytes + outputs_per_core > budget) {
        return ring_with_reason(
            forced,
            fmt::format(
                "placement program CBs ({} B) and moe_compute's L1 outputs ({} B per core) exceed the {} B CB budget",
                placement_cb_bytes,
                outputs_per_core,
                budget));
    }
    // Can the ring program run this call at all (ring_tilize_cb_bytes)?
    const bool ring_fits =
        ring_tilize_cb_bytes(mesh_device, args, shape, specs, tensor_args) + outputs_per_core + kRingFitMargin <=
        budget;
    // Cost model. Every path streams each touched expert's weights once per job: a job is at most 32 of its rows on the
    // ring and with one row tile per job, at most 32 row_tiles rows with weight-stationary jobs. A bank's weights come
    // at 1 / banks of the DRAM peak, so the busiest bank sets the stream time of a job (stream_s). The ring computes
    // each job on one core per bank and pays a ring pass per job; the expert rows path spreads the work over
    // readers_per_bank cores per bank and runs expert_groups groups in parallel. With the combine (SingleDevice,
    // FullCcl) the ring overlaps it with its compute, while the expert rows path sends the rows after they exist, so
    // that path also pays its combine: the device's rows at kFabricCombineBytesPerSecond, or
    // kLocalCombineBytesPerSecond on SingleDevice.
    //   t_ring = jobs(32) * (max(stream_s, ring_tiles * tile_s) + ring * hop)
    //   t_expert_rows = max(jobs(32 row_tiles) * stream_s,
    //                       ceil(jobs(32) / expert_groups) * (busiest * tile_s + reloads * reload))
    //                   [+ rows * row_bytes / combine_bytes_per_s]
    // tile_s: seconds per tile matmul (matmul_tile_cycles at the AI clock); hop and reload (kRingHopCycles,
    // kReloadTileCycles): cycle counts at the clock; jobs(n):
    // expected jobs of at most n rows (expected_jobs: top_k / global experts of each token); row_tiles: row tiles per
    // job; reloads: the partial tiles a row tile of a job of several row tiles writes and reads back between chunks.
    // The expert rows path runs only with a clear predicted margin over the ring (kCostMargin); within it the ring is
    // kept.
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const uint32_t num_shards = shard_banks.size();
    const uint32_t w0_w1_k_rows = hidden_tiles + shape.bias_rows(), w2_k_rows = intermediate_tiles + shape.bias_rows();
    const double clock_hz = ai_clock_hz(mesh_device->arch());
    const CostModel model{
        .shape = shape,
        .num_shards = num_shards,
        .arch = mesh_device->arch(),
        .tile_s = matmul_tile_cycles(args.math_fidelity) / clock_hz,
        .clock_hz = clock_hz};
    ExpertRowsPlan& one_row_tile = std::get<ExpertRowsPlan>(first);
    const PlanCost one_row_tile_cost = model.cost(one_row_tile);
    auto chosen = choose_row_tiles(
        mesh_device, model, shard_banks, budget, partials, CostedPlan{std::move(one_row_tile), one_row_tile_cost});
    if (const auto* why = std::get_if<std::string>(&chosen)) {
        return ring_with_reason(forced, *why);
    }
    const auto& [plan, cost] = std::get<CostedPlan>(chosen);
    const double jobs = expected_jobs(shape, expert_rows::tile_rows);
    uint32_t ring_columns = 0;  // gate/up columns of the ring's busiest core (moe_ring_common.h)
    for (uint32_t position = 0; position < num_shards; ++position) {
        ring_columns = std::max(ring_columns, moe_ring::w0_w1_stored_cols(intermediate_tiles, position, num_shards));
    }
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    // matmul tiles of the ring's busiest core per job: W0 and W1 of its columns, its W2 output groups
    const double ring_tiles =
        2.0 * w0_w1_k_rows * ring_columns + static_cast<double>(prepared.w2_groups.tiles()) * w2_k_rows;
    const double t_ring =
        jobs * (std::max(cost.stream_s, ring_tiles * model.tile_s) + num_shards * kRingHopCycles / clock_hz);
    double t_expert_rows = cost.call_s;
    if (args.path != MoEComputePath::ComputeOnly) {
        const double rows =
            static_cast<double>(shape.tokens) * shape.top_k * shape.local_experts / shape.global_experts;
        const double combine_bytes_per_s =
            args.path == MoEComputePath::SingleDevice ? kLocalCombineBytesPerSecond : kFabricCombineBytesPerSecond;
        t_expert_rows += rows * shape.hidden_size * kBf16Bytes / combine_bytes_per_s;
    }
    if (ring_fits && t_expert_rows * (1.0 + kCostMargin) >= t_ring && forced != ForcedKernel::ExpertRows) {
        return ring_with_reason(
            forced,
            fmt::format(
                "cost model: ring {:.1f} us, expert rows {:.1f} us for {:.1f} expected jobs, no clear margin",
                t_ring * 1e6,
                t_expert_rows * 1e6,
                jobs));
    }
    log_debug(
        tt::LogOp,
        "moe_compute: expert rows kernel (cost model: expert rows {:.1f} us, ring {:.1f} us for {:.1f} expected jobs, "
        "ring program {} L1; {} row tiles per job, {} readers per bank, {} expert groups of {} cores, "
        "x / intermediate slots {} / {}, x chunk {} tiles, W2 chunk {} rows, {} B CBs)",
        t_expert_rows * 1e6,
        t_ring * 1e6,
        jobs,
        ring_fits ? "fits" : "does not fit",
        plan.row_tiles,
        plan.readers_per_bank,
        plan.expert_groups,
        plan.group_cores,
        plan.x_slots,
        plan.intermediate_slots,
        plan.chunk_tiles,
        plan.w2_chunk_rows,
        plan.cbs.total());
    return MoEExpertRowsParams{
        .shape = shape,
        .layer_id = args.layer_id,
        .cluster_axis = args.cluster_axis().value_or(1),
        .cb_budget = plan.cbs.total(),  // the plan search is first-fit: this budget rebuilds the same plan
        .row_tiles = plan.row_tiles,
        .activation_type = args.activation_type,
        .activation_limit = args.activation_limit,
        .math_fidelity = args.math_fidelity,
        .fp32_dest_acc_en = args.fp32_dest_acc_en};
}

}  // namespace

// The choice is kept per call shape (moe_compute's attributes without the layer, its tensors, the device, the CB budget
// and the test hooks): a repeated shape skips the validation, plan search and cost model.
std::optional<MoEExpertRowsParams> select_moe_compute_expert_rows(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args) {
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const ForcedKernel forced = forced_kernel();
    const char* row_tiles = std::getenv(kRowTilesEnv);
    const uint32_t budget = cb_budget_of(*mesh_device);
    MoEComputeParams keyed = args;
    keyed.layer_id = 0;
    const auto key = ttsl::hash::hash_objects_with_default_seed(
        keyed,
        tensor_args,
        mesh_device->id(),
        budget,
        static_cast<uint32_t>(forced),
        std::string(row_tiles == nullptr ? "" : row_tiles));
    static std::mutex mutex;
    static std::unordered_map<ttsl::hash::hash_t, std::optional<MoEExpertRowsParams>> choices;
    {
        const std::lock_guard lock(mutex);
        if (const auto it = choices.find(key); it != choices.end()) {
            auto params = it->second;
            if (params.has_value()) {
                params->layer_id = args.layer_id;
            }
            return params;
        }
    }
    auto params = choose_path(args, tensor_args, forced, budget);
    const std::lock_guard lock(mutex);
    if (choices.size() >= kMaxRememberedChoices) {
        choices.clear();
    }
    choices.emplace(key, params);
    return params;
}

}  // namespace ttnn::experimental::prim
