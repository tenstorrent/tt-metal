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

#include "moe_compute_device_operation.hpp"
#include "moe_expert_rows_plan.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::prim {

using namespace expert_rows_host;

namespace {

// Job cost model, used to pick the row tiles per job and by rows_reader.cpp to deal jobs to the expert groups. Cycles
// are at the AI clock under load, not the live clock, which idles lower and would make the choice history-dependent.
constexpr double kAiClockHzBlackhole = 1.35e9;
constexpr double kAiClockHzWormhole = 1.0e9;
// Peak DRAM bandwidth in bytes / s (Wormhole: 12 x 21.5 GB / s, ttnn/core/operation.cpp)
constexpr double kDramPeakBytesPerSecondBlackhole = 512e9;
constexpr double kDramPeakBytesPerSecondWormhole = 258e9;
// Cycles of one tile matmul per fidelity phase (tech_reports/matrix_engine/matrix_engine.md)
constexpr double kMatmulCyclesPerFidelityPhase = 16.0;
// Cycles measured on Blackhole: one FP32 partial tile written to L1 and reloaded into DEST, and one writer NoC command.
constexpr double kReloadTileCycles = 64.0;
constexpr double kWriterOpCycles = 60.0;
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

// Per-job cost terms of one plan: the weight stream time at the DRAM peak (busiest bank), and per row tile the busiest
// core's matmul tiles and the partial tiles it reloads between chunks, for held and for one-row-tile jobs.
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
// path (over the Wormhole gate; a call that path cannot run is refused with the reason).
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

// Weight-stationary jobs: the most row tiles per job for which the busiest core's matmuls and the writer's NoC commands
// stay within the job's weight stream time while the groups share DRAM. Returns the chosen plan, or why a forced
// row-tile count cannot run.
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
    // Every call the expert rows path can run takes it; the ring runs the rest.
    const double clock_hz = ai_clock_hz(mesh_device->arch());
    const CostModel model{
        .shape = shape,
        .num_shards = static_cast<uint32_t>(shard_banks.size()),
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
    const ExpertRowsPlan& plan = std::get<CostedPlan>(chosen).plan;
    log_debug(
        tt::LogOp,
        "moe_compute: expert rows kernel ({} row tiles per job, {} readers per bank, {} expert groups of {} cores, "
        "x / intermediate slots {} / {}, x chunk {} tiles, W2 chunk {} rows, {} B CBs)",
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
// and the test hooks): a repeated shape skips the validation and the plan search.
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
