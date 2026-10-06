// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/config/enumerating_source.hpp"
#include "ttnn/operations/matmul/device/config/factory_blocking_source.hpp"
#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

namespace {

// Chosen: headroom kept free below the L1 budget, for allocator alignment and small factory-side buffers
constexpr uint32_t L1_HEADROOM_BYTES = 16 * 1024;

tt::tt_metal::TensorMemoryLayout memory_layout(const Placement& t) {
    using tt::tt_metal::TensorMemoryLayout;
    switch (t.layout) {
        case MemoryLayout::Interleaved: return TensorMemoryLayout::INTERLEAVED;
        case MemoryLayout::HeightSharded: return TensorMemoryLayout::HEIGHT_SHARDED;
        case MemoryLayout::WidthSharded: return TensorMemoryLayout::WIDTH_SHARDED;
        case MemoryLayout::BlockSharded: return TensorMemoryLayout::BLOCK_SHARDED;
        case MemoryLayout::NdSharded: return TensorMemoryLayout::ND_SHARDED;
    }
    return TensorMemoryLayout::INTERLEAVED;
}

// The factories' buffer context for this matmul. A fused bias is taken to be interleaved (the selector doesn't
// see the bias tensor), which can only overestimate L1.
BufferContext buffer_context_of(const MatmulDesc& p, const HardwareDesc& hw) {
    return BufferContext{
        .in0_tile = tt::tt_metal::Tile({p.in0_tile_h, TILE_DIM}, p.in0_tile_transposed),
        .in1_tile = tt::tt_metal::Tile({TILE_DIM, p.in1_tile_w}),
        .output_tile = tt::tt_metal::Tile({p.in0_tile_h, p.in1_tile_w}),
        .in0_format = p.in0_format,
        .in1_format = p.in1_format,
        .output_format = p.out_format,
        .bias_tile_bytes = p.bias_tile_bytes,
        .bias_sharded = false,
        .fp32_dest_acc_en = p.fp32_dest_acc_en,
        .packer_l1_acc = p.packer_l1_acc,
        .untilize_out = p.untilize_out,
        .in0_layout = memory_layout(p.a),
        .in1_layout = memory_layout(p.b),
        .out_layout = memory_layout(p.out),
        .in1_in_dram = !p.b.in_l1,
        .in0_shard_width = p.a.shard_w,
        .in1_shard_height = p.b.shard_h,
        .dram_alignment = hw.dram_alignment,
        .Mt = p.Mt,
        .Kt = p.Kt,
    };
}

}  // namespace

HardwareDesc HardwareDesc::for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget) {
    HardwareDesc hw;
    hw.arch = arch;
    hw.grid = grid;
    hw.l1_cb_budget = l1_cb_budget;
    hw.dram_alignment = arch == tt::ARCH::BLACKHOLE ? 64 : 32;
    if (arch == tt::ARCH::BLACKHOLE) {
        hw.noc_bytes_per_cycle = 64;
        hw.dram_bytes_per_cycle = 512.0 / 1.35;  // 512 GB/s at 1.35 GHz
    }
    return hw;
}

uint32_t circular_buffer_bytes(
    const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch) {
    const auto context = buffer_context_of(p, hw);
    const uint32_t a_batches = fuse_batch ? 1 : p.batch_a;
    const uint32_t b_batches = fuse_batch ? 1 : p.batch_b;
    MatmulBuffers buffers;
    switch (family) {
        case Family::Mcast2D: buffers = mcast_2d_buffers(context, b, a_batches); break;
        case Family::Mcast1DIn0: buffers = mcast_1d_in0_buffers(context, b, a_batches, b_batches); break;
        case Family::Mcast1DIn1: buffers = mcast_1d_in1_buffers(context, b, a_batches, b_batches); break;
        case Family::Reuse: buffers = reuse_buffers(context, b); break;
    }
    uint32_t total = buffers.allocated_bytes();
    if (p.out.sharded()) {
        total += b.per_core_M * b.per_core_N * out_tile_bytes(p, p.out_format);  // allocated with the output
    }
    return total;
}

MatmulProgramConfig to_program_config(const MatmulDesc& p, const Candidate& c) {
    const Blocking& b = c.blocking;
    const std::optional<CoreRangeSet> worker_cores =
        c.worker_cores ? std::optional<CoreRangeSet>(CoreRangeSet(*c.worker_cores)) : std::nullopt;
    switch (c.family) {
        case Family::Mcast2D:
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = c.grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .transpose_mcast = c.transpose_mcast,
                .fused_activation = p.activation,
                .fuse_batch = c.fuse_batch,
                .allowed_worker_cores = worker_cores,
            };
        case Family::Mcast1DIn0:
        case Family::Mcast1DIn1:
            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = c.grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .fuse_batch = c.fuse_batch,
                // in0 reuse (broadcast A) can't fuse the activation; matmul then applies it separately
                .fused_activation = broadcasts_a(p) && !p.a.sharded() ? std::nullopt : p.activation,
                .mcast_in0 = c.family == Family::Mcast1DIn0,
                .allowed_worker_cores = worker_cores,
            };
        case Family::Reuse: break;
    }
    return MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = c.grid,
        .in0_block_w = b.in0_block_w,
        .out_subblock_h = b.out_subblock_h,
        .out_subblock_w = b.out_subblock_w,
        .per_core_M = b.per_core_M,
        .per_core_N = b.per_core_N,
        .allowed_worker_cores = worker_cores,
    };
}

std::string factory_limit_error(const MatmulDesc& p, const HardwareDesc& hw, const MatmulProgramConfig& config) {
    return std::visit(
        [&](const auto& c) -> std::string {
            using T = std::decay_t<decltype(c)>;
            constexpr bool reuse = std::is_same_v<T, MatmulMultiCoreReuseProgramConfig>;
            constexpr bool two_d = std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig>;
            constexpr bool one_d = std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>;
            if constexpr (!reuse && !two_d && !one_d) {
                return "";  // the configs the selector doesn't emit have no limits beyond validation here
            } else {
                // Sizes the buffer model divides by (validation rejects zero sizes too)
                if (c.in0_block_w == 0 || c.per_core_M == 0 || c.per_core_N == 0 || c.out_subblock_h == 0 ||
                    c.out_subblock_w == 0) {
                    return "zero block size";
                }
                Family family = Family::Reuse;
                Blocking b{
                    c.per_core_M,
                    c.per_core_N,
                    c.in0_block_w,
                    c.per_core_M,
                    c.per_core_N,
                    c.out_subblock_h,
                    c.out_subblock_w};
                bool fuse_batch = true;
                if constexpr (!reuse) {
                    if (c.out_block_h == 0 || c.out_block_w == 0) {
                        return "zero block size";
                    }
                    family = Family::Mcast2D;
                    if constexpr (one_d) {
                        family = c.mcast_in0 ? Family::Mcast1DIn0 : Family::Mcast1DIn1;
                    }
                    b.out_block_h = c.out_block_h;
                    b.out_block_w = c.out_block_w;
                    fuse_batch = c.fuse_batch;
                }
                if (c.out_subblock_h * c.out_subblock_w > max_subblock_area(p, family)) {
                    return "subblock exceeds the DST capacity the factory computes correctly with";
                }
                // Reuse computes wrong values when it splits K for block-float B with A tiles under 16 rows
                if (reuse && needs_single_k_reuse(p) && c.in0_block_w != p.Kt) {
                    return "block-float B with A tiles under 16 rows needs a single K block";
                }
                if (circular_buffer_bytes(p, hw, family, b, fuse_batch) > hw.l1_cb_budget) {
                    return "circular buffers exceed L1";
                }
                return "";
            }
        },
        config);
}

std::string check(
    const ttnn::prim::MatmulSpecs& specs,
    const MatmulDesc& p,
    const HardwareDesc& hw,
    const MatmulProgramConfig& config) {
    MatmulProgramConfig normalized = config;
    normalize_program_config(normalized, specs.device.grid);
    if (auto error = ttnn::prim::program_config_error(specs, normalized); !error.empty()) {
        return error;
    }
    return factory_limit_error(p, hw, config);
}

std::vector<const Candidate*> rank_by_estimate(
    const MatmulDesc& p,
    const HardwareDesc& hw,
    std::span<const Candidate> options,
    std::span<const std::shared_ptr<const Estimator>> estimators) {
    struct Ranked {
        const Candidate* candidate;
        std::optional<Estimate> estimate;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(options.size());
    for (const auto& option : options) {
        std::optional<Estimate> chosen;
        for (const auto& estimator : estimators) {
            auto e = estimator->estimate(p, hw, option);
            if (e && (!chosen || e->confidence > chosen->confidence)) {
                chosen = e;
            }
        }
        ranked.push_back({&option, chosen});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& x, const Ranked& y) {
        if (x.estimate.has_value() != y.estimate.has_value()) {
            return x.estimate.has_value();
        }
        return x.estimate.has_value() && x.estimate->cycles < y.estimate->cycles;
    });
    std::vector<const Candidate*> result;
    result.reserve(ranked.size());
    for (const auto& r : ranked) {
        result.push_back(r.candidate);
    }
    return result;
}

const Selector& default_selector() {
    static const Selector selector{
        .sources = {std::make_shared<FactoryBlockingSource>()},
        .estimators = {std::make_shared<RooflineEstimator>()},
    };
    return selector;
}

std::optional<Candidate> select(
    const ttnn::prim::MatmulSpecs& specs, const MatmulDesc& p, const HardwareDesc& hw, const Selector& selector) {
    std::vector<Candidate> options;
    for (const auto& source : selector.sources) {
        auto proposed = source->propose(p, hw);
        options.insert(options.end(), proposed.begin(), proposed.end());
    }
    for (const Candidate* candidate : rank_by_estimate(p, hw, options, selector.estimators)) {
        if (check(specs, p, hw, to_program_config(p, *candidate)).empty()) {
            return *candidate;
        }
    }
    return std::nullopt;
}

namespace {

std::string placement_issue(const Placement& t) {
    if (t.dram_sharded()) {
        return "DRAM-sharded tensor";
    }
    if (t.layout == MemoryLayout::NdSharded) {
        return "ND-sharded tensor";
    }
    if (t.has_shard_spec && !t.shard_whole_tiles) {
        return "shard shape not a whole number of tiles";
    }
    return "";
}

// Why the selector has no config for a matmul it can describe (the factories can't run these either), or empty
std::string unsupported_reason(const MatmulDesc& p) {
    if (auto why = placement_issue(p.a); !why.empty()) {
        return "A: " + why;
    }
    if (!dram_sharded_b(p)) {
        if (auto why = placement_issue(p.b); !why.empty()) {
            return "B: " + why;
        }
    }
    if (auto why = placement_issue(p.out); !why.empty()) {
        return "output: " + why;
    }
    // A sharded tensor must carry its shard spec (an output's may be left to the program config)
    if ((p.a.sharded() && !p.a.has_shard_spec) || (p.b.l1_sharded() && !p.b.has_shard_spec)) {
        return "sharded input without a shard spec";
    }
    if (dram_sharded_b(p) && (p.a.sharded() || p.out.sharded())) {
        return "DRAM-sharded B with a sharded A or output";
    }
    // An output tile wider than B's: each core's columns fill whole output tiles (split_n). An output shard spec or an
    // interleaved output is untested.
    if (p.out_tile_w != p.in1_tile_w) {
        if (p.out_tile_w % p.in1_tile_w != 0 || p.Nt % (p.out_tile_w / p.in1_tile_w) != 0) {
            return "output tile width not a multiple of B's tile width, or N not a whole number of output tiles";
        }
        if (!p.out.sharded() || p.out.has_shard_spec) {
            return "output tile wider than B's tile, unless the output is sharded without a shard spec";
        }
    }
    if (p.batch_b > 1 && p.batch_a != p.batch_b) {
        // Only a batch of one broadcasts (1D in1-mcast in0 reuse): interleaved tensors of equal rank >= 3
        if (p.batch_a != 1 || p.rank_a != p.rank_b || p.rank_a < 3) {
            return "A and B batches don't match";
        }
        if (p.a.sharded() || p.b.l1_sharded() || p.out.sharded()) {
            return "broadcast A with a sharded tensor";
        }
    }
    // A sharded tensor's shard spec is physical while transpose_a transposes the logical shape
    if (p.transpose_a && p.a.sharded()) {
        return "transpose_a with a sharded A";
    }
    if (dram_sharded_b(p) && (p.batch_b > 1 || needs_single_k_reuse(p))) {
        return "DRAM-sharded B that only 2D could read, but 2D can't run";
    }
    return "";
}

// The config for a matmul with no layout issue (unsupported_reason is empty): the selected candidate, else, when
// nothing blocked fits, the non-reusing factory if it can run the inputs; nullopt with the reason otherwise.
std::optional<MatmulProgramConfig> choose_config(
    const ttnn::prim::MatmulSpecs& specs, const MatmulDesc& p, const HardwareDesc& hw, std::string& why) {
    if (p.Mt == 0 || p.Kt == 0 || p.Nt == 0 || hw.grid.x == 0 || hw.grid.y == 0) {
        why = "empty matmul or grid";
        return std::nullopt;
    }
    if (const auto chosen = select(specs, p, hw)) {
        return to_program_config(p, *chosen);
    }
    // Nothing blocked fits: the non-reusing factory still runs all-interleaved 32x32 inputs on the device grid
    const bool all_interleaved = !p.a.sharded() && !p.b.sharded() && !p.out.sharded();
    const bool full_tiles = p.in0_tile_h == TILE_DIM && p.in1_tile_w == TILE_DIM;
    if (all_interleaved && full_tiles && !hw.pinned_origin && !broadcasts_a(p)) {
        MatmulProgramConfig multi_core = MatmulMultiCoreProgramConfig{};
        why = ttnn::prim::program_config_error(specs, multi_core);
        return why.empty() ? std::optional(multi_core) : std::nullopt;
    }
    if (sharded_layout(p)) {
        why = "unsupported sharded layout combination, or it doesn't fit L1";
    } else if (needs_single_k_reuse(p) && p.batch_a != p.batch_b) {
        why = "block-float B with A tiles under 16 rows needs Reuse, which can't broadcast B over A's batch";
    } else {
        why = "no config fits L1";
    }
    return std::nullopt;
}

}  // namespace

std::optional<MatmulProgramConfig> select_program_config(
    const ttnn::prim::MatmulSpecs& specs, const HardwareDesc& hw, std::string* unsupported) {
    std::string why;
    auto config = [&]() -> std::optional<MatmulProgramConfig> {
        const auto described = describe_matmul(specs, why);
        if (!described) {
            return std::nullopt;
        }
        if (why = unsupported_reason(*described); !why.empty()) {
            return std::nullopt;
        }
        return choose_config(specs, *described, hw, why);
    }();
    if (!config && unsupported != nullptr) {
        *unsupported = std::move(why);
    }
    return config;
}

namespace {

// What the selection needs to know about a matmul call
struct Problem {
    ttnn::prim::MatmulSpecs specs;
    MatmulDesc matmul;
    HardwareDesc hw;
};

// The matmul call's specs, description and hardware, or nullopt with the reason in `why` for inputs the selector
// has no config for
std::optional<Problem> describe_call(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes,
    std::string& why) {
    auto specs = ttnn::prim::matmul_specs({input_tensor_a, input_tensor_b}, bias, attributes);
    const auto described = describe_matmul(specs, why);
    if (!described) {
        return std::nullopt;
    }
    const MatmulDesc& p = *described;
    if (why = unsupported_reason(p); !why.empty()) {
        return std::nullopt;
    }

    // Worker grid: the device's, or on a sub-device its worker rectangle (the factories anchor their grid at
    // its first core); a user core_grid shrinks it
    auto* device = input_tensor_a.device();
    auto grid = device->compute_with_storage_grid_size();
    CoreCoord origin{0, 0};
    const bool on_sub_device = attributes.sub_device_id.has_value();
    if (on_sub_device) {
        const auto cores =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, attributes.sub_device_id.value());
        const auto bbox = cores.bounding_box();
        if (cores.num_cores() != bbox.size()) {
            why = "sub-device worker cores not a rectangle";
            return std::nullopt;
        }
        grid = bbox.grid_size();
        origin = bbox.start_coord;
    }
    if (attributes.user_core_coord.has_value()) {
        const auto& user = attributes.user_core_coord.value();
        if (user.x > 0 && user.y > 0) {
            grid = CoreCoord(std::min(user.x, grid.x), std::min(user.y, grid.y));
        }
    }

    // L1 left for CBs: free space above the lowest L1 buffer, less this op's own L1 output (not allocated yet)
    uint32_t budget = utilities::get_max_l1_space(input_tensor_a);
    if (attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 && !p.out.sharded()) {
        const uint64_t out_tiles = static_cast<uint64_t>(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
        const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
        const uint64_t out_per_bank =
            div_up(out_tiles, num_banks) * static_cast<uint64_t>(out_tile_bytes(p, p.out_format));
        budget = out_per_bank >= budget ? 0 : budget - static_cast<uint32_t>(out_per_bank);
    }
    budget = budget > L1_HEADROOM_BYTES ? budget - L1_HEADROOM_BYTES : 0;

    auto hw = HardwareDesc::for_arch(device->arch(), grid, budget);
    hw.origin = origin;
    hw.pinned_origin = on_sub_device;
    return Problem{std::move(specs), p, hw};
}

}  // namespace

std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported) {
    std::string why;
    std::optional<MatmulProgramConfig> config;
    if (const auto problem = describe_call(input_tensor_a, input_tensor_b, bias, attributes, why)) {
        config = choose_config(problem->specs, problem->matmul, problem->hw, why);
    }
    if (!config && unsupported != nullptr) {
        *unsupported = std::move(why);
    }
    return config;
}

std::vector<EnumeratedConfig> enumerate_program_configs(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported) {
    std::string why;
    const auto problem = describe_call(input_tensor_a, input_tensor_b, bias, attributes, why);
    if (!problem) {
        if (unsupported != nullptr) {
            *unsupported = std::move(why);
        }
        return {};
    }
    const auto& [specs, p, hw] = *problem;
    // The selection's own config leads, also when it isn't among the proposals (the non-reusing fallback)
    std::vector<EnumeratedConfig> result;
    std::string chosen_text;
    if (auto chosen = choose_config(specs, p, hw, why)) {
        chosen_text = fmt::format("{}", *chosen);
        result.push_back({std::move(*chosen), "heuristic", ""});
    }
    for (const auto& candidate : EnumeratingSource().propose(p, hw)) {
        auto config = to_program_config(p, candidate);
        if (fmt::format("{}", config) != chosen_text) {
            auto error = check(specs, p, hw, config);
            result.push_back({std::move(config), "enumerated", std::move(error)});
        }
    }
    return result;
}

namespace {
thread_local bool record_enumerated = false;
thread_local std::pair<std::vector<EnumeratedConfig>, std::string> last_enumerated;
}  // namespace

void set_record_enumerated_configs(bool on) { record_enumerated = on; }

void record_enumerated_configs(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes) {
    if (record_enumerated) {
        last_enumerated.second.clear();
        last_enumerated.first =
            enumerate_program_configs(input_tensor_a, input_tensor_b, bias, attributes, &last_enumerated.second);
    }
}

std::pair<std::vector<EnumeratedConfig>, std::string> last_enumerated_configs(bool reset) {
    auto result = last_enumerated;
    if (reset) {
        last_enumerated = {};
    }
    return result;
}

}  // namespace ttnn::operations::matmul::auto_config
