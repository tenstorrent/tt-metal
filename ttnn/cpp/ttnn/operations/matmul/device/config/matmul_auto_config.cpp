// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

#include <algorithm>
#include <iterator>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"
#include "ttnn/operations/matmul/device/config/factory_blocking_source.hpp"
#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

namespace {

// Headroom kept free below the L1 budget, for allocator alignment and small factory-side buffers
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

std::string check(const MatmulDesc& p, const HardwareDesc& hw, const MatmulProgramConfig& config) {
    const uint32_t cores = hw.grid.x * hw.grid.y;
    return std::visit(
        [&](const auto& c) -> std::string {
            using T = std::decay_t<decltype(c)>;
            constexpr bool reuse = std::is_same_v<T, MatmulMultiCoreReuseProgramConfig>;
            constexpr bool two_d = std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig>;
            constexpr bool one_d = std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>;
            if constexpr (!reuse && !two_d && !one_d) {
                return "not a config type the selector emits";
            } else {
                if (c.in0_block_w == 0 || p.Kt % c.in0_block_w != 0) {
                    return fmt::format("Kt {} is not a multiple of in0_block_w {}", p.Kt, c.in0_block_w);
                }
                if (c.per_core_M == 0 || c.per_core_N == 0 || c.out_subblock_h == 0 || c.out_subblock_w == 0) {
                    return "zero block size";
                }
                // Block-float B with A tiles under 16 rows: the mcast kernels can't unpack it, and Reuse computes
                // wrong values unless K is a single block
                if (needs_single_k_reuse(p)) {
                    if (!reuse) {
                        return "block-float B with A tiles under 16 rows needs Reuse";
                    }
                    if (c.in0_block_w != p.Kt) {
                        return "block-float B with A tiles under 16 rows needs a single K block";
                    }
                }
                if constexpr (reuse) {
                    if (c.out_subblock_h * c.out_subblock_w > max_subblock_area(p, Family::Reuse)) {
                        return "subblock exceeds DST";
                    }
                    if (c.per_core_N != p.Nt) {
                        return "Reuse needs per_core_N == Nt";
                    }
                    const bool divides = p.Mt % c.per_core_M == 0;
                    const bool whole_batches = c.per_core_M % p.Mt == 0 && (p.batch_a * p.Mt) % c.per_core_M == 0;
                    if (!divides && !whole_batches) {
                        return "Reuse per_core_M neither divides Mt nor covers whole batches";
                    }
                    if (c.per_core_M % c.out_subblock_h != 0 || c.per_core_N % c.out_subblock_w != 0 ||
                        p.Mt % c.out_subblock_h != 0) {
                        return "Reuse subblock doesn't divide the block";
                    }
                    const Blocking b{
                        c.per_core_M,
                        c.per_core_N,
                        c.in0_block_w,
                        c.per_core_M,
                        c.per_core_N,
                        c.out_subblock_h,
                        c.out_subblock_w};
                    if (circular_buffer_bytes(p, hw, Family::Reuse, b, true) > hw.l1_cb_budget) {
                        return "circular buffers exceed L1";
                    }
                    return "";
                } else {
                    Family family = Family::Mcast2D;
                    if constexpr (one_d) {
                        family = c.mcast_in0 ? Family::Mcast1DIn0 : Family::Mcast1DIn1;
                    }
                    if (c.out_subblock_h * c.out_subblock_w > max_subblock_area(p, family)) {
                        return "subblock exceeds DST";
                    }
                    if (c.out_block_h == 0 || c.out_block_w == 0 || c.per_core_M % c.out_block_h != 0 ||
                        c.per_core_N % c.out_block_w != 0 || c.out_block_h % c.out_subblock_h != 0 ||
                        c.out_block_w % c.out_subblock_w != 0) {
                        return "blocks don't divide";
                    }
                    if (c.fuse_batch && p.batch_b > 1) {
                        return "fuse_batch with a batched B";
                    }
                    const uint32_t M = output_rows(p, c.fuse_batch);
                    const uint32_t blocks_y = div_up(M, c.per_core_M);
                    const uint32_t blocks_x = div_up(p.Nt, c.per_core_N);
                    if constexpr (two_d) {
                        if (c.per_core_M > M || blocks_y > hw.grid.y || blocks_x > hw.grid.x) {
                            return "2D blocks exceed the grid";
                        }
                    } else {
                        if (blocks_x * blocks_y > cores) {
                            return "1D blocks exceed the core count";
                        }
                        if (c.mcast_in0 && blocks_y != 1) {
                            return "1D in0-mcast needs one row of blocks";
                        }
                        if (!c.mcast_in0) {
                            if (c.per_core_N != p.Nt || c.per_core_M > M) {
                                return "1D in1-mcast needs per_core_N == Nt";
                            }
                            if (blocks_y == 1 && M % c.out_block_h != 0 && c.per_core_M != c.out_block_h) {
                                return "1D in1-mcast single row block";
                            }
                        }
                    }
                    const Blocking b{
                        c.per_core_M,
                        c.per_core_N,
                        c.in0_block_w,
                        c.out_block_h,
                        c.out_block_w,
                        c.out_subblock_h,
                        c.out_subblock_w};
                    if (circular_buffer_bytes(p, hw, family, b, c.fuse_batch) > hw.l1_cb_budget) {
                        return "circular buffers exceed L1";
                    }
                    return "";
                }
            }
        },
        config);
}

const Candidate& best_by_estimate(
    const MatmulDesc& p,
    const HardwareDesc& hw,
    std::span<const Candidate> options,
    std::span<const std::shared_ptr<const Estimator>> estimators) {
    TT_FATAL(!options.empty(), "best_by_estimate needs at least one option");
    const Candidate* best = &options.front();
    std::optional<Estimate> best_estimate;
    for (const auto& option : options) {
        std::optional<Estimate> chosen;
        for (const auto& estimator : estimators) {
            auto e = estimator->estimate(p, hw, option);
            if (e && (!chosen || e->confidence > chosen->confidence)) {
                chosen = e;
            }
        }
        if (chosen && (!best_estimate || chosen->cycles < best_estimate->cycles)) {
            best = &option;
            best_estimate = chosen;
        }
    }
    return *best;
}

const Selector& default_selector() {
    static const Selector selector{
        .sources = {std::make_shared<FactoryBlockingSource>()},
        .estimators = {std::make_shared<RooflineEstimator>()},
    };
    return selector;
}

std::optional<Candidate> select(const MatmulDesc& p, const HardwareDesc& hw, const Selector& selector) {
    std::vector<Candidate> options;
    for (const auto& source : selector.sources) {
        auto proposed = source->propose(p, hw);
        options.insert(
            options.end(), std::make_move_iterator(proposed.begin()), std::make_move_iterator(proposed.end()));
    }
    if (options.empty()) {
        return std::nullopt;
    }
    return best_by_estimate(p, hw, options, selector.estimators);
}

std::optional<MatmulProgramConfig> select_program_config(const MatmulDesc& p, const HardwareDesc& hw) {
    if (p.Mt == 0 || p.Kt == 0 || p.Nt == 0 || hw.grid.x == 0 || hw.grid.y == 0) {
        return std::nullopt;
    }
    const auto chosen = select(p, hw);
    if (!chosen) {
        return std::nullopt;
    }
    return to_program_config(p, *chosen);
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
    if (p.out_tile_w != p.in1_tile_w) {
        return "output tile wider than B's tile";
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

}  // namespace

std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported) {
    auto reject = [&](std::string reason) -> std::optional<MatmulProgramConfig> {
        if (unsupported != nullptr) {
            *unsupported = std::move(reason);
        }
        return std::nullopt;
    };
    std::string why;
    const auto described = describe_matmul(
        input_tensor_a, input_tensor_b, transpose_a, transpose_b, bias_single_tile_size, attributes, why);
    if (!described) {
        return reject(why);
    }
    const MatmulDesc& p = *described;
    if (auto reason = unsupported_reason(p); !reason.empty()) {
        return reject(reason);
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
            return reject("sub-device worker cores not a rectangle");
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
    if (auto config = select_program_config(p, hw)) {
        return config;
    }
    // Nothing blocked fits: the non-reusing factory still runs all-interleaved 32x32 inputs on the device grid
    const bool all_interleaved = !p.a.sharded() && !p.b.sharded() && !p.out.sharded();
    const bool full_tiles = p.in0_tile_h == TILE_DIM && p.in1_tile_w == TILE_DIM;
    if (all_interleaved && full_tiles && !on_sub_device && !broadcasts_a(p)) {
        return MatmulMultiCoreProgramConfig{};
    }
    if (sharded_layout(p)) {
        return reject("unsupported sharded layout combination, or it doesn't fit L1");
    }
    if (needs_single_k_reuse(p) && p.batch_a != p.batch_b) {
        return reject("block-float B with A tiles under 16 rows needs Reuse, which can't broadcast B over A's batch");
    }
    return reject("no config fits L1");
}

}  // namespace ttnn::operations::matmul::auto_config
