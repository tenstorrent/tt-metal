// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_unified_placement.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar::matmul {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

// The machine the estimate assumes, per arch. Only the ranking of placements matters; Quasar's numbers are
// first-order until the emulator measures them.
struct ArchCostModel {
    double tile_matmul_cycles;         // one 32x32x32 tile product per math fidelity phase, on one compute engine
    double tile_pack_cycles;           // one tile packed out of DST or reloaded into it
    double noc_read_bytes_per_cycle;   // into one core
    double noc_write_bytes_per_cycle;  // out of one core, multicasts included
    double dram_bytes_per_cycle;       // the whole device
    double C_slice_cycles;             // fixed cost per C slice: handshakes and pipeline fill
};

constexpr ArchCostModel WORMHOLE_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .tile_pack_cycles = 16.0,
    .noc_read_bytes_per_cycle = 32.0,
    .noc_write_bytes_per_cycle = 32.0,
    .dram_bytes_per_cycle = 256.0,
    .C_slice_cycles = 1000.0,
};

constexpr ArchCostModel BLACKHOLE_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .tile_pack_cycles = 16.0,
    .noc_read_bytes_per_cycle = 64.0,
    .noc_write_bytes_per_cycle = 64.0,
    .dram_bytes_per_cycle = 380.0,
    .C_slice_cycles = 1000.0,
};

constexpr ArchCostModel QUASAR_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .tile_pack_cycles = 16.0,
    .noc_read_bytes_per_cycle = 128.0,
    .noc_write_bytes_per_cycle = 128.0,
    .dram_bytes_per_cycle = 512.0,
    .C_slice_cycles = 2000.0,
};

// Estimated cycles of a plan: the slowest of the busiest core's compute and data movement and the device's DRAM
// traffic (the buffers overlap them), plus a fixed cost per C slice. Second, to break ties: the three summed
// (decode reads the same DRAM bytes on 8 cores as on 32).
std::pair<double, double> estimate_plan_cycles(
    const ttnn::prim::qsr::UnifiedMatmulPlan& plan,
    tt::ARCH arch,
    tt::tt_metal::MathFidelity math_fidelity,
    bool A_in_dram,
    bool B_in_dram,
    bool C_in_dram) {
    const ArchCostModel& model = arch == tt::ARCH::QUASAR      ? QUASAR_COST_MODEL
                                 : arch == tt::ARCH::BLACKHOLE ? BLACKHOLE_COST_MODEL
                                                               : WORMHOLE_COST_MODEL;
    const uint32_t fidelity_phases = math_fidelity == tt::tt_metal::MathFidelity::LoFi    ? 1
                                     : math_fidelity == tt::tt_metal::MathFidelity::HiFi2 ? 2
                                     : math_fidelity == tt::tt_metal::MathFidelity::HiFi3 ? 3
                                                                                          : 4;
    const double C_slices_on_busiest_core = (double)plan.max_C_slices_per_core * plan.batch_size;
    const double total_C_slices = (double)plan.C_slices_per_batch * plan.batch_size;

    // Compute: the busiest thread's share of each C slice over all of K, plus its packs and partial reloads.
    const uint32_t reloads_per_C_slice =
        plan.packer_l1_acc_en ? 1 : plan.K_chunks_per_C_slice - 1;  // with L1 accumulation only the last K chunk
    const double compute_cycles = C_slices_on_busiest_core * plan.C_entries_per_thread *
                                  (plan.K_tiles * model.tile_matmul_cycles * fidelity_phases +
                                   (plan.K_chunks_per_C_slice + reloads_per_C_slice) * model.tile_pack_cycles);

    // Data movement on the busiest core, per C slice: its A rows and B columns over all of K. A multicast sender
    // also writes them to its row or column once; a receiver reads nothing; a borrowed operand does not move.
    const double A_bytes = plan.borrow_A ? 0.0 : (double)plan.C_slice_M_tiles * plan.K_tiles * plan.A_entry_bytes;
    const double B_bytes = plan.borrow_B ? 0.0 : (double)plan.K_tiles * plan.C_slice_N_tiles * plan.B_entry_bytes;
    const double bias_bytes = (double)plan.C_slice_N_tiles * plan.bias_entry_bytes;
    const double C_bytes = (double)plan.C_slice_M_tiles * plan.C_slice_N_tiles * plan.C_entry_bytes;
    const double read_bytes = C_slices_on_busiest_core * (A_bytes + B_bytes + bias_bytes);
    const double write_bytes =
        C_slices_on_busiest_core * ((plan.A_mcast_num_dests > 0 ? A_bytes : 0.0) +
                                    (plan.B_mcast_num_dests > 0 ? B_bytes : 0.0) + (plan.borrow_C ? 0.0 : C_bytes));
    const double dm_cycles =
        read_bytes / model.noc_read_bytes_per_cycle + write_bytes / model.noc_write_bytes_per_cycle;

    // DRAM over the whole device: a multicast operand is read once per row or column of cores.
    const double dram_bytes =
        total_C_slices * ((A_in_dram ? A_bytes / (plan.A_mcast_num_dests + 1) : 0.0) +
                          (B_in_dram ? B_bytes / (plan.B_mcast_num_dests + 1) : 0.0) + (C_in_dram ? C_bytes : 0.0));
    const double dram_cycles = dram_bytes / model.dram_bytes_per_cycle;

    const double slowest_cycles = std::max({compute_cycles, dm_cycles, dram_cycles});
    return {slowest_cycles + C_slices_on_busiest_core * model.C_slice_cycles, compute_cycles + dm_cycles + dram_cycles};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::optional<MatmulUnifiedProgramConfig> choose_unified_placement(
    const Tensor& A,
    const Tensor& B,
    bool transpose_a,
    bool transpose_b,
    uint32_t bias_single_tile_size,
    const ttnn::prim::qsr::MatmulParams& attributes,
    const MatmulUnifiedProgramConfig& requested) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    TT_FATAL(
        requested.C_slice_M_tiles == 0 && requested.C_slice_N_tiles == 0,
        "MatmulUnifiedProgramConfig: with no cores the C slice is chosen with the placement; leave C_slice_M_tiles and "
        "C_slice_N_tiles at 0");
    // What the unified kernels do not do; the legacy picker takes these. Tiny tiles (any tile but 32x32) are out;
    // tensor dims need not be tile multiples.
    const tt::tt_metal::Tile::TileShape tile_32x32 = {TILE_HEIGHT, TILE_WIDTH};
    const bool output_tile_is_32x32 =
        !attributes.output_tile.has_value() || attributes.output_tile->get_tile_shape() == tile_32x32;
    if (transpose_a || transpose_b || attributes.untilize_out || attributes.global_cb.has_value() ||
        attributes.sub_device_id.has_value() || A.tensor_spec().tile().get_tile_shape() != tile_32x32 ||
        B.tensor_spec().tile().get_tile_shape() != tile_32x32 || !output_tile_is_32x32 ||
        A.logical_shape().rank() < 2 || B.logical_shape().rank() < 2) {
        return std::nullopt;
    }

    const tt::tt_metal::IDevice& device = *A.device();
    const CoreCoord grid = attributes.user_core_coord.value_or(device.compute_with_storage_grid_size());
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const tt::tt_metal::MathFidelity math_fidelity =
        std::get<0>(get_compute_kernel_config_args(device.arch(), attributes.compute_kernel_config.value()));
    // With a 32-bit DST ttnn.linear adds the bias separately (see the plan), so it takes no L1 here.
    const uint32_t bias_tile_bytes = fp32_dest_acc_en ? 0 : bias_single_tile_size;
    const bool B_has_one_batch = attributes.bcast_batch.value_or(false);
    const uint32_t M_per_batch = A.padded_shape()[-2];
    const uint32_t N = B.padded_shape()[-1];
    // Batches of A that share one B are stacked into one [batch * M, N] matmul, as the plan does.
    const uint32_t stacked_batches = B_has_one_batch ? (uint32_t)get_batch_size(A.padded_shape()) : 1;
    const uint32_t M_tiles = stacked_batches * M_per_batch / TILE_HEIGHT;
    const uint32_t N_tiles = N / TILE_WIDTH;
    const tt::tt_metal::MemoryConfig& C_memory = attributes.output_mem_config;

    const auto make_config = [&](CoreRangeSet cores,
                                 uint32_t C_slice_M_tiles,
                                 uint32_t C_slice_N_tiles,
                                 tt::tt_metal::ShardOrientation orientation) {
        MatmulUnifiedProgramConfig config = requested;
        config.cores = std::move(cores);
        config.C_slice_M_tiles = C_slice_M_tiles;
        config.C_slice_N_tiles = C_slice_N_tiles;
        config.orientation = orientation;
        return config;
    };
    const auto rectangle = [](uint32_t columns, uint32_t rows) {
        return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(columns - 1, rows - 1)));
    };
    constexpr tt::tt_metal::ShardOrientation ROW_MAJOR = tt::tt_metal::ShardOrientation::ROW_MAJOR;

    // Each candidate is a placement plus a list of C slice shapes to try in order (0 = auto); the first that fits
    // is the candidate's plan.
    struct Candidate {
        std::string name;
        std::vector<MatmulUnifiedProgramConfig> configs;
    };
    std::vector<Candidate> candidates;
    // C slices of a fixed width (fixed_is_N) or height, the other dimension all of C's, then split into 2, 3, ...
    // pieces until a plan fits. The last piece may be shorter (13 tiles in 4 pieces: 4, 4, 4, 1).
    const auto shrinking_C_slices =
        [&](CoreRangeSet cores, uint32_t fixed_tiles, bool fixed_is_N, tt::tt_metal::ShardOrientation orientation) {
            std::vector<MatmulUnifiedProgramConfig> configs;
            const uint32_t free_tiles = fixed_is_N ? M_tiles : N_tiles;
            uint32_t previous_free_slice_tiles = 0;
            for (uint32_t pieces = 1; pieces <= free_tiles; ++pieces) {
                const uint32_t free_slice_tiles = tt::div_up(free_tiles, pieces);
                if (free_slice_tiles != previous_free_slice_tiles) {
                    configs.push_back(
                        fixed_is_N ? make_config(cores, free_slice_tiles, fixed_tiles, orientation)
                                   : make_config(cores, fixed_tiles, free_slice_tiles, orientation));
                    previous_free_slice_tiles = free_slice_tiles;
                }
            }
            return configs;
        };

    const std::optional<tt::tt_metal::ShardSpec>& C_shard = C_memory.shard_spec();
    if (C_memory.is_sharded() && C_shard.has_value()) {
        // The output shard fixes the cores and the C slice.
        candidates.push_back({"output_shard", {make_config(C_shard->grid, 0, 0, C_shard->orientation)}});
    } else {
        const uint32_t rows = std::min<uint32_t>(grid.y, M_tiles);
        const uint32_t columns = std::min<uint32_t>(grid.x, N_tiles);
        candidates.push_back({"2D", {make_config(rectangle(columns, rows), 0, 0, ROW_MAJOR)}});
        candidates.push_back({"1D_row", {make_config(rectangle(columns, 1), 0, 0, ROW_MAJOR)}});
        candidates.push_back({"1D_column", {make_config(rectangle(1, rows), 0, 0, ROW_MAJOR)}});
        // 1D over whole rows of the grid, one C slice per core: when the C slices span all of C's other dimension
        // every core needs the same A (B), which is multicast to all of them.
        const auto whole_rows = [&](uint32_t num_C_slices) {
            return num_C_slices < grid.x ? rectangle(num_C_slices, 1)
                                         : rectangle(grid.x, std::min<uint32_t>(grid.y, num_C_slices / grid.x));
        };
        const CoreRangeSet across_N_cores = whole_rows(N_tiles);
        const CoreRangeSet across_M_cores = whole_rows(M_tiles);
        candidates.push_back(
            {"1D_across_N",
             shrinking_C_slices(
                 across_N_cores,
                 tt::div_up(N_tiles, across_N_cores.num_cores()),
                 /*fixed_is_N=*/true,
                 ROW_MAJOR)});
        candidates.push_back(
            {"1D_across_M",
             shrinking_C_slices(
                 across_M_cores,
                 tt::div_up(M_tiles, across_M_cores.num_cores()),
                 /*fixed_is_N=*/false,
                 ROW_MAJOR)});
        // Keep a sharded operand where it is: an L1 shard can then be bound in place, and a DRAM shard is read by
        // the core next to its bank (the cores listed in bank order, which the C slice walk follows).
        const auto dram_bank_cores = [&]() {
            std::vector<CoreRange> cores;
            for (const CoreCoord& core :
                 device.get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0)) {
                cores.emplace_back(core);
            }
            return CoreRangeSet(std::move(cores));
        };
        const tt::tt_metal::TensorMemoryLayout A_layout = A.memory_config().memory_layout();
        const tt::tt_metal::TensorMemoryLayout B_layout = B.memory_config().memory_layout();
        if (A.is_sharded() && A_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
            const CoreRangeSet cores = A.memory_config().is_dram() ? dram_bank_cores() : A.shard_spec()->grid;
            const uint32_t A_shard_M_tiles = A.shard_spec()->shape[0] / TILE_HEIGHT;
            candidates.push_back(
                {"A_shard_rows",
                 shrinking_C_slices(cores, A_shard_M_tiles, /*fixed_is_N=*/false, A.shard_spec()->orientation)});
        }
        if (B.is_sharded() && B_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
            const CoreRangeSet cores = B.memory_config().is_dram() ? dram_bank_cores() : B.shard_spec()->grid;
            const uint32_t B_shard_N_tiles = B.shard_spec()->shape[1] / TILE_WIDTH;
            candidates.push_back(
                {"B_shard_columns",
                 shrinking_C_slices(cores, B_shard_N_tiles, /*fixed_is_N=*/true, B.shard_spec()->orientation)});
        }
    }

    std::optional<MatmulUnifiedProgramConfig> best;
    std::pair<double, double> best_estimate;
    for (const Candidate& candidate : candidates) {
        for (const MatmulUnifiedProgramConfig& config : candidate.configs) {
            if (config.cores.bounding_box().end_coord.x >= device.compute_with_storage_grid_size().x ||
                config.cores.bounding_box().end_coord.y >= device.compute_with_storage_grid_size().y) {
                break;
            }
            const std::optional<ttnn::prim::qsr::UnifiedMatmulPlan> plan = ttnn::prim::qsr::plan_unified_matmul(
                device, A, B, bias_tile_bytes, config, attributes, std::nullopt, /*must_fit=*/false);
            if (!plan.has_value()) {
                continue;
            }
            // A sharded output takes the layout the caller asked for, which the C slices must tile.
            if (C_memory.is_sharded() && !C_shard.has_value() &&
                plan->sharded_output_layout != C_memory.memory_layout()) {
                break;
            }
            const std::pair<double, double> estimate = estimate_plan_cycles(
                *plan,
                device.arch(),
                math_fidelity,
                A.memory_config().is_dram(),
                B.memory_config().is_dram(),
                C_memory.is_dram());
            log_debug(
                tt::LogOp,
                "Unified matmul placement: {} with a {}x{}-tile C slice on {} cores: {:.0f} cycles",
                candidate.name,
                plan->C_slice_M_tiles,
                plan->C_slice_N_tiles,
                plan->cores.size(),
                estimate.first);
            if (!best.has_value() || estimate < best_estimate) {
                best = config;
                best_estimate = estimate;
            }
            break;
        }
    }
    return best;
}

}  // namespace ttnn::operations::experimental::quasar::matmul
