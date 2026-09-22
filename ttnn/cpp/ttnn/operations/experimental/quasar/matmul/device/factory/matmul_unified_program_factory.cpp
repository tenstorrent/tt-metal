// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory (GH#41910): one Metal 2.0 program for every placement.
// Kernels address tiles by tile index through the tensor accessor, so operand layout doesn't matter here.

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Names the kernels see: dfb::A_slice / B_slice / C_slice / C_partials and tensor::A / B / C.
const DFBSpecName A_SLICE_DFB{"A_slice"};
const DFBSpecName B_SLICE_DFB{"B_slice"};
const DFBSpecName C_SLICE_DFB{"C_slice"};
const DFBSpecName C_PARTIALS_DFB{"C_partials"};

const TensorParamName A_TENSOR{"A"};
const TensorParamName B_TENSOR{"B"};
const TensorParamName C_TENSOR{"C"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName COMPUTE_KERNEL{"compute"};
const KernelSpecName WRITER_KERNEL{"writer"};

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/";

// A TRISC-visible DFB's extent is a uint16_t of 16-byte units (validate_ring_extent); enforced on
// every arch so a Wormhole-legal config never FATALs on Quasar.
constexpr uint64_t MAX_DFB_EXTENT_BYTES = 65535ull * 16ull;
constexpr uint32_t MAX_AUTO_K_CHUNK_TILES = 8;

// DFB sizing for one candidate K chunk; pure so the K chunk search can size several candidates.
struct DfbSizes {
    uint32_t K_chunk_tiles = 0;
    uint32_t num_K_chunks = 0;
    bool packer_l1_acc_en = false;
    tt::DataFormat C_partials_format{};
    uint32_t A_entry_bytes = 0;
    uint32_t B_entry_bytes = 0;
    uint32_t C_entry_bytes = 0;
    uint32_t C_partials_entry_bytes = 0;
    uint32_t A_slice_entries = 0;
    uint32_t B_slice_entries = 0;
    uint32_t C_slice_entries = 0;
    uint32_t C_partials_entries = 0;
    bool alias_C_partials_onto_C_slice = false;
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;
    uint64_t l1_bytes = 0;

    bool fits(uint64_t l1_budget) const {
        const uint64_t dfb_bytes[] = {
            (uint64_t)A_slice_entries * A_entry_bytes,
            (uint64_t)B_slice_entries * B_entry_bytes,
            (uint64_t)C_slice_entries * C_entry_bytes,
            (uint64_t)C_partials_entries * C_partials_entry_bytes};
        for (uint64_t bytes : dfb_bytes) {
            if (bytes > MAX_DFB_EXTENT_BYTES) {
                return false;
            }
        }
        return l1_bytes <= l1_budget;
    }
};

// Max-volume DST-filling subblock among the shapes the caller's fits predicate accepts (L1 fit and
// borrow preservation stay external): the C slice is rounded up to subblock multiples and the
// overshoot is clipped on write. Ties prefer the least padding waste; 1x1 (no padding) if nothing
// is accepted, and the caller's DFB sizing FATALs with the full breakdown.
template <typename FitsSubblock>
std::pair<uint32_t, uint32_t> maximize_subblock_size(
    uint32_t C_slice_M_tiles, uint32_t C_slice_N_tiles, uint32_t dst_capacity_tiles, FitsSubblock&& fits) {
    std::pair<uint32_t, uint32_t> best{1, 1};
    uint64_t best_volume = 0;
    uint64_t best_padded_area = UINT64_MAX;
    for (uint32_t h = 1; h <= dst_capacity_tiles; ++h) {
        for (uint32_t w = 1; h * w <= dst_capacity_tiles; ++w) {
            const uint64_t volume = h * w;
            const uint64_t padded_area = (uint64_t)tt::round_up(C_slice_M_tiles, h) * tt::round_up(C_slice_N_tiles, w);
            const bool better = volume > best_volume || (volume == best_volume && padded_area < best_padded_area);
            if (better && fits(h, w)) {
                best = {h, w};
                best_volume = volume;
                best_padded_area = padded_area;
            }
        }
    }
    return best;
}

DfbSizes size_dfbs(
    const UnifiedMatmulPlan& plan,
    uint32_t K_chunk_tiles,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    bool A_borrowable,
    bool B_borrowable,
    bool C_borrowable) {
    DfbSizes sizes;
    sizes.K_chunk_tiles = K_chunk_tiles;
    sizes.num_K_chunks = plan.K_tiles / K_chunk_tiles;

    // The packer accumulates partials in L1 only when there are enough K chunks for the reconfig overhead
    // to pay off (the last K chunk spills and reloads either way, so more than two).
    sizes.packer_l1_acc_en = packer_l1_acc && sizes.num_K_chunks > 2;
    sizes.C_partials_format = sizes.packer_l1_acc_en
                                  ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                  : (fp32_dest_acc_en ? tt::DataFormat::Float32 : plan.C_format);
    sizes.C_entry_bytes = tt::tile_size(plan.C_format);
    sizes.C_partials_entry_bytes = tt::tile_size(sizes.C_partials_format);

    const uint32_t C_slice_tiles = plan.C_slice_M_padded_tiles * plan.C_slice_N_padded_tiles;
    sizes.C_slice_entries = C_slice_tiles;
    sizes.C_partials_entries = C_slice_tiles;

    // Copied operands double-buffer when more than one slice passes through; a borrowed DFB is the
    // resident shard itself. A is borrowable only when one K chunk covers K.
    const bool more_than_one_slice = (uint64_t)plan.batch_size * plan.max_C_slices_per_core * sizes.num_K_chunks > 1;
    const uint32_t slice_buffering_factor = more_than_one_slice ? 2 : 1;
    sizes.borrow_A = A_borrowable && sizes.num_K_chunks == 1;
    sizes.borrow_B = B_borrowable;
    sizes.borrow_C = C_borrowable;
    sizes.A_entry_bytes = tt::tile_size(plan.A_format);
    sizes.B_entry_bytes = tt::tile_size(plan.B_format);
    sizes.A_slice_entries = sizes.borrow_A ? plan.C_slice_M_tiles * plan.K_tiles
                                           : plan.C_slice_M_padded_tiles * K_chunk_tiles * slice_buffering_factor;
    sizes.B_slice_entries = sizes.borrow_B ? plan.K_tiles * plan.C_slice_N_tiles
                                           : K_chunk_tiles * plan.C_slice_N_padded_tiles * slice_buffering_factor;

    // Alias C_partials onto C_slice only when partials are never live while C_slice holds unread data
    // (else compute packs slice i+1's partials over slice i before the writer drains it).
    const bool partials_ever_written = sizes.num_K_chunks > 1;
    const bool one_C_slice_per_core = plan.batch_size == 1 && plan.max_C_slices_per_core == 1;
    sizes.alias_C_partials_onto_C_slice =
        (sizes.C_partials_format == plan.C_format) && (!partials_ever_written || one_C_slice_per_core);

    // Borrowed DFBs are the tensors' own memory and cost nothing here.
    sizes.l1_bytes =
        (sizes.borrow_A ? 0 : (uint64_t)sizes.A_slice_entries * sizes.A_entry_bytes) +
        (sizes.borrow_B ? 0 : (uint64_t)sizes.B_slice_entries * sizes.B_entry_bytes) +
        (sizes.borrow_C ? 0 : (uint64_t)sizes.C_slice_entries * sizes.C_entry_bytes) +
        (sizes.alias_C_partials_onto_C_slice ? 0 : (uint64_t)sizes.C_partials_entries * sizes.C_partials_entry_bytes);
    return sizes;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

UnifiedMatmulPlan plan_unified_matmul(
    tt::tt_metal::IDevice& device,
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    UnifiedMatmulPlan plan{};

    TT_FATAL(
        !attributes.transpose_a && !attributes.transpose_b,
        "MatmulUnifiedProgramConfig: transposes are applied to the operands before the op, not in the kernels");
    TT_FATAL(!attributes.untilize_out, "MatmulUnifiedProgramConfig does not support untilize_out");
    TT_FATAL(attributes.bcast_batch.has_value(), "bcast_batch should have been populated");
    TT_FATAL(attributes.compute_kernel_config.has_value(), "compute_kernel_config should have been populated");
    TT_FATAL(attributes.output_dtype.has_value(), "output_dtype should have been populated");

    const tt::tt_metal::Shape& A_shape = A.padded_shape();
    const tt::tt_metal::Shape& B_shape = B.padded_shape();
    const tt::tt_metal::Tile A_tile = A.tensor_spec().tile();
    const tt::tt_metal::Tile B_tile = B.tensor_spec().tile();
    TT_FATAL(
        A_tile.get_height() == TILE_HEIGHT && A_tile.get_width() == TILE_WIDTH && B_tile.get_height() == TILE_HEIGHT &&
            B_tile.get_width() == TILE_WIDTH,
        "MatmulUnifiedProgramConfig supports 32x32 tiles only (A {}x{}, B {}x{})",
        A_tile.get_height(),
        A_tile.get_width(),
        B_tile.get_height(),
        B_tile.get_width());
    if (attributes.output_tile.has_value()) {
        TT_FATAL(
            attributes.output_tile->get_tile_shape()[0] == TILE_HEIGHT &&
                attributes.output_tile->get_tile_shape()[1] == TILE_WIDTH,
            "MatmulUnifiedProgramConfig supports a 32x32 output tile only");
    }

    // ---- GEMM size ----
    plan.batch_size = get_batch_size(A_shape);
    plan.M_tiles = A_shape[-2] / TILE_HEIGHT;
    plan.K_tiles = A_shape[-1] / TILE_WIDTH;
    plan.N_tiles = B_shape[-1] / TILE_WIDTH;
    const bool B_has_one_batch = attributes.bcast_batch.value();  // the op sets bcast_batch iff batch_size(B) == 1
    plan.B_batch_stride_tiles = B_has_one_batch ? 0 : plan.K_tiles * plan.N_tiles;
    TT_FATAL(
        B_has_one_batch || get_batch_size(B_shape) == plan.batch_size,
        "Batched B must match A's batch ({} vs {})",
        get_batch_size(B_shape),
        plan.batch_size);

    // ---- C slice assignment ----
    TT_FATAL(
        config.C_slice_M_tiles > 0 && config.C_slice_N_tiles > 0, "C_slice_M_tiles and C_slice_N_tiles must be > 0");
    plan.C_slice_M_tiles = config.C_slice_M_tiles;
    plan.C_slice_N_tiles = config.C_slice_N_tiles;
    // The C slices of one batch, walked across N then down M; every core produces its C slices for all batches.
    const uint32_t C_slices_across_N = tt::div_up(plan.N_tiles, plan.C_slice_N_tiles);
    const uint32_t C_slices_down_M = tt::div_up(plan.M_tiles, plan.C_slice_M_tiles);
    plan.C_slices_per_batch = C_slices_down_M * C_slices_across_N;
    plan.sharded_output_layout = C_slices_across_N == 1 ? tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED
                                 : C_slices_down_M == 1 ? tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED
                                                        : tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;

    TT_FATAL(config.cores.num_cores() > 0, "MatmulUnifiedProgramConfig.cores is empty");
    const CoreCoord grid = device.compute_with_storage_grid_size();
    const CoreRange bounding_box = config.cores.bounding_box();
    TT_FATAL(
        bounding_box.end_coord.x < grid.x && bounding_box.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    plan.row_major_cores = config.row_major_cores;
    const std::vector<CoreCoord> all_cores = corerange_to_cores(config.cores, std::nullopt, config.row_major_cores);
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), plan.C_slices_per_batch);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    plan.max_C_slices_per_core = tt::div_up(plan.C_slices_per_batch, num_active);

    // ---- Subblock: the C slice's tiles accumulated in DST at once ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(device.arch(), attributes.compute_kernel_config.value()));
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;

    plan.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    plan.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    plan.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    const uint32_t l1_base = device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t l1_ceiling = device.lowest_occupied_compute_l1_address().value_or(device.l1_size_per_core());
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    const uint64_t l1_budget = l1_ceiling - l1_base;

    // ---- Borrowing prerequisites that do not depend on the subblock ----
    // A shard matches when the tensor is L1-sharded with that shard shape and its grid lists the active
    // cores in assignment order (so shard i lives on the core that produces C slice i).
    auto shard_matches = [&](const ttnn::Tensor& tensor, uint32_t shard_M_tiles, uint32_t shard_N_tiles) {
        if (!tensor.is_sharded() || tensor.memory_config().buffer_type() != tt::tt_metal::BufferType::L1) {
            return false;
        }
        const tt::tt_metal::ShardSpec& shard = tensor.shard_spec().value();
        if (shard.shape[0] != shard_M_tiles * TILE_HEIGHT || shard.shape[1] != shard_N_tiles * TILE_WIDTH) {
            return false;
        }
        const std::vector<CoreCoord> shard_cores =
            corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
        return shard_cores == plan.cores;
    };
    const bool one_C_slice_per_core_no_batch = plan.batch_size == 1 && plan.C_slices_per_batch == plan.cores.size();
    const uint32_t A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;
    // A: the C slice's rows for all of K; C slices must span N so no two cores need the same rows. The copy path
    // zeroes A's K padding in the DFB; a borrowed shard is never written, so K must be a tile multiple.
    const bool A_shard_borrowable =
        one_C_slice_per_core_no_batch && C_slices_across_N == 1 && A_last_K_tile_valid_columns == 0 &&
        shard_matches(A, plan.C_slice_M_tiles, plan.K_tiles) &&
        (uint64_t)plan.C_slice_M_tiles * plan.K_tiles * tt::tile_size(plan.A_format) <= MAX_DFB_EXTENT_BYTES;
    // B: the C slice's columns for all of K; C slices must span M.
    const bool B_shard_borrowable =
        one_C_slice_per_core_no_batch && C_slices_down_M == 1 && shard_matches(B, plan.K_tiles, plan.C_slice_N_tiles) &&
        (uint64_t)plan.K_tiles * plan.C_slice_N_tiles * tt::tile_size(plan.B_format) <= MAX_DFB_EXTENT_BYTES;
    // C: packed straight into the shard when subblock-major pack order equals the shard's row-major order.
    const bool C_shard_matches = output.has_value()
                                     ? shard_matches(output.value(), plan.C_slice_M_tiles, plan.C_slice_N_tiles)
                                     : (attributes.output_mem_config.is_sharded() &&
                                        attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1);
    const bool C_shard_borrowable = C_shard_matches && one_C_slice_per_core_no_batch;

    // A candidate is viable when it voids no achievable borrow and its DFBs fit L1, sized at the K
    // chunk the search below bottoms out at, so an accepted candidate is guaranteed to fit.
    const uint32_t K_chunk_floor = config.K_chunk_tiles == 0 ? 1 : config.K_chunk_tiles;
    const auto subblock_viable = [&](uint32_t subblock_M_tiles, uint32_t subblock_N_tiles) {
        UnifiedMatmulPlan candidate = plan;
        candidate.C_slice_M_padded_tiles = tt::round_up(plan.C_slice_M_tiles, subblock_M_tiles);
        candidate.C_slice_N_padded_tiles = tt::round_up(plan.C_slice_N_tiles, subblock_N_tiles);
        const bool M_padded = candidate.C_slice_M_padded_tiles != plan.C_slice_M_tiles;
        const bool N_padded = candidate.C_slice_N_padded_tiles != plan.C_slice_N_tiles;
        // A borrowed shard holds only the true slice dims, so a candidate that voids an achievable
        // borrow is rejected outright: never trade a borrow for subblock volume. C's borrow needs
        // subblocks spanning the C slice width, achievable only when that width fits DST.
        const bool C_borrow_achievable = C_shard_borrowable && plan.C_slice_N_tiles <= dst_capacity_tiles;
        const bool C_borrow_kept = subblock_N_tiles == plan.C_slice_N_tiles && !M_padded;
        const bool voids_A_borrow = A_shard_borrowable && M_padded;
        const bool voids_B_borrow = B_shard_borrowable && N_padded;
        const bool voids_C_borrow = C_borrow_achievable && !C_borrow_kept;
        if (voids_A_borrow || voids_B_borrow || voids_C_borrow) {
            return false;
        }
        return size_dfbs(
                   candidate,
                   K_chunk_floor,
                   fp32_dest_acc_en,
                   packer_l1_acc,
                   /*A_borrowable=*/A_shard_borrowable,
                   /*B_borrowable=*/B_shard_borrowable,
                   /*C_borrowable=*/C_shard_borrowable && C_borrow_kept)
            .fits(l1_budget);
    };

    TT_FATAL(
        (config.subblock_M_tiles == 0) == (config.subblock_N_tiles == 0),
        "subblock_M_tiles and subblock_N_tiles must both be set or both be 0 (auto), got {}x{}",
        config.subblock_M_tiles,
        config.subblock_N_tiles);
    if (config.subblock_M_tiles != 0) {
        plan.subblock_M_tiles = config.subblock_M_tiles;
        plan.subblock_N_tiles = config.subblock_N_tiles;
    } else {
        std::tie(plan.subblock_M_tiles, plan.subblock_N_tiles) =
            maximize_subblock_size(plan.C_slice_M_tiles, plan.C_slice_N_tiles, dst_capacity_tiles, subblock_viable);
    }
    plan.C_slice_M_padded_tiles = tt::round_up(plan.C_slice_M_tiles, plan.subblock_M_tiles);
    plan.C_slice_N_padded_tiles = tt::round_up(plan.C_slice_N_tiles, plan.subblock_N_tiles);
    TT_FATAL(
        plan.subblock_M_tiles * plan.subblock_N_tiles <= dst_capacity_tiles,
        "subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.subblock_M_tiles * plan.subblock_N_tiles,
        dst_capacity_tiles,
        fp32_dest_acc_en);
    // The padded-dim conditions are redundant for auto subblocks (the viability filter enforced them)
    // but load-bearing for explicit ones, which may pad a borrowable operand and must take the copy path.
    const bool A_borrowable = A_shard_borrowable && plan.C_slice_M_padded_tiles == plan.C_slice_M_tiles;
    const bool B_borrowable = B_shard_borrowable && plan.C_slice_N_padded_tiles == plan.C_slice_N_tiles;
    const bool C_borrowable = C_shard_borrowable && plan.subblock_N_tiles == plan.C_slice_N_tiles &&
                              plan.C_slice_M_padded_tiles == plan.C_slice_M_tiles;

    // ---- K chunk and DFB sizing ----
    std::optional<DfbSizes> chosen;
    if (config.K_chunk_tiles == 0) {
        // A resident A shard is only borrowable with a single K chunk, so try that first (main pins
        // in0_block_w == K for height-sharded in0 for the same reason).
        if (A_borrowable) {
            const DfbSizes candidate = size_dfbs(
                plan, plan.K_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
            if (candidate.borrow_A && candidate.fits(l1_budget)) {
                chosen = candidate;
            }
        }
        // Otherwise the largest divisor of K_tiles (capped) whose DFBs fit; 1 is the floor and must fit.
        for (uint32_t K_chunk_tiles = std::min<uint32_t>(plan.K_tiles, MAX_AUTO_K_CHUNK_TILES);
             !chosen.has_value() && K_chunk_tiles >= 1;
             --K_chunk_tiles) {
            if (plan.K_tiles % K_chunk_tiles != 0) {
                continue;
            }
            const DfbSizes candidate = size_dfbs(
                plan, K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
            if (candidate.fits(l1_budget)) {
                chosen = candidate;
            }
        }
        TT_FATAL(
            chosen.has_value(),
            "MatmulUnifiedProgramConfig: a {}x{}-tile C slice does not fit L1 even with K_chunk_tiles=1 "
            "(budget {} B, max DFB extent {} B); shrink C_slice_M_tiles / C_slice_N_tiles",
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles,
            l1_budget,
            MAX_DFB_EXTENT_BYTES);
    } else {
        TT_FATAL(
            plan.K_tiles % config.K_chunk_tiles == 0,
            "K_chunk_tiles ({}) must divide K_tiles ({})",
            config.K_chunk_tiles,
            plan.K_tiles);
        chosen = size_dfbs(
            plan, config.K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
        TT_FATAL(
            chosen->fits(l1_budget),
            "MatmulUnifiedProgramConfig: DFBs for a {}x{}-tile C slice with K_chunk_tiles={} do not fit "
            "(needs {} B, budget {} B, max DFB extent {} B: A slice {} B, B slice {} B, C slice {} B, C partials {} B)",
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles,
            chosen->K_chunk_tiles,
            chosen->l1_bytes,
            l1_budget,
            MAX_DFB_EXTENT_BYTES,
            (uint64_t)chosen->A_slice_entries * chosen->A_entry_bytes,
            (uint64_t)chosen->B_slice_entries * chosen->B_entry_bytes,
            (uint64_t)chosen->C_slice_entries * chosen->C_entry_bytes,
            (uint64_t)chosen->C_partials_entries * chosen->C_partials_entry_bytes);
    }
    // The plan takes the winning candidate exactly once; nothing downstream adjusts it.
    plan.K_chunk_tiles = chosen->K_chunk_tiles;
    plan.num_K_chunks = chosen->num_K_chunks;
    plan.packer_l1_acc_en = chosen->packer_l1_acc_en;
    plan.C_partials_format = chosen->C_partials_format;
    plan.A_entry_bytes = chosen->A_entry_bytes;
    plan.B_entry_bytes = chosen->B_entry_bytes;
    plan.C_entry_bytes = chosen->C_entry_bytes;
    plan.C_partials_entry_bytes = chosen->C_partials_entry_bytes;
    plan.A_slice_entries = chosen->A_slice_entries;
    plan.B_slice_entries = chosen->B_slice_entries;
    plan.C_slice_entries = chosen->C_slice_entries;
    plan.C_partials_entries = chosen->C_partials_entries;
    plan.alias_C_partials_onto_C_slice = chosen->alias_C_partials_onto_C_slice;
    plan.borrow_A = chosen->borrow_A;
    plan.borrow_B = chosen->borrow_B;
    plan.borrow_C = chosen->borrow_C;
    plan.l1_bytes = chosen->l1_bytes;

    // ---- Sharded output: one C slice per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(plan.batch_size == 1, "Sharded output needs batch 1 (a core's C slices would not form one shard)");
        TT_FATAL(
            plan.C_slices_per_batch == plan.cores.size(),
            "Sharded output needs exactly one C slice per core ({} C slices, {} active cores)",
            plan.C_slices_per_batch,
            plan.cores.size());
        if (plan.sharded_output_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const std::vector<CoreRange>& ranges = config.cores.ranges();
            const bool one_rectangle = ranges.size() == 1;
            const uint32_t rectangle_columns = one_rectangle ? ranges[0].grid_size().x : 0;
            const uint32_t rectangle_rows = one_rectangle ? ranges[0].grid_size().y : 0;
            TT_FATAL(
                one_rectangle && rectangle_columns == C_slices_across_N && rectangle_rows == C_slices_down_M,
                "Block-sharded output needs cores to be one {}x{} rectangle (one core per C slice, laid out as "
                "the C slices tile C), got {}",
                C_slices_across_N,
                C_slices_down_M,
                config.cores.str());
        }
    }
    return plan;
}

ttnn::device_operation::ProgramArtifacts MatmulUnifiedProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    TT_FATAL(
        tensor_args.optional_input_tensors.empty() || !tensor_args.optional_input_tensors[0].has_value(),
        "MatmulUnifiedProgramConfig does not fuse bias; the op applies it as a separate add");
    TT_FATAL(
        operation_attributes.program_config.has_value() &&
            std::holds_alternative<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig>(
                operation_attributes.program_config.value()),
        "MatmulUnifiedProgramFactory needs a MatmulUnifiedProgramConfig");
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config =
        std::get<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig>(
            operation_attributes.program_config.value());

    const ttnn::Tensor& A_tensor = tensor_args.input_tensors.at(0);
    const ttnn::Tensor& B_tensor = tensor_args.input_tensors.at(1);
    const tt::tt_metal::MeshTensor& A = A_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& B = B_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& C = tensor_return_value.at(0).mesh_tensor();
    tt::tt_metal::IDevice* device = &A.mutable_device();

    const UnifiedMatmulPlan plan =
        plan_unified_matmul(*device, A_tensor, B_tensor, config, operation_attributes, tensor_return_value.at(0));

    // ---- Tensor parameters: the kernels' tensor accessors are generated from these specs ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = A_TENSOR, .spec = A.tensor_spec()},
        TensorParameter{.unique_id = B_TENSOR, .spec = B.tensor_spec()},
        TensorParameter{.unique_id = C_TENSOR, .spec = C.tensor_spec()},
    };

    // ---- Dataflow buffers ----
    const tt::tt_metal::Tile C_tile = C.tensor_spec().tile();
    log_debug(
        tt::LogOp,
        "MatmulUnifiedProgramConfig: borrow A={} B={} C={} (C slice {}x{}, subblock {}x{}, K chunk {} of {} tiles)",
        plan.borrow_A,
        plan.borrow_B,
        plan.borrow_C,
        plan.C_slice_M_tiles,
        plan.C_slice_N_tiles,
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.K_chunk_tiles,
        plan.K_tiles);
    if (C.is_sharded() && !plan.borrow_C) {
        log_warning(
            tt::LogOp,
            "MatmulUnifiedProgramConfig: sharded C is copied by the writer instead of packed in place. Packing in "
            "place needs an L1 shard grid equal to the active cores, batch 1, one C slice per core and "
            "subblock_N_tiles == C_slice_N_tiles (subblock {}x{} for a {}x{} C slice).",
            plan.subblock_M_tiles,
            plan.subblock_N_tiles,
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles);
    }
    Group<DataflowBufferSpec> dataflow_buffers;
    {
        DataflowBufferSpec A_slice_dfb{
            .unique_id = A_SLICE_DFB,
            .entry_size = plan.A_entry_bytes,
            .num_entries = plan.A_slice_entries,
            .data_format_metadata = plan.A_format,
            .tile_format_metadata = A.tensor_spec().tile(),
        };
        if (plan.borrow_A) {
            A_slice_dfb.borrowed_from = A_TENSOR;  // the resident A shard is the DFB
        }
        DataflowBufferSpec B_slice_dfb{
            .unique_id = B_SLICE_DFB,
            .entry_size = plan.B_entry_bytes,
            .num_entries = plan.B_slice_entries,
            .data_format_metadata = plan.B_format,
            .tile_format_metadata = B.tensor_spec().tile(),
        };
        if (plan.borrow_B) {
            B_slice_dfb.borrowed_from = B_TENSOR;  // the resident B shard is the DFB
        }
        DataflowBufferSpec C_slice_dfb{
            .unique_id = C_SLICE_DFB,
            .entry_size = plan.C_entry_bytes,
            .num_entries = plan.C_slice_entries,
            .data_format_metadata = plan.C_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.borrow_C) {
            C_slice_dfb.borrowed_from = C_TENSOR;  // finished tiles are packed straight into the C shard
        }
        DataflowBufferSpec C_partials_dfb{
            .unique_id = C_PARTIALS_DFB,
            .entry_size = plan.C_partials_entry_bytes,
            .num_entries = plan.C_partials_entries,
            .data_format_metadata = plan.C_partials_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.alias_C_partials_onto_C_slice) {
            C_slice_dfb.advanced_options.alias_with = {C_PARTIALS_DFB};
            C_partials_dfb.advanced_options.alias_with = {C_SLICE_DFB};
            if (plan.borrow_C) {
                C_partials_dfb.borrowed_from = C_TENSOR;  // partials accumulate in the shard too
            }
        }
        dataflow_buffers.push_back(std::move(A_slice_dfb));
        dataflow_buffers.push_back(std::move(B_slice_dfb));
        dataflow_buffers.push_back(std::move(C_slice_dfb));
        dataflow_buffers.push_back(std::move(C_partials_dfb));
    }

    // ---- Reader ----
    const uint32_t A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_reader.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ProducerOf(A_SLICE_DFB, "A_slice"), ProducerOf(B_SLICE_DFB, "B_slice")},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A_TENSOR, .accessor_name = "A"},
                TensorBinding{.tensor_parameter_name = B_TENSOR, .accessor_name = "B"},
            },
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"K_tiles", plan.K_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"B_batch_stride_tiles", plan.B_batch_stride_tiles},
                {"C_slice_M_tiles", plan.C_slice_M_tiles},
                {"C_slice_N_tiles", plan.C_slice_N_tiles},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"K_chunk_tiles", plan.K_chunk_tiles},
                {"num_K_chunks", plan.num_K_chunks},
                {"A_last_K_tile_valid_columns", A_last_K_tile_valid_columns},
                {"A_borrowed", plan.borrow_A ? 1u : 0u},
                {"B_borrowed", plan.borrow_B ? 1u : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"C_slice_first_M_tile", "C_slice_first_N_tile", "num_C_slices"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_writer.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ConsumerOf(C_SLICE_DFB, "C_slice")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_TENSOR, .accessor_name = "C"}},
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"C_slice_M_tiles", plan.C_slice_M_tiles},
                {"C_slice_N_tiles", plan.C_slice_N_tiles},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
                {"C_borrowed", plan.borrow_C ? 1u : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"C_slice_first_M_tile", "C_slice_first_N_tile", "num_C_slices"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Compute ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> compute_defines_map;  // throttle / stagger only
    const ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), plan.cores.size(), compute_defines_map);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), plan.cores.size(), compute_defines_map, throttle_level);
    KernelSpec::CompilerOptions::Defines compute_defines(compute_defines_map);

    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());
    if (fp32_dest_acc_en) {
        // With a 32-bit DST every consumed 32-bit DFB needs an explicit unpack mode. The partials are
        // reloaded with a data copy into DST, so unpack them straight to DST and keep fp32 precision;
        // fp32 operands feed the FPU and go through SrcA/SrcB.
        auto is_32bit = [](tt::DataFormat format) {
            return format == tt::DataFormat::Float32 || format == tt::DataFormat::Int32 ||
                   format == tt::DataFormat::UInt32;
        };
        ComputeUnpackModes& modes = unpack_modes(compute_hw_config);
        if (is_32bit(plan.C_partials_format)) {
            modes.emplace(C_PARTIALS_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
        }
        if (is_32bit(plan.A_format)) {
            modes.emplace(A_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (is_32bit(plan.B_format)) {
            modes.emplace(B_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "compute/unified_matmul_compute.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {
                ConsumerOf(A_SLICE_DFB, "A_slice"),
                ConsumerOf(B_SLICE_DFB, "B_slice"),
                ProducerOf(C_SLICE_DFB, "C_slice"),
                ProducerOf(C_PARTIALS_DFB, "C_partials"),
                ConsumerOf(C_PARTIALS_DFB, "C_partials"),
            },
        .compile_time_args =
            {
                {"batch_size", plan.batch_size},
                {"K_chunk_tiles", plan.K_chunk_tiles},
                {"num_K_chunks", plan.num_K_chunks},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
                {"packer_l1_acc", plan.packer_l1_acc_en ? 1u : 0u},
                {"partials_format_differs", plan.C_partials_format != plan.C_format ? 1u : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_C_slices"}},
        .hw_config = compute_hw_config,
    };

    // ---- One work unit over the active cores ----
    const CoreRangeSet active_cores(ttsl::Span<const CoreCoord>(plan.cores));
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "unified_matmul",
        .kernels = {READER_KERNEL, COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = active_cores,
    }};

    // ---- Per-core runtime args: where each core's run of C slices starts and how long it is ----
    // Each active core takes a contiguous run of the walk, the first (C_slices_per_batch % cores) cores
    // one C slice longer. A core's start is in tile coordinates so the kernels only ever step it by
    // C_slice_M_tiles / C_slice_N_tiles.
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    const uint32_t num_active_cores = plan.cores.size();
    const uint32_t C_slices_across_N = tt::div_up(plan.N_tiles, plan.C_slice_N_tiles);
    const uint32_t C_slices_per_core_floor = plan.C_slices_per_batch / num_active_cores;
    const uint32_t cores_with_extra_C_slice = plan.C_slices_per_batch % num_active_cores;
    uint32_t next_C_slice = 0;  // position in the walk of the next unassigned C slice
    for (uint32_t core = 0; core < num_active_cores; ++core) {
        const uint32_t num_C_slices = C_slices_per_core_floor + (core < cores_with_extra_C_slice ? 1 : 0);
        const std::initializer_list<std::pair<std::string, uint32_t>> run_start = {
            {"C_slice_first_M_tile", (next_C_slice / C_slices_across_N) * plan.C_slice_M_tiles},
            {"C_slice_first_N_tile", (next_C_slice % C_slices_across_N) * plan.C_slice_N_tiles},
            {"num_C_slices", num_C_slices}};
        next_C_slice += num_C_slices;
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, plan.cores[core], {{"num_C_slices", num_C_slices}});
    }

    ProgramSpec spec{
        .name = "matmul_unified",
        .kernels = {reader, compute, writer},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)},
        .tensor_args = {{A_TENSOR, A}, {B_TENSOR, B}, {C_TENSOR, C}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
