// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory (GH#41910): one Metal 2.0 program for every placement.
// The config names the cores and the C slice per core; this file turns that into a C slice assignment,
// four DFB rings (A slice, B slice, C slice, C partials), one reader, one compute kernel and one
// writer. Nothing here depends on operand memory layout: the kernels address tiles by tile index
// through the tensor accessor.

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

// A DFB touched by a TRISC keeps its ring extent in a uint16_t of 16-byte units (see
// validate_ring_extent in dataflow_buffer.cpp). Enforced on every arch so a config that is legal on
// Wormhole never becomes a program-creation FATAL on Quasar.
constexpr uint64_t MAX_DFB_RING_BYTES = 65535ull * 16ull;
constexpr uint32_t MAX_AUTO_K_CHUNK_TILES = 8;

uint64_t l1_budget_bytes(tt::tt_metal::IDevice* device) {
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const std::optional<tt::tt_metal::DeviceAddr> lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(lowest_occupied.value()) : device->l1_size_per_core();
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    return l1_ceiling - l1_base;
}

// Fills the ring sizing of `plan` for a given K chunk. Returns the total footprint in bytes.
struct Borrowable {
    bool A = false;
    bool B = false;
    bool C = false;
};

// Everything about the rings that follows from one candidate K chunk. Pure: the plan is only read, so the
// K chunk search can size several candidates and the plan is written exactly once with the winner.
struct RingSizes {
    uint32_t K_chunk_tiles = 0;
    uint32_t num_K_chunks = 0;
    bool packer_l1_acc_en = false;
    tt::DataFormat C_partials_format{};
    uint32_t A_slot_bytes = 0;
    uint32_t B_slot_bytes = 0;
    uint32_t C_slot_bytes = 0;
    uint32_t C_partials_slot_bytes = 0;
    uint32_t A_slice_ring_slots = 0;
    uint32_t B_slice_ring_slots = 0;
    uint32_t C_slice_ring_slots = 0;
    uint32_t C_partials_ring_slots = 0;
    bool alias_C_partials_onto_C_slice = false;
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;
    uint64_t l1_bytes = 0;

    bool fits(uint64_t l1_budget) const {
        const uint64_t rings[] = {
            (uint64_t)A_slice_ring_slots * A_slot_bytes,
            (uint64_t)B_slice_ring_slots * B_slot_bytes,
            (uint64_t)C_slice_ring_slots * C_slot_bytes,
            (uint64_t)C_partials_ring_slots * C_partials_slot_bytes};
        for (uint64_t ring_bytes : rings) {
            if (ring_bytes > MAX_DFB_RING_BYTES) {
                return false;
            }
        }
        return l1_bytes <= l1_budget;
    }
};

RingSizes size_rings(
    const UnifiedMatmulPlan& plan,
    uint32_t K_chunk_tiles,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    const Borrowable& borrowable) {
    RingSizes r;
    r.K_chunk_tiles = K_chunk_tiles;
    r.num_K_chunks = plan.K_tiles / K_chunk_tiles;

    // The packer accumulates partials in L1 only when there are enough K chunks for the reconfig overhead
    // to pay off (the last K chunk spills and reloads either way, so more than two).
    r.packer_l1_acc_en = packer_l1_acc && r.num_K_chunks > 2;
    r.C_partials_format = r.packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : plan.C_format);
    r.C_slot_bytes = tt::tile_size(plan.C_format);
    r.C_partials_slot_bytes = tt::tile_size(r.C_partials_format);

    const uint32_t C_slice_tiles = plan.C_slice_M_tiles * plan.C_slice_N_tiles;
    r.C_slice_ring_slots = C_slice_tiles;
    r.C_partials_ring_slots = C_slice_tiles;

    // One ring slot per tile for every ring (a 32x32 tile of every format is a multiple of the L1 alignment).
    // Copied operands: double-buffer whenever more than one slice passes through. Borrowed operands: the
    // ring is the resident shard itself, one slot per shard tile.
    // A can only be borrowed when the single K chunk covers all of K; a shard too large for a TRISC ring
    // takes the copy path.
    const bool more_than_one_slice = (uint64_t)plan.batch_size * plan.max_C_slices_per_core * r.num_K_chunks > 1;
    const uint32_t slice_ring_depth = more_than_one_slice ? 2 : 1;
    r.borrow_A = borrowable.A && r.num_K_chunks == 1 &&
                 (uint64_t)plan.C_slice_M_tiles * plan.K_tiles * tt::tile_size(plan.A_format) <= MAX_DFB_RING_BYTES;
    r.borrow_B = borrowable.B &&
                 (uint64_t)plan.K_tiles * plan.C_slice_N_tiles * tt::tile_size(plan.B_format) <= MAX_DFB_RING_BYTES;
    r.borrow_C = borrowable.C;
    r.A_slot_bytes = tt::tile_size(plan.A_format);
    r.B_slot_bytes = tt::tile_size(plan.B_format);
    r.A_slice_ring_slots =
        r.borrow_A ? plan.C_slice_M_tiles * plan.K_tiles : plan.C_slice_M_tiles * K_chunk_tiles * slice_ring_depth;
    r.B_slice_ring_slots =
        r.borrow_B ? plan.K_tiles * plan.C_slice_N_tiles : K_chunk_tiles * plan.C_slice_N_tiles * slice_ring_depth;

    // Aliasing C_partials onto C_slice when a core produces more than one C slice (counting batches) is a
    // race: the writer may still be draining C slice i from C_slice while the compute packs C slice i+1's first
    // partials into the same bytes. Alias only when the partials can never be live while C_slice holds
    // unread data: a single C slice per core, or no partials at all (one K chunk).
    const bool partials_ever_written = r.num_K_chunks > 1;
    const bool one_C_slice_per_core = plan.batch_size == 1 && plan.max_C_slices_per_core == 1;
    r.alias_C_partials_onto_C_slice =
        (r.C_partials_format == plan.C_format) && (!partials_ever_written || one_C_slice_per_core);

    // Borrowed rings are the tensors' own memory and cost nothing here.
    r.l1_bytes = (r.borrow_A ? 0 : (uint64_t)r.A_slice_ring_slots * r.A_slot_bytes) +
                 (r.borrow_B ? 0 : (uint64_t)r.B_slice_ring_slots * r.B_slot_bytes) +
                 (r.borrow_C ? 0 : (uint64_t)r.C_slice_ring_slots * r.C_slot_bytes) +
                 (r.alias_C_partials_onto_C_slice ? 0 : (uint64_t)r.C_partials_ring_slots * r.C_partials_slot_bytes);
    return r;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::TensorMemoryLayout UnifiedMatmulPlan::sharded_output_layout() const {
    if (C_slice_N_tiles >= N_tiles) {  // a C slice spans all of N: C slices are stacked down M
        return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    }
    if (C_slice_M_tiles >= M_tiles) {  // a C slice spans all of M: C slices sit side by side across N
        return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
    }
    return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
}

UnifiedMatmulPlan plan_unified_matmul(
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

    TT_FATAL(config.cores.num_cores() > 0, "MatmulUnifiedProgramConfig.cores is empty");
    const CoreCoord grid = A.device()->compute_with_storage_grid_size();
    const CoreRange bounding_box = config.cores.bounding_box();
    TT_FATAL(
        bounding_box.end_coord.x < grid.x && bounding_box.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    plan.row_major_cores = config.row_major_cores;
    const std::vector<CoreCoord> all_cores = corerange_to_cores(config.cores, std::nullopt, config.row_major_cores);
    // Each active core takes a contiguous run of the walk, the first (C_slices_per_batch % num_active) cores
    // one C slice longer. A core's start is expressed in tile coordinates so the kernels only ever step it by
    // C_slice_M_tiles / C_slice_N_tiles.
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), plan.C_slices_per_batch);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    const uint32_t C_slices_per_core_floor = plan.C_slices_per_batch / num_active;
    const uint32_t cores_with_extra_C_slice = plan.C_slices_per_batch % num_active;
    plan.C_slice_first_M_tile.resize(num_active);
    plan.C_slice_first_N_tile.resize(num_active);
    plan.num_C_slices.resize(num_active);
    uint32_t next_C_slice = 0;  // position in the walk of the next unassigned C slice
    for (uint32_t core = 0; core < num_active; ++core) {
        plan.num_C_slices[core] = C_slices_per_core_floor + (core < cores_with_extra_C_slice ? 1 : 0);
        plan.C_slice_first_M_tile[core] = (next_C_slice / C_slices_across_N) * plan.C_slice_M_tiles;
        plan.C_slice_first_N_tile[core] = (next_C_slice % C_slices_across_N) * plan.C_slice_N_tiles;
        next_C_slice += plan.num_C_slices[core];
    }
    plan.max_C_slices_per_core = plan.num_C_slices.front();

    // ---- Subblock: the C slice's tiles accumulated in DST at once ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    if (config.subblock_M_tiles == 0 && config.subblock_N_tiles == 0) {
        // The chooser's (h, w) is (M tiles, N tiles) of the subblock.
        const std::tuple<uint32_t, uint32_t> subblock =
            operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
                plan.C_slice_M_tiles,
                plan.C_slice_N_tiles,
                /*per_core_M_equals_subblock_h_constraint=*/false,
                // A sharded C is packed straight into the shard when subblocks span the C slice width, so ask
                // for that when it can fit DST; the chooser falls back to 1x1 otherwise.
                /*per_core_N_equals_subblock_w_constraint=*/attributes.output_mem_config.is_sharded() &&
                    plan.C_slice_N_tiles <= (fp32_dest_acc_en ? 4u : 8u),
                fp32_dest_acc_en);
        plan.subblock_M_tiles = std::get<0>(subblock);
        plan.subblock_N_tiles = std::get<1>(subblock);
    } else {
        TT_FATAL(
            config.subblock_M_tiles > 0 && config.subblock_N_tiles > 0,
            "subblock_M_tiles and subblock_N_tiles must both be set or both be 0 (auto)");
        plan.subblock_M_tiles = config.subblock_M_tiles;
        plan.subblock_N_tiles = config.subblock_N_tiles;
    }
    TT_FATAL(
        plan.C_slice_M_tiles % plan.subblock_M_tiles == 0 && plan.C_slice_N_tiles % plan.subblock_N_tiles == 0,
        "subblock {}x{} must divide the per-core C slice {}x{}",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.C_slice_M_tiles,
        plan.C_slice_N_tiles);
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        plan.subblock_M_tiles * plan.subblock_N_tiles <= dst_capacity_tiles,
        "subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.subblock_M_tiles * plan.subblock_N_tiles,
        dst_capacity_tiles,
        fp32_dest_acc_en);

    // ---- Borrowing: which operands are already sitting in L1 exactly as the rings would hold them ----
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
    Borrowable borrowable;
    // A: the C slice's rows for all of K; C slices must span N so no two cores need the same rows. The copy path
    // zeroes A's K padding in the ring; a borrowed shard is never written, so K must be a tile multiple.
    borrowable.A = one_C_slice_per_core_no_batch && C_slices_across_N == 1 && A_last_K_tile_valid_columns == 0 &&
                   shard_matches(A, plan.C_slice_M_tiles, plan.K_tiles);
    // B: the C slice's columns for all of K; C slices must span M.
    borrowable.B =
        one_C_slice_per_core_no_batch && C_slices_down_M == 1 && shard_matches(B, plan.K_tiles, plan.C_slice_N_tiles);
    // C: packed straight into the shard when subblock-major pack order equals the shard's row-major order.
    // A caller-provided output must really be laid out as the plan assumes; one the op allocates from this plan
    // is, by construction.
    const bool C_shard_matches = output.has_value()
                                     ? shard_matches(output.value(), plan.C_slice_M_tiles, plan.C_slice_N_tiles)
                                     : (attributes.output_mem_config.is_sharded() &&
                                        attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1);
    borrowable.C = C_shard_matches && one_C_slice_per_core_no_batch && plan.subblock_N_tiles == plan.C_slice_N_tiles;

    // ---- Formats, K chunk and ring sizing ----
    plan.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    plan.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    plan.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    const uint64_t l1_budget = l1_budget_bytes(A.device());
    std::optional<RingSizes> chosen;
    if (config.K_chunk_tiles == 0) {
        // A resident A shard is only borrowable with a single K chunk, so try that first (main pins
        // in0_block_w == K for height-sharded in0 for the same reason).
        if (borrowable.A) {
            const RingSizes candidate = size_rings(plan, plan.K_tiles, fp32_dest_acc_en, packer_l1_acc, borrowable);
            if (candidate.borrow_A && candidate.fits(l1_budget)) {
                chosen = candidate;
            }
        }
        // Otherwise the largest divisor of K_tiles (capped) whose rings fit; 1 is the floor and must fit.
        for (uint32_t K_chunk_tiles = std::min<uint32_t>(plan.K_tiles, MAX_AUTO_K_CHUNK_TILES);
             !chosen.has_value() && K_chunk_tiles >= 1;
             --K_chunk_tiles) {
            if (plan.K_tiles % K_chunk_tiles != 0) {
                continue;
            }
            const RingSizes candidate = size_rings(plan, K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, borrowable);
            if (candidate.fits(l1_budget)) {
                chosen = candidate;
            }
        }
        TT_FATAL(
            chosen.has_value(),
            "MatmulUnifiedProgramConfig: a {}x{}-tile C slice does not fit L1 even with K_chunk_tiles=1 "
            "(budget {} B, max ring {} B); shrink C_slice_M_tiles / C_slice_N_tiles",
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles,
            l1_budget,
            MAX_DFB_RING_BYTES);
    } else {
        TT_FATAL(
            plan.K_tiles % config.K_chunk_tiles == 0,
            "K_chunk_tiles ({}) must divide K_tiles ({})",
            config.K_chunk_tiles,
            plan.K_tiles);
        chosen = size_rings(plan, config.K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, borrowable);
        TT_FATAL(
            chosen->fits(l1_budget),
            "MatmulUnifiedProgramConfig: rings for a {}x{}-tile C slice with K_chunk_tiles={} do not fit "
            "(needs {} B, budget {} B, max ring {} B: A slice {} B, B slice {} B, C slice {} B, C partials {} B)",
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles,
            chosen->K_chunk_tiles,
            chosen->l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES,
            (uint64_t)chosen->A_slice_ring_slots * chosen->A_slot_bytes,
            (uint64_t)chosen->B_slice_ring_slots * chosen->B_slot_bytes,
            (uint64_t)chosen->C_slice_ring_slots * chosen->C_slot_bytes,
            (uint64_t)chosen->C_partials_ring_slots * chosen->C_partials_slot_bytes);
    }
    // The plan takes the winning candidate exactly once; nothing downstream adjusts it.
    plan.K_chunk_tiles = chosen->K_chunk_tiles;
    plan.num_K_chunks = chosen->num_K_chunks;
    plan.packer_l1_acc_en = chosen->packer_l1_acc_en;
    plan.C_partials_format = chosen->C_partials_format;
    plan.A_slot_bytes = chosen->A_slot_bytes;
    plan.B_slot_bytes = chosen->B_slot_bytes;
    plan.C_slot_bytes = chosen->C_slot_bytes;
    plan.C_partials_slot_bytes = chosen->C_partials_slot_bytes;
    plan.A_slice_ring_slots = chosen->A_slice_ring_slots;
    plan.B_slice_ring_slots = chosen->B_slice_ring_slots;
    plan.C_slice_ring_slots = chosen->C_slice_ring_slots;
    plan.C_partials_ring_slots = chosen->C_partials_ring_slots;
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
        if (plan.sharded_output_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
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
        plan_unified_matmul(A_tensor, B_tensor, config, operation_attributes, tensor_return_value.at(0));

    // ---- Tensor parameters: the kernels' tensor accessors are generated from these specs ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = A_TENSOR, .spec = A.tensor_spec()},
        TensorParameter{.unique_id = B_TENSOR, .spec = B.tensor_spec()},
        TensorParameter{.unique_id = C_TENSOR, .spec = C.tensor_spec()},
    };

    // ---- Dataflow-buffer rings ----
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
            .entry_size = plan.A_slot_bytes,
            .num_entries = plan.A_slice_ring_slots,
            .data_format_metadata = plan.A_format,
            .tile_format_metadata = A.tensor_spec().tile(),
        };
        if (plan.borrow_A) {
            A_slice_dfb.borrowed_from = A_TENSOR;  // the resident A shard is the ring
        }
        DataflowBufferSpec B_slice_dfb{
            .unique_id = B_SLICE_DFB,
            .entry_size = plan.B_slot_bytes,
            .num_entries = plan.B_slice_ring_slots,
            .data_format_metadata = plan.B_format,
            .tile_format_metadata = B.tensor_spec().tile(),
        };
        if (plan.borrow_B) {
            B_slice_dfb.borrowed_from = B_TENSOR;  // the resident B shard is the ring
        }
        DataflowBufferSpec C_slice_dfb{
            .unique_id = C_SLICE_DFB,
            .entry_size = plan.C_slot_bytes,
            .num_entries = plan.C_slice_ring_slots,
            .data_format_metadata = plan.C_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.borrow_C) {
            C_slice_dfb.borrowed_from = C_TENSOR;  // finished tiles are packed straight into the C shard
        }
        DataflowBufferSpec C_partials_dfb{
            .unique_id = C_PARTIALS_DFB,
            .entry_size = plan.C_partials_slot_bytes,
            .num_entries = plan.C_partials_ring_slots,
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
                {"C_slice_M_tiles", plan.C_slice_M_tiles},
                {"C_slice_N_tiles", plan.C_slice_N_tiles},
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
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    for (uint32_t core = 0; core < plan.cores.size(); ++core) {
        const std::initializer_list<std::pair<std::string, uint32_t>> run_start = {
            {"C_slice_first_M_tile", plan.C_slice_first_M_tile[core]},
            {"C_slice_first_N_tile", plan.C_slice_first_N_tile[core]},
            {"num_C_slices", plan.num_C_slices[core]}};
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, plan.cores[core], {{"num_C_slices", plan.num_C_slices[core]}});
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
