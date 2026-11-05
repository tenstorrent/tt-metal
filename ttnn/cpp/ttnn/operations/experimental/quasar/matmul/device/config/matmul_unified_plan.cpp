// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul (GH#41910) planning: plan_unified_matmul turns a MatmulUnifiedProgramConfig with a placement into
// everything the factory builds from, and choose_unified_placement picks the placement whose plan is fastest.

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_unified_plan.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::constants;
using ttnn::operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// A TRISC-visible DFB's extent is a uint16_t of 16-byte units (validate_ring_extent); enforced on
// every arch so a Wormhole-legal config never FATALs on Quasar.
constexpr uint64_t MAX_DFB_EXTENT_BYTES = 65535ull * 16ull;
// DM threads and buffering per core, fixed so the readers and writers keep the NoC busy rather than sized to the L1
// left over. Quasar runs four reader and two writer threads (of its six user DM cores); Wormhole / Blackhole one of
// each. The A and B DFBs hold a slice per reader thread and at least two; C_slice holds two C slices on cores that
// produce several.
constexpr uint32_t QUASAR_NUM_USER_DM_CORES = 6;
constexpr uint32_t QUASAR_NUM_WRITER_THREADS = 2;
constexpr uint32_t MIN_OPERAND_BUFFER_DEPTH = 2;
constexpr uint32_t C_BUFFER_DEPTH = 2;
static_assert(
    QUASAR_NUM_READER_THREADS + QUASAR_NUM_WRITER_THREADS <= QUASAR_NUM_USER_DM_CORES,
    "a Quasar cluster has six user DM cores");
static_assert(
    std::max(MIN_OPERAND_BUFFER_DEPTH, QUASAR_NUM_READER_THREADS) % QUASAR_NUM_READER_THREADS == 0,
    "each reader thread owns whole A and B slices");

// True when every sized DFB fits the extent cap and their total fits the L1 budget.
bool dfbs_fit(const MatmulUnifiedPlan& plan, uint64_t l1_budget) {
    const uint64_t largest_dfb_bytes = std::max(
        {(uint64_t)plan.A_slice_entries * plan.A_entry_bytes,
         (uint64_t)plan.B_slice_entries * plan.B_entry_bytes,
         (uint64_t)plan.C_slice_entries * plan.C_entry_bytes,
         (uint64_t)plan.C_partials_entries * plan.C_partials_entry_bytes,
         (uint64_t)plan.bias_entries * plan.bias_entry_bytes});
    return largest_dfb_bytes <= MAX_DFB_EXTENT_BYTES && plan.l1_bytes <= l1_budget;
}

// What the unified matmul does not do, or nullopt. The placement chooser leaves these cases to the legacy matmul; an
// explicit MatmulUnifiedProgramConfig FATALs with the reason. Every tile must be 32x32 (no tiny tiles); tensor dims
// need not be tile multiples.
std::optional<std::string> find_unsupported_feature(
    const ttnn::Tensor& A, const ttnn::Tensor& B, const MatmulParams& attributes) {
    if (attributes.transpose_a || attributes.transpose_b) {
        return "transposed operands (transpose them before the op)";
    }
    if (attributes.untilize_out) {
        return "untilize_out";
    }
    if (attributes.global_cb.has_value()) {
        return "a global circular buffer";
    }
    if (attributes.sub_device_id.has_value()) {
        return "a sub-device";
    }
    if (A.logical_shape().rank() < 2 || B.logical_shape().rank() < 2) {
        return "operands of rank below 2";
    }
    const tt::tt_metal::Tile::TileShape tile_32x32 = {TILE_HEIGHT, TILE_WIDTH};
    const tt::tt_metal::Tile::TileShape A_tile = A.tensor_spec().tile().get_tile_shape();
    const tt::tt_metal::Tile::TileShape B_tile = B.tensor_spec().tile().get_tile_shape();
    if (A_tile != tile_32x32 || B_tile != tile_32x32) {
        return fmt::format("tiles other than 32x32 (A {}x{}, B {}x{})", A_tile[0], A_tile[1], B_tile[0], B_tile[1]);
    }
    if (attributes.output_tile.has_value() && attributes.output_tile->get_tile_shape() != tile_32x32) {
        return "an output tile other than 32x32";
    }
    return std::nullopt;
}

// The GEMM a plan runs, with only its size fields set. Batches of A that use one B are one [batch * M x N] GEMM:
// A's and C's batches are stacked tile rows, so its C slices spread over the cores like any other rows.
MatmulUnifiedPlan plan_gemm_size(const ttnn::Tensor& A, const ttnn::Tensor& B, const MatmulParams& attributes) {
    const tt::tt_metal::Shape& A_shape = A.padded_shape();
    const tt::tt_metal::Shape& B_shape = B.padded_shape();
    MatmulUnifiedPlan plan{};
    plan.batch_size = get_batch_size(A_shape);
    plan.M_tiles = A_shape[-2] / TILE_HEIGHT;
    plan.K_tiles = A_shape[-1] / TILE_WIDTH;
    plan.N_tiles = B_shape[-1] / TILE_WIDTH;
    const bool B_has_one_batch = attributes.bcast_batch.value();  // the op sets bcast_batch iff batch_size(B) == 1
    plan.B_batch_stride_tiles = B_has_one_batch ? 0 : plan.K_tiles * plan.N_tiles;
    if (B_has_one_batch) {
        plan.M_tiles *= plan.batch_size;
        plan.batch_size = 1;
    }
    plan.A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;
    return plan;
}

// True when every active core already holds, in its own L1, exactly the block of a tensor (with memory config `memory`)
// its C slice uses: the tensor is L1-sharded with that shard shape and its grid lists the active cores in assignment
// order (shard i on the core that produces C slice i). Such a shard is borrowable: bound in place as the core's DFB,
// with no copy.
bool shard_matches(
    const tt::tt_metal::MemoryConfig& memory,
    uint32_t shard_M_tiles,
    uint32_t shard_N_tiles,
    const std::vector<CoreCoord>& cores) {
    if (!memory.is_sharded() || memory.buffer_type() != tt::tt_metal::BufferType::L1 ||
        !memory.shard_spec().has_value()) {
        return false;
    }
    const tt::tt_metal::ShardSpec& shard = *memory.shard_spec();
    if (shard.shape[0] != shard_M_tiles * TILE_HEIGHT || shard.shape[1] != shard_N_tiles * TILE_WIDTH) {
        return false;
    }
    const std::vector<CoreCoord> shard_cores =
        corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
    return shard_cores == cores;
}

// Every field of the plan for one set of choices: C slice, subblock, K chunk and batch groups. `base` holds the fields
// common to every candidate; `cores` are the config's cores, of which the first num_batch_groups runs of
// min(cores / num_batch_groups, C_slices_per_batch) are active.
MatmulUnifiedPlan make_plan(
    const MatmulUnifiedPlan& base,
    const CoreRangeSet& cores,
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    const std::optional<ttnn::Tensor>& output,
    const MatmulParams& attributes,
    uint32_t C_slice_M_tiles,
    uint32_t C_slice_N_tiles,
    uint32_t subblock_M_tiles,
    uint32_t subblock_N_tiles,
    uint32_t K_chunk_tiles,
    uint32_t num_batch_groups) {
    MatmulUnifiedPlan plan = base;
    const std::vector<CoreCoord> all_cores =
        corerange_to_cores(cores, std::nullopt, base.orientation == ShardOrientation::ROW_MAJOR);
    const bool is_quasar = A.device()->arch() == tt::ARCH::QUASAR;
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    plan.C_slice_M_tiles = C_slice_M_tiles;
    plan.C_slice_N_tiles = C_slice_N_tiles;
    // The C slices of one batch, walked across N then down M; each batch group's cores split them, and every core
    // produces its C slices for each of its group's batches.
    const uint32_t C_slices_across_N = tt::div_up(plan.N_tiles, C_slice_N_tiles);
    const uint32_t C_slices_down_M = tt::div_up(plan.M_tiles, C_slice_M_tiles);
    plan.C_slices_across_N = C_slices_across_N;
    plan.C_slices_per_batch = C_slices_down_M * C_slices_across_N;
    plan.num_batch_groups = num_batch_groups;
    const uint32_t cores_per_batch_group =
        std::min<uint32_t>(all_cores.size() / num_batch_groups, plan.C_slices_per_batch);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_batch_groups * cores_per_batch_group);
    plan.max_C_slices_per_core = tt::div_up(plan.C_slices_per_batch, cores_per_batch_group);
    // The busiest core's C slices over all its batches.
    const uint64_t C_slices_per_core =
        (uint64_t)tt::div_up(plan.batch_size, num_batch_groups) * plan.max_C_slices_per_core;
    // A shard is borrowed only with one batch: each core's one C slice then has one block of each operand.
    const bool one_C_slice = plan.batch_size == 1 && C_slices_per_core == 1;

    // Shards that match the C slice.
    // A: the C slice's rows for all of K; C slices must span N so no two cores need the same rows. The copy
    // path zeroes A's K padding in the DFB; a borrowed shard is never written, so K must be a tile multiple.
    plan.A_shard_borrowable =
        one_C_slice && C_slices_across_N == 1 && plan.A_last_K_tile_valid_columns == 0 &&
        shard_matches(A.memory_config(), C_slice_M_tiles, plan.K_tiles, plan.cores) &&
        (uint64_t)C_slice_M_tiles * plan.K_tiles * tt::tile_size(plan.A_format) <= MAX_DFB_EXTENT_BYTES;
    // B: the C slice's columns for all of K; C slices must span M.
    plan.B_shard_borrowable =
        one_C_slice && C_slices_down_M == 1 &&
        shard_matches(B.memory_config(), plan.K_tiles, C_slice_N_tiles, plan.cores) &&
        (uint64_t)plan.K_tiles * C_slice_N_tiles * tt::tile_size(plan.B_format) <= MAX_DFB_EXTENT_BYTES;
    // C: its own shard spec, or the one the op allocates it with. Packed straight into the shard when subblock-major
    // pack order equals the shard's row-major order, which also needs a single compute thread (with several, each
    // thread owns every N-th entry).
    const tt::tt_metal::MemoryConfig& C_memory =
        output.has_value() ? output->memory_config() : attributes.output_mem_config;
    plan.C_slices_are_C_shards = C_memory.shard_spec().has_value()
                                     ? C_slices_per_core == 1 &&
                                           (plan.batch_size == 1 || plan.M_tiles % C_slice_M_tiles == 0) &&
                                           shard_matches(C_memory, C_slice_M_tiles, C_slice_N_tiles, plan.cores)
                                     : C_slice_shard_spec(plan, C_memory).has_value();
    plan.C_shard_borrowable = plan.C_slices_are_C_shards && one_C_slice && plan.num_compute_threads == 1;

    // Multicast.
    // One batch group with one C slice per core on a rectangle laid out as the C slices tile C (walked row-major): the
    // cores of a row then need the same A slice at the same time, and the cores of a column the same B slice. When C
    // is a single row of C slices every core of any rectangle needs the same A slice (one column: the same B slice).
    // Several batch groups work on different batches.
    const std::vector<CoreRange>& core_ranges = cores.ranges();
    const bool one_C_slice_per_core_of_rectangle = num_batch_groups == 1 && core_ranges.size() == 1 &&
                                                   plan.max_C_slices_per_core == 1 &&
                                                   plan.cores.size() == core_ranges[0].size();
    const bool cores_tile_C =
        one_C_slice_per_core_of_rectangle && core_ranges[0].grid_size().x == C_slices_across_N &&
        core_ranges[0].grid_size().y == C_slices_down_M &&
        (base.orientation == ShardOrientation::ROW_MAJOR || C_slices_across_N == 1 || C_slices_down_M == 1);
    plan.A_mcast_to_all_cores = one_C_slice_per_core_of_rectangle && !cores_tile_C && C_slices_down_M == 1;
    plan.B_mcast_to_all_cores = one_C_slice_per_core_of_rectangle && !cores_tile_C && C_slices_across_N == 1;
    const uint32_t A_mcast_group = cores_tile_C ? C_slices_across_N : plan.A_mcast_to_all_cores ? plan.cores.size() : 1;
    const uint32_t B_mcast_group = cores_tile_C ? C_slices_down_M : plan.B_mcast_to_all_cores ? plan.cores.size() : 1;
    plan.A_mcast_num_dests = plan.A_shard_borrowable ? 0 : A_mcast_group - 1;
    plan.B_mcast_num_dests = plan.B_shard_borrowable ? 0 : B_mcast_group - 1;

    // Subblock: the C slice's tiles accumulated in DST at once; the C slice is rounded up to it.
    plan.subblock_M_tiles = subblock_M_tiles;
    plan.subblock_N_tiles = subblock_N_tiles;
    plan.C_slice_M_padded_tiles = tt::round_up(C_slice_M_tiles, subblock_M_tiles);
    plan.C_slice_N_padded_tiles = tt::round_up(C_slice_N_tiles, subblock_N_tiles);
    const bool M_padded = plan.C_slice_M_padded_tiles != C_slice_M_tiles;
    const bool N_padded = plan.C_slice_N_padded_tiles != C_slice_N_tiles;

    // K chunk.
    plan.K_chunk_tiles = K_chunk_tiles;
    plan.K_chunks_per_C_slice = plan.K_tiles / K_chunk_tiles;
    // The packer accumulates partials in L1 when there are enough K chunks for the reconfig overhead to pay off:
    // from two when it adds straight onto the C slice, else more than two (the last K chunk spills and reloads).
    const bool can_accumulate_in_C_slice =
        !base.fp32_dest_acc_en && plan.C_format == tt::DataFormat::Float16_b && !base.has_fused_activation;
    plan.packer_l1_acc_en = packer_l1_acc && plan.K_chunks_per_C_slice > (can_accumulate_in_C_slice ? 1 : 2);
    plan.accumulate_in_C_slice = can_accumulate_in_C_slice && plan.packer_l1_acc_en;
    plan.C_partials_format = base.fp32_dest_acc_en   ? tt::DataFormat::Float32
                             : plan.packer_l1_acc_en ? tt::DataFormat::Float16_b
                                                     : plan.C_format;

    // Borrowed operands: a matching shard bound as the DFB itself, when nothing is padded.
    // A borrowed shard holds only the true C slice dims. A's must also cover all of K in one K chunk, and C's tile
    // order matches the pack order only when subblocks span the C slice width.
    plan.borrow_A = plan.A_shard_borrowable && !M_padded && plan.K_chunks_per_C_slice == 1;
    plan.borrow_B = plan.B_shard_borrowable && !N_padded;
    plan.borrow_C = plan.C_shard_borrowable && subblock_N_tiles == C_slice_N_tiles && !M_padded;

    // DM threads and buffering (fixed; see the constants). Quasar's reader threads each read every
    // QUASAR_NUM_READER_THREADS-th K chunk into their own part of the A and B DFBs, which a borrowed shard cannot be
    // split into; reader threads beyond the K chunks of a C slice would only carry padding. A DFB that only one slice
    // ever passes through holds one.
    plan.num_reader_threads = is_quasar && !plan.borrow_A && !plan.borrow_B
                                  ? std::min(QUASAR_NUM_READER_THREADS, plan.K_chunks_per_C_slice)
                                  : 1;
    plan.K_chunks_per_C_slice_padded = tt::round_up(plan.K_chunks_per_C_slice, plan.num_reader_threads);
    const bool more_than_one_slice = C_slices_per_core * plan.K_chunks_per_C_slice_padded > 1;
    plan.operand_buffer_depth = more_than_one_slice ? std::max(MIN_OPERAND_BUFFER_DEPTH, plan.num_reader_threads) : 1;
    // One C slice's bias per reader thread, and at least two when the core has several, so a single reader reads
    // the next C slice's bias (and the A and B slices behind it) while compute uses this one.
    const uint32_t bias_buffer_depth =
        std::max(plan.num_reader_threads, C_slices_per_core == 1 ? 1u : MIN_OPERAND_BUFFER_DEPTH);

    // DFB sizes.
    // A thread's subblocks of the C slice (back to back, padded to the busiest thread's) take
    // C_entries_per_thread one-tile entries of C_slice and C_partials, pushed and popped as one batch per K chunk:
    // every thread then moves the same credits, and its C_partials entries come back to the same addresses every
    // K chunk, where the packer's L1 accumulation needs them.
    const uint32_t num_subblocks =
        (plan.C_slice_M_padded_tiles / subblock_M_tiles) * (plan.C_slice_N_padded_tiles / subblock_N_tiles);
    plan.C_entries_per_thread =
        tt::div_up(num_subblocks, plan.num_compute_threads) * subblock_M_tiles * subblock_N_tiles;
    plan.C_entry_bytes = tt::tile_size(plan.C_format);
    // Several C slices per core double-buffer C_slice, unless two would not fit one DFB: the next C slice then waits
    // for the writer to drain this one.
    const uint64_t C_slice_bytes = (uint64_t)plan.C_entries_per_thread * plan.num_compute_threads * plan.C_entry_bytes;
    plan.C_buffer_depth =
        C_slices_per_core > 1 && C_BUFFER_DEPTH * C_slice_bytes <= MAX_DFB_EXTENT_BYTES ? C_BUFFER_DEPTH : 1;
    plan.C_partials_entry_bytes = tt::tile_size(plan.C_partials_format);
    plan.C_slice_entries = plan.C_entries_per_thread * plan.num_compute_threads * plan.C_buffer_depth;
    // A copied operand's DFB holds operand_buffer_depth slices; a borrowed DFB is the resident shard itself.
    plan.A_entry_bytes = tt::tile_size(plan.A_format);
    plan.B_entry_bytes = tt::tile_size(plan.B_format);
    plan.A_slice_entries = plan.borrow_A ? C_slice_M_tiles * plan.K_tiles
                                         : plan.C_slice_M_padded_tiles * K_chunk_tiles * plan.operand_buffer_depth;
    plan.B_slice_entries = plan.borrow_B ? plan.K_tiles * C_slice_N_tiles
                                         : K_chunk_tiles * plan.C_slice_N_padded_tiles * plan.operand_buffer_depth;
    // Alias C_partials onto C_slice only when partials are never live while C_slice holds unread data
    // (else compute packs slice i+1's partials over slice i before the writer drains it).
    const bool partials_ever_written = plan.K_chunks_per_C_slice > 1 && !plan.accumulate_in_C_slice;
    plan.alias_C_partials_onto_C_slice = (plan.C_partials_format == plan.C_format) && plan.C_buffer_depth == 1 &&
                                         (!partials_ever_written || C_slices_per_core == 1);
    // Partials that are never written need one entry per thread when they cannot reuse C_slice's L1.
    plan.C_partials_entries = (partials_ever_written || plan.alias_C_partials_onto_C_slice)
                                  ? plan.C_entries_per_thread * plan.num_compute_threads
                                  : plan.num_compute_threads;
    plan.bias_entries = plan.has_bias ? plan.C_slice_N_padded_tiles * bias_buffer_depth : 0;
    // Borrowed DFBs are the tensors' own memory and cost nothing here.
    plan.l1_bytes =
        (plan.borrow_A ? 0 : (uint64_t)plan.A_slice_entries * plan.A_entry_bytes) +
        (plan.borrow_B ? 0 : (uint64_t)plan.B_slice_entries * plan.B_entry_bytes) +
        (plan.borrow_C ? 0 : (uint64_t)plan.C_slice_entries * plan.C_entry_bytes) +
        (plan.alias_C_partials_onto_C_slice ? 0 : (uint64_t)plan.C_partials_entries * plan.C_partials_entry_bytes) +
        (uint64_t)plan.bias_entries * plan.bias_entry_bytes;
    return plan;
}

// Math fidelity phases per tile product. Wormhole and Blackhole issue every phase asked for; Quasar drops the phases
// that need low mantissa bits its source registers do not hold (only Float32 operands carry them; the Quasar LLK's
// math_fidelity_schedule, which takes B in SrcA and A in SrcB).
uint32_t fidelity_phases(
    tt::ARCH arch, tt::tt_metal::MathFidelity math_fidelity, tt::DataFormat A_format, tt::DataFormat B_format) {
    const uint32_t requested = math_fidelity == tt::tt_metal::MathFidelity::LoFi    ? 1
                               : math_fidelity == tt::tt_metal::MathFidelity::HiFi2 ? 2
                               : math_fidelity == tt::tt_metal::MathFidelity::HiFi3 ? 3
                                                                                    : 4;
    if (arch != tt::ARCH::QUASAR) {
        return requested;
    }
    const bool wide_SrcA = B_format == tt::DataFormat::Float32;
    const bool wide_SrcB = A_format == tt::DataFormat::Float32;
    return 1 + (wide_SrcA && requested > 1) + (wide_SrcB && requested > 2) + (wide_SrcA && wide_SrcB && requested > 3);
}

// Estimated cycles of a plan, and the busiest core's pipeline alone (see MatmulUnifiedPlan::estimated_cycles). The
// busiest core runs its C slices' K chunks as a pipeline (reader threads fetch later K chunks while compute multiplies
// this one), and DRAM bandwidth bounds all cores together. The terms follow the kernels; ArchCostModel holds the
// fitted costs.
std::pair<double, double> estimate_plan_cycles(
    const MatmulUnifiedPlan& plan,
    const ArchCostModel& model,
    uint32_t fidelity_phases,
    bool A_in_dram,
    bool B_in_dram,
    bool C_in_dram) {
    constexpr double TILE_BYTES = 2048.0;  // the cost model's tile
    const double C_slices_on_busiest_core =
        (double)tt::div_up(plan.batch_size, plan.num_batch_groups) * plan.max_C_slices_per_core;
    const double K_chunks_on_busiest_core = C_slices_on_busiest_core * plan.K_chunks_per_C_slice;

    // Compute, on the busiest thread: per K tile of a subblock the unpack overlaps the math, and padded tiles cost
    // as much as real ones.
    const uint32_t subblock_tiles = plan.subblock_M_tiles * plan.subblock_N_tiles;
    const uint32_t num_subblocks =
        (plan.C_slice_M_padded_tiles / plan.subblock_M_tiles) * (plan.C_slice_N_padded_tiles / plan.subblock_N_tiles);
    const uint32_t subblocks_per_thread = tt::div_up(num_subblocks, plan.num_compute_threads);
    const double K_tile_cycles = std::max(
        subblock_tiles * model.tile_matmul_cycles * fidelity_phases,
        plan.subblock_M_tiles * model.unpack_A_tile_cycles + plan.subblock_N_tiles * model.unpack_B_tile_cycles);
    const double K_chunk_cycles = subblocks_per_thread * (model.subblock_cycles + plan.K_chunk_tiles * K_tile_cycles +
                                                          subblock_tiles * model.tile_pack_cycles) +
                                  model.K_chunk_cycles;
    // Accumulating in the C slice reloads nothing; packer L1 accumulation in C_partials reloads the partial sums on
    // the last K chunk only, else every K chunk after the first reloads them.
    const uint32_t reloads_per_C_slice = plan.accumulate_in_C_slice ? 0
                                         : plan.packer_l1_acc_en    ? 1
                                                                    : plan.K_chunks_per_C_slice - 1;
    const double compute_cycles = C_slices_on_busiest_core * (plan.K_chunks_per_C_slice * K_chunk_cycles +
                                                              reloads_per_C_slice * subblocks_per_thread *
                                                                  subblock_tiles * model.tile_reload_cycles);

    // Data movement on the busiest core, a multicast sender: per C slice its A rows and B columns over all of K, which
    // it also multicasts, and the C slice it writes. A borrowed operand does not move.
    const double A_tiles =
        plan.borrow_A ? 0.0 : (double)plan.C_slice_M_tiles * plan.K_tiles * plan.A_entry_bytes / TILE_BYTES;
    const double B_tiles =
        plan.borrow_B ? 0.0 : (double)plan.K_tiles * plan.C_slice_N_tiles * plan.B_entry_bytes / TILE_BYTES;
    const double bias_tiles = (double)plan.C_slice_N_tiles * plan.bias_entry_bytes / TILE_BYTES;
    const double C_tiles = (double)plan.C_slice_M_tiles * plan.C_slice_N_tiles * plan.C_entry_bytes / TILE_BYTES;
    const double C_write_cycles = plan.borrow_C ? 0.0 : C_tiles * model.tile_write_cycles;
    // With ragged K one reader thread zero-pads the last tile of each of the C slice's A rows.
    const double A_pad_cycles =
        plan.A_last_K_tile_valid_columns > 0 && !plan.borrow_A ? plan.C_slice_M_tiles * model.A_pad_tile_cycles : 0.0;
    const bool multicast = plan.A_mcast_num_dests > 0 || plan.B_mcast_num_dests > 0;
    const double read_cycles_per_tile =
        std::max(model.tile_read_cycles / plan.num_reader_threads, model.core_tile_read_cycles);
    const double dm_cycles =
        C_slices_on_busiest_core *
            ((A_tiles + B_tiles + bias_tiles) * read_cycles_per_tile +
             ((plan.A_mcast_num_dests > 0 ? A_tiles : 0.0) + (plan.B_mcast_num_dests > 0 ? B_tiles : 0.0)) *
                 model.tile_mcast_cycles +
             C_write_cycles + A_pad_cycles) +
        K_chunks_on_busiest_core * (model.K_chunk_read_cycles + (multicast ? model.K_chunk_mcast_cycles : 0.0)) /
            plan.num_reader_threads;

    // The pipeline. Its fill overlaps nothing: one reader thread fetches (and multicasts) the first K chunk, and the
    // last C slice is written after its last K chunk. Then every later step costs about the slower of its read and its
    // compute (they overlap imperfectly, see ArchCostModel::overlap_exponent). DRAM bandwidth can bound it all instead,
    // the first K chunk's read included.
    const double first_K_chunk_A_tiles =
        (double)plan.C_slice_M_tiles * plan.K_chunk_tiles * plan.A_entry_bytes / TILE_BYTES;
    const double first_K_chunk_B_tiles =
        (double)plan.K_chunk_tiles * plan.C_slice_N_tiles * plan.B_entry_bytes / TILE_BYTES;
    const double fill_cycles = (first_K_chunk_A_tiles + first_K_chunk_B_tiles) * model.tile_read_cycles +
                               ((plan.A_mcast_num_dests > 0 ? first_K_chunk_A_tiles : 0.0) +
                                (plan.B_mcast_num_dests > 0 ? first_K_chunk_B_tiles : 0.0)) *
                                   model.tile_mcast_cycles +
                               model.read_latency_cycles + C_write_cycles +
                               C_slices_on_busiest_core * model.C_slice_cycles +
                               (plan.K_chunks_per_C_slice == 1 ? A_pad_cycles : 0.0);
    const double read_step_cycles = dm_cycles / K_chunks_on_busiest_core;
    const double compute_step_cycles = compute_cycles / K_chunks_on_busiest_core;
    const double step_cycles = std::pow(
        std::pow(read_step_cycles, model.overlap_exponent) + std::pow(compute_step_cycles, model.overlap_exponent),
        1.0 / model.overlap_exponent);
    const double steady_cycles = compute_step_cycles + (K_chunks_on_busiest_core - 1) * step_cycles;

    // DRAM, all cores: a multicast operand is read once per multicast group.
    const double total_C_slices = (double)plan.batch_size * plan.C_slices_per_batch;
    const double dram_tiles =
        total_C_slices * ((A_in_dram ? A_tiles / (plan.A_mcast_num_dests + 1) : 0.0) +
                          (B_in_dram ? B_tiles / (plan.B_mcast_num_dests + 1) : 0.0) + (C_in_dram ? C_tiles : 0.0));
    const double pipeline_cycles = fill_cycles + steady_cycles;
    return {
        std::max(pipeline_cycles, dram_tiles * model.dram_tile_cycles + model.read_latency_cycles + C_write_cycles),
        pipeline_cycles};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::optional<MatmulUnifiedPlan> plan_unified_matmul(
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    uint32_t bias_tile_bytes,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output,
    bool must_fit) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (const std::optional<std::string> unsupported = find_unsupported_feature(A, B, attributes)) {
        TT_THROW("MatmulUnifiedProgramConfig does not support {}", *unsupported);
    }
    const tt::tt_metal::distributed::MeshDevice& device = *A.device();
    const bool is_quasar = device.arch() == tt::ARCH::QUASAR;

    // ---- 1. GEMM size ----
    MatmulUnifiedPlan base = plan_gemm_size(A, B, attributes);

    // ---- 2. Config checks ----
    const CoreCoord grid = device.compute_with_storage_grid_size();
    const CoreRange bounding_box = config.cores.bounding_box();
    TT_FATAL(
        bounding_box.end_coord.x < grid.x && bounding_box.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    // An unset field is auto; a set size is at least one tile.
    TT_FATAL(
        config.C_slice_M_tiles.value_or(1) >= 1 && config.C_slice_N_tiles.value_or(1) >= 1 &&
            config.K_chunk_tiles.value_or(1) >= 1 && config.subblock_M_tiles.value_or(1) >= 1 &&
            config.subblock_N_tiles.value_or(1) >= 1,
        "MatmulUnifiedProgramConfig sizes must be at least one tile; leave a field unset for auto");
    const std::optional<std::size_t>& requested_threads = config.num_compute_threads;  // checked before it narrows
    TT_FATAL(
        !requested_threads.has_value() || *requested_threads == 1 ||
            (is_quasar && (*requested_threads == 2 || *requested_threads == 4)),
        "MatmulUnifiedProgramConfig.num_compute_threads must be 1, 2 or 4 (above 1 needs Quasar), or unset for auto; "
        "got {}",
        *requested_threads);
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        config.subblock_M_tiles.has_value() == config.subblock_N_tiles.has_value(),
        "subblock_M_tiles and subblock_N_tiles must both be set or both be unset (auto)");
    TT_FATAL(
        config.subblock_M_tiles.value_or(1) * config.subblock_N_tiles.value_or(1) <= dst_capacity_tiles,
        "subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        *config.subblock_M_tiles,
        *config.subblock_N_tiles,
        *config.subblock_M_tiles * *config.subblock_N_tiles,
        dst_capacity_tiles,
        fp32_dest_acc_en);
    TT_FATAL(
        config.num_batch_groups.value_or(1) >= 1 && config.num_batch_groups.value_or(1) <= base.batch_size,
        "MatmulUnifiedProgramConfig.num_batch_groups ({}) must be at least 1 and at most the batches with their own B "
        "({})",
        config.num_batch_groups.value_or(1),
        base.batch_size);
    // More batch groups than cores is a placement that does not fit (the chooser tries others).
    if (config.num_batch_groups.value_or(1) > config.cores.num_cores()) {
        TT_FATAL(
            !must_fit,
            "MatmulUnifiedProgramConfig.num_batch_groups ({}) exceeds the cores ({})",
            *config.num_batch_groups,
            config.cores.num_cores());
        return std::nullopt;
    }
    TT_FATAL(
        !config.K_chunk_tiles.has_value() || base.K_tiles % *config.K_chunk_tiles == 0,
        "K_chunk_tiles ({}) must divide K_tiles ({})",
        *config.K_chunk_tiles,
        base.K_tiles);
    TT_FATAL(
        !(bias_tile_bytes > 0 && fp32_dest_acc_en),
        "MatmulUnifiedProgramConfig fuses bias only with a 16-bit DST (fp32_dest_acc_en = false); ttnn.linear adds it "
        "separately otherwise");

    // ---- 3. Fixed for every candidate: cores, threads, formats, L1 budget ----
    base.orientation = config.orientation;
    base.num_compute_threads =
        requested_threads.has_value() ? static_cast<uint32_t>(*requested_threads) : (is_quasar ? 4 : 1);
    // Writer thread t writes the entries of compute threads t, t + num_writer_threads, ...
    base.num_writer_threads =
        is_quasar && base.num_compute_threads % QUASAR_NUM_WRITER_THREADS == 0 ? QUASAR_NUM_WRITER_THREADS : 1;
    base.fp32_dest_acc_en = fp32_dest_acc_en;
    base.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    base.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    base.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    // Shape, layout and batch of the bias are checked by the op; it is one tile row of C's columns.
    base.has_bias = bias_tile_bytes > 0;
    base.bias_entry_bytes = bias_tile_bytes;
    base.has_fused_activation = config.fused_activation.has_value();
    const uint32_t l1_base = device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t l1_ceiling = device.lowest_occupied_compute_l1_address().value_or(device.l1_size_per_core());
    const uint64_t l1_budget = l1_ceiling - l1_base;

    // ---- 4. C slice candidates ----
    // A C slice dim the config gives is fixed, and an auto one divides the largest C slice: M / N split over the
    // bounding box of the cores (one C slice per core). With batches that have their own B and batch groups allowed,
    // the groups keep the cores busy with C slices up to a whole batch's C, so an auto dim may also split M (N) into
    // any number of C slices (the smallest dim for that number). A sharded C's shard shape comes first, so its shards
    // can be the C slices.
    const uint32_t core_rows = bounding_box.grid_size().y;
    const uint32_t core_columns = bounding_box.grid_size().x;
    const bool M_fixed = config.C_slice_M_tiles.has_value();
    const bool N_fixed = config.C_slice_N_tiles.has_value();
    const bool batch_groups_allowed = base.batch_size > 1 && config.num_batch_groups.value_or(2) > 1;
    const uint32_t max_C_slice_M_tiles =
        config.C_slice_M_tiles.value_or(batch_groups_allowed ? base.M_tiles : tt::div_up(base.M_tiles, core_rows));
    const uint32_t max_C_slice_N_tiles =
        config.C_slice_N_tiles.value_or(batch_groups_allowed ? base.N_tiles : tt::div_up(base.N_tiles, core_columns));

    std::vector<std::pair<uint32_t, uint32_t>> C_slice_shapes;
    const std::optional<tt::tt_metal::ShardSpec>& C_shard =
        output.has_value() ? output->memory_config().shard_spec() : attributes.output_mem_config.shard_spec();
    if (C_shard.has_value()) {
        C_slice_shapes.emplace_back(
            config.C_slice_M_tiles.value_or(C_shard->shape[0] / TILE_HEIGHT),
            config.C_slice_N_tiles.value_or(C_shard->shape[1] / TILE_WIDTH));
    }
    for (uint32_t h = max_C_slice_M_tiles; h >= 1; --h) {
        for (uint32_t w = max_C_slice_N_tiles; w >= 1; --w) {
            const bool h_allowed = M_fixed                ? h == max_C_slice_M_tiles
                                   : batch_groups_allowed ? h == tt::div_up(base.M_tiles, tt::div_up(base.M_tiles, h))
                                                          : max_C_slice_M_tiles % h == 0;
            const bool w_allowed = N_fixed                ? w == max_C_slice_N_tiles
                                   : batch_groups_allowed ? w == tt::div_up(base.N_tiles, tt::div_up(base.N_tiles, w))
                                                          : max_C_slice_N_tiles % w == 0;
            if (h_allowed && w_allowed) {
                C_slice_shapes.emplace_back(h, w);
            }
        }
    }
    // ---- 5. Search: the C slice, subblock and K chunk with the lowest estimated cycles ----
    // Every combination whose DFBs fit is estimated; a subblock or K chunk the config gives is the only one tried. An
    // auto subblock may pad the C slice (the estimate charges the padded work) but is no larger than it, which would
    // only add padding; a K chunk divides K_tiles. C slices that are a sharded C's shards (each core writes its own)
    // come before any estimate.
    const ArchCostModel& cost_model = device.arch() == tt::ARCH::QUASAR      ? QUASAR_COST_MODEL
                                      : device.arch() == tt::ARCH::BLACKHOLE ? BLACKHOLE_COST_MODEL
                                                                             : WORMHOLE_COST_MODEL;
    const uint32_t math_fidelity_phases = fidelity_phases(
        device.arch(),
        std::get<0>(get_compute_kernel_config_args(device.arch(), attributes.compute_kernel_config.value())),
        base.A_format,
        base.B_format);
    const bool C_in_dram =
        output.has_value() ? output->memory_config().is_dram() : attributes.output_mem_config.is_dram();
    const uint32_t num_cores = static_cast<uint32_t>(config.cores.num_cores());
    std::optional<MatmulUnifiedPlan> best;
    // After the loops, the last tried (the smallest C slice, a 1x1 subblock and the shortest K chunk): the L1 error
    // reports it.
    MatmulUnifiedPlan candidate;
    for (const auto& [C_slice_M_tiles, C_slice_N_tiles] : C_slice_shapes) {
        // Batch groups: the config's, else one (every core takes every batch) and, with batches that have their own B,
        // the count that leaves the busiest core fewest C slices (the fewest groups on a tie), so that few C slices per
        // batch do not leave cores idle. Only one group can multicast.
        const uint32_t C_slices_per_batch =
            tt::div_up(base.M_tiles, C_slice_M_tiles) * tt::div_up(base.N_tiles, C_slice_N_tiles);
        uint32_t spread_batch_groups = 1;
        uint64_t spread_C_slices_per_core =
            (uint64_t)base.batch_size * tt::div_up(C_slices_per_batch, std::min(num_cores, C_slices_per_batch));
        for (uint32_t groups = 2; groups <= std::min(base.batch_size, num_cores); ++groups) {
            const uint32_t cores_per_group = std::min(num_cores / groups, C_slices_per_batch);
            const uint64_t C_slices_per_core =
                (uint64_t)tt::div_up(base.batch_size, groups) * tt::div_up(C_slices_per_batch, cores_per_group);
            if (C_slices_per_core < spread_C_slices_per_core) {
                spread_batch_groups = groups;
                spread_C_slices_per_core = C_slices_per_core;
            }
        }
        std::vector<uint32_t> batch_group_counts = {1};
        if (config.num_batch_groups.has_value()) {
            batch_group_counts = {static_cast<uint32_t>(*config.num_batch_groups)};
        } else if (spread_batch_groups > 1) {
            batch_group_counts.push_back(spread_batch_groups);
        }
        for (const uint32_t num_batch_groups : batch_group_counts) {
            // Widest first, so a tie keeps the wider subblock (measured slightly faster on Wormhole and Quasar).
            for (uint32_t subblock_N_tiles =
                     config.subblock_N_tiles.value_or(std::min(dst_capacity_tiles, C_slice_N_tiles));
                 subblock_N_tiles >= config.subblock_N_tiles.value_or(1);
                 --subblock_N_tiles) {
                for (uint32_t subblock_M_tiles = config.subblock_M_tiles.value_or(
                         std::min(dst_capacity_tiles / subblock_N_tiles, C_slice_M_tiles));
                     subblock_M_tiles >= config.subblock_M_tiles.value_or(1);
                     --subblock_M_tiles) {
                    for (uint32_t K_chunk_tiles = config.K_chunk_tiles.value_or(base.K_tiles);
                         K_chunk_tiles >= config.K_chunk_tiles.value_or(1);
                         --K_chunk_tiles) {
                        if (base.K_tiles % K_chunk_tiles != 0) {
                            continue;
                        }
                        candidate = make_plan(
                            base,
                            config.cores,
                            A,
                            B,
                            output,
                            attributes,
                            C_slice_M_tiles,
                            C_slice_N_tiles,
                            subblock_M_tiles,
                            subblock_N_tiles,
                            K_chunk_tiles,
                            num_batch_groups);
                        if (!dfbs_fit(candidate, l1_budget)) {
                            continue;
                        }
                        candidate.estimated_cycles = estimate_plan_cycles(
                            candidate,
                            cost_model,
                            math_fidelity_phases,
                            A.memory_config().is_dram(),
                            B.memory_config().is_dram(),
                            C_in_dram);
                        if (!best.has_value() ||
                            std::make_pair(!candidate.C_slices_are_C_shards, candidate.estimated_cycles) <
                                std::make_pair(!best->C_slices_are_C_shards, best->estimated_cycles)) {
                            best = candidate;
                        }
                    }
                }
            }
        }
    }
    if (!best.has_value()) {
        TT_FATAL(
            !must_fit,
            "MatmulUnifiedProgramConfig: the DFBs do not fit for any C slice; the smallest tried, {}x{} tiles with "
            "K_chunk_tiles={}, needs {} B (budget {} B; max DFB extent {} B: A slice {} B, B slice {} B, "
            "C slice {} B, C partials {} B, bias {} B)",
            candidate.C_slice_M_tiles,
            candidate.C_slice_N_tiles,
            candidate.K_chunk_tiles,
            candidate.l1_bytes,
            l1_budget,
            MAX_DFB_EXTENT_BYTES,
            (uint64_t)candidate.A_slice_entries * candidate.A_entry_bytes,
            (uint64_t)candidate.B_slice_entries * candidate.B_entry_bytes,
            (uint64_t)candidate.C_slice_entries * candidate.C_entry_bytes,
            (uint64_t)candidate.C_partials_entries * candidate.C_partials_entry_bytes,
            (uint64_t)candidate.bias_entries * candidate.bias_entry_bytes);
        return std::nullopt;
    }
    return best;
}

std::optional<tt::tt_metal::ShardSpec> C_slice_shard_spec(
    const MatmulUnifiedPlan& plan, const tt::tt_metal::MemoryConfig& C_memory) {
    // One C slice per core over all its batches; with several batches, C slices that tile each batch's rows exactly, so
    // C's rows of C slices run on from one batch to the next (one batch group per batch, in core order).
    if (!C_memory.is_sharded() || C_memory.buffer_type() != tt::tt_metal::BufferType::L1 ||
        plan.max_C_slices_per_core != 1 || tt::div_up(plan.batch_size, plan.num_batch_groups) != 1 ||
        (plan.batch_size > 1 && plan.M_tiles % plan.C_slice_M_tiles != 0)) {
        return std::nullopt;
    }
    const uint32_t C_slices_down_M = plan.batch_size * (plan.C_slices_per_batch / plan.C_slices_across_N);
    const bool row_major = plan.orientation == ShardOrientation::ROW_MAJOR;
    // Height and width sharding place shard i on the grid's i-th core; block sharding places shard (row, column) at
    // that offset in a rectangle (column, row with COL_MAJOR).
    CoreRangeSet grid(ttsl::Span<const CoreCoord>(plan.cores));
    switch (C_memory.memory_layout()) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
            if (plan.C_slices_across_N != 1) {
                return std::nullopt;
            }
            break;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
            if (C_slices_down_M != 1) {
                return std::nullopt;
            }
            break;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: {
            const CoreCoord first = plan.cores.front();
            const uint32_t width = row_major ? plan.C_slices_across_N : C_slices_down_M;
            const uint32_t height = row_major ? C_slices_down_M : plan.C_slices_across_N;
            grid = CoreRangeSet(CoreRange(first, {first.x + width - 1, first.y + height - 1}));
            break;
        }
        default: return std::nullopt;
    }
    if (corerange_to_cores(grid, std::nullopt, row_major) != plan.cores) {
        return std::nullopt;
    }
    return tt::tt_metal::ShardSpec(
        grid, {plan.C_slice_M_tiles * TILE_HEIGHT, plan.C_slice_N_tiles * TILE_WIDTH}, plan.orientation);
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// The cores next to the DRAM banks, in bank order, which the C slice walk follows: each reads its own bank's shard.
CoreRangeSet dram_bank_cores(const tt::tt_metal::distributed::MeshDevice& device) {
    std::vector<CoreRange> cores;
    for (const CoreCoord& core : device.get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0)) {
        cores.emplace_back(core);
    }
    return CoreRangeSet(std::move(cores));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::optional<MatmulUnifiedProgramConfig> choose_unified_placement(
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    uint32_t bias_tile_bytes,
    const MatmulParams& attributes,
    const MatmulUnifiedProgramConfig& requested) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    TT_FATAL(
        !requested.C_slice_M_tiles.has_value() && !requested.C_slice_N_tiles.has_value(),
        "MatmulUnifiedProgramConfig: with no cores the C slice is chosen with the placement; leave C_slice_M_tiles and "
        "C_slice_N_tiles unset");
    if (find_unsupported_feature(A, B, attributes).has_value()) {
        return std::nullopt;
    }

    // ---- 1. Setup ----
    const tt::tt_metal::distributed::MeshDevice& device = *A.device();
    const CoreCoord device_grid = device.compute_with_storage_grid_size();
    const CoreCoord grid = attributes.user_core_coord.value_or(device_grid);
    const uint32_t grid_columns = grid.x;
    const uint32_t grid_rows = grid.y;
    // With a 32-bit DST ttnn.linear adds the bias separately (see the plan), so it takes no L1 here.
    const uint32_t fused_bias_tile_bytes = get_fp32_dest_acc_en(attributes.compute_kernel_config) ? 0 : bias_tile_bytes;
    const MatmulUnifiedPlan gemm = plan_gemm_size(A, B, attributes);
    const uint32_t M_tiles = gemm.M_tiles;
    const uint32_t N_tiles = gemm.N_tiles;
    constexpr tt::tt_metal::ShardOrientation ROW_MAJOR = tt::tt_metal::ShardOrientation::ROW_MAJOR;

    // ---- 2. Placement candidates ----
    // Each is `requested` with a placement: the cores and their orientation, and the C slice and batch group count
    // when the placement fixes them (unset, the plan picks them).
    std::vector<MatmulUnifiedProgramConfig> candidates;
    const auto add_candidate = [&](const CoreRangeSet& cores,
                                   tt::tt_metal::ShardOrientation orientation,
                                   std::optional<uint32_t> C_slice_M_tiles,
                                   std::optional<uint32_t> C_slice_N_tiles,
                                   std::optional<uint32_t> num_batch_groups) {
        MatmulUnifiedProgramConfig config = requested;
        config.cores = cores;
        config.orientation = orientation;
        config.C_slice_M_tiles = C_slice_M_tiles;
        config.C_slice_N_tiles = C_slice_N_tiles;
        if (!requested.num_batch_groups.has_value()) {
            config.num_batch_groups = num_batch_groups;
        }
        candidates.push_back(std::move(config));
    };
    // A 1D placement's C slices are whole columns (rows) of C, or 1/2, 1/3, ... of them, rounded up (13 tiles: 13, 7,
    // 5, 4, 3, 2 and 1 tiles long).
    const auto C_slice_lengths = [](uint32_t C_length_tiles) {
        std::vector<uint32_t> lengths;
        for (uint32_t C_slices_along_length = 1; C_slices_along_length <= C_length_tiles; ++C_slices_along_length) {
            const uint32_t length = tt::div_up(C_length_tiles, C_slices_along_length);
            if (lengths.empty() || length != lengths.back()) {
                lengths.push_back(length);
            }
        }
        return lengths;
    };
    // Batch groups multicast nothing, so they take the whole grid, where their reads spread over the NoC; the
    // placements sized to one batch's C keep one group, the layout their multicast is for.
    constexpr std::optional<uint32_t> ONE_BATCH_GROUP = 1;

    // The grid, 2D: the rectangle the largest C slices (C over the grid's rows and columns, rounded up) fill, so A and
    // B multicast along its rows and columns.
    add_candidate(
        CoreRangeSet(CoreRange(
            {0, 0},
            {tt::div_up(N_tiles, tt::div_up(N_tiles, grid_columns)) - 1,
             tt::div_up(M_tiles, tt::div_up(M_tiles, grid_rows)) - 1})),
        ROW_MAJOR,
        std::nullopt,
        std::nullopt,
        ONE_BATCH_GROUP);
    // The grid, 1D: columns (rows) of C split across N (M) over whole grid rows (part of one row for a narrow C). With
    // one C slice per core every core needs the same A (B), multicast to all of them when the C slices fill every
    // core; so when they do not, also over the most whole rows they fill.
    for (const bool C_slices_are_columns : {true, false}) {
        const uint32_t C_width_tiles = C_slices_are_columns ? N_tiles : M_tiles;
        const uint32_t columns = std::min(grid_columns, C_width_tiles);
        const uint32_t all_rows = std::clamp(C_width_tiles / grid_columns, 1u, grid_rows);
        for (uint32_t rows = all_rows; rows >= 1; --rows) {
            const uint32_t C_slice_width_tiles = tt::div_up(C_width_tiles, columns * rows);
            const bool C_slices_fill_the_cores = tt::div_up(C_width_tiles, C_slice_width_tiles) == columns * rows;
            if (rows == all_rows || C_slices_fill_the_cores) {
                for (const uint32_t length : C_slice_lengths(C_slices_are_columns ? M_tiles : N_tiles)) {
                    add_candidate(
                        CoreRangeSet(CoreRange({0, 0}, {columns - 1, rows - 1})),
                        ROW_MAJOR,
                        C_slices_are_columns ? length : C_slice_width_tiles,
                        C_slices_are_columns ? C_slice_width_tiles : length,
                        ONE_BATCH_GROUP);
                }
            }
            if (C_slices_fill_the_cores) {
                break;
            }
        }
    }
    // The whole grid for batches with their own B, which the plan splits into batch groups when a batch has too few C
    // slices for it.
    if (gemm.batch_size > 1) {
        add_candidate(
            CoreRangeSet(CoreRange({0, 0}, {grid_columns - 1, grid_rows - 1})),
            ROW_MAJOR,
            std::nullopt,
            std::nullopt,
            std::nullopt);
    }
    // Sharded operands' cores, which keep them where they are: an L1-sharded C's own shard grid, where its shards can
    // be the C slices; rows of C on a height-sharded A's cores and columns of C on a width-sharded B's (an L1 shard can
    // then be bound in place, and a DRAM shard is read by the core next to its bank).
    const std::optional<tt::tt_metal::ShardSpec>& C_shard = attributes.output_mem_config.shard_spec();
    if (C_shard.has_value() && attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1) {
        add_candidate(C_shard->grid, C_shard->orientation, std::nullopt, std::nullopt, std::nullopt);
    }
    if (A.is_sharded() && A.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        for (const uint32_t length : C_slice_lengths(N_tiles)) {
            add_candidate(
                A.memory_config().is_dram() ? dram_bank_cores(device) : A.shard_spec()->grid,
                A.shard_spec()->orientation,
                A.shard_spec()->shape[0] / TILE_HEIGHT,
                length,
                ONE_BATCH_GROUP);
        }
    }
    if (B.is_sharded() && B.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        for (const uint32_t length : C_slice_lengths(M_tiles)) {
            add_candidate(
                B.memory_config().is_dram() ? dram_bank_cores(device) : B.shard_spec()->grid,
                B.shard_spec()->orientation,
                length,
                B.shard_spec()->shape[1] / TILE_WIDTH,
                ONE_BATCH_GROUP);
        }
    }

    // ---- 3. Plan every candidate; the best ranked wins ----
    // Ranked by C slices that are a sharded C's shards, then the lowest estimate; the first on a tie. The winner keeps
    // its plan's C slice, subblock, K chunk and batch group count. A candidate does not fit with cores beyond the
    // device grid (a larger user_core_coord) or DFBs beyond L1.
    std::optional<MatmulUnifiedProgramConfig> best;
    std::pair<bool, std::pair<double, double>> best_rank;
    for (MatmulUnifiedProgramConfig& config : candidates) {
        const CoreCoord last_core = config.cores.bounding_box().end_coord;
        if (last_core.x >= device_grid.x || last_core.y >= device_grid.y) {
            continue;
        }
        const std::optional<MatmulUnifiedPlan> plan =
            plan_unified_matmul(A, B, fused_bias_tile_bytes, config, attributes, std::nullopt, /*must_fit=*/false);
        if (!plan.has_value()) {
            continue;
        }
        const std::pair<bool, std::pair<double, double>> rank{!plan->C_slices_are_C_shards, plan->estimated_cycles};
        log_debug(
            tt::LogOp,
            "Unified matmul placement: {}x{}-tile C slice on {}: {:.0f} cycles{}",
            plan->C_slice_M_tiles,
            plan->C_slice_N_tiles,
            config.cores.str(),
            rank.second.first,
            plan->C_slices_are_C_shards ? ", C slices are C's shards" : "");
        if (!best.has_value() || rank < best_rank) {
            config.C_slice_M_tiles = plan->C_slice_M_tiles;
            config.C_slice_N_tiles = plan->C_slice_N_tiles;
            config.subblock_M_tiles = plan->subblock_M_tiles;
            config.subblock_N_tiles = plan->subblock_N_tiles;
            config.K_chunk_tiles = plan->K_chunk_tiles;
            config.num_batch_groups = plan->num_batch_groups;
            best = std::move(config);
            best_rank = rank;
        }
    }
    return best;
}

}  // namespace ttnn::prim::qsr
