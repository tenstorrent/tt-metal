// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul (GH#41910) planning: plan_unified_matmul turns a MatmulUnifiedProgramConfig with a placement into
// everything the factory builds from, and choose_unified_placement picks the placement whose plan is fastest.

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_unified_plan.hpp"

#include <algorithm>
#include <optional>
#include <string>
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
// Auto K chunk cap, for pipelining rather than L1: compute waits for the first K chunk to be read in full and the last
// K chunk's compute overlaps no read, so fewer, longer K chunks leave more of a C slice unoverlapped, which shows on
// cores that produce a single C slice. A C slice of at most SMALL_C_SLICE_TILES tiles has A and B slices cheap enough
// to read in longer K chunks: its cap is K_tiles / min_K_chunks (one K chunk per reader thread on Quasar, two to
// double-buffer elsewhere), still enough K chunks to overlap (#57884). Tuning points, not hardware limits.
constexpr uint32_t MAX_AUTO_K_CHUNK_TILES = 8;
constexpr uint32_t SMALL_C_SLICE_TILES = 4;
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

// True when every active core already holds, in its own L1, exactly the block of `tensor` its C slice uses: the tensor
// is L1-sharded with that shard shape and its grid lists the active cores in assignment order (shard i on the core that
// produces C slice i). Such a shard is borrowable: bound in place as the core's DFB, with no copy.
bool shard_matches(
    const ttnn::Tensor& tensor, uint32_t shard_M_tiles, uint32_t shard_N_tiles, const std::vector<CoreCoord>& cores) {
    if (!tensor.is_sharded() || tensor.memory_config().buffer_type() != tt::tt_metal::BufferType::L1) {
        return false;
    }
    const tt::tt_metal::ShardSpec& shard = tensor.shard_spec().value();
    if (shard.shape[0] != shard_M_tiles * TILE_HEIGHT || shard.shape[1] != shard_N_tiles * TILE_WIDTH) {
        return false;
    }
    const std::vector<CoreCoord> shard_cores =
        corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
    return shard_cores == cores;
}

// Every field of the plan for one set of choices: C slice, subblock and K chunk. `base`
// holds the fields common to every candidate; `cores` are the config's cores, of which the first C_slices_per_batch are
// active.
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
    uint32_t K_chunk_tiles) {
    MatmulUnifiedPlan plan = base;
    const std::vector<CoreCoord> all_cores =
        corerange_to_cores(cores, std::nullopt, base.orientation == ShardOrientation::ROW_MAJOR);
    const bool is_quasar = A.device()->arch() == tt::ARCH::QUASAR;
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    plan.C_slice_M_tiles = C_slice_M_tiles;
    plan.C_slice_N_tiles = C_slice_N_tiles;
    // The C slices of one batch, walked across N then down M; every core produces its C slices for all batches.
    const uint32_t C_slices_across_N = tt::div_up(plan.N_tiles, C_slice_N_tiles);
    const uint32_t C_slices_down_M = tt::div_up(plan.M_tiles, C_slice_M_tiles);
    plan.C_slices_across_N = C_slices_across_N;
    plan.C_slices_per_batch = C_slices_down_M * C_slices_across_N;
    plan.sharded_output_layout = C_slices_across_N == 1 ? tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED
                                 : C_slices_down_M == 1 ? tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED
                                                        : tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), plan.C_slices_per_batch);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    plan.max_C_slices_per_core = tt::div_up(plan.C_slices_per_batch, num_active);
    // The busiest core's C slices over all batches.
    const uint64_t C_slices_per_core = (uint64_t)plan.batch_size * plan.max_C_slices_per_core;

    // Shards that match the C slice.
    // A: the C slice's rows for all of K; C slices must span N so no two cores need the same rows. The copy
    // path zeroes A's K padding in the DFB; a borrowed shard is never written, so K must be a tile multiple.
    plan.A_shard_borrowable =
        C_slices_per_core == 1 && C_slices_across_N == 1 && plan.A_last_K_tile_valid_columns == 0 &&
        shard_matches(A, C_slice_M_tiles, plan.K_tiles, plan.cores) &&
        (uint64_t)C_slice_M_tiles * plan.K_tiles * tt::tile_size(plan.A_format) <= MAX_DFB_EXTENT_BYTES;
    // B: the C slice's columns for all of K; C slices must span M.
    plan.B_shard_borrowable =
        C_slices_per_core == 1 && C_slices_down_M == 1 && shard_matches(B, plan.K_tiles, C_slice_N_tiles, plan.cores) &&
        (uint64_t)plan.K_tiles * C_slice_N_tiles * tt::tile_size(plan.B_format) <= MAX_DFB_EXTENT_BYTES;
    // C: packed straight into the shard when subblock-major pack order equals the shard's row-major order,
    // which also needs a single compute thread (with several, each thread owns every N-th entry).
    const bool C_shard_matches = output.has_value()
                                     ? shard_matches(output.value(), C_slice_M_tiles, C_slice_N_tiles, plan.cores)
                                     : (attributes.output_mem_config.is_sharded() &&
                                        attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1);
    plan.C_shard_borrowable = C_shard_matches && C_slices_per_core == 1 && plan.num_compute_threads == 1;

    // Multicast.
    // One C slice per core on a rectangle laid out as the C slices tile C (walked row-major): the cores of a row
    // then need the same A slice at the same time, and the cores of a column the same B slice. When C is a single
    // row of C slices every core of any rectangle needs the same A slice (one column: the same B slice).
    const std::vector<CoreRange>& core_ranges = cores.ranges();
    const bool one_C_slice_per_core_of_rectangle =
        core_ranges.size() == 1 && plan.max_C_slices_per_core == 1 && plan.cores.size() == core_ranges[0].size();
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
    // The packer accumulates partials in L1 only when there are enough K chunks for the reconfig overhead
    // to pay off (the last K chunk spills and reloads either way, so more than two).
    plan.packer_l1_acc_en = packer_l1_acc && plan.K_chunks_per_C_slice > 2;
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
    plan.C_buffer_depth = C_slices_per_core > 1 ? C_BUFFER_DEPTH : 1;
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
    const bool partials_ever_written = plan.K_chunks_per_C_slice > 1;
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
    const tt::tt_metal::IDevice& device = *A.device();
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
        !config.K_chunk_tiles.has_value() || base.K_tiles % *config.K_chunk_tiles == 0,
        "K_chunk_tiles ({}) must divide K_tiles ({})",
        *config.K_chunk_tiles,
        base.K_tiles);
    TT_FATAL(
        !(bias_tile_bytes > 0 && fp32_dest_acc_en),
        "MatmulUnifiedProgramConfig fuses bias only with a 16-bit DST (fp32_dest_acc_en = false); ttnn.linear adds it "
        "separately otherwise");
    const std::optional<tt::tt_metal::ShardSpec>& C_shard =
        output.has_value() ? output->memory_config().shard_spec() : attributes.output_mem_config.shard_spec();
    if (C_shard.has_value()) {
        const uint32_t C_shard_height = C_shard->shape[0];
        const uint32_t C_shard_width = C_shard->shape[1];
        TT_FATAL(
            C_shard_height % TILE_HEIGHT == 0 && C_shard_width % TILE_WIDTH == 0,
            "A sharded C needs a tile-multiple shard shape, got {}x{}",
            C_shard_height,
            C_shard_width);
        // A sharded C's shard is the C slice.
        const uint32_t C_shard_M_tiles = C_shard_height / TILE_HEIGHT;
        const uint32_t C_shard_N_tiles = C_shard_width / TILE_WIDTH;
        TT_FATAL(
            config.C_slice_M_tiles.value_or(C_shard_M_tiles) == C_shard_M_tiles &&
                config.C_slice_N_tiles.value_or(C_shard_N_tiles) == C_shard_N_tiles,
            "C is sharded, so its {}x{}-tile shard is the C slice; the config asks for a {}x{}-tile C slice",
            C_shard_M_tiles,
            C_shard_N_tiles,
            config.C_slice_M_tiles.value_or(C_shard_M_tiles),
            config.C_slice_N_tiles.value_or(C_shard_N_tiles));
    }

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
    const uint32_t l1_base = device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t l1_ceiling = device.lowest_occupied_compute_l1_address().value_or(device.l1_size_per_core());
    const uint64_t l1_budget = l1_ceiling - l1_base;

    // ---- 4. C slice candidates ----
    // A sharded C's shard is the C slice (checked above against the config's). Otherwise a C slice dim the config gives
    // is fixed, and an auto one divides the largest C slice: M / N split over the bounding box of the cores (one C
    // slice per core).
    const uint32_t core_rows = bounding_box.grid_size().y;
    const uint32_t core_columns = bounding_box.grid_size().x;
    const bool M_fixed = C_shard.has_value() || config.C_slice_M_tiles.has_value();
    const bool N_fixed = C_shard.has_value() || config.C_slice_N_tiles.has_value();
    const uint32_t max_C_slice_M_tiles = C_shard.has_value()                  ? C_shard->shape[0] / TILE_HEIGHT
                                         : config.C_slice_M_tiles.has_value() ? *config.C_slice_M_tiles
                                                                              : tt::div_up(base.M_tiles, core_rows);
    const uint32_t max_C_slice_N_tiles = C_shard.has_value()                  ? C_shard->shape[1] / TILE_WIDTH
                                         : config.C_slice_N_tiles.has_value() ? *config.C_slice_N_tiles
                                                                              : tt::div_up(base.N_tiles, core_columns);

    // Largest area first, then wider (no two candidates tie on both).
    std::vector<std::pair<uint32_t, uint32_t>> C_slice_shapes;
    for (uint32_t h = max_C_slice_M_tiles; h >= 1; --h) {
        for (uint32_t w = max_C_slice_N_tiles; w >= 1; --w) {
            const bool h_allowed = M_fixed ? h == max_C_slice_M_tiles : max_C_slice_M_tiles % h == 0;
            const bool w_allowed = N_fixed ? w == max_C_slice_N_tiles : max_C_slice_N_tiles % w == 0;
            if (h_allowed && w_allowed) {
                C_slice_shapes.emplace_back(h, w);
            }
        }
    }
    std::sort(C_slice_shapes.begin(), C_slice_shapes.end(), [](const auto& a, const auto& b) {
        const uint64_t area_a = (uint64_t)a.first * a.second;
        const uint64_t area_b = (uint64_t)b.first * b.second;
        return area_a != area_b ? area_a > area_b : a.second > b.second;
    });
    // ---- 5. Search: each C slice's subblock, then its K chunk ----
    // The first C slice that fits wins, except one that fits only with K_chunk_tiles == 1 yields to a smaller C slice
    // with a longer K chunk and more work per L1 fill (C slice tiles x K chunk); the largest C slice is never shrunk if
    // it fits (#57884).
    const uint32_t K_chunk_floor = config.K_chunk_tiles.value_or(1);  // the shortest K chunk tried
    const uint32_t min_K_chunks = is_quasar ? QUASAR_NUM_READER_THREADS : MIN_OPERAND_BUFFER_DEPTH;
    std::optional<MatmulUnifiedPlan> best;
    uint64_t best_work = 0;
    MatmulUnifiedPlan candidate;  // after the loop, the last plan tried: the L1 error reports it
    for (const std::pair<uint32_t, uint32_t>& C_slice_shape : C_slice_shapes) {
        const uint32_t C_slice_M_tiles = C_slice_shape.first;
        const uint32_t C_slice_N_tiles = C_slice_shape.second;

        // Subblock: the config's, else the largest that fills DST, voids no achievable borrow and fits L1 at the
        // shortest K chunk, so the K chunk search below always finds one. Never trade a borrow for subblock volume:
        // a subblock that pads a borrowable shard's dims is rejected; C's borrow needs subblocks spanning the C slice
        // width, achievable only when that width fits DST.
        uint32_t subblock_M_tiles = config.subblock_M_tiles.value_or(0);
        uint32_t subblock_N_tiles = config.subblock_N_tiles.value_or(0);
        if (!config.subblock_M_tiles.has_value()) {
            const auto subblock_viable = [&](uint32_t candidate_M_tiles, uint32_t candidate_N_tiles) {
                const MatmulUnifiedPlan at_shortest_K_chunk = make_plan(
                    base,
                    config.cores,
                    A,
                    B,
                    output,
                    attributes,
                    C_slice_M_tiles,
                    C_slice_N_tiles,
                    candidate_M_tiles,
                    candidate_N_tiles,
                    K_chunk_floor);
                const bool M_padded = at_shortest_K_chunk.C_slice_M_padded_tiles != C_slice_M_tiles;
                const bool N_padded = at_shortest_K_chunk.C_slice_N_padded_tiles != C_slice_N_tiles;
                const bool C_borrow_achievable =
                    at_shortest_K_chunk.C_shard_borrowable && C_slice_N_tiles <= dst_capacity_tiles;
                const bool voids_A_borrow = at_shortest_K_chunk.A_shard_borrowable && M_padded;
                const bool voids_B_borrow = at_shortest_K_chunk.B_shard_borrowable && N_padded;
                const bool voids_C_borrow = C_borrow_achievable && !at_shortest_K_chunk.borrow_C;
                return !voids_A_borrow && !voids_B_borrow && !voids_C_borrow &&
                       dfbs_fit(at_shortest_K_chunk, l1_budget);
            };
            std::tie(subblock_M_tiles, subblock_N_tiles) = detail::maximize_subblock_size(
                C_slice_M_tiles, C_slice_N_tiles, dst_capacity_tiles, base.num_compute_threads, subblock_viable);
        }

        // K chunk: all of K when that lets a resident A shard be borrowed (main pins in0_block_w == K for
        // height-sharded in0 for the same reason), else the config's, else the longest divisor of K_tiles that fits,
        // walking down from the cap (see MAX_AUTO_K_CHUNK_TILES).
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
            base.K_tiles);
        const bool single_K_chunk_allows_A_borrow =
            !config.K_chunk_tiles.has_value() && candidate.borrow_A && dfbs_fit(candidate, l1_budget);
        if (!single_K_chunk_allows_A_borrow) {
            const uint32_t K_chunk_cap = C_slice_M_tiles * C_slice_N_tiles > SMALL_C_SLICE_TILES
                                             ? MAX_AUTO_K_CHUNK_TILES
                                             : std::max(MAX_AUTO_K_CHUNK_TILES, base.K_tiles / min_K_chunks);
            const uint32_t K_chunk_start =
                config.K_chunk_tiles.has_value() ? *config.K_chunk_tiles : std::min(base.K_tiles, K_chunk_cap);
            for (uint32_t K_chunk_tiles = K_chunk_start; K_chunk_tiles >= K_chunk_floor; --K_chunk_tiles) {
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
                    K_chunk_tiles);
                if (dfbs_fit(candidate, l1_budget)) {
                    break;
                }
            }
        }

        // Keep the C slice with the most work per L1 fill.
        if (!dfbs_fit(candidate, l1_budget)) {
            continue;
        }
        const uint64_t work = (uint64_t)C_slice_M_tiles * C_slice_N_tiles * candidate.K_chunk_tiles;
        if (work > best_work) {
            best = candidate;
            best_work = work;
        }
        const bool is_largest = C_slice_M_tiles == max_C_slice_M_tiles && C_slice_N_tiles == max_C_slice_N_tiles;
        if (best->K_chunk_tiles > 1 || is_largest) {
            break;
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
    MatmulUnifiedPlan plan = std::move(*best);

    // ---- 6. Sharded output: one C slice per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        const bool one_C_slice_per_core = plan.max_C_slices_per_core == 1;
        const uint32_t C_slices_down_M = tt::div_up(plan.M_tiles, plan.C_slice_M_tiles);
        const std::vector<CoreRange>& ranges = config.cores.ranges();
        const bool block_grid_matches = plan.sharded_output_layout != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                                        (ranges.size() == 1 && ranges[0].grid_size().x == plan.C_slices_across_N &&
                                         ranges[0].grid_size().y == C_slices_down_M);
        if (!must_fit && (plan.batch_size != 1 || !one_C_slice_per_core || !block_grid_matches)) {
            return std::nullopt;
        }
        TT_FATAL(plan.batch_size == 1, "Sharded output needs batch 1 (a core's C slices would not form one shard)");
        TT_FATAL(
            one_C_slice_per_core,
            "Sharded output needs exactly one C slice per core ({} C slices, {} active cores)",
            plan.C_slices_per_batch,
            plan.cores.size());
        TT_FATAL(
            block_grid_matches,
            "Block-sharded output needs cores to be one {}x{} rectangle (one core per C slice, laid out as "
            "the C slices tile C), got {}",
            plan.C_slices_across_N,
            C_slices_down_M,
            config.cores.str());
    }

    return plan;
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Estimated cycles of a plan: the slowest of the busiest core's compute and data movement and the device's DRAM
// traffic (the buffers overlap them), plus a fixed cost per C slice. Second, to break ties: the three summed
// (decode reads the same DRAM bytes on 8 cores as on 32).
std::pair<double, double> estimate_plan_cycles(
    const MatmulUnifiedPlan& plan,
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

    // Compute: the busiest thread's subblocks of each C slice over all of K, plus its packs and partial reloads.
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

// The cores next to the DRAM banks, in bank order, which the C slice walk follows: each reads its own bank's shard.
CoreRangeSet dram_bank_cores(const tt::tt_metal::IDevice& device) {
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

    const tt::tt_metal::IDevice& device = *A.device();
    const CoreCoord device_grid = device.compute_with_storage_grid_size();
    const CoreCoord grid = attributes.user_core_coord.value_or(device_grid);
    const uint32_t grid_columns = grid.x;
    const uint32_t grid_rows = grid.y;
    // With a 32-bit DST ttnn.linear adds the bias separately (see the plan), so it takes no L1 here.
    const uint32_t fused_bias_tile_bytes = get_fp32_dest_acc_en(attributes.compute_kernel_config) ? 0 : bias_tile_bytes;
    const tt::tt_metal::MathFidelity math_fidelity =
        std::get<0>(get_compute_kernel_config_args(device.arch(), attributes.compute_kernel_config.value()));
    const MatmulUnifiedPlan gemm = plan_gemm_size(A, B, attributes);
    const uint32_t M_tiles = gemm.M_tiles;
    const uint32_t N_tiles = gemm.N_tiles;
    const tt::tt_metal::MemoryConfig& C_memory = attributes.output_mem_config;
    const std::optional<tt::tt_metal::ShardSpec>& C_shard = C_memory.shard_spec();

    std::optional<MatmulUnifiedProgramConfig> best;
    std::pair<double, double> best_estimate;
    // Plans the placement (an unset C slice: the plan picks it) and keeps it if its estimate is the lowest so far, the
    // first on a tie. False when it does not fit: cores beyond the device grid (a larger user_core_coord), DFBs beyond
    // L1, or C slices that do not tile a sharded output in the layout it asks for.
    const auto try_placement = [&](const CoreRangeSet& cores,
                                   tt::tt_metal::ShardOrientation orientation,
                                   std::optional<uint32_t> C_slice_M_tiles,
                                   std::optional<uint32_t> C_slice_N_tiles) {
        const CoreCoord last_core = cores.bounding_box().end_coord;
        if (last_core.x >= device_grid.x || last_core.y >= device_grid.y) {
            return false;
        }
        MatmulUnifiedProgramConfig config = requested;
        config.cores = cores;
        config.orientation = orientation;
        config.C_slice_M_tiles = C_slice_M_tiles;
        config.C_slice_N_tiles = C_slice_N_tiles;
        const std::optional<MatmulUnifiedPlan> plan =
            plan_unified_matmul(A, B, fused_bias_tile_bytes, config, attributes, std::nullopt, /*must_fit=*/false);
        if (!plan.has_value()) {
            return false;
        }
        if (C_memory.is_sharded() && !C_shard.has_value() && plan->sharded_output_layout != C_memory.memory_layout()) {
            return false;
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
            "Unified matmul placement: {}x{}-tile C slice on {}: {:.0f} cycles",
            plan->C_slice_M_tiles,
            plan->C_slice_N_tiles,
            cores.str(),
            estimate.first);
        if (!best.has_value() || estimate < best_estimate) {
            best = std::move(config);
            best_estimate = estimate;
        }
        return true;
    };

    // A 1D placement's C slices are columns (C_slices_are_columns) or rows of C, C_slice_width_tiles wide and as long
    // as all of C's other dimension, or if that does not fit L1, 1/2, 1/3, ... of it, rounded up (13 tiles in 4 C
    // slices: 4, 4, 4 and 1).
    const auto try_1D_placement = [&](const CoreRangeSet& cores,
                                      tt::tt_metal::ShardOrientation orientation,
                                      uint32_t C_slice_width_tiles,
                                      bool C_slices_are_columns) {
        const uint32_t C_length_tiles = C_slices_are_columns ? M_tiles : N_tiles;
        for (uint32_t C_slices_along_length = 1; C_slices_along_length <= C_length_tiles; ++C_slices_along_length) {
            const uint32_t C_slice_length_tiles = tt::div_up(C_length_tiles, C_slices_along_length);
            const bool fits = C_slices_are_columns
                                  ? try_placement(cores, orientation, C_slice_length_tiles, C_slice_width_tiles)
                                  : try_placement(cores, orientation, C_slice_width_tiles, C_slice_length_tiles);
            if (fits) {
                return;
            }
        }
    };

    // A sharded output's shard grid fixes the cores, and the plan takes the C slice from the shard: the only placement.
    if (C_memory.is_sharded() && C_shard.has_value()) {
        try_placement(C_shard->grid, C_shard->orientation, std::nullopt, std::nullopt);
        return best;
    }

    constexpr tt::tt_metal::ShardOrientation ROW_MAJOR = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    // 2D: a rectangle of cores over C's rows and columns, the plan picking the C slice.
    try_placement(
        CoreRangeSet(CoreRange({0, 0}, {std::min(grid_columns, N_tiles) - 1, std::min(grid_rows, M_tiles) - 1})),
        ROW_MAJOR,
        std::nullopt,
        std::nullopt);

    // 1D across N (M): whole grid rows (part of one row for a narrow C) splitting C's columns (rows). With one C slice
    // per core every core needs the same A (B), which is multicast to all of them.
    const CoreRangeSet across_N_cores(CoreRange(
        {0, 0}, {std::min(grid_columns, N_tiles) - 1, std::clamp(N_tiles / grid_columns, 1u, grid_rows) - 1}));
    const CoreRangeSet across_M_cores(CoreRange(
        {0, 0}, {std::min(grid_columns, M_tiles) - 1, std::clamp(M_tiles / grid_columns, 1u, grid_rows) - 1}));
    try_1D_placement(
        across_N_cores, ROW_MAJOR, tt::div_up(N_tiles, across_N_cores.num_cores()), /*C_slices_are_columns=*/true);
    try_1D_placement(
        across_M_cores, ROW_MAJOR, tt::div_up(M_tiles, across_M_cores.num_cores()), /*C_slices_are_columns=*/false);

    // 1D on a sharded operand's cores, which keeps it where it is: an L1 shard can then be bound in place, and a DRAM
    // shard is read by the core next to its bank.
    if (A.is_sharded() && A.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        try_1D_placement(
            A.memory_config().is_dram() ? dram_bank_cores(device) : A.shard_spec()->grid,
            A.shard_spec()->orientation,
            A.shard_spec()->shape[0] / TILE_HEIGHT,
            /*C_slices_are_columns=*/false);
    }
    if (B.is_sharded() && B.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        try_1D_placement(
            B.memory_config().is_dram() ? dram_bank_cores(device) : B.shard_spec()->grid,
            B.shard_spec()->orientation,
            B.shard_spec()->shape[1] / TILE_WIDTH,
            /*C_slices_are_columns=*/true);
    }
    return best;
}

}  // namespace ttnn::prim::qsr
