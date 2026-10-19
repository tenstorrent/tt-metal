// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_unified_plan.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// A TRISC-visible DFB's extent is a uint16_t of 16-byte units (validate_ring_extent); enforced on
// every arch so a Wormhole-legal config never FATALs on Quasar.
constexpr uint64_t MAX_DFB_EXTENT_BYTES = 65535ull * 16ull;
// Auto K chunk search (#57884): walk down from this cap through divisors of K_tiles to the first that fits
// L1, except a C slice of at most SMALL_C_SLICE_TILES tiles is not capped (its A and B slices are cheap,
// so a long K chunk saves partial-sum round trips) beyond leaving a K chunk per reader thread to overlap the
// compute. Tuning points, not hardware limits.
constexpr uint32_t MAX_AUTO_K_CHUNK_TILES = 8;
constexpr uint32_t SMALL_C_SLICE_TILES = 4;
// DM threads and buffering per core. Quasar runs up to four reader and two writer threads (its six user DM cores);
// Wormhole / Blackhole one of each. A and B slices in flight: one per reader thread, at least two; C slices in flight
// on cores that produce several.
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
bool dfbs_fit(const UnifiedMatmulPlan& plan, uint64_t l1_budget) {
    const uint64_t dfb_bytes[] = {
        (uint64_t)plan.A_slice_entries * plan.A_entry_bytes,
        (uint64_t)plan.B_slice_entries * plan.B_entry_bytes,
        (uint64_t)plan.C_slice_entries * plan.C_entry_bytes,
        (uint64_t)plan.C_partials_entries * plan.C_partials_entry_bytes,
        (uint64_t)plan.bias_entries * plan.bias_entry_bytes};
    for (uint64_t bytes : dfb_bytes) {
        if (bytes > MAX_DFB_EXTENT_BYTES) {
            return false;
        }
    }
    return plan.l1_bytes <= l1_budget;
}

// Completes a candidate plan for one K chunk: chunking, formats, DFB entry counts and byte totals.
// Takes the plan by value so the K chunk search can size several candidates.
UnifiedMatmulPlan size_dfbs(
    UnifiedMatmulPlan plan,
    uint32_t K_chunk_tiles,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    bool A_borrowable,
    bool B_borrowable,
    bool C_borrowable) {
    plan.K_chunk_tiles = K_chunk_tiles;
    plan.K_chunks_per_C_slice = plan.K_tiles / K_chunk_tiles;
    plan.K_chunks_per_C_slice_padded = tt::round_up(plan.K_chunks_per_C_slice, plan.num_reader_threads);

    // The packer accumulates partials in L1 only when there are enough K chunks for the reconfig overhead
    // to pay off (the last K chunk spills and reloads either way, so more than two).
    plan.packer_l1_acc_en = packer_l1_acc && plan.K_chunks_per_C_slice > 2;
    plan.C_partials_format = plan.packer_l1_acc_en
                                 ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                 : (fp32_dest_acc_en ? tt::DataFormat::Float32 : plan.C_format);
    // A thread's share of the C slice (its subblocks back to back, padded to the busiest thread's) is
    // C_entries_per_thread one-tile entries of C_slice and C_partials, pushed and popped as one batch per K chunk:
    // every thread then moves the same credits, and its C_partials entries come back to the same addresses every
    // K chunk, where the packer's L1 accumulation needs them.
    const uint32_t num_subblocks =
        (plan.C_slice_M_padded_tiles / plan.subblock_M_tiles) * (plan.C_slice_N_padded_tiles / plan.subblock_N_tiles);
    plan.C_entries_per_thread =
        tt::div_up(num_subblocks, plan.num_compute_threads) * plan.subblock_M_tiles * plan.subblock_N_tiles;
    plan.C_entry_bytes = tt::tile_size(plan.C_format);
    plan.C_partials_entry_bytes = tt::tile_size(plan.C_partials_format);
    plan.C_slice_entries = plan.C_entries_per_thread * plan.num_compute_threads * plan.C_buffer_depth;

    // A copied operand's DFB holds operand_buffer_depth slices, one when a single slice passes through; a borrowed
    // DFB is the resident shard itself. A is borrowable only when one K chunk covers K.
    const bool more_than_one_slice =
        (uint64_t)plan.batch_size * plan.max_C_slices_per_core * plan.K_chunks_per_C_slice_padded > 1;
    plan.operand_buffer_depth = more_than_one_slice ? std::max(MIN_OPERAND_BUFFER_DEPTH, plan.num_reader_threads) : 1;
    plan.borrow_A = A_borrowable && plan.K_chunks_per_C_slice == 1;
    plan.borrow_B = B_borrowable;
    plan.borrow_C = C_borrowable;
    plan.A_entry_bytes = tt::tile_size(plan.A_format);
    plan.B_entry_bytes = tt::tile_size(plan.B_format);
    plan.A_slice_entries = plan.borrow_A ? plan.C_slice_M_tiles * plan.K_tiles
                                         : plan.C_slice_M_padded_tiles * K_chunk_tiles * plan.operand_buffer_depth;
    plan.B_slice_entries = plan.borrow_B ? plan.K_tiles * plan.C_slice_N_tiles
                                         : K_chunk_tiles * plan.C_slice_N_padded_tiles * plan.operand_buffer_depth;

    // Alias C_partials onto C_slice only when partials are never live while C_slice holds unread data
    // (else compute packs slice i+1's partials over slice i before the writer drains it).
    const bool partials_ever_written = plan.K_chunks_per_C_slice > 1;
    const bool one_C_slice_per_core = plan.batch_size == 1 && plan.max_C_slices_per_core == 1;
    plan.alias_C_partials_onto_C_slice = (plan.C_partials_format == plan.C_format) && plan.C_buffer_depth == 1 &&
                                         (!partials_ever_written || one_C_slice_per_core);
    // Partials that are never written need one entry per thread when they cannot share C_slice's L1.
    plan.C_partials_entries = (partials_ever_written || plan.alias_C_partials_onto_C_slice)
                                  ? plan.C_entries_per_thread * plan.num_compute_threads
                                  : plan.num_compute_threads;
    // One C slice's bias per reader thread, and at least two when the core has several, so a single reader reads the
    // next C slice's bias (and the A and B slices behind it) while compute uses this one.
    const uint32_t bias_buffer_depth =
        std::max(plan.num_reader_threads, one_C_slice_per_core ? 1u : MIN_OPERAND_BUFFER_DEPTH);
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

std::optional<UnifiedMatmulPlan> plan_unified_matmul(
    const tt::tt_metal::IDevice& device,
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    uint32_t bias_tile_bytes,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output,
    bool must_fit) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    UnifiedMatmulPlan base{};

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
            attributes.output_tile->get_height() == TILE_HEIGHT && attributes.output_tile->get_width() == TILE_WIDTH,
            "MatmulUnifiedProgramConfig supports a 32x32 output tile only");
    }

    // ---- GEMM size ----
    base.batch_size = get_batch_size(A_shape);
    base.M_tiles = A_shape[-2] / TILE_HEIGHT;
    base.K_tiles = A_shape[-1] / TILE_WIDTH;
    base.N_tiles = B_shape[-1] / TILE_WIDTH;
    const bool B_has_one_batch = attributes.bcast_batch.value();  // the op sets bcast_batch iff batch_size(B) == 1
    base.B_batch_stride_tiles = B_has_one_batch ? 0 : base.K_tiles * base.N_tiles;
    TT_FATAL(
        B_has_one_batch || get_batch_size(B_shape) == base.batch_size,
        "Batched B must match A's batch ({} vs {})",
        get_batch_size(B_shape),
        base.batch_size);
    if (B_has_one_batch) {
        // Every batch of A multiplies the same B, and A's and C's batches are stacked tile rows: one [batch * M x N]
        // GEMM, whose C slices spread over the cores like any other rows.
        base.M_tiles *= base.batch_size;
        base.batch_size = 1;
    }
    base.A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;

    // ---- Cores ----
    TT_FATAL(config.cores.num_cores() > 0, "MatmulUnifiedProgramConfig.cores is empty");
    const CoreCoord grid = device.compute_with_storage_grid_size();
    const CoreRange bounding_box = config.cores.bounding_box();
    TT_FATAL(
        bounding_box.end_coord.x < grid.x && bounding_box.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    base.orientation = config.orientation;
    const std::vector<CoreCoord> all_cores =
        corerange_to_cores(config.cores, std::nullopt, config.orientation == ShardOrientation::ROW_MAJOR);

    // ---- Compute threads ----
    const bool is_quasar = device.arch() == tt::ARCH::QUASAR;
    const std::size_t requested_threads = config.num_compute_threads;  // checked before it narrows to uint32_t
    TT_FATAL(
        requested_threads == 0 || requested_threads == 1 || requested_threads == 2 || requested_threads == 4,
        "MatmulUnifiedProgramConfig.num_compute_threads must be 0 (auto), 1, 2 or 4, got {}",
        requested_threads);
    base.num_compute_threads = requested_threads != 0 ? static_cast<uint32_t>(requested_threads) : (is_quasar ? 4 : 1);
    TT_FATAL(
        base.num_compute_threads == 1 || is_quasar,
        "MatmulUnifiedProgramConfig.num_compute_threads = {} needs Quasar; this device has one compute engine per "
        "core",
        base.num_compute_threads);

    // ---- Writer threads: thread t writes the shares of compute threads t, t + num_writer_threads, ... ----
    base.num_writer_threads =
        is_quasar && base.num_compute_threads % QUASAR_NUM_WRITER_THREADS == 0 ? QUASAR_NUM_WRITER_THREADS : 1;
    // The C slice, subblock and K chunk are chosen with one reader thread and one C slice in flight; more are added
    // at the end only where they fit.
    base.num_reader_threads = 1;
    base.C_buffer_depth = 1;

    // ---- Accumulation mode, formats, L1 budget ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    base.fp32_dest_acc_en = fp32_dest_acc_en;
    TT_FATAL(
        !(bias_tile_bytes > 0 && fp32_dest_acc_en),
        "MatmulUnifiedProgramConfig fuses bias only with a 16-bit DST (fp32_dest_acc_en = false); ttnn.linear adds it "
        "separately otherwise");
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(device.arch(), attributes.compute_kernel_config.value()));
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    base.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    base.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    base.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    // Shape, layout and batch of the bias are checked by the op; it is one tile row of C's columns.
    base.has_bias = bias_tile_bytes > 0;
    base.bias_entry_bytes = bias_tile_bytes;
    const uint32_t l1_base = device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t l1_ceiling = device.lowest_occupied_compute_l1_address().value_or(device.l1_size_per_core());
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    const uint64_t l1_budget = l1_ceiling - l1_base;

    TT_FATAL(
        (config.subblock_M_tiles == 0) == (config.subblock_N_tiles == 0),
        "subblock_M_tiles and subblock_N_tiles must both be set or both be 0 (auto), got {}x{}",
        config.subblock_M_tiles,
        config.subblock_N_tiles);
    TT_FATAL(
        config.K_chunk_tiles == 0 || base.K_tiles % config.K_chunk_tiles == 0,
        "K_chunk_tiles ({}) must divide K_tiles ({})",
        config.K_chunk_tiles,
        base.K_tiles);

    // A shard matches when the tensor is L1-sharded with that shard shape and its grid lists the active
    // cores in assignment order (so shard i lives on the core that produces C slice i).
    const auto shard_matches = [&](const ttnn::Tensor& tensor,
                                   uint32_t shard_M_tiles,
                                   uint32_t shard_N_tiles,
                                   const std::vector<CoreCoord>& cores) {
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
    };

    // Everything downstream of the C slice: core assignment, borrowability, subblock, K chunk and DFB
    // sizing. Returns nullopt when no K chunk fits L1 unless must_fit, which FATALs with the breakdown.
    const auto complete_plan =
        [&](uint32_t C_slice_M_tiles, uint32_t C_slice_N_tiles, bool must_fit) -> std::optional<UnifiedMatmulPlan> {
        UnifiedMatmulPlan plan = base;
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

        // ---- Borrowing prerequisites that do not depend on the subblock ----
        const bool one_C_slice_per_core_no_batch = plan.batch_size == 1 && plan.C_slices_per_batch == plan.cores.size();
        // A: the C slice's rows for all of K; C slices must span N so no two cores need the same rows. The copy
        // path zeroes A's K padding in the DFB; a borrowed shard is never written, so K must be a tile multiple.
        const bool A_shard_borrowable =
            one_C_slice_per_core_no_batch && C_slices_across_N == 1 && plan.A_last_K_tile_valid_columns == 0 &&
            shard_matches(A, C_slice_M_tiles, plan.K_tiles, plan.cores) &&
            (uint64_t)C_slice_M_tiles * plan.K_tiles * tt::tile_size(plan.A_format) <= MAX_DFB_EXTENT_BYTES;
        // B: the C slice's columns for all of K; C slices must span M.
        const bool B_shard_borrowable =
            one_C_slice_per_core_no_batch && C_slices_down_M == 1 &&
            shard_matches(B, plan.K_tiles, C_slice_N_tiles, plan.cores) &&
            (uint64_t)plan.K_tiles * C_slice_N_tiles * tt::tile_size(plan.B_format) <= MAX_DFB_EXTENT_BYTES;
        // C: packed straight into the shard when subblock-major pack order equals the shard's row-major order,
        // which also needs a single compute thread (with several, each thread owns every N-th entry).
        const bool C_shard_matches = output.has_value()
                                         ? shard_matches(output.value(), C_slice_M_tiles, C_slice_N_tiles, plan.cores)
                                         : (attributes.output_mem_config.is_sharded() &&
                                            attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1);
        const bool C_shard_borrowable =
            C_shard_matches && one_C_slice_per_core_no_batch && plan.num_compute_threads == 1;

        // ---- Multicast ----
        // One C slice per core on a rectangle laid out as the C slices tile C (walked row-major): the cores of a row
        // then need the same A slice at the same time, and the cores of a column the same B slice. When C is a single
        // row of C slices every core of any rectangle needs the same A slice (one column: the same B slice).
        const std::vector<CoreRange>& core_ranges = config.cores.ranges();
        const bool one_C_slice_per_core_of_rectangle = core_ranges.size() == 1 &&
                                                       plan.cores.size() == plan.C_slices_per_batch &&
                                                       plan.cores.size() == core_ranges[0].size();
        const bool cores_tile_C =
            one_C_slice_per_core_of_rectangle && core_ranges[0].grid_size().x == C_slices_across_N &&
            core_ranges[0].grid_size().y == C_slices_down_M &&
            (config.orientation == ShardOrientation::ROW_MAJOR || C_slices_across_N == 1 || C_slices_down_M == 1);
        plan.A_mcast_to_all_cores = one_C_slice_per_core_of_rectangle && !cores_tile_C && C_slices_down_M == 1;
        plan.B_mcast_to_all_cores = one_C_slice_per_core_of_rectangle && !cores_tile_C && C_slices_across_N == 1;
        const uint32_t A_mcast_group = cores_tile_C                ? C_slices_across_N
                                       : plan.A_mcast_to_all_cores ? plan.cores.size()
                                                                   : 1;
        const uint32_t B_mcast_group = cores_tile_C                ? C_slices_down_M
                                       : plan.B_mcast_to_all_cores ? plan.cores.size()
                                                                   : 1;
        plan.A_mcast_num_dests = A_shard_borrowable ? 0 : A_mcast_group - 1;
        plan.B_mcast_num_dests = B_shard_borrowable ? 0 : B_mcast_group - 1;

        // ---- Subblock: the C slice's tiles accumulated in DST at once ----
        // A candidate is viable when it voids no achievable borrow and its DFBs fit L1, sized at the K
        // chunk the search below bottoms out at, so an accepted candidate is guaranteed to fit.
        const uint32_t K_chunk_floor = config.K_chunk_tiles == 0 ? 1 : config.K_chunk_tiles;
        const auto subblock_viable = [&](uint32_t subblock_M_tiles, uint32_t subblock_N_tiles) {
            UnifiedMatmulPlan candidate = plan;
            candidate.subblock_M_tiles = subblock_M_tiles;
            candidate.subblock_N_tiles = subblock_N_tiles;
            candidate.C_slice_M_padded_tiles = tt::round_up(C_slice_M_tiles, subblock_M_tiles);
            candidate.C_slice_N_padded_tiles = tt::round_up(C_slice_N_tiles, subblock_N_tiles);
            const bool M_padded = candidate.C_slice_M_padded_tiles != C_slice_M_tiles;
            const bool N_padded = candidate.C_slice_N_padded_tiles != C_slice_N_tiles;
            // A borrowed shard holds only the true slice dims, so a candidate that voids an achievable
            // borrow is rejected outright: never trade a borrow for subblock volume. C's borrow needs
            // subblocks spanning the C slice width, achievable only when that width fits DST.
            const bool C_borrow_achievable = C_shard_borrowable && C_slice_N_tiles <= dst_capacity_tiles;
            const bool C_borrow_kept = subblock_N_tiles == C_slice_N_tiles && !M_padded;
            const bool voids_A_borrow = A_shard_borrowable && M_padded;
            const bool voids_B_borrow = B_shard_borrowable && N_padded;
            const bool voids_C_borrow = C_borrow_achievable && !C_borrow_kept;
            if (voids_A_borrow || voids_B_borrow || voids_C_borrow) {
                return false;
            }
            return dfbs_fit(
                size_dfbs(
                    std::move(candidate),
                    K_chunk_floor,
                    fp32_dest_acc_en,
                    packer_l1_acc,
                    /*A_borrowable=*/A_shard_borrowable,
                    /*B_borrowable=*/B_shard_borrowable,
                    /*C_borrowable=*/C_shard_borrowable && C_borrow_kept),
                l1_budget);
        };
        if (config.subblock_M_tiles != 0) {
            plan.subblock_M_tiles = config.subblock_M_tiles;
            plan.subblock_N_tiles = config.subblock_N_tiles;
        } else {
            std::tie(plan.subblock_M_tiles, plan.subblock_N_tiles) = detail::maximize_subblock_size(
                C_slice_M_tiles, C_slice_N_tiles, dst_capacity_tiles, plan.num_compute_threads, subblock_viable);
        }
        plan.C_slice_M_padded_tiles = tt::round_up(C_slice_M_tiles, plan.subblock_M_tiles);
        plan.C_slice_N_padded_tiles = tt::round_up(C_slice_N_tiles, plan.subblock_N_tiles);
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
        const bool A_borrowable = A_shard_borrowable && plan.C_slice_M_padded_tiles == C_slice_M_tiles;
        const bool B_borrowable = B_shard_borrowable && plan.C_slice_N_padded_tiles == C_slice_N_tiles;
        const bool C_borrowable = C_shard_borrowable && plan.subblock_N_tiles == C_slice_N_tiles &&
                                  plan.C_slice_M_padded_tiles == C_slice_M_tiles;

        // ---- K chunk and DFB sizing ----
        if (config.K_chunk_tiles != 0) {
            UnifiedMatmulPlan candidate = size_dfbs(
                plan, config.K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
            const bool fits = dfbs_fit(candidate, l1_budget);
            TT_FATAL(
                fits || !must_fit,
                "MatmulUnifiedProgramConfig: DFBs for a {}x{}-tile C slice with K_chunk_tiles={} do not fit "
                "(needs {} B, budget {} B, max DFB extent {} B: A slice {} B, B slice {} B, C slice {} B, C partials "
                "{} B, bias {} B)",
                C_slice_M_tiles,
                C_slice_N_tiles,
                candidate.K_chunk_tiles,
                candidate.l1_bytes,
                l1_budget,
                MAX_DFB_EXTENT_BYTES,
                (uint64_t)candidate.A_slice_entries * candidate.A_entry_bytes,
                (uint64_t)candidate.B_slice_entries * candidate.B_entry_bytes,
                (uint64_t)candidate.C_slice_entries * candidate.C_entry_bytes,
                (uint64_t)candidate.C_partials_entries * candidate.C_partials_entry_bytes,
                (uint64_t)candidate.bias_entries * candidate.bias_entry_bytes);
            return fits ? std::optional<UnifiedMatmulPlan>(std::move(candidate)) : std::nullopt;
        }
        // A resident A shard is only borrowable with a single K chunk, so try that first (main pins
        // in0_block_w == K for height-sharded in0 for the same reason).
        if (A_borrowable) {
            UnifiedMatmulPlan candidate = size_dfbs(
                plan, plan.K_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
            if (candidate.borrow_A && dfbs_fit(candidate, l1_budget)) {
                return candidate;
            }
        }
        // Otherwise the largest divisor of K_tiles whose DFBs fit, walking down from the cap (#57884: a
        // small C slice makes the A and B slices cheap, so it is not capped, but keeps the readers busy).
        const uint32_t min_K_chunks = is_quasar ? QUASAR_NUM_READER_THREADS : MIN_OPERAND_BUFFER_DEPTH;
        const uint32_t K_chunk_start = std::min<uint32_t>(
            plan.K_tiles,
            C_slice_M_tiles * C_slice_N_tiles > SMALL_C_SLICE_TILES
                ? MAX_AUTO_K_CHUNK_TILES
                : std::max(MAX_AUTO_K_CHUNK_TILES, plan.K_tiles / min_K_chunks));
        for (uint32_t K_chunk_tiles = K_chunk_start; K_chunk_tiles >= 1; --K_chunk_tiles) {
            if (plan.K_tiles % K_chunk_tiles != 0) {
                continue;
            }
            UnifiedMatmulPlan candidate = size_dfbs(
                plan, K_chunk_tiles, fp32_dest_acc_en, packer_l1_acc, A_borrowable, B_borrowable, C_borrowable);
            if (dfbs_fit(candidate, l1_budget)) {
                return candidate;
            }
        }
        TT_FATAL(
            !must_fit,
            "MatmulUnifiedProgramConfig: a {}x{}-tile C slice does not fit L1 even with K_chunk_tiles=1 "
            "(budget {} B, max DFB extent {} B); shrink C_slice_M_tiles / C_slice_N_tiles",
            C_slice_M_tiles,
            C_slice_N_tiles,
            l1_budget,
            MAX_DFB_EXTENT_BYTES);
        return std::nullopt;
    };

    // ---- C slice ----
    // Each core's share of C: the output shard when C is sharded, else M / N split over the bounding box of
    // the cores. An explicit or shard-derived C slice is taken as is; an auto one is the largest piece of the
    // share that fits L1 (#57884).
    const std::optional<tt::tt_metal::ShardSpec>& C_shard =
        output.has_value() ? output->memory_config().shard_spec() : attributes.output_mem_config.shard_spec();
    const bool C_slice_from_shard = C_shard.has_value() && (config.C_slice_M_tiles == 0 || config.C_slice_N_tiles == 0);
    uint32_t C_shard_M_tiles = 0;
    uint32_t C_shard_N_tiles = 0;
    if (C_slice_from_shard) {
        const uint32_t C_shard_height = C_shard->shape[0];
        const uint32_t C_shard_width = C_shard->shape[1];
        TT_FATAL(
            C_shard_height % TILE_HEIGHT == 0 && C_shard_width % TILE_WIDTH == 0,
            "A sharded C needs a tile-multiple shard shape to derive the C slice from, got {}x{}",
            C_shard_height,
            C_shard_width);
        C_shard_M_tiles = C_shard_height / TILE_HEIGHT;
        C_shard_N_tiles = C_shard_width / TILE_WIDTH;
    }
    const uint32_t core_rows = bounding_box.grid_size().y;
    const uint32_t core_columns = bounding_box.grid_size().x;
    const uint32_t share_M_tiles = config.C_slice_M_tiles != 0 ? config.C_slice_M_tiles
                                   : C_slice_from_shard        ? C_shard_M_tiles
                                                               : tt::div_up(base.M_tiles, core_rows);
    const uint32_t share_N_tiles = config.C_slice_N_tiles != 0 ? config.C_slice_N_tiles
                                   : C_slice_from_shard        ? C_shard_N_tiles
                                                               : tt::div_up(base.N_tiles, core_columns);
    const bool M_fixed = config.C_slice_M_tiles != 0 || C_slice_from_shard;
    const bool N_fixed = config.C_slice_N_tiles != 0 || C_slice_from_shard;

    UnifiedMatmulPlan plan;
    if (M_fixed && N_fixed) {
        std::optional<UnifiedMatmulPlan> fixed = complete_plan(share_M_tiles, share_N_tiles, must_fit);
        if (!fixed.has_value()) {
            return std::nullopt;
        }
        plan = std::move(*fixed);
    } else {
        // Candidates are divisor pieces of the share, largest area first (ties: wider). The first that fits
        // wins, except one that fits only with K_chunk_tiles == 1 yields to a smaller piece with a longer K
        // chunk and more work per L1 fill (C slice tiles x K chunk); the whole share is never shrunk if it fits.
        const auto divisors_descending = [](uint32_t n, bool fixed) {
            std::vector<uint32_t> out;
            for (uint32_t d = n; d >= 1; --d) {
                if (fixed ? d == n : n % d == 0) {
                    out.push_back(d);
                }
            }
            return out;
        };
        std::vector<std::pair<uint32_t, uint32_t>> candidates;
        for (uint32_t h : divisors_descending(share_M_tiles, M_fixed)) {
            for (uint32_t w : divisors_descending(share_N_tiles, N_fixed)) {
                candidates.emplace_back(h, w);
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            const uint64_t area_a = (uint64_t)a.first * a.second;
            const uint64_t area_b = (uint64_t)b.first * b.second;
            return area_a != area_b ? area_a > area_b : a.second > b.second;
        });
        std::optional<UnifiedMatmulPlan> best;
        uint64_t best_work = 0;
        for (const auto& [h, w] : candidates) {
            std::optional<UnifiedMatmulPlan> candidate = complete_plan(h, w, /*must_fit=*/false);
            if (!candidate.has_value()) {
                continue;
            }
            const uint64_t work = (uint64_t)h * w * candidate->K_chunk_tiles;
            if (!best.has_value() || work > best_work) {
                best = std::move(candidate);
                best_work = work;
            }
            const bool whole_share = h == share_M_tiles && w == share_N_tiles;
            if (best->K_chunk_tiles > 1 || whole_share) {
                break;
            }
        }
        TT_FATAL(
            best.has_value() || !must_fit,
            "MatmulUnifiedProgramConfig: no piece of the {}x{}-tile per-core share of C fits L1 (budget {} B, max DFB "
            "extent {} B)",
            share_M_tiles,
            share_N_tiles,
            l1_budget,
            MAX_DFB_EXTENT_BYTES);
        if (!best.has_value()) {
            return std::nullopt;
        }
        plan = std::move(*best);
    }

    // ---- Reader threads, then C slices in flight, each in the L1 the C slice, subblock and K chunk leave ----
    // Several reader threads each own part of the A and B DFBs, which a borrowed shard cannot be split into, and
    // reader threads beyond the K chunks of a C slice would only carry padding. Four are tried, then two, which take no
    // more L1 than one: an operand DFB holds at least MIN_OPERAND_BUFFER_DEPTH (two) slices.
    const auto resized = [&](UnifiedMatmulPlan candidate) {
        return size_dfbs(
            std::move(candidate),
            plan.K_chunk_tiles,
            fp32_dest_acc_en,
            packer_l1_acc,
            plan.borrow_A,
            plan.borrow_B,
            plan.borrow_C);
    };
    if (is_quasar && !plan.borrow_A && !plan.borrow_B) {
        for (uint32_t num_reader_threads : {QUASAR_NUM_READER_THREADS, MIN_OPERAND_BUFFER_DEPTH}) {
            UnifiedMatmulPlan candidate = plan;
            candidate.num_reader_threads = std::min(num_reader_threads, plan.K_chunks_per_C_slice);
            candidate = resized(std::move(candidate));
            if (dfbs_fit(candidate, l1_budget)) {
                plan = std::move(candidate);
                break;
            }
        }
    }
    if ((uint64_t)plan.batch_size * plan.max_C_slices_per_core > 1) {
        UnifiedMatmulPlan candidate = plan;
        candidate.C_buffer_depth = C_BUFFER_DEPTH;
        candidate = resized(std::move(candidate));
        if (dfbs_fit(candidate, l1_budget)) {
            plan = std::move(candidate);
        }
    }

    // ---- Sharded output: one C slice per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        const bool one_C_slice_per_core = plan.C_slices_per_batch == plan.cores.size();
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

}  // namespace ttnn::prim::qsr
