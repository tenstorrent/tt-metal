// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"

#include <algorithm>
#include <limits>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

bool is_plain_interleaved(const tt::tt_metal::MemoryConfig& config) {
    const bool on_supported_buffer =
        config.buffer_type() == tt::tt_metal::BufferType::DRAM || config.buffer_type() == tt::tt_metal::BufferType::L1;
    return config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && !config.is_sharded() &&
           !config.nd_shard_spec().has_value() && on_supported_buffer;
}

bool standard_tile_with_tight_padding(const Tensor& input) {
    const auto& tile = input.tensor_spec().tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        tile.get_transpose_of_faces() || tile.get_transpose_within_face()) {
        return false;
    }
    const auto& logical = input.logical_shape();
    const auto& padded = input.padded_shape();
    if (logical.rank() < 2 || padded.rank() != logical.rank()) {
        return false;
    }
    for (size_t i = 0; i + 2 < logical.rank(); ++i) {
        if (padded[i] != logical[i]) {
            return false;
        }
    }
    return padded[-1] == tt::round_up(logical[-1], tile.get_width()) &&
           padded[-2] == tt::round_up(logical[-2], tile.get_height());
}

// Checks that need neither a plan nor a spec, cheapest first.
const char* scope_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || !input.is_allocated()) {
        return "input is not an allocated device tensor";
    }
    // buffer() throws when no device of the mesh is local; native's launch skips such a mesh instead.
    if (input.device()->is_remote_only()) {
        return "the mesh has no local devices";
    }
    // Every program moves bytes, so any 2 or 4 B type the kernels copy in halfwords or words works alike.
    switch (input.dtype()) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::UINT16: break;
        default: return "dtype is not bfloat16, float32, int32, uint32 or uint16";
    }
    if (!is_plain_interleaved(input.memory_config()) || !is_plain_interleaved(output_mem_config)) {
        return "input or output is not interleaved in DRAM or L1";
    }
    const auto& in_shape = input.logical_shape();
    if (in_shape == logical_output_shape) {
        return "output shape equals the input shape";
    }
    if (in_shape.volume() == 0 || in_shape.volume() != logical_output_shape.volume()) {
        return "input is empty or the volume changes";
    }
    if (input.layout() == Layout::TILE) {
        if (!standard_tile_with_tight_padding(input)) {
            return "input is not TILE with 32x32 tiles padded only in the last two dims";
        }
    } else if (input.layout() != Layout::ROW_MAJOR || input.padded_shape() != in_shape) {
        // Row-major programs address input sticks by the logical shape.
        return "input is neither TILE nor unpadded ROW_MAJOR";
    }
    if (!supported_execution_controls(input, sub_core_grids)) {
        return "sub_core_grids is not a non-empty subset of the compute grid";
    }
    return nullptr;
}

// Native reshapes one output tile per core per round and reads every source tile it touches; tile rows reads each
// source once per batch. Tile rows wins on any of three clauses: native reads at least 8x what the batches read,
// the batches save at least 16 reads per output tile, or every output tile is whole face halves of its sources.
// Both thresholds are judged on the full plan. Fitted on Blackhole; Wormhole cross-arch CI shows no routed loss at
// this cut.
constexpr uint64_t kTileRowsMinReadCut = 8;
// Source reads tile rows saves per output tile that pay for copying its rows one run at a time, whatever the ratio.
// Width changes below the read cut save at most 6 and lose; a merge of a few heads gathers each output tile from 32
// padded source tiles and saves 24. Fitted on Blackhole, holds on Wormhole.
constexpr uint64_t kTileRowsMinReadsSavedPerTile = 16;
// Two runs per output tile means every output tile is whole face halves of its sources.
constexpr uint64_t kWholeHalfRunsPerTile = 2;

bool native_faster(const Tensor& input, const TileRowsChoice& choice, uint32_t num_cores) {
    if (input.dtype() == DataType::UINT16) {
        return false;
    }
    // A width change copies one row per run and loses below the read cut unless the reads it saves outweigh the runs;
    // a refold of whole face halves wins once native needs more than one round.
    const uint64_t out_pages = choice.layout.num_output_pages;
    const bool read_cut = choice.native_reads >= kTileRowsMinReadCut * choice.batch_reads;
    const bool reads_saved = choice.native_reads >= choice.batch_reads + kTileRowsMinReadsSavedPerTile * out_pages;
    const bool whole_halves = choice.runs <= kWholeHalfRunsPerTile * out_pages && out_pages > num_cores;
    return !read_cut && !reads_saved && !whole_halves;
}

constexpr const char* kTileCapsViolation = "no TILE program takes these shapes within its size caps";
constexpr const char* kTileStaticFitViolation = "no TILE program fits the static L1 budget even at its smallest plan";

// The TILE program a call takes, chosen against the static L1 window so the choice does not move with live
// occupancy. `beats_native` is whether that program measures faster than native's tiled reshape.
struct TileRouteChoice {
    std::optional<LaunchPlan> plan;
    bool beats_native = false;
    const char* violation = nullptr;
    std::optional<tt::tt_metal::TensorSpec> output_spec;
};

// Tries, in order: tile rows where it beats native, then tile rows and the segment map where they do not; the first
// whose smallest plan fits the static window is taken. The segment map beats native only on widths that are not
// multiples of 32, where tile rows does not apply. Tile rows falls back to the segment map when live L1 holds none of
// its plans, and its smallest plan needs no more L1 than native's tiled reshape, so tile rows beats native only while
// the segment map is within its size caps. Native rejects uint16, so every program beats it there. With
// `winners_only`, returns no plan before building any output spec when no program can beat native.
TileRouteChoice choose_tile_route(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t num_cores,
    bool winners_only) {
    const auto& in_shape = input.logical_shape();
    const bool native_rejects = input.dtype() == DataType::UINT16;
    const bool row_permutation = is_tile_row_permutation(in_shape, logical_output_shape);
    auto rows = plan_tile_rows(in_shape, logical_output_shape);
    const bool rows_win = rows.has_value() && !native_faster(input, *rows, num_cores);
    if (winners_only && !native_rejects && !rows_win && row_permutation) {
        return {};
    }

    TileRouteChoice route{.output_spec = output_spec(input, logical_output_shape, output_mem_config)};
    const uint64_t budget = static_l1_budget(input, *route.output_spec);
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const auto rows_fit_budget = [&](const TileRowsChoice& c) { return tile_rows_l1_bytes(c, tile_bytes) <= budget; };
    const bool rows_fit = rows.has_value() && (rows_fit_budget(*rows) ||
                                               plan_tile_rows_fitting(in_shape, logical_output_shape, rows_fit_budget));
    const bool mapped_within_caps =
        fit_tile_mapped(input, logical_output_shape, num_cores, std::numeric_limits<uint64_t>::max()).has_value();
    const bool live_fallback = native_rejects || mapped_within_caps;

    const auto take = [&](TileRoute tile_route, bool beats_native) {
        route.plan = LaunchPlan{.tile_route = tile_route};
        if (tile_route == TileRoute::TileRows) {
            route.plan->tile_rows = std::move(*rows);
        }
        route.beats_native = beats_native;
        return route;
    };
    if (rows_fit && rows_win) {
        return take(TileRoute::TileRows, live_fallback);
    }
    if (rows_fit) {
        return take(TileRoute::TileRows, false);
    }
    if (fit_tile_mapped(input, logical_output_shape, num_cores, budget).has_value()) {
        return take(TileRoute::TileMapped, native_rejects || !row_permutation);
    }
    const bool within_caps = rows.has_value() || mapped_within_caps;
    route.violation = within_caps ? kTileStaticFitViolation : kTileCapsViolation;
    return route;
}

// Native's row-major factory treats a stick as NOC-aligned at this multiple.
constexpr uint32_t kNativeNocAlignment = 16;

// Native's row-major kernel runs as two instances per core, each on its own RISC and NOC, unless its output is DRAM
// with sticks that are not a 16 B multiple; then a single instance moves everything.
bool native_runs_two_kernels(uint32_t new_stick, const tt::tt_metal::MemoryConfig& output_mem_config) {
    return output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 || new_stick % kNativeNocAlignment == 0;
}

// Merges issue every read from one RISC on one NOC while native spreads them over two. From L1 a read of
// kMaxReadSize is alone between barriers, and a 32 KB new stick is one push the writer waits on in full.
constexpr uint32_t kSinglePushNewStick = 16 * kMaxReadSize;

bool coalesced_l1_merge_native_faster(uint32_t piece, uint32_t new_stick) {
    return piece >= kMaxReadSize || (2 * piece >= kMaxReadSize && new_stick >= kSinglePushNewStick);
}

// Unaligned sticks are repacked by a RISC at a cost per byte; native pays a round trip per piece, so it wins from a
// piece size set by that round trip and its kernel count. Merges pay a read round trip per piece, splits only a
// staged write. Fitted on Blackhole.
constexpr uint32_t kMinNativeMergePieceTwoKernels = 40;
constexpr uint32_t kMinNativeSplitPieceTwoKernels = 20;
constexpr uint32_t kMinNativeMergePieceOneKernel = 128;
constexpr uint32_t kMinNativeSplitPieceOneKernel = 40;

uint32_t min_native_piece(bool two_kernels, bool merge) {
    if (two_kernels) {
        return merge ? kMinNativeMergePieceTwoKernels : kMinNativeSplitPieceTwoKernels;
    }
    return merge ? kMinNativeMergePieceOneKernel : kMinNativeSplitPieceOneKernel;
}

// Native hands each of its two kernels half of a core's source sticks; at two sticks per core each kernel moves one,
// while the port's reader and writer pass both through one pipeline. Up to this many pieces per stick, the second
// stick costs the port more than its repack saves.
constexpr uint32_t kMaxPairedSplitPieces = 24;

// Whether an L1 split puts exactly two source sticks on the port's busiest core. In DRAM the port still wins there.
bool paired_l1_split_native_faster(
    uint32_t ratio, uint64_t num_old_sticks, uint32_t num_cores, const tt::tt_metal::MemoryConfig& output_mem_config) {
    return output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 && ratio <= kMaxPairedSplitPieces &&
           num_old_sticks > num_cores && num_old_sticks <= 2 * static_cast<uint64_t>(num_cores);
}

bool staged_native_faster(
    uint32_t piece,
    uint32_t new_stick,
    uint32_t ratio,
    uint64_t num_old_sticks,
    uint32_t num_cores,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool merge) {
    const bool two_kernels = native_runs_two_kernels(new_stick, output_mem_config);
    // Pieces that are not a multiple of 4 B make native copy byte by byte; one kernel doing that never catches up.
    if (!two_kernels && piece % 4 != 0) {
        return false;
    }
    return piece >= min_native_piece(two_kernels, merge) ||
           (!merge && paired_l1_split_native_faster(ratio, num_old_sticks, num_cores, output_mem_config));
}

// Slack native's row-major factory adds to each source read and destination slot, and the cap on its destination ring.
constexpr uint64_t kNativeSourceReadSlack = 128;
constexpr uint64_t kNativeDestSlotSlack = 80;
constexpr uint64_t kNativeMaxDestSlots = 8;

// Native's row-major kernel fits the scratch and destination ring it selects in `budget`. Native picks its ring from
// unaligned sizes but places every scratchpad DRAM-aligned, so the ring it picks can overrun `budget`; native then
// fails where a row-major program still fits.
bool native_rm_fits_l1(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t budget) {
    const uint32_t elem = input.element_size();
    const uint64_t old_stick = static_cast<uint64_t>(input.logical_shape()[-1]) * elem;
    const uint64_t new_stick = static_cast<uint64_t>(logical_output_shape[-1]) * elem;
    // Per instance: a scratchpad of two source pages padded to 64 B plus 128 B, and one of `slots` destination slots
    // each padded to 64 B plus 80 B. Every scratchpad starts DRAM-aligned, so each takes its size rounded up to that
    // alignment.
    const uint64_t alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint64_t source_scratch = 2 * (((old_stick - 1) & ~uint64_t{63}) + kNativeSourceReadSlack);
    const uint64_t dest_slot = ((new_stick - 1) & ~uint64_t{63}) + kNativeDestSlotSlack;
    const bool divisible = old_stick % new_stick == 0 || new_stick % old_stick == 0;
    const uint64_t kernels =
        divisible && native_runs_two_kernels(static_cast<uint32_t>(new_stick), output_mem_config) ? 2 : 1;
    const auto footprint = [&](uint64_t slots) {
        return kernels * (tt::round_up(source_scratch, alignment) + tt::round_up(dest_slot * slots, alignment));
    };
    if (footprint(1) > budget) {
        return false;
    }
    const bool pages_noc_aligned = old_stick % kNativeNocAlignment == 0 && new_stick % kNativeNocAlignment == 0;
    const uint64_t slots =
        pages_noc_aligned
            ? 1
            : std::clamp((budget - kernels * source_scratch) / (kernels * dest_slot), uint64_t{1}, kNativeMaxDestSlots);
    return footprint(slots) <= budget;
}

// Wormhole thresholds, fitted from the same forced sweeps run on Wormhole. Its DRAM reads stay transaction-bound up to
// 2 KB pieces, its L1 merges lose to native from 1 KB, and a single-kernel split with at most one source stick per
// core leaves the port's pipeline unfilled, so native wins from 20 B.
constexpr uint32_t kWhMinDramMergePiece = 2048;
constexpr uint32_t kWhMinL1MergePiece = 1024;
constexpr uint32_t kWhMinUnderfilledSplitPiece = 20;
// Blackhole's counterpart of kWhMinDramMergePiece; the reason is at its use in rm_native_faster.
constexpr uint32_t kBhMinDramMergePiece = 256;

// Whether an rm_native_faster verdict may be taken once the static fit is known. Native's scratch and the stick program
// rm_native_faster was measured against must both fit the static L1 window (see native_rm_fits_l1). When only the slab
// program fits, it reads a batch of windows behind one barrier, as for widths that do not divide.
bool demotion_applies(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t static_budget,
    bool stick_program_fits) {
    return stick_program_fits && native_rm_fits_l1(input, logical_output_shape, output_mem_config, static_budget);
}

// Whether native's kernel measures faster than the ROW_MAJOR program the gate admits. Thresholds fitted on Blackhole
// from forced port-vs-native sweeps of same-placement copies, with the Wormhole thresholds above. Widths that do not
// divide are never demoted: native waits on a read barrier per input page, the slab program reads every window of a
// batch of slabs behind one barrier. Mixed DRAM/L1 placement is not demoted either: the output's alignment is not an
// attribute of the input, and the rule was fitted on same-placement copies only.
bool rm_native_faster(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t num_cores) {
    if (input.layout() != Layout::ROW_MAJOR || output_mem_config.buffer_type() != input.memory_config().buffer_type()) {
        return false;
    }
    const uint32_t elem = input.element_size();
    const uint32_t old_stick = input.logical_shape()[-1] * elem;
    const uint32_t new_stick = logical_output_shape[-1] * elem;
    // The shorter stick is the piece both programs move one at a time.
    const uint32_t piece = std::min(old_stick, new_stick);
    if (old_stick == new_stick || std::max(old_stick, new_stick) % piece != 0) {
        return false;
    }
    const uint32_t alignment = input.buffer()->alignment();
    const bool merge = new_stick > old_stick;
    const bool wormhole = input.device()->arch() == tt::ARCH::WORMHOLE_B0;
    if (old_stick % alignment == 0 && new_stick % alignment == 0) {
        // Splits keep every short write in flight; native waits on each one. From 256 B a DRAM read is no longer
        // transaction-bound and the port's single read NOC caps it.
        if (input.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM) {
            return merge && piece >= (wormhole ? kWhMinDramMergePiece : kBhMinDramMergePiece);
        }
        if (wormhole) {
            return merge && piece >= kWhMinL1MergePiece;
        }
        return merge && coalesced_l1_merge_native_faster(piece, new_stick);
    }
    const uint64_t num_old_sticks = input.logical_shape().volume() / input.logical_shape()[-1];
    if (wormhole && !merge && !native_runs_two_kernels(new_stick, output_mem_config) && piece % 4 == 0 &&
        piece >= kWhMinUnderfilledSplitPiece && num_old_sticks <= num_cores) {
        return true;
    }
    return staged_native_faster(
        piece, new_stick, std::max(old_stick, new_stick) / piece, num_old_sticks, num_cores, output_mem_config, merge);
}

// The budget-independent half of the row-major programs, fitted to the static budget for the gate. The launch
// carries it unfitted for live_rm_choice. The slab geometry is only built when the stick program does not fit.
class RmGeometry {
public:
    RmGeometry(
        const Tensor& input,
        const ttnn::Shape& logical_output_shape,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        uint32_t num_cores) :
        input_(input),
        logical_output_shape_(logical_output_shape),
        output_mem_config_(output_mem_config),
        num_cores_(num_cores),
        output_spec_(reshape_codegen::output_spec(input, logical_output_shape, output_mem_config)),
        out_aligned_page_(output_aligned_page(input, output_spec_)),
        stick_(rm_geometry(input, logical_output_shape, out_aligned_page_, num_cores_)),
        static_budget_(static_l1_budget(input, output_spec_)) {}

    uint64_t static_budget() const { return static_budget_; }
    const tt::tt_metal::TensorSpec& output_spec() const { return output_spec_; }

    std::optional<RmPlan> fit_stick(uint64_t l1_budget) const {
        return stick_.has_value() ? fit_rm(*stick_, l1_budget) : std::nullopt;
    }

    std::optional<RmArbitraryPlan> fit_slab(uint64_t l1_budget) {
        if (!slab_built_) {
            slab_ = rm_arbitrary_geometry(
                input_, logical_output_shape_, out_aligned_page_, output_mem_config_.buffer_type(), num_cores_);
            slab_built_ = true;
        }
        return slab_.has_value() ? fit_rm_arbitrary(*slab_, l1_budget) : std::nullopt;
    }

    // Whether the smallest stick or slab plan fits the static window.
    bool fits_static() { return fit_stick(static_budget_).has_value() || fit_slab(static_budget_).has_value(); }

    LaunchPlan launch_plan() const {
        return LaunchPlan{.out_aligned_page = out_aligned_page_, .rm = stick_, .rm_arbitrary = slab_};
    }

private:
    const Tensor& input_;
    const ttnn::Shape& logical_output_shape_;
    const tt::tt_metal::MemoryConfig& output_mem_config_;
    uint32_t num_cores_;
    tt::tt_metal::TensorSpec output_spec_;
    uint32_t out_aligned_page_;
    std::optional<RmPlan> stick_;
    uint64_t static_budget_;
    bool slab_built_ = false;
    std::optional<RmArbitraryPlan> slab_;
};

uint32_t rm_grid_cores(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return static_cast<uint32_t>(reshape_core_grid(input, sub_core_grids).num_cores());
}

constexpr const char* kRmStaticFitViolation = "no row-major program fits the static L1 budget";

}  // namespace

const char* call_contract_violation(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<ttnn::PadValue>& pad_value) {
    if (padded_shape != logical_shape || logical_shape.volume() != input.logical_volume()) {
        return "explicit padded shape or volume change";
    }
    // A rank-one output is a single row, so its tile pads the height from 1 to 32.
    const bool padded_tile_output = logical_shape.rank() < 2 || logical_shape[-1] % tt::constants::TILE_WIDTH != 0 ||
                                    logical_shape[-2] % tt::constants::TILE_HEIGHT != 0;
    if (input.layout() == Layout::TILE && pad_value.has_value() && padded_tile_output) {
        return "explicit pad_value on a padded tile output";
    }
    return nullptr;
}

bool supported_execution_controls(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return !sub_core_grids.has_value() ||
           (sub_core_grids->num_cores() > 0 && reshape_core_grid(input).contains(*sub_core_grids));
}

const char* unsupported_reason(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids)) {
        return violation;
    }
    if (input.layout() == Layout::ROW_MAJOR) {
        RmGeometry geometry(input, logical_output_shape, output_mem_config, rm_grid_cores(input, sub_core_grids));
        return geometry.fits_static() ? nullptr : kRmStaticFitViolation;
    }
    return choose_tile_route(
               input,
               logical_output_shape,
               output_mem_config,
               reshape_core_grid(input, sub_core_grids).num_cores(),
               /*winners_only=*/false)
        .violation;
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return unsupported_reason(input, logical_output_shape, output_mem_config, sub_core_grids) == nullptr;
}

bool supported_by_codegen(const Tensor& input, const ttnn::Shape& logical_output_shape) {
    return supported_by_codegen(input, logical_output_shape, input.memory_config());
}

bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.layout() == Layout::ROW_MAJOR) {
        const uint32_t num_cores = rm_grid_cores(input, sub_core_grids);
        // The stick-size rule runs before any output spec is built; native's device operation rejects uint16.
        if (input.dtype() == DataType::UINT16 ||
            !rm_native_faster(input, logical_output_shape, output_mem_config, num_cores)) {
            return false;
        }
        RmGeometry geometry(input, logical_output_shape, output_mem_config, num_cores);
        const uint64_t static_budget = geometry.static_budget();
        return demotion_applies(
            input,
            logical_output_shape,
            output_mem_config,
            static_budget,
            geometry.fit_stick(static_budget).has_value());
    }
    // Built against the static window only, so the verdict does not move with live occupancy.
    return !choose_tile_route(
                input,
                logical_output_shape,
                output_mem_config,
                reshape_core_grid(input, sub_core_grids).num_cores(),
                /*winners_only=*/true)
                .beats_native;
}

LaunchPlan launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids);
    TT_FATAL(violation == nullptr, "reshape_codegen: outside the codegen scope: {}", violation);
    if (input.layout() == Layout::ROW_MAJOR) {
        RmGeometry geometry(input, logical_output_shape, output_mem_config, rm_grid_cores(input, sub_core_grids));
        TT_FATAL(geometry.fits_static(), "reshape_codegen: outside the codegen scope: {}", kRmStaticFitViolation);
        return geometry.launch_plan();
    }
    auto route = choose_tile_route(
        input,
        logical_output_shape,
        output_mem_config,
        reshape_core_grid(input, sub_core_grids).num_cores(),
        /*winners_only=*/false);
    TT_FATAL(route.plan.has_value(), "reshape_codegen: outside the codegen scope: {}", route.violation);
    return std::move(*route.plan);
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
