// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"

#include <algorithm>

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
    // The writers walk 16x16 faces; a 32x32 tile built with another face shape lays its datums out differently.
    const auto face = tile.get_face_shape();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        face[0] != tt::constants::FACE_HEIGHT || face[1] != tt::constants::FACE_WIDTH ||
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

// Why the codegen program cannot run this call before any plan is built (placement, layout, dtype, shapes, grid), or
// nullptr. Cheapest first.
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
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::INT32:
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::UINT16: break;
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
    if (input.layout() == tt::tt_metal::Layout::TILE) {
        if (!standard_tile_with_tight_padding(input)) {
            return "input is not TILE with 32x32 tiles padded only in the last two dims";
        }
    } else if (input.layout() != tt::tt_metal::Layout::ROW_MAJOR || input.padded_shape() != in_shape) {
        // Row-major programs address input sticks by the logical shape.
        return "input is neither TILE nor unpadded ROW_MAJOR";
    }
    if (!supported_execution_controls(input, sub_core_grids)) {
        return "sub_core_grids is not a non-empty subset of the compute grid";
    }
    return nullptr;
}

// Native reads every source tile each output tile touches; tile rows reads each source once per batch. Thresholds
// are judged on the full plan, fitted on Blackhole and checked on Wormhole.
constexpr uint64_t kTileRowsMinReadCut = 8;
// Source reads saved per output tile that pay for copying its rows one run at a time, whatever the ratio. Width
// changes below the read cut save at most 6 and lose; a merge of a few heads saves 24.
constexpr uint64_t kTileRowsMinReadsSavedPerTile = 16;
// Two runs per output tile means every output tile is whole face halves of its sources.
constexpr uint64_t kWholeHalfRunsPerTile = 2;

bool native_faster(const TileRowsChoice& choice, uint32_t num_cores) {
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
// while the codegen reader and writer pass both through one pipeline. Up to this many pieces per stick, the second
// stick costs codegen more than its repack saves.
constexpr uint32_t kMaxPairedSplitPieces = 24;

// Whether an L1 split puts exactly two source sticks on codegen's busiest core. In DRAM codegen still wins there.
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

// Scratchpads native's row-major kernel takes per core. Per instance: one of two source pages padded to 64 B plus
// 128 B, and one of `slots` destination slots each padded to 64 B plus 80 B. Every scratchpad starts DRAM-aligned, so
// each takes its size rounded up to that alignment.
struct NativeRmScratch {
    uint64_t source_scratch = 0;
    uint64_t dest_slot = 0;
    uint64_t kernels = 1;
    bool pages_noc_aligned = false;

    uint64_t footprint(uint64_t slots) const {
        const uint64_t alignment = tt::tt_metal::hal::get_dram_alignment();
        return kernels * (tt::round_up(source_scratch, alignment) + tt::round_up(dest_slot * slots, alignment));
    }
};

NativeRmScratch native_rm_scratch(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config) {
    const uint32_t elem = input.element_size();
    const uint64_t old_stick = static_cast<uint64_t>(input.logical_shape()[-1]) * elem;
    const uint64_t new_stick = static_cast<uint64_t>(logical_output_shape[-1]) * elem;
    const bool divisible = old_stick % new_stick == 0 || new_stick % old_stick == 0;
    const bool two_kernels = divisible && native_runs_two_kernels(static_cast<uint32_t>(new_stick), output_mem_config);
    return NativeRmScratch{
        .source_scratch = 2 * (((old_stick - 1) & ~uint64_t{63}) + kNativeSourceReadSlack),
        .dest_slot = ((new_stick - 1) & ~uint64_t{63}) + kNativeDestSlotSlack,
        .kernels = two_kernels ? uint64_t{2} : uint64_t{1},
        .pages_noc_aligned = old_stick % kNativeNocAlignment == 0 && new_stick % kNativeNocAlignment == 0};
}

// Native's row-major kernel fits the scratch and destination ring it selects in `budget`. Native picks its ring from
// unaligned sizes but places every scratchpad DRAM-aligned, so the ring it picks can overrun `budget`; native then
// fails where a row-major program still fits.
bool native_rm_fits_l1(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t budget) {
    const NativeRmScratch native = native_rm_scratch(input, logical_output_shape, output_mem_config);
    if (native.footprint(1) > budget) {
        return false;
    }
    uint64_t slots = 1;
    if (!native.pages_noc_aligned) {
        slots = std::clamp(
            (budget - native.kernels * native.source_scratch) / (native.kernels * native.dest_slot),
            uint64_t{1},
            kNativeMaxDestSlots);
    }
    return native.footprint(slots) <= budget;
}

// Wormhole thresholds, fitted from the same forced sweeps run on Wormhole. Its DRAM reads stay transaction-bound up to
// 2 KB pieces, its L1 merges lose to native from 1 KB, and a single-kernel split with at most one source stick per
// core leaves the codegen pipeline unfilled, so native wins from 20 B.
constexpr uint32_t kWhMinDramMergePiece = 2048;
constexpr uint32_t kWhMinL1MergePiece = 1024;
constexpr uint32_t kWhMinUnderfilledSplitPiece = 20;
// Blackhole's counterpart of kWhMinDramMergePiece; the reason is at its use in rm_native_faster.
constexpr uint32_t kBhMinDramMergePiece = 256;

// Whether native's kernel measures faster than the ROW_MAJOR program the gate admits. Thresholds fitted on Blackhole
// from forced codegen-vs-native sweeps of same-placement copies, with the Wormhole thresholds above. Widths that do not
// divide are never demoted: native waits on a read barrier per input page, the slab program reads every window of a
// batch of slabs behind one barrier. Mixed DRAM/L1 placement is not demoted either: the output's alignment is not an
// attribute of the input, and the rule was fitted on same-placement copies only.
bool rm_native_faster(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t num_cores) {
    if (input.layout() != tt::tt_metal::Layout::ROW_MAJOR ||
        output_mem_config.buffer_type() != input.memory_config().buffer_type()) {
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
        // transaction-bound and codegen's single read NOC caps it.
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

    // The geometries `plan` carries, which the constructor above built for the same call. A slab geometry the plan
    // lacks is built from the shapes, as live_rm_choice builds it.
    RmGeometry(
        const Tensor& input,
        const ttnn::Shape& logical_output_shape,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        uint32_t num_cores,
        const LaunchPlan& plan) :
        input_(input),
        logical_output_shape_(logical_output_shape),
        output_mem_config_(output_mem_config),
        num_cores_(num_cores),
        output_spec_(reshape_codegen::output_spec(input, logical_output_shape, output_mem_config)),
        out_aligned_page_(plan.out_aligned_page),
        stick_(plan.rm),
        static_budget_(static_l1_budget(input, output_spec_)),
        slab_built_(plan.rm_arbitrary.has_value()),
        slab_(plan.rm_arbitrary) {}

    uint64_t static_budget() const { return static_budget_; }
    const tt::tt_metal::TensorSpec& output_spec() const { return output_spec_; }

    std::optional<RmPlan> fit_stick(uint64_t l1_budget) const {
        return stick_.has_value() ? fit_rm(*stick_, l1_budget) : std::nullopt;
    }

    std::optional<RmArbitraryPlan> fit_slab(uint64_t l1_budget) {
        const auto& slab_geometry = slab();
        return slab_geometry.has_value() ? fit_rm_arbitrary(*slab_geometry, l1_budget) : std::nullopt;
    }

    // Whether the smallest stick or slab plan fits `l1_budget`, as live_rm_choice tries them.
    bool fits(uint64_t l1_budget) { return fit_stick(l1_budget).has_value() || fit_slab(l1_budget).has_value(); }

    bool fits_static() { return fits(static_budget_); }

    // Carries the slab geometry only when the stick program misses the static window.
    LaunchPlan launch_plan() {
        if (!fit_stick(static_budget_).has_value()) {
            slab();
        }
        return LaunchPlan{.out_aligned_page = out_aligned_page_, .rm = stick_, .rm_arbitrary = slab_};
    }

private:
    const std::optional<RmArbitraryPlan>& slab() {
        if (!slab_built_) {
            slab_ = rm_arbitrary_geometry(
                input_, logical_output_shape_, out_aligned_page_, output_mem_config_.buffer_type(), num_cores_);
            slab_built_ = true;
        }
        return slab_;
    }

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
    if (input.layout() == tt::tt_metal::Layout::TILE && pad_value.has_value() && padded_tile_output) {
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
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan* plan) {
    if (const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids)) {
        return violation;
    }
    const uint32_t num_cores = reshape_core_grid(input, sub_core_grids).num_cores();
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        if (plan == nullptr) {
            return kRmStaticFitViolation;
        }
        RmGeometry geometry(input, logical_output_shape, output_mem_config, num_cores, *plan);
        return geometry.fits_static() ? nullptr : kRmStaticFitViolation;
    }
    const auto& in_shape = input.logical_shape();
    if (plan == nullptr) {
        const bool within_caps = plan_tile_rows(in_shape, logical_output_shape).has_value() ||
                                 segment_map_within_caps(input, logical_output_shape, num_cores);
        return within_caps ? kTileStaticFitViolation : kTileCapsViolation;
    }
    const uint64_t budget = static_l1_budget(input, output_spec(input, logical_output_shape, output_mem_config));
    if (plan->tile_route == TileRoute::TileRows) {
        const uint32_t tile_bytes =
            input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
        const auto fits = [&](const TileRowsChoice& c) { return tile_rows_l1_bytes(c, tile_bytes) <= budget; };
        if (fits(plan->tile_rows) || plan_tile_rows_fitting(in_shape, logical_output_shape, fits).has_value()) {
            return nullptr;
        }
    } else if (fit_tile_mapped(input, logical_output_shape, num_cores, budget).has_value()) {
        return nullptr;
    }
    return kTileStaticFitViolation;
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan& plan) {
    return unsupported_reason(input, logical_output_shape, output_mem_config, sub_core_grids, &plan) == nullptr;
}

std::optional<LaunchPlan> plan_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids) != nullptr) {
        return std::nullopt;
    }
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        RmGeometry geometry(
            input, logical_output_shape, output_mem_config, reshape_core_grid(input, sub_core_grids).num_cores());
        LaunchPlan plan = geometry.launch_plan();
        if (supported_by_codegen(input, logical_output_shape, output_mem_config, sub_core_grids, plan)) {
            return plan;
        }
        return std::nullopt;
    }
    if (auto rows = plan_tile_rows(input.logical_shape(), logical_output_shape)) {
        LaunchPlan plan{.tile_route = TileRoute::TileRows, .tile_rows = std::move(*rows)};
        if (supported_by_codegen(input, logical_output_shape, output_mem_config, sub_core_grids, plan)) {
            return plan;
        }
    }
    LaunchPlan mapped{.tile_route = TileRoute::TileMapped};
    if (supported_by_codegen(input, logical_output_shape, output_mem_config, sub_core_grids, mapped)) {
        return mapped;
    }
    return std::nullopt;
}

bool segment_map_within_caps(const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores) {
    return tile_mapped_within_caps(
        input.logical_shape(),
        logical_output_shape,
        static_cast<uint32_t>(input.element_size()),
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype())),
        num_cores);
}

bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t num_cores,
    const LaunchPlan& plan) {
    // Native rejects uint16, so demoting would fail the call.
    if (input.dtype() == tt::tt_metal::DataType::UINT16) {
        return false;
    }
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        RmGeometry geometry(input, logical_output_shape, output_mem_config, num_cores, plan);
        // The program factory raises when live L1 holds no row-major program, so the call runs native unless every L1
        // window native runs in also holds the smallest one.
        if (!geometry.fits(native_rm_scratch(input, logical_output_shape, output_mem_config).footprint(1))) {
            return true;
        }
        if (!rm_native_faster(input, logical_output_shape, output_mem_config, num_cores)) {
            return false;
        }
        // The verdict holds only where native's scratch and the stick program it was measured against both fit the
        // static window (see native_rm_fits_l1). When only the slab program fits, it reads a batch of windows behind
        // one barrier, as for widths that do not divide.
        const uint64_t static_budget = geometry.static_budget();
        return geometry.fit_stick(static_budget).has_value() &&
               native_rm_fits_l1(input, logical_output_shape, output_mem_config, static_budget);
    }
    if (plan.tile_route == TileRoute::TileRows) {
        // Tile rows falls back to the segment map under live L1 pressure, so it beats native only while the segment
        // map is within its size caps.
        return native_faster(plan.tile_rows, num_cores) ||
               !segment_map_within_caps(input, logical_output_shape, num_cores);
    }
    // The segment map loses to native only on a tile-row permutation: both widths whole tiles.
    return input.logical_shape()[-1] % kTileDim == 0 && logical_output_shape[-1] % kTileDim == 0;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
