// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

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
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return "input is not an allocated device tensor";
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
        if (!is_tile_row_permutation(in_shape, logical_output_shape)) {
            return "input or output last dim is not a multiple of 32";
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

// Gates on the smallest plan, not the full one: the program factory shrinks the plan to what live L1 holds, so the
// call is in scope whenever any plan fits the static window.
const char* static_fit_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const TileRowsChoice& choice) {
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const uint64_t budget = static_l1_budget(input, output_spec(input, logical_output_shape, output_mem_config));
    const auto fits = [&](const TileRowsChoice& c) { return tile_rows_l1_bytes(c, tile_bytes) <= budget; };
    if (fits(choice) || plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits)) {
        return nullptr;
    }
    return "tile-row buffers do not fit the static L1 budget even at the smallest plan";
}

constexpr const char* kPlanCapsViolation = "tile-row plan exceeds its size caps";

// Native reshapes one output tile per core per round and reads every source tile it touches; tile rows reads each
// source once per batch. Tile rows wins on any of three clauses: native reads at least 8x what the batches read,
// the batches save at least 16 reads per output tile, or every output tile is whole face halves of its sources.
// Both thresholds are judged on the full plan. Fitted on Blackhole.
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

// Native's row-major factory treats a stick as NOC-aligned at this multiple.
constexpr uint32_t kNativeNocAlignment = 16;

// Native's row-major kernel runs as two instances per core, each on its own RISC and NOC, unless its output is DRAM
// with sticks that are not a 16 B multiple; then a single instance moves everything.
bool native_runs_two_kernels(uint32_t new_stick, const tt::tt_metal::MemoryConfig& output_mem_config) {
    return output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 || new_stick % kNativeNocAlignment == 0;
}

// Merges issue every read from one RISC on one NOC while native spreads them over two. From L1 a read of
// kMaxReadSize is alone between barriers, and a 32 KB new stick is one push the writer waits on in full.
bool coalesced_l1_merge_native_faster(uint32_t piece, uint32_t new_stick) {
    return piece >= kMaxReadSize || (2 * piece >= kMaxReadSize && new_stick >= 16 * kMaxReadSize);
}

// Unaligned sticks are repacked by a RISC at a cost per byte; native pays a round trip per piece, so it wins from a
// piece size set by that round trip and its kernel count. Merges pay a read round trip per piece, splits only a
// staged write.
uint32_t min_native_piece(bool two_kernels, bool merge) {
    if (two_kernels) {
        return merge ? 40 : 20;
    }
    return merge ? 128 : 40;
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

// Native's row-major kernel fits its minimum scratch in the static L1 window. Native sizes that scratch from live L1
// and fails when it does not fit.
bool native_rm_fits_l1(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t static_budget) {
    const uint32_t elem = input.element_size();
    const uint64_t old_stick = static_cast<uint64_t>(input.logical_shape()[-1]) * elem;
    const uint64_t new_stick = static_cast<uint64_t>(logical_output_shape[-1]) * elem;
    // Per instance: two source pages padded to 64 B plus 128 B, one destination slot padded to 64 B plus 80 B.
    const uint64_t per_kernel =
        (2 * (((old_stick - 1) & ~uint64_t{63}) + 128)) + ((new_stick - 1) & ~uint64_t{63}) + 80;
    const bool divisible = old_stick % new_stick == 0 || new_stick % old_stick == 0;
    const uint64_t kernels =
        divisible && native_runs_two_kernels(static_cast<uint32_t>(new_stick), output_mem_config) ? 2 : 1;
    return kernels * per_kernel <= static_budget;
}

// Whether an rm_native_faster verdict may be taken once the static fit is known. Native's scratch must fit, and the
// stick program rm_native_faster was measured against must fit the static L1 window. When only the slab program
// fits, it reads a batch of windows behind one barrier, as for widths that do not divide.
bool demotion_applies(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t static_budget,
    bool stick_program_fits) {
    return stick_program_fits && native_rm_fits_l1(input, logical_output_shape, output_mem_config, static_budget);
}

// Whether native's kernel measures faster than the ROW_MAJOR program the gate admits. Thresholds fitted on Blackhole
// from forced port-vs-native sweeps of same-placement copies. Widths that do not divide are never demoted: native
// waits on a read barrier per input page, the slab program reads every window of a batch of slabs behind one barrier.
// Mixed DRAM/L1 placement is not demoted either: the output's alignment is not an attribute of the input, and the
// rule was fitted on same-placement copies only.
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
    if (old_stick % alignment == 0 && new_stick % alignment == 0) {
        // Splits keep every short write in flight; native waits on each one. From 256 B a DRAM read is no longer
        // transaction-bound and the port's single read NOC caps it.
        if (input.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM) {
            return merge && piece >= 256;
        }
        return merge && coalesced_l1_merge_native_faster(piece, new_stick);
    }
    const uint64_t num_old_sticks = input.logical_shape().volume() / input.logical_shape()[-1];
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
    const bool padded_tile_output = logical_shape.rank() >= 2 && (logical_shape[-1] % tt::constants::TILE_WIDTH != 0 ||
                                                                  logical_shape[-2] % tt::constants::TILE_HEIGHT != 0);
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
    const auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    if (!choice) {
        return kPlanCapsViolation;
    }
    return static_fit_violation(input, logical_output_shape, output_mem_config, *choice);
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
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.layout() != Layout::TILE) {
        return false;
    }
    const auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    return choice.has_value() && native_faster(input, *choice, reshape_core_grid(input, sub_core_grids).num_cores());
}

std::optional<LaunchPlan> routed_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids) != nullptr) {
        return std::nullopt;
    }
    if (input.layout() == Layout::ROW_MAJOR) {
        const uint32_t num_cores = rm_grid_cores(input, sub_core_grids);
        // The shape-only half of demotion runs before any output spec is built; native's device operation rejects
        // uint16.
        const bool native_may_be_faster = input.dtype() != DataType::UINT16 &&
                                          rm_native_faster(input, logical_output_shape, output_mem_config, num_cores);
        RmGeometry geometry(input, logical_output_shape, output_mem_config, num_cores);
        const uint64_t static_budget = geometry.static_budget();
        const bool stick_fits = geometry.fit_stick(static_budget).has_value();
        if (!stick_fits && !geometry.fit_slab(static_budget).has_value()) {
            return std::nullopt;
        }
        if (native_may_be_faster &&
            demotion_applies(input, logical_output_shape, output_mem_config, static_budget, stick_fits)) {
            return std::nullopt;
        }
        // Native sizes its scratch to live L1, so a call no row-major program fits once the output is allocated runs
        // there instead of reaching the program factory's no-program error.
        auto plan = geometry.launch_plan();
        const auto reserved = pending_output_l1_bytes(input, geometry.output_spec());
        if (!live_rm_choice(input, logical_output_shape, output_mem_config.buffer_type(), num_cores, plan, reserved)) {
            return std::nullopt;
        }
        return plan;
    }
    auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    // Demotion is settled before the static fit, so a native-routed call builds no output spec.
    if (!choice || native_faster(input, *choice, reshape_core_grid(input, sub_core_grids).num_cores()) ||
        static_fit_violation(input, logical_output_shape, output_mem_config, *choice) != nullptr) {
        return std::nullopt;
    }
    // Native needs only a few CB pages, so a call whose smallest plan does not fit live L1 runs there instead of
    // reaching the program factory's no-plan error. The output is reserved because the op allocates it before the
    // program factory samples live L1.
    const auto reserved = pending_output_l1_bytes(input, output_spec(input, logical_output_shape, output_mem_config));
    if (!live_tile_rows_choice(input, logical_output_shape, *choice, reserved)) {
        return std::nullopt;
    }
    return LaunchPlan{.tile_rows = std::move(*choice)};
}

LaunchPlan forced_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids);
    TT_FATAL(violation == nullptr, "reshape_force_codegen: outside the codegen scope: {}", violation);
    if (input.layout() == Layout::ROW_MAJOR) {
        RmGeometry geometry(input, logical_output_shape, output_mem_config, rm_grid_cores(input, sub_core_grids));
        TT_FATAL(geometry.fits_static(), "reshape_force_codegen: outside the codegen scope: {}", kRmStaticFitViolation);
        return geometry.launch_plan();
    }
    auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    TT_FATAL(choice.has_value(), "reshape_force_codegen: outside the codegen scope: {}", kPlanCapsViolation);
    violation = static_fit_violation(input, logical_output_shape, output_mem_config, *choice);
    TT_FATAL(violation == nullptr, "reshape_force_codegen: outside the codegen scope: {}", violation);
    return LaunchPlan{.tile_rows = std::move(*choice)};
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
