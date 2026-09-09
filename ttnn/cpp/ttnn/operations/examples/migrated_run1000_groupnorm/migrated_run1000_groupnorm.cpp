// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "migrated_run1000_groupnorm.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::migration::generated_migrated_run1000_groupnorm {

namespace metal = tt::tt_metal;

namespace {

// ===========================================================================
// Constants transcribed from the source module (never re-tuned)
// ===========================================================================

constexpr std::int64_t BATCH_BLOCK = 1;    // batch elements per work unit
constexpr std::int64_t CLUSTER_BLOCK = 1;  // clusters per work unit
constexpr std::int64_t INPUT_DEPTH = 2;    // cb_input_tiles depth
constexpr std::int64_t OUTPUT_DEPTH = 2;   // cb_output_tiles depth
constexpr std::int64_t RM_DEPTH = 2;       // cb_rm_sticks depth (ROW_MAJOR only)
constexpr std::int64_t TARGET_HW_BLOCKS = 8;
constexpr std::int64_t MIN_BLOCK_HW_TILES = 2;

// L1 budget left for this op's circular buffers, in bytes.
constexpr std::int64_t USABLE_L1 = 900 * 1024;

// CB indices (semantic names; the numeric slot is only an index)
constexpr std::uint32_t CB_INPUT_TILES = 0;
constexpr std::uint32_t CB_OUTPUT_TILES = 1;
constexpr std::uint32_t CB_MASK_TILES = 2;
constexpr std::uint32_t CB_SCALER_ONES = 3;
constexpr std::uint32_t CB_MOMENT_SUM = 4;
constexpr std::uint32_t CB_MOMENT_SQ = 5;
constexpr std::uint32_t CB_MASKED_MOMENT = 6;
constexpr std::uint32_t CB_MOMENT_SCALAR = 7;
constexpr std::uint32_t CB_MEAN_TILES = 8;
constexpr std::uint32_t CB_SCALE_TILES = 9;
constexpr std::uint32_t CB_GAMMA_TILES = 10;
constexpr std::uint32_t CB_BETA_TILES = 11;
constexpr std::uint32_t CB_RM_STICKS = 12;
constexpr std::uint32_t CB_MEAN_SCALAR = 13;
constexpr std::uint32_t CB_SCALE_SCALAR = 14;
constexpr std::uint32_t CB_AFFINE_SCRATCH = 15;
constexpr std::uint32_t CB_ROW_MASK = 16;
constexpr std::uint32_t CB_PARTIAL_MOMENTS = 17;
constexpr std::uint32_t CB_MOMENT_TOTAL_OUT = 18;
constexpr std::uint32_t CB_MOMENT_TOTAL_IN = 19;

constexpr std::int64_t TILE_HW = 32;

// Regimes (pinned into the shared `regime_id` compile-time arg).
constexpr std::int64_t REGIME_CLUSTER_PARALLEL = 0;
constexpr std::int64_t REGIME_HW_SPLIT = 1;

// Semaphore ids. Mcast1D owns id 0 (data_ready); the gather counter is ours.
constexpr std::uint32_t SEM_MCAST_BASE = 0;
constexpr std::uint32_t SEM_GATHER = 1;

// Number of scalar args the mcast helper contributes (McastArgs' CT/RT blocks).
constexpr std::size_t MCAST_CT_ARGS = 6;
constexpr std::size_t MCAST_RT_ARGS = 4;

// `bfloat8_b` is BLOCK-quantized: there is no scalar element size, so `1` is the
// sentinel meaning "the operand is bfp8_b, decode it".
constexpr std::int64_t BFP8_ELEM_SENTINEL = 1;

// Indicates that no consumer-ready semaphore is configured (source SDK helper).
constexpr std::uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// ===========================================================================
// Small Python-semantics helpers
// ===========================================================================

[[noreturn]] void raise_error(PyErrKind kind, std::string message) { throw HostError(kind, std::move(message)); }

std::int64_t py_floordiv(std::int64_t a, std::int64_t b) {
    if (b == 0) {
        raise_error(PyErrKind::ZeroDivisionError, "integer division or modulo by zero");
    }
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

std::int64_t py_mod(std::int64_t a, std::int64_t b) {
    if (b == 0) {
        raise_error(PyErrKind::ZeroDivisionError, "integer modulo by zero");
    }
    std::int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        r += b;
    }
    return r;
}

std::uint32_t f32_bits(double value) { return std::bit_cast<std::uint32_t>(static_cast<float>(value)); }

// `int(os.environ[name])` semantics, including the ValueError on garbage.
std::int64_t parse_int_env(const char* name, const char* text) {
    std::string_view sv(text);
    std::size_t begin = 0;
    std::size_t end = sv.size();
    const auto is_space = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (begin < end && is_space(sv[begin])) {
        ++begin;
    }
    while (end > begin && is_space(sv[end - 1])) {
        --end;
    }
    std::string body(sv.substr(begin, end - begin));
    bool negative = false;
    std::size_t index = 0;
    if (index < body.size() && (body[index] == '+' || body[index] == '-')) {
        negative = body[index] == '-';
        ++index;
    }
    if (index >= body.size()) {
        raise_error(
            PyErrKind::ValueError,
            std::string("invalid literal for int() with base 10: '") + text + "' (from " + name + ")");
    }
    std::int64_t value = 0;
    for (; index < body.size(); ++index) {
        const char c = body[index];
        if (c < '0' || c > '9') {
            raise_error(
                PyErrKind::ValueError,
                std::string("invalid literal for int() with base 10: '") + text + "' (from " + name + ")");
        }
        value = value * 10 + (c - '0');
    }
    return negative ? -value : value;
}

// `_knob(name, default)` -- sweep hook for the block/depth co-tune. An unset or
// empty variable keeps the module-level knob as the single source of truth.
std::int64_t knob(const char* name, std::int64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return fallback;
    }
    return parse_int_env(name, text);
}

std::string dtype_repr(metal::DataType dtype) {
    switch (dtype) {
        case metal::DataType::BFLOAT16: return "DataType.BFLOAT16";
        case metal::DataType::FLOAT32: return "DataType.FLOAT32";
        case metal::DataType::UINT32: return "DataType.UINT32";
        case metal::DataType::BFLOAT8_B: return "DataType.BFLOAT8_B";
        case metal::DataType::BFLOAT4_B: return "DataType.BFLOAT4_B";
        case metal::DataType::UINT8: return "DataType.UINT8";
        case metal::DataType::UINT16: return "DataType.UINT16";
        case metal::DataType::INT32: return "DataType.INT32";
        case metal::DataType::FP8_E4M3: return "DataType.FP8_E4M3";
        case metal::DataType::INT8: return "DataType.INT8";
        case metal::DataType::INVALID: return "DataType.INVALID";
    }
    return "DataType.<unknown>";
}

std::string layout_repr(metal::Layout layout) {
    switch (layout) {
        case metal::Layout::ROW_MAJOR: return "Layout.ROW_MAJOR";
        case metal::Layout::TILE: return "Layout.TILE";
        case metal::Layout::INVALID: return "Layout.INVALID";
    }
    return "Layout.<unknown>";
}

std::string shape_repr(const metal::Shape& shape) {
    std::string out = "(";
    for (std::size_t i = 0; i < shape.rank(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(static_cast<std::int64_t>(shape[i]));
    }
    if (shape.rank() == 1) {
        out += ",";
    }
    out += ")";
    return out;
}

// ===========================================================================
// Faithful reproduction of the source SDK host helper `Mcast1D`
// ===========================================================================
//
// The operation constructs exactly one configuration:
//
//   Mcast1D(device, all_cores, Mcast1DShape.PerRow, /*starting_sender_index=*/0,
//           McastConfig(noc=NOC.RISCV_1_default, handshake=False,
//                       base_sem_id=SEM_MCAST_BASE))
//
// i.e. one dense receiver rectangle, per-row lines, uniform sender placement,
// non-rotating sender taken from the receiver grid itself, Flag data-ready mode
// and helper-owned semaphores. Every value this class derives (semaphore
// descriptors, the 6-word CT block, the 4-word per-core RT block, and the
// NOC_1 rectangle-corner ordering) mirrors `mcast_host.cpp` for that
// configuration.
class Mcast1DPerRow {
public:
    Mcast1DPerRow(
        metal::distributed::MeshDevice* device,
        const metal::CoreRangeSet& receiver_grid,
        std::uint32_t starting_sender_index,
        std::uint32_t base_sem_id) :
        device_(device), starting_sender_index_(starting_sender_index) {
        TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");

        const auto receiver_box = receiver_grid.bounding_box();
        TT_FATAL(
            receiver_grid.num_cores() == receiver_box.size(),
            "Mcast1D: receiver grid must be one dense rectangle (bounding box has {} cores, set has {})",
            receiver_box.size(),
            receiver_grid.num_cores());

        origin_x_ = static_cast<std::uint32_t>(receiver_box.start_coord.x);
        origin_y_ = static_cast<std::uint32_t>(receiver_box.start_coord.y);
        const auto columns = static_cast<std::uint32_t>(receiver_box.end_coord.x - receiver_box.start_coord.x) + 1u;
        const auto rows = static_cast<std::uint32_t>(receiver_box.end_coord.y - receiver_box.start_coord.y) + 1u;

        // PerRow: each line is one grid row; the sender grid equals the receiver
        // grid, so every line spans the full receiver width.
        receiver_span_ = columns;
        span_ = columns;
        num_lines_ = rows;

        TT_FATAL(
            starting_sender_index_ < span_,
            "Mcast1D: starting_sender_index {} must be less than the sender-line span {}",
            starting_sender_index_,
            span_);

        // Every line's sender lies inside the receiver grid, so its fan-out is
        // receiver_span_ - 1 and it is uniform across lines.
        const std::uint32_t fanout = receiver_span_ - 1u;
        active_ = fanout > 0u;
        ack_count_ = fanout;
        grid_ = receiver_grid;

        data_ready_id_ = base_sem_id;
        consumer_ready_id_ = UNUSED_SEM_ID;  // handshake = false
    }

    std::vector<metal::SemaphoreDescriptor> owned_semaphores() const {
        std::vector<metal::SemaphoreDescriptor> semaphores;
        semaphores.push_back(
            metal::SemaphoreDescriptor{.id = data_ready_id_, .core_ranges = grid_, .initial_value = 0});
        return semaphores;
    }

    std::array<std::uint32_t, MCAST_CT_ARGS> compile_time_args() const {
        // flags: bit0 = pre-handshake (off), bit1 = counter data-ready (off).
        return {active_ ? 1u : 0u, data_ready_id_, consumer_ready_id_, ack_count_, 0u, 0u};
    }

    std::array<std::uint32_t, MCAST_RT_ARGS> runtime_args(const metal::CoreCoord& core) const {
        if (is_sender(core)) {
            return sender_rect(core);
        }
        const auto sender = sender_of(core);
        const auto virtual_sender = virt(sender);
        return {virtual_sender.first, virtual_sender.second, 0u, 0u};
    }

    bool is_sender(const metal::CoreCoord& core) const {
        const auto x = static_cast<std::uint32_t>(core.x);
        const auto y = static_cast<std::uint32_t>(core.y);
        return x == origin_x_ + starting_sender_index_ && y >= origin_y_ && y < origin_y_ + num_lines_;
    }

private:
    std::pair<std::uint32_t, std::uint32_t> virt(const metal::CoreCoord& logical) const {
        const auto worker = device_->worker_core_from_logical_core(logical);
        return {static_cast<std::uint32_t>(worker.x), static_cast<std::uint32_t>(worker.y)};
    }

    metal::CoreCoord sender_of(const metal::CoreCoord& core) const {
        return metal::CoreCoord{origin_x_ + starting_sender_index_, core.y};
    }

    metal::CoreCoord line_coord(const metal::CoreCoord& core, std::uint32_t i) const {
        return metal::CoreCoord{origin_x_ + i, core.y};
    }

    // NOC_1 (== RISCV_1_default) orders the rectangle corners high-to-low.
    static std::array<std::uint32_t, MCAST_RT_ARGS> noc_ordered_bbox(
        const std::vector<std::pair<std::uint32_t, std::uint32_t>>& coordinates) {
        std::uint32_t xlo = coordinates[0].first;
        std::uint32_t xhi = coordinates[0].first;
        std::uint32_t ylo = coordinates[0].second;
        std::uint32_t yhi = coordinates[0].second;
        for (const auto& coordinate : coordinates) {
            xlo = std::min(xlo, coordinate.first);
            xhi = std::max(xhi, coordinate.first);
            ylo = std::min(ylo, coordinate.second);
            yhi = std::max(yhi, coordinate.second);
        }
        return {xhi, yhi, xlo, ylo};
    }

    std::array<std::uint32_t, MCAST_RT_ARGS> sender_rect(const metal::CoreCoord& core) const {
        const auto sender = sender_of(core);

        if (receiver_span_ == 1u) {
            return noc_ordered_bbox({virt(sender)});
        }

        const std::uint32_t sender_index = static_cast<std::uint32_t>(sender.x) - origin_x_;
        if (sender_index == 0u) {
            return noc_ordered_bbox({virt(line_coord(core, 1u)), virt(line_coord(core, receiver_span_ - 1u))});
        }
        if (sender_index == receiver_span_ - 1u) {
            return noc_ordered_bbox({virt(line_coord(core, 0u)), virt(line_coord(core, receiver_span_ - 2u))});
        }

        std::vector<std::pair<std::uint32_t, std::uint32_t>> coordinates;
        coordinates.reserve(receiver_span_);
        for (std::uint32_t i = 0; i < receiver_span_; ++i) {
            coordinates.push_back(virt(line_coord(core, i)));
        }
        return noc_ordered_bbox(coordinates);
    }

    metal::distributed::MeshDevice* device_ = nullptr;
    metal::CoreRangeSet grid_;
    std::uint32_t starting_sender_index_ = 0;
    std::uint32_t origin_x_ = 0;
    std::uint32_t origin_y_ = 0;
    std::uint32_t span_ = 1;
    std::uint32_t receiver_span_ = 1;
    std::uint32_t num_lines_ = 1;
    bool active_ = false;
    std::uint32_t ack_count_ = 0;
    std::uint32_t data_ready_id_ = 0;
    std::uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

// ===========================================================================
// Kernel path resolution (relative to this translation unit, never hardcoded)
// ===========================================================================

const std::filesystem::path& kernel_dir() {
    static const std::filesystem::path dir = std::filesystem::path(__FILE__).parent_path() / "source" / "kernels";
    return dir;
}

std::string kernel_path(const char* file_name) { return (kernel_dir() / file_name).string(); }

// ===========================================================================
// Planner helpers
// ===========================================================================

struct ClusterGeometry {
    std::int64_t Cg = 0;
    std::int64_t cluster_channels = 0;
    std::int64_t groups_per_cluster = 0;
    std::int64_t cluster_c_tiles = 0;
    std::int64_t num_clusters = 0;
};

// `_cluster_geometry()`. `C % 32 != 0` falls back to the degenerate cluster:
// the WHOLE channel axis is one cluster.
ClusterGeometry cluster_geometry(std::int64_t C, std::int64_t num_groups) {
    ClusterGeometry geometry;
    geometry.Cg = py_floordiv(C, num_groups);
    if (py_mod(C, TILE_HW) != 0) {
        geometry.cluster_channels = C;
        geometry.groups_per_cluster = num_groups;
        geometry.cluster_c_tiles = py_floordiv(C + TILE_HW - 1, TILE_HW);
        geometry.num_clusters = 1;
        return geometry;
    }
    const std::int64_t g = std::gcd(TILE_HW, geometry.Cg);
    geometry.cluster_channels = py_floordiv(TILE_HW * geometry.Cg, g);
    geometry.groups_per_cluster = py_floordiv(TILE_HW, g);
    geometry.cluster_c_tiles = py_floordiv(geometry.Cg, g);
    geometry.num_clusters = py_floordiv(C, geometry.cluster_channels);
    return geometry;
}

struct MaskSpans {
    std::int64_t num_mask_tiles = 0;
    std::int64_t max_span = 0;
};

// `_mask_spans()` -- per-group channel-tile span inside a cluster.
MaskSpans mask_spans(std::int64_t Cg, std::int64_t groups_per_cluster) {
    MaskSpans spans;
    bool have_span = false;
    for (std::int64_t j = 0; j < groups_per_cluster; ++j) {
        const std::int64_t lo = j * Cg;
        const std::int64_t hi = lo + Cg;
        const std::int64_t t0 = py_floordiv(lo, TILE_HW);
        const std::int64_t t1 = py_floordiv(hi - 1, TILE_HW);
        const std::int64_t span = t1 - t0 + 1;
        spans.num_mask_tiles += span;
        if (!have_span) {
            spans.max_span = span;
            have_span = true;
        } else {
            spans.max_span = std::max(spans.max_span, span);
        }
    }
    return spans;
}

// `_elem_size()`
std::int64_t elem_size(const std::optional<Tensor>& tensor) {
    if (!tensor.has_value()) {
        return 4;
    }
    if (tensor->dtype() == metal::DataType::BFLOAT8_B) {
        return BFP8_ELEM_SENTINEL;
    }
    return static_cast<std::int64_t>(tensor->element_size());
}

std::int64_t elem_size(const Tensor& tensor) {
    if (tensor.dtype() == metal::DataType::BFLOAT8_B) {
        return BFP8_ELEM_SENTINEL;
    }
    return static_cast<std::int64_t>(tensor.element_size());
}

// `_affine_axes()` -- presence plus the "none"-sentinel dtype/layout axes.
enum class AffineAxis : std::uint8_t { NoAffine, GammaOnly, GammaBeta };

struct AffineAxes {
    AffineAxis affine = AffineAxis::NoAffine;
    bool has_ref = false;
    metal::DataType dtype = metal::DataType::BFLOAT16;
    metal::Layout layout = metal::Layout::TILE;
};

AffineAxes affine_axes(const std::optional<Tensor>& gamma, const std::optional<Tensor>& beta) {
    AffineAxes axes;
    if (!gamma.has_value() && !beta.has_value()) {
        axes.affine = AffineAxis::NoAffine;
        axes.has_ref = false;
        return axes;
    }
    const Tensor& ref = gamma.has_value() ? *gamma : *beta;
    axes.affine = (gamma.has_value() && beta.has_value()) ? AffineAxis::GammaBeta : AffineAxis::GammaOnly;
    axes.has_ref = true;
    axes.dtype = ref.dtype();
    axes.layout = ref.layout();
    return axes;
}

std::string affine_repr(AffineAxis affine) {
    switch (affine) {
        case AffineAxis::NoAffine: return "'no_affine'";
        case AffineAxis::GammaOnly: return "'gamma_only'";
        case AffineAxis::GammaBeta: return "'gamma_beta'";
    }
    return "'<unknown>'";
}

// `tag_alignment()`
enum class Alignment : std::uint8_t { TileAligned, HwNonAligned, CNonAligned };

Alignment tag_alignment(std::int64_t HW, std::int64_t C) {
    if (py_mod(C, TILE_HW) != 0) {
        return Alignment::CNonAligned;
    }
    if (py_mod(HW, TILE_HW) != 0) {
        return Alignment::HwNonAligned;
    }
    return Alignment::TileAligned;
}

void append(std::vector<std::uint32_t>& target, const std::vector<std::uint32_t>& values) {
    target.insert(target.end(), values.begin(), values.end());
}

std::vector<std::uint32_t> accessor_compile_time_args(const std::optional<Tensor>& tensor) {
    if (!tensor.has_value()) {
        return metal::TensorAccessorArgs{}.get_compile_time_args();
    }
    return metal::TensorAccessorArgs(*tensor->buffer()).get_compile_time_args();
}

std::vector<std::uint32_t> accessor_compile_time_args(const Tensor& tensor) {
    return metal::TensorAccessorArgs(*tensor.buffer()).get_compile_time_args();
}

// ===========================================================================
// The descriptor half of `create_program_descriptor()`
// ===========================================================================

metal::ProgramDescriptor build_descriptor(
    const Plan& plan,
    const Tensor& input_tensor,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const Tensor& output_tensor) {
    metal::ProgramDescriptor desc;

    // ------------------------- Circular buffers ---------------------------
    const auto add_cb =
        [&desc, &plan](
            std::uint32_t index, std::uint32_t page_size, std::uint32_t num_pages, metal::DataType data_format) {
            metal::CBDescriptor cb;
            cb.total_size = page_size * num_pages;
            cb.core_ranges = plan.all_cores;
            cb.format_descriptors.push_back(metal::CBFormatDescriptor{
                .buffer_index = static_cast<std::uint8_t>(index),
                .data_format = metal::datatype_to_dataformat_converter(data_format),
                .page_size = page_size});
            desc.cbs.push_back(std::move(cb));
        };

    add_cb(CB_INPUT_TILES, plan.tile_in, plan.input_cb_pages, plan.input_dtype);
    add_cb(CB_OUTPUT_TILES, plan.tile_out, plan.output_depth * plan.block_tiles, plan.output_dtype);
    add_cb(CB_MASK_TILES, plan.tile_bf16, plan.num_mask_tiles, metal::DataType::BFLOAT16);
    add_cb(CB_SCALER_ONES, plan.tile_bf16, 1, metal::DataType::BFLOAT16);
    add_cb(CB_MOMENT_SUM, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_MOMENT_SQ, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_MASKED_MOMENT, plan.tile_f32, plan.max_span, metal::DataType::FLOAT32);
    add_cb(CB_MOMENT_SCALAR, plan.tile_f32, 2, metal::DataType::FLOAT32);
    add_cb(CB_MEAN_TILES, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_SCALE_TILES, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_GAMMA_TILES, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_BETA_TILES, plan.tile_f32, plan.cluster_c_tiles, metal::DataType::FLOAT32);
    add_cb(CB_MEAN_SCALAR, plan.tile_f32, 1, metal::DataType::FLOAT32);
    add_cb(CB_SCALE_SCALAR, plan.tile_f32, 1, metal::DataType::FLOAT32);
    add_cb(CB_AFFINE_SCRATCH, plan.affine_scratch_bytes, 1, metal::DataType::FLOAT32);
    add_cb(CB_ROW_MASK, plan.tile_bf16, 1, metal::DataType::BFLOAT16);

    if (plan.regime == static_cast<std::uint32_t>(REGIME_HW_SPLIT)) {
        // Landing zone on the combiner for the (S-1) remote partial pairs; the
        // CB set is declared on the same core range so the base address is
        // identical on every core.
        const std::uint32_t partial_pages =
            std::max<std::uint32_t>(1u, (plan.hw_split_factor - 1u) * 2u * plan.cluster_c_tiles);
        add_cb(CB_PARTIAL_MOMENTS, plan.tile_f32, partial_pages, metal::DataType::FLOAT32);
        add_cb(CB_MOMENT_TOTAL_OUT, plan.tile_f32, 2u * plan.cluster_c_tiles, metal::DataType::FLOAT32);
        add_cb(CB_MOMENT_TOTAL_IN, plan.tile_f32, 2u * plan.cluster_c_tiles, metal::DataType::FLOAT32);
    }

    if (plan.input_is_rm) {
        add_cb(
            CB_RM_STICKS,
            static_cast<std::uint32_t>(TILE_HW) * plan.in_elem,
            plan.rm_depth * static_cast<std::uint32_t>(TILE_HW) * plan.block_hw_tiles,
            plan.input_dtype);
    }

    // ------------------- Shared scalar compile-time args -------------------
    const std::vector<std::uint32_t> geom_ct = {
        plan.tensor_hw_tiles,
        plan.tensor_c_tiles,
        plan.cluster_c_tiles,
        plan.groups_per_cluster,
        plan.num_clusters,
        plan.block_hw_tiles,
        plan.num_hw_blocks,
        plan.Cg,
        plan.cluster_channels,
        plan.num_mask_tiles,
        plan.max_span,
        plan.hw_tail,                                      // (11) valid rows in the LAST HW tile; 0 => tile-aligned
        plan.c_tail,                                       // (12) valid channels in the LAST channel tile
        plan.regime,                                       // (13)
        plan.hw_split_factor,                              // (14)
        static_cast<std::uint32_t>(plan.resident ? 1 : 0)  // (15)
    };

    // ------------------------------ Reader ---------------------------------
    std::vector<std::uint32_t> reader_ct = geom_ct;
    reader_ct.push_back(plan.has_gamma ? 1u : 0u);
    reader_ct.push_back(plan.has_beta ? 1u : 0u);
    reader_ct.push_back(plan.input_is_rm ? 1u : 0u);
    reader_ct.push_back(plan.in_elem);
    reader_ct.push_back(plan.affine_is_rm ? 1u : 0u);
    reader_ct.push_back(plan.affine_elem);
    reader_ct.push_back(plan.C);
    reader_ct.push_back(plan.HW);
    reader_ct.push_back(plan.affine_page_bytes);  // (24)
    append(reader_ct, accessor_compile_time_args(input_tensor));
    append(reader_ct, accessor_compile_time_args(plan.has_gamma ? gamma : std::optional<Tensor>{}));
    append(reader_ct, accessor_compile_time_args(plan.has_beta ? beta : std::optional<Tensor>{}));

    // ---------------- Writer (owns the cross-core combine) ------------------
    std::vector<std::uint32_t> writer_ct = geom_ct;
    for (std::size_t i = 0; i < MCAST_CT_ARGS; ++i) {
        writer_ct.push_back(plan.mcast_ct[i]);
    }
    append(writer_ct, accessor_compile_time_args(output_tensor));

    // ------------------------------ Compute --------------------------------
    std::vector<std::uint32_t> compute_ct = geom_ct;
    compute_ct.push_back(plan.inv_n_g_bits);
    compute_ct.push_back(plan.eps_bits);
    compute_ct.push_back(plan.input_is_rm ? 1u : 0u);

    metal::KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kernel_path("groupnorm_sc_N_1_HW_C_reader.cpp");
    reader_kernel.source_type = metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = plan.all_cores;
    reader_kernel.compile_time_args = std::move(reader_ct);
    reader_kernel.config = metal::ReaderConfigDescriptor{};

    metal::KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kernel_path("groupnorm_sc_N_1_HW_C_writer.cpp");
    writer_kernel.source_type = metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = plan.all_cores;
    writer_kernel.compile_time_args = std::move(writer_ct);
    writer_kernel.config = metal::WriterConfigDescriptor{};

    metal::KernelDescriptor compute_kernel;
    compute_kernel.kernel_source = kernel_path("groupnorm_sc_N_1_HW_C_compute.cpp");
    compute_kernel.source_type = metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = plan.all_cores;
    compute_kernel.compile_time_args = std::move(compute_ct);
    compute_kernel.config = metal::ComputeConfigDescriptor{
        .math_fidelity = plan.compute.math_fidelity,
        .fp32_dest_acc_en = plan.compute.fp32_dest_acc_en,
        .dst_full_sync_en = plan.compute.dst_full_sync_en,
        .math_approx_mode = plan.compute.math_approx_mode,
    };

    // ---------------------- Per-core runtime args --------------------------
    metal::Buffer* in_buffer = input_tensor.buffer();
    metal::Buffer* out_buffer = output_tensor.buffer();
    metal::Buffer* gamma_buffer = plan.has_gamma ? gamma->buffer() : nullptr;
    metal::Buffer* beta_buffer = plan.has_beta ? beta->buffer() : nullptr;

    for (const CoreArgs& core_args : plan.core_args) {
        metal::KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(6);
        reader_rt.push_back(in_buffer);
        reader_rt.push_back(gamma_buffer);
        reader_rt.push_back(beta_buffer);
        reader_rt.push_back(core_args.unit_start);
        reader_rt.push_back(core_args.units);
        reader_rt.push_back(core_args.hw_tile_start);
        reader_kernel.emplace_runtime_args(core_args.core, reader_rt);

        metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.reserve(6 + MCAST_RT_ARGS);
        writer_rt.push_back(out_buffer);
        writer_rt.push_back(core_args.unit_start);
        writer_rt.push_back(core_args.units);
        writer_rt.push_back(core_args.hw_tile_start);
        writer_rt.push_back(core_args.is_sender);
        writer_rt.push_back(core_args.gather_slot);
        for (std::size_t i = 0; i < MCAST_RT_ARGS; ++i) {
            writer_rt.push_back(core_args.mcast_rt[i]);
        }
        writer_kernel.emplace_runtime_args(core_args.core, writer_rt);

        compute_kernel.runtime_args.emplace_back(
            core_args.core,
            metal::KernelDescriptor::CoreRuntimeArgs{
                core_args.unit_start, core_args.units, core_args.is_sender, core_args.owns_hw_tail});
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    for (const auto& semaphore : plan.semaphores) {
        desc.semaphores.push_back(semaphore);
    }

    return desc;
}

}  // namespace

// ===========================================================================
// `_validate_args()`
// ===========================================================================

void validate_arguments(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta) {
    const metal::Shape& shape = input_tensor.logical_shape();
    if (shape.rank() != 4) {
        raise_error(
            PyErrKind::ValueError, "groupnorm_sc_N_1_HW_C: input must be rank 4, got shape " + shape_repr(shape));
    }
    if (static_cast<std::int64_t>(shape[1]) != 1) {
        raise_error(
            PyErrKind::ValueError,
            "groupnorm_sc_N_1_HW_C: input dim[1] must be 1, got " +
                std::to_string(static_cast<std::int64_t>(shape[1])));
    }
    if (num_groups < 1) {
        raise_error(
            PyErrKind::ValueError,
            "groupnorm_sc_N_1_HW_C: num_groups must be a positive int, got " + std::to_string(num_groups));
    }
    const std::int64_t C = static_cast<std::int64_t>(shape[3]);
    if (py_mod(C, num_groups) != 0) {
        raise_error(
            PyErrKind::ValueError,
            "groupnorm_sc_N_1_HW_C: channels C=" + std::to_string(C) +
                " not divisible by num_groups=" + std::to_string(num_groups));
    }
    const std::pair<const char*, const std::optional<Tensor>*> affine[2] = {{"gamma", &gamma}, {"beta", &beta}};
    for (const auto& [name, tensor] : affine) {
        if (!tensor->has_value()) {
            continue;
        }
        const metal::Shape& affine_shape = (*tensor)->logical_shape();
        const bool matches = affine_shape.rank() == 4 && static_cast<std::int64_t>(affine_shape[0]) == 1 &&
                             static_cast<std::int64_t>(affine_shape[1]) == 1 &&
                             static_cast<std::int64_t>(affine_shape[2]) == 1 &&
                             static_cast<std::int64_t>(affine_shape[3]) == C;
        if (!matches) {
            raise_error(
                PyErrKind::ValueError,
                std::string("groupnorm_sc_N_1_HW_C: ") + name + " shape must be (1, 1, 1, " + std::to_string(C) +
                    "), got " + shape_repr(affine_shape));
        }
    }
}

// ===========================================================================
// `validate()` -- INPUT_TAGGERS / SUPPORTED / EXCLUSIONS
// ===========================================================================

void validate_support(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta) {
    const AffineAxes affine = affine_axes(gamma, beta);

    const metal::Shape& shape = input_tensor.logical_shape();
    // `(N, 1, HW, C)`: last two dims are HW (-2) and C (-1).
    const std::int64_t HW = static_cast<std::int64_t>(shape[shape.rank() - 2]);
    const std::int64_t C = static_cast<std::int64_t>(shape[shape.rank() - 1]);

    // `num_groups` is intentionally absent from SUPPORTED: it is an unbounded
    // integer coupled to the shape, not a finite axis. Tagged for completeness.
    (void)num_groups;

    // SUPPORTED["dtype"]
    const metal::DataType dtype = input_tensor.dtype();
    const bool dtype_ok =
        dtype == metal::DataType::BFLOAT16 || dtype == metal::DataType::FLOAT32 || dtype == metal::DataType::BFLOAT8_B;
    if (!dtype_ok) {
        raise_error(
            PyErrKind::UnsupportedAxisValue,
            "groupnorm_sc_N_1_HW_C: dtype=" + dtype_repr(dtype) +
                " not in SUPPORTED [DataType.BFLOAT16, DataType.FLOAT32, DataType.BFLOAT8_B]");
    }

    // SUPPORTED["layout"]
    const metal::Layout layout = input_tensor.layout();
    const bool layout_ok = layout == metal::Layout::TILE || layout == metal::Layout::ROW_MAJOR;
    if (!layout_ok) {
        raise_error(
            PyErrKind::UnsupportedAxisValue,
            "groupnorm_sc_N_1_HW_C: layout=" + layout_repr(layout) +
                " not in SUPPORTED [Layout.TILE, Layout.ROW_MAJOR]");
    }

    // SUPPORTED["affine"] -- every tagger value is enumerated, so this can only
    // pass; kept so the contract stays visible and future EXCLUSIONS can use it.
    const bool affine_ok = affine.affine == AffineAxis::NoAffine || affine.affine == AffineAxis::GammaOnly ||
                           affine.affine == AffineAxis::GammaBeta;
    if (!affine_ok) {
        raise_error(
            PyErrKind::UnsupportedAxisValue,
            "groupnorm_sc_N_1_HW_C: affine=" + affine_repr(affine.affine) +
                " not in SUPPORTED ['gamma_beta', 'gamma_only', 'no_affine']");
    }

    // SUPPORTED["affine_dtype"] -- "none" = no affine tensor supplied.
    if (affine.has_ref) {
        const bool affine_dtype_ok = affine.dtype == metal::DataType::BFLOAT16 ||
                                     affine.dtype == metal::DataType::FLOAT32 ||
                                     affine.dtype == metal::DataType::BFLOAT8_B;
        if (!affine_dtype_ok) {
            raise_error(
                PyErrKind::UnsupportedAxisValue,
                "groupnorm_sc_N_1_HW_C: affine_dtype=" + dtype_repr(affine.dtype) +
                    " not in SUPPORTED [DataType.BFLOAT16, DataType.FLOAT32, DataType.BFLOAT8_B, 'none']");
        }

        // SUPPORTED["affine_layout"]
        const bool affine_layout_ok = affine.layout == metal::Layout::TILE || affine.layout == metal::Layout::ROW_MAJOR;
        if (!affine_layout_ok) {
            raise_error(
                PyErrKind::UnsupportedAxisValue,
                "groupnorm_sc_N_1_HW_C: affine_layout=" + layout_repr(affine.layout) +
                    " not in SUPPORTED [Layout.TILE, Layout.ROW_MAJOR, 'none']");
        }
    }

    // SUPPORTED["alignment"] -- all three tagger values are supported; the
    // tagger is total, so this axis can never refuse.
    const Alignment alignment = tag_alignment(HW, C);
    switch (alignment) {
        case Alignment::TileAligned:
        case Alignment::HwNonAligned:
        case Alignment::CNonAligned: break;
    }

    // EXCLUSIONS is empty: no cell inside SUPPORTED is refused. An ExcludedCell
    // is therefore unreachable, but the identity stays wired for the contract.
}

// ===========================================================================
// `create_program_descriptor()` planner half
// ===========================================================================

Plan build_plan(
    const Tensor& input_tensor,
    metal::DataType output_dtype,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    double eps,
    const ComputeConfigFields& compute) {
    metal::distributed::MeshDevice* device = input_tensor.device();
    TT_FATAL(device != nullptr, "groupnorm_sc_N_1_HW_C: input tensor must be allocated on a device");

    const metal::Shape& shape = input_tensor.logical_shape();
    const std::int64_t N = static_cast<std::int64_t>(shape[0]);
    const std::int64_t HW = static_cast<std::int64_t>(shape[2]);
    const std::int64_t C = static_cast<std::int64_t>(shape[3]);

    const ClusterGeometry geometry = cluster_geometry(C, num_groups);
    const MaskSpans spans = mask_spans(geometry.Cg, geometry.groups_per_cluster);

    const std::int64_t tensor_hw_tiles = py_floordiv(HW + TILE_HW - 1, TILE_HW);
    const std::int64_t tensor_c_tiles = py_floordiv(C + TILE_HW - 1, TILE_HW);

    const bool input_is_rm = input_tensor.layout() == metal::Layout::ROW_MAJOR;
    const std::int64_t in_elem = elem_size(input_tensor);
    const std::int64_t tile_in = static_cast<std::int64_t>(metal::tile_size(input_tensor.dtype()));
    const std::int64_t tile_out = static_cast<std::int64_t>(metal::tile_size(output_dtype));
    const std::int64_t tile_f32 = static_cast<std::int64_t>(metal::tile_size(metal::DataType::FLOAT32));
    const std::int64_t tile_bf16 = static_cast<std::int64_t>(metal::tile_size(metal::DataType::BFLOAT16));

    const bool has_gamma = gamma.has_value();
    const bool has_beta = beta.has_value();
    const std::optional<Tensor>& affine_ref = has_gamma ? gamma : beta;
    const bool affine_is_rm = affine_ref.has_value() && affine_ref->layout() == metal::Layout::ROW_MAJOR;
    const std::int64_t affine_elem = elem_size(affine_ref);
    // Bytes of ONE affine page as the reader must ask the NoC for it: a whole
    // (1,1,1,C) row-major row, or one tile page (1088 B for bfp8_b).
    const std::int64_t affine_page_bytes =
        affine_is_rm
            ? C * affine_elem
            : (affine_ref.has_value() ? static_cast<std::int64_t>(metal::tile_size(affine_ref->dtype())) : tile_f32);

    // ---------------- BLOCK_HW_TILES -- the coarsest extent that fits L1 -----
    const std::int64_t affine_scratch_bytes = std::max(geometry.cluster_channels * 4, tile_f32);
    const std::int64_t fixed_inventory = spans.num_mask_tiles * tile_bf16           // cb_mask_tiles
                                         + tile_bf16                                // cb_scaler_ones
                                         + tile_bf16                                // cb_row_mask
                                         + 4 * geometry.cluster_c_tiles * tile_f32  // sum/sq/mean/scale
                                         + spans.max_span * tile_f32                // cb_masked_moment
                                         + 2 * tile_f32                             // cb_moment_scalar
                                         + 2 * tile_f32                             // mean/scale scalar
                                         + 2 * geometry.cluster_c_tiles * tile_f32  // gamma/beta tiles
                                         + affine_scratch_bytes;                    // cb_affine_scratch

    const std::int64_t input_depth = knob("GN_INPUT_DEPTH", INPUT_DEPTH);
    const std::int64_t output_depth = knob("GN_OUTPUT_DEPTH", OUTPUT_DEPTH);
    const std::int64_t rm_depth = knob("GN_RM_DEPTH", RM_DEPTH);

    std::int64_t per_block_unit = geometry.cluster_c_tiles * (input_depth * tile_in + output_depth * tile_out);
    if (input_is_rm) {
        // cb_rm_sticks holds one channel-tile column of the block at a time.
        per_block_unit += rm_depth * TILE_HW * TILE_HW * in_elem;
    }

    const std::int64_t budget_base = USABLE_L1 - fixed_inventory;
    if (budget_base < per_block_unit) {
        raise_error(
            PyErrKind::NotImplementedError,
            "groupnorm_sc_N_1_HW_C: cluster of " + std::to_string(geometry.cluster_c_tiles) +
                " channel-tiles does not fit L1 (fixed inventory " + std::to_string(fixed_inventory) +
                " B, per-HW-tile " + std::to_string(per_block_unit) + " B)");
    }

    const std::int64_t units_total = py_floordiv(N, BATCH_BLOCK) * py_floordiv(geometry.num_clusters, CLUSTER_BLOCK);
    const metal::CoreCoord grid_size = device->compute_with_storage_grid_size();
    const std::int64_t grid_x = static_cast<std::int64_t>(grid_size.x);
    const std::int64_t grid_y = static_cast<std::int64_t>(grid_size.y);

    // ---------------- Regime selection --------------------------------------
    std::int64_t regime = REGIME_CLUSTER_PARALLEL;
    std::int64_t hw_split_factor = 1;
    std::int64_t hw_split_extra = 0;
    if (units_total <= grid_y && grid_x >= 2) {
        // Test hook: GN_HW_SPLIT_FACTOR pins S for the regime-pinned tests.
        // Unset in production, where S is chosen greedily.
        std::vector<std::int64_t> candidates;
        const char* pinned = std::getenv("GN_HW_SPLIT_FACTOR");
        if (pinned != nullptr && pinned[0] != '\0') {
            candidates.push_back(parse_int_env("GN_HW_SPLIT_FACTOR", pinned));
        } else {
            for (std::int64_t candidate = std::min(grid_x, py_floordiv(tensor_hw_tiles, 2)); candidate > 1;
                 --candidate) {
                candidates.push_back(candidate);
            }
        }
        for (const std::int64_t candidate : candidates) {
            if (py_mod(tensor_hw_tiles, candidate) != 0) {
                continue;
            }
            // Extra fixed L1 the combine costs on every core.
            const std::int64_t extra =
                ((candidate - 1) * 2 * geometry.cluster_c_tiles + 4 * geometry.cluster_c_tiles) * tile_f32;
            if (budget_base - extra >= per_block_unit) {
                regime = REGIME_HW_SPLIT;
                hw_split_factor = candidate;
                hw_split_extra = extra;
                break;
            }
        }
    }

    const std::int64_t per_core_hw_tiles = py_floordiv(tensor_hw_tiles, hw_split_factor);
    const std::int64_t budget = budget_base - hw_split_extra;

    // ---------------- Lever 1: residency fast path -------------------------
    const std::int64_t resident_input_bytes = per_core_hw_tiles * geometry.cluster_c_tiles * tile_in;
    const std::int64_t per_block_unit_no_input = per_block_unit - geometry.cluster_c_tiles * input_depth * tile_in;
    const char* resident_env = std::getenv("GN_RESIDENT");
    const std::string resident_text = resident_env != nullptr ? std::string(resident_env) : std::string("1");
    const bool resident = resident_text != "0" && (budget - resident_input_bytes >= per_block_unit_no_input);

    std::int64_t block_hw_tiles = 0;
    if (resident) {
        const std::int64_t block_budget = budget - resident_input_bytes;
        block_hw_tiles =
            std::max<std::int64_t>(1, std::min(py_floordiv(block_budget, per_block_unit_no_input), per_core_hw_tiles));
    } else {
        block_hw_tiles = std::max<std::int64_t>(1, std::min(py_floordiv(budget, per_block_unit), per_core_hw_tiles));
    }
    // Pipelining cap: keep ~TARGET_HW_BLOCKS blocks per core, never below the
    // granularity floor.
    const std::int64_t target_blocks = knob("GN_TARGET_HW_BLOCKS", TARGET_HW_BLOCKS);
    const std::int64_t floor_tiles = knob("GN_MIN_BLOCK_HW_TILES", MIN_BLOCK_HW_TILES);
    block_hw_tiles = std::min(block_hw_tiles, std::max(floor_tiles, py_floordiv(per_core_hw_tiles, target_blocks)));
    block_hw_tiles = std::min(block_hw_tiles, knob("GN_BLOCK_HW_TILES", block_hw_tiles));
    // Snap DOWN to a divisor of the PER-CORE HW extent so every HW block is the
    // same size and every CB transaction stays block-aligned.
    while (py_mod(per_core_hw_tiles, block_hw_tiles) != 0) {
        block_hw_tiles -= 1;
    }
    const std::int64_t num_hw_blocks = py_floordiv(per_core_hw_tiles, block_hw_tiles);

    const std::int64_t block_tiles = block_hw_tiles * geometry.cluster_c_tiles;
    const std::int64_t input_cb_pages =
        resident ? (per_core_hw_tiles * geometry.cluster_c_tiles) : (input_depth * block_tiles);

    // ---------------- Work distribution ------------------------------------
    metal::CoreRangeSet all_cores;
    std::vector<metal::CoreCoord> cores;
    metal::CoreRangeSet core_group_1;
    metal::CoreRangeSet core_group_2;
    std::int64_t units_per_core_g1 = 0;
    std::int64_t units_per_core_g2 = 0;

    if (regime == REGIME_HW_SPLIT) {
        // Core map: unit `u` owns grid ROW u, columns 0..S-1.
        all_cores = metal::CoreRangeSet(metal::CoreRange(
            metal::CoreCoord{0, 0},
            metal::CoreCoord{
                static_cast<std::size_t>(hw_split_factor - 1), static_cast<std::size_t>(units_total - 1)}));
        cores.reserve(static_cast<std::size_t>(units_total * hw_split_factor));
        for (std::int64_t y = 0; y < units_total; ++y) {
            for (std::int64_t x = 0; x < hw_split_factor; ++x) {
                cores.push_back(metal::CoreCoord{static_cast<std::size_t>(x), static_cast<std::size_t>(y)});
            }
        }
    } else {
        auto [num_cores, split_all_cores, group_1, group_2, work_1, work_2] =
            metal::split_work_to_cores(grid_size, static_cast<std::uint32_t>(units_total), true);
        all_cores = split_all_cores;
        core_group_1 = group_1;
        core_group_2 = group_2;
        units_per_core_g1 = static_cast<std::int64_t>(work_1);
        units_per_core_g2 = static_cast<std::int64_t>(work_2);
        cores = metal::grid_to_cores(
            num_cores, static_cast<std::uint32_t>(grid_size.x), static_cast<std::uint32_t>(grid_size.y), true);
    }

    // ---------------- Mcast1D (writer owns the cross-core combine) ----------
    std::array<std::uint32_t, MCAST_CT_ARGS> mcast_ct{0u, 0u, UNUSED_SEM_ID, 0u, 0u, 0u};
    std::vector<metal::SemaphoreDescriptor> semaphores;
    std::optional<Mcast1DPerRow> mcast;
    if (regime == REGIME_HW_SPLIT) {
        // The writer runs on RISCV_0 / NOC_1 (WriterConfigDescriptor), so the
        // rectangle corners are ordered for NOC_1.
        mcast.emplace(device, all_cores, /*starting_sender_index=*/0u, SEM_MCAST_BASE);
        mcast_ct = mcast->compile_time_args();
        semaphores = mcast->owned_semaphores();
        semaphores.push_back(metal::SemaphoreDescriptor{
            .id = SEM_GATHER, .core_ranges = all_cores, .initial_value = 0});  // gather counter
    }

    // ---------------- Per-core runtime args --------------------------------
    std::vector<CoreArgs> core_args;
    core_args.reserve(cores.size());
    std::int64_t unit_start = 0;
    for (const metal::CoreCoord& core : cores) {
        CoreArgs args;
        args.core = core;
        if (regime == REGIME_HW_SPLIT) {
            // unit == grid row; HW slice == grid column.
            unit_start = static_cast<std::int64_t>(core.y);
            const std::int64_t units = 1;
            args.unit_start = static_cast<std::uint32_t>(unit_start);
            args.units = static_cast<std::uint32_t>(units);
            args.hw_tile_start = static_cast<std::uint32_t>(static_cast<std::int64_t>(core.x) * per_core_hw_tiles);
            args.is_sender = core.x == 0 ? 1u : 0u;
            args.gather_slot = core.x == 0 ? 0u : static_cast<std::uint32_t>(core.x - 1);
            args.owns_hw_tail = static_cast<std::int64_t>(core.x) == hw_split_factor - 1 ? 1u : 0u;
            args.mcast_rt = mcast->runtime_args(core);
        } else {
            std::int64_t units = 0;
            if (core_group_1.contains(core)) {
                units = units_per_core_g1;
            } else if (core_group_2.contains(core)) {
                units = units_per_core_g2;
            } else {
                units = 0;
            }
            args.unit_start = static_cast<std::uint32_t>(unit_start);
            args.units = static_cast<std::uint32_t>(units);
            args.hw_tile_start = 0;
            args.is_sender = 0;
            args.gather_slot = 0;
            args.owns_hw_tail = 1;
            args.mcast_rt = {0u, 0u, 0u, 0u};
            unit_start += units;
        }
        core_args.push_back(args);
    }

    // ---------------- Assemble the plan ------------------------------------
    Plan plan;
    plan.N = static_cast<std::uint32_t>(N);
    plan.HW = static_cast<std::uint32_t>(HW);
    plan.C = static_cast<std::uint32_t>(C);
    plan.num_groups = static_cast<std::uint32_t>(num_groups);

    plan.Cg = static_cast<std::uint32_t>(geometry.Cg);
    plan.cluster_channels = static_cast<std::uint32_t>(geometry.cluster_channels);
    plan.groups_per_cluster = static_cast<std::uint32_t>(geometry.groups_per_cluster);
    plan.cluster_c_tiles = static_cast<std::uint32_t>(geometry.cluster_c_tiles);
    plan.num_clusters = static_cast<std::uint32_t>(geometry.num_clusters);
    plan.num_mask_tiles = static_cast<std::uint32_t>(spans.num_mask_tiles);
    plan.max_span = static_cast<std::uint32_t>(spans.max_span);

    plan.tensor_hw_tiles = static_cast<std::uint32_t>(tensor_hw_tiles);
    plan.tensor_c_tiles = static_cast<std::uint32_t>(tensor_c_tiles);
    plan.hw_tail = static_cast<std::uint32_t>(py_mod(HW, TILE_HW));
    plan.c_tail = static_cast<std::uint32_t>(py_mod(C, TILE_HW));

    plan.input_is_rm = input_is_rm;
    plan.in_elem = static_cast<std::uint32_t>(in_elem);
    plan.tile_in = static_cast<std::uint32_t>(tile_in);
    plan.tile_out = static_cast<std::uint32_t>(tile_out);
    plan.tile_f32 = static_cast<std::uint32_t>(tile_f32);
    plan.tile_bf16 = static_cast<std::uint32_t>(tile_bf16);
    plan.input_dtype = input_tensor.dtype();
    plan.output_dtype = output_dtype;

    plan.has_gamma = has_gamma;
    plan.has_beta = has_beta;
    plan.affine_is_rm = affine_is_rm;
    plan.affine_elem = static_cast<std::uint32_t>(affine_elem);
    plan.affine_page_bytes = static_cast<std::uint32_t>(affine_page_bytes);
    plan.affine_scratch_bytes = static_cast<std::uint32_t>(affine_scratch_bytes);

    plan.input_depth = static_cast<std::uint32_t>(input_depth);
    plan.output_depth = static_cast<std::uint32_t>(output_depth);
    plan.rm_depth = static_cast<std::uint32_t>(rm_depth);
    plan.block_hw_tiles = static_cast<std::uint32_t>(block_hw_tiles);
    plan.num_hw_blocks = static_cast<std::uint32_t>(num_hw_blocks);
    plan.block_tiles = static_cast<std::uint32_t>(block_tiles);
    plan.input_cb_pages = static_cast<std::uint32_t>(input_cb_pages);
    plan.per_core_hw_tiles = static_cast<std::uint32_t>(per_core_hw_tiles);
    plan.resident = resident;

    plan.regime = static_cast<std::uint32_t>(regime);
    plan.hw_split_factor = static_cast<std::uint32_t>(hw_split_factor);
    plan.units_total = static_cast<std::uint32_t>(units_total);

    plan.inv_n_g_bits = f32_bits(1.0 / static_cast<double>(geometry.Cg * HW));
    plan.eps_bits = f32_bits(eps);

    plan.all_cores = all_cores;
    plan.core_args = std::move(core_args);
    plan.mcast_ct = mcast_ct;
    plan.semaphores = std::move(semaphores);
    plan.compute = compute;

    return plan;
}

// ===========================================================================
// Device operation
// ===========================================================================

metal::ProgramDescriptor MigratedRun1000GroupNormDeviceOperation::ClusterParallelTwoPass::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return build_descriptor(
        operation_attributes.plan, tensor_args.input_tensor, tensor_args.gamma, tensor_args.beta, tensor_return_value);
}

void MigratedRun1000GroupNormDeviceOperation::ClusterParallelTwoPass::override_runtime_arguments(
    metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-derive the whole per-dispatch descriptor and bulk-copy every runtime
    // argument (which rebinds every buffer base address) plus any dynamic CB
    // address into the cached program. Correct by construction on cache hits:
    // nothing is inferred, and no address participates in the cache key.
    const metal::ProgramDescriptor desc = build_descriptor(
        operation_attributes.plan, tensor_args.input_tensor, tensor_args.gamma, tensor_args.beta, tensor_return_value);
    metal::apply_descriptor_runtime_args(program, desc);
}

MigratedRun1000GroupNormDeviceOperation::program_factory_t
MigratedRun1000GroupNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ClusterParallelTwoPass{};
}

void MigratedRun1000GroupNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto* device = tensor_args.input_tensor.device();
    TT_FATAL(device != nullptr, "groupnorm_sc_N_1_HW_C: input tensor must be allocated on a device");
    if (tensor_args.gamma.has_value()) {
        TT_FATAL(
            tensor_args.gamma->device() == device,
            "groupnorm_sc_N_1_HW_C: gamma must live on the same device as the input tensor");
    }
    if (tensor_args.beta.has_value()) {
        TT_FATAL(
            tensor_args.beta->device() == device,
            "groupnorm_sc_N_1_HW_C: beta must live on the same device as the input tensor");
    }
    TT_FATAL(
        !operation_attributes.plan.core_args.empty(),
        "groupnorm_sc_N_1_HW_C: the work split produced no cores for {} units",
        operation_attributes.plan.units_total);
}

MigratedRun1000GroupNormDeviceOperation::spec_return_value_t
MigratedRun1000GroupNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Output: same shape, input dtype, ALWAYS TILE_LAYOUT, interleaved DRAM.
    const Tensor& input_tensor = tensor_args.input_tensor;
    return metal::TensorSpec(
        input_tensor.logical_shape(),
        metal::TensorLayout(input_tensor.dtype(), metal::PageConfig(metal::Layout::TILE), metal::MemoryConfig{}));
}

MigratedRun1000GroupNormDeviceOperation::tensor_return_value_t
MigratedRun1000GroupNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t MigratedRun1000GroupNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Plan& plan = operation_attributes.plan;

    ttsl::hash::hash_t hash = ttsl::hash::hash_objects_with_default_seed(
        static_cast<std::uint32_t>(ttsl::hash::type_hash<MigratedRun1000GroupNormDeviceOperation>));

    hash = ttsl::hash::hash_objects(
        hash, plan.N, plan.HW, plan.C, plan.num_groups, plan.Cg, plan.cluster_channels, plan.groups_per_cluster);
    hash = ttsl::hash::hash_objects(
        hash, plan.cluster_c_tiles, plan.num_clusters, plan.num_mask_tiles, plan.max_span, plan.tensor_hw_tiles);
    hash = ttsl::hash::hash_objects(hash, plan.tensor_c_tiles, plan.hw_tail, plan.c_tail);
    hash = ttsl::hash::hash_objects(
        hash, plan.input_is_rm, plan.in_elem, plan.tile_in, plan.tile_out, plan.tile_f32, plan.tile_bf16);
    hash = ttsl::hash::hash_objects(
        hash,
        static_cast<std::uint32_t>(plan.input_dtype),
        static_cast<std::uint32_t>(plan.output_dtype),
        plan.has_gamma,
        plan.has_beta,
        plan.affine_is_rm,
        plan.affine_elem,
        plan.affine_page_bytes,
        plan.affine_scratch_bytes);
    hash = ttsl::hash::hash_objects(
        hash,
        plan.input_depth,
        plan.output_depth,
        plan.rm_depth,
        plan.block_hw_tiles,
        plan.num_hw_blocks,
        plan.block_tiles,
        plan.input_cb_pages,
        plan.per_core_hw_tiles,
        plan.resident);
    hash = ttsl::hash::hash_objects(
        hash, plan.regime, plan.hw_split_factor, plan.units_total, plan.inv_n_g_bits, plan.eps_bits);
    hash = ttsl::hash::hash_objects(
        hash,
        static_cast<std::uint32_t>(plan.compute.math_fidelity),
        plan.compute.fp32_dest_acc_en,
        plan.compute.math_approx_mode,
        plan.compute.dst_full_sync_en);

    // Core geometry (structural: the kernels are placed on these ranges).
    hash = ttsl::hash::hash_objects(hash, static_cast<std::uint32_t>(plan.all_cores.ranges().size()));
    for (const auto& range : plan.all_cores.ranges()) {
        hash = ttsl::hash::hash_objects(
            hash,
            static_cast<std::uint32_t>(range.start_coord.x),
            static_cast<std::uint32_t>(range.start_coord.y),
            static_cast<std::uint32_t>(range.end_coord.x),
            static_cast<std::uint32_t>(range.end_coord.y));
    }

    // Per-core scalar runtime args (never any buffer address).
    hash = ttsl::hash::hash_objects(hash, static_cast<std::uint32_t>(plan.core_args.size()));
    for (const CoreArgs& args : plan.core_args) {
        hash = ttsl::hash::hash_objects(
            hash,
            static_cast<std::uint32_t>(args.core.x),
            static_cast<std::uint32_t>(args.core.y),
            args.unit_start,
            args.units,
            args.hw_tile_start,
            args.is_sender,
            args.gather_slot,
            args.owns_hw_tail);
        for (const std::uint32_t value : args.mcast_rt) {
            hash = ttsl::hash::hash_objects(hash, value);
        }
    }

    // Mcast compile-time block and semaphores.
    for (const std::uint32_t value : plan.mcast_ct) {
        hash = ttsl::hash::hash_objects(hash, value);
    }
    hash = ttsl::hash::hash_objects(hash, static_cast<std::uint32_t>(plan.semaphores.size()));
    for (const auto& semaphore : plan.semaphores) {
        hash = ttsl::hash::hash_objects(
            hash,
            semaphore.id,
            semaphore.initial_value,
            static_cast<std::uint32_t>(semaphore.core_type),
            static_cast<std::uint32_t>(semaphore.core_ranges.num_cores()));
    }

    // TensorAccessor compile-time args: structural page/shard descriptions of
    // every operand. These are baked into the kernels, so they belong in the
    // key; none of them is a buffer base address.
    const auto fold_accessor_args = [&hash](const std::vector<std::uint32_t>& args) {
        hash = ttsl::hash::hash_objects(hash, static_cast<std::uint32_t>(args.size()));
        for (const std::uint32_t value : args) {
            hash = ttsl::hash::hash_objects(hash, value);
        }
    };
    fold_accessor_args(accessor_compile_time_args(tensor_args.input_tensor));
    fold_accessor_args(accessor_compile_time_args(plan.has_gamma ? tensor_args.gamma : std::optional<Tensor>{}));
    fold_accessor_args(accessor_compile_time_args(plan.has_beta ? tensor_args.beta : std::optional<Tensor>{}));

    return hash;
}

// ===========================================================================
// Public entry point
// ===========================================================================

Tensor invoke(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    double eps,
    const ComputeConfigFields& compute) {
    validate_arguments(input_tensor, num_groups, gamma, beta);
    validate_support(input_tensor, num_groups, gamma, beta);

    // The output carries the input dtype and is ALWAYS TILE_LAYOUT; the device
    // operation allocates it from `compute_output_specs()`.
    Plan plan = build_plan(input_tensor, input_tensor.dtype(), num_groups, gamma, beta, eps, compute);

    using OperationType = MigratedRun1000GroupNormDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{std::move(plan), eps},
        OperationType::tensor_args_t{input_tensor, gamma, beta});
}

}  // namespace ttnn::migration::generated_migrated_run1000_groupnorm
