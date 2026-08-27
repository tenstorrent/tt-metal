// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::compact {

// Online model ABI. It intentionally contains only workload facts and
// program-config fields. Compute-kernel knobs and every other caller-owned
// execution field are unrepresentable here.
enum class ProgramConfigFeature : std::uint8_t {
    LogicalM,
    LogicalK,
    LogicalN,
    PaddedM,
    PaddedK,
    PaddedN,
    GridX,
    GridY,
    In0BlockW,
    OutSubblockH,
    OutSubblockW,
    PerCoreM,
    PerCoreN,
    Family,
    OutBlockH,
    OutBlockW,
    NumGlobalCbReceivers,
    FuseBatch,
    McastIn0,
    TransposeMcast,
    FusedActivationPresent,
    GatherIn0,
    HopCoresPresent,
    UntilizeOut,
    StreamIn1,
    Count,
};

inline constexpr std::size_t kProgramConfigFeatureCount = static_cast<std::size_t>(ProgramConfigFeature::Count);

// A candidate is a program config, not a benchmark row. Offline production
// must collapse all rows that differ only in CKC or measurement metadata into
// one candidate before emitting this POD.
struct ProgramConfigDescriptor {
    ProgramFamily family{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint32_t in0_block_w{};
    std::uint32_t out_subblock_h{};
    std::uint32_t out_subblock_w{};
    std::uint32_t per_core_m{};
    std::uint32_t per_core_n{};
    std::uint32_t out_block_h{};
    std::uint32_t out_block_w{};
    std::uint32_t num_global_cb_receivers{};
    bool allowed_worker_cores_present{};
    // Family-specific program-config fields. They are explicit even when a
    // native constructor currently supplies the same default.
    bool fuse_batch{};
    bool mcast_in0{};
    bool transpose_mcast{};
    bool fused_activation_present{};
    bool gather_in0{};
    bool hop_cores_present{};
    bool untilize_out{};
    bool stream_in1{};

    auto operator<=>(const ProgramConfigDescriptor&) const = default;
};

struct ProgramConfigCandidate {
    ProgramConfigDescriptor program_config{};
    RegistryEntryId candidate_id{};

    auto operator<=>(const ProgramConfigCandidate&) const = default;
};

// Exact runtime entries are deliberately distinct from legacy ReplayDescriptor
// rows.  The type cannot represent CKC, untilize, output memory, or any other
// call state: an exact hit owns only the selected program_config.
struct ProgramConfigExactEntry {
    RegistryEntryId entry_id{};
    KeyDescriptor key{};
    ProgramConfigDescriptor program_config{};

    auto operator<=>(const ProgramConfigExactEntry&) const = default;
};

static_assert(std::is_trivially_copyable_v<ProgramConfigExactEntry>);
static_assert(std::is_standard_layout_v<ProgramConfigExactEntry>);

inline constexpr const ProgramConfigExactEntry* lookup_program_config_exact(
    const KeyDescriptor& key, const std::span<const ProgramConfigExactEntry> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(),
        entries.end(),
        key,
        [](const ProgramConfigExactEntry& entry, const KeyDescriptor& requested_key) {
            return entry.key < requested_key;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}

// Direct-bank measurements describe the one-chip architecture and legal
// worker grid, not a particular board serial/topology. Normalize only those
// physical attestation fields; device count and mesh shape remain semantic
// deployment axes and must still match exactly.
constexpr KeyDescriptor direct_bank_key(KeyDescriptor key) noexcept {
    key.board_capability_class = 0;
    key.topology_sha256 = {};
    return key;
}

inline constexpr const ProgramConfigExactEntry* lookup_program_config_exact_direct_bank(
    const KeyDescriptor& key, const std::span<const ProgramConfigExactEntry> entries) noexcept {
    const auto normalized = direct_bank_key(key);
    const auto candidate = std::lower_bound(
        entries.begin(),
        entries.end(),
        normalized,
        [](const ProgramConfigExactEntry& entry, const KeyDescriptor& requested_key) {
            return direct_bank_key(entry.key) < requested_key;
        });
    return candidate != entries.end() && direct_bank_key(candidate->key) == normalized ? &*candidate : nullptr;
}

struct GbdtNode {
    // A leaf has feature == Count and stores its signed score in leaf_value.
    ProgramConfigFeature feature = ProgramConfigFeature::Count;
    std::uint64_t threshold{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::int64_t leaf_value{};
};

struct GbdtTree {
    std::uint32_t node_offset{};
    std::uint32_t node_count{};
};

enum class GbdtScoreOrientation : std::uint8_t {
    // Python's pairwise model is higher-is-better. Its exporter must negate
    // every leaf/base margin exactly once before emitting this runtime form.
    LowerIsBetterNegatedPairwiseMargin = 1,
};

enum class ShapeScaleClass : std::uint8_t { Decode, SmallBatch, Prefill, LongPrefill };
enum class ShapeGeometryClass : std::uint8_t { ContractWide, SquareKn, OutputWide };

struct ProgramConfigModelSupport {
    std::uint32_t architecture{};
    std::uint32_t board_capability_class{};
    std::uint16_t device_count{};
    std::uint16_t mesh_rows{};
    std::uint16_t mesh_cols{};
    Sha256 topology_sha256{};
    Domain domain{};
    TensorDescriptor input_a{};
    TensorDescriptor input_b{};
    TensorDescriptor output{};
    ShapeScaleClass shape_scale{};
    ShapeGeometryClass shape_geometry{};
    std::uint64_t minimum_m{};
    std::uint64_t maximum_m{};
    std::uint64_t minimum_k{};
    std::uint64_t maximum_k{};
    std::uint64_t minimum_n{};
    std::uint64_t maximum_n{};
};

struct TrainingShapeLandmark {
    std::uint64_t logical_m{};
    std::uint64_t logical_k{};
    std::uint64_t logical_n{};

    auto operator<=>(const TrainingShapeLandmark&) const = default;
};

inline constexpr std::uint64_t MAX_NORMALIZED_SHAPE_DISTANCE_PPM = 250'000;

struct ProgramConfigGbdtModel {
    std::uint16_t schema_version{};
    bool enabled{};
    GbdtScoreOrientation score_orientation{};
    Sha256 feature_schema_sha256{};
    Sha256 model_sha256{};
    Sha256 training_table_sha256{};
    Sha256 safety_evidence_sha256{};
    Sha256 candidate_policy_sha256{};
    // Sealed full-source/context/candidate-policy/native-sublane lineage.
    // This prevents a projected table with an intentionally empty unsafe-row
    // ledger from becoming detached from the source safety evidence.
    Sha256 lineage_sha256{};
    Sha256 evaluation_model_payload_sha256{};
    Sha256 quality_evaluation_sha256{};
    Sha256 unseen_abstention_policy_sha256{};
    Sha256 support_sha256{};
    // Must equal TableMetadata.online_model_bundle_binding_sha256. The emitter
    // reconstructs this non-self-referential binding over exact-entry and
    // model inventory digests.
    Sha256 bundle_binding_sha256{};
    ProgramConfigModelSupport support{};
    std::int64_t base_score{};
    // Scores are fixed-point integers. A positive multiplier preserves order;
    // keeping it in the artifact makes quantization reviewable.
    std::uint32_t score_scale{};
    // Fixed-point distance required between the best and runner-up scores.
    // A single legal candidate or a smaller margin always abstains.
    std::uint64_t minimum_score_margin{};
    // Calibrated nearest-development-shape L-infinity radius. Axis deltas
    // are normalized by this model support's full M/K/N span in integer ppm.
    std::uint64_t maximum_normalized_shape_distance_ppm{};
    std::span<const TrainingShapeLandmark> training_shapes;
    std::span<const ProgramConfigCandidate> candidates;
    std::span<const GbdtTree> trees;
    std::span<const GbdtNode> nodes;
};

enum class ProgramConfigLookupSource : std::uint8_t { None, Exact, Gbdt };

struct ProgramConfigLookupResult {
    ProgramConfigLookupSource source = ProgramConfigLookupSource::None;
    std::optional<ProgramConfigDescriptor> program_config = std::nullopt;
    const RegistryEntryId* identity = nullptr;
};

constexpr ProgramConfigDescriptor exact_program_config(const ReplayDescriptor& replay) noexcept {
    const auto& config = replay.program_config;
    return ProgramConfigDescriptor{
        .family = replay.family,
        .compute_grid_x = config.compute_grid_x,
        .compute_grid_y = config.compute_grid_y,
        .in0_block_w = config.in0_block_w,
        .out_subblock_h = config.out_subblock_h,
        .out_subblock_w = config.out_subblock_w,
        .per_core_m = config.per_core_m,
        .per_core_n = config.per_core_n,
        .allowed_worker_cores_present = config.allowed_worker_cores_present,
        .fuse_batch = false,
        .mcast_in0 = false,
        .transpose_mcast = false,
    };
}

constexpr bool nonzero_sha256(const Sha256& value) noexcept {
    for (const auto byte : value) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

constexpr ShapeScaleClass shape_scale_class(const std::uint64_t m) noexcept {
    if (m <= 64) {
        return ShapeScaleClass::Decode;
    }
    if (m <= 256) {
        return ShapeScaleClass::SmallBatch;
    }
    if (m <= 1024) {
        return ShapeScaleClass::Prefill;
    }
    return ShapeScaleClass::LongPrefill;
}

constexpr ShapeGeometryClass shape_geometry_class(const std::uint64_t k, const std::uint64_t n) noexcept {
    // Exact integer form of geometry_v1's inclusive 0.8 <= K/N <= 1.25.
    if (k <= std::numeric_limits<std::uint64_t>::max() / 5 && n <= std::numeric_limits<std::uint64_t>::max() / 5 &&
        5 * k >= 4 * n && 4 * k <= 5 * n) {
        return ShapeGeometryClass::SquareKn;
    }
    return k > n ? ShapeGeometryClass::ContractWide : ShapeGeometryClass::OutputWide;
}

constexpr bool is_canonical_tile_padding(
    const std::uint64_t logical, const std::uint64_t padded, const std::uint64_t tile_extent) noexcept {
    if (logical == 0 || tile_extent == 0) {
        return false;
    }
    const auto whole_tiles = logical / tile_extent;
    const auto has_partial_tile = logical % tile_extent != 0;
    if (has_partial_tile && whole_tiles == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    const auto tile_count = whole_tiles + has_partial_tile;
    return tile_count <= std::numeric_limits<std::uint64_t>::max() / tile_extent && padded == tile_count * tile_extent;
}

constexpr std::uint64_t calibrated_axis_delta(const std::uint64_t span, const std::uint64_t threshold_ppm) noexcept {
    constexpr std::uint64_t million = 1'000'000;
    // floor(span * threshold / 1e6), decomposed to avoid overflow.
    return (span / million) * threshold_ppm + ((span % million) * threshold_ppm) / million;
}

constexpr bool model_shape_is_near_training_data(
    const KeyDescriptor& key, const ProgramConfigGbdtModel& model) noexcept {
    if (model.training_shapes.empty() || model.maximum_normalized_shape_distance_ppm == 0 ||
        model.maximum_normalized_shape_distance_ppm > MAX_NORMALIZED_SHAPE_DISTANCE_PPM) {
        return false;
    }
    const auto& support = model.support;
    const auto m_span = support.maximum_m - support.minimum_m;
    const auto k_span = support.maximum_k - support.minimum_k;
    const auto n_span = support.maximum_n - support.minimum_n;
    const auto m_delta_limit = calibrated_axis_delta(m_span, model.maximum_normalized_shape_distance_ppm);
    const auto k_delta_limit = calibrated_axis_delta(k_span, model.maximum_normalized_shape_distance_ppm);
    const auto n_delta_limit = calibrated_axis_delta(n_span, model.maximum_normalized_shape_distance_ppm);
    for (const auto& landmark : model.training_shapes) {
        const auto m_delta = key.logical_m >= landmark.logical_m ? key.logical_m - landmark.logical_m
                                                                 : landmark.logical_m - key.logical_m;
        const auto k_delta = key.logical_k >= landmark.logical_k ? key.logical_k - landmark.logical_k
                                                                 : landmark.logical_k - key.logical_k;
        const auto n_delta = key.logical_n >= landmark.logical_n ? key.logical_n - landmark.logical_n
                                                                 : landmark.logical_n - key.logical_n;
        const bool m_near = m_span == 0 ? m_delta == 0 : m_delta <= m_delta_limit;
        const bool k_near = k_span == 0 ? k_delta == 0 : k_delta <= k_delta_limit;
        const bool n_near = n_span == 0 ? n_delta == 0 : n_delta <= n_delta_limit;
        if (m_near && k_near && n_near) {
            return true;
        }
    }
    return false;
}

// Training landmarks are an exclusion set for online inference. Their measured
// winners belong in the typed exact table; if an exact entry was not promoted,
// the runtime must fall through rather than substitute a model prediction for
// a known training shape.
constexpr bool model_shape_is_training_landmark(
    const KeyDescriptor& key, const ProgramConfigGbdtModel& model) noexcept {
    const auto requested = TrainingShapeLandmark{
        .logical_m = key.logical_m,
        .logical_k = key.logical_k,
        .logical_n = key.logical_n,
    };
    return std::binary_search(model.training_shapes.begin(), model.training_shapes.end(), requested);
}

constexpr bool model_supports(
    const KeyDescriptor& key,
    const ProgramConfigGbdtModel& model,
    const Sha256& expected_bundle_binding_sha256,
    const bool direct_bank_scope = false) noexcept {
    const auto& support = model.support;
    return model.enabled && model.schema_version == 1 && nonzero_sha256(model.feature_schema_sha256) &&
           nonzero_sha256(model.model_sha256) && nonzero_sha256(model.training_table_sha256) &&
           nonzero_sha256(model.safety_evidence_sha256) && nonzero_sha256(model.candidate_policy_sha256) &&
           nonzero_sha256(model.lineage_sha256) && nonzero_sha256(model.evaluation_model_payload_sha256) &&
           nonzero_sha256(model.quality_evaluation_sha256) && nonzero_sha256(model.unseen_abstention_policy_sha256) &&
           model.minimum_score_margin != 0 && nonzero_sha256(model.support_sha256) &&
           nonzero_sha256(model.bundle_binding_sha256) &&
           model.bundle_binding_sha256 == expected_bundle_binding_sha256 && key.architecture == support.architecture &&
           (direct_bank_scope || key.board_capability_class == support.board_capability_class) &&
           key.device_count == support.device_count && key.mesh_rows == support.mesh_rows &&
           key.mesh_cols == support.mesh_cols &&
           (direct_bank_scope || key.topology_sha256 == support.topology_sha256) && key.domain == support.domain &&
           key.input_a == support.input_a && key.input_b == support.input_b && key.output == support.output &&
           // The raw-feature trainer synthesizes exactly ceil-to-tile padded
           // dimensions. Reject larger custom padding rather than scoring an
           // unmeasured feature combination.
           is_canonical_tile_padding(key.logical_m, key.padded_m, key.input_a.tile_height) &&
           is_canonical_tile_padding(key.logical_m, key.padded_m, key.output.tile_height) &&
           is_canonical_tile_padding(key.logical_k, key.padded_k, key.input_a.tile_width) &&
           is_canonical_tile_padding(key.logical_k, key.padded_k, key.input_b.tile_height) &&
           is_canonical_tile_padding(key.logical_n, key.padded_n, key.input_b.tile_width) &&
           is_canonical_tile_padding(key.logical_n, key.padded_n, key.output.tile_width) &&
           shape_scale_class(key.logical_m) == support.shape_scale &&
           shape_geometry_class(key.logical_k, key.logical_n) == support.shape_geometry &&
           key.logical_m >= support.minimum_m && key.logical_m <= support.maximum_m &&
           key.logical_k >= support.minimum_k && key.logical_k <= support.maximum_k &&
           key.logical_n >= support.minimum_n && key.logical_n <= support.maximum_n &&
           !model_shape_is_training_landmark(key, model) && model_shape_is_near_training_data(key, model);
}

constexpr std::uint64_t nonnegative_score_distance(const std::int64_t lower, const std::int64_t upper) noexcept {
    if (upper < lower) {
        return 0;
    }
    if (lower >= 0 || upper < 0) {
        return static_cast<std::uint64_t>(upper - lower);
    }
    // Avoid signed overflow for a range crossing zero. Negating INT64_MIN is
    // also undefined, so form its magnitude as -(x + 1) + 1.
    return static_cast<std::uint64_t>(-(lower + 1)) + 1 + static_cast<std::uint64_t>(upper);
}

constexpr std::uint64_t feature_value(
    const KeyDescriptor& key, const ProgramConfigCandidate& candidate, const ProgramConfigFeature feature) noexcept {
    const auto& config = candidate.program_config;
    switch (feature) {
        case ProgramConfigFeature::LogicalM: return key.logical_m;
        case ProgramConfigFeature::LogicalK: return key.logical_k;
        case ProgramConfigFeature::LogicalN: return key.logical_n;
        case ProgramConfigFeature::PaddedM: return key.padded_m;
        case ProgramConfigFeature::PaddedK: return key.padded_k;
        case ProgramConfigFeature::PaddedN: return key.padded_n;
        case ProgramConfigFeature::GridX: return config.compute_grid_x;
        case ProgramConfigFeature::GridY: return config.compute_grid_y;
        case ProgramConfigFeature::In0BlockW: return config.in0_block_w;
        case ProgramConfigFeature::OutSubblockH: return config.out_subblock_h;
        case ProgramConfigFeature::OutSubblockW: return config.out_subblock_w;
        case ProgramConfigFeature::PerCoreM: return config.per_core_m;
        case ProgramConfigFeature::PerCoreN: return config.per_core_n;
        case ProgramConfigFeature::Family: return static_cast<std::uint64_t>(config.family);
        case ProgramConfigFeature::OutBlockH: return config.out_block_h;
        case ProgramConfigFeature::OutBlockW: return config.out_block_w;
        case ProgramConfigFeature::NumGlobalCbReceivers: return config.num_global_cb_receivers;
        case ProgramConfigFeature::FuseBatch: return config.fuse_batch;
        case ProgramConfigFeature::McastIn0: return config.mcast_in0;
        case ProgramConfigFeature::TransposeMcast: return config.transpose_mcast;
        case ProgramConfigFeature::FusedActivationPresent: return config.fused_activation_present;
        case ProgramConfigFeature::GatherIn0: return config.gather_in0;
        case ProgramConfigFeature::HopCoresPresent: return config.hop_cores_present;
        case ProgramConfigFeature::UntilizeOut: return config.untilize_out;
        case ProgramConfigFeature::StreamIn1: return config.stream_in1;
        case ProgramConfigFeature::Count: return 0;
    }
    return 0;
}

// Conservative runtime legality for the first native family. This mirrors the
// compact exact-entry validator; an emitted GBDT candidate cannot bypass it.
constexpr bool legal_program_config_candidate(
    const KeyDescriptor& key, const ProgramConfigCandidate& candidate) noexcept {
    const auto& program = candidate.program_config;
    if (program.fused_activation_present || program.gather_in0 || program.hop_cores_present || program.untilize_out ||
        program.stream_in1) {
        return false;
    }
    if (program.compute_grid_x == 0 || program.compute_grid_y == 0 || program.compute_grid_x > key.compute_grid_x ||
        program.compute_grid_y > key.compute_grid_y || program.in0_block_w == 0 || program.out_subblock_h == 0 ||
        program.out_subblock_w == 0 || program.per_core_m == 0 || program.per_core_n == 0 ||
        program.allowed_worker_cores_present || key.input_a.tile_height == 0 || key.input_a.tile_width == 0 ||
        key.input_b.tile_height == 0 || key.input_b.tile_width == 0 || key.padded_m % key.input_a.tile_height != 0 ||
        key.padded_k % key.input_a.tile_width != 0 || key.padded_k % key.input_b.tile_height != 0 ||
        key.padded_n % key.input_b.tile_width != 0) {
        return false;
    }
    const auto m_tiles = key.padded_m / key.input_a.tile_height;
    const auto a_k_tiles = key.padded_k / key.input_a.tile_width;
    const auto b_k_tiles = key.padded_k / key.input_b.tile_height;
    const auto n_tiles = key.padded_n / key.input_b.tile_width;
    if (a_k_tiles != b_k_tiles || a_k_tiles % program.in0_block_w != 0 ||
        program.per_core_m % program.out_subblock_h != 0 || program.per_core_n % program.out_subblock_w != 0 ||
        program.out_subblock_h * program.out_subblock_w > 4) {
        return false;
    }
    switch (program.family) {
        case ProgramFamily::MultiCoreReuse:
            return program.out_block_h == 0 && program.out_block_w == 0 && program.num_global_cb_receivers == 0 &&
                   !program.fuse_batch && !program.mcast_in0 && !program.transpose_mcast &&
                   m_tiles % program.per_core_m == 0 && n_tiles == program.per_core_n;
        case ProgramFamily::MultiCast1D: {
            if (!program.fuse_batch || program.transpose_mcast || program.per_core_n > 64 ||
                program.out_block_h != program.per_core_m || program.out_block_w != program.per_core_n ||
                program.num_global_cb_receivers != 1) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            const auto core_count = static_cast<std::uint64_t>(program.compute_grid_x) * program.compute_grid_y;
            const bool complete_axis =
                program.mcast_in0 ? program.per_core_m == m_tiles : program.per_core_n == n_tiles;
            return complete_axis && n_blocks != 0 && m_blocks <= core_count / n_blocks;
        }
        case ProgramFamily::MultiCast2D: {
            if (!program.fuse_batch || program.mcast_in0 || program.out_block_h != program.per_core_m ||
                program.out_block_w != program.per_core_n || program.num_global_cb_receivers != 0) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            // Native validation swaps output-block axes when transpose_mcast is
            // selected (matmul_device_operation.cpp). Mirror that exact grid
            // extent check so valid banked transpose candidates remain usable.
            return program.transpose_mcast ? m_blocks <= program.compute_grid_x && n_blocks <= program.compute_grid_y
                                           : m_blocks <= program.compute_grid_y && n_blocks <= program.compute_grid_x;
        }
    }
    return false;
}

constexpr std::optional<std::int64_t> score_program_config_candidate(
    const KeyDescriptor& key,
    const ProgramConfigCandidate& candidate,
    const ProgramConfigGbdtModel& model,
    const Sha256& expected_bundle_binding_sha256,
    const bool direct_bank_scope = false) noexcept {
    if (!model_supports(key, model, expected_bundle_binding_sha256, direct_bank_scope) ||
        model.score_orientation != GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin || model.score_scale == 0) {
        return std::nullopt;
    }
    std::int64_t score = model.base_score;
    for (const auto& tree : model.trees) {
        if (tree.node_count == 0 || tree.node_offset > model.nodes.size() ||
            tree.node_count > model.nodes.size() - tree.node_offset) {
            return std::nullopt;
        }
        std::uint32_t index = 0;
        for (std::uint32_t depth = 0; depth < tree.node_count; ++depth) {
            const auto& node = model.nodes[tree.node_offset + index];
            if (node.feature == ProgramConfigFeature::Count) {
                if ((node.leaf_value > 0 && score > std::numeric_limits<std::int64_t>::max() - node.leaf_value) ||
                    (node.leaf_value < 0 && score < std::numeric_limits<std::int64_t>::min() - node.leaf_value)) {
                    return std::nullopt;
                }
                score += node.leaf_value;
                break;
            }
            index = feature_value(key, candidate, node.feature) <= node.threshold ? node.left : node.right;
            if (index >= tree.node_count) {
                return std::nullopt;
            }
            if (depth + 1 == tree.node_count) {
                return std::nullopt;
            }
        }
    }
    return score;
}

// The runtime contract in one function: exact canonical key first, then GBDT
// for a supported non-landmark shape over emitted legal program configs, then
// an empty result so TTNN's existing heuristic remains authoritative. Lower
// model scores are better; ties use the stable candidate ID.
inline ProgramConfigLookupResult lookup_program_config(
    const KeyDescriptor& key,
    const std::span<const ProgramConfigExactEntry> exact_entries,
    const ProgramConfigGbdtModel& model,
    const Sha256& expected_bundle_binding_sha256,
    const bool direct_bank_scope = false) noexcept {
    const auto* exact = direct_bank_scope ? lookup_program_config_exact_direct_bank(key, exact_entries)
                                          : lookup_program_config_exact(key, exact_entries);
    if (exact != nullptr) {
        if (!legal_program_config_candidate(
                key,
                ProgramConfigCandidate{.program_config = exact->program_config, .candidate_id = exact->entry_id})) {
            return {};
        }
        return {
            .source = ProgramConfigLookupSource::Exact,
            .program_config = exact->program_config,
            .identity = &exact->entry_id,
        };
    }

    const ProgramConfigCandidate* best = nullptr;
    const ProgramConfigCandidate* runner_up = nullptr;
    const ProgramConfigCandidate* previous = nullptr;
    std::int64_t best_score = std::numeric_limits<std::int64_t>::max();
    std::int64_t runner_up_score = std::numeric_limits<std::int64_t>::max();
    for (const auto& candidate : model.candidates) {
        // Generated candidates are canonically ordered and unique by the
        // complete program config. A duplicate would commonly mean offline
        // rows that differed only by hidden CKC knobs leaked into the online
        // universe; fail closed instead of selecting between them.
        if (previous != nullptr && !(previous->program_config < candidate.program_config)) {
            return {};
        }
        previous = &candidate;
        if (!legal_program_config_candidate(key, candidate)) {
            continue;
        }
        const auto score =
            score_program_config_candidate(key, candidate, model, expected_bundle_binding_sha256, direct_bank_scope);
        if (!score.has_value()) {
            continue;
        }
        if (best == nullptr || *score < best_score ||
            (*score == best_score && candidate.candidate_id < best->candidate_id)) {
            runner_up = best;
            runner_up_score = best_score;
            best = &candidate;
            best_score = *score;
        } else if (
            runner_up == nullptr || *score < runner_up_score ||
            (*score == runner_up_score && candidate.candidate_id < runner_up->candidate_id)) {
            runner_up = &candidate;
            runner_up_score = *score;
        }
    }
    if (best == nullptr || runner_up == nullptr ||
        nonnegative_score_distance(best_score, runner_up_score) < model.minimum_score_margin) {
        return {};
    }
    return {
        .source = ProgramConfigLookupSource::Gbdt,
        .program_config = best->program_config,
        .identity = &best->candidate_id,
    };
}

static_assert(std::is_trivially_copyable_v<ProgramConfigCandidate>);
static_assert(std::is_trivially_copyable_v<GbdtNode>);

}  // namespace ttnn::operations::matmul::registry::compact
