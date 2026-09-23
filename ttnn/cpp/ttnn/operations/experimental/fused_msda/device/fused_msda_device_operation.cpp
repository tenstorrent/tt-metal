// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_msda_device_operation.hpp"

#include <string_view>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::fused_msda {

namespace {

using tt::tt_metal::TensorMemoryLayout;

// The reader holds per-level (H, W, level_start) in a fixed-size array so the
// staging loops can be unrolled against a compile-time bound. Bumping this is a
// one-line change in the kernels, but it has to stay in sync, so it is checked
// here rather than discovered as a stack smash on device.
constexpr uint32_t MAX_LEVELS = 8;

// Largest H_l or W_l the op supports, set by bf16's 8 significant bits: the
// floored bilinear corner crosses from the SFPU geometry to the reader as bf16
// and has to decode to the integer that was floored, exactly. See
// device/kernels/compute/msda_geometry.hpp.
constexpr uint32_t MAX_SPATIAL_EXTENT = 256;

void check_common_tensor(const Tensor& t, std::string_view name, const Tensor& ref) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "fused_msda: {} must be on device", name);
    TT_FATAL(t.device() == ref.device(), "fused_msda: {} must be on the same device as value", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "fused_msda: {} must be BFLOAT16, got {}", name, t.dtype());
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "fused_msda: {} must be ROW_MAJOR, got {}", name, t.layout());
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "fused_msda: {} memory_layout must be INTERLEAVED (sharded inputs are not supported)",
        name);
}

}  // namespace

MSDAShapes derive_shapes(
    const FusedMSDAOperation::operation_attributes_t& attrs, const FusedMSDAOperation::tensor_args_t& args) {
    MSDAShapes s;

    const auto& value = args.value;
    const auto& attn = args.attention_weights;

    TT_FATAL(value.storage_type() == StorageType::DEVICE, "fused_msda: value must be on device");
    check_common_tensor(value, "value", value);
    check_common_tensor(attn, "attention_weights", value);

    // ---- spatial_shapes ----
    const auto& hw = attrs.spatial_shapes_hw;
    TT_FATAL(hw.size() % 2 == 0, "fused_msda: internal error, spatial_shapes_hw must have even length");
    s.num_levels = static_cast<uint32_t>(hw.size() / 2);
    TT_FATAL(s.num_levels >= 1, "fused_msda: spatial_shapes must contain at least one level");
    TT_FATAL(
        s.num_levels <= MAX_LEVELS,
        "fused_msda: at most {} feature levels are supported, got {}",
        MAX_LEVELS,
        s.num_levels);

    uint32_t total_keys = 0;
    for (uint32_t l = 0; l < s.num_levels; ++l) {
        const uint32_t h = hw[2 * l];
        const uint32_t w = hw[2 * l + 1];
        TT_FATAL(h > 0 && w > 0, "fused_msda: spatial_shapes[{}] must be positive, got ({}, {})", l, h, w);
        // The floored bilinear corner crosses from the compute kernel to the
        // reader as bf16, which carries 8 significant bits and so represents
        // every integer up to MAX_SPATIAL_EXTENT exactly and nothing beyond it
        // reliably. Past that an in-bounds corner index would round to a
        // different, still in-bounds pixel — a silently wrong sample rather
        // than a failure. Refuse instead.
        TT_FATAL(
            h <= MAX_SPATIAL_EXTENT && w <= MAX_SPATIAL_EXTENT,
            "fused_msda: spatial_shapes[{}] = ({}, {}) exceeds the {}-pixel per-axis limit. The sampling geometry "
            "runs on the SFPU and hands the reader the floored corner as bf16, which is exact only for integers up "
            "to {}",
            l,
            h,
            w,
            MAX_SPATIAL_EXTENT,
            MAX_SPATIAL_EXTENT);
        total_keys += h * w;
    }

    // ---- value: (B, S, H, D) or packed (B, S, H*D) ----
    // Packed is the layout a Linear over embed_dims emits; D is recovered from
    // attention_weights' head dim after that tensor is parsed below.
    const auto& vs = value.logical_shape();
    TT_FATAL(
        vs.rank() == 4 || vs.rank() == 3,
        "fused_msda: value rank must be 4 (B, S, H, D) or 3 (B, S, H*D), got shape {}",
        vs);
    s.batch = vs[0];
    s.num_keys = vs[1];
    s.value_packed = vs.rank() == 3;
    TT_FATAL(s.batch > 0, "fused_msda: batch must be > 0");
    TT_FATAL(
        s.num_keys == total_keys,
        "fused_msda: value's S dim ({}) must equal sum of H_l * W_l over spatial_shapes ({})",
        s.num_keys,
        total_keys);
    if (!s.value_packed) {
        s.num_heads = vs[2];
        s.head_dim = vs[3];
        TT_FATAL(s.num_heads > 0, "fused_msda: num_heads must be > 0");
    }

    // ---- attention_weights: (B, Q, H, L, P) or (B, Q, H, L*P) ----
    const auto& as = attn.logical_shape();
    TT_FATAL(
        as.rank() == 5 || as.rank() == 4,
        "fused_msda: attention_weights rank must be 5 (B, Q, H, L, P) or 4 (B, Q, H, L*P), got shape {}",
        as);
    TT_FATAL(
        static_cast<uint32_t>(as[0]) == s.batch,
        "fused_msda: attention_weights batch ({}) != value batch ({})",
        as[0],
        s.batch);
    if (s.value_packed) {
        s.num_heads = as[2];
        TT_FATAL(s.num_heads > 0, "fused_msda: num_heads must be > 0");
        TT_FATAL(
            vs[2] % s.num_heads == 0,
            "fused_msda: packed value last dim ({}) must be divisible by attention_weights heads ({})",
            vs[2],
            s.num_heads);
        s.head_dim = vs[2] / s.num_heads;
    } else {
        TT_FATAL(
            static_cast<uint32_t>(as[2]) == s.num_heads,
            "fused_msda: attention_weights heads ({}) != value heads ({})",
            as[2],
            s.num_heads);
    }
    // A D-wide stick is scattered across ceil(D/32) tiles in 16-value face
    // halves, and the writer places head h at byte offset h * D * 2 inside the
    // output page — a multiple of 16 keeps both exact and the offset 32-B aligned.
    TT_FATAL(
        s.head_dim > 0 && s.head_dim % 16 == 0,
        "fused_msda: head_dim D must be a positive multiple of 16, got {}",
        s.head_dim);
    s.num_queries = as[1];
    TT_FATAL(s.num_queries > 0, "fused_msda: num_queries Q must be > 0");
    if (as.rank() == 5) {
        s.weights_packed = false;
        TT_FATAL(
            static_cast<uint32_t>(as[3]) == s.num_levels,
            "fused_msda: attention_weights levels ({}) != spatial_shapes levels ({})",
            as[3],
            s.num_levels);
        s.num_points = as[4];
    } else {
        s.weights_packed = true;
        const uint32_t lp = as[3];
        TT_FATAL(
            lp % s.num_levels == 0,
            "fused_msda: packed attention_weights last dim ({}) must be divisible by num_levels ({})",
            lp,
            s.num_levels);
        s.num_points = lp / s.num_levels;
    }
    TT_FATAL(s.num_points > 0, "fused_msda: num_points P must be > 0");

    // ---- sampling_locations (V1) / sampling_offsets (V2) ----
    const std::string_view loc_name = attrs.from_offsets ? "sampling_offsets" : "sampling_locations";
    const Tensor& loc = attrs.from_offsets ? *args.sampling_offsets : *args.sampling_locations;
    check_common_tensor(loc, loc_name, value);

    const auto& ls = loc.logical_shape();
    TT_FATAL(
        ls.rank() == 6 || ls.rank() == 4,
        "fused_msda: {} rank must be 6 (B, Q, H, L, P, 2) or 4 (B, Q, H, L*P*2), got shape {}",
        loc_name,
        ls);
    TT_FATAL(
        static_cast<uint32_t>(ls[0]) == s.batch && static_cast<uint32_t>(ls[1]) == s.num_queries &&
            static_cast<uint32_t>(ls[2]) == s.num_heads,
        "fused_msda: {} leading dims {} do not match (B, Q, H) = ({}, {}, {})",
        loc_name,
        ls,
        s.batch,
        s.num_queries,
        s.num_heads);
    if (ls.rank() == 6) {
        s.locations_packed = false;
        TT_FATAL(
            static_cast<uint32_t>(ls[3]) == s.num_levels && static_cast<uint32_t>(ls[4]) == s.num_points,
            "fused_msda: {} (L, P) = ({}, {}) does not match attention_weights (L, P) = ({}, {})",
            loc_name,
            ls[3],
            ls[4],
            s.num_levels,
            s.num_points);
        TT_FATAL(ls[5] == 2, "fused_msda: {} last dim must be 2 (x, y), got {}", loc_name, ls[5]);
    } else {
        s.locations_packed = true;
        const uint32_t expected = s.num_levels * s.num_points * 2;
        TT_FATAL(
            static_cast<uint32_t>(ls[3]) == expected,
            "fused_msda: packed {} last dim ({}) must equal L*P*2 = {}",
            loc_name,
            ls[3],
            expected);
    }

    // ---- reference_points (V2 only) ----
    if (attrs.from_offsets) {
        const Tensor& ref = *args.reference_points;
        check_common_tensor(ref, "reference_points", value);
        const auto& rs = ref.logical_shape();
        TT_FATAL(rs.rank() == 4, "fused_msda: reference_points rank must be 4 (B, Q, R, 2), got shape {}", rs);
        TT_FATAL(
            static_cast<uint32_t>(rs[0]) == s.batch && static_cast<uint32_t>(rs[1]) == s.num_queries,
            "fused_msda: reference_points leading dims {} do not match (B, Q) = ({}, {})",
            rs,
            s.batch,
            s.num_queries);
        TT_FATAL(
            rs[3] == 2,
            "fused_msda: reference_points last dim must be 2 (normalized x, y), got {}. The 4-D box form "
            "(cx, cy, w, h), which scales offsets by ref_wh / (2*P) instead of 1 / [W_l, H_l], is a different "
            "operator and is not supported — compute sampling_locations on host and call fused_msda instead",
            rs[3]);
        s.num_refs = rs[2];
        TT_FATAL(s.num_refs > 0, "fused_msda: reference_points R dim must be > 0");
        if (attrs.reference_mode == MSDAReferenceMode::Level) {
            TT_FATAL(
                s.num_refs == s.num_levels,
                "fused_msda: reference_mode=\"level\" requires one reference point per level: R ({}) != L ({})",
                s.num_refs,
                s.num_levels);
        } else {
            TT_FATAL(
                s.num_points % s.num_refs == 0,
                "fused_msda: reference_mode=\"pillar\" requires P ({}) to be divisible by R ({})",
                s.num_points,
                s.num_refs);
        }
    }

    return s;
}

void FusedMSDAOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(
        attrs.from_offsets == args.sampling_offsets.has_value(),
        "fused_msda: internal error, from_offsets does not match the supplied tensors");
    TT_FATAL(
        attrs.from_offsets == args.reference_points.has_value(),
        "fused_msda: internal error, from_offsets does not match the supplied tensors");
    TT_FATAL(
        attrs.from_offsets != args.sampling_locations.has_value(),
        "fused_msda: internal error, from_offsets does not match the supplied tensors");
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "fused_msda: output_memory_config memory_layout must be INTERLEAVED");
    (void)derive_shapes(attrs, args);
}

void FusedMSDAOperation::validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

FusedMSDAOperation::spec_return_value_t FusedMSDAOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto s = derive_shapes(attrs, args);
    // Heads are concatenated by the writer, which places head h at byte offset
    // h * D * 2 inside query (b, q)'s page — so (B, Q, H*D) costs nothing extra
    // and saves every consumer a reshape + permute. See README.md §2.3.
    Shape out_shape({s.batch, s.num_queries, s.num_heads * s.head_dim});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), attrs.output_memory_config));
}

FusedMSDAOperation::tensor_return_value_t FusedMSDAOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.value.device());
}

}  // namespace ttnn::operations::experimental::fused_msda

namespace ttnn::prim {

namespace {

using OperationType = ttnn::operations::experimental::fused_msda::FusedMSDAOperation;

ttsl::SmallVector<uint32_t> flatten_spatial_shapes(const std::vector<std::array<uint32_t, 2>>& spatial_shapes) {
    TT_FATAL(!spatial_shapes.empty(), "fused_msda: spatial_shapes must not be empty");
    ttsl::SmallVector<uint32_t> flat;
    flat.reserve(2 * spatial_shapes.size());
    for (const auto& hw : spatial_shapes) {
        flat.push_back(hw[0]);
        flat.push_back(hw[1]);
    }
    return flat;
}

}  // namespace

ttnn::Tensor fused_msda(
    const Tensor& value,
    const Tensor& sampling_locations,
    const Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners,
    bool locations_in_grid_space,
    const std::optional<MemoryConfig>& memory_config) {
    auto attrs = OperationType::operation_attributes_t{
        .output_memory_config = memory_config.value_or(value.memory_config()),
        .spatial_shapes_hw = flatten_spatial_shapes(spatial_shapes),
        .align_corners = align_corners,
        .locations_in_grid_space = locations_in_grid_space,
        .from_offsets = false,
        .reference_mode = ttnn::experimental::MSDAReferenceMode::Level,
    };
    auto args = OperationType::tensor_args_t{
        .value = value,
        .attention_weights = attention_weights,
        .sampling_locations = sampling_locations,
        .reference_points = std::nullopt,
        .sampling_offsets = std::nullopt,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

ttnn::Tensor fused_msda_from_offsets(
    const Tensor& value,
    const Tensor& reference_points,
    const Tensor& sampling_offsets,
    const Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    ttnn::experimental::MSDAReferenceMode reference_mode,
    bool align_corners,
    const std::optional<MemoryConfig>& memory_config) {
    auto attrs = OperationType::operation_attributes_t{
        .output_memory_config = memory_config.value_or(value.memory_config()),
        .spatial_shapes_hw = flatten_spatial_shapes(spatial_shapes),
        .align_corners = align_corners,
        // V2 forms locations as ref + offset / [W, H], which is by construction
        // the MSDA [0, 1] space. A grid-space variant would need a different
        // reference-point convention too, so it is not offered here.
        .locations_in_grid_space = false,
        .from_offsets = true,
        .reference_mode = reference_mode,
    };
    auto args = OperationType::tensor_args_t{
        .value = value,
        .attention_weights = attention_weights,
        .sampling_locations = std::nullopt,
        .reference_points = reference_points,
        .sampling_offsets = sampling_offsets,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
