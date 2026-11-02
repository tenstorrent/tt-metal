// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_msda_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/fused_msda/fused_msda.hpp"

namespace ttnn::operations::experimental::fused_msda::detail {

namespace {

// `ttnn::bind_function` wraps a raw function pointer, so the string -> enum
// translation lives in a free function rather than a lambda.
ttnn::experimental::MSDAReferenceMode parse_reference_mode(const std::string& mode) {
    if (mode == "level") {
        return ttnn::experimental::MSDAReferenceMode::Level;
    }
    if (mode == "pillar") {
        return ttnn::experimental::MSDAReferenceMode::Pillar;
    }
    TT_THROW(
        "fused_msda_from_offsets: reference_mode must be \"level\" (one reference point per feature level, "
        "R == L) or \"pillar\" (one per z-anchor, P % R == 0), got \"{}\"",
        mode);
}

ttnn::experimental::MSDAValueLayout parse_value_layout(const std::string& layout) {
    if (layout == "auto") {
        return ttnn::experimental::MSDAValueLayout::Auto;
    }
    if (layout == "head_major") {
        return ttnn::experimental::MSDAValueLayout::HeadMajor;
    }
    TT_THROW(
        "fused_msda: value_layout must be \"auto\" ((B, S, H, D) or packed (B, S, H*D), by rank) or \"head_major\" "
        "((B, H, ceil(S/32), 32*D)), got \"{}\"",
        layout);
}

ttnn::Tensor fused_msda_py(
    const ttnn::Tensor& value,
    const ttnn::Tensor& sampling_locations,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners,
    bool locations_in_grid_space,
    const std::string& value_layout,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::experimental::fused_msda(
        value,
        sampling_locations,
        attention_weights,
        spatial_shapes,
        align_corners,
        locations_in_grid_space,
        parse_value_layout(value_layout),
        memory_config);
}

ttnn::Tensor fused_msda_from_offsets_py(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    const std::string& reference_mode,
    bool align_corners,
    const std::string& value_layout,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<uint32_t>>& num_valid_queries,
    const std::optional<ttnn::Tensor>& num_valid_queries_tensor) {
    return ttnn::experimental::fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        parse_reference_mode(reference_mode),
        align_corners,
        parse_value_layout(value_layout),
        memory_config,
        num_valid_queries,
        num_valid_queries_tensor);
}

bool can_use_fused_msda_from_offsets_l1_value_py(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    const std::string& reference_mode,
    bool align_corners,
    bool device_valid_counts,
    const std::string& value_layout) {
    return ttnn::experimental::can_use_fused_msda_from_offsets_l1_value(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        parse_reference_mode(reference_mode),
        align_corners,
        device_valid_counts,
        parse_value_layout(value_layout));
}

}  // namespace

void bind_fused_msda(nb::module_& mod) {
    const auto* v1_doc =
        R"doc(
        Generic multi-scale deformable attention: bilinear sampling, attention
        weighting and the reduction over levels x points fused into one kernel.

        For each (b, q, h)::

            out[b, q, h*D:(h+1)*D] = sum_l sum_p
                attention_weights[b, q, h, l, p]
                * bilinear(value[b, level l, h], sampling_locations[b, q, h, l, p])

        Args:
            * :attr:`value`: (B, S, H, D) ROW_MAJOR bfloat16, S = sum_l H_l * W_l,
              packed (B, S, H*D), or head-major (B, H, ceil(S/32), 32*D) with
              value_layout="head_major". Packed is what a Linear over embed_dims
              emits and costs one page per (b, s) instead of one per (b, s, h).
              Head-major, built by fused_msda_head_major_value, is fastest for
              D % 32 == 0: a horizontal corner pair is one read instead of two.
              Packed with H > 1 needs D*2 to be a multiple of the buffer's
              alignment (on Blackhole, D % 32 == 16 in DRAM is rejected, e.g.
              D = 16 or 48; use (B, S, H, D) or L1 there).
            * :attr:`sampling_locations`: (B, Q, H, L, P, 2) ROW_MAJOR bfloat16, or the
              packed form (B, Q, H, L*P*2). The packed form costs one DRAM page per
              (b, q, h) instead of one per sample point and avoids 8x page padding —
              prefer it where the producer can emit it. (B, Q, H*L*P*2), the heads of a
              query in one page as a Linear emits them, is also accepted when each head's
              L*P*2 values fill whole alignment units of their buffer.
            * :attr:`attention_weights`: (B, Q, H, L, P) ROW_MAJOR bfloat16, or packed (B, Q, H, L*P)
            * :attr:`spatial_shapes`: host-side list of (H_l, W_l) per level, in level order.
              Not a device tensor: it is a static property of the feature pyramid and is
              needed for address arithmetic inside the reader.
            * :attr:`align_corners`: pixel mapping within the chosen coordinate space
            * :attr:`locations_in_grid_space`: False (default) treats locations as normalized
              [0, 1] coordinates; True treats them as normalized [-1, 1] coordinates
            * :attr:`value_layout`: "auto" (default) tells (B, S, H, D) and packed
              (B, S, H*D) apart by rank; "head_major" declares (B, H, ceil(S/32), 32*D).
              Declared rather than inferred: that shape can coincide with a canonical one.
            * :attr:`memory_config`: output memory config

        Returns:
            (B, Q, H*D) ROW_MAJOR bfloat16.

        Out-of-bounds sampling corners contribute zero (padding_mode="zeros").
        Constraints: all tensors bfloat16 / ROW_MAJOR / INTERLEAVED, D a positive
        multiple of 16, 1 <= L <= 8, and each level's H_l and W_l <= 256. Q need
        not be a multiple of 32. Head-major value also needs D * 2 to be a
        multiple of its buffer's alignment.
        )doc";

    ttnn::bind_function<"fused_msda", "ttnn.experimental.">(
        mod,
        v1_doc,
        &fused_msda_py,
        nb::arg("value"),
        nb::arg("sampling_locations"),
        nb::arg("attention_weights"),
        nb::arg("spatial_shapes"),
        nb::kw_only(),
        nb::arg("align_corners") = false,
        nb::arg("locations_in_grid_space") = false,
        nb::arg("value_layout") = "auto",
        nb::arg("memory_config") = nb::none());

    const auto* v2_doc =
        R"doc(
        fused_msda with sampling-location generation fused into the op. No
        (B, Q, H, L, P, 2) location tensor is materialized::

            sampling_locations[b, q, h, l, p] =
                reference_points[b, q, r(l, p)]
                + sampling_offsets[b, q, h, l, p] * scale_l

        scale_l is [1/W_l, 1/H_l] when align_corners is false and
        [1/(W_l-1), 1/(H_l-1)] when it is true (0 on a singleton axis), so one
        offset unit is one feature-map pixel under either mapping.

        Args:
            * :attr:`value`: (B, S, H, D) or packed (B, S, H*D) ROW_MAJOR bfloat16, or head-major as in fused_msda
            * :attr:`reference_points`: (B, Q, R, 2) ROW_MAJOR bfloat16 or float32, normalized
              (x, y) in [0, 1]. float32 keeps the sampling position sub-pixel on wide maps.
              The 4-D box form is rejected, not reinterpreted.
            * :attr:`sampling_offsets`: (B, Q, H, L, P, 2), packed (B, Q, H, L*P*2) or
              heads-packed (B, Q, H*L*P*2) as for sampling_locations, ROW_MAJOR bfloat16. Raw, in feature-map pixel units — the op accounts
              for scale_l (1/[W_l, H_l], or 1/[W_l-1, H_l-1] when align_corners is
              set).
            * :attr:`attention_weights`: as in fused_msda
            * :attr:`spatial_shapes`: as in fused_msda
            * :attr:`reference_mode`:
                - "level"  (default): R == L, r(l, p) = l. DINO-family decoders, and
                  BEVFormer temporal attention with an ego shift (its maps are levels).
                - "pillar": P % R == 0, r(l, p) = p % R, with the point axis laid out
                  as (P // R, R). BEVFormer spatial attention, and temporal attention
                  without a shift (R == 1).
            * :attr:`align_corners`: pixel mapping
            * :attr:`value_layout`: as in fused_msda
            * :attr:`memory_config`: output memory config
            * :attr:`num_valid_queries`: optional list of B counts, each in [0, Q] and at
              least one nonzero. Only queries [0, num_valid_queries[b]) of batch b are
              computed; None computes every query. Changing the counts reuses the cached
              program, and a captured trace replays the counts it was captured with.
            * :attr:`num_valid_queries_tensor`: the counts as a uint32 ROW_MAJOR
              INTERLEAVED tensor on value's device, all B of them in the last dim (one
              unpadded page; (B,) is the natural shape). The kernels read it on every
              run, so a captured trace computes what the captured tensor holds at
              replay; write new counts into that tensor (e.g. with
              ttnn.copy_host_to_device_tensor), since the trace keeps its address.
              A device read cannot fail, so the rules differ from the list's: a count
              above Q is treated as Q and all-zero counts compute nothing. It compiles
              a program separate from the list form's. Pass at most one of
              num_valid_queries and num_valid_queries_tensor.

        Returns:
            (B, Q, H*D) ROW_MAJOR bfloat16, equivalent to calling fused_msda on the
            materialized locations, up to rounding. With either count form, that holds for
            rows below each batch's count; the rows past it are unwritten and hold
            undefined data (possibly NaN or Inf), so drop them with a select such as
            ttnn.where, not by multiplying with a 0/1 mask.

        Constraints: as fused_msda, including each level's H_l and W_l <= 256.
        )doc";

    ttnn::bind_function<"fused_msda_from_offsets", "ttnn.experimental.">(
        mod,
        v2_doc,
        &fused_msda_from_offsets_py,
        nb::arg("value"),
        nb::arg("reference_points"),
        nb::arg("sampling_offsets"),
        nb::arg("attention_weights"),
        nb::arg("spatial_shapes"),
        nb::kw_only(),
        nb::arg("reference_mode") = "level",
        nb::arg("align_corners") = false,
        nb::arg("value_layout") = "auto",
        nb::arg("memory_config") = nb::none(),
        nb::arg("num_valid_queries") = nb::none(),
        nb::arg("num_valid_queries_tensor") = nb::none());

    const auto* l1_value_fit_doc =
        R"doc(
        Return whether fused_msda_from_offsets can safely stage value and output
        in interleaved L1 under the device's current allocator state.

        The check counts value as it will be staged in L1 (not counted again when
        it is already ROW_MAJOR in L1) and the bfloat16 output, each rounded up to
        the allocator's per-bank granule, plus the operation's circular buffers,
        under the 95% program-L1 margin. value may still be TILE or in DRAM.

        Args beyond fused_msda_from_offsets':
            * :attr:`device_valid_counts`: True when the op will get
              num_valid_queries_tensor, whose staging adds circular buffers.
            * :attr:`value_layout`: the layout the op will read. "auto" charges value
              staged ROW_MAJOR in its own shape. "head_major" takes value either
              already head-major or packed (B, S, H*D) that the caller will convert
              with fused_msda_head_major_value, and charges the head-major tensor,
              which pads S to a multiple of 32 per head.

        The answer reflects the allocator's current state; call it right before the
        op and do not cache it.
        )doc";

    ttnn::bind_function<"can_use_fused_msda_from_offsets_l1_value", "ttnn.experimental.">(
        mod,
        l1_value_fit_doc,
        &can_use_fused_msda_from_offsets_l1_value_py,
        nb::arg("value"),
        nb::arg("reference_points"),
        nb::arg("sampling_offsets"),
        nb::arg("attention_weights"),
        nb::arg("spatial_shapes"),
        nb::kw_only(),
        nb::arg("reference_mode") = "level",
        nb::arg("align_corners") = false,
        nb::arg("device_valid_counts") = false,
        nb::arg("value_layout") = "auto");

    const auto* head_major_doc =
        R"doc(
        Converts the tiled packed value (B, S, H*D) that value_proj emits into the
        ROW_MAJOR head-major layout (B, H, ceil(S/32), 32*D). Each page holds 32
        consecutive keys of one head, so fused_msda / fused_msda_from_offsets read
        both corners of a horizontal bilinear pair in one NoC read. Pass the result
        with value_layout="head_major".

        Args:
            * :attr:`value`: (B, S, H*D) TILE bfloat16, INTERLEAVED; D a multiple of 32;
              32x32 tiles padded only up to a whole tile of S, as ttnn.linear and
              ttnn.to_layout produce
            * :attr:`num_heads`: H
            * :attr:`memory_config`: output memory config; None keeps value's. L1 gives
              the fastest gather.

        Returns:
            (B, H, ceil(S/32), 32*D) ROW_MAJOR bfloat16. Rows past S in each head's
            last page hold the input's tile padding and are never read by the op.
        )doc";

    ttnn::bind_function<"fused_msda_head_major_value", "ttnn.experimental.">(
        mod,
        head_major_doc,
        &ttnn::experimental::fused_msda_head_major_value,
        nb::arg("value"),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::fused_msda::detail
