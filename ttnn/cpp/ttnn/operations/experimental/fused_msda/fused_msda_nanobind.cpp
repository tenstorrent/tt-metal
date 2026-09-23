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

ttnn::Tensor fused_msda_from_offsets_py(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    const std::string& reference_mode,
    bool align_corners,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::experimental::fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        parse_reference_mode(reference_mode),
        align_corners,
        memory_config);
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
              or packed (B, S, H*D). The packed form is what a Linear over
              embed_dims emits and costs one DRAM page per (b, s) instead of one
              per (b, s, h) — prefer it.
            * :attr:`sampling_locations`: (B, Q, H, L, P, 2) ROW_MAJOR bfloat16, or the
              packed form (B, Q, H, L*P*2). The packed form costs one DRAM page per
              (b, q, h) instead of one per sample point and avoids 8x page padding —
              prefer it where the producer can emit it.
            * :attr:`attention_weights`: (B, Q, H, L, P) ROW_MAJOR bfloat16, or packed (B, Q, H, L*P)
            * :attr:`spatial_shapes`: host-side list of (H_l, W_l) per level, in level order.
              Not a device tensor: it is a static property of the feature pyramid and is
              needed for address arithmetic inside the reader.
            * :attr:`align_corners`: pixel mapping within the chosen coordinate space
            * :attr:`locations_in_grid_space`: False (default) treats locations as MSDA/mmcv
              [0, 1]; True treats them as grid_sample [-1, 1]
            * :attr:`memory_config`: output memory config

        Returns:
            (B, Q, H*D) ROW_MAJOR bfloat16.

        Out-of-bounds sampling corners contribute zero (padding_mode="zeros").
        Constraints: all tensors bfloat16 / ROW_MAJOR / INTERLEAVED, D a positive
        multiple of 16, 1 <= L <= 8, and each level's H_l and W_l <= 256. Q need
        not be a multiple of 32.
        )doc";

    ttnn::bind_function<"fused_msda", "ttnn.experimental.">(
        mod,
        v1_doc,
        &ttnn::experimental::fused_msda,
        nb::arg("value"),
        nb::arg("sampling_locations"),
        nb::arg("attention_weights"),
        nb::arg("spatial_shapes"),
        nb::kw_only(),
        nb::arg("align_corners") = false,
        nb::arg("locations_in_grid_space") = false,
        nb::arg("memory_config") = nb::none());

    const auto* v2_doc =
        R"doc(
        fused_msda with sampling-location generation fused into the op. No
        (B, Q, H, L, P, 2) location tensor is materialized::

            sampling_locations[b, q, h, l, p] =
                reference_points[b, q, r(l, p)]
                + sampling_offsets[b, q, h, l, p] / [W_l, H_l]

        Args:
            * :attr:`value`: (B, S, H, D) or packed (B, S, H*D) ROW_MAJOR bfloat16
            * :attr:`reference_points`: (B, Q, R, 2) ROW_MAJOR bfloat16, normalized (x, y)
              in [0, 1]. The 4-D box form is rejected, not reinterpreted.
            * :attr:`sampling_offsets`: (B, Q, H, L, P, 2) or packed (B, Q, H, L*P*2)
              ROW_MAJOR bfloat16. Raw, in feature-map pixel units — the op applies
              the per-level / [W_l, H_l] normalization on device.
            * :attr:`attention_weights`: as in fused_msda
            * :attr:`spatial_shapes`: as in fused_msda
            * :attr:`reference_mode`:
                - "level"  (default): R == L, r(l, p) = l. DINO-family decoders.
                - "pillar": P % R == 0, r(l, p) = p % R, with the point axis laid out
                  as (P // R, R). BEVFormer spatial/temporal attention.
            * :attr:`align_corners`: pixel mapping
            * :attr:`memory_config`: output memory config

        Returns:
            (B, Q, H*D) ROW_MAJOR bfloat16, identical to calling fused_msda on the
            equivalent materialized locations.
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
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::fused_msda::detail
