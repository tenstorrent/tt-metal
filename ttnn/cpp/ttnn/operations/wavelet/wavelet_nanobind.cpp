// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/wavelet_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/wavelet/wavelet.hpp"

namespace ttnn::operations::wavelet {

void bind_wavelet_operations(nb::module_& mod) {
    ttnn::bind_function<"dwt_coeff_len">(
        mod,
        R"doc(
Return the valid coefficient count for one level of ``ttnn.dwt``.
)doc",
        &ttnn::dwt_coeff_len,
        nb::arg("input_length"),
        nb::arg("wavelet"));

    ttnn::bind_function<"dwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D Discrete Wavelet Transform.

``input`` must be a row-major INTERLEAVED FLOAT32 tensor with shape ``[W]`` or
``[B,1,1,W]`` in DRAM or L1 on one physical device. Outputs remain INTERLEAVED
DRAM tensors and preserve the optional batch dimensions.
``wavelet`` names one of the 106 supported discrete wavelet schemes and
``boundary_mode`` is one of ``zero``, ``constant``, ``symmetric``,
``reflect``, ``periodic``, ``smooth``, ``antisymmetric``, or ``antireflect``.

Returns ``(approximation, detail)``. Let
``C = ttnn.dwt_coeff_len(input.shape[-1], wavelet)``. Each output has
stick-native shape ``[ceil(C/32), 32]`` or ``[B,1,ceil(C/32),32]`` and a
128-byte physical page per stick. Only the first ``C`` flattened values per
batch item are valid; unused final-stick lanes are unspecified. When supplied,
``output_tensors`` must contain two non-aliasing tensors with the exact inferred
specification.
)doc",
        &ttnn::dwt,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"idwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D Inverse Discrete Wavelet Transform.

``approximation`` and ``detail`` must be non-aliasing, equal-shaped row-major
INTERLEAVED FLOAT32 tensors with canonical shape ``[Wc]``/``[B,1,1,Wc]`` or
stick-native shape ``[S,32]``/``[B,1,S,32]`` in DRAM or L1 on the same device.
Their placements may differ. The output remains an INTERLEAVED DRAM tensor.
``original_length`` restores the exact odd or even logical length and must be
consistent with the coefficient shape, wavelet, and boundary mode.

Returns a stick-native tensor with shape ``[ceil(original_length/32),32]``
or ``[B,1,ceil(original_length/32),32]``. Only the first
``original_length`` flattened values per batch item are valid; unused lanes
are unspecified. ``output_tensor`` may provide exact-spec preallocated
storage and must not alias either input.
)doc",
        &ttnn::idwt,
        nb::arg("approximation").noconvert(),
        nb::arg("detail").noconvert(),
        nb::arg("wavelet"),
        nb::arg("original_length"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    ttnn::bind_function<"dwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D discrete wavelet transform.

``input`` must be a standard 32x32 tile-layout INTERLEAVED FLOAT32 tensor with
shape ``[H,W]`` or ``[B,1,H,W]`` in DRAM or L1 on one physical device. Outputs
remain INTERLEAVED DRAM tensors and preserve the optional batch dimensions. The operation preserves
the standalone vertical-first execution order and returns ``(LL, LH, HL, HH)``.

``output_tensors`` may provide four pairwise non-aliasing tensors with the exact
inferred specifications.
)doc",
        &ttnn::dwt_2d,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"idwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D inverse discrete wavelet transform.

``ll``, ``lh``, ``hl``, and ``hh`` must be pairwise non-aliasing, equal-shaped,
standard 32x32 tile-layout INTERLEAVED FLOAT32 tensors with shape ``[Hc,Wc]``
or ``[B,1,Hc,Wc]`` in DRAM or L1 on the same physical device. Their placements
may differ. The output remains an
INTERLEAVED DRAM tensor. ``output_shape=(height, width)`` restores the exact odd or even logical
dimensions and must be consistent with the coefficient shape.

Returns one tensor matching the input rank and batch. ``output_tensor`` may provide exact-spec
preallocated storage and must not alias an input band.
)doc",
        &ttnn::idwt_2d,
        nb::arg("ll").noconvert(),
        nb::arg("lh").noconvert(),
        nb::arg("hl").noconvert(),
        nb::arg("hh").noconvert(),
        nb::arg("wavelet"),
        nb::arg("output_shape"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::wavelet
