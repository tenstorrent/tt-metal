// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "turbo_quant_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/turbo_quant/turbo_quant.hpp"

namespace ttnn::operations::experimental::turbo_quant {

void bind_turbo_quant_operations(nb::module_& mod) {
    ttnn::bind_function<"turbo_quant_bucketize", "ttnn.experimental.">(
        mod,
        R"doc(
Fused TurboQuant bucketize.  Maps normalised rotated values to integer
bucket indices using a single device kernel (replaces 13 cascaded TTNN ops).

Args:
    input_tensor: BF16 TILE tensor of normalised values.
    boundaries:   List of inner boundary floats (len = 2^bits - 1).

Returns:
    BF16 TILE tensor with integer indices 0 … 2^bits-1.
)doc",
        &ttnn::turbo_quant_bucketize,
        nb::arg("input_tensor").noconvert(),
        nb::arg("boundaries"));

    ttnn::bind_function<"turbo_quant_gather_centroids", "ttnn.experimental.">(
        mod,
        R"doc(
Fused TurboQuant gather centroids.  Maps integer indices to their
corresponding centroid values using a single device kernel (replaces 21
cascaded TTNN ops).

Args:
    input_tensor: BF16 TILE tensor with integer indices.
    centroids:    List of centroid floats (len = 2^bits).

Returns:
    BF16 TILE tensor with centroid values.
)doc",
        &ttnn::turbo_quant_gather_centroids,
        nb::arg("input_tensor").noconvert(),
        nb::arg("centroids"));
}

}  // namespace ttnn::operations::experimental::turbo_quant
