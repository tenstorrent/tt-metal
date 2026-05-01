// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moe_compute_nanobind.hpp"
#include "moe_compute.hpp"
#include "device/kernels/moe_ring_common.h"
#include "device/hostdevcommon/config.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_moe_compute(nb::module_& mod) {
    // Bind the activation function enum
    nb::enum_<ttnn::experimental::prim::detail::MoEActivationFunction>(mod, "MoEActivationFunction")
        .value("SILU", ttnn::experimental::prim::detail::MoEActivationFunction::SILU)
        .value("SWIGLU", ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU);
    ttnn::bind_function<"moe_compute", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental fused MoE compute for DeepSeek-class models.

        This operation performs the expert matmuls (gate/up projection via W0/W1, down
        projection via W2) and activation (SILU or SwiGLU) in a fused compute kernel.

        Note: This is the **compute** portion of the MoE pipeline. The A2A dispatch
        (producing the sparse buffer consumed by this op) and the A2A combine (reducing
        expert outputs) are handled by separate collective operations; see the MoE
        module tests for the full flow.

        **Weight tensor layout (CRITICAL)**

        The API takes **two packed weight tensor arguments** that contain **three logical
        expert weight matrices** (W0, W1, W2):

        - ``matmul_w0_w1_tensor``: Interleaved W0 and W1 (gate + up projection weights).
        - ``matmul_w2_tensor``: W2 (down projection weights).

        The exact byte layout expected by the kernels is model- and config-dependent.
        Callers must match the layout the device op expects or use the reference packer
        from ``ttnn.experimental.moe_compute_utils`` (see below).

        **Bias support (optional)**

        - ``has_bias=False`` (default): tensors contain only weights; original no-bias
          layout (K=7168 elements for the reference config, N+192=2240).
        - ``has_bias=True``: tensors must include fused bias tiles. Bias values are
          appended to the weight tensors in a kernel-specific format:

          - For W0/W1: Bias tiles (shape expanded to TILE_SIZE rows with row 0 populated)
            are concatenated along the K dimension, then K is padded to a multiple of
            14 tiles (W0_W1_TILES_PER_TXN). In the reference config this yields K_padded
            = 7616 elements (238 tiles).
          - For W2: The bias tile is appended along the N (intermediate) dimension
            **without** ring-rotation (matching GPT-OSS behavior). N is padded to 70
            tiles (2240 elements), same total size as the non-bias path.

        ``has_bias`` must match the actual layout of the provided tensors; mismatch
        produces silent wrong results or UB.

        **Reference packer (optional)**

        ``ttnn.experimental.moe_compute_utils`` provides reference implementations that
        produce the expected layout:

        - No bias: ``prepare_w0_w1_tensor_for_moe_compute``,
          ``prepare_w2_tensor_for_moe_compute``
        - With bias: ``prepare_w0_w1_tensor_with_bias``,
          ``prepare_w2_tensor_with_bias``

        These functions are kept in sync with the test suite and can be used as
        "executable documentation" for the layout contract; they are not a required
        public API.

        See also: ``ttnn.experimental.moe_compute_utils`` module docstring for full
        layout details, constants (TILES_PER_TXN, ring tables), and constraints.
        )doc",
        &ttnn::experimental::moe_compute,
        nb::arg("tilize_input_tensor").noconvert(),
        nb::arg("tilize_expert_indices_tensor").noconvert(),
        nb::arg("tilize_expert_scores_tensor").noconvert(),
        nb::arg("tilize_expert_mapping_tensor").noconvert(),
        nb::arg("matmul_w0_w1_tensor").noconvert(),
        nb::arg("matmul_w2_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("layer_id"),
        nb::arg("output_height_shard_dim"),
        nb::arg("has_bias") = false,
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("mux_core_range_set") = nb::none(),
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none(),
        nb::arg("optional_cross_device_semaphore") = nb::none(),
        nb::arg("activation_type") = nb::none());
}

void bind_get_moe_combine_cores(nb::module_& mod) {
    const auto* doc = R"doc(Return the ordered list of cores assigned to A2A Combine for the MoE module flow )doc";
    ttnn::bind_function<"get_moe_combine_cores", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<ttnn::MeshDevice*, const uint32_t, const uint32_t>(
                &ttnn::experimental::get_moe_combine_cores),
            nb::arg("mesh_device"),
            nb::arg("combine_token_parallel_cores"),
            nb::arg("combine_data_parallel_cores")));
}
}  // namespace ttnn::operations::experimental::ccl
