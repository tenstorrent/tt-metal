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
        Experimental fused MoE compute supporting arbitrary ``(hidden_size, intermediate_size)`` pairs.

        This operation performs the expert matmuls (gate/up projection via W0/W1, down
        projection via W2) and activation (SILU or SwiGLU) in a fused compute kernel.
        Tile distribution across the 12-core ring is derived at compile time from
        ``hidden_size`` and ``intermediate_size`` using Euclidean-rhythm (Bresenham)
        shard formulas — no model-specific configuration tables are needed.

        Note: This is the **compute** portion of the MoE pipeline. The A2A dispatch
        (producing the sparse buffer consumed by this op) and the A2A combine (reducing
        expert outputs) are handled by separate collective operations; see the MoE
        module tests for the full flow.

        **Weight tensor layout (CRITICAL)**

        The API takes **two packed weight tensor arguments** that contain **three logical
        expert weight matrices** (W0, W1, W2):

        - ``matmul_w0_w1_tensor``: Interleaved W0 and W1 (gate + up projection weights).
        - ``matmul_w2_tensor``: W2 (down projection weights).

        The exact byte layout expected by the kernels depends on ``hidden_size`` and
        ``intermediate_size``. Callers must match the layout the device op expects or
        use the reference packer from ``ttnn.experimental.moe_compute_utils`` (see below).

        **Key parameters**

        - ``intermediate_size`` (**required**, added in this version): The MoE
          intermediate (expert FFN) dimension. Together with ``hidden_size`` (inferred
          from the input tensor), this determines the per-core tile shard counts via
          ``shard_tiles()`` / ``w2_shard_tiles()`` and the number of data-parallel
          cores (``num_data_parallel_cores``). Previously, tile distributions were
          selected by a model-specific enum; this parameter replaces that mechanism.

        - ``output_height_shard_dim``: Number of tile columns per output shard. Use
          ``auto_output_width_shard_dim(hidden_size)`` from ``moe_compute_utils`` to
          compute the optimal value for a given hidden size.

        **Bias support (optional)**

        - ``has_bias=False`` (default): tensors contain only weights.
        - ``has_bias=True``: tensors must include fused bias tiles. Bias values are
          appended to the weight tensors in a kernel-specific format:

          - For W0/W1: Bias tiles (shape expanded to TILE_SIZE rows with row 0 populated)
            are concatenated along the K dimension, then K is padded to a multiple of
            BLOCK_TILES_H (7) tiles.
          - For W2: The bias tile is appended along the N (intermediate) dimension
            **without** ring-rotation. N is then padded to a multiple of BLOCK_TILES_H
            tiles.

        ``has_bias`` must match the actual layout of the provided tensors; mismatch
        produces silent wrong results or UB.

        **Reference packer (optional)**

        ``ttnn.experimental.moe_compute_utils`` provides reference implementations that
        produce the expected layout:

        - No bias: ``prepare_w0_w1_tensor_for_moe_compute``,
          ``prepare_w2_tensor_for_moe_compute``
        - With bias: ``prepare_w0_w1_tensor_with_bias``,
          ``prepare_w2_tensor_with_bias``
        - Shard maps: ``get_weight_core_shard_maps(mesh_device, hidden_size, intermediate_size)``
        - Memory configs: ``get_weight_mem_configs(...)``
        - Output shard dim: ``auto_output_width_shard_dim(hidden_size)``

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
        nb::arg("intermediate_size"),
        nb::arg("has_bias") = false,
        nb::arg("cluster_axis"),
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
