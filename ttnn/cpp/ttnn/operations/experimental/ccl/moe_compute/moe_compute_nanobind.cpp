// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "moe_compute_nanobind.hpp"
#include "moe_compute.hpp"
#include "moe_compute_utils.hpp"
#include "device/kernels/moe_ring_common.h"
#include "device/hostdevcommon/config.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_moe_compute(nb::module_& mod) {
    // Bind the activation function enum
    nb::enum_<ttnn::experimental::prim::detail::MoEActivationFunction>(mod, "MoEActivationFunction")
        .value("SILU", ttnn::experimental::prim::detail::MoEActivationFunction::SILU)
        .value("SWIGLU", ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU)
        .value("GELU", ttnn::experimental::prim::detail::MoEActivationFunction::GELU);
    ttnn::bind_function<"moe_compute", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental fused MoE compute supporting arbitrary ``(hidden_size, intermediate_size)`` pairs.

        This operation performs the expert matmuls (gate/up projection via W0/W1, down
        projection via W2) and activation (SILU, SwiGLU, or GELU) in a fused compute kernel.
        Tile distribution across the matmul ring (12 cores on Wormhole; 8 cores on
        Blackhole — auto-detected from the architecture) is derived at compile time from
        ``hidden_size`` and ``intermediate_size`` using Euclidean-rhythm (Bresenham) shard
        formulas — no model-specific configuration tables are needed.

        Note: This is the **compute** portion of the MoE pipeline. The A2A dispatch
        (producing the sparse buffer consumed by this op) and the A2A combine (reducing
        expert outputs) are handled by separate collective operations; see the MoE
        module tests for the full flow.

        **Hardware / device configuration**

        - **Unharvested Wormhole chips.** The multi-device (6U/Galaxy) flow assumes the
          full Wormhole compute grid: the WH worker layout hardcodes the drain tilize core
          at logical ``(6, 9)`` and the combine cores at columns 5–6, so the ``y=9`` compute
          row must be present. Harvested WH SKUs that drop that row are not supported on
          this path (the single-card test path derives the drain core dynamically instead).
        - **``DispatchCoreAxis.COL``.** The mesh device must be opened with
          ``dispatch_core_axis=ttnn.DispatchCoreAxis.COL`` so dispatch cores occupy a column
          edge and do not overlap the op's tilize/matmul/combine worker cores.

        See https://github.com/tenstorrent/tt-metal/issues/41132 for details.

        **Weight tensor layout (CRITICAL)**

        The API takes **two packed weight tensor arguments** that contain **three logical
        expert weight matrices** (W0, W1, W2):

        - ``matmul_w0_w1_tensor``: Interleaved W0 and W1 (gate + up projection weights).
        - ``matmul_w2_tensor``: W2 (down projection weights).

        The exact byte layout expected by the kernels depends on ``hidden_size`` and
        ``intermediate_size``. Callers must match the layout the device op expects or
        use the reference packer from ``ttnn.experimental.moe_compute_utils`` (see below).

        **Input tensors**

        The first four tensors are the per-device outputs of the A2A dispatch op:

        - ``tilize_input_tensor``: The dispatched token activations (sparse buffer) for
          the experts that live on this device. ``hidden_size`` is inferred from its last
          dim and ``total_tokens`` from its first two dims.
        - ``tilize_expert_indices_tensor``: Per-token selected expert ids;
          ``select_experts_k`` is inferred from its last dim.
        - ``tilize_expert_scores_tensor``: Per-token routing scores (gates) applied to
          the expert outputs.
        - ``tilize_expert_mapping_tensor``: The expert → device mapping that tells the
          kernel which experts are resident locally.

        **Key parameters**

        - ``intermediate_size`` (**required**, added in this version): The MoE
          intermediate (expert FFN) dimension. Together with ``hidden_size`` (inferred
          from the input tensor), this determines the per-core tile shard counts via
          ``shard_tiles()`` / ``w2_shard_tiles()`` and the number of data-parallel
          cores (``num_data_parallel_cores``).

        - ``output_height_shard_dim``: Number of token-parallel (height) cores used
          for the combine output.

        - ``num_shared_experts_per_device`` (optional, default ``None`` ≡ 0): How many
          of the per-device experts are **shared** experts (run on every token) rather
          than routed. Shared experts occupy the **tail** slots of the experts dimension
          Note: this parameter differs from `num_shared_experts` used by all_to_all_dispatch_metadata
          that op counts all logical shared experts. This one counts *physical* shared experts
          per device, eg: 2x routed shared experts each residing on half of the devices
          would be num_shared_experts_per_device=1 and num_shared_experts=2

        - ``layer_id`` (**required**): Selects which layer's weight block to read when
          multiple layers are packed into a single DRAM-resident weight tensor.

        - ``activation_type`` (optional, default ``None`` ≡ ``SILU``): The expert FFN
          activation function — one of ``ttnn.experimental.MoEActivationFunction``
          ``{SILU, SWIGLU, GELU}`` — applied between the W0/W1 and W2 projections.

        - ``compute_only`` (default ``False``): When ``True``, run only the expert
          matmuls and skip the A2A combine. The op then returns **5** tensors (the
          matmul output is the final output, slot 4) instead of 6, and all combine-path
          arguments below must be left unset (notably ``cluster_axis`` must be ``None``).

        The matmul ring size is **auto-detected** from the architecture — 8 on Blackhole,
        12 on Wormhole (one per DRAM bank) — and is not exposed on this API. The
        ``prepare_*`` / ``get_weight_mem_configs`` helpers that pack the weights must be
        called with the matching ring size (see ``effective_matmul_ring_size``).

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

        **Combine path (Full mode only)**

        These arguments configure the cross-device A2A combine that reduces expert
        outputs. They apply only when ``compute_only=False``; with ``compute_only=True``
        they must be left at their defaults.

        - ``cluster_axis`` (**required when** ``compute_only=False``): The mesh axis along
          which the combine reduces. Must be ``None`` when ``compute_only=True``.
        - ``topology`` (optional, default ``None`` ≡ fabric default): Combine fabric
          topology; only ``ttnn.Topology.Linear`` and ``ttnn.Topology.Ring`` are
          supported. If the fabric default is Torus/Mesh, pass ``Linear`` or ``Ring``
          explicitly (BH Loudbox callers must pass ``Linear``).
        - ``num_links`` (optional, default ``None``): Number of fabric links for the
          combine; auto-detected from the mesh and ``cluster_axis`` when ``None``.
        - ``mux_core_range_set`` (optional, default ``None`` ≡ empty): Cores assigned to
          the fabric mux on the combine path.
        - ``output_memory_config`` (optional, default ``None`` ≡ ``DRAM_MEMORY_CONFIG``):
          Memory config for the combine output tensor.
        - ``optional_output_tensor`` (optional): Preallocated tensor to receive the
          combine output instead of allocating a new one. Must be ``None`` when
          ``compute_only=True`` (no combine output is produced).
        - ``optional_cross_device_semaphore`` (optional): Global semaphore used to
          synchronize the cross-device combine.

        **Reference input packer **

        ``ttnn.experimental.moe_compute_utils`` provides reference implementations that
        produce the expected layout:

        - add_shared_expert_weights
        - prepare_w0_w1_tensor_for_moe_compute/prepare_w0_w1_tensor_with_bias
        - prepare_w2_tensor_for_moe_compute/prepare_w2_tensor_with_bias
        - quantize_weights_via_host (slower but higer quality) or ttnn.typecast (faster)

        These functions are kept in sync with the test suite and can be used as
        "executable documentation" for the layout contract; they are not a required
        public API but they are recommended.

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
        // cluster_axis is required when compute_only=False; pass None for compute_only=True paths.
        // (Two breaking changes vs prior versions: (1) intermediate_size is now required positional
        // from PR #43932; (2) cluster_axis became optional, new compute_only knob. The matmul ring
        // size is auto-detected from the arch (8 BH / 12 WH); bh_ring_size remains only on prim.)
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("mux_core_range_set") = nb::none(),
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none(),
        nb::arg("optional_cross_device_semaphore") = nb::none(),
        nb::arg("activation_type") = nb::none(),
        nb::arg("compute_only") = false,
        nb::arg("num_shared_experts_per_device") = nb::none());
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

    const auto* bbox_doc =
        R"doc(Return the logical CoreRange bounding box of tilize + matmul + combine worker cores.
This matches the `all_worker_cores_bounding_box` used by the tilize kernel's per-expert count mcast.)doc";
    ttnn::bind_function<"get_moe_worker_mcast_bounding_box", "ttnn.experimental.">(
        mod,
        bbox_doc,
        ttnn::overload_t(
            nb::overload_cast<ttnn::MeshDevice*, const uint32_t, const uint32_t, const uint32_t>(
                &ttnn::experimental::get_moe_worker_mcast_bounding_box),
            nb::arg("mesh_device"),
            nb::arg("combine_token_parallel_cores"),
            nb::arg("combine_data_parallel_cores"),
            nb::arg("hidden_size")));
}

void bind_moe_compute_utils(nb::module_& mod) {
    nb::class_<ttnn::experimental::WeightCoreShardMaps>(mod, "WeightCoreShardMaps")
        .def_ro("w0_w1_shard_map", &ttnn::experimental::WeightCoreShardMaps::w0_w1_shard_map)
        .def_ro("w2_shard_map", &ttnn::experimental::WeightCoreShardMaps::w2_shard_map)
        .def_ro("dram_core_range_set", &ttnn::experimental::WeightCoreShardMaps::dram_core_range_set);

    ttnn::bind_function<"get_weight_core_shard_maps", "ttnn.experimental.">(
        mod,
        R"doc(
        Compute per-ring-position shard maps for W0/W1 and W2 weight tensors,
        plus the DRAM-bank ``CoreRangeSet`` used by the packed weight tensors'
        memory configs.

        Uses ``shard_tiles`` (Euclidean rhythm) for W0/W1 and ``w2_shard_tiles``
        (complementary when ``Nt%n_cores + Ht%n_cores == n_cores``) for W2. Ring
        ordering: DRAM bank logical coords sorted by ``(y, x)`` descending.

        The matmul ring size is auto-detected from the architecture — 8 on Blackhole,
        12 on Wormhole (the DRAM-bank count) — matching ``ttnn.experimental.moe_compute``,
        so the packed weights always line up with the op.

        Returns an object with ``w0_w1_shard_map``, ``w2_shard_map``, and
        ``dram_core_range_set`` attributes.
        )doc",
        &ttnn::experimental::get_weight_core_shard_maps,
        nb::arg("mesh_device"),
        nb::arg("hidden_size"),
        nb::arg("intermediate_size"));

    nb::class_<ttnn::experimental::WeightMemoryConfigs>(mod, "WeightMemoryConfigs")
        .def_ro("w0_w1", &ttnn::experimental::WeightMemoryConfigs::w0_w1)
        .def_ro("w2", &ttnn::experimental::WeightMemoryConfigs::w2);

    ttnn::bind_function<"get_weight_mem_configs", "ttnn.experimental.">(
        mod,
        R"doc(
        Build the DRAM-sharded ``MemoryConfig`` for the packed W0/W1 and W2
        weight tensors. Shard maps and DRAM-bank ``CoreRangeSet`` are computed
        internally from ``hidden_size`` and ``intermediate_size``.

        When ``has_bias`` is true, padded K (for W0/W1) and N (for W2) grow by
        one tile and are re-aligned to a multiple of ``BLOCK_TILES_H`` tiles.

        Returns an object with ``w0_w1`` and ``w2`` ``MemoryConfig`` attributes.
        )doc",
        &ttnn::experimental::get_weight_mem_configs,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("num_layers"),
        nb::arg("experts_per_device"),
        nb::arg("hidden_size"),
        nb::arg("intermediate_size"),
        nb::arg("has_bias") = false);

    ttnn::bind_function<"add_shared_expert_weights", "ttnn.experimental.">(
        mod,
        R"doc(
        Append per-device shared experts after routed experts along the experts dim.

        Inputs are multi-device tensors sharded on dim 1 (experts). Each device's
        routed shard holds its assigned routed experts; each device's shared shard
        holds its assigned shared experts, already in the correct slot order.
        Callers own the device → shared-expert mapping and produce the
        pre-arranged ``shared_w*`` tensors.

        The shared experts are tensor-parallel split on the intermediate dim across
        ``1 - cluster_axis`` (via ``mesh_partition``). Each ring core's real TpNt
        slice is front-packed into the front of that core's full-Nt shard (zeros
        after), using the same per-core shard maps the kernel derives, and the
        identical mapping is applied to W0/W1 (intermediate dim) and W2 (contraction
        dim) so real columns stay paired with their real W2 rows. This keeps the
        downstream prep + DRAM layout uniform (full-Nt per-expert stride) while
        letting the kernel walk only the per-core prefixes as a balanced TpNt ring.
        The shard-map generator auto-detects the ring size from the arch (8 on Blackhole,
        12 on Wormhole), matching ``prepare_*`` / ``get_weight_mem_configs`` and the op.

        Returns ``(output_w0, output_w1, output_w2)``, each the result of
        concatenating routed + shared along dim 1.
        )doc",
        &ttnn::experimental::add_shared_expert_weights,
        nb::arg("routed_w0").noconvert(),
        nb::arg("routed_w1").noconvert(),
        nb::arg("routed_w2").noconvert(),
        nb::arg("shared_w0").noconvert(),
        nb::arg("shared_w1").noconvert(),
        nb::arg("shared_w2").noconvert(),
        nb::arg("cluster_axis"));

    ttnn::bind_function<"prepare_w0_w1_tensor_for_moe_compute", "ttnn.experimental.">(
        mod,
        R"doc(
        Pack W0/W1 into the interleaved, padded, per-core layout the MoE kernel
        reads. See ``ttnn.experimental.moe_compute_utils`` for the layout
        contract. Output local shape:
        ``(num_cores, L, E, groups_per_core, K_padded, 4*TILE_SIZE)`` in TILE_LAYOUT.

        The per-core shard map is derived internally from ``K`` (hidden_size)
        and ``N`` (intermediate_size) via ``get_weight_core_shard_maps``.
        )doc",
        &ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute,
        nb::arg("tt_w0").noconvert(),
        nb::arg("tt_w1").noconvert(),
        nb::kw_only(),
        nb::arg("L"),
        nb::arg("E"),
        nb::arg("K"),
        nb::arg("N"));

    ttnn::bind_function<"prepare_w2_tensor_for_moe_compute", "ttnn.experimental.">(
        mod,
        R"doc(
        Pack W2 into the ring-rotated per-core layout the MoE kernel reads.
        Output local shape:
        ``(num_cores, L, E, w2_groups_per_core, N_padded, 4*TILE_SIZE)`` in TILE_LAYOUT.

        The per-core shard maps are derived internally from ``K`` (hidden_size)
        and ``N`` (intermediate_size) via ``get_weight_core_shard_maps``.
        )doc",
        &ttnn::experimental::prepare_w2_tensor_for_moe_compute,
        nb::arg("tt_w2").noconvert(),
        nb::kw_only(),
        nb::arg("L"),
        nb::arg("E"),
        nb::arg("N"),
        nb::arg("K"));

    ttnn::bind_function<"prepare_w0_w1_tensor_with_bias", "ttnn.experimental.">(
        mod,
        R"doc(
        Bias-aware W0/W1 packer. Concatenates kernel-format bias tiles after
        weight tiles along K, then delegates to
        ``prepare_w0_w1_tensor_for_moe_compute``.

        Bias inputs are PyTorch-format ``(L, E, N)``; they get expanded to a
        ``(L, E, TILE_SIZE, N)`` tile with only row 0 populated before concat.
        )doc",
        &ttnn::experimental::prepare_w0_w1_tensor_with_bias,
        nb::arg("tt_w0").noconvert(),
        nb::arg("tt_w1").noconvert(),
        nb::arg("tt_b0").noconvert(),
        nb::arg("tt_b1").noconvert(),
        nb::kw_only(),
        nb::arg("L"),
        nb::arg("E"),
        nb::arg("K"),
        nb::arg("N"));

    ttnn::bind_function<"prepare_w2_tensor_with_bias", "ttnn.experimental.">(
        mod,
        R"doc(
        Bias-aware W2 packer. Weight tiles get ring-rotated as usual; the bias
        tile row is column-sharded and concatenated along N **without**
        rotation, then N is padded to a multiple of BLOCK_TILES_H tiles.

        Bias input is PyTorch-format ``(L, E, K)``; it gets expanded to a
        ``(L, E, TILE_SIZE, K)`` tile with only row 0 populated.
        )doc",
        &ttnn::experimental::prepare_w2_tensor_with_bias,
        nb::arg("tt_w2").noconvert(),
        nb::arg("tt_b2").noconvert(),
        nb::kw_only(),
        nb::arg("L"),
        nb::arg("E"),
        nb::arg("N"),
        nb::arg("K"));

    ttnn::bind_function<"quantize_weights_via_host", "ttnn.experimental.">(
        mod,
        R"doc(
        Round-trip a device tensor through host to change its dtype and
        re-upload it under the supplied memory config. Used to quantize the
        packed MoE weight tensors to ``bfloat4_b`` on the DRAM-sharded mem
        config the kernel consumes.
        )doc",
        &ttnn::experimental::quantize_weights_via_host,
        nb::arg("device_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("dtype"),
        nb::arg("memory_config"));
}
}  // namespace ttnn::operations::experimental::ccl
