// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "high_bw_all_gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather::detail {

void bind_experimental_high_bw_all_gather_operation(nb::module_& mod) {
    ttnn::bind_function<"high_bw_all_gather", "ttnn.experimental.">(
        mod,
        R"doc(
            Gathers a large row-major or tile-layout DRAM tensor over one device-mesh
            axis, or over a direct-neighbor ring linearized across a complete 2D mesh.
            This is a one-dimensional Fabric line or ring collective. The
            operation uses a native one-hop store-and-forward transport and does not
            provide a composite fallback.

            Fabric2D supports a direct physical line or ring. A Torus may wrap
            ``cluster_axis`` when the axis has a distinct wrap neighbor and every
            ring edge is one physical hop. A size-two torus axis is treated as a
            line because its ordinary and wrap neighbors are the same device.
            Plain ``FABRIC_2D`` uses the direct-line schedule; only a Torus
            configuration that wraps ``cluster_axis`` selects the ring schedule.

            Channel trimming compatibility:
                Channel trimming is selected during Fabric initialization and is shared
                by all CCLs in the workload; it is not configurable per operation. A
                trimming capture must cover every CCL and shape that will run. When
                using a trimming profile with this operation, set
                ``TT_METAL_FABRIC_TRIMMING_OVERRIDE`` to an override that force-enables
                all VC0 sender and receiver channels. This avoids a trim-derived VC0
                fast path that can substantially regress this high-rate, multi-worker
                collective while preserving correctness. See this operation's README
                for the YAML and launch example.

            Args:
                input_tensor: Row-major or tile-layout device tensor in DRAM.
                dim: Tensor dimension along which device shards are concatenated.
                output_tensor: Preallocated persistent output tensor.

            Keyword Args:
                cluster_axis: Device-mesh axis (0 or 1) participating in the
                    one-dimensional collective. Other mesh axes run independent
                    all-gathers. Pass ``None`` to gather across every device in a 2D
                    mesh using a snake Hamiltonian ring. The full-mesh mode requires
                    at least one even mesh dimension, direct links for every edge of
                    a row or column snake, and a tensor sharded over ``dim`` across
                    all mesh devices. The host prefers a row snake, then tries a
                    column snake if the row route cannot close directly.
                num_links: Optional number of Fabric links to use. ``None`` uses every
                    link reported usable across the selected axis, or the minimum
                    discovered across both axes in full-mesh mode. An explicit value
                    must be greater than zero and cannot exceed that discovered count;
                    use ``2`` to keep the same link count across QuietBox, LoudBox, and
                    Galaxy.
                subdevice_id: Subdevice containing the worker cores.
                sub_core_grids: Optional worker-core restriction.
                input_batch_index: Optional batch slot selected from a persistent input cache.
                    When set, input has shape [B, 1, ...], output has batch 1, and only that
                    slot is transported.
                gathered_dim_size: Optional active global gathered extent along ``dim``. The
                    output tensor must still be allocated at its worst-case full gathered size.
                    Each rank writes its active local prefix into that rank's fixed worst-case
                    slot; bytes outside those prefixes are left unchanged. ``gathered_dim_size``
                    is the total valid extent, not a contiguous output prefix: consumers must
                    preserve the fixed per-rank stride when locating every rank's valid data.
                page_bundle_indices: Optional uint16 ROW_MAJOR DRAM table mapping this request's
                    logical local KV pages to physical bundles. When present, ``input_tensor`` is a
                    shared ND-sharded pool shaped
                    ``[physical_bundles*kv_cache_num_layers, 1, kv_cache_page_size, D]`` and the
                    logical gathered source is ``[1, 1, table_length*kv_cache_page_size, D]``.
                    This mode is incompatible with ``input_batch_index`` because the table selects
                    the request.
                kv_cache_page_size: Token rows in each physical KV page. Defaults to 32.
                kv_cache_num_layers: Layers stored in every physical bundle. Defaults to 1.
                kv_cache_layer_idx: Layer selected from every physical bundle. Defaults to 0.
        )doc",
        &high_bw_all_gather,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim"),
        nb::arg("output_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("input_batch_index") = nb::none(),
        nb::arg("gathered_dim_size") = nb::none(),
        nb::arg("page_bundle_indices").noconvert() = nb::none(),
        nb::arg("kv_cache_page_size") = 32,
        nb::arg("kv_cache_num_layers") = 1,
        nb::arg("kv_cache_layer_idx") = 0);
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather::detail
