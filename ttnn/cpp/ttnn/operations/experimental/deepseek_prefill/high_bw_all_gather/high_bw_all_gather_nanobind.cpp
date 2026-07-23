// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "high_bw_all_gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather::detail {

void bind_experimental_high_bw_all_gather_operation(nb::module_& mod) {
    ttnn::bind_function<"high_bw_all_gather", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Gathers a large row-major or tile-layout DRAM tensor over one required
            device-mesh axis. This is a one-dimensional direct-neighbor Fabric line or
            ring collective: it does not gather across both axes of a 2D mesh. The
            operation uses a native one-hop store-and-forward transport and does not
            provide a composite fallback.

            Fabric2D supports a direct physical line or ring. A Torus may wrap
            ``cluster_axis`` when every ring edge is a direct physical neighbor.
            Fabric handles a size-two torus dimension without applying torus
            bubble flow control to its collapsed ordinary/wrap connection.
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
                cluster_axis: Required device-mesh axis (0 or 1) participating in the
                    one-dimensional collective. The selected axis must contain at least
                    two devices; other mesh axes run independent all-gathers.
                subdevice_id: Subdevice containing the worker cores.
                sub_core_grids: Optional worker-core restriction.
        )doc",
        &high_bw_all_gather,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim"),
        nb::arg("output_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather::detail
