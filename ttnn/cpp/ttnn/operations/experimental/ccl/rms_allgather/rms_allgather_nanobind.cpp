// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_fused_rms_minimal(nb::module_& mod) {
    ttnn::bind_function<"fused_rms_minimal">(
        mod,
        R"doc(
            Fused distributed RMS normalization. Fuses the pre-RMS reduction, the cross-device
            all-gather of the partial statistics, post-RMS normalization, optional residual add,
            gamma (weight) scaling, and output resharding into a single op. The computation is
            width-sharded across a core grid: each core owns a slice of the hidden dimension and
            the per-shard statistics are gathered across cores and devices.

            The op is intentionally sharding-specific and validates the following requirements;
            it does not reshard inputs/outputs internally, so callers (and compilers) must satisfy
            the layout contract:

            - input_tensor: shape (1, 1, M, N) with M <= 32 and N a multiple of 32; TILE layout;
              dtype FLOAT32, BFLOAT16, or BFLOAT8_B. Must be width-sharded in L1 with ROW_MAJOR
              shard orientation, sharded on the same core grid that ``global_semaphore`` was created
              on. Interleaved / DRAM inputs are not accepted (reshard to width-sharded L1 first), and
              Blackhole DRAM is unsupported.
            - weight (gamma): required. ROW_MAJOR layout with 2-D shape (N/32, 32); dtype FLOAT32 or
              BFLOAT16. Passing weight=None is not supported.
            - stats: optional. Backs the op's internal all-gather circular buffer. Prefer omitting it
              (the default), in which case the op allocates its own transient, trace-native L1
              scratch. If provided, it must be a tiled, width-sharded L1 tensor replicated across the
              mesh, with a per-core shard shape of (32, 32 * number of devices on ``cluster_axis``);
              a narrower shard is rejected, since the circular buffer is bound to a single core's
              bank. Its dtype must be consistent with the compute config accumulation (FLOAT32 when
              fp32_dest_acc_en is enabled, else BFLOAT16).
            - memory_config (output): must be a sharded config whose buffer type and memory layout
              match the input's. Interleaved output configs are not accepted; reshard the result
              afterward if a downstream consumer needs interleaved/DRAM.
            - residual_input_tensor (if provided): must be sharded with the same shard spec and memory
              config as input_tensor.
            - persistent_output_tensor: a pre-allocated tiled tensor for the intermediate all-gather,
              per-device shape (32, 32), sharded with a shard shape (32, 32) on core (0, 0).
        )doc",
        &ttnn::fused_rms_minimal,
        nb::arg("input_tensor"),
        nb::arg("program_config"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("global_semaphore"),
        nb::kw_only(),
        // all gather
        nb::arg("persistent_output_tensor") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = ttnn::ccl::Topology::Linear,
        nb::arg("subdevice_id") = nb::none(),
        // common
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        // on pre only
        nb::arg("residual_input_tensor") = nb::none(),
        // on post only
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("stats") = nb::none(),
        nb::arg("use_noc1_only") = false);
}
}  // namespace ttnn::operations::experimental::ccl
