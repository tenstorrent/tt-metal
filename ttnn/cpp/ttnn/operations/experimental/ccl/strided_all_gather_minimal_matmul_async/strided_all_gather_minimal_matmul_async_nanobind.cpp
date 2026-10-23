// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_minimal_matmul_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_strided_all_gather_minimal_matmul_async(nb::module_& mod) {
    nb::enum_<ttnn::experimental::prim::MMSignalAggregatorMode>(mod, "MMSignalAggregatorMode")
        .value("Auto", ttnn::experimental::prim::MMSignalAggregatorMode::Auto)
        .value("On", ttnn::experimental::prim::MMSignalAggregatorMode::On)
        .value("Off", ttnn::experimental::prim::MMSignalAggregatorMode::Off);

    ttnn::bind_function<"strided_all_gather_minimal_matmul_async", "ttnn.experimental.">(
        mod,
        R"doc(strided_all_gather_minimal_matmul_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`topology` (ttnn.Topology): Communication topology for the all-gather. Defaults to `ttnn.Topology.Ring`.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`fused_activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`fused_ternary_input_a` (Optional[ttnn.Tensor]): addcmul residual/base tensor (added to the result).
            * :attr:`fused_ternary_input_b` (Optional[ttnn.Tensor]): addcmul multiplier/gate tensor; a single tile-row broadcasts across M.
            * :attr:`fused_ternary_scalar` (Optional[float]): addcmul scale; output = a + scalar * matmul_out * b. Requires both a and b.
            * :attr:`chunks` (int): split the matmul output into this many tensors along N (default 1). Returns [all_gather_output, matmul_chunk_0, ..., matmul_chunk_{chunks-1}]. N must be divisible by chunks.
            * :attr:`mm_signal_aggregator_mode` (ttnn.MMSignalAggregatorMode): whether the all-gather signals the matmul through per-direction aggregator cores. These cost one worker core per direction on top of `num_links * (num_workers_per_link + 1) * 2` mux/worker cores, all placed from `strided_all_gather_core_grid_offset`. `Auto` (default) uses them when they fit and otherwise falls back to reader-signaled matmul with a warning, `On` requires them, `Off` never uses them.
            * :attr:`fuse_swiglu` (bool): If True, applies SwiGLU fused into the matmul: the weight's N columns
              are interpreted as a tile-pair-interleaved [gate|up] layout — column tile 2p is the gate and 2p+1
              the up projection for each pair p (``models.tt_dit.utils.tensor.prepare_for_fused_swiglu`` produces
              this layout). The op computes silu(gate) * up, so the matmul output width is N/2. The bias (if
              provided) must use the same column layout. N must be divisible by 2*32, and by 2*32*chunks when
              chunking. Mutually exclusive with fused_activation and the fused ternary (addcmul) inputs.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.strided_all_gather_minimal_matmul_async(tensor, weight_tensor, dim=0, (0, 0))

        )doc",
        &ttnn::experimental::strided_all_gather_minimal_matmul_async,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("persistent_output_buffer"),
        nb::arg("dim"),
        nb::arg("multi_device_global_semaphore"),
        nb::arg("strided_all_gather_core_grid_offset"),
        nb::kw_only(),
        nb::arg("num_links") = 1,
        nb::arg("memory_config_ag") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("config") = nb::none(),
        nb::arg("memory_config_mm") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("num_buffers_per_channel") = nb::none(),
        nb::arg("read_local_slice_from_input") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("chunks") = 1,
        nb::arg("mm_signal_aggregator_mode") = nb::cast(ttnn::experimental::prim::MMSignalAggregatorMode::Auto),
        nb::arg("fuse_swiglu") = false);
}

}  // namespace ttnn::operations::experimental::ccl
