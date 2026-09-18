// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_sp_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/matmul_reduce_scatter_sp_async.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_matmul_reduce_scatter_sp_async(nb::module_& mod) {
    ttnn::bind_function<"matmul_reduce_scatter_sp_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused sequence-parallel "multiply-then-scatter" on dim 2:
        ``reduce_scatter(input_tensor @ weight_tensor, dim=2, cluster_axis)``.

        ``input_tensor`` is ``[B, 1, S, K]`` (this rank's K shard), ``weight_tensor`` is ``[1, 1, K, N]`` or
        ``[1, 1, N, K]`` with ``transpose_b=True``. Returns ``[B, 1, S/T, N]`` where ``T`` is the mesh extent along
        ``cluster_axis`` (every device gets its own S/T rows of the sum over the T partial products). The full
        ``[B, 1, S, N]`` partial and the reduce-scatter intermediates are allocated by the op and released on
        return; no caller-owned buffers. No bias (add it after the reduction).

        The matmul computes one (batch, sequence-slice) block per batch iteration and signals the reduce-scatter
        workers after each, so the collective starts while the matmul is still running.

        Args:
            input_tensor (ttnn.Tensor): ``[B, 1, S, K]`` tiled, interleaved DRAM, bfloat16 or float32.
            weight_tensor (ttnn.Tensor): ``[1, 1, K, N]`` (or ``[1, 1, N, K]`` with ``transpose_b``), tiled, interleaved DRAM.
            cluster_axis (int): mesh axis to reduce-scatter over.
            multi_device_global_semaphore (List[ttnn.GlobalSemaphore]): 3 semaphores, as ``reduce_scatter_minimal_async``.

        Keyword Args:
            barrier_semaphore (ttnn.GlobalSemaphore, optional): start-up barrier semaphore.
            transpose_b (bool): weight is ``[1, 1, N, K]``. Defaults to ``False``.
            num_links (int, optional): fabric links to use. Defaults to all links available on ``cluster_axis``.
            topology (ttnn.Topology): ``Ring`` (even T, demoted to ``Linear`` if the axis is not wrap-wired) or ``Linear``.
            ccl_core_rows (int): bottom rows of the compute grid reserved for the reduce-scatter workers; the matmul
                uses the rows above. Defaults to 2 (measured best: 24 cores hold 2 links x 2 directions x (5 workers
                + 1 mux); with T=4, S=2048 the matmul only needs 8 rows anyway).
            num_workers_per_link (int, optional): reduce-scatter workers per direction per link. Defaults to the
                measured best (5 on Ring, 4 on Linear), halved until the worker+mux cores fit the reserved rows.
            memory_config (ttnn.MemoryConfig, optional): memory config of the matmul partial and the output.
            dtype (ttnn.DataType, optional): output dtype. Defaults to the input dtype.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): matmul compute config. Defaults to the
                ``ttnn.matmul`` defaults (HiFi2, L1 accumulation) so fused and unfused results match.
            program_config (ttnn.MatmulMultiCoreReuseMultiCastProgramConfig, optional): override for the derived
                per-slice matmul config (``fuse_batch`` must be ``False``).
            sub_device_id (ttnn.SubDeviceId, optional)
            debug_serialize_reduce_scatter (bool): measurement knob, the reduce-scatter waits for the whole matmul
                (same result, no overlap). Defaults to ``False``.

        Returns:
            ttnn.Tensor: ``[B, 1, S/T, N]``.
        )doc",
        &ttnn::experimental::matmul_reduce_scatter_sp_async,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("transpose_b") = false,
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("ccl_core_rows") = 2,
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("sub_device_id") = nb::none(),
        nb::arg("debug_serialize_reduce_scatter") = false);

    ttnn::bind_function<"matmul_reduce_scatter_sp_rs_first_touch_order", "ttnn.experimental.">(
        mod,
        R"doc(
        Order in which rank ``ring_index``'s reduce-scatter readers first read their local input slices
        (derived from the ring/line reader kernels): a permutation of ``0..ring_size-1`` ending with ``ring_index``.
        This is the per-batch matmul sub-batch order the sequence-parallel slice schedule will use.
        )doc",
        &ttnn::experimental::matmul_reduce_scatter_sp_rs_first_touch_order,
        nb::arg("topology"),
        nb::arg("ring_size"),
        nb::arg("ring_index"));
}

}  // namespace ttnn::operations::experimental::ccl
