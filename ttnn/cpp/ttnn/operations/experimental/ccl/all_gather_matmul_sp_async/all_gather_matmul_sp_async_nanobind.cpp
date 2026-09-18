// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_sp_async_nanobind.hpp"

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
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/all_gather_matmul_sp_async.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_all_gather_matmul_sp_async(nb::module_& mod) {
    ttnn::bind_function<"all_gather_matmul_sp_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused sequence-parallel gather-then-multiply on dim 2 of a [B,1,S/T,K] multi-device tensor.

        Returns ``[gathered, mm]``: ``gathered`` is ``all_gather(input, dim=2, cluster_axis)`` ([B,1,S,K]) and
        ``mm = gathered @ weight (+ bias)`` ([B,1,S,N]); the matmul consumes each gathered sequence slice as soon
        as the all-gather delivers it (local slice first, straight from the input). Both tensors are allocated by
        the op. The all-gather workers use the bottom ``ccl_core_rows`` rows of the core grid, the matmul the rest.

        Args:
            * :attr:`input` (ttnn.Tensor): [B,1,S/T,K], TILE, DRAM interleaved, sequence-sharded along ``cluster_axis``.
            * :attr:`weight` (ttnn.Tensor): [1,1,K,N], or [1,1,N,K] with ``transpose_b``.
            * :attr:`cluster_axis` (int): mesh axis to gather along (T = its extent).
            * :attr:`multi_device_global_semaphore` (List[ttnn.GlobalSemaphore]): 2 semaphores, as all_gather_async.

        Keyword Args:
            * :attr:`barrier_semaphore` (ttnn.GlobalSemaphore, optional)
            * :attr:`transpose_b` (bool): weight stored as [N,K]. Defaults to False.
            * :attr:`bias` (ttnn.Tensor, optional): row-broadcast [1,1,1,N].
            * :attr:`num_links` (int, optional): defaults to the links available on ``cluster_axis``.
            * :attr:`topology` (ttnn.Topology): Ring (demoted to Linear when the axis is not wrap-wired) or Linear.
            * :attr:`ccl_core_rows` (int): bottom rows reserved for the all-gather workers. Defaults to 2
              (kDefaultAllGatherMatmulSpCclCoreRows).
            * :attr:`num_workers_per_link` (int, optional): defaults to the largest of 4/2/1 that fits the rows.
            * :attr:`memory_config` (ttnn.MemoryConfig, optional): for both outputs. Defaults to the input's.
            * :attr:`dtype` (ttnn.DataType, optional): matmul output dtype. Defaults to the input's.
            * :attr:`compute_kernel_config` (DeviceComputeKernelConfig, optional): matmul numerics (default HiFi2).
            * :attr:`program_config` (ttnn.MatmulProgramConfig, optional): 2D-mcast, fuse_batch=False, on the
              [B*T,1,S/T,K] view; derived when omitted.
            * :attr:`subdevice_id` (ttnn.SubDeviceId, optional)
        )doc",
        &ttnn::experimental::all_gather_matmul_sp_async,
        nb::arg("input"),
        nb::arg("weight"),
        nb::arg("cluster_axis"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("transpose_b") = false,
        nb::arg("bias") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("ccl_core_rows") = ttnn::experimental::kDefaultAllGatherMatmulSpCclCoreRows,
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none());

    ttnn::bind_function<"all_gather_matmul_sp_ag_schedule", "ttnn.experimental.">(
        mod,
        R"doc(
        Host-side: the matmul schedule all_gather_matmul_sp_async uses on rank ``ring_index`` of ``ring_size``,
        one row per matmul iteration: [in0_idx, out_idx, wait_dir, wait_count, is_local].
        )doc",
        &ttnn::experimental::all_gather_matmul_sp_ag_schedule,
        nb::arg("topology"),
        nb::arg("ring_size"),
        nb::arg("ring_index"),
        nb::arg("batch"));
}

}  // namespace ttnn::operations::experimental::ccl
