// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_prefill_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn/operations/experimental/ccl/all_gather_matmul_prefill/all_gather_matmul_prefill.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_all_gather_matmul_prefill(nb::module_& mod) {
    ttnn::bind_function<"all_gather_matmul_prefill", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused TP all-gather + matmul: every device holds a K shard ``[1, 1, M, K / ranks]`` of in0 along
        ``cluster_axis`` and the whole weight ``[1, 1, K, N]``; returns the gathered in0 times the weight,
        ``[1, 1, M, N]``. The matmul runs ``program_config`` (2D multicast) on its grid, which must leave the worker
        column right of it to the gather; it starts with this device's own K slice and takes each other slice as soon
        as the gather has landed it, so the gather runs beside the matmul.

        Args:
            input_tensor (ttnn.Tensor): BFLOAT16 TILE interleaved ``[1, 1, M, K / ranks]`` shard.
            weight (ttnn.Tensor): TILE interleaved ``[1, 1, K, N]`` weight.
            gathered (ttnn.Tensor): persistent BFLOAT16 TILE DRAM interleaved ``[1, 1, M, K]`` buffer the op fills
                with the gathered in0 (also this device's slice).
            ready_semaphore (ttnn.GlobalSemaphore): persistent, on the gather column's cores, zero before the
                first call.
            slice_counters (list[ttnn.GlobalSemaphore]): ``ranks - 1`` persistent semaphores on the matmul's first
                column, zero before the first call; every call leaves them zero.

        Keyword Args:
            cluster_axis (int): mesh axis the input is sharded over.
            program_config (ttnn.MatmulMultiCoreReuseMultiCastProgramConfig): the matmul's schedule.
            num_links (int): fabric links per direction the gather uses. Defaults to 1.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to HiFi2, FP32 accumulation.
            memory_config (ttnn.MemoryConfig, optional): interleaved output memory; defaults to DRAM.
            dtype (ttnn.DataType, optional): output data type; defaults to the input's.
        )doc",
        &ttnn::experimental::all_gather_matmul_prefill,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("gathered").noconvert(),
        nb::arg("ready_semaphore"),
        nb::arg("slice_counters"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("program_config"),
        nb::arg("num_links") = 1,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
