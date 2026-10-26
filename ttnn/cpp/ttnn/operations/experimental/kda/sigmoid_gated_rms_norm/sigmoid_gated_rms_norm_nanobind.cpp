// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_nanobind.hpp"
#include "sigmoid_gated_rms_norm.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail {

void bind_sigmoid_gated_rms_norm(nb::module_& mod) {
    ttnn::bind_function<"sigmoid_gated_rms_norm", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply per-head RMS normalization followed by sigmoid gating.

        For input head ``h``:

            normalized = input / sqrt(mean(input², dim=V) + epsilon)
            output = normalized * weight * sigmoid(gate)

        The operation converts head-first input ``[B*H, T, V]`` into time-first
        output ``[B, T, H*V]`` for the following output projection.

        Args:
            input (ttnn.Tensor): Input tensor ``[B*H, T, V]``. Must be a TILE-layout
                device tensor with FLOAT32 or BFLOAT16 dtype, interleaved or sharded. Sharded, the
                op deals its (row tile, head) items round-robin over the shard cores: an input (and
                activated gate) sharded one ``[1, 32, V]`` item per shard, round-robin, is then read
                from each core's own L1.
            gate (ttnn.Tensor): Sigmoid gate ``[B, T, W]`` with ``W >= gate_column_offset + H*V``.
                Must be an interleaved TILE-layout BFLOAT16 device tensor.
            weight (ttnn.Tensor): Per-value RMSNorm weight ``[V]``. Must be an
                interleaved TILE-layout BFLOAT16 device tensor.
            num_heads (int): Number of heads ``H``. The input leading dimension
                must be divisible by ``H``.

        Keyword Args:
            epsilon (float): Finite positive RMSNorm epsilon. Defaults to ``1e-5``.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.
            output_dtype (ttnn.DataType): Output dtype, either FLOAT32 or BFLOAT16.
                Defaults to FLOAT32.
            gate_column_offset (int): Tile-aligned first gate column; the gate is read from
                columns ``[offset, offset + H*V)``, so a fused projection's gate columns need not
                be sliced out first. Defaults to 0.
            activated_gate (ttnn.Tensor, optional): FLOAT32 ``[B*Ha, T, V]`` (laid out as the input) holding
                sigmoid(gate) of heads ``[0, Ha)``, activated elsewhere. The gate then holds only heads ``[Ha, H)``,
                from ``gate_column_offset``, and the op deals its (row tile, head) items round-robin over its cores,
                so the items that still need a sigmoid spread evenly. An activated gate holding the FP32-accurate
                sigmoid of the same BFLOAT16 gate values, computed in an FP32 DST, gives the same output bit for bit.
                Defaults to None (no head activated).
            prefetch_source (ttnn.Tensor, optional): With ``prefetch_jobs`` and ``prefetch_cores``, an L1 prefetch for
                a later op rides along: once its reads are done, the reader of ``prefetch_cores[j]`` moves job ``j`` of
                the per-device job table (``ttnn/cpp/ttnn/kernel_lib/l1_prefetch_dataflow.hpp``) from this tensor
                into L1 and multicasts it. Every job writes inside ``[prefetch_l1_begin, prefetch_l1_end)``, which
                must be free while the norm runs (checked: above its circular buffers, below every L1 buffer).
            prefetch_jobs (ttnn.Tensor, optional): ROW_MAJOR UINT32 ``[..., jobs, 16]`` job table, one row per job.
            prefetch_cores (List[ttnn.CoreCoord]): norm cores that run a job, in job order.
            prefetch_l1_begin (int), prefetch_l1_end (int): the L1 region the jobs write.

        Returns:
            ttnn.Tensor: A new TILE-layout tensor with shape ``[B, T, H*V]``.

        Note:
            ``T`` and ``V`` must be positive and tile-aligned. All input tensors
            must be allocated on the same device. Inputs are not modified.
        )doc",
        &ttnn::experimental::kda::sigmoid_gated_rms_norm,
        nb::arg("input").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("gate_column_offset") = 0,
        nb::arg("activated_gate") = nb::none(),
        nb::arg("prefetch_source") = nb::none(),
        nb::arg("prefetch_jobs") = nb::none(),
        nb::arg("prefetch_cores") = std::vector<tt::tt_metal::CoreCoord>{},
        nb::arg("prefetch_l1_begin") = 0,
        nb::arg("prefetch_l1_end") = 0);
}

}  // namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail
