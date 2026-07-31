// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "all_gather_regime_a_matmul_async.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail {

void bind_all_gather_regime_a_matmul_async(nb::module_& mod) {
    ttnn::bind_function<"all_gather_regime_a_matmul_async", "ttnn.experimental.">(
        mod,
        R"doc(
        all_gather_regime_a_matmul_async(input_tensor, weight_tensor, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None)

        Experimental DRAM-bandwidth-optimal matrix multiply (A @ B) for low-arithmetic-intensity
        (M << N or N << M) "Regime-A" shapes, with optional fused epilogue. Numerics are FIXED:
        BFLOAT16 in/out, HiFi2 math, FP32 dest accumulation, DRAM-interleaved output (there are no
        dtype / memory_config / compute_kernel_config arguments).

        Fusions (applied at the output/compute stage; for split-K they run exactly once after reduction):
          - bias:       Y = A@B + bias
          - activation: Y = activation(A@B + bias)                (bias applied before activation)
          - addcmul:    Y = residual + scalar*(A@B + bias)*gate   (activation and addcmul are exclusive)

        The activation A ([.., M, K]) is DRAM interleaved. The weight B ([.., K, N]) must be DRAM
        WIDTH_SHARDED across 8 banks — build its MemoryConfig with
        ``ttnn.create_regime_a_weight_memory_config``. Output is [.., M, N] in TILE layout.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation A. TILE layout, BFLOAT16, on device. Shape [.., M, K] (leading dims must be 1).
        weight_tensor : ttnn.Tensor
            Weight B. TILE layout, BFLOAT16, on device, DRAM WIDTH_SHARDED. Shape [.., K, N].
        config : Optional[RegimeAMatmulConfig], default: None
            Manual execution config. None => auto-select via the FLUX/LTX picker.
        bias_tensor : Optional[ttnn.Tensor], default: None
            Row-broadcast bias [.., 1, N] / [.., N], TILE, on device.
        fused_activation : Optional[UnaryWithParam], default: None
            Fused unary activation applied after bias.
        fused_ternary_scalar : Optional[float], default: None
            addcmul scalar. If set, fused_ternary_input_a (residual) and fused_ternary_input_b (gate)
            are required and fused_activation must be None.
        fused_ternary_input_a : Optional[ttnn.Tensor], default: None
            addcmul residual [M, N], BFLOAT16, TILE.
        fused_ternary_input_b : Optional[ttnn.Tensor], default: None
            addcmul gate [1, N] (broadcast) or [M, N] (full), TILE.

        Returns
        -------
        ttnn.Tensor
            Output tensor [.., M, N], TILE layout, BFLOAT16, DRAM interleaved.
        )doc",
        &ttnn::experimental::all_gather_regime_a_matmul_async,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none(),
        nb::arg("multi_device_global_semaphore") = std::vector<GlobalSemaphore>{},
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("cluster_axis") = nb::none());

    ttnn::bind_function<"all_gather_regime_a_matmul_async_split", "ttnn.experimental.">(
        mod,
        R"doc(
        all_gather_regime_a_matmul_async_split(input_tensor, weight_tensor, chunks, dim=-1, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None)

        Output column-split sibling of all_gather_regime_a_matmul_async. Returns `chunks` equal-width [.., M, N/chunks]
        output tensors, written directly (no full-output materialize + slice). Requires dim==-1,
        N % chunks == 0 and N/chunks tile-aligned. All fusions compose with chunking. Fixed numerics
        (BFLOAT16, HiFi2, FP32 acc, DRAM interleaved).

        Returns
        -------
        List[ttnn.Tensor]
            `chunks` output tensors [.., M, N/chunks], TILE layout.
        )doc",
        &ttnn::experimental::all_gather_regime_a_matmul_async_split,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("chunks"),
        nb::arg("dim") = -1,
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none(),
        nb::arg("multi_device_global_semaphore") = std::vector<GlobalSemaphore>{},
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("cluster_axis") = nb::none());

    // NOTE: RegimeAMatmulConfig and create_regime_a_weight_memory_config are deliberately NOT
    // re-registered here. They are already bound by ttnn.experimental.regime_a_matmul and are shared
    // verbatim by this op; binding them twice collides on the same Python names at import time.
}

}  // namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail
