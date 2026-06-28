// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/matmul_decode/matmul_decode.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul_decode {

void bind_matmul_decode_operation(nb::module_& mod) {
    ttnn::bind_function<"matmul_decode">(
        mod,
        R"doc(matmul_decode(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, partial_width_sharded: bool = False, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Returns the matrix product of two tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied.

        Keyword Args:
            partial_width_sharded (bool, optional): force the partial width-sharded program
                factory, where B is sharded along both K and N and the K-partials are reduced
                across cores. Defaults to False (factory chosen automatically).
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): math fidelity /
                fp32 dest accumulation / approx-mode config for the compute kernel. Defaults to
                None (HiFi4).

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        &ttnn::matmul_decode,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("partial_width_sharded") = false,
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("fused_gelu") = false,
        nb::arg("interleaved_output") = false,
        nb::arg("fused_gelu_approx") = false,
        nb::arg("reshard_input") = false,
        nb::arg("reshard_cores") = 2);

    ttnn::bind_function<"gate_up_matmul_decode">(
        mod,
        R"doc(gate_up_matmul_decode(input_tensor_a: ttnn.Tensor, gate_b: ttnn.Tensor, up_b: ttnn.Tensor, *, dtype: Optional[ttnn.DataType] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None, fused_gelu_approx: bool = False, reshard_input: bool = False, reshard_cores: int = 2) -> ttnn.Tensor

        Fused GeGLU gate+up projection: ONE gather of A, TWO partial-width-sharded weights, ONE
        output. Returns hid = gelu(A @ gate_w) * (A @ up_w). gate_b and up_b are partial-width-
        sharded resident-L1 weights laid out on the SAME core grid. Replaces two separate
        matmul_decode(partial_width_sharded, reshard_input) calls + the eltwise multiply -- sharing
        the x-gather, halving the cross-core reduce/dispatch, and folding the GeGLU multiply in.

        Args:
            input_tensor_a (ttnn.Tensor): activation A (gathered once).
            gate_b (ttnn.Tensor): partial-width-sharded gate weight.
            up_b (ttnn.Tensor): partial-width-sharded up weight (same geometry as gate_b).

        Keyword Args:
            dtype (ttnn.DataType, optional): output dtype. Defaults to A's dtype.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): math fidelity config.
            fused_gelu_approx (bool, optional): tanh-approx (True) vs exact-erf gelu for the gate.
            reshard_input (bool, optional): reader reshards A internally. Required True.
            reshard_cores (int, optional): number of A-reshard sender cores. Defaults to 2.

        Returns:
            ttnn.Tensor: hid = gelu(A @ gate_w) * (A @ up_w), width-sharded across N_blocks cores.
        )doc",
        &ttnn::gate_up_matmul_decode,
        nb::arg("input_tensor_a"),
        nb::arg("gate_b"),
        nb::arg("up_b"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("fused_gelu_approx") = false,
        nb::arg("reshard_input") = false,
        nb::arg("reshard_cores") = 2);
}

}  // namespace ttnn::operations::matmul_decode
