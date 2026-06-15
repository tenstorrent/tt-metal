// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <cstdint>

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
        R"doc(matmul_decode(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, partial_width_sharded: bool = False, dtype: Optional[ttnn.DataType] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None) -> ttnn.Tensor

        Returns the matrix product of two tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied.

        Keyword Args:
            partial_width_sharded (bool, optional): force the partial width-sharded program
                factory, where B is sharded along both K and N and the K-partials are reduced
                across cores. Defaults to False (factory chosen automatically).
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel
                configuration for the matmul_decode operation. Resolves (mirroring ttnn.matmul)
                the per-factory math_fidelity / fp32_dest_acc_en / math_approx_mode. Defaults to
                None, which resolves to math_fidelity=HiFi4 and fp32_dest_acc_en=False (fp32 DST
                accumulation is OPT-IN: pass a config with fp32_dest_acc_en=True to enable the
                higher-precision K-reduction at the cost of device time).
            out_subblock_h (int, optional): explicit fat-fill M-rows per matmul_block (the systolic
                rt_dim). When None the factory auto-derives via the ported native get_subblock_sizes
                (out_w-only unless M-fill is enabled). out_subblock_h>1 (M-fill) requires the
                A-relayout. Defaults to None (auto).
            out_subblock_w (int, optional): explicit fat-fill N-cols per matmul_block (ct_dim).
                Defaults to None (auto).
            in0_block_w (int, optional): K-tiles per inner matmul_block step (K-reuse / fewer
                invocations on large-K shapes). Defaults to 1 (byte-identical to the shipped path).
            k_stream (bool, optional): enable the WIDTH-temporal stream_k codepath (double-buffered
                per-K-slice gather + on-core fp32 K-accumulation) for large-K shapes that bust the
                one-shot full-A gather. When False the full-K one-shot path runs. Defaults to False.
                NOTE: the substring stream_k in this docstring is the capability probe the blocked
                wrapper uses to detect temporal support.
            k_slice_tiles (int, optional): K-tiles per temporal slice when k_stream is True (0 ==
                auto-derive the largest CBCAP-fitting divisor of K_tiles). Defaults to 0.

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
        nb::arg("out_subblock_h") = nb::none(),
        nb::arg("out_subblock_w") = nb::none(),
        nb::arg("in0_block_w") = static_cast<uint32_t>(1),
        // Python-facing kwarg is "stream_k" (matches the blocked wrapper's call site +
        // the docstring capability probe). Internally threaded as the k_stream attr.
        nb::arg("stream_k") = false,
        nb::arg("k_slice_tiles") = static_cast<uint32_t>(0));
}

}  // namespace ttnn::operations::matmul_decode
