// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_split_nanobind.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "minimal_matmul_split.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::minimal_matmul::detail {

namespace nb = nanobind;

void bind_minimal_matmul_split(nb::module_& mod) {
    ttnn::bind_function<"minimal_matmul_split", "ttnn.experimental.">(
        mod,
        R"doc(
        minimal_matmul_split(input_tensor, weight_tensor, *, chunks=3, dim=-1, bias_tensor=None, fused_activation=None, config=None, memory_config=None, dtype=None, compute_kernel_config=None, fuse_swiglu=False)

        Experimental, high-performance matrix multiply (A @ B [+ bias]) with output splitting along the last dimension.
        This op performs a matmul and splits the result into `chunks` separate output tensors, fusing the common
        Q/K/V projection pattern: Q, K, V = ttnn.minimal_matmul_split(qkv, qkv_proj, chunks=3)

        This is equivalent to:
        ```python
        outs = torch.matmul(qkv, qkv_proj)
        if bias is not None:
            outs = outs + bias
        Q, K, V = torch.chunk(outs, chunks=3, dim=-1)
        ```

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation/input matrix A.
            - Layout: TILE (required).
            - Device: must be on device and allocated in a device buffer.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B, DataType::FLOAT32}.
            - Shape: [..., M, K]. Upper (leading) dimensions are broadcast over rows (folded into M).

        weight_tensor : ttnn.Tensor
            Weight matrix B.
            - Layout: TILE (required).
            - Device: same device as `input_tensor`; must be allocated in a device buffer.
            - DType: must match `input_tensor` dtype (same set of supported dtypes).
            - Shape: [..., K, N] with no batching; all leading dimensions (dims < -2) must be 1.
            - N must be divisible by `chunks`, and N/chunks must be a multiple of 32 (tile-aligned).

        chunks : int, default: 3
            Number of output tensors to split into. Any number of chunks is supported as long as N is divisible by `chunks`.
            Output dimension N will be split into `chunks` equal parts along dimension `dim`.

        dim : int, default: -1
            Dimension along which to split. **Currently only dim=-1 is supported.**

        bias_tensor : Optional[ttnn.Tensor], default: None
            Optional row-broadcast bias added to the full matmul result before splitting.
            - Layout: TILE (required if provided).
            - Device: same device as inputs; must be allocated in a device buffer.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B, DataType::FLOAT32}.
            - Shape: [..., N] where all dims except the last are 1. The last dim must equal N (full output width).
            - The bias is applied to the full output before splitting into chunks.

        fused_activation : Optional[ttnn.operations.eltwise.unary.common.UnaryWithParam], default: None
            Optional fused unary activation applied per output tile during packing. Applied after bias (if any)
            and before writing to output buffers.

        fuse_swiglu : bool, default: False
            If True, applies SwiGLU activation fused into the matmul: the weight tensor's N columns are
            interpreted as a tile-pair-interleaved [gate|up] layout — column tile 2p is the gate and 2p+1 is
            the up projection for each pair p (``models.tt_dit.utils.tensor.prepare_for_fused_swiglu`` can be used to produce this layout).
            Each output chunk of width N/chunks has SwiGLU applied, so the output width
            per chunk is N/chunks/2. The bias (if provided) must use the same column layout. N/chunks must be
            divisible by 2*32. Mutually exclusive with fused_activation.

        config : Optional[MinimalMatmulConfig], default: None
            Execution configuration in tile units. If omitted, reasonable defaults are selected based on tensor
            sizes and kernel flags. See ttnn.experimental.minimal_matmul documentation for config details.

        memory_config : Optional[ttnn.MemoryConfig], default: None
            Memory configuration for all output tensors. If not provided, outputs inherit the memory configuration
            of `input_tensor`. All outputs share the same memory configuration.

        dtype : Optional[ttnn.DataType], default: None
            Data type of all output tensors. If not provided, outputs inherit the data type of `input_tensor`.
            All outputs share the same dtype.

        compute_kernel_config : Optional[ttnn.operations.core.compute_kernel.DeviceComputeKernelConfig], default: None
            Compute kernel configuration. If omitted, defaults are selected via `init_device_compute_kernel_config`.

        Returns
        -------
        List[ttnn.Tensor]
            List of `chunks` output tensors, each in TILE layout with the specified dtype.
            - fuse_swiglu=False: each tensor has shape [..., M, N/chunks]. E.g For chunks=3, returns [output0, output1, output2].
            - fuse_swiglu=True: each tensor has shape [..., M, N/chunks/2] (SwiGLU halves the output width per chunk).

        Constraints
        -----------
        - dim must be -1 (last dimension)
        - N must be divisible by chunks (N % chunks == 0)
        - N/chunks must be a multiple of 32 (tile-aligned: (N/chunks) % 32 == 0)
        - If fuse_swiglu=True: N/chunks must additionally be divisible by 2 (each chunk holds interleaved gate/up pairs);
          fuse_swiglu is mutually exclusive with fused_activation
        - All tensors must be on the same device and allocated in device buffers
        - All tensors must be in TILE layout
        - Weight and bias must have 1 in all leading dimensions (dims < -2)
        )doc",
        &ttnn::experimental::minimal_matmul_split,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::kw_only(),
        nb::arg("chunks") = 3,
        nb::arg("dim") = -1,
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("fuse_swiglu") = false);
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
