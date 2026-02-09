// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_matmul_addcmul_fused_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "dit_minimal_matmul_addcmul_fused.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer {

void bind_dit_minimal_matmul_addcmul_fused(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::dit_minimal_matmul_addcmul_fused,
        R"doc(
        dit_minimal_matmul_addcmul_fused(matmul_input_tensor, matmul_weight_tensor, scalar, addcmul_input_tensor1, addcmul_input_tensor2, bias_tensor=None, *, config=None, memory_config=None, dtype=None, compute_kernel_config=None)

        Experimental fused operation combining minimal_matmul and addcmul for improved performance in DiT transformer blocks.
        This operation is designed to optimize the common pattern: output = input1 + (scalar * matmul(input, weight) * input2)

        **Implementation:**
        Delegates to minimal_matmul with fused addcmul (ternary) parameters; addcmul is computed inline in the matmul kernels.

        **Intended Mathematical Operation:**

        .. math::
            \text{{matmul\_output}} = \text{{minimal\_matmul}}(\text{{matmul\_input\_tensor}}, \text{{matmul\_weight\_tensor}})

            \text{{output}} = \text{{addcmul\_input\_tensor1}} + (\text{{scalar}} \times \text{{matmul\_output}} \times \text{{addcmul\_input\_tensor2}})

        This is equivalent to:
            intermediate = minimal_matmul(matmul_input_tensor, matmul_weight_tensor, bias_tensor)
            output = addcmul(addcmul_input_tensor1, intermediate, addcmul_input_tensor2, value=scalar)

        Parameters
        ----------
        matmul_input_tensor : ttnn.Tensor
            Activation/input matrix for matmul (A in A @ B).
            - Layout: TILE (required).
            - Device: must be on device and allocated in a device buffer.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}.
            - Shape: [..., M, K]. Upper dimensions are broadcast over rows (folded into M).

        matmul_weight_tensor : ttnn.Tensor
            Weight matrix for matmul (B in A @ B).
            - Layout: TILE (required).
            - Device: same device as `matmul_input_tensor`.
            - DType: must match `matmul_input_tensor` dtype.
            - Shape: [..., K, N] with no batching; all leading dimensions must be 1.

        scalar : float
            Scalar constant multiplier for the addcmul operation.
            Typically 1.0 in DiT transformer blocks.

        addcmul_input_tensor1 : ttnn.Tensor
            Residual/base for addcmul.
            - Layout: TILE (required).
            - Device: same device as matmul tensors.
            - Shape: [..., M, N] - must match matmul output shape.

        addcmul_input_tensor2 : ttnn.Tensor
            Gate/multiplier tensor for addcmul (broadcast like bias).
            - Layout: TILE (required).
            - Device: same device as matmul tensors.
            - Shape: [..., 1, N] - broadcast row across all M rows.

        bias_tensor : Optional[ttnn.Tensor], default: None
            Optional row-broadcast bias added to the matmul result.
            - Layout: TILE (required if provided).
            - Device: same device as inputs.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}.
            - Shape: [..., N] where all dims except the last are 1.

        Keyword Args
        ------------
        config : Optional[MinimalMatmulConfig], default: None
            Execution configuration for the matmul in tile units. If omitted, defaults are selected.
            Fields (all in tiles):
            - M_block_size: Number of output tiles along M per block.
            - K_block_size: Number of K tiles per block.
            - N_block_size: Number of output tiles along N per block.
            - subblock_h: Sub-block height for compute kernel.
            - subblock_w: Sub-block width for compute kernel.
            - compute_with_storage_grid_size: Core grid (x, y) to run the op on.

        memory_config : Optional[ttnn.MemoryConfig], default: None
            Memory configuration for the output tensor.
            If not provided, inherits from `matmul_input_tensor`.

        dtype : Optional[ttnn.DataType], default: None
            Data type of the output tensor.
            If not provided, inherits from `matmul_input_tensor`.

        compute_kernel_config : Optional[ttnn.operations.core.compute_kernel.DeviceComputeKernelConfig], default: None
            Compute kernel configuration. If omitted, defaults are selected
            (e.g., MathFidelity::HiFi2, fp32 accumulation enabled).

        Returns
        -------
        ttnn.Tensor
            Output tensor with shape [..., M, N], TILE layout, and specified dtype.
            Fused result: addcmul_input_tensor1 + (scalar * matmul_output * addcmul_input_tensor2).

        Shape Semantics
        ---------------
        - matmul_input_tensor: [..., M, K]
        - matmul_weight_tensor: [..., K, N]
        - addcmul_input_tensor1: [..., M, N] (full output shape)
        - addcmul_input_tensor2: [..., 1, N] (broadcast like bias)
        - output: [..., M, N]

        Limitations & Requirements
        --------------------------
        - All tensors must be on the same device and allocated in device buffers.
        - All tensors must be in TILE layout.
        - Supported dtypes: BF16, BF8_B, BF4_B, FLOAT32.
        - Weight and bias (if present) must have 1 in all leading dimensions.
        - Activation may have arbitrary upper dimensions (folded into M).
        - addcmul is fused and computed inline with matmul.

        Use Case
        --------
        This operation is designed for DiT (Diffusion Transformer) models like Wan2.2, where the pattern
        `base_value + gate * feedforward(x)` is common. This fusion eliminates intermediate tensor writes
        and reads, improving performance.

        Typical usage in Wan2.2 transformer block:
            output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                input_normed,          # input after normalization [M, K]
                ff_weight,             # feedforward weight [K, N]
                1.0,                   # scalar multiplier
                gate_tensor,           # gate from timestep embedding [M, N]
                base_value             # broadcast base value [1, N]
            )

        Notes on Implementation
        -----------------------
        - addcmul is computed inline in minimal_matmul kernels (no intermediate matmul tensor write/read).
        - Performance and memory footprint are sensitive to block sizes and subblock shapes.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::dit_minimal_matmul_addcmul_fused)& self,
               const ttnn::Tensor& matmul_input_tensor,
               const ttnn::Tensor& matmul_weight_tensor,
               float scalar,
               const ttnn::Tensor& addcmul_input_tensor1,
               const ttnn::Tensor& addcmul_input_tensor2,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<const DataType> dtype,
               std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    matmul_input_tensor,
                    matmul_weight_tensor,
                    scalar,
                    addcmul_input_tensor1,
                    addcmul_input_tensor2,
                    bias_tensor,
                    config,
                    memory_config,
                    dtype,
                    compute_kernel_config);
            },
            nb::arg("matmul_input_tensor"),
            nb::arg("matmul_weight_tensor"),
            nb::arg("scalar"),
            nb::arg("addcmul_input_tensor1"),
            nb::arg("addcmul_input_tensor2"),
            nb::arg("bias_tensor") = nb::none(),
            nb::kw_only(),
            nb::arg("config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
