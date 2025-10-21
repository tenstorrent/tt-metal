// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_pybind.hpp"

#include <optional>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "minimal_matmul.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::minimal_matmul::detail {

void py_bind_minimal_matmul(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::experimental::minimal_matmul,
        R"doc(
        minimal_matmul(input_tensor, weight_tensor, bias_tensor=None, *, fused_activation=None, config=None, memory_config=None, compute_kernel_config=None)

        Experimental, high-performance matrix multiply (A @ B [+ bias]) with optional fused activation.
        This op expects TILE layout tensors on device and operates in tile units internally. It is designed
        to be minimal and fast, with a small set of required constraints explicitly validated at runtime.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation/input matrix A.
            - Layout: TILE (required).
            - Device: must be on device and allocated in a device buffer.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}.
            - Shape: [..., M, K] with no batching; all leading dimensions (dims < -2) must be 1.

        weight_tensor : ttnn.Tensor
            Weight matrix B.
            - Layout: TILE (required).
            - Device: same device as `input_tensor`; must be allocated in a device buffer.
            - DType: must match `input_tensor` dtype (same set of supported dtypes).
            - Shape: [..., K, N] with no batching; all leading dimensions (dims < -2) must be 1.

        bias_tensor : Optional[ttnn.Tensor], default: None
            Optional row-broadcast bias added to the matmul result before the optional activation.
            - Layout: TILE (required if provided).
            - Device: same device as inputs; must be allocated in a device buffer.
            - DType: one of {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}.
            - Shape: [..., N] where all dims except the last are 1. The last dim must equal N.

        fused_activation : Optional[ttnn.operations.eltwise.unary.common.UnaryWithParam], default: None
            Optional fused unary activation applied per output tile during packing. See ttnn unary utilities
            for supported ops and parameters. Typical examples include relu/gelu/etc. If provided, it is applied
            after bias (if any) and before the tile is written out.

        config : Optional[MinimalMatmulConfig], default: None
            Execution configuration in tile units. If omitted, reasonable defaults are selected based on tensor
            sizes and kernel flags.
            Fields (all values are in tiles):
            - M_block_size: Number of output tiles along M per block.
            - K_block_size: Number of K tiles per block (inner dimension chunk).
            - N_block_size: Number of output tiles along N per block.
            - subblock_h: Sub-block height (M dimension) used by compute kernel within a block.
            - subblock_w: Sub-block width (N dimension) used by compute kernel within a block.
            - compute_with_storage_grid_size: Core grid (x, y) to run the op on.

            Required invariants:
            - All block/subblock sizes must be > 0.
            - M_block_size % subblock_h == 0.
            - N_block_size % subblock_w == 0.
            - Grid size must be within the device grid; typically at least 2x2 is recommended.
            - Internally, K is rounded up to a multiple of K_block_size as needed (zero-padded reads).

            Defaults behavior (when `config` is None):
            - Block sizes and subblock sizes are chosen automatically; subblocks are typically 2x2, and may switch
              to 2x4 or 4x2 depending on aspect ratio and accumulation mode to balance NOC/dataflow.
            - The core grid defaults to the device compute-with-storage grid.

        memory_config : Optional[ttnn.MemoryConfig], default: None
            Memory configuration for the output tensor. If not provided, the output inherits the memory configuration
            of `input_tensor`. The output is produced in TILE layout.

        compute_kernel_config : Optional[ttnn.operations.core.compute_kernel.DeviceComputeKernelConfig], default: None
            Compute kernel configuration. If omitted, defaults are selected via `init_device_compute_kernel_config`
            (e.g., MathFidelity::HiFi2, fp32 accumulation enabled, packer accumulation enabled).

        Returns
        -------
        ttnn.Tensor
            Output tensor with shape [..., M, N], TILE layout, and the same dtype as `input_tensor`.

        Shape Semantics
        ----------------
        - input_tensor: [..., M, K]
        - weight_tensor: [..., K, N]
        - bias_tensor (optional): [..., N] (row-broadcast)
        - output: [..., M, N]
        Note: All leading dims (dims < -2) are required to be 1 (no batching). Tensors are read/written in tile units;
        if logical sizes are not tile-aligned, padding is handled internally (reads fill zeros; writes skip outside
        logical bounds).

        Limitations & Requirements
        --------------------------
        - All tensors must be on the same device and allocated in device buffers.
        - All tensors must be in TILE layout (sharded tensors must be tile-aligned at shard boundaries).
        - Supported dtypes for inputs: BF16, BF8_B, BF4_B. Bias (if present)
          must be one of the supported dtypes. The dtype of the output is the same as the dtype of the inputs.
        - No implicit transpose flags are supported; provide `weight_tensor` with logical shape [..., K, N].
        - No batching: all leading dimensions for inputs and bias must be 1.
        - Performance and memory footprint are sensitive to block sizes and subblock shapes. Providing non-sensible
          values in `config` may degrade performance. Defaults are generally a good starting point.

        Notes on Implementation
        -----------------------
        - Data movement reads A in MxK blocks and B in KxN blocks with serpentine ordering and reuse across
          subblocks to reduce NOC pressure; writes are deferred to reduce congestion.
        - K is processed in blocks of size `K_block_size`, with zero-padding as needed when K is not a multiple.
        - If `fused_activation` is provided, it is applied per tile just before packing to the output buffer.

        Example
        -------
        >>> import ttnn
        >>> a = ...  # TILE tensor with shape [M, K], dtype=ttnn.bfloat16, on device
        >>> b = ...  # TILE tensor with shape [K, N], same dtype/device as `a`
        >>> bias = ...  # Optional TILE tensor with shape [N]
        >>> y = ttnn.experimental.minimal_matmul(
        ...     input_tensor=a,
        ...     weight_tensor=b,
        ...     bias_tensor=bias,
        ...     fused_activation=(ttnn.UnaryOpType.GELU, False),
        ...     config=ttnn.MinimalMatmulConfig(
        ...         M_block_size=8,
        ...         K_block_size=8,
        ...         N_block_size=8,
        ...         subblock_h=2,
        ...         subblock_w=2,
        ...         compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        ...     ),
        ... )
        >>> y.shape  # [M, N]
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::minimal_matmul)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const std::optional<unary::UnaryWithParam>& fused_activation,
               const std::optional<const MinimalMatmulConfig>& config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(
                    input_tensor,
                    weight_tensor,
                    bias_tensor,
                    fused_activation,
                    config,
                    memory_config,
                    compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("fused_activation") = std::nullopt,
            py::arg("config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    auto py_minimal_matmul_config = py::class_<MinimalMatmulConfig>(
                                        module,
                                        "MinimalMatmulConfig",
                                        R"doc(
                            Configuration for the MinimalMatmul operation.
                            )doc")
                                        .def(py::init<>())
                                        .def(
                                            py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, CoreCoord>(),
                                            py::kw_only(),
                                            py::arg("M_block_size") = 1,
                                            py::arg("K_block_size") = 1,
                                            py::arg("N_block_size") = 1,
                                            py::arg("subblock_h") = 1,
                                            py::arg("subblock_w") = 1,
                                            py::arg("compute_with_storage_grid_size") = CoreCoord{1, 1});

    py_minimal_matmul_config.def_readwrite("M_block_size", &MinimalMatmulConfig::M_block_size, "");
    py_minimal_matmul_config.def_readwrite("K_block_size", &MinimalMatmulConfig::K_block_size, "");
    py_minimal_matmul_config.def_readwrite("N_block_size", &MinimalMatmulConfig::N_block_size, "");
    py_minimal_matmul_config.def_readwrite("subblock_h", &MinimalMatmulConfig::subblock_h, "");
    py_minimal_matmul_config.def_readwrite("subblock_w", &MinimalMatmulConfig::subblock_w, "");
    py_minimal_matmul_config.def_readwrite(
        "compute_with_storage_grid_size", &MinimalMatmulConfig::compute_with_storage_grid_size, "");

    py_minimal_matmul_config.def(
        "__repr__", [](const MinimalMatmulConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
