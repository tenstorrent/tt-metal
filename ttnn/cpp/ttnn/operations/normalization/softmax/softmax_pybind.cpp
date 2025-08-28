// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "softmax.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_softmax_program_config_operation(py::module& module) {
    py::class_<SoftmaxProgramConfig>(module, "SoftmaxProgramConfig").def(py::init<>());

    py::class_<SoftmaxDefaultProgramConfig>(module, "SoftmaxDefaultProgramConfig").def(py::init<>());

    py::class_<SoftmaxShardedMultiCoreProgramConfig>(module, "SoftmaxShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert())
        .def_readwrite("block_w", &SoftmaxShardedMultiCoreProgramConfig::block_w);
}

void bind_normalization_softmax_operation(py::module& module) {
    auto doc =
        R"doc(softmax(input_tensor: ttnn.Tensor, dim: int, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute softmax over :attr:`input_tensor` along :attr:`dim`.

            .. math::
                \sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K

            Args:
                input_tensor (ttnn.Tensor): The input tensor.
                dim (int): The dimension along which to compute softmax.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. If not provided, the memory configuration of the input tensor is used.
                compute_kernel_config (ttnn.ComputeKernelConfig, optional): the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Note:
              Supported data types and layouts by tensor:

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

              .. list-table:: mask (optional)
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE, ROW_MAJOR

              .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - matches input_tensor dtype
                  - TILE

            Limitations:
              - All tensors must be on-device.
              - Unsharded inputs are interleaved; sharded runs require M and K to be divisible by TILE_WIDTH.
              - If a mask is provided:
                - If sharded: mask must have identical padded shape to input.
                - If unsharded and ROW_MAJOR: last padded dim must be TILE_WIDTH; broadcast dims must be 1 except the last two; padded last two dims must align to input's width tiles and TILE_WIDTH.

            Example:
                .. code-block:: python

                  tensor = ttnn.zeros((1, 1, 64, 32), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                  output = ttnn.softmax(tensor, -1)

        )doc";

    using OperationType = decltype(ttnn::softmax);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) {
                return self(input_tensor, dim, memory_config, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim") = -1,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("numeric_stable").noconvert() = false});
}

void bind_normalization_scale_mask_softmax_operation(py::module& module) {
    auto doc =
        R"doc(scale_mask_softmax(input_tensor: ttnn.Tensor, scale: Optional[float] = None, mask: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None, is_causal_mask: Optional[bool] = False, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute fused scale->attention_mask->softmax operation over :attr:`input_tensor` on the last dim.

            Args:
                * :attr:`input_tensor`: the input tensor
                * :attr:`scale`: the scale to be multiplied with input tensor
                * :attr:`mask`: the input mask tensor to be applied to input tensor

            Keyword Args:
                * :attr:`memory_config`: the memory configuration for the output tensor. If not provided, the memory configuration of the input tensor is used.
                * :attr:`is_causal_mask`: determines whether the mask tensor is causal or not. If not provided, non-causal mask will be used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Note:
              Supported data types and layouts by tensor:

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

              .. list-table:: mask
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE, ROW_MAJOR

              .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - matches input_tensor dtype
                  - TILE

            Limitations:
              - All tensors must be on-device
              - When mask is ROW_MAJOR, intermediate dims except the last two must be 1; last dim must equal TILE_WIDTH; width (in tiles) must align to input.
              - Internal block size constraints on width-in-tiles may restrict in-place variants for very large widths.


            Example:
                .. code-block:: python

                  compute_grid_size = device.compute_with_storage_grid_size()
                  fuse_head = 2
                  batch = compute_grid_size.x
                  num_cores_r = compute_grid_size.y

                  input_shape = (batch, num_cores_r, fuse_head * 384, 768)

                  attention_mask_t = ttnn.rand((batch, 1, 1, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  tt_output = ttnn.scale_mask_softmax(
                      input_tensor=input_tensor,
                      scale=1.0,
                      mask=attention_mask_t,
                  )

        )doc";

    using OperationType = decltype(ttnn::scale_mask_softmax);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_mask_softmax,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<float> scale,
               const std::optional<const Tensor>& mask,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const bool is_causal_mask,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) {
                return self(
                    input_tensor, scale, mask, memory_config, is_causal_mask, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("mask").noconvert() = std::nullopt,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("is_causal_mask") = false,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("numeric_stable") = false});
}

void bind_normalization_softmax_in_place_operation(py::module& module) {
    auto doc =
        R"doc(softmax_in_place(input_tensor: ttnn.Tensor, program_config: Optional[SoftmaxProgramConfig], compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute softmax over :attr:`input_tensor` along the last dim, input and output tensor are in-placed on the same L1 address.

            .. math::
                \sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K

            Args:
                * :attr:`input_tensor`: the input tensor

            Keyword Args:
                * :attr:`program_config`: the program configuration for op. If not provided, SoftmaxDefaultProgramConfig is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Note:
              Supported data types and layouts by tensor:

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

              .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - matches input_tensor dtype
                  - TILE

            Limitations:
              - In-place variant is disabled for very wide tensors (when CBs would consume more than 90% of L1) and will fall back to standard softmax.

            Example:
                shape = [1, 1, 32, 32]
                input_tensor = ttnn.rand(shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                output_tensor = ttnn.softmax_in_place(input_tensor)

        )doc";

    using OperationType = decltype(ttnn::softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const SoftmaxProgramConfig& program_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) {
                return self(input_tensor, program_config, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("program_config") = SoftmaxDefaultProgramConfig{},
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("numeric_stable") = false});
}

void bind_normalization_scale_mask_softmax_in_place_operation(py::module& module) {
    auto doc =
        R"doc(scale_mask_softmax_in_place(input_tensor: ttnn.Tensor, scale: Optional[float] = None, mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig], compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute fused scale->attention_mask->softmax over :attr:`input_tensor` along the last dim, input and output tensor are in-placed on the same L1 address.

            Args:
                * :attr:`input_tensor`: the input tensor

            Keyword Args:
                * :attr:`program_config`: the program configuration for op. If not provided, SoftmaxDefaultProgramConfig is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Note:
              Supported data types and layouts by tensor:

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

              .. list-table:: mask
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE, ROW_MAJOR

              .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - matches input_tensor dtype
                  - TILE

            Limitations:
              - All tensors must be on-device
              - Mask broadcasting: for unsharded ROW_MAJOR mask, intermediate dims except the last two must be 1; last dim must equal TILE_WIDTH; width (in tiles) must align to input.
              - Sharded inputs require TILE layout mask with identical padded shape to input.
              - Internal block size constraints on width-in-tiles may restrict in-place variants for very large widths.

            Example:
              input_shape = (1, 1, 32, 32)

              attention_mask_t = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
              input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

              tt_output = ttnn.scale_mask_softmax_in_place(
                  input_tensor=input_tensor,
                  scale=1.0,
                  mask=attention_mask_t,
              )

            Example (Sharded):
                .. code-block:: python

                  compute_grid_size = device.compute_with_storage_grid_size()
                  fuse_head = 2
                  batch = compute_grid_size.x
                  num_cores_r = compute_grid_size.y

                  input_shape = (batch, num_cores_r, fuse_head * 384, 768)

                  attention_mask_t = ttnn.rand((batch, 1, 384, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  input_tensor = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  #Shard the input tensor
                  grid_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
                  shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
                  shard_shape = [fuse_head * 384, 768]
                  shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
                  sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

                  input_sharded = ttnn.to_memory_config(input_tensor, sharded_mem_config)

                  #Create sharded program config
                  program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                      compute_with_storage_grid_size=compute_grid_size,
                      subblock_w=8,
                      block_h=12 * fuse_head,
                      block_w=24,
                  )

                  tt_output = ttnn.scale_mask_softmax_in_place(
                      input_tensor=input_sharded,
                      scale=1.0,
                      mask=attention_mask_t,
                      program_config=program_config,
                  )

        )doc";

    using OperationType = decltype(ttnn::scale_mask_softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_mask_softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<float> scale,
               const std::optional<const Tensor>& mask,
               const SoftmaxProgramConfig& program_config,
               const bool is_causal_mask,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) {
                return self(
                    input_tensor, scale, mask, program_config, is_causal_mask, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("mask").noconvert() = std::nullopt,
            py::kw_only(),
            py::arg("program_config") = SoftmaxDefaultProgramConfig{},
            py::arg("is_causal_mask") = false,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("numeric_stable") = false});
}

void bind_normalization_scale_causal_mask_hw_dims_softmax_in_place_operation(py::module& module) {
    auto doc =
        R"doc(scale_causal_mask_hw_dims_softmax_in_place(input_tensor: ttnn.Tensor, scale: Optional[float] = None, mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig], compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute fused scale->attention_mask->softmax over :attr:`input_tensor` along the last dim, input and output tensor are in-placed on the same L1 address.

            Args:
                * :attr:`input_tensor`: the input tensor

            Keyword Args:
                * :attr:`program_config`: the program configuration for op. If not provided, SoftmaxDefaultProgramConfig is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Note:
              Supported data types and layouts by tensor:

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

              .. list-table:: mask
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE

              .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - matches input_tensor
                  - TILE

            Limitations:
              - All tensors must be on-device.
              - Requires sharded input (with ROW_MAJOR shard orientation) and a sharded softmax program config.
              - Mask must not be sharded. It should have shape [1, 1, H, W]
              - Scale must be provided.

              Example:
                .. code-block:: python

                  compute_grid_size = device.compute_with_storage_grid_size()
                  batch = compute_grid_size.x
                  num_cores_r = compute_grid_size.y

                  input_shape = (batch, num_cores_r, 384, 768)
                  attention_mask_t = ttnn.rand((1, 1, 384, 768), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  input_tiled = ttnn.rand(input_shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

                  # We must shard the input tensor in ROW_MAJOR orientation
                  grid_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
                  shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
                  shard_shape = [384, 768]
                  shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
                  sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

                  input_sharded = ttnn.to_memory_config(input_tiled, sharded_mem_config)

                  # We must also use the sharded softmax program config
                  program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                      compute_with_storage_grid_size=compute_grid_size,
                      subblock_w=8,
                      block_h=12,
                      block_w=24,
                  )

                  tt_output_sharded = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
                      input_tensor=input_sharded,
                      scale=1.0,
                      mask=attention_mask_t,
                      program_config=program_config,
                  )

        )doc";

    using OperationType = decltype(ttnn::scale_causal_mask_hw_dims_softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_causal_mask_hw_dims_softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<float> scale,
               const std::optional<const Tensor>& mask,
               const SoftmaxProgramConfig& program_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) {
                return self(input_tensor, scale, mask, program_config, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("mask").noconvert() = std::nullopt,
            py::kw_only(),
            py::arg("program_config") = SoftmaxDefaultProgramConfig{},
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("numeric_stable") = false});
}

void bind_normalization_softmax(py::module& module) {
    bind_normalization_softmax_program_config_operation(module);
    bind_normalization_softmax_operation(module);
    bind_normalization_scale_mask_softmax_operation(module);
    bind_normalization_softmax_in_place_operation(module);
    bind_normalization_scale_mask_softmax_in_place_operation(module);
    bind_normalization_scale_causal_mask_hw_dims_softmax_in_place_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
