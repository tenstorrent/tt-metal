// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_pybind.hpp"

#include "softmax.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::normalization::detail {
namespace py = pybind11;

/**
 * @brief Binds Softmax program configuration classes to Python module.
 *
 * This function exposes three Softmax program configuration classes to Python:
 *
 * - SoftmaxProgramConfig: Base program configuration class with default constructor
 * - SoftmaxDefaultProgramConfig: Default program configuration with standard settings
 * - SoftmaxShardedMultiCoreProgramConfig: Multi-core sharded configuration with the following parameters:
 *   - compute_with_storage_grid_size: Grid size for compute cores with storage
 *   - subblock_w: Width of sub-blocks for computation
 *   - block_h: Height of blocks for processing
 *   - block_w: Width of blocks for processing (also exposed as read/write property)
 *
 * @param module The Python module to bind the classes to
 */
void bind_normalization_softmax_program_config_operation(py::module& module) {
    py::class_<SoftmaxProgramConfig>(module, "SoftmaxProgramConfig", R"doc(
        Base program configuration class for Softmax operations.

        This is the base class for all Softmax program configurations. It provides
        a common interface for different types of program configurations used in
        softmax operations.
    )doc")
        .def(py::init<>());

    py::class_<SoftmaxDefaultProgramConfig>(module, "SoftmaxDefaultProgramConfig", R"doc(
        Default program configuration for Softmax operations.

        This configuration uses the default settings for Softmax operations, providing
        standard behavior suitable for most use cases. It automatically selects
        appropriate parameters based on the input tensor characteristics.

        Example:
            .. code-block:: python

                config = ttnn.SoftmaxDefaultProgramConfig()
                result = ttnn.softmax_in_place(tensor, program_config=config)
    )doc")
        .def(py::init<>());

    py::class_<SoftmaxShardedMultiCoreProgramConfig>(module, "SoftmaxShardedMultiCoreProgramConfig", R"doc(
        Multi-core sharded program configuration for Softmax operations.

        This configuration is designed for sharded tensors and enables multi-core
        execution with customizable block sizes and compute grid configuration.
        It provides fine-grained control over the computation parameters for
        optimal performance on sharded data.

        Args:
            compute_with_storage_grid_size (CoreCoord): The grid size for compute cores with storage capability.
            subblock_w (int): Width of sub-blocks for computation. Must be divisible by the tensor's width.
            block_h (int): Height of blocks for processing. Controls the vertical granularity of computation.
            block_w (int): Width of blocks for processing. Controls the horizontal granularity of computation. Can be modified after creation.

        Note:
            * This configuration is specifically designed for sharded tensors.
            * Block dimensions must be compatible with the tensor's shard specification.
            * Proper block sizing can significantly impact performance.

        Example:
            .. code-block:: python

                compute_grid = device.compute_with_storage_grid_size()
                config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=compute_grid,
                    subblock_w=8,
                    block_h=32,
                    block_w=24
                )
                # Modify block_w if needed
                config.block_w = 32
                result = ttnn.softmax_in_place(sharded_tensor, program_config=config)
    )doc")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert())
        .def_readwrite("block_w", &SoftmaxShardedMultiCoreProgramConfig::block_w);
}

// Softmax operation base
void bind_normalization_softmax_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes the softmax function over the specified dimension of the input tensor.

            The softmax function is defined as:

            .. math::
                \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}

            Args:
                input_tensor (ttnn.Tensor): The input tensor to apply softmax to.
                dim (int, optional): The dimension along which to compute softmax. Defaults to -1 (last dimension).

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, inherits from input tensor.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: Output tensor with softmax applied along the specified dimension.

            Supported dtypes and layouts

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            Example:
                .. code-block:: python

                    tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    result = ttnn.softmax(tensor, dim=-1)
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
               const bool numeric_stable) -> ttnn::Tensor {
                return self(input_tensor, dim, memory_config, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim") = -1,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("numeric_stable").noconvert() = false});
}

// Softmax with scale and mask
void bind_normalization_softmax_scale_mask_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes a fused scale-mask-softmax operation along the last dimension of the input tensor.

            This operation performs the following sequence:
            1. Optionally scales the input: ``scaled = input_tensor * scale`` (if scale is provided)
            2. Optionally applies mask: ``masked = scaled + mask`` (if mask is provided, with broadcasting)
            3. Computes softmax: ``output = softmax(masked)``

            This fused operation is commonly used in attention mechanisms where scaling and masking
            are applied before the softmax operation for efficiency.

            Args:
                input_tensor (ttnn.Tensor): The input tensor to process.
                scale (float, optional): Scaling factor to multiply with input tensor.
                mask (ttnn.Tensor, optional): Attention mask tensor to add to scaled input.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, inherits from input tensor.
                is_causal_mask (bool, optional): Whether the mask is a causal mask. Defaults to False.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: Output tensor with the fused scale-mask-softmax operation applied.

            Supported dtypes and layouts:

            .. list-table:: Input Tensor
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            .. list-table:: Mask Tensor (optional)
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B
                 - TILE, ROW_MAJOR

            Note:
                * All tensors must be on-device.
                * For ROW_MAJOR masks: intermediate dimensions (except last two) must be 1; last dimension must equal TILE_WIDTH; width must align to input tensor's tile width.

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
               const bool numeric_stable) -> ttnn::Tensor {
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

// Softmax in-place operation
void bind_normalization_softmax_inplace_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes the softmax function along the last dimension of the input tensor in-place.

            This operation modifies the input tensor directly, making it memory-efficient by avoiding
            additional tensor allocation. The softmax is computed as:

            .. math::
                \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}

            Args:
                input_tensor (ttnn.Tensor): The input tensor to apply softmax to. This tensor is modified in-place.

            Keyword Args:
                program_config (SoftmaxProgramConfig, optional): Program configuration for the operation. Defaults to SoftmaxDefaultProgramConfig().
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: The same tensor as input with softmax applied in-place.

            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            Note:
                * The input tensor is modified in-place to save memory.
                * For very wide tensors, the operation may fall back to standard softmax if circular buffers would consume more than 90% of L1 memory.
                * Supports both default and sharded multi-core program configurations.

            Example:
                .. code-block:: python

                shape = [1, 1, 32, 32]
                input_tensor = ttnn.rand(shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                output_tensor = ttnn.softmax_in_place(input_tensor)

            Example (with sharded configuration):
                .. code-block:: python

                    compute_grid = device.compute_with_storage_grid_size()
                    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=compute_grid,
                        subblock_w=8,
                        block_h=32,
                        block_w=32
                    )
                    result = ttnn.softmax_in_place(tensor, program_config=program_config)

    )doc";

    using OperationType = decltype(ttnn::softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const SoftmaxProgramConfig& program_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) -> ttnn::Tensor {
                return self(input_tensor, dim, program_config, compute_kernel_config, numeric_stable);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim") = -1,
            py::kw_only(),
            py::arg("program_config") = SoftmaxDefaultProgramConfig{},
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("numeric_stable") = false});
}

// Softmax with scale and mask in-place operation
void bind_normalization_softmax_scale_mask_inplace_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes a fused scale-mask-softmax operation along the last dimension in-place.

            This operation modifies the input tensor directly and performs the following sequence:
            1. Optionally scales the input: ``input_tensor *= scale`` (if scale is provided)
            2. Optionally applies mask: ``input_tensor += mask`` (if mask is provided, with broadcasting)
            3. Computes softmax: ``input_tensor = softmax(input_tensor)``

            This in-place fused operation is commonly used in attention mechanisms and is memory-efficient
            as it reuses the input tensor for output, avoiding additional memory allocation.

            Args:
                input_tensor (ttnn.Tensor): The input tensor to process. This tensor is modified in-place.
                scale (float, optional): Scaling factor to multiply with input tensor.
                mask (ttnn.Tensor, optional): Attention mask tensor to add to scaled input.

            Keyword Args:
                program_config (SoftmaxProgramConfig, optional): Program configuration for the operation. Defaults to SoftmaxDefaultProgramConfig().
                is_causal_mask (bool, optional): Whether the mask is a causal mask. Defaults to False.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: The same tensor as input with the fused scale-mask-softmax operation applied in-place.

            Supported dtypes and layouts:

            .. list-table:: Input Tensor
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            .. list-table:: Mask Tensor (optional)
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B
                 - TILE, ROW_MAJOR
                 - 2, 3, 4

            Note:
                * All tensors must be on-device.
                * For unsharded ROW_MAJOR masks: intermediate dimensions (except last two) must be 1; last dimension must equal TILE_WIDTH; width must align to input tensor.
                * For sharded inputs: mask must be TILE layout with identical padded shape to input.
                * Internal block size constraints may restrict in-place operation for very large width tensors.

            Example:
                .. code-block:: python

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

                    # Shard the input tensor
                    grid_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
                    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
                    shard_shape = [fuse_head * 384, 768]
                    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
                    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

                    input_sharded = ttnn.to_memory_config(input_tensor, sharded_mem_config)

                    # Create sharded program config
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
               const bool numeric_stable) -> ttnn::Tensor {
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

// Softmax with scale and causal mask in-place operation
void bind_normalization_softmax_scale_casual_mask_HW_inplace_operation(py::module& module) {
    const auto doc =
        R"doc(
            Specialized in-place operation for causal masked softmax with height-width dimension constraints.

            This is an optimized version of scale_mask_softmax_in_place specifically designed for transformer
            attention patterns where the causal mask only affects the height and width dimensions. This operation
            provides better performance for specific use cases with the following constraints:

            **Requirements:**
            * Input tensor should be sharded for optimal performance
            * Attention mask must be interleaved and have shape [1, 1, H, W] (hw_dims_only)
            * The mask is treated as a causal mask by design
            * Scale parameter is typically provided for attention scaling

            The operation performs:
            1. Scales the input: ``input_tensor *= scale`` (if scale is provided)
            2. Applies causal mask: ``input_tensor += mask`` (with broadcasting from [1, 1, H, W])
            3. Computes softmax: ``input_tensor = softmax(input_tensor)``

            Args:
                input_tensor (ttnn.Tensor): The input tensor to process. Must be sharded for optimal performance.
                scale (float, optional): Scaling factor to multiply with input tensor (typically 1/√d_k for attention).
                mask (ttnn.Tensor, optional): Causal attention mask tensor with shape [1, 1, H, W].

            Keyword Args:
                program_config (SoftmaxProgramConfig, optional): Program configuration for the operation. Defaults to SoftmaxDefaultProgramConfig().
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: The same tensor as input with the specialized causal scale-mask-softmax operation applied in-place.

            Supported dtypes and layouts:

            .. list-table:: Input Tensor (Sharded)
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            .. list-table:: Mask Tensor [1, 1, H, W]
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B
                 - TILE (interleaved)

            Note:
                * This is an experimental/specialized feature optimized for specific transformer attention patterns.
                * Input tensor must be sharded for optimal performance.
                * Mask shape is constrained to [1, 1, H, W] format.
                * Provides better performance than general scale_mask_softmax_in_place for these specific constraints.

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
               const bool numeric_stable) -> ttnn::Tensor {
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
    bind_normalization_softmax_scale_mask_operation(module);
    bind_normalization_softmax_inplace_operation(module);
    bind_normalization_softmax_scale_mask_inplace_operation(module);
    bind_normalization_softmax_scale_casual_mask_HW_inplace_operation(module);
}
}  // namespace ttnn::operations::normalization::detail
