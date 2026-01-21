// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_nanobind.hpp"

#include <cstddef>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "softmax.hpp"

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::operations::normalization::detail {

struct SoftmaxProgramConfigPlaceholder {};

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
void bind_normalization_softmax_program_config_operation(nb::module_& mod) {
    nb::class_<SoftmaxProgramConfigPlaceholder>(mod, "SoftmaxProgramConfig", R"doc(
        Base program configuration variant for Softmax operations.

        This is the variant for all Softmax program configurations. It provides
        a common interface for different types of program configurations used in
        softmax operations.
    )doc");

    nb::class_<SoftmaxDefaultProgramConfig>(mod, "SoftmaxDefaultProgramConfig", R"doc(
        Default program configuration for Softmax operations.

        This configuration uses the default settings for Softmax operations, providing
        standard behavior suitable for most use cases. It automatically selects
        appropriate parameters based on the input tensor characteristics.
    )doc")
        .def(nb::init<>());

    nb::class_<SoftmaxShardedMultiCoreProgramConfig>(mod, "SoftmaxShardedMultiCoreProgramConfig", R"doc(
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
    )doc")
        .def(
            nb::init<CoreCoord, std::size_t, std::size_t, std::size_t>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("subblock_w").noconvert(),
            nb::arg("block_h").noconvert(),
            nb::arg("block_w").noconvert())
        .def_rw("compute_with_storage_grid_size", &SoftmaxShardedMultiCoreProgramConfig::compute_with_storage_grid_size)
        .def_rw("subblock_w", &SoftmaxShardedMultiCoreProgramConfig::subblock_w)
        .def_rw("block_h", &SoftmaxShardedMultiCoreProgramConfig::block_h)
        .def_rw("block_w", &SoftmaxShardedMultiCoreProgramConfig::block_w);
}

// Softmax operation base
void bind_normalization_softmax_operation(nb::module_& mod) {
    const auto* const doc =
        R"doc(
            Computes the softmax function over the specified dimension of the input tensor.

            The softmax function is defined as:

            .. math::
                \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}

            Args:
                input_tensor (ttnn.Tensor): The input tensor to apply softmax to. Must be on the device.
                dim (int, optional): The dimension along which to compute softmax. Defaults to -1 (last dimension).

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, inherits from input tensor.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to True.

            Returns:
                ttnn.Tensor: Output tensor with softmax applied along the specified dimension.

            Note:
                The tensors support the following data types and layouts:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - BFLOAT16, FLOAT32, BFLOAT8_B
                      - TILE

                The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded (L1): Height sharded

            Limitations:
                * All tensors must be on-device, interleaved, and tile layout.
                * Using the attention-optimized kernels requires a 4D input tensor and reducing on the last dimension.
    )doc";

    using OperationType = decltype(ttnn::softmax);

    ttnn::bind_registered_operation(
        mod,
        ttnn::softmax,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) -> ttnn::Tensor {
                return self(input_tensor, dim, memory_config, compute_kernel_config, numeric_stable);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim") = -1,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none(),
            nb::arg("numeric_stable").noconvert() = true});
}

// Softmax with scale and mask
void bind_normalization_softmax_scale_mask_operation(nb::module_& mod) {
    const auto* const doc =
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
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to True.

            Returns:
                ttnn.Tensor: Output tensor with the fused scale-mask-softmax operation applied.

            Note:
                The tensors support the following data types and layouts:

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

                The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

            Limitations:
                * All tensors must be on-device.
                * For ROW_MAJOR masks: intermediate dimensions (except last two) must be 1; last dimension must equal TILE_WIDTH; width must align to input tensor's tile width.
    )doc";
    using OperationType = decltype(ttnn::scale_mask_softmax);

    ttnn::bind_registered_operation(
        mod,
        ttnn::scale_mask_softmax,
        doc,
        ttnn::nanobind_overload_t{
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
            nb::arg("input_tensor").noconvert(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("mask").noconvert() = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("is_causal_mask") = false,
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("numeric_stable") = true});
}

// Softmax in-place operation
void bind_normalization_softmax_inplace_operation(nb::module_& mod) {
    const auto* const doc =
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
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to True.

            Returns:
                ttnn.Tensor: The same tensor as input with softmax applied in-place.

            Note:
                The tensors support the following data types and layouts:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - BFLOAT16, FLOAT32, BFLOAT8_B
                      - TILE

                The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

            Limitations:
                * The input tensor is modified in-place to save memory. Must already be on the device.
                * For very wide tensors, the operation may fall back to standard softmax if circular buffers would consume more than 90% of L1 memory.
                * Supports both default and sharded multi-core program configurations.
    )doc";

    using OperationType = decltype(ttnn::softmax_in_place);

    ttnn::bind_registered_operation(
        mod,
        ttnn::softmax_in_place,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const SoftmaxProgramConfig& program_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) -> ttnn::Tensor {
                return self(input_tensor, dim, program_config, compute_kernel_config, numeric_stable);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim") = -1,
            nb::kw_only(),
            nb::arg("program_config") = nb::cast(SoftmaxDefaultProgramConfig{}),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("numeric_stable") = true});
}

// Softmax with scale and mask in-place operation
void bind_normalization_softmax_scale_mask_inplace_operation(nb::module_& mod) {
    const auto* const doc =
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

            Note:
                The tensors support the following data types and layouts:

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

                The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

            Limitations:
                * All tensors must be on-device.
                * For unsharded ROW_MAJOR masks: intermediate dimensions (except last two) must be 1; last dimension must equal TILE_WIDTH; width must align to input tensor.
                * For sharded inputs: mask must be TILE layout with identical padded shape to input.
                * Internal block size constraints may restrict in-place operation for very large width tensors.
    )doc";

    using OperationType = decltype(ttnn::scale_mask_softmax_in_place);

    ttnn::bind_registered_operation(
        mod,
        ttnn::scale_mask_softmax_in_place,
        doc,
        ttnn::nanobind_overload_t{
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
            nb::arg("input_tensor").noconvert(),
            nb::arg("scale") = nb::none(),
            nb::arg("mask") = nb::none(),
            nb::kw_only(),
            nb::arg("program_config") = nb::cast(SoftmaxDefaultProgramConfig{}),
            nb::arg("is_causal_mask") = false,
            nb::arg("compute_kernel_config") = nb::none(),
            // TODO: switch the default value to 'true' once model accuracy is fixed
            // See issue #28531
            nb::arg("numeric_stable") = false});
}

// Softmax with scale and causal mask in-place operation
void bind_normalization_softmax_scale_casual_mask_HW_inplace_operation(nb::module_& mod) {
    const auto* const doc =
        R"doc(
            Specialized in-place operation for causal masked softmax with height-width dimension constraints.

            This is an optimized version of scale_mask_softmax_in_place specifically designed for transformer
            attention patterns where the causal mask only affects the height and width dimensions.
            This operation provides better performance than general :func:`ttnn.scale_mask_softmax_in_place` for these specific constraints.

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

            Note:
                The tensors support the following data types and layouts:

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

                The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

            Limitations:
                * This is an experimental/specialized feature optimized for specific transformer attention patterns.
                * Inputs must be on the device.
                * Input tensor must be sharded for optimal performance.
                * Attention mask must be interleaved and have shape [1, 1, H, W] (i.e. hw_dims_only)
                * The mask is treated as a causal mask by design
                * Scale parameter is typically provided for attention scaling
    )doc";

    using OperationType = decltype(ttnn::scale_causal_mask_hw_dims_softmax_in_place);

    ttnn::bind_registered_operation(
        mod,
        ttnn::scale_causal_mask_hw_dims_softmax_in_place,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<float> scale,
               const std::optional<const Tensor>& mask,
               const SoftmaxProgramConfig& program_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
               const bool numeric_stable) -> ttnn::Tensor {
                return self(input_tensor, scale, mask, program_config, compute_kernel_config, numeric_stable);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("mask").noconvert() = nb::none(),
            nb::kw_only(),
            nb::arg("program_config") = nb::cast(SoftmaxDefaultProgramConfig{}),
            nb::arg("compute_kernel_config") = nb::none(),
            // TODO: switch the default value to 'true' once model accuracy is fixed
            // See issue #28531
            nb::arg("numeric_stable") = false});
}

void bind_normalization_softmax(nb::module_& mod) {
    bind_normalization_softmax_program_config_operation(mod);
    bind_normalization_softmax_operation(mod);
    bind_normalization_softmax_scale_mask_operation(mod);
    bind_normalization_softmax_inplace_operation(mod);
    bind_normalization_softmax_scale_mask_inplace_operation(mod);
    bind_normalization_softmax_scale_casual_mask_HW_inplace_operation(mod);
}
}  // namespace ttnn::operations::normalization::detail

// NOLINTEND(bugprone-unused-raii)
