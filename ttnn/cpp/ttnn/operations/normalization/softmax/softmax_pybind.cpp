// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "softmax.hpp"


namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_softmax_program_config_operation(py::module& module) {
    py::class_<SoftmaxProgramConfig>(module, "SoftmaxProgramConfig").def(py::init<>());

    py::class_<SoftmaxDefaultProgramConfig>(module, "SoftmaxDefaultProgramConfig")
        .def(py::init<>());

    py::class_<SoftmaxShardedMultiCoreProgramConfig>(module, "SoftmaxShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert()
        )
        .def_readwrite("block_w", &SoftmaxShardedMultiCoreProgramConfig::block_w);
}

void bind_normalization_softmax_operation(py::module& module) {

    auto doc =
        R"doc(softmax(input_tensor: ttnn.Tensor, dim: int, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute softmax over :attr:`input_tensor` along :attr:`dim`.

            Args:
                * :attr:`input_tensor`: the input tensor
                * :attr:`dim`: the dimension along which to compute softmax.

            Keyword Args:
                * :attr:`memory_config`: the memory configuration for the output tensor. If not provided, the memory configuration of the input tensor is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.softmax(tensor, -1)
                >>> print(output[0, 0, 0, :3])
                ttnn.Tensor([ 0.0310059, 0.0310059, 0.0310059], dtype=bfloat16 )
        )doc";

    using OperationType = decltype(ttnn::softmax);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const int8_t dim,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                    return self(input_tensor, dim, memory_config, compute_kernel_config);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("dim") = -1,
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("compute_kernel_config").noconvert() = std::nullopt});
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

            Example:

        )doc";

    using OperationType = decltype(ttnn::scale_mask_softmax);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_mask_softmax,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::optional<float> scale,
                const std::optional<const Tensor> mask,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const bool is_causal_mask,
                const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                    return self(input_tensor, scale, mask, memory_config, is_causal_mask, compute_kernel_config);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("scale").noconvert() = std::nullopt,
                py::arg("mask").noconvert() = std::nullopt,
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("is_causal_mask") = false,
                py::arg("compute_kernel_config") = std::nullopt});
}

void bind_normalization_softmax_in_place_operation(py::module& module) {

    auto doc =
        R"doc(softmax_in_place(input_tensor: ttnn.Tensor, program_config: Optional[SoftmaxProgramConfig], compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute softmax over :attr:`input_tensor` along the last dim, input and output tensor are in-placed on the same L1 address.

            Args:
                * :attr:`input_tensor`: the input tensor

            Keyword Args:
                * :attr:`program_config`: the program configuration for op. If not provided, SoftmaxDefaultProgramConfig is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Example:

        )doc";

    using OperationType = decltype(ttnn::softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const SoftmaxProgramConfig& program_config,
                const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                    return self(input_tensor, program_config, compute_kernel_config);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("program_config") = SoftmaxDefaultProgramConfig{},
                py::arg("compute_kernel_config") = std::nullopt});
}

void bind_normalization_scale_mask_softmax_in_place_operation(py::module& module) {

    auto doc =
        R"doc(softmax_in_place(input_tensor: ttnn.Tensor, scale: Optional[float] = None, mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig], compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Compute fused scale->attention_mask->softmax over :attr:`input_tensor` along the last dim, input and output tensor are in-placed on the same L1 address.

            Args:
                * :attr:`input_tensor`: the input tensor

            Keyword Args:
                * :attr:`program_config`: the program configuration for op. If not provided, SoftmaxDefaultProgramConfig is used.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.

            Example:

        )doc";

    using OperationType = decltype(ttnn::scale_mask_softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_mask_softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::optional<float> scale,
                const std::optional<const Tensor> mask,
                const SoftmaxProgramConfig& program_config,
                const bool is_causal_mask,
                const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                    return self(input_tensor, scale, mask, program_config, is_causal_mask, compute_kernel_config);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("scale").noconvert() = std::nullopt,
                py::arg("mask").noconvert() = std::nullopt,
                py::kw_only(),
                py::arg("program_config") = SoftmaxDefaultProgramConfig{},
                py::arg("is_causal_mask") = false,
                py::arg("compute_kernel_config") = std::nullopt});
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

            Example:

        )doc";

    using OperationType = decltype(ttnn::scale_causal_mask_hw_dims_softmax_in_place);

    ttnn::bind_registered_operation(
        module,
        ttnn::scale_causal_mask_hw_dims_softmax_in_place,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::optional<float> scale,
                const std::optional<const Tensor> mask,
                const SoftmaxProgramConfig& program_config,
                const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                    return self(input_tensor, scale, mask, program_config, compute_kernel_config);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("scale").noconvert() = std::nullopt,
                py::arg("mask").noconvert() = std::nullopt,
                py::kw_only(),
                py::arg("program_config") = SoftmaxDefaultProgramConfig{},
                py::arg("compute_kernel_config") = std::nullopt});
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
