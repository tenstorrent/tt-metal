// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/work_split.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace core {

void py_module_types(py::module& module) {
    py::class_<DeviceComputeKernelConfig>(module, "DeviceComputeKernelConfig");

    py::class_<GrayskullComputeKernelConfig>(module, "GrayskullComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true)
        .def_readwrite("math_fidelity", &GrayskullComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &GrayskullComputeKernelConfig::math_approx_mode);

    py::class_<WormholeComputeKernelConfig>(module, "WormholeComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool, bool, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true,
            py::arg("fp32_dest_acc_en") = false,
            py::arg("packer_l1_acc") = false)
        .def_readwrite("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
        .def_readwrite("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
        .def_readwrite("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc);

}

void py_module(py::module& module) {
    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const ttnn::Shape& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 1>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 2>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 3>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 4>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 5>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def("unsqueeze_to_4D", &ttnn::unsqueeze_to_4D, py::arg("tensor"));

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, Device*, const std::optional<MemoryConfig>&>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt);

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, MeshDevice*, const std::optional<MemoryConfig>&>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt);

    module.def("from_device", &ttnn::operations::core::from_device, py::arg("tensor"), py::arg("blocking") = true, py::kw_only(), py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def("deallocate", &ttnn::operations::core::deallocate, py::arg("tensor"), py::arg("force") = true);

    module.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return reallocate(input_tensor, memory_config); },
        py::arg("tensor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
            Deallocates device tensor and returns a reallocated tensor

            Args:
                * :attr:`input_tensor`: Input Tensor
        )doc");

    bind_registered_operation(
        module,
        ttnn::to_memory_config,
        R"doc(to_memory_config(tensor: ttnn.Tensor, memory_config: MemoryConfig, dtype: Optional[DataType] = None) -> ttnn.Tensor

            Converts a tensor to the desired mem_config, used for converting tensors to sharded tensors or interleaved, and to convert DRAM to L1 and vice versa


            Args:
                * :attr:`tensor`: the ttnn.Tensor
                * :attr:`memory_config`: the desired MemoryConfig
                * :attr:`dtype`: the optional `ttnn` data type.


                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
                >>> tensor = ttnn.to_memory_config(tensor, memory_config)
        )doc",
        ttnn::pybind_arguments_t{py::arg("tensor"), py::arg("memory_config"), py::arg("dtype") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::to_dtype,
        R"doc(to_dtype(tensor: ttnn.Tensor, dtype: DataType = None) -> ttnn.Tensor

            Converts a tensor to the desired dtype


            Args:
                * :attr:`tensor`: the ttnn.Tensor
                * :attr:`dtype`: `ttnn` data type.

            Example:
                >>> tensor = ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16))
                >>> tensor = ttnn.to_dtype(tensor, dtype=ttnn.uint16)
        )doc",
        ttnn::pybind_arguments_t{py::arg("tensor"), py::arg("dtype")});

    module.def(
        "allocate_tensor_on_device",
        py::overload_cast<
            const ttnn::Shape&,
            ttnn::DataType,
            ttnn::Layout,
            Device*,
            const std::optional<ttnn::MemoryConfig>&>(&ttnn::operations::core::allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt);

    module.def(
        "allocate_tensor_on_device",
        py::overload_cast<
            const ttnn::Shape&,
            ttnn::DataType,
            ttnn::Layout,
            MeshDevice*,
            const std::optional<ttnn::MemoryConfig>&>(&ttnn::operations::core::allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("mesh_device"),
        py::arg("memory_config") = std::nullopt);

    module.def(
        "copy_host_to_device_tensor",
        &ttnn::operations::core::copy_host_to_device_tensor,
        py::arg("host_tensor"),
        py::arg("device_tensor"),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "begin_trace_capture",
        py::overload_cast<Device*, const uint8_t>(&ttnn::operations::core::begin_trace_capture),
        py::arg("device"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "end_trace_capture",
        py::overload_cast<Device*, const uint32_t, const uint8_t>(&ttnn::operations::core::end_trace_capture),
        py::arg("device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "execute_trace",
        py::overload_cast<Device*, const uint32_t, const uint8_t, bool>(&ttnn::operations::core::execute_trace),
        py::arg("device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId,
        py::arg("blocking") = true);

    module.def(
        "release_trace",
        py::overload_cast<Device*, const uint32_t>(&ttnn::operations::core::release_trace),
        py::arg("device"),
        py::arg("trace_id"));

    module.def(
        "begin_trace_capture",
        py::overload_cast<MeshDevice*, const uint8_t>(&ttnn::operations::core::begin_trace_capture),
        py::arg("mesh_device"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "end_trace_capture",
        py::overload_cast<MeshDevice*, const uint32_t, const uint8_t>(&ttnn::operations::core::end_trace_capture),
        py::arg("mesh_device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "execute_trace",
        py::overload_cast<MeshDevice*, const uint32_t, const uint8_t, bool>(&ttnn::operations::core::execute_trace),
        py::arg("mesh_device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId,
        py::arg("blocking") = true);

    module.def(
        "release_trace",
        py::overload_cast<MeshDevice*, const uint32_t>(&ttnn::operations::core::release_trace),
        py::arg("mesh_device"),
        py::arg("trace_id"));

    bind_registered_operation(
        module,
        ttnn::to_layout,
        R"doc(to_layout(tensor: ttnn.Tensor, layout: Layout, dtype: Optional[DataType] = None, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

    Organizes the `ttnn.Tensor` :attr:`tensor` into either ROW_MAJOR_LAYOUT or TILE_LAYOUT.  When requesting ROW_MAJOR_LAYOUT
    the tensor will be returned unpadded in the last two dimensions.   When requesting TILE_LAYOUT the tensor will be automatically
    padded where the width and height become multiples of 32.
    In the case where the layout is the same, the operation simply pad or unpad the last two dimensions depending on layout requested.

    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`layout`: the layout of either ttnn.ROW_MAJOR_LAYOUT or ttnn.TILE_LAYOUT.
        * :attr:`dtype`: the optional output data type.
        * :attr:`memory_config`: the optional output memory configuration.
        * :attr:`device`: Device/MeshDevice whose worker thread on host should be used for the layout conversion

    Example:
        >>> device_id = 0
        >>> device = ttnn.open_device(device_id=device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
        >>> print(tensor[0,0,:3])
        Tensor([ 1.42188, -1.25, -0.398438], dtype=bfloat16 ))doc",
        ttnn::pybind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               Device* device) -> ttnn::Tensor { return self(tensor, layout, dtype, memory_config, device); },
            py::arg("tensor"),
            py::arg("layout"),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("device") = nullptr},
        ttnn::pybind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               MeshDevice* device) -> ttnn::Tensor { return self(tensor, layout, dtype, memory_config, device); },
            py::arg("tensor"),
            py::arg("layout"),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("device") = nullptr});

}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
