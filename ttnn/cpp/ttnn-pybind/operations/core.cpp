// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::core {

void py_module_types(py::module& module) {
    py::enum_<ttnn::operations::compute_throttle_utils::ThrottleLevel>(module, "ThrottleLevel", R"doc(
        Enum for controlling compute throttling.

        Higher levels insert NOP instructions to reduce compute throughput:
        - LEVEL_1: Throttle to 73% of max performance
        - LEVEL_2: Throttle to 67% of max performance
        - LEVEL_3: Throttle to 50% of max performance
        - LEVEL_4: Throttle to 40% of max performance
        - LEVEL_5: Throttle to 33% of max performance

        Used to prevent di/dt (power supply current) issues on large core counts.
    )doc")
        .value("NO_THROTTLE", ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
        .value("LEVEL_1", ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_1)
        .value("LEVEL_2", ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_2)
        .value("LEVEL_3", ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_3)
        .value("LEVEL_4", ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_4)
        .value("LEVEL_5", ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_5);

    py::class_<DeviceComputeKernelConfig>(module, "DeviceComputeKernelConfig");

    py::class_<GrayskullComputeKernelConfig>(module, "GrayskullComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true,
            py::arg("dst_full_sync_en") = false)
        .def_readwrite("math_fidelity", &GrayskullComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &GrayskullComputeKernelConfig::math_approx_mode)
        .def_readwrite("dst_full_sync_en", &GrayskullComputeKernelConfig::dst_full_sync_en);

    py::class_<WormholeComputeKernelConfig>(module, "WormholeComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool, bool, bool, bool, ttnn::operations::compute_throttle_utils::ThrottleLevel>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true,
            py::arg("fp32_dest_acc_en") = false,
            py::arg("packer_l1_acc") = false,
            py::arg("dst_full_sync_en") = false,
            py::arg("throttle_level") = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
        .def_readwrite("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
        .def_readwrite("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
        .def_readwrite("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc)
        .def_readwrite("dst_full_sync_en", &WormholeComputeKernelConfig::dst_full_sync_en)
        .def_readwrite("throttle_level", &WormholeComputeKernelConfig::throttle_level);
}

void py_module(py::module& module) {
    module.def(
        "init_device_compute_kernel_config",
        &ttnn::init_device_compute_kernel_config,
        py::arg("arch"),
        py::arg("device_kernel_config") = std::nullopt,
        py::kw_only(),
        py::arg("math_fidelity") = MathFidelity::LoFi,
        py::arg("math_approx_mode") = true,
        py::arg("fp32_dest_acc_en") = false,
        py::arg("packer_l1_acc") = false,
        py::arg("dst_full_sync_en") = false,
        py::arg("throttle_level") = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    module.def("unsqueeze_to_4D", &ttnn::unsqueeze_to_4D, py::arg("tensor"));

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, IDevice*, const std::optional<MemoryConfig>&, QueueId>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt,
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, MeshDevice*, const std::optional<MemoryConfig>&, QueueId>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt,
        py::arg("cq_id") = ttnn::DefaultQueueId,
        R"doc(
            Copy tensor from host to device.

            Args:
                tensor (ttnn.Tensor): The tensor to be copied from host to device.
                device (ttnn.Device | ttnn.MeshDevice): The target device where the tensor will be copied.
                memory_config (ttnn.MemoryConfig, optional): The memory configuration to use. Defaults to `None`.
                cq_id (int, optional): The command queue ID to use. Defaults to `0`.

            Returns:
                ttnn.Tensor: The device tensor copy.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> tensor = ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16))
                >>> device_tensor = ttnn.to_device(tensor=tensor, device=device)
        )doc");

    module.def(
        "from_device",
        &ttnn::operations::core::from_device,
        py::arg("tensor"),
        py::arg("blocking") = true,
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId,
        R"doc(
            Copy tensor from device to host.

            Args:
                tensor (ttnn.Tensor): the tensor to be copied from device to host.
                blocking (bool, optional): whether the operation should be blocked until the copy is complete. Defaults to `True`.

            Keyword args:
                cq_id (int, optional): the command queue ID to use. Defaults to `0`.

            Returns:
                ttnn.Tensor: the host tensor copy.

            Example:
                >>> device = ttnn.open_device(0)
                >>> tensor = ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16))
                >>> device_tensor = ttnn.to_device(tensor=tensor, device=device)
                >>> # non-blocking mode
                >>> host_tensor = ttnn.from_device(tensor=device_tensor, blocking=False)
        )doc");

    module.def(
        "deallocate",
        &ttnn::operations::core::deallocate,
        py::arg("tensor"),
        py::arg("force") = true,
        R"doc(
        Deallocates device tensor. Releases the resources for `ttnn.Tensor` :attr:`tensor` explicitly.

        Args:
            tensor (ttnn.Tensor): the input tensor.
            force (bool, optional): force deallocation even if the buffer may have multiple references. Defaults to `True`.

        Returns:
            `None`: deallocates the tensor.

        Example:
            >>> device_id = 0
            >>> device = ttnn.open_device(device_id=device_id)
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
            >>> ttnn.deallocate(tensor=tensor, force=False)
    )doc");

    module.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return reallocate(input_tensor, memory_config); },
        py::arg("tensor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
            Deallocates device tensor and returns a reallocated tensor.

            Args:
                tensor (ttnn.Tensor): the input tensor.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the reallocated tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: the reallocated tensor.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
                >>> new_tensor = ttnn.reallocate(tensor, memory_config=my_memory_config)
        )doc");

    bind_registered_operation(
        module,
        ttnn::to_memory_config,
        R"doc(
        Converts a tensor to the desired memory configuration. Used for converting tensors to sharded tensors, interleaved tensors, or converting between DRAM and L1 memory.

        Args:
            tensor (ttnn.Tensor): the input tensor to be converted.
            memory_config (ttnn.MemoryConfig): the desired memory configuration for the tensor.
            dtype (ttnn.DataType, optional): the optional `ttnn` data type. Defaults to `None`.

        Returns:
            ttnn.Tensor: the converted tensor.

        Example:
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

    module
        .def(
            "allocate_tensor_on_device",
            [](const ttnn::TensorSpec& spec, MeshDevice* device) {
                return tt::tt_metal::allocate_tensor_on_device(spec, device);
            },
            py::arg("tensor_spec"),
            py::arg("mesh_device"))
        .def(
            "allocate_tensor_on_host",
            [](const ttnn::TensorSpec& spec, MeshDevice* device) {
                return tt::tt_metal::allocate_tensor_on_host(spec, device);
            },
            py::arg("tensor_spec"),
            py::arg("mesh_device"));

    module
        .def(
            "allocate_tensor_on_device",
            [](const ttnn::Shape& shape,
               ttnn::DataType dtype,
               ttnn::Layout layout,
               MeshDevice* device,
               const std::optional<ttnn::MemoryConfig>& mem_config) {
                return tt::tt_metal::allocate_tensor_on_device(
                    TensorSpec(
                        shape,
                        tt::tt_metal::TensorLayout(
                            dtype, tt::tt_metal::PageConfig(layout), mem_config.value_or(MemoryConfig{}))),
                    device);
            },
            py::arg("shape"),
            py::arg("dtype"),
            py::arg("layout"),
            py::arg("mesh_device"),
            py::arg("memory_config") = std::nullopt)
        .def(
            "allocate_tensor_on_host",
            [](const ttnn::Shape& shape,
               ttnn::DataType dtype,
               ttnn::Layout layout,
               MeshDevice* device,
               const std::optional<ttnn::MemoryConfig>& mem_config) {
                return tt::tt_metal::allocate_tensor_on_host(
                    TensorSpec(
                        shape,
                        tt::tt_metal::TensorLayout(
                            dtype, tt::tt_metal::PageConfig(layout), mem_config.value_or(MemoryConfig{}))),
                    device);
            },
            py::arg("shape"),
            py::arg("dtype"),
            py::arg("layout"),
            py::arg("mesh_device"),
            py::arg("memory_config") = std::nullopt);

    module.def(
        "copy_host_to_device_tensor",
        [](const ttnn::Tensor& host_tensor, ttnn::Tensor& device_tensor, QueueId cq_id = ttnn::DefaultQueueId) {
            tt::tt_metal::write_tensor(host_tensor, device_tensor, /*blocking=*/false, cq_id);
        },
        py::arg("host_tensor"),
        py::arg("device_tensor"),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "copy_device_to_host_tensor",
        [](const ttnn::Tensor& device_tensor,
           ttnn::Tensor& host_tensor,
           bool blocking = true,
           QueueId cq_id = ttnn::DefaultQueueId) {
            tt::tt_metal::write_tensor(device_tensor, host_tensor, blocking, cq_id);
        },
        py::arg("device_tensor"),
        py::arg("host_tensor"),
        py::arg("blocking") = true,
        py::arg("cq_id") = ttnn::DefaultQueueId);

    bind_registered_operation(
        module,
        ttnn::to_layout,
        R"doc(
        Organizes the `ttnn.Tensor` tensor into either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.

        When requesting `ttnn.ROW_MAJOR_LAYOUT`, the tensor will be returned unpadded in the last two dimensions.
        When requesting `ttnn.TILE_LAYOUT`, the tensor will be automatically padded where the width and height
        become multiples of 32. In the case where the layout is the same, the operation simply pads or unpads
        the last two dimensions depending on the requested layout.

        Args:
            tensor (ttnn.Tensor): the input tensor to be organized.
            layout (ttnn.Layout): the desired layout, either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.
            dtype (ttnn.DataType, optional): the optional output data type.
            memory_config (ttnn.MemoryConfig, optional): the optional output memory configuration.

        Returns:
            ttnn.Tensor: the tensor with the requested layout.

        Example:
            >>> device_id = 0
            >>> device = ttnn.open_device(device_id=device_id)
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
            >>> print(tensor[0,0,:3])
            Tensor([1.42188, -1.25, -0.398438], dtype=bfloat16)
        )doc",
        ttnn::pybind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(tensor, layout, dtype, memory_config);
            },
            py::arg("tensor"),
            py::arg("layout"),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt});

    module.def(
        "num_cores_to_corerangeset",
        py::overload_cast<const uint32_t, const CoreCoord, const bool>(&tt::tt_metal::num_cores_to_corerangeset),
        R"doc(Create a CoreRangeSet containing the specified number of cores)doc");

    module.def(
        "num_cores_to_corerangeset_in_subcoregrids",
        py::overload_cast<const CoreCoord, const uint32_t, const CoreRangeSet&, const bool>(
            &tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids),
        R"doc(Create a CoreRangeSet containing the specified number of cores starting from start_core in given subcoregrids)doc");
}

}  // namespace ttnn::operations::core
