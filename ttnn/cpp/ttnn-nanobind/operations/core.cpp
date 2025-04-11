// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/work_split.hpp>

namespace nb = nanobind;

namespace ttnn::operations::core {

void py_module_types(nb::module_& mod) {
    nb::class_<DeviceComputeKernelConfig>(mod, "DeviceComputeKernelConfig");

    nb::class_<GrayskullComputeKernelConfig>(mod, "GrayskullComputeKernelConfig")
        .def(
            nb::init<MathFidelity, bool, bool>(),
            nb::kw_only(),
            nb::arg("math_fidelity") = MathFidelity::Invalid,
            nb::arg("math_approx_mode") = true,
            nb::arg("dst_full_sync_en") = false)
        .def_rw("math_fidelity", &GrayskullComputeKernelConfig::math_fidelity)
        .def_rw("math_approx_mode", &GrayskullComputeKernelConfig::math_approx_mode)
        .def_rw("dst_full_sync_en", &GrayskullComputeKernelConfig::dst_full_sync_en);

    nb::class_<WormholeComputeKernelConfig>(mod, "WormholeComputeKernelConfig")
        .def(
            nb::init<MathFidelity, bool, bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("math_fidelity") = MathFidelity::Invalid,
            nb::arg("math_approx_mode") = true,
            nb::arg("fp32_dest_acc_en") = false,
            nb::arg("packer_l1_acc") = false,
            nb::arg("dst_full_sync_en") = false)
        .def_rw("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
        .def_rw("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
        .def_rw("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
        .def_rw("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc)
        .def_rw("dst_full_sync_en", &WormholeComputeKernelConfig::dst_full_sync_en);
}

void py_module(nb::module_& mod) {

    mod.def("init_device_compute_kernel_config", &ttnn::init_device_compute_kernel_config,
            nb::arg("arch"),
            nb::arg("device_kernel_config") = std::nullopt,
            nb::kw_only(),
            nb::arg("math_fidelity") = MathFidelity::LoFi,
            nb::arg("math_approx_mode") = true,
            nb::arg("fp32_dest_acc_en") = false,
            nb::arg("packer_l1_acc") = false,
            nb::arg("dst_full_sync_en") = false
        );
    mod.def("unsqueeze_to_4D", &ttnn::unsqueeze_to_4D, nb::arg("tensor"));

    mod.def(
        "to_device",
        nb::overload_cast<const ttnn::Tensor&, IDevice*, const std::optional<MemoryConfig>&, QueueId>(
            &ttnn::operations::core::to_device),
        nb::arg("tensor"),
        nb::arg("device"),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "to_device",
        nb::overload_cast<const ttnn::Tensor&, MeshDevice*, const std::optional<MemoryConfig>&, QueueId>(
            &ttnn::operations::core::to_device),
        nb::arg("tensor"),
        nb::arg("device"),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("cq_id") = ttnn::DefaultQueueId,
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

    mod.def(
        "from_device",
        &ttnn::operations::core::from_device,
        nb::arg("tensor"),
        nb::arg("blocking") = true,
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId,
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

    mod.def(
        "deallocate",
        &ttnn::operations::core::deallocate,
        nb::arg("tensor"),
        nb::arg("force") = true,
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

    mod.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return reallocate(input_tensor, memory_config); },
        nb::arg("tensor"),
        nb::arg("memory_config") = std::nullopt,
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
        mod,
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
        ttnn::nanobind_arguments_t{nb::arg("tensor"), nb::arg("memory_config"), nb::arg("dtype") = std::nullopt});

    bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{nb::arg("tensor"), nb::arg("dtype")});

    mod.def(
        "allocate_tensor_on_device",
        nb::overload_cast<const ttnn::TensorSpec&, IDevice*>(&ttnn::operations::core::allocate_tensor_on_device),
        nb::arg("tensor_spec"),
        nb::arg("device"));

    mod.def(
        "allocate_tensor_on_device",
        [](const ttnn::TensorSpec& spec, MeshDevice* device) {
            return tt::tt_metal::allocate_tensor_on_mesh(spec, device);
        },
        nb::arg("tensor_spec"),
        nb::arg("mesh_device"));

    mod.def(
        "allocate_tensor_on_device",
        nb::overload_cast<
            const ttnn::Shape&,
            ttnn::DataType,
            ttnn::Layout,
            IDevice*,
            const std::optional<ttnn::MemoryConfig>&>(&ttnn::operations::core::allocate_tensor_on_device),
        nb::arg("shape"),
        nb::arg("dtype"),
        nb::arg("layout"),
        nb::arg("device"),
        nb::arg("memory_config") = std::nullopt);

    mod.def(
        "allocate_tensor_on_device",
        [](const ttnn::Shape& shape,
           ttnn::DataType dtype,
           ttnn::Layout layout,
           MeshDevice* device,
           const std::optional<ttnn::MemoryConfig>& mem_config) {
            return tt::tt_metal::allocate_tensor_on_mesh(
                TensorSpec(
                    shape,
                    tt::tt_metal::TensorLayout(
                        dtype, tt::tt_metal::PageConfig(layout), mem_config.value_or(MemoryConfig{}))),
                device);
        },
        nb::arg("shape"),
        nb::arg("dtype"),
        nb::arg("layout"),
        nb::arg("mesh_device"),
        nb::arg("memory_config") = std::nullopt);

    mod.def(
        "copy_host_to_device_tensor",
        [](const ttnn::Tensor& host_tensor, ttnn::Tensor device_tensor, QueueId cq_id = ttnn::DefaultQueueId) {
            // Copies `device_tensor`, to be able to asynchronously populate metadata in tensor attributes, stored as a
            // shared pointer.
            tt::tt_metal::write_tensor(host_tensor, std::move(device_tensor), cq_id);
        },
        nb::arg("host_tensor"),
        nb::arg("device_tensor"),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    bind_registered_operation(
        mod,
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
            device (ttnn.Device | ttnn.MeshDevice): the device/mesh device whose worker thread on the host should be used for the layout conversion.

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
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               IDevice* device) -> ttnn::Tensor { return self(tensor, layout, dtype, memory_config, device); },
            nb::arg("tensor"),
            nb::arg("layout"),
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("device") = nullptr},
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               MeshDevice* device) -> ttnn::Tensor { return self(tensor, layout, dtype, memory_config, device); },
            nb::arg("tensor"),
            nb::arg("layout"),
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("device") = nullptr});

    mod.def(
        "num_cores_to_corerangeset",
        nb::overload_cast<const uint32_t, const CoreCoord, const bool>(&tt::tt_metal::num_cores_to_corerangeset),
        R"doc(Create a CoreRangeSet containing the specified number of cores)doc");

    mod.def(
        "num_cores_to_corerangeset_in_subcoregrids",
        nb::overload_cast<const CoreCoord, const uint32_t, const CoreRangeSet&, const bool>(
            &tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids),
        R"doc(Create a CoreRangeSet containing the specified number of cores starting from start_core in given subcoregrids)doc");
}

}  // namespace ttnn::operations::core 
