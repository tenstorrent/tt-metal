// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::core {

void py_module_types(py::module& module) {
    py::enum_<compute_throttle_utils::ThrottleLevel>(module, "ThrottleLevel", R"doc(
        Enum for controlling compute throttling.

        Higher levels insert NOP instructions to reduce compute throughput:
        - LEVEL_1: Throttle to 73% of max performance
        - LEVEL_2: Throttle to 67% of max performance
        - LEVEL_3: Throttle to 50% of max performance
        - LEVEL_4: Throttle to 40% of max performance
        - LEVEL_5: Throttle to 33% of max performance

        Used to prevent di/dt (power supply current) issues on large core counts.
    )doc")
        .value("NO_THROTTLE", compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
        .value("LEVEL_1", compute_throttle_utils::ThrottleLevel::LEVEL_1)
        .value("LEVEL_2", compute_throttle_utils::ThrottleLevel::LEVEL_2)
        .value("LEVEL_3", compute_throttle_utils::ThrottleLevel::LEVEL_3)
        .value("LEVEL_4", compute_throttle_utils::ThrottleLevel::LEVEL_4)
        .value("LEVEL_5", compute_throttle_utils::ThrottleLevel::LEVEL_5);

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
            py::arg("throttle_level") = compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
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
        py::overload_cast<
            const ttnn::Tensor&,
            MeshDevice*,
            const std::optional<MemoryConfig>&,
            std::optional<ttnn::QueueId>>(&ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt,
        py::kw_only(),
        py::arg("queue_id") = std::nullopt,
        R"doc(
            Copy tensor from host to device.

            Args:
                tensor (ttnn.Tensor): The tensor to be copied from host to device.
                device (ttnn.Device | ttnn.MeshDevice): The target device where the tensor will be copied.
                memory_config (ttnn.MemoryConfig, optional): The memory configuration to use. Defaults to `None`.

            Kwargs:
                queue_id (ttnn.QueueId, optional): The queue id to use. Defaults to `null`.

            Returns:
                ttnn.Tensor: The device tensor copy.
        )doc");

    module.def(
        "from_device",
        &ttnn::operations::core::from_device,
        py::arg("tensor"),
        py::arg("blocking") = true,
        py::kw_only(),
        py::arg("queue_id") = std::nullopt,
        R"doc(
            Copy tensor from device to host.

            Args:
                tensor (ttnn.Tensor): the tensor to be copied from device to host.
                blocking (bool, optional): whether the operation should be blocked until the copy is complete. Defaults to `True`.

            Kwargs:
                queue_id (ttnn.QueueId, optional): The queue id to use. Defaults to `null`.

            Returns:
                ttnn.Tensor: the host tensor copy.
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
            output_tensor (ttnn.Tensor, optional): the optional output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the converted tensor.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::arg("memory_config"),
            py::arg("dtype") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});

    bind_registered_operation(
        module,
        ttnn::to_dtype,
        R"doc(
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
        [](const ttnn::Tensor& host_tensor,
           ttnn::Tensor& device_tensor,
           const std::optional<QueueId>& cq_id = std::nullopt) {
            tt::tt_metal::tensor_impl::copy_to_device(host_tensor, device_tensor, cq_id);
        },
        py::arg("host_tensor"),
        py::arg("device_tensor"),
        py::arg("cq_id") = std::nullopt);

    module.def(
        "copy_device_to_host_tensor",
        [](const ttnn::Tensor& device_tensor,
           ttnn::Tensor& host_tensor,
           bool blocking = true,
           std::optional<ttnn::QueueId> cq_id = std::nullopt) {
            tt::tt_metal::tensor_impl::copy_to_host(device_tensor, host_tensor, blocking, cq_id);
        },
        py::arg("device_tensor"),
        py::arg("host_tensor"),
        py::arg("blocking") = true,
        py::arg("cq_id") = std::nullopt);

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
        )doc",
        ttnn::pybind_overload_t{
            [](const std::decay_t<decltype(ttnn::to_layout)> self,
               const ttnn::Tensor& tensor,
               const ttnn::Layout layout,
               const std::optional<ttnn::DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(tensor, layout, dtype, memory_config, sub_core_grids);
            },
            py::arg("tensor"),
            py::arg("layout"),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});

    module.def(
        "num_cores_to_corerangeset",
        py::overload_cast<const uint32_t, const CoreCoord, const bool>(&tt::tt_metal::num_cores_to_corerangeset),
        R"doc(Create a CoreRangeSet containing the specified number of cores)doc");

    module.def(
        "num_cores_to_corerangeset_in_subcoregrids",
        py::overload_cast<const CoreCoord, const uint32_t, const CoreRangeSet&, const bool>(
            &tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids),
        R"doc(Create a CoreRangeSet containing the specified number of cores starting from start_core in given subcoregrids)doc");

    module.def(
        "split_work_to_cores",
        py::overload_cast<const CoreCoord, const uint32_t, const bool>(&tt::tt_metal::split_work_to_cores),
        py::arg("grid_size"),
        py::arg("units_to_divide"),
        py::arg("row_wise") = false,
        R"doc(
        Split work units across cores in a grid.

        This function divides a specified number of work units across cores in a grid.
        It returns information about how the work is distributed, including core ranges
        for different groups if work cannot be evenly divided.

        Args:
            grid_size (ttnn.CoreCoord): The size of the core grid (x, y dimensions).
            units_to_divide (int): The total number of work units to distribute.
            row_wise (bool, optional): Whether to distribute work by iterating row-wise. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - num_cores (int): Number of cores being used
                - all_cores (CoreRangeSet): All cores involved
                - core_group_1 (CoreRangeSet): Cores doing more work
                - core_group_2 (CoreRangeSet): Cores doing less work (empty if evenly divisible)
                - units_per_core_group_1 (int): Work units per core in group 1
                - units_per_core_group_2 (int): Work units per core in group 2

        Example:
        >>> # Split 100 tiles across an 8x8 core grid
        >>> num_cores, all_cores, core_group_1, core_group_2, units_1, units_2 = \\
        ...     ttnn.split_work_to_cores(ttnn.CoreCoord(8, 8), 100)
        >>> print(f"Using {num_cores} cores, {units_1} units per core in group 1, {units_2} in group 2")
        )doc");

    module.def(
        "split_work_to_cores",
        py::overload_cast<const CoreRangeSet&, const uint32_t, const bool>(&tt::tt_metal::split_work_to_cores),
        py::arg("core_grid"),
        py::arg("units_to_divide"),
        py::arg("row_wise") = false,
        R"doc(
        Split work units across cores in a CoreRangeSet.

        This function divides a specified number of work units across cores in a CoreRangeSet.
        It returns information about how the work is distributed, including core ranges
        for different groups if work cannot be evenly divided.

        Args:
            core_grid (ttnn.CoreRangeSet): The set of core ranges to distribute work across.
            units_to_divide (int): The total number of work units to distribute.
            row_wise (bool, optional): Whether to distribute work by iterating row-wise. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - num_cores (int): Number of cores being used
                - all_cores (CoreRangeSet): All cores involved
                - core_group_1 (CoreRangeSet): Cores doing more work
                - core_group_2 (CoreRangeSet): Cores doing less work (empty if evenly divisible)
                - units_per_core_group_1 (int): Work units per core in group 1
                - units_per_core_group_2 (int): Work units per core in group 2

        Example:
        >>> # Split 100 tiles across an 8x8 core grid
        >>> core_rangeset = ttnn.CoreRangeSet(ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(7,7)))
        >>> num_cores, all_cores, core_group_1, core_group_2, units_1, units_2 = \\
        ...     ttnn.split_work_to_cores(core_rangeset, 100)
        >>> print(f"Using {num_cores} cores, {units_1} units per core in group 1, {units_2} in group 2")
        )doc");
}

}  // namespace ttnn::operations::core
