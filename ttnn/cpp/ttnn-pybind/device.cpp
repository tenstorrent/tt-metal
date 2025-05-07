// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "small_vector_caster.hpp"  // NOLINT - for pybind11 SmallVector binding support.
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/trace.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "tools/profiler/op_profiler.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;

namespace {
void DumpDeviceProfiler(IDevice* device) {
    ProfilerOptionalMetadata prof_metadata(tt::tt_metal::op_profiler::runtime_id_to_opname_.export_map());
    tt::tt_metal::detail::DumpDeviceProfileResults(device, ProfilerDumpState::NORMAL, prof_metadata);
}
}  // namespace

namespace ttnn {
namespace device {
namespace detail {

void ttnn_device(py::module& module) {
    module.def(
        "open_device",
        &ttnn::open_mesh_device,
        py::kw_only(),
        py::arg("device_id"),
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("dispatch_core_config") = tt::tt_metal::DispatchCoreConfig{},
        py::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        py::return_value_policy::reference,
        R"doc(
            Open a device with the given device_id. If the device is already open, return the existing device.

            Keyword Args:
                device_id (int): The device ID to open.
                l1_small_size (int, optional): The size of the L1 small buffer. Defaults to `ttnn.device.DEFAULT_L1_SMALL_SIZE`.
                trace_region_size (int, optional): The size of the trace region. Defaults to `ttnn.device.DEFAULT_TRACE_REGION_SIZE`.
                worker_l1_size (int, optional): The size of the user-allocatable L1 buffer. Defaults to a hardware-specific value.
                dispatch_core_type (ttnn.device.DispatchCoreType, optional): The type of dispatch core to use. Defaults to `ttnn.device.DispatchCoreType.WORKER`.

            Returns:
                ttnn.Device: The device with the given device_id.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> print(device)
                <ttnn._ttnn.device.Device object at 0x7fbac5bfc1b0>
        )doc");

    module.def("close_device", [](MeshDevice& device) { ttnn::close_device(device); }, py::arg("device"));

    module.def(
        "deallocate_buffers", [](MeshDevice* device) { ttnn::deallocate_buffers(device); }, py::arg("device"), R"doc(
        Deallocate all buffers associated with Device handle
    )doc");
}

}  // namespace detail

void py_device_module_types(py::module& m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE);

    py::enum_<tt::tt_metal::DispatchCoreType>(m_device, "DispatchCoreType", "Enum of types of dispatch cores.")
        .value("WORKER", tt::tt_metal::DispatchCoreType::WORKER)
        .value("ETH", tt::tt_metal::DispatchCoreType::ETH);

    py::enum_<tt::tt_metal::DispatchCoreAxis>(
        m_device, "DispatchCoreAxis", "Enum of axis (row or col) of dispatch cores.")
        .value("ROW", tt::tt_metal::DispatchCoreAxis::ROW)
        .value("COL", tt::tt_metal::DispatchCoreAxis::COL);

    py::class_<tt::tt_metal::DispatchCoreConfig>(
        m_device, "DispatchCoreConfig", "Class representing dispatch core configuration.")
        .def(
            py::init<>(),
            "Default constructor initializing type to WORKER and axis to default value on platform architecture.")
        .def(
            py::init<tt::tt_metal::DispatchCoreType>(),
            "Constructor with specified dispatch core type and default axis on platform architecture.",
            py::arg("type"))
        .def(
            py::init<tt::tt_metal::DispatchCoreType, tt::tt_metal::DispatchCoreAxis>(),
            "Constructor with specified dispatch core type and axis.",
            py::arg("type"),
            py::arg("axis"));

    py::class_<SubDevice>(m_device, "SubDevice", "Class describing a sub-device of a Tenstorrent accelerator device.");

    py::class_<SubDeviceId>(m_device, "SubDeviceId", "ID of a sub-device.");

    py::class_<SubDeviceManagerId>(m_device, "SubDeviceManagerId", "ID of a sub-device manager.");

    py::class_<tt::tt_metal::detail::MemoryView>(
        m_device, "MemoryView", "Class representing view of the memory (dram, l1, l1_small, trace) of a device.")
        .def_readonly("num_banks", &tt::tt_metal::detail::MemoryView::num_banks)
        .def_readonly("total_bytes_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_per_bank)
        .def_readonly(
            "total_bytes_allocated_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_allocated_per_bank)
        .def_readonly("total_bytes_free_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_free_per_bank)
        .def_readonly(
            "largest_contiguous_bytes_free_per_bank",
            &tt::tt_metal::detail::MemoryView::largest_contiguous_bytes_free_per_bank)
        .def_readonly("block_table", &tt::tt_metal::detail::MemoryView::block_table);
}

void device_module(py::module& m_device) {
    auto pySubDevice = static_cast<py::class_<SubDevice>>(m_device.attr("SubDevice"));
    pySubDevice.def(
        py::init<>([](std::vector<CoreRangeSet> cores) { return SubDevice(cores); }),
        py::arg("cores"),
        R"doc(
            Creates a SubDevice object from a list of CoreRangeSet objects, where each CoreRangeSet object
            represents the cores from a specific CoreType.
            The order of cores is Tensix, then Ethernet.
        )doc");

    auto pySubDeviceId = static_cast<py::class_<SubDeviceId>>(m_device.attr("SubDeviceId"));
    pySubDeviceId
        .def(
            py::init<uint8_t>(),
            py::arg("id"),
            R"doc(
            Creates a SubDeviceId object with the given ID.
        )doc")
        .def(
            "__repr__",
            [](const SubDeviceId& self) { return "SubDeviceId(" + std::to_string(static_cast<int>(*self)) + ")"; })
        .def(py::self == py::self)
        .def(py::self != py::self);

    m_device.def(
        "CreateDevice",
        [](int device_id,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
           size_t worker_l1_size) {
            return MeshDevice::create_unit_mesh(
                device_id,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config,
                /*l1_bank_remap=*/{},
                worker_l1_size);
        },
        R"doc(
        Creates an instance of TT device.

        +------------------+------------------------+---------------------+------------------------------+----------+
        | Argument         | Description            | Data type           | Valid range                  | Required |
        +==================+========================+=====================+==============================+==========+
        | device_id        | Device index           | int                 |                              | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
    )doc",
        py::arg("device_id"),
        py::arg("num_command_queues") = 1,
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("DispatchCoreConfig") = tt::tt_metal::DispatchCoreConfig{},
        py::kw_only(),
        py::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    m_device.def(
        "CreateDevices",
        [](const std::vector<int>& device_ids,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
           size_t worker_l1_size) {
            return MeshDevice::create_unit_meshes(
                device_ids,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config,
                /*l1_bank_remap=*/{},
                worker_l1_size);
        },
        R"doc(
        Creates an instance of TT device.

        +------------------+------------------------+---------------------+------------------------------+----------+
        | Argument         | Description            | Data type           | Valid range                  | Required |
        +==================+========================+=====================+==============================+==========+
        | device_id        | Device index           | int                 |                              | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
    )doc",
        py::arg("device_ids"),
        py::arg("num_command_queues") = 1,
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("DispatchCoreConfig") = tt::tt_metal::DispatchCoreConfig{},
        py::kw_only(),
        py::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    m_device.def("CloseDevice", [](MeshDevice* device) { device->close(); }, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | ttnn.Device           |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def(
        "CloseDevices",
        [](const std::map<chip_id_t, MeshDevice*>& devices) {
            for (const auto& device_entry : devices) {
                device_entry.second->close();
            }
        },
        R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | ttnn.Device           |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetNumAvailableDevices", &tt::tt_metal::GetNumAvailableDevices, R"doc(
        Returns number of Tenstorrent devices that can be targeted.
    )doc");

    m_device.def("GetNumPCIeDevices", &tt::tt_metal::GetNumPCIeDevices, R"doc(
        Returns number of Tenstorrent devices that are connected to host via PCIe and can be targeted.
    )doc");

    m_device.def("GetPCIeDeviceID", &tt::tt_metal::GetPCIeDeviceID, R"doc(
        Returns associated mmio device of give device id.
    )doc");

    m_device.def(
        "SetDefaultDevice",
        [](MeshDevice* device) { ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice(device); },
        R"doc(
            Sets the default device to use for operations when inputs are not on the device.

            Args:
                device (ttnn.Device): The TT device to use.

            Note:
                This functionality is planned for deprecation in the future.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id = device_id)
                >>> ttnn.SetDefaultDevice(device)
        )doc");

    m_device.def(
        "GetDefaultDevice",
        []() {
            return dynamic_cast<MeshDevice*>(
                ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice());
        },
        R"doc(
            Gets the default device to use for ops when inputs aren't on device.

            Returns:
                ttnn.Device: The default device to use.

            Note:
                This functionality is planned for deprecation in the future.

            Example:
                >>> device = ttnn.GetDefaultDevice()
        )doc");

    m_device.def(
        "format_input_tensor",
        [](const Tensor& input,
           MeshDevice* device,
           const ttnn::Shape& padded_shape,
           float pad_value,
           Layout target_layout,
           std::optional<MemoryConfig> target_mem_config) {
            return ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
                input, device, padded_shape, pad_value, target_layout, std::move(target_mem_config));
        },
        py::arg("input").noconvert(),
        py::arg("device").noconvert(),
        py::arg("padded_shape"),
        py::arg("pad_value"),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
        Formats tensor to target layout and pads to padded shape.

        Args:
            input (ttnn.Tensor): Input tensor to format.
            device (ttnn.device.Device): Device where the tensor will be moved.
            padded_shape (ttnn.Shape): Desired shape of the tensor.
            pad_value (float): Value to pad with.
            target_layout (ttnn.Layout): Desired tensor layout.
            target_mem_config (ttnn.MemoryConfig, optional): Desired memory config. Defaults to `None`.

        Returns:
            ttnn.Tensor: Formatted tensor.

        Note:
            This functionality is planned for deprecation in the future.

        Example:
            >>> input_tensor = ttnn.ones([1, 2, 2, 2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            >>> padded_tensor = ttnn.format_input_tensor(input_tensor, device=device, padded_shape=[1, 2, 4, 4], pad_value=0.0, target_layout=ttnn.TILE_LAYOUT, output_mem_config)
        )doc");

    m_device.def(
        "format_output_tensor",
        [](const Tensor& output,
           const ttnn::SmallVector<uint32_t>& shape,
           MeshDevice* device,
           Layout target_layout,
           std::optional<MemoryConfig> target_mem_config) {
            return operations::experimental::auto_format::AutoFormat::format_output_tensor(
                output, ttnn::Shape(shape), device, target_layout, std::move(target_mem_config));
        },
        py::arg("output").noconvert(),
        py::arg("shape"),
        py::arg("device").noconvert(),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
        Formats tensor to target layout and unpads to shape.

        Args:
            output (ttnn.Tensor): Output tensor to format.
            shape (ttnn.Shape): Desired shape of the tensor.
            device (ttnn.device.Device): Device where the tensor will be moved.
            target_layout (ttnn.Layout): Desired tensor layout.
            target_mem_config (ttnn.MemoryConfig, optional): Desired memory config. Defaults to `None`.

        Returns:
            ttnn.Tensor: Formatted tensor.

        Note:
            This functionality is planned for deprecation in the future.

        Example:
            >>> # Assuming we have a padded tensor of shape [1, 2, 4, 4] with padding of [1, 1, 1, 1] of layout=ttnn.TILE_LAYOUT
            >>> unpadded_tensor = ttnn.format_output_tensor(output_tensor, shape=[1, 2, 2, 2], device=device, target_layout=ttnn.ROW_MAJOR_LAYOUT, output_mem_config)
        )doc");

    m_device.def(
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape) -> std::vector<uint32_t> {
            auto result =
                ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(ttnn::Shape(unpadded_shape));
            return std::vector<uint32_t>(result.cbegin(), result.cend());
        },
        py::arg("unpadded_shape"),
        R"doc(
        Pads the given shape to tile shape based on specified padding options.

        Args:
            unpadded_shape (List of [int]): The original shape of the tensor to pad.

        Returns:
            List of [int]: The padded shape.

        Note:
            This functionality is planned for deprecation in the future.

        Example:
            >>> padded_shape = ttnn.pad_to_tile_shape(unpadded_shape=[1, 2, 2, 2])

        )doc");

    m_device.def("EnablePersistentKernelCache", &tt::tt_metal::detail::EnablePersistentKernelCache, R"doc(
        Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
    )doc");
    m_device.def("DisablePersistentKernelCache", &tt::tt_metal::detail::DisablePersistentKernelCache, R"doc(
        Disables kernel compilation cache from being persistent across runs
    )doc");
    m_device.def("EnableMemoryReports", &tt::tt_metal::detail::EnableMemoryReports, R"doc(
        Enables tt-metal to generate reports of memory allocation statistics
    )doc");
    m_device.def("DisableMemoryReports", &tt::tt_metal::detail::DisableMemoryReports, R"doc(
        Disables generation of memory allocation statistics reports in tt-metal
    )doc");

    constexpr std::string_view dump_device_memory_state_doc = R"doc(
        Generates reports to dump device memory state. Three reports are generated:
        - `<prefix>l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
        - `<prefix>memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
        - `<prefix>detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | ttnn.Device           |             | Yes      |
        | prefix           | Dumped report filename prefix    | str                   |             | No       |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc";
    m_device.def(
        "DumpDeviceMemoryState",
        &tt::tt_metal::detail::DumpDeviceMemoryState,
        py::arg().noconvert(),
        py::arg("prefix").noconvert() = std::string(""),
        dump_device_memory_state_doc.data());
    m_device.def(
        "DumpDeviceMemoryState",
        [](MeshDevice* device, const std::string& prefix) {
            tt::tt_metal::detail::DumpDeviceMemoryState(device, prefix);
        },
        py::arg().noconvert(),
        py::arg("prefix").noconvert() = std::string(""),
        dump_device_memory_state_doc.data());

    constexpr std::string_view get_memory_view_doc = R"doc(
        Populates MemoryView for BufferType [dram, l1, l1 small, trace]. Used when storing to disk is not an option.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | ttnn.Device           |             | Yes      |
        | buffer_type      | Type of buffer for memory view   | ttnn.BufferType       |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc";
    m_device.def(
        "GetMemoryView",
        &tt::tt_metal::detail::GetMemoryView,
        py::arg().noconvert(),
        py::arg().noconvert(),
        get_memory_view_doc.data());
    m_device.def(
        "GetMemoryView",
        [](MeshDevice* device, const BufferType& buffer_type) {
            return tt::tt_metal::detail::GetMemoryView(device, buffer_type);
        },
        py::arg().noconvert(),
        py::arg().noconvert(),
        get_memory_view_doc.data());

    constexpr std::string_view synchronize_device_doc = R"doc(
                Synchronize the device with host by waiting for all operations to complete.
                If cq_id is provided then only the operations associated with that cq_id are waited for,
                otherwise operations for all command queues are waited on.
                If the device has been configured with sub-devices, then sub_device_ids can be provided to only wait
                for the operations that ran on the specified sub-devices, otherwise all sub-devices (the entire chip) are waited on.

                Args:
                    device (ttnn.device.Device): The device to synchronize with.
                    cq_id (int, optional): The command queue ID to synchronize. Defaults to `None`.
                    sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to synchronize. Defaults to sub-devices set by set_sub_device_stall_group.

                Returns:
                    `None`: The op ensures that all operations are completed.

                Example:
                    >>> device_id = 0
                    >>> device = ttnn.open_device(device_id=device_id)
                    >>> # Assume some operations are queued on the device
                    >>> ttnn.synchronize_device(device)
            )doc";
    m_device.def(
        "synchronize_device",
        [](IDevice* device, std::optional<QueueId> cq_id, const std::vector<SubDeviceId>& sub_device_ids) {
            Synchronize(device, cq_id.has_value() ? std::make_optional(**cq_id) : std::nullopt, sub_device_ids);
        },
        synchronize_device_doc.data(),
        py::arg("device"),
        py::arg("cq_id") = std::nullopt,
        py::arg("sub_device_ids") = std::vector<SubDeviceId>());
    m_device.def(
        "synchronize_device",
        [](MeshDevice* device, std::optional<QueueId> cq_id, const std::vector<SubDeviceId>& sub_device_ids) {
            tt::tt_metal::distributed::Synchronize(
                device, cq_id.has_value() ? std::make_optional(**cq_id) : std::nullopt, sub_device_ids);
        },
        synchronize_device_doc.data(),
        py::arg("device"),
        py::arg("cq_id") = std::nullopt,
        py::arg("sub_device_ids") = std::vector<SubDeviceId>());
    m_device.def(
        "DumpDeviceProfiler",
        [](MeshDevice* mesh_device) {
            for (auto device : mesh_device->get_devices()) {
                DumpDeviceProfiler(device);
            }
        },
        py::arg("device"),
        R"doc(
        Dump device side profiling data.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump profiling data of | ttnn.Device           |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("get_arch_name", &tt::tt_metal::hal::get_arch_name, "Return the name of the architecture present.");

    m_device.attr("DEFAULT_L1_SMALL_SIZE") = py::int_(DEFAULT_L1_SMALL_SIZE);
    m_device.attr("DEFAULT_TRACE_REGION_SIZE") = py::int_(DEFAULT_TRACE_REGION_SIZE);
    m_device.attr("DEFAULT_WORKER_L1_SIZE") = py::int_(DEFAULT_WORKER_L1_SIZE);

    m_device.def(
        "get_max_worker_l1_unreserved_size",
        &tt::tt_metal::hal::get_max_worker_l1_unreserved_size,
        "Return the maximum size of the worker L1 unreserved memory.");

    m_device.attr("DefaultQueueId") = ttnn::DefaultQueueId;
}

void py_device_module(py::module& module) {
    detail::ttnn_device(module);
    device_module(module);
}

}  // namespace device
}  // namespace ttnn
