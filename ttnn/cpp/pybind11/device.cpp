// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;

namespace {
inline void DumpDeviceProfiler(IDevice* device, bool last_dump) {
    tt::tt_metal::detail::DumpDeviceProfileResults(device, last_dump);
}
}  // namespace

namespace ttnn {
namespace device {
namespace detail {

void ttnn_device(py::module& module) {
    module.def(
        "open_device",
        &ttnn::open_device,
        py::kw_only(),
        py::arg("device_id"),
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("dispatch_core_config") = tt::tt_metal::DispatchCoreConfig{},
        py::return_value_policy::reference,
        R"doc(
            Open a device with the given device_id. If the device is already open, return the existing device.

            Keyword Args:
                device_id (int): The device ID to open.
                l1_small_size (int, optional): The size of the L1 small buffer. Defaults to `ttnn.device.DEFAULT_L1_SMALL_SIZE`.
                trace_region_size (int, optional): The size of the trace region. Defaults to `ttnn.device.DEFAULT_TRACE_REGION_SIZE`.
                dispatch_core_type (ttnn.device.DispatchCoreType, optional): The type of dispatch core to use. Defaults to `ttnn.device.DispatchCoreType.WORKER`.

            Returns:
                ttnn.Device: The device with the given device_id.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> print(device)
                <ttnn._ttnn.device.Device object at 0x7fbac5bfc1b0>
        )doc");

    module.def("close_device", &ttnn::close_device, py::arg("device"));

    module.def("enable_program_cache", &ttnn::enable_program_cache, py::arg("device"));

    module.def("disable_and_clear_program_cache", &ttnn::disable_and_clear_program_cache, py::arg("device"));

    module.def("deallocate_buffers", &ttnn::deallocate_buffers, py::arg("device"), R"doc(
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
        .def(py::init<>(), "Default constructor initializing type to WORKER and axis to ROW.")
        .def(
            py::init<tt::tt_metal::DispatchCoreType, tt::tt_metal::DispatchCoreAxis>(),
            "Constructor with specified dispatch core type and axis.",
            py::arg("type"),
            py::arg("axis"));

    py::class_<IDevice, std::unique_ptr<IDevice, py::nodelete>>(
        m_device, "IDevice", "Base class describing a Tenstorrent accelerator device.");

    py::class_<tt::tt_metal::Device, IDevice, std::unique_ptr<Device, py::nodelete>>(
        m_device, "Device", "Class describing a Tenstorrent accelerator device.");

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
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto pyIDevice = static_cast<py::class_<IDevice, std::unique_ptr<IDevice, py::nodelete>>>(m_device.attr("IDevice"))
        .def("id", &IDevice::id, "Device's ID")
        .def("arch", &IDevice::arch, "Device's arch")
        .def(
            "compute_with_storage_grid_size",
            &IDevice::compute_with_storage_grid_size,
            "Grid size (x, y) denoting region that can be targeted by ops")
        .def("dram_grid_size", &IDevice::dram_grid_size, "Grid size (x, y) denoting dram cores that can be targeted")
        .def(
            "worker_core_from_logical_core",
            &IDevice::worker_core_from_logical_core,
            "Convert a logical core coordinate into a physical worker core coordinate")
        .def(
            "enable_program_cache",
            &IDevice::enable_program_cache,
            "Enable caching for all programs sent to this device")
        .def(
            "disable_and_clear_program_cache",
            &IDevice::disable_and_clear_program_cache,
            "Disable and clear program cache for this device")
        .def(
            "num_program_cache_entries",
            &IDevice::num_program_cache_entries,
            "Number of entries in the program cache for this device")
        .def("enable_async", &IDevice::enable_async)
        .def(
            "create_sub_device_manager",
            [](IDevice* device,
               const std::vector<SubDevice>& sub_devices,
               DeviceAddr local_l1_size) -> SubDeviceManagerId {
                SubDeviceManagerId sub_device_manager_id;
                device->push_work(
                    [device, sub_devices, local_l1_size, &sub_device_manager_id] {
                        sub_device_manager_id = device->create_sub_device_manager(sub_devices, local_l1_size);
                    },
                    /*blocking=*/true);
                return sub_device_manager_id;
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given device.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    SubDeviceManagerId: The ID of the created sub-device manager.
            )doc")
        .def(
            "create_sub_device_manager_with_fabric",
            [](IDevice* device, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                std::tuple<SubDeviceManagerId, SubDeviceId> manager_and_sub_device_ids;
                device->push_work(
                    [device, sub_devices, local_l1_size, &manager_and_sub_device_ids] {
                        manager_and_sub_device_ids =
                            device->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
                    },
                    /*blocking=*/true);
                return manager_and_sub_device_ids;
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given device. This will automatically create a sub-device of ethernet cores for use with fabric.
                Note that this is a temporary API until migration to actual fabric is complete.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager. No ethernet cores should be included in this list.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    SubDeviceManagerId: The ID of the created sub-device manager.
                    SubDeviceId: The ID of the sub-device that will be used for fabric.
            )doc")
        .def(
            "load_sub_device_manager",
            [](IDevice* device, SubDeviceManagerId sub_device_manager_id) {
                device->push_work(
                    [device, sub_device_manager_id] { device->load_sub_device_manager(sub_device_manager_id); });
            },
            py::arg("sub_device_manager_id"),
            R"doc(
                Loads the sub-device manager with the given ID.

                Args:
                    sub_device_manager_id (SubDeviceManagerId): The ID of the sub-device manager to load.
            )doc")
        .def(
            "clear_loaded_sub_device_manager",
            [](IDevice* device) { device->push_work([device] { device->clear_loaded_sub_device_manager(); }); },
            R"doc(
                Clears the loaded sub-device manager for the given device.
            )doc")
        .def(
            "remove_sub_device_manager",
            [](IDevice* device, SubDeviceManagerId sub_device_manager_id) {
                device->push_work(
                    [device, sub_device_manager_id] { device->remove_sub_device_manager(sub_device_manager_id); });
            },
            py::arg("sub_device_manager_id"),
            R"doc(
                Removes the sub-device manager with the given ID.

                Args:
                    sub_device_manager_id (SubDeviceManagerId): The ID of the sub-device manager to remove.
            )doc")
        .def(
            "set_sub_device_stall_group",
            [](IDevice* device, const std::vector<SubDeviceId>& sub_device_ids) {
                device->push_work([device, sub_device_ids] { device->set_sub_device_stall_group(sub_device_ids); });
            },
            py::arg("sub_device_ids"),
            R"doc(
                Set the SubDevice IDs that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing.
                Stalling here refers to the Fast Dispatch cores waiting for programs to complete execution on the specified SubDevices before proceeding with the specified instruction.
                The default SubDevice IDs to stall on are set to all SubDevice IDs, and whenever a new SubDevice Manager is loaded.

                Args:
                    sub_device_ids (List[SubDeviceId]): The IDs of the SubDevices to stall on.
            )doc")
        .def(
            "reset_sub_device_stall_group",
            [](IDevice* device) { device->push_work([device] { device->reset_sub_device_stall_group(); }); },
            R"doc(
                Resets the sub_device_ids that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing
                back to all SubDevice IDs.
            )doc")
        .def("sfpu_eps", &IDevice::sfpu_eps, R"doc(Returns machine epsilon value for current device.)doc")
        .def("sfpu_nan", &IDevice::sfpu_nan, R"doc(Returns NaN value for current device.)doc")
        .def("sfpu_inf", &IDevice::sfpu_inf, R"doc(Returns Infinity value for current device.)doc");

    auto pyDevice = static_cast<py::class_<tt::tt_metal::Device, IDevice, std::unique_ptr<tt::tt_metal::Device, py::nodelete>>>(m_device.attr("Device"));
    pyDevice
        .def(
            py::init<>([](int device_id, size_t l1_small_size, size_t trace_region_size) {
                return tt::tt_metal::Device(device_id, 1, l1_small_size, trace_region_size);
            }),
            "Create device.",
            py::arg("device_id"),
            py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
            py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE);

    // *** eps constant ***
    m_device.attr("EPS_GS") = EPS_GS;
    m_device.attr("EPS_WHB0") = EPS_WHB0;
    m_device.attr("EPS_BH") = EPS_BH;

    m_device.attr("NAN_GS") = NAN_GS;
    m_device.attr("NAN_WHB0") = NAN_WHB0;
    m_device.attr("NAN_BH") = NAN_BH;

    m_device.attr("INF_GS") = INF_GS;
    m_device.attr("INF_WHB0") = INF_WHB0;
    m_device.attr("INF_BH") = INF_BH;

    m_device.def(
        "CreateDevice",
        [](int device_id,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
            return tt::tt_metal::CreateDevice(
                device_id, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config);
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
        py::arg("DispatchCoreConfig") = tt::tt_metal::DispatchCoreConfig{});
    m_device.def(
        "CreateDevices",
        [](const std::vector<int>& device_ids,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
            return tt::tt_metal::detail::CreateDevices(
                device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config);
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
        py::arg("DispatchCoreConfig") = tt::tt_metal::DispatchCoreConfig{});
    m_device.def("CloseDevice", &tt::tt_metal::CloseDevice, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | ttnn.Device           |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def("CloseDevices", &tt::tt_metal::detail::CloseDevices, R"doc(
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
        &ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice,
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
        &ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice,
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
        &ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor,
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
        &ttnn::operations::experimental::auto_format::AutoFormat::format_output_tensor,
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
        [](const std::array<uint32_t, 4>& unpadded_shape,
           bool pad_c = false,
           bool pad_n = false,
           bool pad_h = true,
           bool pad_w = true) -> tt::tt_metal::LegacyShape {
            return ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(
                unpadded_shape, pad_c, pad_n, pad_h, pad_w);
        },
        py::arg("unpadded_shape"),
        py::arg("pad_c") = false,
        py::arg("pad_n") = false,
        py::arg("pad_h") = true,
        py::arg("pad_w") = true,
        R"doc(
        Pads the given shape to tile shape based on specified padding options.

        Args:
            unpadded_shape (List of [int]): The original shape of the tensor to pad.
            pad_c (bool, optional): Pad the channel dimension. Defaults to `False`.
            pad_n (bool, optional): Pad the batch dimension. Defaults to `False`.
            pad_h (bool, optional): Pad the height dimension. Defaults to `True`.
            pad_w (bool, optional): Pad the width dimension. Defaults to `True`.

        Returns:
            ttnn.tt_metal.LegacyShape: The padded shape.

        Note:
            This functionality is planned for deprecation in the future.

        Example:
            >>> padded_shape = ttnn.pad_to_tile_shape(unpadded_shape=[1, 2, 2, 2], pad_c=False, pad_n=False, pad_h=True, pad_w=True)

        )doc");

    m_device.def("EnablePersistentKernelCache", &tt::tt_metal::detail::EnablePersistentKernelCache, R"doc(
        Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
    )doc");
    m_device.def("DisablePersistentKernelCache", &tt::tt_metal::detail::DisablePersistentKernelCache, R"doc(
        Disables kernel compilation cache from being persistent across runs
    )doc");
    m_device.def("EnableCompilationReports", &tt::tt_metal::detail::EnableCompilationReports, R"doc(
        Enables tt-metal to generate reports of compilation statistics
    )doc");
    m_device.def("DisableCompilationReports", &tt::tt_metal::detail::DisableCompilationReports, R"doc(
        Disables generation of compilation statistics reports in tt-metal
    )doc");

    m_device.def("EnableMemoryReports", &tt::tt_metal::detail::EnableMemoryReports, R"doc(
        Enables tt-metal to generate reports of memory allocation statistics
    )doc");
    m_device.def("DisableMemoryReports", &tt::tt_metal::detail::DisableMemoryReports, R"doc(
        Disables generation of memory allocation statistics reports in tt-metal
    )doc");

    m_device.def(
        "DumpDeviceMemoryState",
        &tt::tt_metal::detail::DumpDeviceMemoryState,
        py::arg().noconvert(),
        py::arg("prefix").noconvert() = std::string(""),
        R"doc(
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
    )doc");

    m_device.def(
        "GetMemoryView",
        &tt::tt_metal::detail::GetMemoryView,
        py::arg().noconvert(),
        py::arg().noconvert(),
        R"doc(
        Populates MemoryView for BufferType [dram, l1, l1 small, trace]. Used when storing to disk is not an option.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | ttnn.Device           |             | Yes      |
        | buffer_type      | Type of buffer for memory view   | ttnn.BufferType       |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def(
        "synchronize_device",
        [](IDevice* device, const std::optional<uint8_t> cq_id, const std::vector<SubDeviceId>& sub_device_ids) {
            // Send finish command to issue queue through worker thread
            // Worker thread will stall until the device is flushed.
            device->push_work(
                [device, cq_id, &sub_device_ids]() mutable { Synchronize(device, cq_id, sub_device_ids); });
            // Main thread stalls until worker is complete (full device and worker queue flush).
            device->synchronize();
        },
        R"doc(
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
            )doc",
        py::arg("device"),
        py::arg("cq_id") = std::nullopt,
        py::arg("sub_device_ids") = std::vector<SubDeviceId>());
    m_device.def("SetLazyCommandQueueMode", &tt::tt_metal::detail::SetLazyCommandQueueMode, R"doc(
        If set to true, the host does not notify the device that there are commands available other than
        the FinishCommand. Once set to false, all subsequent commands will immediately notify the device
        that the write pointer has been updated.
    )doc");
    m_device.def("DumpDeviceProfiler", DumpDeviceProfiler, py::arg("device"), py::arg("last_dump") = false, R"doc(
        Dump device side profiling data.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump profiling data of | ttnn.Device           |             | Yes      |
        | last_dump        | Last dump before process dies    | bool                  |             | No       |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.attr("DEFAULT_L1_SMALL_SIZE") = py::int_(DEFAULT_L1_SMALL_SIZE);
    m_device.attr("DEFAULT_TRACE_REGION_SIZE") = py::int_(DEFAULT_TRACE_REGION_SIZE);

    m_device.attr("DefaultQueueId") = ttnn::DefaultQueueId;
}

void py_device_module(py::module& module) {
    detail::ttnn_device(module);
    device_module(module);
}

}  // namespace device
}  // namespace ttnn
