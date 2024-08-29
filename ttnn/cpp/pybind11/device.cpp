// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"


namespace py = pybind11;

namespace {
    inline void DumpDeviceProfiler(Device * device, bool last_dump)
    {
        tt::tt_metal::detail::DumpDeviceProfileResults(device, last_dump);
    }
}

namespace ttnn {
namespace device {
namespace detail {
void ttnn_device(py::module& module) {
    module.def(
        "open_device",
        &ttnn::open_device,
        py::kw_only(),
        py::arg("device_id"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("dispatch_core_type"),
        py::return_value_policy::reference);

    module.def("close_device", &ttnn::close_device, py::arg("device"), py::kw_only());

    module.def("enable_program_cache", &ttnn::enable_program_cache, py::arg("device"), py::kw_only());

    module.def("disable_and_clear_program_cache", &ttnn::disable_and_clear_program_cache, py::arg("device"), py::kw_only());

    module.def("deallocate_buffers",
    &ttnn::deallocate_buffers, py::arg("device"), R"doc(
        Deallocate all buffers associated with Device handle
    )doc");
}

void device_module(py::module &m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE);

    py::enum_<tt::tt_metal::DispatchCoreType>(m_device, "DispatchCoreType", "Enum of types of dispatch cores.")
        .value("WORKER", tt::tt_metal::DispatchCoreType::WORKER)
        .value("ETH", tt::tt_metal::DispatchCoreType::ETH);

    auto pyDevice = py::class_<Device, std::unique_ptr<Device, py::nodelete>>(m_device, "Device", "Class describing a Tenstorrent accelerator device.");
    pyDevice
        .def(
            py::init<>([](int device_id, size_t l1_small_size, size_t trace_region_size) { return Device(device_id, 1, l1_small_size, trace_region_size); }),
            "Create device.",
            py::arg("device_id"),
            py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
            py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE)
        .def("id", &Device::id, "Device's ID")
        .def("arch", &Device::arch, "Device's arch")
        .def(
            "compute_with_storage_grid_size",
            &Device::compute_with_storage_grid_size,
            "Grid size (x, y) denoting region that can be targeted by ops")
        .def(
            "dram_grid_size",
            &Device::dram_grid_size,
            "Grid size (x, y) denoting dram cores that can be targeted")
        .def(
            "worker_core_from_logical_core",
            &Device::worker_core_from_logical_core,
            "Convert a logical core coordinate into a physical worker core coordinate")
        .def(
            "enable_program_cache",
            &Device::enable_program_cache,
            "Enable caching for all programs sent to this device")
        .def(
            "disable_and_clear_program_cache",
            &Device::disable_and_clear_program_cache,
            "Disable and clear program cache for this device")
        .def(
            "num_program_cache_entries",
            &Device::num_program_cache_entries,
            "Number of entries in the program cache for this device")
        .def("enable_async", &Device::enable_async);
    // *** eps constant ***
    m_device.attr("EPS_GS") = EPS_GS;
    m_device.attr("EPS_WHB0") = EPS_WHB0;
    m_device.attr("EPS_BH") = EPS_BH;

    pyDevice.def("sfpu_eps", &Device::sfpu_eps, R"doc(
        Machine epsilon value for current device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | return machine epsilon | tt_lib.device.Device  |     NA      | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
        )doc");
    m_device.def(
        "CreateDevice",
        [](int device_id, uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::tt_metal::DispatchCoreType dispatch_core_type) { return tt::tt_metal::CreateDevice(device_id, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type); },
        R"doc(
        Creates an instance of TT device.

        +------------------+------------------------+---------------------+------------------------------+----------+
        | Argument         | Description            | Data type           | Valid range                  | Required |
        +==================+========================+=====================+==============================+==========+
        | device_id        | Device index           | int                 |                              | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
    )doc",
        py::arg("device_id"),
        py::arg("num_hw_cqs") = 1,
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("dispatch_core_type") = tt::tt_metal::DispatchCoreType::WORKER);
    m_device.def(
        "CreateDevices",
        [](std::vector<int> device_ids, uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::tt_metal::DispatchCoreType dispatch_core_type) {
            return tt::tt_metal::detail::CreateDevices(device_ids, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type);
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
        py::arg("num_hw_cqs") = 1,
        py::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        py::arg("dispatch_core_type") = tt::tt_metal::DispatchCoreType::WORKER);
    m_device.def("CloseDevice", &tt::tt_metal::CloseDevice, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def("CloseDevices", &tt::tt_metal::detail::CloseDevices, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | tt_lib.device.Device  |             | Yes      |
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

    m_device.def("SetDefaultDevice", &ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device. This will be deprecated soon.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetDefaultDevice", &ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.This will be deprecated soon.
    )doc");

    m_device.def(
        "format_input_tensor",
        &ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor,
        py::arg("input").noconvert(),
        py::arg("device").noconvert(),
        py::arg("padded_shape"),
        py::arg("pad_value"),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt);

    m_device.def(
        "format_output_tensor",
        &ttnn::operations::experimental::auto_format::AutoFormat::format_output_tensor,
        py::arg("output").noconvert(),
        py::arg("shape"),
        py::arg("device").noconvert(),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt);

    m_device.def(
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape,
        bool pad_c = false,
        bool pad_n = false,
        bool pad_h = true,
        bool pad_w = true) -> tt::tt_metal::Shape {
            return ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w);
        });

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

    m_device.def("DumpDeviceMemoryState", &tt::tt_metal::detail::DumpDeviceMemoryState, py::arg().noconvert(), py::arg("prefix").noconvert() = std::string(""), R"doc(
        Generates reports to dump device memory state. Three reports are generated:
        - `<prefix>l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
        - `<prefix>memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
        - `<prefix>detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | tt_lib.device.Device  |             | Yes      |
        | prefix           | Dumped report filename prefix    | str                   |             | No       |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("Synchronize",
        [] (Device* device, const std::optional<uint8_t> cq_id) {
            // Send finish command to issue queue through worker thread
            // Worker thread will stall until the device is flushed.
            device->push_work([device, cq_id] () mutable {
                Synchronize(device, cq_id);
            });
            // Main thread stalls until worker is complete (full device and worker queue flush).
            device->synchronize();
        }, R"doc(
        Synchronize the device with host by waiting for all operations to complete.
        If cq_id is provided then only the operations associated with that cq_id are waited for,
        otherwise operations for all command queues are waited on.
    )doc",
        py::arg("device"),
        py::arg("cq_id") = std::nullopt);
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
        | device           | Device to dump profiling data of | tt_lib.device.Device  |             | Yes      |
        | last_dump        | Last dump before process dies    | bool                  |             | No       |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.attr("DEFAULT_L1_SMALL_SIZE") = py::int_(DEFAULT_L1_SMALL_SIZE);
    m_device.attr("DEFAULT_TRACE_REGION_SIZE") = py::int_(DEFAULT_TRACE_REGION_SIZE);
}

}  // namespace detail

void py_module(py::module& module) {
   detail::ttnn_device(module);
   detail::device_module(module);
}
}  // namespace device
}  // namespace ttnn
