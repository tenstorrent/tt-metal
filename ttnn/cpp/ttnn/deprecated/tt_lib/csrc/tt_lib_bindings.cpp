// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings.hpp"

#include "operations/module.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "type_caster.hpp"

namespace py = pybind11;

namespace tt {

namespace tt_metal {

void DeviceModule(py::module &m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE);

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
        [](int device_id, uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size) { return CreateDevice(device_id, num_hw_cqs, l1_small_size, trace_region_size); },
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
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE);
    m_device.def(
        "CreateDevices",
        [](std::vector<int> device_ids, uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size) {
            return tt::tt_metal::detail::CreateDevices(device_ids, num_hw_cqs, l1_small_size, trace_region_size);
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
        py::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE);
    m_device.def("CloseDevice", &CloseDevice, R"doc(
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

    m_device.def("GetNumAvailableDevices", &GetNumAvailableDevices, R"doc(
        Returns number of Tenstorrent devices that can be targeted.
    )doc");

    m_device.def("GetNumPCIeDevices", &GetNumPCIeDevices, R"doc(
        Returns number of Tenstorrent devices that are connected to host via PCIe and can be targeted.
    )doc");

    m_device.def("GetPCIeDeviceID", &GetPCIeDeviceID, R"doc(
        Returns associated mmio device of give device id.
    )doc");

    m_device.def("SetDefaultDevice", &AutoFormat::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetDefaultDevice", &AutoFormat::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.
    )doc");

    m_device.def("EnablePersistentKernelCache", &detail::EnablePersistentKernelCache, R"doc(
        Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
    )doc");
    m_device.def("DisablePersistentKernelCache", &detail::DisablePersistentKernelCache, R"doc(
        Disables kernel compilation cache from being persistent across runs
    )doc");
    m_device.def("EnableCompilationReports", &detail::EnableCompilationReports, R"doc(
        Enables tt-metal to generate reports of compilation statistics
    )doc");
    m_device.def("DisableCompilationReports", &detail::DisableCompilationReports, R"doc(
        Disables generation of compilation statistics reports in tt-metal
    )doc");

    m_device.def("EnableMemoryReports", &detail::EnableMemoryReports, R"doc(
        Enables tt-metal to generate reports of memory allocation statistics
    )doc");
    m_device.def("DisableMemoryReports", &detail::DisableMemoryReports, R"doc(
        Disables generation of memory allocation statistics reports in tt-metal
    )doc");

    m_device.def("DumpDeviceMemoryState", &detail::DumpDeviceMemoryState, py::arg().noconvert(), py::arg("prefix").noconvert() = std::string(""), R"doc(
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
    m_device.def("SetLazyCommandQueueMode", &detail::SetLazyCommandQueueMode, R"doc(
        If set to true, the host does not notify the device that there are commands available other than
        the FinishCommand. Once set to false, all subsequent commands will immediately notify the device
        that the write pointer has been updated.
    )doc");
    m_device.def("DumpDeviceProfiler", &detail::DumpDeviceProfiler, py::arg("device"), py::arg("last_dump") = false, R"doc(
        Dump device side profiling data.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump profiling data of | tt_lib.device.Device  |             | Yes      |
        | last_dump        | Last dump before process dies    | bool                  |             | No       |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");
    m_device.def("DeallocateBuffers",
    [] (Device* device) {
        device->push_work([device] () mutable {
            device->deallocate_buffers();
        });
    }, R"doc(
        Deallocate all buffers associated with Device handle
    )doc");
    m_device.def("BeginTraceCapture",
        [] (Device* device, const uint8_t cq_id) {
            uint32_t tid = Trace::next_id();
            device->push_work([device, cq_id, tid] () mutable {
                device->begin_trace(cq_id, tid);
            });
            return tid;
        }, R"doc(
        Begin trace capture on Device handle
    )doc");
    m_device.def("EndTraceCapture",
        [] (Device* device, const uint8_t cq_id, const uint32_t tid) {
            device->push_work([device, cq_id, tid] () mutable {
                device->end_trace(cq_id, tid);
            });
        }, R"doc(
        End trace capture on Device handle
    )doc");
    m_device.def("ReplayTrace",
        [] (Device* device, const uint8_t cq_id, const uint32_t tid, bool blocking) {
            // If blocking, ensure that worker thread blocks until trace is completed
            device->push_work([device, cq_id, tid, blocking] {
                device->replay_trace(cq_id, tid, blocking);
            });
            // If blocking, wait until worker threads have completed
            if (blocking) {
                device->synchronize();
            }
        }, R"doc(
        Replay captured trace on Device handle
    )doc");
    m_device.def("ReleaseTrace",
        [] (Device* device, const uint32_t tid) {
            device->push_work([device, tid] {
                device->release_trace(tid);
            });
        }, R"doc(
        Release captured Trace on Device handle
    )doc");

    auto pyEvent = py::class_<Event, std::shared_ptr<Event>>(m_device, "Event", "Event class");
    m_device.def("CreateEvent",
        [] () {
            return std::make_shared<Event>();
        }, R"doc(
        Create new event
    )doc");
    m_device.def("RecordEvent",
        [] (Device* device, const uint8_t cq_id, std::shared_ptr<Event> event) {
            device->push_work([device, cq_id, event] {
                EnqueueRecordEvent(device->command_queue(cq_id), event);
            });
        }, R"doc(
        Record an event
    )doc");
    m_device.def("WaitForEvent",
        [] (Device* device, const uint8_t cq_id, std::shared_ptr<Event> event) {
            device->push_work([device, cq_id, event] {
                EnqueueWaitForEvent(device->command_queue(cq_id), event);
            });
        }, R"doc(
        Wait for an event
    )doc");

    m_device.attr("DEFAULT_L1_SMALL_SIZE") = py::int_(DEFAULT_L1_SMALL_SIZE);
    m_device.attr("DEFAULT_TRACE_REGION_SIZE") = py::int_(DEFAULT_TRACE_REGION_SIZE);
}

void ProfilerModule(py::module &m_profiler) {
    m_profiler.def("start_tracy_zone",&op_profiler::start_tracy_zone,
            py::arg("source"), py::arg("functName"),py::arg("lineNum"), py::arg("color") = 0, R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | source           | Source file for the zone                       | string                |             | Yes      |
        | functName        | Function of the zone                           | string                |             | Yes      |
        | lineNum          | Line number of the zone marker                 | int                   |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("stop_tracy_zone",&op_profiler::stop_tracy_zone, py::arg("name") = "", py::arg("color") = 0, R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "tracy_message",
        &op_profiler::tracy_message,
        py::arg("message"),
        py::arg("color") = 0xf0f8ff,
        R"doc(
        Emit a message signpost into the tracy profile.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | message          | Message description for this signpost.         | string                |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "tracy_frame",
        &op_profiler::tracy_frame,
        R"doc(
        Emit a tracy frame signpost.
    )doc");
}

} // end namespace tt_metal

void bind_deprecated(py::module m) {
    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    tt::tt_metal::TensorModule(m_tensor);

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::tt_metal::DeviceModule(m_device);

    py::module_ m_profiler = m.def_submodule("profiler", "Submodule defining the profiler");
    tt::tt_metal::ProfilerModule(m_profiler);

    py::module_ m_operations = m.def_submodule("operations", "Submodule for experimental operations");
    tt::operations::py_module(m_operations);

#if defined(TRACY_ENABLE)
    py::function tracy_decorator = py::module::import("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_device);
    tracy_decorator(m_tensor);
    tracy_decorator(m_operations);
#endif
}

} // end namespace tt
