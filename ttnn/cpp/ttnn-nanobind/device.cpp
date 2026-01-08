// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/device.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "small_vector_caster.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
// #include "ttnn/operations/experimental/auto_format/auto_format.hpp" // TODO_NANOBIND
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/experimental/kernel_cache.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;
namespace nb = nanobind;

// NOLINTBEGIN(bugprone-unused-raii)

namespace {

void ttnn_device(nb::module_& mod) {
    mod.def(
        "open_device",
        &ttnn::open_mesh_device,
        nb::sig("def open_device(\\*, device_id: int, l1_small_size: int, trace_region_size: int, "
                "dispatch_core_config: ttnn.device.DispatchCoreConfig, worker_l1_size: int)"),
        nb::kw_only(),
        nb::arg("device_id"),
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("num_command_queues") = 1,
        nb::arg("dispatch_core_config") = nb::cast(tt::tt_metal::DispatchCoreConfig{}),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        nb::rv_policy::reference,  // cleanup has to happen in c++ land
        R"doc(
            Open a device with the given device_id. If the device is already open, return the existing device.

            Keyword Args:
                device_id (int): The device ID to open.
                l1_small_size (int, optional): The size of the L1 small buffer. Defaults to `ttnn.device.DEFAULT_L1_SMALL_SIZE`.
                trace_region_size (int, optional): The size of the trace region. Defaults to `ttnn.device.DEFAULT_TRACE_REGION_SIZE`.
                num_command_queues (int, optional): The number of command queues to open. Defaults to 1.
                dispatch_core_config (ttnn.device.DispatchCoreConfig, optional): The dispatch core config to use. Defaults to a hardware-specific value.
                worker_l1_size (int, optional): The size of the user-allocatable L1 buffer. Defaults to a hardware-specific value.

            Returns:
                ttnn.Device: The device with the given device_id.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> print(device)
                <ttnn._ttnn.device.Device object at 0x7fbac5bfc1b0>
        )doc");

    mod.def("close_device", [](ttnn::MeshDevice& device) { ttnn::close_device(device); }, nb::arg("device"));

    mod.def(
        "deallocate_buffers",
        [](ttnn::MeshDevice* device) { ttnn::deallocate_buffers(device); },
        nb::arg("device"),
        R"doc(
        Deallocate all buffers associated with Device handle
    )doc");
}

}  // namespace

namespace ttnn::device {

void py_device_module_types(nb::module_& m_device) {
    nb::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE);

    nb::enum_<tt::tt_metal::DispatchCoreType>(m_device, "DispatchCoreType", "Enum of types of dispatch cores.")
        .value("WORKER", tt::tt_metal::DispatchCoreType::WORKER)
        .value("ETH", tt::tt_metal::DispatchCoreType::ETH);

    nb::enum_<tt::tt_metal::DispatchCoreAxis>(
        m_device, "DispatchCoreAxis", "Enum of axis (row or col) of dispatch cores.")
        .value("ROW", tt::tt_metal::DispatchCoreAxis::ROW)
        .value("COL", tt::tt_metal::DispatchCoreAxis::COL);

    nb::class_<tt::tt_metal::DispatchCoreConfig>(
        m_device, "DispatchCoreConfig", "Class representing dispatch core configuration.")
        .def(
            nb::init<>(),
            "Default constructor initializing type to WORKER and axis to default value on platform architecture.")
        .def(
            nb::init<tt::tt_metal::DispatchCoreType>(),
            "Constructor with specified dispatch core type and default axis on platform architecture.",
            nb::arg("type"))
        .def(
            nb::init<tt::tt_metal::DispatchCoreType, tt::tt_metal::DispatchCoreAxis>(),
            "Constructor with specified dispatch core type and axis.",
            nb::arg("type"),
            nb::arg("axis"));

    nb::class_<SubDevice>(m_device, "SubDevice", "Class describing a sub-device of a Tenstorrent accelerator device.");

    nb::class_<SubDeviceId>(m_device, "SubDeviceId", "ID of a sub-device.");

    nb::class_<SubDeviceManagerId>(m_device, "SubDeviceManagerId", "ID of a sub-device manager.");

    nb::class_<tt::tt_metal::detail::MemoryView>(
        m_device, "MemoryView", "Class representing view of the memory (dram, l1, l1_small, trace) of a device.")
        .def_ro("num_banks", &tt::tt_metal::detail::MemoryView::num_banks)
        .def_ro("total_bytes_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_per_bank)
        .def_ro("total_bytes_allocated_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_allocated_per_bank)
        .def_ro("total_bytes_free_per_bank", &tt::tt_metal::detail::MemoryView::total_bytes_free_per_bank)
        .def_ro(
            "largest_contiguous_bytes_free_per_bank",
            &tt::tt_metal::detail::MemoryView::largest_contiguous_bytes_free_per_bank)
        .def_ro("block_table", &tt::tt_metal::detail::MemoryView::block_table);
}

void device_module(nb::module_& m_device) {
    auto pySubDevice = static_cast<nb::class_<SubDevice>>(m_device.attr("SubDevice"));
    pySubDevice.def(
        "__init__",
        [](SubDevice* t, const std::vector<CoreRangeSet>& cores) { new (t) SubDevice(cores); },
        nb::arg("cores"),
        R"doc(
            Creates a SubDevice object from a list of CoreRangeSet objects, where each CoreRangeSet object
            represents the cores from a specific CoreType.
            The order of cores is Tensix, then Ethernet.
        )doc");

    auto pySubDeviceId = static_cast<nb::class_<SubDeviceId>>(m_device.attr("SubDeviceId"));
    pySubDeviceId
        .def(
            nb::init<uint8_t>(),
            nb::arg("id"),
            R"doc(
            Creates a SubDeviceId object with the given ID.
        )doc")
        .def(
            "__repr__",
            [](const SubDeviceId& self) { return "SubDeviceId(" + std::to_string(static_cast<int>(*self)) + ")"; })
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

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
        nb::arg("device_id"),
        nb::arg("num_command_queues") = 1,
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("DispatchCoreConfig") = nb::cast(tt::tt_metal::DispatchCoreConfig{}),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
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
        nb::arg("device_ids"),
        nb::arg("num_command_queues") = 1,
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("DispatchCoreConfig") = nb::cast(tt::tt_metal::DispatchCoreConfig{}),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
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
        [](const std::map<tt::ChipId, MeshDevice*>& devices) {
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

    m_device.def("SetRootDir", &tt::tt_metal::SetRootDir, nb::arg("root_dir"), R"doc(
        Sets the root directory for TT Metal operations.
        Args:
            root_dir (str): Path to the root directory to set.
        Example:
            >>> ttnn.device.SetRootDir("/path/to/tt_metal_home")
    )doc");

    m_device.def(  // afuller
        "SetDefaultDevice",
        [](std::optional<MeshDevice*> device) { ttnn::SetDefaultDevice(device.value_or(nullptr)); },
        nb::arg("device") = nb::none(),
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

    m_device.def(  // afuller
        "ClearDefaultDevice",
        []() { ttnn::SetDefaultDevice(nullptr); },
        R"doc(Clears the default device (sets it to None).)doc");

    m_device.def(
        "GetDefaultDevice",
        []() { return ttnn::GetDefaultDevice(); },
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
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape) -> std::vector<uint32_t> {
            auto result = ttnn::operations::data_movement::pad_to_tile_shape(ttnn::Shape(unpadded_shape));
            return std::vector<uint32_t>(result.cbegin(), result.cend());
        },
        nb::arg("unpadded_shape"),
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

    m_device.def("ClearKernelCache", &tt::tt_metal::experimental::ClearKernelCache, R"doc(
        Clear the in-memory kernel compilation hash lookup cache.

        Note:
            This only clears the in-memory HashLookup cache.
            The compiler rebuilds binaries when:
            1. Kernel hash is NOT in HashLookup (cleared by this function), AND
            2. Binaries do not exist on disk (or persistent cache is disabled)

            To also clear disk-cached
                kernel binaries, you must delete the files in:
                ~/.cache/tt-metal-cache/<git_hash>/<build_id>/kernels/

        Example:
            >>> import ttnn
            >>> ttnn.device.ClearKernelCache()
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
        nb::arg("device").noconvert(),
        nb::arg("prefix").noconvert() = std::string(""),
        dump_device_memory_state_doc.data());
    m_device.def(
        "DumpDeviceMemoryState",
        [](MeshDevice* device, const std::string& prefix) {
            tt::tt_metal::detail::DumpDeviceMemoryState(device, prefix);
        },
        nb::arg("device").noconvert(),
        nb::arg("prefix").noconvert() = std::string(""),
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
        nb::arg("device").noconvert(),
        nb::arg("buffer_type").noconvert(),
        get_memory_view_doc.data());
    m_device.def(
        "GetMemoryView",
        [](MeshDevice* device, const BufferType& buffer_type) {
            return tt::tt_metal::detail::GetMemoryView(device, buffer_type);
        },
        nb::arg("device").noconvert(),
        nb::arg("buffer_type").noconvert(),
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
        [](MeshDevice* device, std::optional<QueueId> cq_id, const std::vector<SubDeviceId>& sub_device_ids) {
            tt::tt_metal::distributed::Synchronize(device, raw_optional(cq_id), sub_device_ids);
        },
        synchronize_device_doc.data(),
        nb::arg("device"),
        nb::arg("cq_id") = nb::none(),
        nb::arg("sub_device_ids") = std::vector<SubDeviceId>());
    m_device.def(
        "ReadDeviceProfiler",
        [](MeshDevice* mesh_device) {
            ProfilerOptionalMetadata prof_metadata(tt::tt_metal::op_profiler::runtime_id_to_opname_.export_map());
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device, ProfilerReadState::NORMAL, prof_metadata);
        },
        nb::arg("device"),
        R"doc(
        Read device side profiling data.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to read profiling data of | ttnn.Device           |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def(
        "get_arch_name",
        &tt::tt_metal::detail::get_platform_architecture_name,
        "Return the name of the architecture present.");

    m_device.attr("DEFAULT_L1_SMALL_SIZE") = nb::int_(DEFAULT_L1_SMALL_SIZE);
    m_device.attr("DEFAULT_TRACE_REGION_SIZE") = nb::int_(DEFAULT_TRACE_REGION_SIZE);
    m_device.attr("DEFAULT_WORKER_L1_SIZE") = nb::int_(DEFAULT_WORKER_L1_SIZE);

    m_device.def(
        "get_max_worker_l1_unreserved_size",
        &tt::tt_metal::hal::get_max_worker_l1_unreserved_size,
        "Return the maximum size of the worker L1 unreserved memory.");
}

void py_device_module(nb::module_& mod) {
    ttnn_device(mod);
    device_module(mod);
}

}  // namespace ttnn::device

// NOLINTEND(bugprone-unused-raii)
