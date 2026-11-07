// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/device.hpp"

#include <tt-metalium/device_types.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "small_vector_caster.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
// #include "ttnn/operations/experimental/auto_format/auto_format.hpp" // TODO_NANOBIND
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/kernel_cache.hpp>
#include <tt-metalium/experimental/operation_timeout.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_state.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;
namespace nb = nanobind;

// NOLINTBEGIN(bugprone-unused-raii)

namespace {

// Prevent Python callback GC while registered.  Keyed by handle returned from C++
// RegisterProgramRealtimeProfilerCallback. Access only from the Python thread (always under GIL), so no mutex needed.
std::unordered_map<uint64_t, PyObject*> python_realtime_callback_refs;

struct PythonProgramRealtimeRecordBatch {
    std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord> records;
    uint64_t dropped = 0;
};

namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;

// Device duration of each program that runs while the collector is started, from the streaming profiler.
// Aggregates the firmware "*-KERNEL" zones (one per RISC per program launch) into one record per
// (chip, runtime id, execution): first kernel start to last kernel end over every core and RISC of that
// program, on the host steady_clock timeline. Same aggregation as the streaming profiler's ops CSV.
// Aggregation runs in C++ on the profiler's callback thread, so Python never sees per-zone traffic.
class StreamingProgramCollector {
public:
    static constexpr std::size_t kNumProcessors = 7;  // streaming_profiler::Processor

    struct ProgramRecord {
        uint32_t chip_id = 0;
        uint32_t runtime_id = 0;
        uint32_t execution = 0;  // k-th launch of this runtime id on this chip since start()
        int64_t start_ns = std::numeric_limits<int64_t>::max();  // steady_clock, since epoch
        int64_t end_ns = std::numeric_limits<int64_t>::min();
        uint64_t start_cycles = std::numeric_limits<uint64_t>::max();
        uint64_t end_cycles = 0;
        std::array<int64_t, kNumProcessors> processor_start_ns{};
        std::array<int64_t, kNumProcessors> processor_end_ns{};
        std::set<uint32_t> cores;                 // physical (y << 16) | x
        std::vector<std::string> kernel_sources;  // filled by records()
        // Sums over the program's "*-KERNEL" zones, for the average device clock while it ran.
        uint64_t zone_cycles = 0;
        int64_t zone_ns = 0;

        ProgramRecord() {
            processor_start_ns.fill(std::numeric_limits<int64_t>::max());
            processor_end_ns.fill(std::numeric_limits<int64_t>::min());
        }
        double duration_ns() const { return end_ns > start_ns ? static_cast<double>(end_ns - start_ns) : 0.0; }
        // Average device clock over the program's kernels, GHz (cycles per ns); below nominal when the chip throttled.
        double clock_ghz() const {
            return zone_ns > 0 ? static_cast<double>(zone_cycles) / static_cast<double>(zone_ns) : 0.0;
        }
    };

    StreamingProgramCollector() = default;
    StreamingProgramCollector(const StreamingProgramCollector&) = delete;
    StreamingProgramCollector& operator=(const StreamingProgramCollector&) = delete;
    ~StreamingProgramCollector() { stop(); }

    void start() {
        stop();
        {
            std::lock_guard lock(mutex_);
            programs_.clear();
            executions_seen_.clear();
            dropped_bytes_ = 0;
            stall_count_ = 0;
            num_kernel_zones_ = 0;
        }
        callback_ = streaming_profiler::RegisterCallback(
            [this](const streaming_profiler::Batch<streaming_profiler::Zone>& batch) { on_batch(batch); },
            "ttnn-program-collector");
    }

    // Unregisters the callback; waits for an in-flight invocation to return.
    void stop() { callback_.reset(); }

    // Programs seen so far, ordered by start time.
    std::vector<ProgramRecord> records() const {
        std::vector<ProgramRecord> out;
        {
            std::lock_guard lock(mutex_);
            out.reserve(programs_.size());
            for (const auto& [key, record] : programs_) {
                out.push_back(record);
            }
        }
        std::ranges::sort(out, {}, &ProgramRecord::start_ns);
        for (ProgramRecord& record : out) {
            for (std::string_view source : streaming_profiler::KernelSources(record.runtime_id)) {
                record.kernel_sources.emplace_back(source);
            }
        }
        return out;
    }

    uint64_t dropped_bytes() const {
        std::lock_guard lock(mutex_);
        return dropped_bytes_;
    }
    uint64_t stall_count() const {
        std::lock_guard lock(mutex_);
        return stall_count_;
    }
    uint64_t num_kernel_zones() const {
        std::lock_guard lock(mutex_);
        return num_kernel_zones_;
    }

private:
    void on_batch(const streaming_profiler::Batch<streaming_profiler::Zone>& batch) {
        std::lock_guard lock(mutex_);
        dropped_bytes_ += batch.dropped_bytes();
        stall_count_ += batch.stall_count();
        for (const streaming_profiler::Zone& zone : batch.records<streaming_profiler::Zone>()) {
            if (zone.runtime_id() == 0 || !zone.descriptor().name.ends_with("-KERNEL")) {
                continue;
            }
            const streaming_profiler::Core core = zone.core();
            const auto processor = static_cast<uint32_t>(core.processor);
            const uint32_t core_key =
                (static_cast<uint32_t>(core.physical.y) << 16) | static_cast<uint32_t>(core.physical.x);
            // A "-KERNEL" zone never nests, so the k-th one on a RISC for a runtime id is that program's k-th launch.
            uint32_t& execution = executions_seen_[{core.chip_id, core_key, processor, zone.runtime_id()}];
            ProgramRecord& record = programs_[{core.chip_id, zone.runtime_id(), execution}];
            record.chip_id = core.chip_id;
            record.runtime_id = zone.runtime_id();
            record.execution = execution;
            execution++;
            num_kernel_zones_++;

            const int64_t start_ns = zone.start_time().time_since_epoch().count();
            const int64_t end_ns = zone.end_time().time_since_epoch().count();
            record.start_ns = std::min(record.start_ns, start_ns);
            record.end_ns = std::max(record.end_ns, end_ns);
            record.zone_cycles += zone.end_device_cycles() - zone.start_device_cycles();
            record.zone_ns += end_ns - start_ns;
            record.start_cycles = std::min(record.start_cycles, zone.start_device_cycles());
            record.end_cycles = std::max(record.end_cycles, zone.end_device_cycles());
            if (processor < kNumProcessors) {
                record.processor_start_ns[processor] = std::min(record.processor_start_ns[processor], start_ns);
                record.processor_end_ns[processor] = std::max(record.processor_end_ns[processor], end_ns);
            }
            record.cores.insert(core_key);
        }
    }

    mutable std::mutex mutex_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, ProgramRecord> programs_;  // (chip, runtime id, execution)
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> executions_seen_;
    uint64_t dropped_bytes_ = 0;
    uint64_t stall_count_ = 0;
    uint64_t num_kernel_zones_ = 0;
    streaming_profiler::Callback callback_;
};

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
        nb::arg("dispatch_core_config") = nb::none(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        nb::rv_policy::reference,  // cleanup has to happen in c++ land
        R"doc(
            Open a device with the given device_id. If the device is already open, return the existing device.

            Keyword Args:
                device_id (int): The device ID to open.
                l1_small_size (int, optional): The size of the L1 small buffer. Defaults to `ttnn.device.DEFAULT_L1_SMALL_SIZE`. The default is 0: ops that allocate from the L1_SMALL region (conv2d and pooling with kernel > 1, CCL semaphores) fail with an `L1_SMALL` out-of-memory error, or fall back to L1, unless this is large enough for their allocations. The conv and pool unit tests use 16384 or 32768; model demos under `models/` use 8192 to 131072.
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

    mod.def(
        "close_device",
        [](ttnn::MeshDevice& device) { ttnn::close_device(device); },
        nb::arg("device"),
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "deallocate_buffers",
        [](ttnn::MeshDevice* device) { ttnn::deallocate_buffers(device); },
        nb::arg("device"),
        R"doc(
        Deallocate all buffers associated with Device handle. Also clears the program cache, since cached
        programs keep their kernel binaries in device buffers.
    )doc");
}

}  // namespace

namespace ttnn::device {

void py_device_module_types(nb::module_& m_device) {
    nb::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE)
        .value("QUASAR", tt::ARCH::QUASAR);

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
            nb::arg("axis"))
        .def_prop_ro("type", [](const tt::tt_metal::DispatchCoreConfig& self) { return self.get_dispatch_core_type(); })
        .def_prop_ro(
            "axis", [](const tt::tt_metal::DispatchCoreConfig& self) { return self.get_dispatch_core_axis(); });

    nb::class_<SubDevice>(m_device, "SubDevice", "Class describing a sub-device of a Tenstorrent accelerator device.");

    nb::class_<SubDeviceId>(m_device, "SubDeviceId", "ID of a sub-device.");

    nb::class_<SubDeviceManagerId>(m_device, "SubDeviceManagerId", "ID of a sub-device manager.");

    nb::class_<tt::tt_metal::experimental::ProgramRealtimeRecord>(
        m_device, "ProgramRealtimeRecord", "Record containing real-time profiler data from a device.")
        .def_ro("runtime_id", &tt::tt_metal::experimental::ProgramRealtimeRecord::runtime_id, "Runtime ID")
        .def_ro(
            "start_timestamp",
            &tt::tt_metal::experimental::ProgramRealtimeRecord::start_timestamp,
            "Device start timestamp (raw ticks)")
        .def_ro(
            "end_timestamp",
            &tt::tt_metal::experimental::ProgramRealtimeRecord::end_timestamp,
            "Device end timestamp (raw ticks)")
        .def_ro(
            "frequency",
            &tt::tt_metal::experimental::ProgramRealtimeRecord::frequency,
            "Device clock frequency (cycles per ns)")
        .def_ro("chip_id", &tt::tt_metal::experimental::ProgramRealtimeRecord::chip_id, "Device chip ID")
        .def_prop_ro(
            "kernel_sources",
            [](const tt::tt_metal::experimental::ProgramRealtimeRecord& record) {
                return std::vector<std::string>(record.kernel_sources.begin(), record.kernel_sources.end());
            },
            "Kernel source paths associated with this runtime ID; valid until the callback returns");

    nb::class_<PythonProgramRealtimeRecordBatch>(
        m_device, "ProgramRealtimeRecordBatch", "Batch of real-time profiler records delivered to a callback.")
        .def_ro(
            "records",
            &PythonProgramRealtimeRecordBatch::records,
            "ProgramRealtimeRecord entries in this batch; non-empty, oldest first")
        .def_ro(
            "dropped",
            &PythonProgramRealtimeRecordBatch::dropped,
            "Records lost since this callback last ran; nonzero if the callback could not keep up with incoming "
            "profiler data");

    using StreamingProgramRecord = StreamingProgramCollector::ProgramRecord;
    nb::class_<StreamingProgramRecord>(
        m_device, "StreamingProgramRecord", "One program launch on one chip, aggregated from streaming profiler zones.")
        .def_ro("chip_id", &StreamingProgramRecord::chip_id)
        .def_ro("runtime_id", &StreamingProgramRecord::runtime_id)
        .def_ro("execution", &StreamingProgramRecord::execution, "k-th launch of this runtime id on this chip")
        .def_ro("start_ns", &StreamingProgramRecord::start_ns, "First kernel start, host steady_clock ns")
        .def_ro("end_ns", &StreamingProgramRecord::end_ns, "Last kernel end, host steady_clock ns")
        .def_ro("start_cycles", &StreamingProgramRecord::start_cycles)
        .def_ro("end_cycles", &StreamingProgramRecord::end_cycles)
        .def_ro("kernel_sources", &StreamingProgramRecord::kernel_sources, "Kernel source paths of the program")
        .def_prop_ro(
            "duration_ns",
            &StreamingProgramRecord::duration_ns,
            "First kernel start to last kernel end, elapsed host time (includes clock throttling)")
        .def_prop_ro(
            "clock_ghz", &StreamingProgramRecord::clock_ghz, "Average device clock over the program's kernels, GHz")
        .def_prop_ro("num_cores", [](const StreamingProgramRecord& record) { return record.cores.size(); })
        .def_prop_ro(
            "processor_duration_ns",
            [](const StreamingProgramRecord& record) {
                static constexpr std::array<const char*, StreamingProgramCollector::kNumProcessors> kNames = {
                    "BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2", "ERISC0", "ERISC1"};
                std::map<std::string, double> out;
                for (std::size_t i = 0; i < kNames.size(); i++) {
                    if (record.processor_end_ns[i] > record.processor_start_ns[i]) {
                        out[kNames[i]] = static_cast<double>(record.processor_end_ns[i] - record.processor_start_ns[i]);
                    }
                }
                return out;
            },
            "Per-RISC span (first start to last end over cores), ns; RISCs with no kernel are omitted");

    nb::class_<StreamingProgramCollector>(
        m_device,
        "StreamingProgramCollector",
        "Collects per-program device durations from the streaming profiler (TT_METAL_STREAMING_PROFILER=1).")
        .def(nb::init<>())
        .def("start", &StreamingProgramCollector::start, nb::call_guard<nb::gil_scoped_release>())
        .def("stop", &StreamingProgramCollector::stop, nb::call_guard<nb::gil_scoped_release>())
        .def("records", &StreamingProgramCollector::records, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("dropped_bytes", &StreamingProgramCollector::dropped_bytes)
        .def_prop_ro("stall_count", &StreamingProgramCollector::stall_count)
        .def_prop_ro("num_kernel_zones", &StreamingProgramCollector::num_kernel_zones);

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
        "create_dispatch_core_config",
        [](std::optional<tt::tt_metal::DispatchCoreType> type,
           std::optional<tt::tt_metal::DispatchCoreAxis> axis,
           std::optional<tt::tt_fabric::FabricTensixConfig> fabric_tensix_config) {
            return ttnn::device::create_cluster_aware_dispatch_config(
                type, axis, fabric_tensix_config.value_or(tt::tt_fabric::FabricTensixConfig::DISABLED));
        },
        nb::kw_only(),
        nb::arg("type") = nb::none(),
        nb::arg("axis") = nb::none(),
        nb::arg("fabric_tensix_config") = nb::none(),
        "Build a cluster/arch-aware DispatchCoreConfig (TTNN default that prefers maximum available worker cores).");

    m_device.def(
        "CreateDevice",
        [](int device_id,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const std::optional<tt::tt_metal::DispatchCoreConfig>& dispatch_core_config,
           size_t worker_l1_size) {
            return MeshDevice::create_unit_mesh(
                device_id,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config.value_or(ttnn::device::create_cluster_aware_dispatch_config()),
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
        nb::arg("DispatchCoreConfig") = nb::none(),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    m_device.def(
        "CreateDevices",
        [](const std::vector<int>& device_ids,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const std::optional<tt::tt_metal::DispatchCoreConfig>& dispatch_core_config,
           size_t worker_l1_size) {
            return MeshDevice::create_unit_meshes(
                device_ids,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config.value_or(ttnn::device::create_cluster_aware_dispatch_config()),
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
        nb::arg("DispatchCoreConfig") = nb::none(),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    m_device.def(
        "CloseDevice",
        [](MeshDevice* device) { device->close(); },
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        [](const std::array<uint32_t, 4>& unpadded_shape) -> nb::list {
            auto result = ttnn::operations::data_movement::pad_to_tile_shape(ttnn::Shape(unpadded_shape));
            nb::list py_list;
            for (auto val : result) {
                py_list.append(val);
            }
            return py_list;
        },
        nb::arg("unpadded_shape"),
        R"doc(
        Pads the given shape to tile shape (rounds last two dims up to multiples of 32).

        .. deprecated::
            This function is deprecated and will be removed in a future release.
            Use ``ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)`` which handles
            tile-alignment automatically.

            If you only need the padded shape without converting layout, align
            dimensions manually::

                import math
                TILE = 32
                shape = list(original_shape)
                shape[-1] = math.ceil(shape[-1] / TILE) * TILE
                if len(shape) >= 2:
                    shape[-2] = math.ceil(shape[-2] / TILE) * TILE

        Args:
            unpadded_shape (List of [int]): The original shape of the tensor to pad.

        Returns:
            List of [int]: The padded shape.

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
    m_device.def(
        "GetOperationTimeoutSeconds",
        [](const MeshDevice& mesh_device) {
            return tt::tt_metal::experimental::GetOperationTimeout(mesh_device).count();
        },
        nb::arg("mesh_device").noconvert(),
        R"doc(
        Get the operation timeout in seconds (0 = off), initialized from TT_METAL_OPERATION_TIMEOUT_SECONDS.

        A command queue wait that sees no dispatch progress for this long runs
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE (if set) and raises.

        Args:
            mesh_device (ttnn.MeshDevice): Selects the environment; all mesh devices in it share one timeout.

        Example:
            >>> import ttnn
            >>> timeout_s = ttnn.device.GetOperationTimeoutSeconds(mesh_device)
    )doc");
    m_device.def(
        "SetOperationTimeoutSeconds",
        [](MeshDevice& mesh_device, float seconds) {
            tt::tt_metal::experimental::SetOperationTimeout(mesh_device, std::chrono::duration<float>(seconds));
        },
        nb::arg("mesh_device").noconvert(),
        nb::arg("seconds"),
        R"doc(
        Set the operation timeout in seconds (0 = off), overriding TT_METAL_OPERATION_TIMEOUT_SECONDS.

        Each wait reads the value when it starts, so a wait already in progress keeps its old value.

        Args:
            mesh_device (ttnn.MeshDevice): Selects the environment; all mesh devices in it share one timeout.
            seconds (float): The new timeout; 0 turns it off.

        Example:
            >>> import ttnn
            >>> ttnn.device.SetOperationTimeoutSeconds(mesh_device, 300)
    )doc");
    m_device.def(
        "get_dispatch_core_axis",
        [](MeshDevice* device) {
            return tt::tt_metal::experimental::DispatchContext::get().get_dispatch_core_axis(device);
        },
        nb::arg("device").noconvert(),
        R"doc(
        Return the axis (ROW or COL) of the device's live dispatch-core configuration.

        COL means dispatch sits on the last column of the worker grid (the Blackhole default); ROW means
        the last row (Blackhole with fabric tensix enabled, or Wormhole). tt-blaze's two-phase upload uses
        it to find that edge: cargo shards off the edge are written under Fast Dispatch, and shards on it
        are written under Slow Dispatch after the session, because dispatch firmware occupies those cores
        during it.

        Args:
            device (ttnn.Device): The mesh device to query.

        Returns:
            ttnn.device.DispatchCoreAxis: ROW or COL.
    )doc");
    m_device.def(
        "initialize_fast_dispatch",
        [](MeshDevice* device, bool allow_destructive) {
            tt::tt_metal::experimental::FastDispatchSetupOptions options;
            options.allow_destructive = allow_destructive;
            tt::tt_metal::experimental::DispatchContext::get().initialize_fast_dispatch(device, options);
        },
        nb::arg("device").noconvert(),
        nb::arg("allow_destructive") = false,
        R"doc(
        Dynamically enable Fast Dispatch on a MeshDevice that was opened in Slow Dispatch mode.

        Refuses, before any firmware is written, if an L1 or L1_SMALL allocation is resident on a core
        that Fast Dispatch will claim; dispatch cores may not hold L1 allocations while Fast Dispatch is
        active.

        Limitations:
            - Any interleaved L1 buffer resident when the session opens is refused, whatever its size:
              its pages are spread over every L1 bank.
            - A non-default sub-device manager still loaded on any view over an active chip is refused,
              even with allow_destructive: the check sees only the default manager's allocations.
              Sub-device managers aren't supported with manual Fast Dispatch; loading one during a
              session raises.
            - Every active device must be MMIO-attached; otherwise this raises before Fast Dispatch is
              enabled.

        Args:
            device (ttnn.Device): The mesh device to enable Fast Dispatch on.
            allow_destructive (bool): Warn and proceed instead of refusing. The resident L1 may be corrupted.
    )doc");
    m_device.def(
        "terminate_fast_dispatch",
        [](MeshDevice* device) { tt::tt_metal::experimental::DispatchContext::get().terminate_fast_dispatch(device); },
        nb::arg("device").noconvert(),
        R"doc(
        Disable Fast Dispatch on a MeshDevice, returning it to Slow Dispatch mode.

        Args:
            device (ttnn.Device): The mesh device to disable Fast Dispatch on.
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

    constexpr std::string_view get_allocator_base_address_doc = R"doc(
        Return the base address (in bytes) of the device's allocator region for the given buffer type.

        For ``ttnn.BufferType.L1`` this is the worker-L1 unreserved base, i.e. the lowest address
        that the allocator is allowed to hand out for tensor / buffer storage on Tensix worker
        cores. Combined with the worker-L1 size (queryable via :py:func:`ttnn.get_memory_view`), it
        defines the address range available for compute-time L1 allocations.

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to query                  | ttnn.Device           |             | Yes      |
        | buffer_type      | Allocator region (L1 / DRAM)     | ttnn.BufferType       |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc";
    auto buffer_type_to_hal_mem_type = [](const BufferType& buffer_type) {
        switch (buffer_type) {
            case BufferType::DRAM: return HalMemType::DRAM;
            case BufferType::L1:
            case BufferType::L1_SMALL:
            case BufferType::TRACE: return HalMemType::L1;
            default:
                throw std::invalid_argument(
                    "GetAllocatorBaseAddress: unsupported buffer_type=" +
                    std::to_string(static_cast<int>(buffer_type)));
        }
    };
    m_device.def(
        "GetAllocatorBaseAddress",
        [buffer_type_to_hal_mem_type](IDevice* device, const BufferType& buffer_type) {
            return device->allocator()->get_base_allocator_addr(buffer_type_to_hal_mem_type(buffer_type));
        },
        nb::arg("device").noconvert(),
        nb::arg("buffer_type").noconvert(),
        get_allocator_base_address_doc.data());
    m_device.def(
        "GetAllocatorBaseAddress",
        [buffer_type_to_hal_mem_type](MeshDevice* device, const BufferType& buffer_type) {
            return device->allocator()->get_base_allocator_addr(buffer_type_to_hal_mem_type(buffer_type));
        },
        nb::arg("device").noconvert(),
        nb::arg("buffer_type").noconvert(),
        get_allocator_base_address_doc.data());

    constexpr std::string_view get_l1_occupied_ranges_doc = R"doc(
        Experimental. Return the occupied L1 ``(start, end)`` byte ranges (end exclusive) on one device of the mesh:
        lockstep allocations, the core's per-core allocations and persistent L1, sorted and coalesced.
        With ``core``, returns a list for that core; without it, a dict mapping every L1-bank core to its list.
        Requires the device to be opened with ``TT_METAL_ALLOCATOR_MODE_HYBRID=1``.
    )doc";
    m_device.def(
        "ExperimentalGetL1OccupiedRanges",
        [](const MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord, const CoreCoord& core) {
            return tt::tt_metal::experimental::per_core_allocation::get_l1_occupied_ranges(
                mesh_device, device_coord, core);
        },
        nb::arg("mesh_device").noconvert(),
        nb::arg("device_coord"),
        nb::arg("core"),
        get_l1_occupied_ranges_doc.data());
    m_device.def(
        "ExperimentalGetL1OccupiedRanges",
        [](const MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord) {
            return tt::tt_metal::experimental::per_core_allocation::get_l1_occupied_ranges(mesh_device, device_coord);
        },
        nb::arg("mesh_device").noconvert(),
        nb::arg("device_coord"),
        get_l1_occupied_ranges_doc.data());

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
            // Guard before touching mesh_command_queue(): the queues are torn down on close, so looking one up on
            // an uninitialized device would assert. Synchronize() is a no-op there anyway.
            if (!device->is_initialized()) {
                return;
            }
            if (cq_id.has_value()) {
                tt::tt_metal::distributed::Synchronize(
                    *device, device->mesh_command_queue(cq_id->get()), sub_device_ids);
            } else {
                tt::tt_metal::distributed::Synchronize(*device, std::nullopt, sub_device_ids);
            }
        },
        synchronize_device_doc.data(),
        nb::arg("device"),
        nb::arg("cq_id") = nb::none(),
        nb::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        // Release GIL: sync can block a long time; other threads need it to run Python (e.g. real-time profiler
        // callbacks).
        nb::call_guard<nb::gil_scoped_release>());
    m_device.def(
        "is_trace_capture_active",
        [](MeshDevice* device) {
            if (!device->is_initialized()) {
                return false;
            }
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
                if (device->mesh_command_queue(cq_id).trace_id().has_value()) {
                    return true;
                }
            }
            return false;
        },
        nb::arg("device"),
        R"doc(
                Returns True if a metal trace is currently being captured on any command queue of the device.

                Operations that synchronize with host (event recording, reads, writes) are not supported while a
                trace is being captured, so callers that would otherwise synchronize must skip doing so.

                Args:
                    device (ttnn.MeshDevice): The device to query.

                Returns:
                    `bool`: whether a trace capture is in progress.
            )doc");
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

    m_device.def(
        "get_dram_alignment",
        &tt::tt_metal::hal::get_dram_alignment,
        "Return the DRAM alignment requirement in bytes for the current architecture.");

    m_device.def(
        "get_l1_alignment",
        &tt::tt_metal::hal::get_l1_alignment,
        "Return the L1 alignment requirement in bytes for the current architecture.");

    m_device.def(
        "get_optimal_dram_bank_to_logical_worker_assignment",
        [](MeshDevice* device, tt::tt_metal::NOC noc) {
            return device->get_optimal_dram_bank_to_logical_worker_assignment(noc);
        },
        nb::arg("device"),
        nb::arg("noc"),
        R"doc(
            [EXPERIMENTAL] Returns the optimal DRAM bank to logical worker core assignment.

            This function returns an ordered list of logical worker core coordinates that are optimally
            placed to interface with DRAM banks. Placing DRAM reader or writer kernels on these worker
            cores will minimize NOC congestion and the number of NOC hops required to complete a DRAM
            read or write.

            Args:
                device (ttnn.Device): The TT device to query.
                noc (ttnn.NOC): The Network-on-Chip to use (NOC_0 or NOC_1, or RISCV_0_default/RISCV_1_default).

            Returns:
                List[ttnn.CoreCoord]: A list of logical worker core coordinates, where index i corresponds
                to the optimal worker core for DRAM bank i.

            Example:
                >>> import ttnn
                >>> device = ttnn.open_device(device_id=0)
                >>> cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(
                ...     device, ttnn.NOC.RISCV_0_default
                ... )
                >>> print(f"DRAM bank 0 maps to core: {cores[0]}")
        )doc");

    m_device.def(
        "enable_asynchronous_slow_dispatch",
        [](tt::tt_metal::distributed::MeshDevice* device) {
            tt::tt_metal::experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(device);
        },
        nb::arg("device"),
        R"doc(
        Experimental: If Slow Dispatch is enabled, this function enables running multiple non-overlapping programs concurrently on the same device.
        )doc");
    m_device.def(
        "disable_asynchronous_slow_dispatch",
        [](tt::tt_metal::distributed::MeshDevice* device) {
            tt::tt_metal::experimental::DispatchContext::get().disable_asynchronous_slow_dispatch(device);
        },
        nb::arg("device"),
        R"doc(
        Experimental: If Slow Dispatch is enabled, this function disables the ability to run multiple non-overlapping programs concurrently on the same device.
        )doc");
    m_device.def(
        "is_asynchronous_slow_dispatch_enabled",
        [](tt::tt_metal::distributed::MeshDevice* device) {
            return tt::tt_metal::experimental::DispatchContext::get().is_asynchronous_slow_dispatch_enabled(device);
        },
        nb::arg("device"),
        R"doc(
        Experimental: Returns whether asynchronous slow dispatch is currently enabled on the given device.
        )doc");

    m_device.def(
        "RegisterProgramRealtimeProfilerCallback",
        [](const nb::callable& callback) -> uint64_t {
            PyObject* raw_cb = callback.ptr();
            Py_INCREF(raw_cb);

            uint64_t handle = 0;
            {
                nb::gil_scoped_release release;
                handle = tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback(
                    [raw_cb](const tt::tt_metal::experimental::ProgramRealtimeRecordBatch& batch) {
                        PythonProgramRealtimeRecordBatch py_batch{
                            .records = std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>(
                                batch.records.begin(), batch.records.end()),
                            .dropped = batch.dropped,
                        };
                        nb::gil_scoped_acquire gil;
                        (nb::handle(raw_cb))(nb::cast(std::move(py_batch), nb::rv_policy::move));
                    });
            }

            python_realtime_callback_refs[handle] = raw_cb;
            return handle;
        },
        nb::arg("callback"),
        R"doc(
            Register a callback to be invoked when real-time profiler data arrives from a device.
            The callback receives a ProgramRealtimeRecordBatch and is called from its own thread.
            Callbacks that are too slow to keep up with incoming profiler data may miss records;
            this is reported by ProgramRealtimeRecordBatch.dropped.

            Multiple callbacks can be registered.

            Args:
                callback: A callable that accepts a single ProgramRealtimeRecordBatch argument.

            Returns:
                int: A handle that can be passed to UnregisterProgramRealtimeProfilerCallback.

            Example:
                >>> def my_callback(batch):
                ...     for record in batch.records:
                ...         print(f"runtime_id={record.runtime_id} on chip {record.chip_id}")
                >>> handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(my_callback)
        )doc");

    m_device.def(
        "UnregisterProgramRealtimeProfilerCallback",
        [](uint64_t handle) {
            {
                nb::gil_scoped_release release;
                tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback(handle);
            }
            auto it = python_realtime_callback_refs.find(handle);
            if (it != python_realtime_callback_refs.end()) {
                Py_DECREF(it->second);
                python_realtime_callback_refs.erase(it);
            }
        },
        nb::arg("handle"),
        R"doc(
            Unregister a previously registered real-time profiler callback.
            This call waits for any in-flight invocations of the callback to finish.

            Args:
                handle (int): The handle returned by RegisterProgramRealtimeProfilerCallback.
        )doc");

    m_device.def(
        "IsProgramRealtimeProfilerActive",
        []() { return tt::tt_metal::experimental::IsProgramRealtimeProfilerActive(); },
        R"doc(
            Returns True if the real-time profiler is currently running on at least one chip.

            Returns False when the profiler was silently disabled for the current run
            (for example, ETH dispatch, remote-only mesh, or a platform that can't reserve
            a tensix dispatch core). Tests that want to collect RT records should call this
            after opening the device and skip their verification when it returns False —
            registered callbacks will simply never fire in that case.

            Example:
                >>> mesh_device = ttnn.open_mesh_device(...)
                >>> if not ttnn.device.IsProgramRealtimeProfilerActive():
                ...     pytest.skip("Real-time profiler not active on this configuration")
        )doc");

    m_device.def(
        "SetStreamingProfilerEnabled",
        [](bool enabled, uint32_t host_fifo_mb) { streaming_profiler::SetEnabled(enabled, host_fifo_mb); },
        nb::arg("enabled"),
        nb::arg("host_fifo_mb") = 16,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Turn the streaming profiler on or off for the devices opened next, in place of
            TT_METAL_STREAMING_PROFILER. Call it with every device closed and fabric DISABLED.
        )doc");

    m_device.def(
        "IsStreamingProfilerActive",
        []() { return streaming_profiler::IsActive(); },
        R"doc(
            Returns True if the streaming profiler (TT_METAL_STREAMING_PROFILER=1) is running on at least one device.
            Call after opening the device.
        )doc");
}

void py_device_module(nb::module_& mod) {
    ttnn_device(mod);
    device_module(mod);
}

}  // namespace ttnn::device

// NOLINTEND(bugprone-unused-raii)
