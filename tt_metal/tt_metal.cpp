// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>

#include "dev_msgs.h"
#include "llrt/hal.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/dispatch/command_queue.hpp"
#include "tools/profiler/profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

#include "tt_metal/graph/graph_tracking.hpp"

namespace tt {

namespace tt_metal {

namespace {

CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet> &specified_core_spec) {
    ZoneScoped;
    return std::visit(
        [](auto&& core_spec) -> CoreRangeSet
        {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRangeSet({CoreRange(core_spec, core_spec)});
            }
            else if constexpr (std::is_same_v<T, CoreRange>) {
                return CoreRangeSet({core_spec});
            }
            else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                return core_spec;
            }
        },
        specified_core_spec
    );
}

struct DataMovementConfigStatus {
    bool riscv0_in_use;
    bool riscv1_in_use;
    bool noc0_in_use;
    bool noc1_in_use;
};

DataMovementConfigStatus CheckDataMovementConfig(Program &program, const CoreRangeSet &core_ranges) {
    DataMovementConfigStatus data_movement_config_status{
        .riscv0_in_use = false, .riscv1_in_use = false, .noc0_in_use = false, .noc1_in_use = false};

    auto set_global_and_local_noc_usage = [&](KernelHandle kernel_id, bool &local_noc0_usage, bool &local_noc1_usage) {
        const auto kernel = detail::GetKernel(program, kernel_id);
        auto kernel_config = std::get<DataMovementConfig>(kernel->config());
        auto noc_value = magic_enum::enum_integer(kernel_config.noc);
        local_noc0_usage = noc_value == 0;
        local_noc1_usage = noc_value == 1;
        data_movement_config_status.noc0_in_use = local_noc0_usage;
        data_movement_config_status.noc1_in_use = local_noc1_usage;
    };

    for (const auto &core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup *kernel_group = program.kernels_on_core(
                    CoreCoord(x, y), hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX));
                if (kernel_group != nullptr) {
                    bool local_noc0_in_use = false;
                    bool local_noc1_in_use = false;
                    if (kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].has_value()) {
                        data_movement_config_status.riscv0_in_use = true;
                        set_global_and_local_noc_usage(
                            kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value(),
                            local_noc0_in_use,
                            local_noc1_in_use);
                    }
                    if (kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].has_value()) {
                        data_movement_config_status.riscv1_in_use = true;
                        set_global_and_local_noc_usage(
                            kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value(),
                            local_noc0_in_use,
                            local_noc1_in_use);
                    }
                    if (kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].has_value() and
                        kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].has_value()) {
                        TT_FATAL(
                            local_noc0_in_use and local_noc1_in_use,
                            "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, "
                            "doing so results in hangs!",
                            CoreCoord(x, y).str());
                    }
                }
            }
        }
    }

    return data_movement_config_status;
}

void ConfigureKernelGroup(
    const Program &program, const KernelGroup *kernel_group, Device *device, const CoreCoord &logical_core) {

    for (auto& optional_id : kernel_group->kernel_ids) {
        if (optional_id) {
            detail::GetKernel(program, optional_id.value())->configure(device, logical_core);
        }
    }
}

std::optional<uint32_t> get_semaphore_id(const Program &program, const CoreRange &core_range) {
    std::optional<uint32_t> semaphore_id = std::nullopt;
    std::vector<uint32_t> semaphore_histogram(NUM_SEMAPHORES, 0);
    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            CoreCoord logical_core(x, y);
            auto semaphores = program.semaphores_on_core(logical_core);
            if (semaphores.size() == NUM_SEMAPHORES) {
                TT_THROW(
                    "Cannot add semaphore on core {}. Max number of semaphores ({}) reached!", logical_core.str(), NUM_SEMAPHORES);
            }

            for (const auto &semaphore : semaphores) {
                semaphore_histogram[semaphore.get().id()]++;
            }
        }
    }

    std::optional<uint32_t> uninitialized_sem_id = std::nullopt;
    for (int sem_id = 0; sem_id < semaphore_histogram.size(); sem_id++) {
        if (semaphore_histogram.at(sem_id) == 0) {
            uninitialized_sem_id = sem_id;
            break;
        }
    }

    if (uninitialized_sem_id.has_value()) {
        semaphore_id =  uninitialized_sem_id.value();
    } else {
        TT_THROW("Unable to initialize semaphores on core range {}", core_range.str());
    }

    return semaphore_id;
}

inline void SetRuntimeArgsImpl(
    const Program &program, KernelHandle kernel_id, const CoreCoord &c, stl::Span<const uint32_t> runtime_args) {
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgsImpl(
    const Program &program,
    KernelHandle kernel_id,
    const CoreRange &core_range,
    stl::Span<const uint32_t> runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = detail::GetKernel(program, kernel_id);
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
            }
        }
    }
}

inline void SetRuntimeArgsImpl(
    const Program &program,
    KernelHandle kernel_id,
    const CoreRangeSet &core_range_set,
    stl::Span<const uint32_t> runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = detail::GetKernel(program, kernel_id);
        for (const auto &core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

inline void SetRuntimeArgsImpl(
    CommandQueue &cq,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode
    std::visit(
        [&](auto &&core_spec) {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                EnqueueSetRuntimeArgs(cq, kernel, core_spec, runtime_args, blocking);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                for (auto x = core_spec.start_coord.x; x <= core_spec.end_coord.x; x++) {
                    for (auto y = core_spec.start_coord.y; y <= core_spec.end_coord.y; y++) {
                        EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                    }
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto &core_range : core_spec.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                        }
                    }
                }
            }
        },
        core_spec);
}

inline void SetRuntimeArgsImpl(
    CommandQueue &cq,
    const std::shared_ptr<Kernel> kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode (support vector of runtime args)
    for (size_t i = 0; i < core_spec.size(); i++) {
        EnqueueSetRuntimeArgs(cq, kernel, core_spec[i], runtime_args[i], blocking);
    }
}

}  // namespace

// #define DEBUG_PRINT_SHARD

namespace detail {

bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
{
    bool pass = true;
    TT_FATAL(address >= device->get_base_allocator_addr(HalMemType::DRAM), "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!", device->get_base_allocator_addr(HalMemType::DRAM));
    tt::Cluster::instance().write_dram_vec(host_buffer, tt_target_dram{device->id(), dram_channel, 0}, address);
    return pass;
}

bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
{
    bool pass = true;
    tt::Cluster::instance().dram_barrier(device->id());
    tt::Cluster::instance().read_dram_vec(host_buffer, size, tt_target_dram{device->id(), dram_channel, 0}, address);
    return pass;
}

bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer, CoreType core_type)
{
    ZoneScoped;
    auto worker_core = device->physical_core_from_logical_core(logical_core, core_type);
    llrt::write_hex_vec_to_core(device->id(), worker_core, host_buffer, address);
    return true;
}

bool WriteRegToDevice(Device *device, const CoreCoord &logical_core, uint32_t address, const uint32_t &regval)
{
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    tt::Cluster::instance().write_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
{
    tt::Cluster::instance().l1_barrier(device->id());
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    host_buffer = llrt::read_hex_vec_from_core(device->id(), worker_core, address, size);
    return true;
}

bool ReadRegFromDevice(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t &regval)
{
    tt::Cluster::instance().l1_barrier(device->id());
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    tt::Cluster::instance().read_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

std::map<chip_id_t, Device *> CreateDevices(
    const std::vector<chip_id_t>& device_ids,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    DispatchCoreType dispatch_core_type,
    const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    bool is_galaxy = tt::Cluster::instance().is_galaxy_cluster();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type);
    const auto devices = tt::DevicePool::instance().get_all_active_devices();
    std::map<chip_id_t, Device *> ret_devices;
    //Only include the mmio device in the active devices set returned to the caller if we are not running
    //on a Galaxy cluster.
    //On Galaxy, gateway (mmio devices) cannot run compute workloads.

    for (Device * dev: devices) {
        if (is_galaxy and dev->is_mmio_capable()) {
            continue;
        }
        ret_devices.insert({dev->id(), dev});
    }

    return ret_devices;
}

void CloseDevices(std::map<chip_id_t, Device *> devices) {
    // Global Sync across all devices in the pool.
    // We need to ensure that commands sent to each device have been completed
    // before closing any device + modifying routing info.
    // If this is not done, non-blocking CCLs followed by a close will hang, since
    // the main thread will modify device state while the CCL is running on device.
    for (const auto &[device_id, dev] : devices) {
        dev->synchronize(); // Synchronize worker queue
        Synchronize(dev); // Synchronize device
    }
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    std::map<chip_id_t, v1::DeviceHandle> mmio_devices = {};
    bool is_galaxy = tt::Cluster::instance().is_galaxy_cluster();

    if (is_galaxy) {
        //On Galaxy, gateway wormhole devices (mmio devices) are not included in the set of devices
        //created by CreateDevices(). So when closing devices, we need to find the corresponding
        //gateway chips for all the tunneled devcies.
        for (const auto &[device_id, dev] : devices) {
            const auto &mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
            if (mmio_devices.find(mmio_device_id) == mmio_devices.end()) {
                auto dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
                mmio_devices.insert({mmio_device_id, dev_handle});
            }
        }
    } else {
        for (const auto &[device_id, dev] : devices) {
            if(dev->is_mmio_capable()) {
                mmio_devices.insert({device_id, tt::DevicePool::instance().get_handle(dev)});
            }
        }
        for (const auto &[device_id, dev] : mmio_devices) {
            devices.erase(device_id);
        }
    }

    for (const auto &[device_id, dev] : mmio_devices) {
        //For each mmio device, first close all the remote tunneled devices.
        //Close the farthest tunneled device first.
        auto tunnels_from_mmio = dev->tunnels_from_mmio_;
        //iterate over all tunnels origination from this mmio device
        for (auto t : tunnels_from_mmio) {
            //iterate over all tunneled devices (tunnel stops) in this tunnel and close them.
            for (uint32_t ts = t.size() - 1; ts > 0; ts--) {
                if (devices.find(t[ts]) != devices.end()) {
                    devices[t[ts]]->close();
                    // When a device is closed, its worker thread is joined. Stop tracking this
                    // worker thread.
                    tt::DevicePool::instance().unregister_worker_thread_for_device(tt::DevicePool::instance().get_handle(devices[t[ts]]));
                }
            }
        }
        //finally close the mmio device
        dev->close();
        tt::DevicePool::instance().unregister_worker_thread_for_device(dev);
    }
}

bool InWorkerThread() {
    // These are values are cached per thread. in_worker_thread is a 1:1 function of the thread_id.
    // Therefore it does not need to be recomputed or looked up using the worker_thread_ids each time.
    // This is a performance optimization, since looking up the thread id inside worker_thread_ids for
    // each function call significantly degrades runtime perf.
    thread_local static bool in_worker_thread = false;
    thread_local static bool is_thread_status_checked = false;
    if (not is_thread_status_checked) {
        auto worker_thread_ids = tt::DevicePool::instance().get_worker_thread_ids();
        in_worker_thread = worker_thread_ids.find(std::this_thread::get_id()) != worker_thread_ids.end();
        is_thread_status_checked = true;
    }
    return in_worker_thread;
}

void print_page(
    uint32_t dev_page_id,
    CoreCoord core,
    uint32_t host_page_id,
    CoreCoord noc_coordinates,
    uint32_t l1_address,
    uint32_t bank_id,
    std::vector<uint32_t> page) {
    std::cout << "dev_page_index " << dev_page_id << " on core " << core.str() << std::endl;
    std::cout << "host_page_index " << host_page_id << std::endl;
    std::cout << "noc coordinates " << noc_coordinates.str() << std::endl;
    std::cout << "l1_address " << l1_address << std::endl;
    std::cout << "bank id " << bank_id << std::endl;

    std::cout << "0x";
    for (auto entry : page) {
        std::cout << std::hex << entry << std::dec;
    }
    std::cout << std::dec << std::endl;
}

void WriteToDeviceSharded(Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_ASSERT(page_size % bytes_per_page_entry == 0);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    auto total_pages = buffer.num_pages();
    for (int host_page_id = 0; host_page_id < total_pages; host_page_id++) {
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page_id];
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto absolute_address = buffer.sharded_page_address(bank_id, dev_page_id);
        auto data_index = host_page_id * num_entries_per_page;
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);

        auto noc_coordinates = buffer.noc_coordinates(bank_id);
        llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
    }
}

void WriteToDeviceInterleavedContiguous(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_FATAL(
        buffer.size() % page_size == 0,
        "Invalid buffer size: {}. Buffer size must be a multiple of page size {}.",
        buffer.size(),
        page_size);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_FATAL(page_size % bytes_per_page_entry == 0,
        "Invalid page size: {}. Page size  must be a multiple of bytes per page entry {}.", page_size, bytes_per_page_entry);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());
    uint32_t bank_index = 0;
    int data_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
            } break;
            default: TT_THROW("Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}

void WriteToDevice(Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED ||
        buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK) {
        WriteToDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        WriteToDeviceSharded(buffer, host_buffer);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void WriteToBuffer(std::shared_ptr<Buffer> buffer, const std::vector<uint32_t> &host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

void WriteToBuffer(Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:  // fallthrough
        case BufferType::L1:    // fallthrough
        case BufferType::L1_SMALL: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Writing to host memory is unsupported!");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

void ReadFromDeviceInterleavedContiguous(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    host_buffer.clear();  // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_FATAL(
        buffer.size() % page_size == 0,
        "Invalid buffer size: {}. Buffer size must be a multiple of page size {}.",
        buffer.size(),
        page_size);
    uint32_t num_pages = buffer.size() / page_size;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());

    uint32_t bank_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::TRACE:
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
            } break;
            default: TT_THROW("Unsupported buffer type to read from device!");
        }

        // Copy page into host buffer
        for (uint32_t entry : page) {
            host_buffer.push_back(entry);
        }

        bank_index = (bank_index + 1) % num_banks;
    }
}

void read_pages_to_host_helper(
    Device *device,
    Buffer &dev_buffer,
    std::vector<uint32_t> &host_buffer,
    const uint32_t &page_size,
    const uint32_t &host_page_id,
    const uint32_t &dev_page_id,
    const uint32_t &bank_id) {
    auto absolute_address = dev_buffer.sharded_page_address(bank_id, dev_page_id);
    auto noc_coordinates = dev_buffer.noc_coordinates(bank_id);
    uint32_t num_entries_per_page = page_size / sizeof(uint32_t);
    uint32_t host_buffer_start = host_page_id * num_entries_per_page;
    tt::Cluster::instance().read_core(host_buffer.data() + host_buffer_start, page_size, tt_cxy_pair(device->id(), noc_coordinates), absolute_address);
}

void ReadFromDeviceSharded(Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    TensorMemoryLayout buffer_layout = buffer.buffer_layout();

    auto device = buffer.device();
#ifdef DEBUG_PRINT_SHARD
    std::cout << "Reading From Device Height Sharded " << std::endl;
#endif

    int output_page_index = 0;
    auto total_pages = buffer.num_dev_pages();
    uint32_t page_size = buffer.page_size();
    uint32_t bytes_per_page_entry = sizeof(uint32_t);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    host_buffer = std::vector<uint32_t>(total_pages * num_entries_per_page);

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    for (int dev_page_id = 0; dev_page_id < total_pages; dev_page_id++) {
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto host_page_id = buffer_page_mapping.dev_page_to_host_page_mapping_[dev_page_id];
        if (host_page_id.has_value()) {
            if (!shard_order) {
                read_pages_to_host_helper(
                    device, buffer, host_buffer, page_size, host_page_id.value(), dev_page_id, bank_id);
            } else {
                read_pages_to_host_helper(device, buffer, host_buffer, page_size, dev_page_id, dev_page_id, bank_id);
            }
        }
    }
}

void ReadFromDevice(Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    ZoneScoped;
    host_buffer.clear();  // overwrite the data
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED ||
        buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK) {
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        ReadFromDeviceSharded(buffer, host_buffer, shard_order);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromBuffer(std::shared_ptr<Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    ReadFromBuffer(*buffer, host_buffer, shard_order);
}

void ReadFromBuffer(Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    Device *device = buffer.device();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::TRACE:
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            if (buffer.is_dram()) {
                tt::Cluster::instance().dram_barrier(device->id());
            } else {
                tt::Cluster::instance().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer, shard_order);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Reading from host memory is unsupported!");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

void ReadShard(Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t &core_id) {
    Device *device = buffer.device();
    TT_ASSERT(is_sharded(buffer.buffer_layout()));
    host_buffer.clear();  // overwrite the data

    uint32_t num_entries_per_page = buffer.page_size() / sizeof(uint32_t);
    uint32_t num_entries_per_shard = num_entries_per_page * buffer.shard_spec().size();
    host_buffer = std::vector<uint32_t>(num_entries_per_shard);

    std::vector<uint32_t> page_ids;
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    for (uint32_t i = 0; i < buffer_page_mapping.dev_page_to_core_mapping_.size(); i++) {
        if (buffer_page_mapping.dev_page_to_core_mapping_[i] == core_id) {
            page_ids.push_back(i);
        }
    }

    uint32_t host_page_id = 0;
    for (auto dev_page_id : page_ids) {
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        read_pages_to_host_helper(device, buffer, host_buffer, buffer.page_size(), host_page_id, dev_page_id, bank_id);
        host_page_id++;
    }
}

void LaunchProgram(Device *device, std::shared_ptr<Program> program, bool wait_until_cores_done) {
    LaunchProgram(device, *program, wait_until_cores_done);
}

void LaunchProgram(Device *device, Program &program, bool wait_until_cores_done) {
    {  // Profiler scope start
        ZoneScoped;
        detail::DispatchStateCheck(false);
        detail::CompileProgram(device, program);
        if (!program.is_finalized()) {
            program.finalize(device);
        }

        detail::WriteRuntimeArgsToDevice(device, program);
        detail::ConfigureDeviceWithProgram(device, program);

        auto device_id = device->id();

        tt::Cluster::instance().dram_barrier(device_id);

        // Note: the l1_barrier below is needed to be sure writes to cores that
        // don't get the GO mailbox (eg, storage cores) have all landed
        tt::Cluster::instance().l1_barrier(device->id());

        std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
        std::unordered_set<CoreCoord> not_done_cores;
        for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size(); programmable_core_type_index++) {
            CoreType core_type = hal.get_core_type(programmable_core_type_index);
            for (const auto &logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
                launch_msg_t *msg = &program.kernels_on_core(logical_core, programmable_core_type_index)->launch_msg;
                go_msg_t* go_msg = &program.kernels_on_core(logical_core, programmable_core_type_index)->go_msg;
                msg->kernel_config.host_assigned_id = program.get_runtime_id();

                auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
                not_done_cores.insert(physical_core);
                tt::llrt::write_launch_msg_to_core(device->id(), physical_core, msg, go_msg, device->get_dev_addr(physical_core, HalL1MemAddrType::LAUNCH));
            }
        }
        if (wait_until_cores_done) {
            // Wait for all cores to be done
            llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
        }
    }  // Profiler scope end
    if (wait_until_cores_done) {
        DumpDeviceProfileResults(device, program);
    }
}

void WaitProgramDone(Device *device, Program &program) {
    auto device_id = device->id();
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
    std::unordered_set<CoreCoord> not_done_cores;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto & logical_cores = logical_cores_used_in_program[index];
        CoreType core_type = hal.get_core_type(index);
        for (const auto &logical_core : logical_cores) {
            auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
            not_done_cores.insert(physical_core);
        }
    }
    // Wait for all cores to be done
    llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
    DumpDeviceProfileResults(device, program);
}

bool ConfigureDeviceWithProgram(Device *device, Program &program, bool fd_bootloader_mode) {
    ZoneScoped;
    bool pass = true;
    // This is function is shared between FD and SD.
    // We call this function when initializing HW Command Queues (tracked as fd_bootloader_mode) for Fast Dispatch.
    // Used to Launch programs for Slow dispatch.
    bool using_fast_dispatch = fd_bootloader_mode;
    detail::DispatchStateCheck(using_fast_dispatch);

    auto device_id = device->id();

    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto & logical_cores = logical_cores_used_in_program[index];
        CoreType core_type = hal.get_core_type(index);
        for (const auto &logical_core : logical_cores) {
            KernelGroup *kernel_group = program.kernels_on_core(logical_core, index);
            CoreCoord physical_core = device->physical_core_from_logical_core(logical_core, core_type);

            ConfigureKernelGroup(program, kernel_group, device, logical_core);
            // TODO: add support for CB for ethernet cores
            if (core_type == CoreType::WORKER) {
                // CircularBufferConfigVec -- common across all kernels, so written once to the core
                std::vector<uint32_t> circular_buffer_config_vec(program.get_program_config(index).cb_size / sizeof(uint32_t));

                auto cbs_on_core = program.circular_buffers_on_core(logical_core);
                for (auto circular_buffer : cbs_on_core) {
                    for (uint32_t buffer_index : circular_buffer->buffer_indices()) {
                        uint32_t addr_in_bytes = circular_buffer->address();
                        uint32_t size_in_bytes = circular_buffer->size();
                        uint32_t num_pages = circular_buffer->num_pages(buffer_index);
                        uint32_t page_size = size_in_bytes / num_pages;
                        circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index) =
                            addr_in_bytes >> 4;  // convert to addr in 16B words
                        circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index + 1) =
                            size_in_bytes >> 4;  // convert to addr in 16B words
                        circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index + 2) = num_pages;
                        circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index + 3) = page_size >> 4;
                    }
                }  // PROF_END("CBS")

                if (cbs_on_core.size()) {
                    uint64_t kernel_config_base = hal.get_dev_addr(index, HalL1MemAddrType::KERNEL_CONFIG);
                    uint64_t addr = kernel_config_base + program.get_program_config(index).cb_offset;
                    llrt::write_hex_vec_to_core(device_id, physical_core, circular_buffer_config_vec, addr);
                }
            }
            program.init_semaphores(*device, logical_core, index);
        }
    }

    return pass;
}

void WriteRuntimeArgsToDevice(Device *device, Program &program) {
    ZoneScoped;
    auto device_id = device->id();
    detail::DispatchStateCheck(false);

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        uint32_t processor_classes = hal.get_processor_classes_count(index);
        for (auto& kg : program.get_kernel_groups(index)) {
            uint32_t kernel_config_base = kg.launch_msg.kernel_config.kernel_config_base[index];
            for (const CoreRange &core_range : kg.core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core(x, y);
                        auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
                        for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
                            auto& optional_id = kg.kernel_ids[dispatch_class];
                            if (optional_id) {
                                const auto &kernel = detail::GetKernel(program, optional_id.value());
                                const auto &rt_args = kernel->runtime_args(logical_core);

                                if (rt_args.size() > 0) {
                                    auto rt_args_addr = kernel_config_base + kg.launch_msg.kernel_config.rta_offset[dispatch_class].rta_offset;
                                    log_trace(
                                              tt::LogMetal,
                                              "{} - Writing {} unique rtargs to core {} (physical: {}) addr 0x{:x} => args: {}",
                                              __FUNCTION__,
                                              rt_args.size(),
                                              logical_core.str(),
                                              physical_core.str(),
                                              rt_args_addr,
                                              rt_args);
                                    tt::llrt::write_hex_vec_to_core(device_id, physical_core, rt_args, rt_args_addr);
                                }

                                const auto &common_rt_args = kernel->common_runtime_args();
                                if (common_rt_args.size() > 0) {
                                    auto common_rt_args_addr = kernel_config_base + kg.launch_msg.kernel_config.rta_offset[dispatch_class].crta_offset;
                                    log_trace(
                                              tt::LogMetal,
                                              "{} - Writing {} common rtargs to core {} (physical: {}) addr 0x{:x} => args: {}",
                                              __FUNCTION__,
                                              common_rt_args.size(),
                                              logical_core.str(),
                                              physical_core.str(),
                                              common_rt_args_addr,
                                              common_rt_args);
                                    tt::llrt::write_hex_vec_to_core(device_id, physical_core, common_rt_args, common_rt_args_addr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void CompileProgram(Device *device, Program &program, bool fd_bootloader_mode) {
    ZoneScoped;
    program.compile(device, fd_bootloader_mode);
}

void AllocateBuffer(Buffer *buffer, bool bottom_up) {
    if(GraphTracker::instance().hook_allocate(buffer, bottom_up)) {
        GraphTracker::instance().track_allocate(buffer, bottom_up);
        return;
    }
    EnqueueAllocateBuffer(buffer->device()->command_queue(), buffer, bottom_up, false);
    GraphTracker::instance().track_allocate(buffer, bottom_up);
}

void DeallocateBuffer(Buffer *buffer) {
    GraphTracker::instance().track_deallocate(buffer);
    if(GraphTracker::instance().hook_deallocate(buffer)) {
        return;
    }
    EnqueueDeallocateBuffer(
        buffer->device()->command_queue(),
        *(buffer->device()->allocator_),
        buffer->address(),
        buffer->buffer_type(),
        false);
}

}  // namespace detail

inline namespace v0 {

size_t GetNumAvailableDevices() {
    return tt::Cluster::instance().number_of_user_devices();
}

bool IsGalaxyCluster() {
    return tt::Cluster::instance().is_galaxy_cluster();
}

size_t GetNumPCIeDevices() {
    return tt::Cluster::instance().number_of_pci_devices();
}

chip_id_t GetPCIeDeviceID(chip_id_t device_id){
    return tt::Cluster::instance().get_associated_mmio_device(device_id);
}

Device *CreateDevice(
    chip_id_t device_id,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    DispatchCoreType dispatch_core_type,
    const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;

    tt::DevicePool::initialize({device_id}, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type, l1_bank_remap);
    auto dev = tt::DevicePool::instance().get_active_device(device_id);
    return dev;
}

Device *CreateDeviceMinimal(chip_id_t device_id, const uint8_t num_hw_cqs, DispatchCoreType dispatch_core_type) {
    ZoneScoped;
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_type, num_hw_cqs);
    Device *dev = new Device(device_id, num_hw_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, {}, true);
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(Device *device) {
    ZoneScoped;
    auto device_id = device->id();
    return tt::DevicePool::instance().close_device(device_id);
}

Program CreateProgram() { return Program(); }

KernelHandle CreateDataMovementKernel(
    Program &program,
    const KernelSource &kernel_src,
    const CoreRangeSet &core_range_set,
    const DataMovementConfig &config) {
    const DataMovementConfigStatus &data_movement_config_status = CheckDataMovementConfig(program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    string kernel_name;
    if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
        kernel_name = kernel_src.source_;
    } else {
        TT_FATAL(kernel_src.source_type_ == KernelSource::SOURCE_CODE, "Unsupported kernel source type!");
        kernel_name = "kernel";
    }

    TT_FATAL(
        !(are_both_riscv_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel_name);
    TT_FATAL(
        !(are_both_noc_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel_name);

    std::shared_ptr<Kernel> kernel = std::make_shared<DataMovementKernel>(kernel_src, core_range_set, config);
    return detail::AddKernel(program, kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateComputeKernel(
    Program &program, const KernelSource &kernel_src, const CoreRangeSet &core_range_set, const ComputeConfig &config) {
    std::shared_ptr<Kernel> kernel = std::make_shared<ComputeKernel>(kernel_src, core_range_set, config);
    return detail::AddKernel(program, kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateEthernetKernel(
    Program &program,
    const KernelSource &kernel_src,
    const CoreRangeSet &core_range_set,
    const EthernetConfig &config) {
    KernelHandle kernel_handle;
    std::shared_ptr<Kernel> kernel = std::make_shared<EthernetKernel>(kernel_src, core_range_set, config);
    if (config.eth_mode == Eth::IDLE) {
        kernel_handle = detail::AddKernel(program, kernel, HalProgrammableCoreType::IDLE_ETH);
    } else {
        kernel_handle = detail::AddKernel(program, kernel, HalProgrammableCoreType::ACTIVE_ETH);
    }
    return kernel_handle;
}

KernelHandle CreateKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return std::visit(
        [&](auto &&cfg) -> KernelHandle {
            CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
            KernelSource kernel_src(file_name, KernelSource::FILE_PATH);
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            } else if constexpr (std::is_same_v<T, ComputeConfig>) {
                return CreateComputeKernel(program, kernel_src, core_ranges, cfg);
            } else if constexpr (std::is_same_v<T, EthernetConfig>) {
                return CreateEthernetKernel(program, kernel_src, core_ranges, cfg);
            }
        },
        config);
}

KernelHandle CreateKernelFromString(
    Program &program,
    const std::string &kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return std::visit(
        [&](auto &&cfg) -> KernelHandle {
            CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
            KernelSource kernel_src(kernel_src_code, KernelSource::SOURCE_CODE);
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            } else if constexpr (std::is_same_v<T, ComputeConfig>) {
                return CreateComputeKernel(program, kernel_src, core_ranges, cfg);
            } else if constexpr (std::is_same_v<T, EthernetConfig>) {
                return CreateEthernetKernel(program, kernel_src, core_ranges, cfg);
            }
        },
        config);
}

CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    return program.add_circular_buffer(core_ranges, config);
}

const CircularBufferConfig &GetCircularBufferConfig(Program &program, CBHandle cb_handle) {
    return detail::GetCircularBuffer(program, cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program &program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBuffer> circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program &program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    detail::GetCircularBuffer(program, cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program &program, CBHandle cb_handle, const Buffer &buffer) {
    auto circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}

uint32_t CreateSemaphore(
    Program &program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return std::visit(
        [&](auto &&c) -> uint32_t {
            using T = std::decay_t<decltype(c)>;
            CoreRangeSet crs({});
            if constexpr (std::is_same_v<T, CoreRange>) {
                crs = CoreRangeSet({c});
            } else {
                crs = c;
            }
            std::optional<uint32_t> semaphore_id;
            TT_FATAL(crs.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
            for (const auto &core_range : crs.ranges()) {
                CoreCoord start_core = core_range.start_coord;
                CoreCoord end_core = core_range.end_coord;
                std::optional<uint32_t> semaphore_id_candidate = get_semaphore_id(program, core_range);
                if (!semaphore_id.has_value()) {
                    semaphore_id = semaphore_id_candidate;
                } else {
                    semaphore_id = std::max(semaphore_id.value(), semaphore_id_candidate.value());
                }
            }
            TT_FATAL(semaphore_id.has_value(), "Unable to initialize Semaphore!");

            program.add_semaphore(crs, semaphore_id.value(), initial_value, core_type);

            return semaphore_id.value();
        },
        core_spec);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device, config.size, config.page_size, config.buffer_type, config.buffer_layout, std::nullopt, std::nullopt, config.allocate);
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters,
        std::nullopt,
        config.allocate);
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }

void AssignGlobalBufferToProgram(
    std::shared_ptr<Buffer> buffer, Program& program) {
    detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
    EnqueueAddBufferToProgram(buffer->device()->command_queue(), buffer, program, false);
}

void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    stl::Span<const uint32_t> runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast "
        "Dispatch.");
    std::visit([&](auto &&core_spec) { SetRuntimeArgsImpl(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::vector<uint32_t>> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast "
        "Dispatch.");
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mistmatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    auto k = detail::GetKernel(program, kernel);
    for (size_t i = 0; i < core_spec.size(); i++) k->set_runtime_args(core_spec[i], runtime_args[i]);
}

void SetRuntimeArgs(
    Device *device,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args) {
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgsImpl(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetRuntimeArgs(
    Device *device,
    const std::shared_ptr<Kernel> kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args) {
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mismatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgsImpl(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetCommonRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for "
        "Fast Dispatch.");
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData &GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>> &GetRuntimeArgs(const Program &program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->runtime_args_data();
}

RuntimeArgsData &GetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->common_runtime_args_data();
}

uint32_t BeginTraceCapture(Device *device, const uint8_t cq_id) {
    const uint32_t tid = Trace::next_id();
    device->begin_trace(cq_id, tid);
    return tid;
}

void EndTraceCapture(Device *device, const uint8_t cq_id, const uint32_t tid) { device->end_trace(cq_id, tid); }

void ReplayTrace(Device *device, const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    device->replay_trace(cq_id, tid, blocking);
}

void ReleaseTrace(Device *device, const uint32_t tid) { device->release_trace(tid); }

void Synchronize(Device *device, const std::optional<uint8_t> cq_id) {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        if (cq_id.has_value()) {
            Finish(device->command_queue(cq_id.value()));
        } else {
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
                Finish(device->command_queue(cq_id));
            }
        }
    }
}

}  // namespace v0
}  // namespace tt_metal

}  // namespace tt
