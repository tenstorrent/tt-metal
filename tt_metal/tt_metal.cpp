// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/api_backdoor.hpp"

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

struct Program {
    MetalProgram *metal_program;
};

namespace {

inline void SetRuntimeArgs(
    const MetalProgram &program, KernelHandle kernel_id, const CoreCoord &c, const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        program.get_kernel(kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgs(
    const MetalProgram &program,
    KernelHandle kernel_id,
    const CoreRange &core_range,
    const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = program.get_kernel(kernel_id);
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
            }
        }
    }
}

inline void SetRuntimeArgs(
    const MetalProgram &program,
    KernelHandle kernel_id,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = program.get_kernel(kernel_id);
        for (const auto &core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

inline void SetRuntimeArgs(
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

inline void SetRuntimeArgs(
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

MetalProgram *GetMetalProgram(const Program *program) {
    return program->metal_program;
}

bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
{
    bool pass = true;
    TT_FATAL(address >= DRAM_UNRESERVED_BASE, "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!", DRAM_UNRESERVED_BASE);
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
    if (is_galaxy) {
        TT_FATAL(num_hw_cqs < 2, "Multiple Command Queues are not Currently Supported on Galaxy Systems");
    }
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
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
    std::map<chip_id_t, Device *> mmio_devices = {};
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
                mmio_devices.insert({device_id, dev});
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
                }
            }
        }
        //finally close the mmio device
        dev->close();
    }
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

void WriteToDeviceSharded(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
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

    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
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
    TT_FATAL(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_FATAL(page_size % bytes_per_page_entry == 0);
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
            default: TT_FATAL(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}

void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
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

void WriteToBuffer(std::shared_ptr<const Buffer> buffer, const std::vector<uint32_t> &host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:  // fallthrough
        case BufferType::L1:    // fallthrough
        case BufferType::L1_SMALL: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_FATAL(false && "Writing to host memory is unsupported!");
        } break;
        default: TT_FATAL(false && "Unsupported buffer type!");
    }
}

void ReadFromDeviceInterleavedContiguous(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    host_buffer.clear();  // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_FATAL(buffer.size() % page_size == 0);
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
            default: TT_FATAL(false && "Unsupported buffer type to read from device!");
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
    const Buffer &dev_buffer,
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

void ReadFromDeviceSharded(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
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

    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
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

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
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

void ReadFromBuffer(std::shared_ptr<const Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    ReadFromBuffer(*buffer, host_buffer, shard_order);
}

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
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
            TT_FATAL(false && "Reading from host memory is unsupported!");
        } break;
        default: TT_FATAL(false && "Unsupported buffer type!");
    }
}

void ReadShard(const Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t &core_id) {
    Device *device = buffer.device();
    TT_ASSERT(is_sharded(buffer.buffer_layout()));
    host_buffer.clear();  // overwrite the data

    uint32_t num_entries_per_page = buffer.page_size() / sizeof(uint32_t);
    uint32_t num_entries_per_shard = num_entries_per_page * buffer.shard_spec().size();
    host_buffer = std::vector<uint32_t>(num_entries_per_shard);

    std::vector<uint32_t> page_ids;
    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
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

// TODO: This is a problem
void LaunchProgram(Device *device, std::shared_ptr<Program> program, bool wait_until_cores_done, bool force_slow_dispatch) {
    LaunchProgram(device, program, wait_until_cores_done, force_slow_dispatch);
}


void LaunchProgram(Device *device, Program *program, bool wait_until_cores_done, bool force_slow_dispatch) {
    device->launch_program(program->metal_program, wait_until_cores_done, force_slow_dispatch);
}

void WaitProgramDone(Device *device, Program *program) {
    auto device_id = device->id();
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program->metal_program->logical_cores();
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
    DumpDeviceProfileResults(device);
}

bool ConfigureDeviceWithProgram(Device *device, Program *program, bool fd_bootloader_mode) {
    return device->configure_with_program(program->metal_program, fd_bootloader_mode);
}

void WriteRuntimeArgsToDevice(Device *device, Program *program, bool force_slow_dispatch) {
    return device->write_runtime_args(program->metal_program, force_slow_dispatch);
}

void CompileProgram(Device *device, Program *program, bool fd_bootloader_mode) {
    ZoneScoped;
    program->metal_program->compile(device, fd_bootloader_mode);
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

size_t GetNumAvailableDevices() {
    return tt::Cluster::instance().number_of_user_devices();
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
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_type);
    Device *dev = new Device(device_id, num_hw_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, {}, true);
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(Device *device) {
    ZoneScoped;
    auto device_id = device->id();
    return tt::DevicePool::instance().close_device(device_id);
}

Program *CreateProgram() {
    struct Program *program = (struct Program *) malloc(sizeof(struct Program));
    program->metal_program = new MetalProgram;
    return program;
}

void DestroyProgram(Program *program) {
    delete program->metal_program;
    free(program);
}

KernelHandle CreateKernel(
    Program *program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return program->metal_program->create_kernel(file_name, core_spec, config);
}

CBHandle CreateCircularBuffer(
    Program *program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = CoreRangeSet::get_core_range_set(core_spec);
    return program->metal_program->add_circular_buffer(core_ranges, config);
}

const CircularBufferConfig &GetCircularBufferConfig(Program *program, CBHandle cb_handle) {
    return program->metal_program->get_circular_buffer(cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program *program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBuffer> circular_buffer = program->metal_program->get_circular_buffer(cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program->metal_program->invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program *program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    program->metal_program->get_circular_buffer(cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program *program, CBHandle cb_handle, const Buffer &buffer) {
    auto circular_buffer = program->metal_program->get_circular_buffer(cb_handle);
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}

uint32_t CreateSemaphore(
    Program *program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return program->metal_program->create_semaphore(core_spec, initial_value, core_type);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device, config.size, config.page_size, config.buffer_type, config.buffer_layout);
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters);
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }

void AssignGlobalBufferToProgram(
    std::shared_ptr<Buffer> buffer, Program *program) {
    detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
    EnqueueAddBufferToProgram(buffer->device()->command_queue(), buffer, program->metal_program, false);
}

void SetRuntimeArgs(
    const Program *program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast "
        "Dispatch.");
    std::visit([&](auto &&core_spec) { SetRuntimeArgs(*program->metal_program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program *program,
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
    auto k = program->metal_program->get_kernel(kernel);
    for (size_t i = 0; i < core_spec.size(); i++) k->set_runtime_args(core_spec[i], runtime_args[i]);
}

void SetRuntimeArgs(
    Device *device,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args) {
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
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
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetCommonRuntimeArgs(const Program *program, KernelHandle kernel_id, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetCommonRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for "
        "Fast Dispatch.");
    if (runtime_args.size() != 0) {
        program->metal_program->get_kernel(kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData &GetRuntimeArgs(const Program *program, KernelHandle kernel_id, const CoreCoord &logical_core) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return program->metal_program->get_kernel(kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>> &GetRuntimeArgs(const Program *program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return program->metal_program->get_kernel(kernel_id)->runtime_args_data();
}

RuntimeArgsData &GetCommonRuntimeArgs(const Program *program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return program->metal_program->get_kernel(kernel_id)->common_runtime_args_data();
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

void EnqueueProgram(CommandQueue& cq, Program *program, bool blocking) {
    EnqueueProgram(cq, program->metal_program, blocking);
}

}  // namespace tt_metal

}  // namespace tt
