// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <circular_buffer.hpp>
#include <circular_buffer_constants.h>
#include <tt_stl/assert.hpp>
#include <cstdint>
#include "device/device_manager.hpp"
#include <global_circular_buffer.hpp>
#include <global_semaphore.hpp>
#include <host_api.hpp>
#include <enchantum/enchantum.hpp>
#include <memory>
#include <sub_device_types.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>

#include "buffer_types.hpp"
#include "circular_buffer_config.hpp"
#include "data_types.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <filesystem>
#include "device.hpp"
#include "context/metal_context.hpp"
#include "kernels/kernel.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "device/device_impl.hpp"
#include "hal_types.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>
#include <tt-metalium/experimental/lightmetal/lightmetal_api.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_metal_profiler.hpp>
#include <program.hpp>
#include "program/program_impl.hpp"
#include "impl/buffers/semaphore.hpp"
#include "tracy/Tracy.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt_stl/enum.hpp>
#include "fabric/hw/inc/fabric_routing_mode.h"
#include <graph_tracking.hpp>
#include <tt_stl/overloaded.hpp>
#include "get_platform_architecture.hpp"
#include "common/tt_backend_api_types.hpp"
#include <experimental/fabric/control_plane.hpp>
#include "impl/buffers/circular_buffer.hpp"

namespace tt::tt_metal {
enum class FabricConfig : uint32_t;
struct RuntimeArgsData;
struct TraceDescriptor;

namespace {

CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet>& specified_core_spec) {
    ZoneScoped;
    return std::visit(
        ttsl::overloaded{
            [](const CoreCoord& core_spec) { return CoreRangeSet(CoreRange(core_spec, core_spec)); },
            [](const CoreRange& core_spec) { return CoreRangeSet(core_spec); },
            [](const CoreRangeSet& core_spec) { return core_spec; },
        },
        specified_core_spec);
}

struct DataMovementConfigStatus {
    bool riscv0_in_use;
    bool riscv1_in_use;
    bool noc0_in_use;
    bool noc1_in_use;
};

DataMovementConfigStatus CheckDataMovementConfig(
    const HalProgrammableCoreType& programmable_core, Program& program, const CoreRangeSet& core_ranges) {
    DataMovementConfigStatus data_movement_config_status{
        .riscv0_in_use = false, .riscv1_in_use = false, .noc0_in_use = false, .noc1_in_use = false};

    auto set_global_and_local_noc_usage = [&](const std::shared_ptr<Kernel>& kernel,
                                              bool& local_noc0_usage,
                                              bool& local_noc1_usage) {
        int noc_value;
        switch (programmable_core) {
            case HalProgrammableCoreType::TENSIX:
                noc_value = enchantum::to_underlying(std::get<DataMovementConfig>(kernel->config()).noc);
                break;
            case HalProgrammableCoreType::ACTIVE_ETH:
            case HalProgrammableCoreType::IDLE_ETH:
                noc_value = enchantum::to_underlying(std::get<EthernetConfig>(kernel->config()).noc);
                break;
            default:
                TT_THROW(
                    "Checking NoC and DataMovementProcessor is unsupported for programmable core {}",
                    enchantum::to_string(programmable_core));
        }
        local_noc0_usage = noc_value == 0;
        local_noc1_usage = noc_value == 1;
        data_movement_config_status.noc0_in_use = local_noc0_usage;
        data_movement_config_status.noc1_in_use = local_noc1_usage;
    };

    const auto& hal = MetalContext::instance().hal();
    for (const auto& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup* kernel_group = program.impl().kernels_on_core(
                    CoreCoord(x, y), hal.get_programmable_core_type_index(programmable_core));
                if (kernel_group != nullptr) {
                    bool local_noc0_in_use = false;
                    bool local_noc1_in_use = false;
                    bool has_dm0 = false;
                    bool has_dm1 = false;
                    for (auto kernel_id : kernel_group->kernel_ids) {
                        const auto kernel = program.impl().get_kernel(kernel_id);
                        if (kernel->get_kernel_processor_class() == HalProcessorClassType::DM) {
                            switch (kernel->get_kernel_processor_type(0)) {
                                case 0:
                                    has_dm0 = true;
                                    data_movement_config_status.riscv0_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                case 1:
                                    has_dm1 = true;
                                    data_movement_config_status.riscv1_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                default: TT_THROW("Unknown DataMovementProcessor type"); break;
                            }
                        }
                    }
                    if (has_dm0 and has_dm1) {
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
    Program& program,
    uint32_t programmable_core_type_index,
    const KernelGroup* kernel_group,
    IDevice* device,
    const CoreCoord& logical_core) {
    const auto& hal = MetalContext::instance().hal();
    uint32_t kernel_config_base =
        hal.get_dev_addr(hal.get_programmable_core_type(programmable_core_type_index), HalL1MemAddrType::KERNEL_CONFIG);
    for (auto kernel_id : kernel_group->kernel_ids) {
        // Need the individual offsets of each bin
        // TODO: make configure take a std::span
        program.impl().get_kernel(kernel_id)->configure(
            device, logical_core, kernel_config_base, kernel_group->kernel_text_offsets.data());
    }
}

std::optional<uint32_t> get_semaphore_id(const Program& program, const CoreRange& core_range, CoreType core_type) {
    std::optional<uint32_t> semaphore_id = std::nullopt;
    std::vector<uint32_t> semaphore_histogram(NUM_SEMAPHORES, 0);
    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            CoreCoord logical_core(x, y);
            auto semaphores = program.impl().semaphores_on_core(logical_core, core_type);
            if (semaphores.size() == NUM_SEMAPHORES) {
                TT_THROW(
                    "Cannot add semaphore on core {}. Max number of semaphores ({}) reached!",
                    logical_core.str(),
                    NUM_SEMAPHORES);
            }

            for (const auto& semaphore : semaphores) {
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
        semaphore_id = uninitialized_sem_id;
    } else {
        TT_THROW("Unable to initialize semaphores on core range {}", core_range.str());
    }

    return semaphore_id;
}

inline void SetRuntimeArgsImpl(
    const Program& program, KernelHandle kernel_id, const CoreCoord& c, stl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        program.impl().get_kernel(kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgsImpl(
    const Program& program,
    KernelHandle kernel_id,
    const CoreRange& core_range,
    stl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        auto kernel = program.impl().get_kernel(kernel_id);
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
            }
        }
    }
}

inline void SetRuntimeArgsImpl(
    const Program& program,
    KernelHandle kernel_id,
    const CoreRangeSet& core_range_set,
    stl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        auto kernel = program.impl().get_kernel(kernel_id);
        for (const auto& core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

}  // namespace

namespace detail {

bool WriteToDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, std::span<const std::uint8_t> host_buffer) {
    TT_FATAL(
        address >= device->allocator()->get_base_allocator_addr(HalMemType::DRAM),
        "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!",
        device->allocator()->get_base_allocator_addr(HalMemType::DRAM));
    MetalContext::instance().get_cluster().write_dram_vec(
        host_buffer.data(), host_buffer.size(), device->id(), dram_channel, address);
    return true;
}

bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer) {
    return WriteToDeviceDRAMChannel(
        device,
        dram_channel,
        address,
        std::span(reinterpret_cast<const std::uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(uint32_t)));
}

bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer) {
    bool pass = true;
    MetalContext::instance().get_cluster().dram_barrier(device->id());
    MetalContext::instance().get_cluster().read_dram_vec(
        host_buffer.data(), host_buffer.size(), device->id(), dram_channel, address);
    return pass;
}

bool ReadFromDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer) {
    host_buffer.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    return ReadFromDeviceDRAMChannel(
        device, dram_channel, address, std::span(reinterpret_cast<std::uint8_t*>(host_buffer.data()), size));
}

bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const std::uint8_t> host_buffer,
    CoreType core_type) {
    ZoneScoped;
    auto worker_core = device->virtual_core_from_logical_core(logical_core, core_type);
    MetalContext::instance().get_cluster().write_core(device->id(), worker_core, host_buffer, address);
    return true;
}

bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    return WriteToDeviceL1(
        device,
        logical_core,
        address,
        std::span(reinterpret_cast<const std::uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(uint32_t)),
        core_type);
}

bool WriteRegToDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, const uint32_t& regval) {
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    MetalContext::instance().get_cluster().write_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<uint8_t> host_buffer,
    CoreType core_type) {
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    MetalContext::instance().get_cluster().read_core(
        host_buffer.data(), host_buffer.size(), tt_cxy_pair(device->id(), virtual_core), address);
    return true;
}

bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    host_buffer = MetalContext::instance().get_cluster().read_core(device->id(), virtual_core, address, size);
    return true;
}

bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval) {
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    MetalContext::instance().get_cluster().read_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

std::string get_platform_architecture_name() { return tt::get_string_lowercase(get_platform_architecture({})); }

IDevice* GetActiveDevice(ChipId device_id) {
    IDevice* device = nullptr;
    if (MetalContext::instance().device_manager()->is_device_active(device_id)) {
        device = MetalContext::instance().device_manager()->get_active_device(device_id);
    }
    return device;
}

std::map<ChipId, IDevice*> CreateDevices(
    const std::vector<ChipId>& device_ids,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& /*l1_bank_remap*/,
    const size_t worker_l1_size,
    bool init_profiler,
    [[maybe_unused]] bool ignored,
    bool initialize_fabric_and_dispatch_fw) {
    ZoneScoped;
    bool is_galaxy = MetalContext::instance().get_cluster().is_galaxy_cluster();
    MetalContext::instance().initialize_device_manager(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        init_profiler,
        initialize_fabric_and_dispatch_fw);

    const auto devices = MetalContext::instance().device_manager()->get_all_active_devices();
    std::map<ChipId, IDevice*> ret_devices;
    // Only include the mmio device in the active devices set returned to the caller if we are not running
    // on a Galaxy cluster.
    // On Galaxy, gateway (mmio devices) cannot run compute workloads.

    for (IDevice* dev : devices) {
        if (is_galaxy and dev->is_mmio_capable()) {
            continue;
        }
        ret_devices.insert({dev->id(), dev});
    }

    return ret_devices;
}

void CloseDevices(const std::map<ChipId, IDevice*>& devices) {
    std::vector<IDevice*> devices_to_close;
    devices_to_close.reserve(devices.size());
    for (const auto& [id, device] : devices) {
        devices_to_close.push_back(device);
    }
    MetalContext::instance().device_manager()->close_devices(devices_to_close);
}

void print_page(
    uint32_t dev_page_id,
    CoreCoord core,
    uint32_t host_page_id,
    CoreCoord noc_coordinates,
    uint32_t l1_address,
    uint32_t bank_id,
    const std::vector<uint32_t>& page) {
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

void WriteToDeviceSharded(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
    TT_FATAL(
        host_buffer.size() <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer.size(),
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(page_size == 0 ? buffer.size() == 0 : buffer.size() % page_size == 0);

    auto* device = buffer.device();
    const auto& allocator = device->allocator();

    const auto& cluster = MetalContext::instance().get_cluster();
    const size_t alignment_req = cluster.get_alignment_requirements(device->id(), page_size);
    const size_t aligned_bytes = alignment_req ? (page_size / alignment_req) * alignment_req : page_size;
    const size_t remainder_bytes = page_size - aligned_bytes;
    TT_ASSERT(buffer.aligned_page_size() >= page_size);  // Check that we don't write to the end of the buffer
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    for (auto mapped_page : buffer_page_mapping) {
        auto core = buffer_page_mapping.all_cores[mapped_page.core_id];
        auto bank_id = allocator->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto bank_offset = allocator->get_bank_offset(buffer.buffer_type(), bank_id);
        auto data_index = mapped_page.host_page * page_size;
        auto write_chunk = [&](size_t offset, size_t size_in_bytes) {
            if (size_in_bytes == 0) {
                return;
            }
            std::span<const std::uint8_t> page(host_buffer.data() + data_index + offset, size_in_bytes);
            if (buffer.is_l1()) {
                auto absolute_address =
                    buffer.address() + bank_offset + (mapped_page.device_page * buffer.aligned_page_size()) + offset;
                auto core_coordinates =
                    device->worker_core_from_logical_core(buffer.allocator()->get_logical_core_from_bank_id(bank_id));
                MetalContext::instance().get_cluster().write_core(
                    device->id(), core_coordinates, page, absolute_address);
            } else {
                auto bank_local_address =
                    buffer.address() + (mapped_page.device_page * buffer.aligned_page_size()) + offset;
                WriteToDeviceDRAMChannel(device, bank_id, bank_local_address, page);
            }
        };

        write_chunk(0, aligned_bytes);
        write_chunk(aligned_bytes, remainder_bytes);
    }
}

DeviceAddr CalculateAddressDeviceInterleavedContiguous(const Buffer& buffer, uint64_t bank_index, uint64_t page_index) {
    DeviceAddr addr = 0;
    if (buffer.is_dram()) {
        uint32_t num_banks = buffer.allocator()->get_num_banks(buffer.buffer_type());
        uint32_t pages_offset_within_bank = page_index / num_banks;
        addr = buffer.address() + pages_offset_within_bank * buffer.aligned_page_size();
    } else {
        TT_ASSERT(buffer.is_l1());
        addr = buffer.page_address(bank_index, page_index);
    }

    return addr;
}

void WriteToDeviceInterleavedContiguous(const Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
    if (GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }

    size_t host_buffer_size_bytes = host_buffer.size();
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    size_t page_size = buffer.page_size();
    size_t num_pages = buffer.num_pages();

    auto* device = buffer.device();
    size_t num_banks = device->allocator()->get_num_banks(buffer.buffer_type());
    size_t bank_index = 0;
    size_t data_index = 0;

    const auto& cluster = MetalContext::instance().get_cluster();
    const size_t alignment_req = cluster.get_alignment_requirements(device->id(), page_size);
    const size_t aligned_bytes = alignment_req ? (page_size / alignment_req) * alignment_req : page_size;
    const size_t remainder_bytes = page_size - aligned_bytes;
    TT_ASSERT(buffer.aligned_page_size() >= page_size);  // Check that we don't write to the end of the buffer
    for (size_t page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        auto write_chunk = [&](size_t offset, size_t size_in_bytes) {
            if (size_in_bytes == 0) {
                return;
            }
            std::span<const std::uint8_t> page(host_buffer.data() + data_index + offset, size_in_bytes);
            switch (buffer.buffer_type()) {
                case BufferType::DRAM: WriteToDeviceDRAMChannel(device, bank_index, address + offset, page); break;
                case BufferType::L1:
                case BufferType::L1_SMALL: {
                    CoreCoord logical_core = buffer.allocator()->get_logical_core_from_bank_id(bank_index);
                    WriteToDeviceL1(device, logical_core, address + offset, page, CoreType::WORKER);
                } break;
                default: TT_THROW("Unsupported buffer type to write to device!");
            }
        };

        write_chunk(0, aligned_bytes);
        write_chunk(aligned_bytes, remainder_bytes);

        bank_index = (bank_index + 1) % num_banks;
        data_index += page_size;
    }
}

void WriteToDevice(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED) {
        WriteToDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        WriteToDeviceSharded(buffer, host_buffer);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void WriteToBuffer(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
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

void ReadFromDeviceInterleavedContiguous(const Buffer& buffer, uint8_t* host_buffer) {
    size_t page_size = buffer.page_size();
    size_t num_pages = buffer.num_pages();

    auto* device = buffer.device();
    size_t num_banks = device->allocator()->get_num_banks(buffer.buffer_type());

    size_t host_idx = 0;
    size_t bank_index = 0;

    const auto& cluster = MetalContext::instance().get_cluster();
    size_t aligned_page_size = tt::align(page_size, cluster.get_alignment_requirements(device->id(), page_size));

    std::vector<uint8_t> page(aligned_page_size);
    for (size_t page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::TRACE: {
                ReadFromDeviceDRAMChannel(device, bank_index, address, std::span<uint8_t>(page));
            } break;
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto core_coordinates = device->worker_core_from_logical_core(
                    buffer.allocator()->get_logical_core_from_bank_id(bank_index));
                tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    page.data(), aligned_page_size, tt_cxy_pair(device->id(), core_coordinates), address);
            } break;
            default: TT_THROW("Unsupported buffer type to read from device!");
        }

        // Copy page into host buffer
        std::memcpy(host_buffer + host_idx, page.data(), page_size);
        host_idx += page_size;

        bank_index = (bank_index + 1) % num_banks;
    }
}

void read_pages_to_host_helper(
    IDevice* device,
    Buffer& dev_buffer,
    uint8_t* host_buffer,
    const uint32_t& page_size,
    const uint32_t& host_page_id,
    const uint32_t& core_page_id,
    const uint32_t& bank_id) {
    uint64_t host_buffer_start = uint64_t(host_page_id) * page_size;
    const auto& cluster = MetalContext::instance().get_cluster();
    size_t aligned_page_size = tt::align(page_size, cluster.get_alignment_requirements(device->id(), page_size));

    if (dev_buffer.is_l1()) {
        auto core_coordinates =
            device->worker_core_from_logical_core(dev_buffer.allocator()->get_logical_core_from_bank_id(bank_id));
        auto bank_offset = device->allocator()->get_bank_offset(dev_buffer.buffer_type(), bank_id);
        auto absolute_address = dev_buffer.address() + bank_offset + (core_page_id * dev_buffer.aligned_page_size());
        if (aligned_page_size > page_size) {
            std::vector<uint8_t> page(aligned_page_size);
            MetalContext::instance().get_cluster().read_core(
                page.data(), aligned_page_size, tt_cxy_pair(device->id(), core_coordinates), absolute_address);
            std::memcpy(host_buffer + host_buffer_start, page.data(), page_size);
        } else {
            MetalContext::instance().get_cluster().read_core(
                host_buffer + host_buffer_start,
                page_size,
                tt_cxy_pair(device->id(), core_coordinates),
                absolute_address);
        }
    } else {
        std::vector<uint8_t> page(aligned_page_size);
        auto bank_local_address = dev_buffer.address() + (core_page_id * dev_buffer.aligned_page_size());
        ReadFromDeviceDRAMChannel(device, bank_id, bank_local_address, std::span<uint8_t>(page));
        std::memcpy(host_buffer + host_buffer_start, page.data(), page_size);
    }
}

void ReadFromDeviceSharded(Buffer& buffer, uint8_t* host_buffer) {
    auto* device = buffer.device();

    uint32_t page_size = buffer.page_size();
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();

    for (auto mapped_page : buffer_page_mapping) {
        auto core = buffer_page_mapping.all_cores[mapped_page.core_id];
        auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        read_pages_to_host_helper(
            device, buffer, host_buffer, page_size, mapped_page.host_page, mapped_page.device_page, bank_id);
    }
}

void ReadFromDevice(Buffer& buffer, uint8_t* host_buffer) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED) {
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        ReadFromDeviceSharded(buffer, host_buffer);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<uint32_t>& host_buffer) {
    ReadFromBuffer(*buffer, host_buffer);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer) {
    IDevice* device = buffer.device();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::TRACE:
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            if (buffer.is_dram()) {
                MetalContext::instance().get_cluster().dram_barrier(device->id());
            } else {
                MetalContext::instance().get_cluster().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Reading from host memory is unsupported!");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id) {
    IDevice* device = buffer.device();
    TT_ASSERT(is_sharded(buffer.buffer_layout()));

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    auto core = buffer_page_mapping.all_cores[core_id];
    auto core_page_mappings = buffer_page_mapping.core_page_mappings[core_id];
    auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];

    if (core_page_mappings.empty()) {
        return;
    }
    size_t shard_offset = core_page_mappings[0].host_ranges[0].host_page_start;

    for (const auto& core_mapping : core_page_mappings) {
        for (auto host_page_it = core_mapping.begin(); host_page_it != core_mapping.end(); host_page_it++) {
            if (!*host_page_it) {
                continue;
            }
            auto host_page_id = **host_page_it - shard_offset;
            auto core_page_id = core_mapping.device_start_page + host_page_it.device_page_offset();
            read_pages_to_host_helper(
                device, buffer, host_buffer, buffer.page_size(), host_page_id, core_page_id, bank_id);
        }
    }
}

void LaunchProgram(
    IDevice* device, const std::shared_ptr<Program>& program, bool wait_until_cores_done, bool force_slow_dispatch) {
    LaunchProgram(device, *program, wait_until_cores_done, force_slow_dispatch);
}

void LaunchProgram(IDevice* device, Program& program, bool wait_until_cores_done, bool force_slow_dispatch) {
    {  // Profiler scope start
        ZoneScoped;
        /// This function is shared between FD and SD.
        // We call this function when initializing HW Command Queues or when reading Profiler Device to Device
        // sync information from the accelerators.
        // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
        if (!force_slow_dispatch) {
            detail::DispatchStateCheck(false);
        } else {
            TT_ASSERT(!MetalContext::instance().device_manager()->is_dispatch_firmware_active());
        }

        detail::CompileProgram(device, program);
        if (!program.impl().is_finalized()) {
            program.impl().finalize_offsets(device);
        }

        detail::WriteRuntimeArgsToDevice(device, program, force_slow_dispatch);
        detail::ConfigureDeviceWithProgram(device, program, force_slow_dispatch);

        auto device_id = device->id();

        MetalContext::instance().get_cluster().dram_barrier(device_id);

        // Note: the l1_barrier below is needed to be sure writes to cores that
        // don't get the GO mailbox (eg, storage cores) have all landed
        MetalContext::instance().get_cluster().l1_barrier(device->id());

        std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
        std::unordered_set<CoreCoord> not_done_cores;
        const auto& hal = MetalContext::instance().hal();
        for (uint32_t programmable_core_type_index = 0;
             programmable_core_type_index < logical_cores_used_in_program.size();
             programmable_core_type_index++) {
            CoreType core_type = hal.get_core_type(programmable_core_type_index);
            for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
                auto* kg = program.impl().kernels_on_core(logical_core, programmable_core_type_index);
                kg->launch_msg.view().kernel_config().host_assigned_id() = program.get_runtime_id();

                auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
                not_done_cores.insert(physical_core);
                if (force_slow_dispatch) {
                    tt::llrt::send_reset_go_signal(device->id(), physical_core);
                }

                tt::llrt::write_launch_msg_to_core(
                    device->id(),
                    physical_core,
                    kg->launch_msg.view(),
                    kg->go_msg.view(),
                    device->get_dev_addr(physical_core, HalL1MemAddrType::LAUNCH));
            }
        }
        if (wait_until_cores_done) {
            // Wait for all cores to be done
            llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_GO, not_done_cores);
        }
    }  // Profiler scope end
    if (wait_until_cores_done) {
        detail::ReadDeviceProfilerResults(device);
    }
}

void WaitProgramDone(IDevice* device, Program& program, bool read_device_profiler_results) {
    auto device_id = device->id();
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    std::unordered_set<CoreCoord> not_done_cores;
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_cores = logical_cores_used_in_program[index];
        CoreType core_type = hal.get_core_type(index);
        for (const auto& logical_core : logical_cores) {
            auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
            not_done_cores.insert(physical_core);
        }
    }
    llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_GO, not_done_cores);
    if (read_device_profiler_results) {
        detail::ReadDeviceProfilerResults(device);
    }
}

bool ConfigureDeviceWithProgram(IDevice* device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    bool pass = true;
    // This function is shared between FD and SD.
    // We call this function when initializing HW Command Queues or when reading Profiler Device to Device
    // sync information from the accelerators.
    // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
    if (!force_slow_dispatch) {
        detail::DispatchStateCheck(false);
    }

    auto device_id = device->id();

    program.impl().allocate_circular_buffers(device);
    program.impl().validate_circular_buffer_region(device);

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_cores = logical_cores_used_in_program[index];
        CoreType core_type = hal.get_core_type(index);
        for (const auto& logical_core : logical_cores) {
            KernelGroup* kernel_group = program.impl().kernels_on_core(logical_core, index);
            CoreCoord physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
            ConfigureKernelGroup(program, index, kernel_group, device, logical_core);
            // TODO: add support for CB for ethernet cores
            if (core_type == CoreType::WORKER) {
                const auto& cbs_on_core = program.impl().circular_buffers_on_core(logical_core);
                if (!cbs_on_core.empty()) {
                    // CircularBufferConfigVec -- common across all kernels, so written once to the core
                    std::vector<uint32_t> circular_buffer_config_vec(
                        program.impl().get_program_config(index).cb_size / sizeof(uint32_t));

                    uint32_t remote_offset_index =
                        program.impl().get_program_config(index).local_cb_size / sizeof(uint32_t);
                    for (const auto& circular_buffer : cbs_on_core) {
                        for (uint32_t buffer_index : circular_buffer->local_buffer_indices()) {
                            uint32_t base_index = buffer_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
                            uint32_t addr_in_bytes = circular_buffer->address();
                            uint32_t size_in_bytes = circular_buffer->size();
                            uint32_t num_pages = circular_buffer->num_pages(buffer_index);
                            uint32_t page_size = size_in_bytes / num_pages;
                            circular_buffer_config_vec[base_index] = addr_in_bytes;      // convert to addr in 16B words
                            circular_buffer_config_vec[base_index + 1] = size_in_bytes;  // convert to addr in 16B words
                            circular_buffer_config_vec[base_index + 2] = num_pages;
                            circular_buffer_config_vec[base_index + 3] = page_size;
                        }
                        for (uint32_t buffer_index : circular_buffer->remote_buffer_indices()) {
                            uint32_t base_index =
                                remote_offset_index + ((NUM_CIRCULAR_BUFFERS - 1 - buffer_index) *
                                                       UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG);
                            uint32_t config_address = circular_buffer->config_address();
                            circular_buffer_config_vec[base_index] = config_address;
                            circular_buffer_config_vec[base_index + 1] = circular_buffer->page_size(buffer_index);
                        }
                    }  // PROF_END("CBS")
                    uint64_t kernel_config_base =
                        hal.get_dev_addr(hal.get_programmable_core_type(index), HalL1MemAddrType::KERNEL_CONFIG);
                    uint64_t addr = kernel_config_base + program.impl().get_program_config(index).cb_offset;
                    MetalContext::instance().get_cluster().write_core(
                        device_id, physical_core, circular_buffer_config_vec, addr);
                }
            }
            program.impl().init_semaphores(*device, logical_core, index);
        }
    }

    return pass;
}

void WriteRuntimeArgsToDevice(IDevice* device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    auto device_id = device->id();
    // This function is shared between FD and SD.
    // We call this function when initializing HW Command Queues or when reading Profiler Device to Device
    // sync information from the accelerators.
    // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
    if (!force_slow_dispatch) {
        detail::DispatchStateCheck(false);
    }

    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        for (const auto& kg : program.impl().get_kernel_groups(index)) {
            auto kernel_config = kg->launch_msg.view().kernel_config();
            uint32_t kernel_config_base = kernel_config.kernel_config_base()[index];
            for (const CoreRange& core_range : kg->core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core(x, y);
                        auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
                        for (auto kernel_id : kg->kernel_ids) {
                            const auto& kernel = program.impl().get_kernel(kernel_id);
                            const auto& rt_args = kernel->runtime_args(logical_core);

                            // RTA/CRTA offsets are the same for all binaries of the kernel, pick any binary.
                            uint32_t processor_index = hal.get_processor_index(
                                kernel->get_kernel_programmable_core_type(),
                                kernel->get_kernel_processor_class(),
                                kernel->get_kernel_processor_type(0));
                            auto rta_offset = kernel_config.rta_offset()[processor_index];
                            if (!rt_args.empty()) {
                                auto rt_args_addr = kernel_config_base + rta_offset.rta_offset();
                                log_trace(
                                    tt::LogMetal,
                                    "{} - Writing {} unique rtargs to core {} (physical: {}) addr 0x{:x} => args: "
                                    "{}",
                                    __FUNCTION__,
                                    rt_args.size(),
                                    logical_core.str(),
                                    physical_core.str(),
                                    rt_args_addr,
                                    rt_args);
                                MetalContext::instance().get_cluster().write_core(
                                    device_id, physical_core, rt_args, rt_args_addr);
                            }

                            const auto& common_rt_args = kernel->common_runtime_args();
                            if (!common_rt_args.empty()) {
                                auto common_rt_args_addr = kernel_config_base + rta_offset.crta_offset();
                                log_trace(
                                    tt::LogMetal,
                                    "{} - Writing {} common rtargs to core {} (physical: {}) addr 0x{:x} => args: "
                                    "{}",
                                    __FUNCTION__,
                                    common_rt_args.size(),
                                    logical_core.str(),
                                    physical_core.str(),
                                    common_rt_args_addr,
                                    common_rt_args);
                                MetalContext::instance().get_cluster().write_core(
                                    device_id, physical_core, common_rt_args, common_rt_args_addr);
                            }
                        }
                    }
                }
            }
        }
    }
}

void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    program.impl().compile(device, force_slow_dispatch);
}

}  // namespace detail

size_t GetNumAvailableDevices() { return MetalContext::instance().get_cluster().number_of_user_devices(); }

bool IsGalaxyCluster() { return MetalContext::instance().get_cluster().is_galaxy_cluster(); }

size_t GetNumPCIeDevices() { return MetalContext::instance().get_cluster().number_of_pci_devices(); }

ChipId GetPCIeDeviceID(ChipId device_id) {
    return MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
}

ClusterType GetClusterType() { return MetalContext::instance().get_cluster().get_cluster_type(); }

std::string SerializeClusterDescriptor() {
    std::filesystem::path path = tt::umd::Cluster::create_cluster_descriptor()->serialize_to_file();
    return path.string();
}

// This function is used to set a default root directory for the tt_metal library.
void SetRootDir(const std::string& root_dir) { tt::llrt::RunTimeOptions::set_root_dir(root_dir); }

IDevice* CreateDevice(
    ChipId device_id,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& l1_bank_remap,
    const size_t worker_l1_size) {
    ZoneScoped;

    // MMIO devices do not support dispatch on galaxy cluster
    // Suggest the user to use the CreateDevices API
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        TT_FATAL(
            !(MetalContext::instance().get_cluster().is_galaxy_cluster() &&
              MetalContext::instance().get_cluster().get_cluster_desc()->is_chip_mmio_capable(device_id)),
            "Galaxy cluster does not support dispatch on mmio devices. Please use CreateDevices API to open all "
            "devices for dispatch.");
    }

    // This API may not be used to create single remote device or multi chip clusters
    // CreateDevices should be used instead to ensure proper init/teardown
    TT_FATAL(
        MetalContext::instance().get_cluster().get_associated_mmio_device(device_id) == device_id,
        "CreateDevice(device_id={}) may only be used for opening single MMIO capable devices. For multi chip clusters, "
        "must use "
        "CreateDevices().",
        device_id);

    MetalContext::instance().initialize_device_manager(
        {device_id}, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_config, l1_bank_remap, worker_l1_size);
    auto* dev = MetalContext::instance().device_manager()->get_active_device(device_id);
    return dev;
}

IDevice* CreateDeviceMinimal(
    ChipId device_id, const uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    ZoneScoped;
    MetalContext::instance().initialize(dispatch_core_config, num_hw_cqs, {}, DEFAULT_L1_SMALL_SIZE, true);
    auto* dev = new Device(device_id, num_hw_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, {}, true);
    MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(IDevice* device) {
    ZoneScoped;
    auto device_id = device->id();

    // This API may not be used to close single remote device or multi chip clusters
    // CloseDevices should be used instead to ensure proper init/teardown
    TT_FATAL(
        MetalContext::instance().get_cluster().get_associated_mmio_device(device_id) == device_id,
        "CloseDevice(device_id={}) may only be used for opening single MMIO capable devices. For multi chip clusters, "
        "must use "
        "CloseDevices().",
        device_id);

    return MetalContext::instance().device_manager()->close_device(device_id);
}

Program CreateProgram() { return Program(); }

KernelHandle CreateDataMovementKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const DataMovementConfig& config) {
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(HalProgrammableCoreType::TENSIX, program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    std::string kernel_name;
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
    auto& control_plane = MetalContext::instance().get_control_plane();
    auto mode = control_plane.get_routing_mode();
    if (mode != ROUTING_MODE_UNDEFINED) {
        kernel->add_defines({{"ROUTING_MODE", std::to_string(static_cast<int>(mode))}});
        auto udm_mode = MetalContext::instance().get_fabric_udm_mode();
        if (udm_mode == tt::tt_fabric::FabricUDMMode::ENABLED) {
            kernel->add_defines({{"UDM_MODE", std::to_string(static_cast<int>(udm_mode))}});
        }
    }
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateComputeKernel(
    Program& program, const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config) {
    std::shared_ptr<Kernel> kernel = std::make_shared<ComputeKernel>(kernel_src, core_range_set, config);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateEthernetKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const EthernetConfig& config) {
    HalProgrammableCoreType eth_core_type =
        config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(eth_core_type, program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    std::shared_ptr<Kernel> kernel = std::make_shared<EthernetKernel>(kernel_src, core_range_set, config);
    auto& control_plane = MetalContext::instance().get_control_plane();
    auto mode = control_plane.get_routing_mode();
    if (mode != ROUTING_MODE_UNDEFINED) {
        kernel->add_defines({{"ROUTING_MODE", std::to_string(static_cast<int>(mode))}});
        auto udm_mode = MetalContext::instance().get_fabric_udm_mode();
        if (udm_mode == tt::tt_fabric::FabricUDMMode::ENABLED) {
            kernel->add_defines({{"UDM_MODE", std::to_string(static_cast<int>(udm_mode))}});
        }
    }

    TT_FATAL(
        ttsl::as_underlying_type<DataMovementProcessor>(config.processor) <
            MetalContext::instance().hal().get_num_risc_processors(eth_core_type),
        "EthernetKernel creation failure: {} kernel cannot target processor {} because Ethernet core only has {} "
        "processors. "
        "Update DataMovementProcessor in the config.",
        kernel->name(),
        enchantum::to_string(config.processor),
        MetalContext::instance().hal().get_num_risc_processors(eth_core_type));
    TT_FATAL(
        !(are_both_riscv_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel->name());
    TT_FATAL(
        !(are_both_noc_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel->name());

    //
    // Valid configurations for Blackhole ERISC
    //
    // |                | Valid NOC Configuration     |                             |
    // |----------------|-----------------------------|-----------------------------|
    // | **ERISC Mode** | **Physical ERISC0**         | **Physical ERISC1**         |
    // | Single         | Not enabled for dispatch    | Dedicated NOC1              |
    // | Dual           | Dedicated NOC0, Dynamic NOC | Dedicated NOC1, Dynamic NOC |
    //
    if (!MetalContext::instance().hal().get_eth_fw_is_cooperative() && config.eth_mode != Eth::IDLE &&
        config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC) {
        bool is_dual_erisc_mode = MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
        bool is_erisc0 = (config.processor == DataMovementProcessor::RISCV_0);
        bool is_erisc1 = (config.processor == DataMovementProcessor::RISCV_1);

        if (is_dual_erisc_mode) {
            // Dual ERISC mode: ERISC0 uses NOC0, ERISC1 uses NOC1 (when in dedicated mode)
            if (is_erisc0) {
                TT_FATAL(
                    config.noc == NOC::NOC_0,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC0 in dedicated mode must use NOC0. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_0. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            } else if (is_erisc1) {
                TT_FATAL(
                    config.noc == NOC::NOC_1,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC1 in dedicated mode must use NOC1. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_1. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            }
        } else {
            // ERISC1 must use NOC1 in dedicated mode
            TT_FATAL(
                config.noc == NOC::NOC_1,
                "EthernetKernel creation failure: In single ERISC mode, ERISC0 must use NOC1. "
                "Kernel: {}, Current NOC: {}, Required NOC: NOC_1.",
                kernel->name(),
                config.noc);
        }
    }

    // Dynamic noc is not supported on single erisc mode
    if (!MetalContext::instance().hal().get_eth_fw_is_cooperative() &&
        !MetalContext::instance().rtoptions().get_enable_2_erisc_mode()) {
        TT_FATAL(
            config.noc_mode == NOC_MODE::DM_DEDICATED_NOC,
            "EthernetKernel creation failure: Dynamic NOC is not supported on single ERISC mode. "
            "Kernel: {}, Current NOC Mode: {}, Required NOC Mode: DM_DEDICATED_NOC.",
            kernel->name(),
            config.noc_mode);
    }

    if (MetalContext::instance().hal().get_eth_fw_is_cooperative()) {
        // Dynamic NOC is not supported with this configuration
        TT_FATAL(
            config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC,
            "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
            "cores because NOC Mode {} is not supported on this platform",
            kernel->name(),
            config.noc_mode);
    }
    return program.impl().add_kernel(kernel, eth_core_type);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {

    // Validate the defines in the config
    std::visit(
        [](const auto& cfg) {
            for (const auto& [key, value] : cfg.defines) {
                if (value.find('\0') != std::string::npos) {
                    throw std::invalid_argument(
                        "Define value for key '" + key + "' contains null character");
                }
            }
        },
        config);

    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    KernelSource kernel_src(file_name, KernelSource::FILE_PATH);
    KernelHandle kernel = std::visit(
        ttsl::overloaded{
            [&](const DataMovementConfig& cfg) {
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            },
            [&](const ComputeConfig& cfg) { return CreateComputeKernel(program, kernel_src, core_ranges, cfg); },
            [&](const EthernetConfig& cfg) { return CreateEthernetKernel(program, kernel_src, core_ranges, cfg); },
        },
        config);

    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateKernel, kernel, program, file_name, core_spec, config);
    return kernel;
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    KernelSource kernel_src(kernel_src_code, KernelSource::SOURCE_CODE);
    return std::visit(
        ttsl::overloaded{
            [&](const DataMovementConfig& cfg) {
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            },
            [&](const ComputeConfig& cfg) { return CreateComputeKernel(program, kernel_src, core_ranges, cfg); },
            [&](const EthernetConfig& cfg) { return CreateEthernetKernel(program, kernel_src, core_ranges, cfg); },
        },
        config);
}

CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    auto cb_handle = program.impl().add_circular_buffer(core_ranges, config);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateCircularBuffer, cb_handle, program, core_spec, config);
    return cb_handle;
}

const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle) {
    return program.impl().get_circular_buffer(cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBufferImpl> circular_buffer = program.impl().get_circular_buffer(cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.impl().invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    program.impl().get_circular_buffer(cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    circular_buffer->config().set_globally_allocated_address_and_total_size(buffer, total_size);
    circular_buffer->assign_global_address();
}

uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    CoreRangeSet crs = std::visit(
        ttsl::overloaded{
            [](const CoreRange& c) { return CoreRangeSet(c); },
            [](const CoreRangeSet& c) {
                // Merge ranges to reduce the number of multicasts needed to initialize semaphores.
                return c.merge_ranges();
            },
        },
        core_spec);
    std::optional<uint32_t> semaphore_id;
    TT_FATAL(!crs.ranges().empty(), "Expecting a non-empty CoreRangeSet!");
    TT_FATAL(
        MetalContext::instance().is_coord_in_range((crs.ranges().back()).end_coord, core_type),
        "Coordinates out of range");
    for (const auto& core_range : crs.ranges()) {
        std::optional<uint32_t> semaphore_id_candidate = get_semaphore_id(program, core_range, core_type);
        if (!semaphore_id.has_value()) {
            semaphore_id = semaphore_id_candidate;
        } else {
            semaphore_id = std::max(semaphore_id.value(), semaphore_id_candidate.value());
        }
    }
    TT_FATAL(semaphore_id.has_value(), "Unable to initialize Semaphore!");

    program.impl().add_semaphore(crs, semaphore_id.value(), initial_value, core_type);

    return semaphore_id.value();
}

GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(device, cores, initial_value, buffer_type);
}

GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(device, std::move(cores), initial_value, buffer_type);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config) {
    return Buffer::create(config.device, config.size, config.page_size, config.buffer_type);
}
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(config.device, address, config.size, config.page_size, config.buffer_type);
}
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device, config.size, config.page_size, config.buffer_type, std::nullopt, std::nullopt, sub_device_id);
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(
        config.device,
        address,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout),
        std::nullopt,
        sub_device_id);
}

void DeallocateBuffer(Buffer& buffer) { buffer.deallocate(); }

void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program) {
    detail::DispatchStateCheck(MetalContext::instance().rtoptions().get_fast_dispatch());
    program.impl().add_buffer(buffer);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    stl::Span<const uint32_t> runtime_args) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32, program, kernel_id, core_spec, runtime_args);
    std::visit([&](auto&& core_spec) { SetRuntimeArgsImpl(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::initializer_list<uint32_t> runtime_args) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32, program, kernel_id, core_spec, runtime_args);
    ZoneScoped;
    std::visit([&](auto&& core_spec) { SetRuntimeArgsImpl(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args) {
    ZoneScoped;
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32VecPerCore, program, kernel, core_spec, runtime_args);
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mistmatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    auto k = program.impl().get_kernel(kernel);
    for (size_t i = 0; i < core_spec.size(); i++) {
        k->set_runtime_args(core_spec[i], runtime_args[i]);
    }
}

void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args) {
    ZoneScoped;
    if (!runtime_args.empty()) {
        program.impl().get_kernel(kernel_id)->set_common_runtime_args(runtime_args);
    }
}

void SetCommonRuntimeArgs(
    const Program& program, KernelHandle kernel_id, std::initializer_list<uint32_t> runtime_args) {
    ZoneScoped;
    if (runtime_args.size() != 0) {
        program.impl().get_kernel(kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData& GetRuntimeArgs(const Program& program, KernelHandle kernel_id, const CoreCoord& logical_core) {
    return program.impl().get_kernel(kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    return program.impl().get_kernel(kernel_id)->runtime_args_data();
}

RuntimeArgsData& GetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    return program.impl().get_kernel(kernel_id)->common_runtime_args_data();
}

namespace experimental::lightmetal {

// This is nop if compile time define not set.
void LightMetalBeginCapture() {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    log_debug(tt::LogMetalTrace, "Begin LightMetalBinary Capture");
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    lm_capture_ctx.reset();            // Clear previous traces if any, ensure tracing disabled
    lm_capture_ctx.set_tracing(true);  // Enable tracing
#else
    log_warning(tt::LogMetalTrace, "TT_ENABLE_LIGHT_METAL_TRACE!=1, ignoring LightMetalBeginCapture()");
#endif
}

// This is nop if compile time define not set, return empty vector.
LightMetalBinary LightMetalEndCapture() {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    log_debug(tt::LogMetalTrace, "End LightMetalBinary Capture");
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    TT_ASSERT(lm_capture_ctx.is_tracing(), "Light Metal Capture was not enabled.");
    lm_capture_ctx.set_tracing(false);  // Disable tracing
    return lm_capture_ctx.create_light_metal_binary();
#else
    log_warning(tt::LogMetalTrace, "TT_ENABLE_LIGHT_METAL_TRACE!=1, ignoring LightMetalEndCapture()");
    return {};
#endif
}

}  // namespace experimental::lightmetal

void PushCurrentCommandQueueIdForThread(uint8_t cq_id) {
    auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    cq_stack.push_back(cq_id);
}

uint8_t PopCurrentCommandQueueIdForThread() {
    auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    TT_FATAL(!cq_stack.empty(), "Current command queue id stack is empty!");
    uint8_t cq_id = cq_stack.back();
    cq_stack.pop_back();
    return cq_id;
}

uint8_t GetCurrentCommandQueueIdForThread() {
    const auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    if (cq_stack.empty()) {
        return 0;
    }
    return cq_stack.back();
}

namespace experimental {

GlobalCircularBuffer CreateGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(device, sender_receiver_core_mapping, size, buffer_type);
}

CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer) {
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    return program.impl().add_circular_buffer(core_ranges, config, global_circular_buffer);
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(circular_buffer->is_global_circular_buffer(), "CircularBuffer must be linked to a GlobalCircularBuffer!");
    circular_buffer->set_global_circular_buffer(global_circular_buffer);
}

}  // namespace experimental

}  // namespace tt::tt_metal
