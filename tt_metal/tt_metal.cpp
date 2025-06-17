// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <circular_buffer.hpp>
#include <circular_buffer_constants.h>
#include "dev_msgs.h"
#include <device_pool.hpp>
#include <global_circular_buffer.hpp>
#include <global_semaphore.hpp>
#include <host_api.hpp>
#include <kernel.hpp>
#include <magic_enum/magic_enum.hpp>
#include <sub_device_types.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "buffer_types.hpp"
#include "circular_buffer_config.hpp"
#include "data_types.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "kernels/kernel_impl.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "device/device_impl.hpp"
#include "hal_types.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include "lightmetal_binary.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/program.hpp"
#include "program/program_impl.hpp"
#include "semaphore.hpp"
#include "trace/trace.hpp"
#include "tracy/Tracy.hpp"
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>
#include "utils.hpp"
#include "fabric/hw/inc/fabric_routing_mode.h"

namespace tt {

namespace tt_metal {
enum class FabricConfig : uint32_t;
struct RuntimeArgsData;
struct TraceDescriptor;

namespace {

CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet>& specified_core_spec) {
    ZoneScoped;
    return std::visit(
        [](auto&& core_spec) -> CoreRangeSet {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRangeSet(CoreRange(core_spec, core_spec));
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                return CoreRangeSet(core_spec);
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                return core_spec;
            }
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

    auto set_global_and_local_noc_usage = [&](KernelHandle kernel_id, bool& local_noc0_usage, bool& local_noc1_usage) {
        const auto kernel = detail::GetKernel(program, kernel_id);
        int noc_value;
        switch (programmable_core) {
            case HalProgrammableCoreType::TENSIX:
                noc_value = magic_enum::enum_integer(std::get<DataMovementConfig>(kernel->config()).noc);
                break;
            case HalProgrammableCoreType::ACTIVE_ETH:
            case HalProgrammableCoreType::IDLE_ETH:
                noc_value = magic_enum::enum_integer(std::get<EthernetConfig>(kernel->config()).noc);
                break;
            default:
                TT_THROW(
                    "Checking NoC and DataMovementProcessor is unsupported for programmable core {}",
                    magic_enum::enum_name(programmable_core));
        }
        local_noc0_usage = noc_value == 0;
        local_noc1_usage = noc_value == 1;
        data_movement_config_status.noc0_in_use = local_noc0_usage;
        data_movement_config_status.noc1_in_use = local_noc1_usage;
    };

    // TODO (abhullar): Clean this up when brisc/ncrisc are moved to be one processor class with two data movement
    // processor types
    uint32_t dm0_idx =
        programmable_core == HalProgrammableCoreType::TENSIX ? DISPATCH_CLASS_TENSIX_DM0 : DISPATCH_CLASS_ETH_DM0;
    uint32_t dm1_idx =
        programmable_core == HalProgrammableCoreType::TENSIX ? DISPATCH_CLASS_TENSIX_DM1 : DISPATCH_CLASS_ETH_DM1;
    const auto& hal = MetalContext::instance().hal();
    for (const auto& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup* kernel_group = program.impl().kernels_on_core(
                    CoreCoord(x, y), hal.get_programmable_core_type_index(programmable_core));
                if (kernel_group != nullptr) {
                    bool local_noc0_in_use = false;
                    bool local_noc1_in_use = false;
                    if (kernel_group->kernel_ids[dm0_idx].has_value()) {
                        data_movement_config_status.riscv0_in_use = true;
                        set_global_and_local_noc_usage(
                            kernel_group->kernel_ids[dm0_idx].value(), local_noc0_in_use, local_noc1_in_use);
                    }
                    if (kernel_group->kernel_ids[dm1_idx].has_value()) {
                        data_movement_config_status.riscv1_in_use = true;
                        set_global_and_local_noc_usage(
                            kernel_group->kernel_ids[dm1_idx].value(), local_noc0_in_use, local_noc1_in_use);
                    }
                    if (kernel_group->kernel_ids[dm0_idx].has_value() and
                        kernel_group->kernel_ids[dm1_idx].has_value()) {
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
    uint32_t kernel_config_base =
        MetalContext::instance().hal().get_dev_addr(programmable_core_type_index, HalL1MemAddrType::KERNEL_CONFIG);
    for (auto& optional_id : kernel_group->kernel_ids) {
        if (optional_id) {
            // Need the individual offsets of each bin
            detail::GetKernel(program, optional_id.value())
                ->configure(device, logical_core, kernel_config_base, kernel_group->kernel_text_offsets);
        }
    }
}

std::optional<uint32_t> get_semaphore_id(const Program &program, const CoreRange& core_range, CoreType core_type) {
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
        semaphore_id = uninitialized_sem_id.value();
    } else {
        TT_THROW("Unable to initialize semaphores on core range {}", core_range.str());
    }

    return semaphore_id;
}

inline void SetRuntimeArgsImpl(
    const Program& program, KernelHandle kernel_id, const CoreCoord& c, stl::Span<const uint32_t> runtime_args) {
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgsImpl(
    const Program& program,
    KernelHandle kernel_id,
    const CoreRange& core_range,
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
    const Program& program,
    KernelHandle kernel_id,
    const CoreRangeSet& core_range_set,
    stl::Span<const uint32_t> runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = detail::GetKernel(program, kernel_id);
        for (const auto& core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

void SetRuntimeArgsImpl(
    const std::shared_ptr<Kernel>& kernel,
    const CoreCoord& core_coord,
    const std::shared_ptr<RuntimeArgs>& runtime_args_ptr,
    bool /*blocking*/) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve(runtime_args_ptr->size());

    for (const auto& arg : *(runtime_args_ptr)) {
        std::visit(
            [&resolved_runtime_args](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, Buffer*>) {
                    resolved_runtime_args.push_back(a->address());
                } else {
                    resolved_runtime_args.push_back(a);
                }
            },
            arg);
    }
    kernel->set_runtime_args(core_coord, resolved_runtime_args);
}

inline void SetRuntimeArgsImpl(
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::shared_ptr<RuntimeArgs>& runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode
    std::visit(
        [&](auto&& core_spec) {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                SetRuntimeArgsImpl(kernel, core_spec, runtime_args, blocking);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                for (auto x = core_spec.start_coord.x; x <= core_spec.end_coord.x; x++) {
                    for (auto y = core_spec.start_coord.y; y <= core_spec.end_coord.y; y++) {
                        SetRuntimeArgsImpl(kernel, CoreCoord(x, y), runtime_args, blocking);
                    }
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto& core_range : core_spec.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            SetRuntimeArgsImpl(kernel, CoreCoord(x, y), runtime_args, blocking);
                        }
                    }
                }
            }
        },
        core_spec);
}

inline void SetRuntimeArgsImpl(
    const std::shared_ptr<Kernel>& kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>>& runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode (support vector of runtime args)
    for (size_t i = 0; i < core_spec.size(); i++) {
        SetRuntimeArgsImpl(kernel, core_spec[i], runtime_args[i], blocking);
    }
}

}  // namespace

namespace detail {

bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer) {
    bool pass = true;
    TT_FATAL(
        address >= device->allocator()->get_base_allocator_addr(HalMemType::DRAM),
        "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!",
        device->allocator()->get_base_allocator_addr(HalMemType::DRAM));
    tt::tt_metal::MetalContext::instance().get_cluster().write_dram_vec(
        host_buffer.data(), host_buffer.size() * sizeof(uint32_t), device->id(), dram_channel, address);
    return pass;
}

bool ReadFromDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer) {
    bool pass = true;
    tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device->id());
    host_buffer.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    tt::tt_metal::MetalContext::instance().get_cluster().read_dram_vec(
        host_buffer.data(), size, device->id(), dram_channel, address);
    return pass;
}

bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    ZoneScoped;
    auto worker_core = device->virtual_core_from_logical_core(logical_core, core_type);
    llrt::write_hex_vec_to_core(device->id(), worker_core, host_buffer, address);
    return true;
}

bool WriteRegToDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, const uint32_t& regval) {
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        &regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    host_buffer = llrt::read_hex_vec_from_core(device->id(), virtual_core, address, size);
    return true;
}

bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval) {
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(
        &regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

void SetFabricConfig(FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(fabric_config, num_routing_planes);
}

std::map<chip_id_t, IDevice*> CreateDevices(
    const std::vector<chip_id_t>& device_ids,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& /*l1_bank_remap*/,
    const size_t worker_l1_size,
    bool init_profiler,
    bool use_max_eth_core_count_on_all_devices,
    bool initialize_fabric_and_dispatch_fw) {
    // Issue #19729: use_max_eth_core_count_on_all_devices is a workaround
    // to allow TT-Mesh Workload dispatch to target active ethernet cores.
    ZoneScoped;
    bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
    tt::DevicePool::initialize(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        init_profiler,
        use_max_eth_core_count_on_all_devices,
        initialize_fabric_and_dispatch_fw);

    const auto devices = tt::DevicePool::instance().get_all_active_devices();
    std::map<chip_id_t, IDevice*> ret_devices;
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

void CloseDevices(const std::map<chip_id_t, IDevice*>& devices) {
    std::vector<IDevice*> devices_to_close;
    for (auto& [id, device] : devices) {
        devices_to_close.push_back(device);
    }
    tt::DevicePool::instance().close_devices(devices_to_close);
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

    auto device = buffer.device();

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    auto total_pages = buffer.num_pages();
    std::vector<uint32_t> page;
    page.resize(page_size / sizeof(uint32_t));
    for (int host_page_id = 0; host_page_id < total_pages; host_page_id++) {
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping[host_page_id];
        auto core = buffer_page_mapping.all_cores[buffer_page_mapping.dev_page_to_core_mapping[dev_page_id]];
        auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto absolute_address = buffer.sharded_page_address(bank_id, dev_page_id);
        auto bank_local_address = buffer.bank_local_page_address(bank_id, dev_page_id);
        auto data_index = host_page_id * page_size;
        std::memcpy(page.data(), host_buffer.data() + data_index, page_size);
        if (buffer.is_l1()) {
            auto core_coordinates =
                device->worker_core_from_logical_core(buffer.allocator()->get_logical_core_from_bank_id(bank_id));
            llrt::write_hex_vec_to_core(device->id(), core_coordinates, page, absolute_address);
        } else {
            WriteToDeviceDRAMChannel(device, bank_id, bank_local_address, page);
        }
    }
}

DeviceAddr CalculateAddressDeviceInterleavedContiguous(const Buffer& buffer, uint32_t bank_index, uint32_t page_index) {
    DeviceAddr addr = 0;
    if (buffer.is_dram()) {
        addr = buffer.bank_local_page_address(bank_index, page_index);
    } else {
        TT_ASSERT(buffer.is_l1());
        addr = buffer.page_address(bank_index, page_index);
    }

    return addr;
}

void WriteToDeviceInterleavedContiguous(const Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size();
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    uint32_t page_size = buffer.page_size();
    uint32_t num_pages = buffer.num_pages();

    auto device = buffer.device();
    auto num_banks = device->allocator()->get_num_banks(buffer.buffer_type());
    uint32_t bank_index = 0;
    int data_index = 0;
    std::vector<uint32_t> page;
    page.resize(page_size / sizeof(uint32_t));
    for (int page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        std::memcpy(page.data(), host_buffer.data() + data_index, page_size);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM: WriteToDeviceDRAMChannel(device, bank_index, address, page); break;
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                CoreCoord logical_core = buffer.allocator()->get_logical_core_from_bank_id(bank_index);
                WriteToDeviceL1(device, logical_core, address, page, CoreType::WORKER);
            } break;
            default: TT_THROW("Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += page_size;
    }
}

void WriteToDevice(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer) {
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
    uint32_t page_size = buffer.page_size();
    uint32_t num_pages = buffer.num_pages();

    auto device = buffer.device();
    auto num_banks = device->allocator()->get_num_banks(buffer.buffer_type());

    size_t host_idx = 0;
    uint32_t bank_index = 0;
    std::vector<uint32_t> page;
    page.resize(page_size / sizeof(uint32_t));
    for (int page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        page.clear();
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::TRACE: ReadFromDeviceDRAMChannel(device, bank_index, address, page_size, page); break;
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto core_coordinates = device->worker_core_from_logical_core(
                    buffer.allocator()->get_logical_core_from_bank_id(bank_index));
                tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    page.data(), page_size, tt_cxy_pair(device->id(), core_coordinates), address);
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
    const uint32_t& dev_page_id,
    const uint32_t& bank_id) {
    auto absolute_address = dev_buffer.sharded_page_address(bank_id, dev_page_id);
    uint32_t host_buffer_start = host_page_id * page_size;
    if (dev_buffer.is_l1()) {
        auto core_coordinates =
            device->worker_core_from_logical_core(dev_buffer.allocator()->get_logical_core_from_bank_id(bank_id));
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            host_buffer + host_buffer_start, page_size, tt_cxy_pair(device->id(), core_coordinates), absolute_address);
    } else {
        std::vector<uint32_t> page;
        page.resize(page_size / sizeof(uint32_t));
        auto bank_local_address = dev_buffer.bank_local_page_address(bank_id, dev_page_id);
        ReadFromDeviceDRAMChannel(device, bank_id, bank_local_address, page_size, page);
        std::memcpy(host_buffer + host_buffer_start, page.data(), page_size);
    }
}

void ReadFromDeviceSharded(Buffer& buffer, uint8_t* host_buffer, bool shard_order) {
    TensorMemoryLayout buffer_layout = buffer.buffer_layout();

    auto device = buffer.device();

    auto total_pages = buffer.num_dev_pages();
    uint32_t page_size = buffer.page_size();

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    for (int dev_page_id = 0; dev_page_id < total_pages; dev_page_id++) {
        auto core = buffer_page_mapping.all_cores[buffer_page_mapping.dev_page_to_core_mapping[dev_page_id]];
        auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto host_page_id = buffer_page_mapping.dev_page_to_host_page_mapping[dev_page_id];
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

void ReadFromDevice(Buffer& buffer, uint8_t* host_buffer, bool shard_order) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED ||
        buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK) {
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        ReadFromDeviceSharded(buffer, host_buffer, shard_order);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<uint32_t>& host_buffer, bool shard_order) {
    ReadFromBuffer(*buffer, host_buffer, shard_order);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer, bool shard_order) {
    IDevice* device = buffer.device();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::TRACE:
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            if (buffer.is_dram()) {
                tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device->id());
            } else {
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer, shard_order);
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

    std::vector<uint32_t> page_ids;
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    for (uint32_t i = 0; i < buffer_page_mapping.dev_page_to_core_mapping.size(); i++) {
        if (buffer_page_mapping.dev_page_to_core_mapping[i] == core_id) {
            page_ids.push_back(i);
        }
    }

    uint32_t host_page_id = 0;
    for (auto dev_page_id : page_ids) {
        auto core = buffer_page_mapping.all_cores[buffer_page_mapping.dev_page_to_core_mapping[dev_page_id]];
        auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        read_pages_to_host_helper(device, buffer, host_buffer, buffer.page_size(), host_page_id, dev_page_id, bank_id);
        host_page_id++;
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
            TT_ASSERT(!tt::DevicePool::instance().is_dispatch_firmware_active());
        }

        detail::CompileProgram(device, program);
        if (!program.is_finalized()) {
            program.finalize_offsets(device);
        }

        detail::WriteRuntimeArgsToDevice(device, program, force_slow_dispatch);
        detail::ConfigureDeviceWithProgram(device, program, force_slow_dispatch);

        auto device_id = device->id();

        tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device_id);

        // Note: the l1_barrier below is needed to be sure writes to cores that
        // don't get the GO mailbox (eg, storage cores) have all landed
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());

        std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
        std::unordered_set<CoreCoord> not_done_cores;
        const auto& hal = MetalContext::instance().hal();
        for (uint32_t programmable_core_type_index = 0;
             programmable_core_type_index < logical_cores_used_in_program.size();
             programmable_core_type_index++) {
            CoreType core_type = hal.get_core_type(programmable_core_type_index);
            for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
                launch_msg_t* msg =
                    &program.impl().kernels_on_core(logical_core, programmable_core_type_index)->launch_msg;
                go_msg_t* go_msg = &program.impl().kernels_on_core(logical_core, programmable_core_type_index)->go_msg;
                msg->kernel_config.host_assigned_id = program.get_runtime_id();

                auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
                not_done_cores.insert(physical_core);
                if (force_slow_dispatch) {
                    tt::llrt::send_reset_go_signal(device->id(), physical_core);
                }

                tt::llrt::write_launch_msg_to_core(
                    device->id(),
                    physical_core,
                    msg,
                    go_msg,
                    device->get_dev_addr(physical_core, HalL1MemAddrType::LAUNCH));
            }
        }
        if (wait_until_cores_done) {
            // Wait for all cores to be done
            llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
        }
    }  // Profiler scope end
    if (wait_until_cores_done) {
        detail::DumpDeviceProfileResults(device);
    }
}

void WaitProgramDone(IDevice* device, Program& program, bool dump_device_profile_results) {
    auto device_id = device->id();
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
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
    llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
    if (dump_device_profile_results) {
        detail::DumpDeviceProfileResults(device);
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

    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
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
                if (cbs_on_core.size()) {
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
                                remote_offset_index + (NUM_CIRCULAR_BUFFERS - 1 - buffer_index) *
                                                          UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
                            uint32_t config_address = circular_buffer->config_address();
                            circular_buffer_config_vec[base_index] = config_address;
                            circular_buffer_config_vec[base_index + 1] = circular_buffer->page_size(buffer_index);
                        }
                    }  // PROF_END("CBS")
                    uint64_t kernel_config_base = hal.get_dev_addr(index, HalL1MemAddrType::KERNEL_CONFIG);
                    uint64_t addr = kernel_config_base + program.impl().get_program_config(index).cb_offset;
                    llrt::write_hex_vec_to_core(device_id, physical_core, circular_buffer_config_vec, addr);
                }
            }
            program.init_semaphores(*device, logical_core, index);
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
        uint32_t processor_classes = hal.get_processor_classes_count(index);
        for (const auto& kg : program.impl().get_kernel_groups(index)) {
            uint32_t kernel_config_base = kg->launch_msg.kernel_config.kernel_config_base[index];
            for (const CoreRange& core_range : kg->core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core(x, y);
                        auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
                        for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
                            auto& optional_id = kg->kernel_ids[dispatch_class];
                            if (optional_id) {
                                const auto& kernel = detail::GetKernel(program, optional_id.value());
                                const auto& rt_args = kernel->runtime_args(logical_core);

                                if (rt_args.size() > 0) {
                                    auto rt_args_addr =
                                        kernel_config_base +
                                        kg->launch_msg.kernel_config.rta_offset[dispatch_class].rta_offset;
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
                                    tt::llrt::write_hex_vec_to_core(device_id, physical_core, rt_args, rt_args_addr);
                                }

                                const auto& common_rt_args = kernel->common_runtime_args();
                                if (common_rt_args.size() > 0) {
                                    auto common_rt_args_addr =
                                        kernel_config_base +
                                        kg->launch_msg.kernel_config.rta_offset[dispatch_class].crta_offset;
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
                                    tt::llrt::write_hex_vec_to_core(
                                        device_id, physical_core, common_rt_args, common_rt_args_addr);
                                }
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
    program.compile(device, force_slow_dispatch);
}

}  // namespace detail

size_t GetNumAvailableDevices() {
    return tt::tt_metal::MetalContext::instance().get_cluster().number_of_user_devices();
}

bool IsGalaxyCluster() { return tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster(); }

size_t GetNumPCIeDevices() { return tt::tt_metal::MetalContext::instance().get_cluster().number_of_pci_devices(); }

chip_id_t GetPCIeDeviceID(chip_id_t device_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
}

IDevice* CreateDevice(
    chip_id_t device_id,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& l1_bank_remap,
    const size_t worker_l1_size) {
    ZoneScoped;

    tt::DevicePool::initialize(
        {device_id}, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_config, l1_bank_remap, worker_l1_size);
    auto dev = tt::DevicePool::instance().get_active_device(device_id);
    return dev;
}

IDevice* CreateDeviceMinimal(
    chip_id_t device_id, const uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    ZoneScoped;
    tt::tt_metal::MetalContext::instance().initialize(dispatch_core_config, num_hw_cqs, {});
    auto dev = new Device(device_id, num_hw_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, {}, true);
    tt::tt_metal::MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(IDevice* device) {
    ZoneScoped;
    auto device_id = device->id();
    return tt::DevicePool::instance().close_device(device_id);
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
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mode = control_plane.get_routing_mode();
    if (mode != ROUTING_MODE_UNDEFINED) {
        kernel->add_defines({{"ROUTING_MODE", std::to_string(static_cast<int>(mode))}});
    }
    return detail::AddKernel(program, kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateComputeKernel(
    Program& program, const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config) {
    std::shared_ptr<Kernel> kernel = std::make_shared<ComputeKernel>(kernel_src, core_range_set, config);
    return detail::AddKernel(program, kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateEthernetKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const EthernetConfig& config) {
    KernelHandle kernel_handle;
    HalProgrammableCoreType eth_core_type =
        config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(eth_core_type, program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    std::shared_ptr<Kernel> kernel = std::make_shared<EthernetKernel>(kernel_src, core_range_set, config);
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mode = control_plane.get_routing_mode();
    if (mode != ROUTING_MODE_UNDEFINED) {
        kernel->add_defines({{"ROUTING_MODE", std::to_string(static_cast<int>(mode))}});
    }

    TT_FATAL(
        utils::underlying_type<DataMovementProcessor>(config.processor) <
            MetalContext::instance().hal().get_processor_classes_count(eth_core_type),
        "EthernetKernel creation failure: {} kernel cannot target processor {} because Ethernet core only has {} "
        "processors. "
        "Update DataMovementProcessor in the config.",
        kernel->name(),
        magic_enum::enum_name(config.processor),
        MetalContext::instance().hal().get_processor_classes_count(eth_core_type));
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

    return detail::AddKernel(program, kernel, eth_core_type);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    KernelHandle kernel = std::visit(
        [&](auto&& cfg) -> KernelHandle {
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

    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateKernel, kernel, program, file_name, core_spec, config);
    return kernel;
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    return std::visit(
        [&](auto&& cfg) -> KernelHandle {
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
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = GetCoreRangeSet(core_spec);
    auto cb_handle = program.add_circular_buffer(core_ranges, config);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateCircularBuffer, cb_handle, program, core_spec, config);
    return cb_handle;
}

const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle) {
    return detail::GetCircularBuffer(program, cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBuffer> circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    detail::GetCircularBuffer(program, cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer) {
    auto circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size) {
    auto circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    circular_buffer->config().set_globally_allocated_address_and_total_size(buffer, total_size);
    circular_buffer->assign_global_address();
}

uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return std::visit(
        [&](auto&& c) -> uint32_t {
            using T = std::decay_t<decltype(c)>;
            CoreRangeSet crs;
            if constexpr (std::is_same_v<T, CoreRange>) {
                crs = CoreRangeSet(c);
            } else {
                // Merge ranges to reduce the number of multicasts needed to initialize semaphores.
                crs = c.merge_ranges();
            }
            std::optional<uint32_t> semaphore_id;
            TT_FATAL(crs.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
            for (const auto& core_range : crs.ranges()) {
                std::optional<uint32_t> semaphore_id_candidate = get_semaphore_id(program, core_range, core_type);
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

GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(device, cores, initial_value, buffer_type);
}

GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(device, std::move(cores), initial_value, buffer_type);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(
        config.device,
        address,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        std::nullopt,
        std::nullopt);
}
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        std::nullopt,
        std::nullopt,
        sub_device_id);
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters,
        std::nullopt,
        std::nullopt);
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(
        config.device,
        address,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters,
        std::nullopt,
        std::nullopt);
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters,
        std::nullopt,
        sub_device_id);
}

void DeallocateBuffer(Buffer& buffer) { buffer.deallocate(); }

void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program) {
    detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
    program.add_buffer(buffer);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    stl::Span<const uint32_t> runtime_args) {
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
    auto k = detail::GetKernel(program, kernel);
    for (size_t i = 0; i < core_spec.size(); i++) {
        k->set_runtime_args(core_spec[i], runtime_args[i]);
    }
}

void SetRuntimeArgs(
    IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::shared_ptr<RuntimeArgs>& runtime_args) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgs, device, kernel, core_spec, runtime_args);
    SetRuntimeArgsImpl(kernel, core_spec, std::move(runtime_args), false);
}

void SetRuntimeArgs(
    IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>>& runtime_args) {
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mismatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgsImpl(kernel, core_spec, runtime_args, false);
}

void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args) {
    ZoneScoped;
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData& GetRuntimeArgs(const Program& program, KernelHandle kernel_id, const CoreCoord& logical_core) {
    return detail::GetKernel(program, kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    return detail::GetKernel(program, kernel_id)->runtime_args_data();
}

RuntimeArgsData& GetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    return detail::GetKernel(program, kernel_id)->common_runtime_args_data();
}

uint32_t BeginTraceCapture(IDevice* device, const uint8_t cq_id) {
    const uint32_t tid = Trace::next_id();
    device->begin_trace(cq_id, tid);
    return tid;
}

void EndTraceCapture(IDevice* device, const uint8_t cq_id, const uint32_t tid) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    device->end_trace(cq_id, tid);
    // When light metal tracing is enabled, TraceDescriptor will be serialized via end_trace() and this
    // will serialize the LightMetalLoadTraceId call to be used during replay to load trace back to device.
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureLoadTrace, device, cq_id, tid);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureReplayTrace, device, cq_id, tid, true);  // blocking=true
}

void ReplayTrace(IDevice* device, const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureReplayTrace, device, cq_id, tid, blocking);
    device->replay_trace(cq_id, tid, blocking /* block_on_device */, blocking /* block_on_worker_thread */);
}

void ReleaseTrace(IDevice* device, const uint32_t tid) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureReleaseTrace, device, tid);
    device->release_trace(tid);
}

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

void LoadTrace(IDevice* device, const uint8_t cq_id, const uint32_t trace_id, const TraceDescriptor& trace_desc) {
    device->load_trace(cq_id, trace_id, trace_desc);
}

void Synchronize(IDevice* device, const std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        if (cq_id.has_value()) {
            Finish(device->command_queue(cq_id.value()), sub_device_ids);
        } else {
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
                Finish(device->command_queue(cq_id), sub_device_ids);
            }
        }
    }
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
    return program.add_circular_buffer(core_ranges, config, global_circular_buffer);
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer) {
    auto circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    TT_FATAL(circular_buffer->is_global_circular_buffer(), "CircularBuffer must be linked to a GlobalCircularBuffer!");
    circular_buffer->set_global_circular_buffer(global_circular_buffer);
}

}  // namespace experimental

}  // namespace tt_metal

}  // namespace tt
