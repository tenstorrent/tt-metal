// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <string>

#include "impl/allocator/allocator.hpp"
#include "impl/dispatch/command_queue.hpp"
#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"
#include "dev_msgs.h"

#include "tools/profiler/profiler.hpp"
#include "tools/cpuprof/cpuprof.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/program.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt {

namespace tt_metal {

namespace {

void ConfigureKernelGroup(const Program &program, const KernelGroup *kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group->compute_id.has_value()) {
        detail::GetKernel(program, kernel_group->compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group->riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group->riscv1_id.value())->configure(device, logical_core);
    }
    if (kernel_group->riscv0_id.has_value()) {
        detail::GetKernel(program, kernel_group->riscv0_id.value())->configure(device, logical_core);
    }
    if (kernel_group->erisc_id.has_value()) {
        detail::GetKernel(program, kernel_group->erisc_id.value())->configure(device, logical_core);
    }
}

std::optional<uint32_t> get_semaphore_address(const Program &program, const CoreRange &core_range) {
    std::optional<uint32_t> address;
    auto start_core = core_range.start;
    auto end_core = core_range.end;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = CoreCoord{x, y};
            auto num_semaphores = program.num_semaphores(logical_core);
            if (num_semaphores == NUM_SEMAPHORES) {
                TT_THROW(
                    "Cannot add semaphore on core " + logical_core.str() + ". Max number of semaphores (" +
                    std::to_string(NUM_SEMAPHORES) + ") reached!");
            }
            uint32_t addr = num_semaphores == 0
                                ? SEMAPHORE_BASE
                                : program.semaphore_address(num_semaphores - 1) + L1_ALIGNMENT;
            if (!address.has_value()) {
                address = addr;
            } else if (addr != address) {
                TT_THROW(
                    "Expected semaphore on logical core " + logical_core.str() + " to be initialized at L1 address " +
                    std::to_string(address.value()) + " but it is at " + std::to_string(addr));
            }
        }
    }
    return address;
}


inline void SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &c, const std::vector<uint32_t> &runtime_args)
{
    detail::GetKernel(program, kernel_id)->set_runtime_args(c, runtime_args);
}


inline void SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args)
{
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            SetRuntimeArgs(program, kernel_id, CoreCoord(x,y), runtime_args);
        }
    }
}

inline void SetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, std::shared_ptr<RuntimeArgs> runtime_args, bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode
    std::visit([&](auto&& core_spec) {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                EnqueueSetRuntimeArgs(cq, kernel, core_spec, runtime_args, blocking);
            }
            else if constexpr (std::is_same_v<T, CoreRange>) {
                for (auto x = core_spec.start.x; x <= core_spec.end.x; x++) {
                    for (auto y = core_spec.start.y; y <= core_spec.end.y; y++) {
                        EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                    }
                }
            }
            else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto& core_range : core_spec.ranges()) {
                    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                                EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                        }
                    }
                }
            }
        },
        core_spec
    );
}

inline void SetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const std::vector< CoreCoord > & core_spec, const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args, bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode (support vector of runtime args)
    for (size_t i = 0; i < core_spec.size(); i++) {
        EnqueueSetRuntimeArgs(cq, kernel, core_spec[i], runtime_args[i], blocking);
    }
}

}  // namespace

//#define DEBUG_PRINT_SHARD

namespace detail {

std::map<chip_id_t, Device *> CreateDevices(
    std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, const std::vector<uint32_t> &l1_bank_remap) {
    std::map<chip_id_t, Device *> active_devices;  // TODO: pass this to CloseDevices
    for (const auto &device_id : device_ids) {
        const auto &mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        if (active_devices.find(mmio_device_id) == active_devices.end()) {
            for (const auto &mmio_controlled_device_id :
                 tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
                Device * dev = new Device(mmio_controlled_device_id, num_hw_cqs, l1_bank_remap);
                active_devices.insert({mmio_controlled_device_id, dev});
            }
        }
    }
    // TODO: need to only enable routing for used mmio chips
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    return active_devices;
}

void CloseDevices(std::map<chip_id_t, Device *> devices) {
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    for (const auto &[device_id, dev] : devices) {
        dev->close();
    }
}

    void print_page(uint32_t dev_page_id, CoreCoord core, uint32_t host_page_id, CoreCoord noc_coordinates, uint32_t l1_address, uint32_t bank_id, std::vector<uint32_t> page){
        std::cout << "dev_page_index " << dev_page_id << " on core " << core.str() << std::endl;
        std::cout << "host_page_index " << host_page_id << std::endl;
        std::cout << "noc coordinates " << noc_coordinates.str() << std::endl;
        std::cout << "l1_address " << l1_address << std::endl;
        std::cout << "bank id " << bank_id << std::endl;

        std::cout << "0x";
        for(auto entry: page){
            std::cout << std::hex << entry << std::dec;
        }
        std::cout << std::dec << std::endl;
    }

    void WriteToDeviceSharded(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
        uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
        TT_ASSERT(
            host_buffer_size_bytes <= buffer.size(),
            "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", host_buffer_size_bytes, buffer.size());

        uint32_t page_size = buffer.page_size();
        TT_ASSERT(buffer.size() % page_size == 0);

        static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
        TT_ASSERT(page_size % bytes_per_page_entry == 0);
        uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

        auto device = buffer.device();

        TT_ASSERT(buffer.buffer_type() == BufferType::L1 && "Only L1 Buffers support sharding");

        auto total_pages = buffer.num_pages();
        for(int host_page_id=0; host_page_id<total_pages; host_page_id++){
            auto dev_page_id = buffer.get_host_to_dev_mapped_page_id(host_page_id);
            auto core = buffer.get_core_from_dev_page_id(dev_page_id);
            auto bank_id = device->bank_ids_from_logical_core(core)[0];
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
            "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", host_buffer_size_bytes, buffer.size());

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
                case BufferType::DRAM: {
                    auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                    tt::Cluster::instance().write_dram_vec(page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address);
                } break;
                case BufferType::L1: {
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
        detail::ProfileTTMetalScope profile_this =
            detail::ProfileTTMetalScope(std::string("WriteToDevice ") + std::to_string(buffer.device()->id()));
        if(buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED || buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK){
            WriteToDeviceInterleavedContiguous(buffer, host_buffer);
        }
        else if(is_sharded(buffer.buffer_layout())){
            WriteToDeviceSharded(buffer, host_buffer);
        }
        else{
            TT_ASSERT(false && "Unsupported buffer layout");
        }
    }

    void WriteToBuffer( std::shared_ptr<const Buffer> buffer, const std::vector<uint32_t> &host_buffer){
        WriteToBuffer ( *buffer, host_buffer );
    }

    void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1: {
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
                case BufferType::DRAM: {
                    auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                    tt::Cluster::instance().read_dram_vec(page, page_size, tt_target_dram{device->id(), dram_channel, 0}, absolute_address);
                } break;
                case BufferType::L1: {
                    auto noc_coordinates = buffer.noc_coordinates(bank_index);
                    page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
                } break;
                default: TT_FATAL(false && "Unsupported buffer type to write to device!");
            }

            // Copy page into host buffer
            for (uint32_t entry : page) {
                host_buffer.push_back(entry);
            }

            bank_index = (bank_index + 1) % num_banks;
        }

    }

    void read_pages_to_host_helper(
                                  Device * device,
                                  const Buffer & dev_buffer,
                                  std::vector<uint32_t> &host_buffer,
                                  const uint32_t & page_size,
                                  const uint32_t & host_page_id,
                                  const uint32_t & dev_page_id,
                                  const uint32_t & bank_id
    ){
        auto absolute_address = dev_buffer.sharded_page_address(bank_id, dev_page_id);
        auto noc_coordinates = dev_buffer.noc_coordinates(bank_id);

        uint32_t num_entries_per_page = page_size/sizeof(uint32_t);
        auto page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
        uint32_t host_buffer_start = host_page_id * num_entries_per_page;
        uint32_t dev_page_index = 0;
        for(uint32_t host_buffer_index = host_buffer_start; host_buffer_index < host_buffer_start + num_entries_per_page; host_buffer_index++){
            host_buffer[host_buffer_index] = page[dev_page_index];
            dev_page_index++;
        }

    }

    void ReadFromDeviceSharded(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order){

        TensorMemoryLayout buffer_layout = buffer.buffer_layout();

        auto device = buffer.device();
        #ifdef DEBUG_PRINT_SHARD
            std::cout << "Reading From Device Height Sharded " << std::endl;
        #endif


        int output_page_index = 0;
        auto total_pages = buffer.num_pages();
        uint32_t page_size = buffer.page_size();
        uint32_t bytes_per_page_entry = sizeof(uint32_t);
        uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

        host_buffer = std::vector<uint32_t>(total_pages * num_entries_per_page);

        for(int dev_page_id=0; dev_page_id<total_pages; dev_page_id++){
            auto core = buffer.get_core_from_dev_page_id(dev_page_id);
            auto bank_id = device->bank_ids_from_logical_core(core)[0];
            auto host_page_id = buffer.get_dev_to_host_mapped_page_id(dev_page_id);
            if(!shard_order){
                read_pages_to_host_helper(
                    device,
                    buffer,
                    host_buffer,
                    page_size,
                    host_page_id,
                    dev_page_id,
                    bank_id
                );
            }
            else{
                read_pages_to_host_helper(
                    device,
                    buffer,
                    host_buffer,
                    page_size,
                    dev_page_id,
                    dev_page_id,
                    bank_id
                );
            }
        }

    }


    void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer,  bool shard_order) {
        ZoneScoped;
        detail::ProfileTTMetalScope profile_this =
            detail::ProfileTTMetalScope(std::string("ReadFromDevice ") + std::to_string(buffer.device()->id()));

        host_buffer.clear();  // overwrite the data
        if(buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED
            || buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK){
            ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
        }
        else if(is_sharded(buffer.buffer_layout())){
            TT_ASSERT(buffer.buffer_type() == BufferType::L1 && "Only L1 Buffers support sharding");
            ReadFromDeviceSharded(buffer, host_buffer, shard_order);
        }
        else{
            TT_ASSERT(false && "Unsupported buffer layout");
        }
    }

    void ReadFromBuffer(std::shared_ptr<const Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order)
    {
        ReadFromBuffer(*buffer, host_buffer, shard_order);
    }

    void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
        Device *device = buffer.device();
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1: {
                if (buffer.buffer_type() == BufferType::DRAM) {
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

    void ReadShard(const Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t & core_id) {

        Device *device = buffer.device();
        TT_ASSERT(is_sharded(buffer.buffer_layout()));
        host_buffer.clear();  // overwrite the data

        uint32_t num_entries_per_page = buffer.page_size() / sizeof(uint32_t);
        uint32_t num_entries_per_shard = num_entries_per_page * buffer.shard_spec().size();
        host_buffer = std::vector<uint32_t>(num_entries_per_shard);

        auto page_ids = buffer.dev_pages_in_shard(core_id);

        uint32_t host_page_id = 0;
        for(auto dev_page_id: page_ids){
            auto core = buffer.get_core_from_dev_page_id(dev_page_id);
            auto bank_id = device->bank_ids_from_logical_core(core)[0];
            read_pages_to_host_helper(
                device,
                buffer,
                host_buffer,
                buffer.page_size(),
                host_page_id,
                dev_page_id,
                bank_id
            );
            host_page_id++;

        }


    }

    void LaunchProgram(Device *device, std::shared_ptr<Program> program){
        LaunchProgram(device, *program);
    }

    void LaunchProgram(Device *device, Program &program) {
        {//Profiler scope start
        ZoneScoped;
        detail::DispatchStateCheck( false );
        detail::ProfileTTMetalScope profile_this =
            detail::ProfileTTMetalScope(std::string("LaunchProgram ") + std::to_string(device->id()));
        detail::CompileProgram(device, program);
        detail::WriteRuntimeArgsToDevice(device, program);
        detail::ConfigureDeviceWithProgram(device, program);
        auto device_id = device->id();

        tt::Cluster::instance().dram_barrier(device_id);

        // Note: the l1_barrier below is needed to be sure writes to cores that
        // don't get the GO mailbox (eg, storage cores) have all landed
        tt::Cluster::instance().l1_barrier(device->id());

        std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
        std::unordered_set<CoreCoord> not_done_cores;
        for (const auto &[core_type, logical_cores] : logical_cores_used_in_program) {
            for (const auto &logical_core : logical_cores) {
                launch_msg_t *msg = &program.kernels_on_core(logical_core, core_type)->launch_msg;
                auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
                not_done_cores.insert(physical_core);
                tt::llrt::write_launch_msg_to_core(device->id(), physical_core, msg);
            }
        }
        // Wait for all cores to be done
        llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);

        }//Profiler scope end
        DumpDeviceProfileResults(device, program);
    }

    bool ConfigureDeviceWithProgram(Device *device, Program &program, bool fd_bootloader_mode) {
        ZoneScoped;
        bool pass = true;
        // This is function is shared between FD and SD.
        // We call this function when initializing HW Command Queues (tracked as fd_bootloader_mode) for Fast Dispatch.
        // Used to Launch programs for Slow dispatch.
        bool using_fast_dispatch = fd_bootloader_mode;
        detail::DispatchStateCheck( using_fast_dispatch );
        detail::ProfileTTMetalScope profile_this =
            detail::ProfileTTMetalScope(std::string("ConfigureDeviceWithProgram ") + std::to_string(device->id()));

        auto device_id = device->id();

        program.allocate_circular_buffers();
        detail::ValidateCircularBufferRegion(program, device);

        std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
        for (const auto &[core_type, logical_cores] : logical_cores_used_in_program) {
            for (const auto &logical_core : logical_cores) {
                KernelGroup *kernel_group = program.kernels_on_core(logical_core, core_type);
                CoreCoord physical_core = device->physical_core_from_logical_core(logical_core, core_type);

                ConfigureKernelGroup(
                    program, kernel_group, device, logical_core);  // PROF_BEGIN("CONF_KERN") PROF_END("CONF_KERN")
                // TODO: add support for CB for ethernet cores
                if (core_type == CoreType::WORKER) {
                    // CircularBufferConfigVec -- common across all kernels, so written once to the core
                    llrt::CircularBufferConfigVec circular_buffer_config_vec =
                        llrt::create_circular_buffer_config_vector();

                    auto cbs_on_core = program.circular_buffers_on_core(logical_core);  // PROF_BEGIN("CBS")
                    for (auto circular_buffer : cbs_on_core) {
                        for (uint32_t buffer_index : circular_buffer->buffer_indices()) {
                            llrt::set_config_for_circular_buffer(
                                circular_buffer_config_vec,
                                buffer_index,
                                circular_buffer->address(),
                                circular_buffer->size(),
                                circular_buffer->num_pages(buffer_index));
                        }
                    }  // PROF_END("CBS")

                    if (cbs_on_core.size()) {
                        llrt::write_circular_buffer_config_vector_to_core(
                            device_id,
                            physical_core,
                            circular_buffer_config_vec);  // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")
                    }

                    program.init_semaphores(*device, logical_core);
                }
            }
        }

        return pass;
    }

    void WriteRuntimeArgsToDevice(Device *device, const Program &program) {
        ZoneScoped;
        auto device_id = device->id();
        detail::DispatchStateCheck( false );

        auto get_l1_arg_base_addr = [](const RISCV &riscv) {
            uint32_t l1_arg_base = 0;
            switch (riscv) {
                case RISCV::BRISC: {
                    l1_arg_base = BRISC_L1_ARG_BASE;
                } break;
                case RISCV::NCRISC: {
                    l1_arg_base = NCRISC_L1_ARG_BASE;
                } break;
                case RISCV::ERISC: {
                    l1_arg_base = eth_l1_mem::address_map::ERISC_L1_ARG_BASE;
                } break;
                case RISCV::COMPUTE: {
                    l1_arg_base = TRISC_L1_ARG_BASE;
                }
                break;
                default: TT_THROW("Unsupported {} processor does not support runtime args", riscv);
            }
            return l1_arg_base;
        };

        for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
            const auto kernel = detail::GetKernel(program, kernel_id);
            auto processor = kernel->processor();
            for (const auto &logical_core : kernel->cores_with_runtime_args()) {
                auto physical_core = device->physical_core_from_logical_core(logical_core, kernel->get_kernel_core_type());
                const auto & rt_args = kernel->runtime_args(logical_core);
                tt::llrt::write_hex_vec_to_core(device_id, physical_core, rt_args, get_l1_arg_base_addr(processor));
            }
        }
    }

    void CompileProgram(Device *device, Program &program){
        ZoneScoped;
        program.compile(device);
    }

    void AllocateBuffer(Buffer* buffer, bool bottom_up) {
        detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
        EnqueueAllocateBuffer(buffer->device()->command_queue(), buffer, bottom_up, false);
    }

    void DeallocateBuffer(Buffer *buffer) {
        detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
        EnqueueDeallocateBuffer(buffer->device()->command_queue(), *(buffer->device()->allocator_), buffer->address(), buffer->buffer_type(), false);
    }

    void GetBufferAddress(const Buffer* buffer, uint32_t* address_on_host) {
        detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
        EnqueueGetBufferAddr(buffer->device()->command_queue(), address_on_host, buffer, false);
    }

}   // namespace detail

size_t GetNumAvailableDevices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_devices();
#else
    return 1;
#endif
}

size_t GetNumPCIeDevices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_pci_devices();
#else
    return 1;
#endif
}

Device *CreateDevice(chip_id_t device_id, const uint8_t num_hw_cqs, const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    Device * dev = new Device(device_id, num_hw_cqs, l1_bank_remap);
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(Device *device) {
    ZoneScoped;
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    return device->close();
}

Program CreateProgram(){
    return Program();
}

KernelHandle CreateKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return std::visit( [&](auto&& cfg) -> KernelHandle
                        {
                            CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
                            std::shared_ptr<Kernel> kernel;
                            using T = std::decay_t<decltype(cfg)>;
                            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                                detail::CheckDataMovementConfig(program, file_name, core_ranges);
                                kernel = std::make_shared<DataMovementKernel>(file_name, core_ranges, cfg);
                                return detail::AddKernel(program, kernel, CoreType::WORKER);
                            }
                            else if constexpr (std::is_same_v<T, ComputeConfig>) {
                                kernel = std::make_shared<ComputeKernel>(file_name, core_ranges, cfg);
                                return detail::AddKernel(program, kernel, CoreType::WORKER);
                            } else if constexpr (std::is_same_v<T, EthernetConfig>) {
                                kernel = std::make_shared<EthernetKernel>(file_name, core_ranges, cfg);
                                return detail::AddKernel(program, kernel, CoreType::ETH);
                            }
                        },
                        config
                    );
}

CBHandle CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
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
    if (buffer.buffer_type() != BufferType::L1) {
        TT_FATAL("Only L1 buffers can have an associated circular buffer!");
    }
    detail::GetCircularBuffer(program, cb_handle)->config().set_globally_allocated_address(buffer);
    detail::GetCircularBuffer(program, cb_handle)->assign_global_address();
}

uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value) {
    return std::visit(
        [&](auto&& c) -> uint32_t
        {
            using T = std::decay_t<decltype(c)>;
            CoreRangeSet crs({});
            if constexpr (std::is_same_v<T, CoreRange>) {
                crs = CoreRangeSet({c});
            } else{
                crs = c;
            }
            std::optional<uint32_t> address;
            TT_FATAL(crs.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
            for (const auto& core_range : crs.ranges()) {
                CoreCoord start_core = core_range.start;
                CoreCoord end_core = core_range.end;
                TT_FATAL(start_core == end_core or start_core < end_core && "Invalid core range!");
                auto addr = get_semaphore_address(program, core_range);
                if (!address.has_value()) {
                    address = addr;
                } else {
                    TT_ASSERT(addr == address);
                }
            }
            TT_FATAL(address.has_value(), "Expecting a valid Semaphore address!");

            program.add_semaphore(crs, address.value(), initial_value);

            return address.value();
        },
        core_spec);
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

void AssignGlobalBufferToProgram(std::shared_ptr<Buffer> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
    EnqueueAddBufferToProgram(buffer-> device()->command_queue(), buffer, program, false);
}

void SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &core_spec, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    TT_FATAL( not CommandQueue::async_mode_set(), "This variant of SetRuntimeArgs can only be called when Asyncrhonous SW Command Queues are disabled for Fast Dispatch.");
    std::visit(
        [&](auto&& core_spec)
        {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreRange> || std::is_same_v<T, CoreCoord> ) {
                SetRuntimeArgs(program, kernel_id, core_spec, runtime_args);
            }
            else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto& core_range : core_spec.ranges()) {
                    SetRuntimeArgs(program, kernel_id, core_range, runtime_args);
                }
            }
        },
        core_spec
    );
}

void SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)
{
    ZoneScoped;
    TT_FATAL( not CommandQueue::async_mode_set(), "This variant of SetRuntimeArgs can only be called when Asyncrhonous SW Command Queues are disabled for Fast Dispatch.");
    TT_FATAL( core_spec.size() == runtime_args.size(), "Mistmatch between number of cores {} and number of runtime args {} getting updated", core_spec.size(), runtime_args.size());
    auto k = detail::GetKernel(program, kernel);
    for (size_t i = 0; i < core_spec.size(); i++)
        k->set_runtime_args(core_spec[i], runtime_args[i]);
}

void SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::variant<CoreCoord, CoreRange,CoreRangeSet> &core_spec, std::shared_ptr<RuntimeArgs> runtime_args) {
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::vector< CoreCoord > & core_spec, const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args) {
    TT_FATAL( core_spec.size() == runtime_args.size(), "Mistmatch between number of cores {} and number of runtime args {} getting updated", core_spec.size(), runtime_args.size());
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
}

std::vector<uint32_t> & GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core) {
    TT_FATAL( not CommandQueue::async_mode_set(), "GetRuntimeArgs can only be called when Asyncrhonous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->runtime_args(logical_core);
}

void UpdateRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args) {
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    EnqueueUpdateRuntimeArgs(device->command_queue(), kernel, core_coord, update_idx, runtime_args, false);
}

}  // namespace tt_metal

}  // namespace tt
