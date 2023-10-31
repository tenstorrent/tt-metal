// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <string>

#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"
// XXXX TODO(PGK): fix include paths so device can export interfaces
#include "tt_metal/src/firmware/riscv/common/dev_msgs.h"

#include "tools/profiler/profiler.hpp"
#include "tools/cpuprof/cpuprof.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/program.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

#include "common/core_coord.h"
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
                                : program.semaphore_address(num_semaphores - 1) + ALIGNED_SIZE_PER_SEMAPHORE;
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
}  // namespace

namespace detail {

    bool ConfigureDeviceWithProgram(Device *device, Program &program) {
        ZoneScoped;
        bool pass = true;
        detail::DispatchStateCheck( false );
        detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ConfigureDeviceWithProgram");

        program.allocate_circular_buffers();

        std::unordered_set<CoreCoord> worker_cores;
        auto device_id = device->id();

        std::vector<CoreCoord> logical_cores_used_in_program = program.logical_cores();
        for (const auto &logical_core : logical_cores_used_in_program) {
            KernelGroup *kernel_group = program.kernels_on_core(logical_core);
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            worker_cores.insert(worker_core);

            if (program.circular_buffers_on_core(logical_core).size()) {
                detail::ValidateCircularBufferRegion(program, device, logical_core);
            }
            // CircularBufferConfigVec -- common across all kernels, so written once to the core
            llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

            ConfigureKernelGroup(program, kernel_group, device, logical_core); // PROF_BEGIN("CONF_KERN") PROF_END("CONF_KERN")

            auto cbs_on_core = program.circular_buffers_on_core(logical_core);         // PROF_BEGIN("CBS")
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
                    worker_core,
                    circular_buffer_config_vec);  // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")
            }

            program.init_semaphores(*device, logical_core);
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
                case RISCV::COMPUTE: {
                    l1_arg_base = TRISC_L1_ARG_BASE;
                }
                break;
                default: log_assert(false, "Unsupported {} processor does not support runtime args", riscv);
            }
            return l1_arg_base;
        };

        for (auto kernel_id : program.kernel_ids()) {
            const auto kernel = detail::GetKernel(program, kernel_id);
            auto processor = kernel->processor();
            for (const auto &[logical_core, rt_args] : kernel->runtime_args()) {
                auto worker_core = device->worker_core_from_logical_core(logical_core);
                tt::llrt::write_hex_vec_to_core(device_id, worker_core, rt_args, get_l1_arg_base_addr(processor));
            }
        }
    }

    void CompileProgram(Device *device, Program &program){
        ZoneScoped;
        program.compile(device);
    }
}

Device *CreateDevice(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap) {
    Device * dev = new Device(device_id, l1_bank_remap);
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        detail::GLOBAL_CQ = std::make_unique<CommandQueue>(dev);
    }
    return dev;
}

bool CloseDevice(Device *device) {
    // Needed to ensure that GLOBAL_CQ doesn't contain a closed device
    if (detail::GLOBAL_CQ) {
        detail::GLOBAL_CQ.reset(nullptr);
    }
    return device->close();
}

KernelID CreateDataMovementKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<DataMovementConfig> &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto dm_config = detail::GetDataMovementConfig(program, file_name, core_ranges, config);
    auto kernel = new DataMovementKernel(file_name, core_ranges, dm_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

KernelID CreateComputeKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<ComputeConfig> &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto compute_config = config.has_value() ? config.value() : ComputeConfig{};
    auto kernel = new ComputeKernel(file_name, core_ranges, compute_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

CircularBufferID CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return program.add_circular_buffer(core_ranges, config);
}

CircularBufferConfig &GetCircularBufferConfig(Program &program, CircularBufferID circular_buffer_id) {
    program.invalidate_circular_buffer_allocation();
    return detail::GetCircularBuffer(program, circular_buffer_id)->config();
}

uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value) {
    return CreateSemaphore(program, CoreRangeSet({core_range}), initial_value);
}

uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value) {
    std::optional<uint32_t> address;
    TT_ASSERT(core_range_set.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
    for (auto core_range : core_range_set.ranges()) {
        auto start_core = core_range.start;
        auto end_core = core_range.end;
        TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
        auto addr = get_semaphore_address(program, core_range);
        if (!address.has_value()) {
            address = addr;
        } else {
            TT_ASSERT(addr == address);
        }
    }
    TT_ASSERT(address.has_value(), "Expecting a valid Semaphore address!");

    program.add_semaphore(core_range_set, address.value(), initial_value);

    return address.value();
}

void WriteToDeviceInterleavedContiguous(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_ASSERT(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", host_buffer_size_bytes, buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_ASSERT(page_size % bytes_per_page_entry == 0);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_storage());
    uint32_t bank_index = 0;
    int data_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
        switch (buffer.buffer_storage()) {
            case BufferStorage::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                tt::Cluster::instance().write_dram_vec(page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address);
            } break;
            case BufferStorage::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
            } break;
            default: TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}


void WriteToDeviceHeightSharded(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
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

    TT_ASSERT(buffer.buffer_storage() == BufferStorage::L1 && "Only L1 Buffers support sharding");

    auto shard_grid_set = buffer.shard_grid();
    std::cout << "Writing to Device Sharded " << std::endl;
    for(auto core_range : shard_grid_set.ranges()){
        auto start_coord = core_range.start;
        auto end_coord = core_range.end;
        auto num_cores_x = (end_coord.x+1) - start_coord.x;
        auto num_cores_y = (end_coord.y+1) - start_coord.y;
        auto row_major = buffer.shard_orientation() == ShardOrientation::ROW_MAJOR;


        auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, row_major);
        int data_index = 0;
        int global_page_index = 0;
        std::cout << "In core range " << core_range.str() << std::endl;
        for(auto core: cores){
            std::cout << "Core: " << core.str() << std::endl;
            auto bank_ids = device->bank_ids_from_logical_core(core);
            for(auto bank_index: bank_ids){
                auto num_pages = buffer.shard_size();
                for(int page_index = 0; page_index < num_pages; page_index++){
                    std::cout << "global_page_index " << global_page_index << std::endl;
                    auto absolute_address = buffer.page_address(bank_index, global_page_index);
                    std::cout << "l1_address " << absolute_address << std::endl;
                    std::vector<uint32_t> page;
                    page.insert(
                        page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
                    auto noc_coordinates = buffer.noc_coordinates(bank_index);
                    llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
                    std::cout << "writing page " << std::endl << "0x";
                    for(auto entry: page){
                        std::cout << std::hex << entry << std::dec;
                    }
                    std::cout << std::dec << std::endl;

                    data_index += num_entries_per_page;
                    global_page_index++;


                }
            }
        }
    }
}

void WriteToDeviceWidthSharded(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
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

    TT_ASSERT(buffer.buffer_storage() == BufferStorage::L1 && "Only L1 Buffers support sharding");

    auto shard_grid_set = buffer.shard_grid();
    std::cout << "Writing to Device Sharded " << std::endl;
    for(auto core_range : shard_grid_set.ranges()){
        auto start_coord = core_range.start;
        auto end_coord = core_range.end;
        auto num_cores_x = (end_coord.x+1) - start_coord.x;
        auto num_cores_y = (end_coord.y+1) - start_coord.y;
        auto row_major = buffer.shard_orientation() == ShardOrientation::ROW_MAJOR;


        auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, row_major);
        int data_index = 0;
        int global_page_index = 0;
        std::cout << "In core range " << core_range.str() << std::endl;
        for(auto core: cores){
            std::cout << "Core: " << core.str() << std::endl;
            auto bank_ids = device->bank_ids_from_logical_core(core);
            for(auto bank_index: bank_ids){
                auto num_pages = buffer.shard_size();
                for(int page_index = 0; page_index < num_pages; page_index++){
                    std::cout << "global_page_index " << global_page_index << std::endl;
                    auto absolute_address = buffer.page_address(bank_index, global_page_index);
                    std::cout << "l1_address " << absolute_address << std::endl;
                    std::vector<uint32_t> page;
                    page.insert(
                        page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
                    auto noc_coordinates = buffer.noc_coordinates(bank_index);
                    llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
                    std::cout << "writing page " << std::endl << "0x";
                    for(auto entry: page){
                        std::cout << std::hex << entry << std::dec;
                    }
                    std::cout << std::dec << std::endl;

                    data_index += num_entries_per_page;
                    global_page_index++;


                }
            }
        }
    }
}



void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("WriteToDevice");
    if(buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED || buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK){
        WriteToDeviceInterleavedContiguous(buffer, host_buffer);
    }
    else if(is_sharded(buffer.buffer_layout())){
        if(buffer.buffer_layout() == TensorMemoryLayout::HEIGHT_SHARDED){
            WriteToDeviceHeightSharded(buffer, host_buffer);
        }
        else{
            TT_ASSERT(false && "Unsupported buffer layout");
        }
    }
    else{
        TT_ASSERT(false && "Unsupported buffer layout");
    }

}

void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_storage()) {
        case BufferStorage::DRAM:
        case BufferStorage::L1: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferStorage::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Writing to host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void ReadFromDeviceInterleavedContiguous(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_storage());

    uint32_t bank_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        switch (buffer.buffer_storage()) {
            case BufferStorage::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                tt::Cluster::instance().read_dram_vec(page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address, page_size);
            } break;
            case BufferStorage::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
            } break;
            default: TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        // Copy page into host buffer
        for (uint32_t entry : page) {
            host_buffer.push_back(entry);
        }

        bank_index = (bank_index + 1) % num_banks;
    }
}

void ReadFromDeviceHeightSharded(const Buffer &buffer, std::vector<uint32_t> &host_buffer, std::optional<TensorMemoryLayout> override_layout){

    TensorMemoryLayout buffer_layout;
    if(override_layout == std::nullopt)
        buffer_layout = buffer.buffer_layout();
    else
        buffer_layout = override_layout.value();

    auto shard_grid_set = buffer.shard_grid();
    auto device = buffer.device();
    uint32_t page_size = buffer.page_size();
    std::cout << "Reading From Device Height Sharded " << std::endl;
    for(auto core_range : shard_grid_set.ranges()){
        auto start_coord = core_range.start;
        auto end_coord = core_range.end;
        auto num_cores_x = (end_coord.x + 1) - start_coord.x;
        auto num_cores_y = (end_coord.y + 1) - start_coord.y;
        auto row_major = buffer.shard_orientation() == ShardOrientation::ROW_MAJOR;


        std::cout << "In core range " << core_range.str() << std::endl;
        auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, row_major);
        int global_page_index =0;
        for(auto core: cores){
            std::cout << "Core: " << core.str() << std::endl;
            auto bank_ids = device->bank_ids_from_logical_core(core);
            for(auto bank_index: bank_ids){
                std::cout << "Bank Index: " << bank_index << std::endl;
                auto num_pages = buffer.shard_size();
                for(int page_index = 0; page_index < num_pages; page_index++){
                    std::cout << "global_page_index " << global_page_index << std::endl;
                    auto absolute_address = buffer.page_address(bank_index, global_page_index);
                    std::cout << "l1_address " << absolute_address << std::endl;
                    auto noc_coordinates = buffer.noc_coordinates(bank_index);
                    auto page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
                    host_buffer.insert(host_buffer.end(), page.begin(), page.end());
                    std::cout << "reading page " << std::endl << "0x";
                    for(auto entry: page){
                        std::cout << std::hex << entry << std::dec;
                    }
                    std::cout << std::dec << std::endl;
                    global_page_index++;
                }
            }
        }

    }
}

void ReadFromDeviceWidthSharded(const Buffer &buffer, std::vector<uint32_t> &host_buffer, std::optional<TensorMemoryLayout> override_layout){


    TensorMemoryLayout buffer_layout;
    if(override_layout == std::nullopt)
        buffer_layout = buffer.buffer_layout();
    else
        buffer_layout = override_layout.value();

    auto shard_grid_set = buffer.shard_grid();
    auto device = buffer.device();
    uint32_t page_size = buffer.page_size();
    std::cout << "Reading From Device Width Sharded " << std::endl;
    for(auto core_range : shard_grid_set.ranges()){
        auto start_coord = core_range.start;
        auto end_coord = core_range.end;
        auto num_cores_x = (end_coord.x + 1) - start_coord.x;
        auto num_cores_y = (end_coord.y + 1) - start_coord.y;
        auto row_major = buffer.shard_orientation() == ShardOrientation::ROW_MAJOR;


        std::cout << "In core range " << core_range.str() << std::endl;
        auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, row_major);
        int global_page_index =0;


        //each column is on a core
        for(auto core: cores){
            std::cout << "Core/Column: " << core.str() << std::endl;
            auto bank_ids = device->bank_ids_from_logical_core(core);
            for(auto bank_index: bank_ids){
                std::cout << "Bank Index: " << bank_index << std::endl;
                auto num_pages = buffer.shard_size();
                for(int page_index = 0; page_index < num_pages; page_index++){
                    std::cout << "global_page_index " << global_page_index << std::endl;
                    auto absolute_address = buffer.page_address(bank_index, global_page_index);
                    std::cout << "l1_address " << absolute_address << std::endl;
                    auto noc_coordinates = buffer.noc_coordinates(bank_index);
                    auto page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
                    host_buffer.insert(host_buffer.end(), page.begin(), page.end());
                    std::cout << "reading page " << std::endl << "0x";
                    for(auto entry: page){
                        std::cout << std::hex << entry << std::dec;
                    }
                    std::cout << std::dec << std::endl;
                    global_page_index++;
                }
            }
        }

    }


}

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer, std::optional<TensorMemoryLayout> override_layout) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ReadFromDevice");

    host_buffer.clear();  // overwrite the data
    if(buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED
        || buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK){
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    }
    else if(is_sharded(buffer.buffer_layout())){
        TT_ASSERT(buffer.buffer_storage() == BufferStorage::L1 && "Only L1 Buffers support sharding");
        if(buffer.buffer_layout() == TensorMemoryLayout::HEIGHT_SHARDED){
            ReadFromDeviceHeightSharded(buffer, host_buffer, std::nullopt);
        }
        else if(buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED){
            ReadFromDeviceWidthSharded(buffer, host_buffer, std::nullopt);
        }
        else{
            TT_ASSERT(false && "Unsupported buffer layout");
        }
    }
    else{
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, std::optional<TensorMemoryLayout> override_layout) {
    Device *device = buffer.device();
    switch (buffer.buffer_storage()) {
        case BufferStorage::DRAM:
        case BufferStorage::L1: {
            if (buffer.buffer_storage() == BufferStorage::DRAM) {
                tt::Cluster::instance().dram_barrier(device->id());
            } else {
                tt::Cluster::instance().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer, override_layout);
        } break;
        case BufferStorage::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Reading from host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

Buffer CreateBuffer(Device *device, std::uint64_t size, std::uint64_t page_size, const BufferStorage buffer_storage,
            const TensorMemoryLayout buffer_layout, std::optional<ShardSpec> shard_parameter)
{
    return Buffer(device, size, page_size, buffer_storage, buffer_layout, shard_parameter);
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }


void ConfigureKernelGroup(const Program &program, const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group.compute_id.has_value()) {
        detail::GetKernel(program, kernel_group.compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group.riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv1_id.value())->configure(device, logical_core);
    }
    if (kernel_group.riscv0_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv0_id.value())->configure(device, logical_core);
    }
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    detail::GetKernel(program, kernel_id)->set_runtime_args(logical_core, runtime_args);
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            CoreCoord logical_core(x, y);
            SetRuntimeArgs(program, kernel_id, logical_core, runtime_args);
        }
    }
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    for (auto core_range : core_range_set.ranges()) {
        SetRuntimeArgs(program, kernel_id, core_range, runtime_args);
    }
}

std::vector<uint32_t> GetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core) {
    return detail::GetKernel(program, kernel_id)->runtime_args(logical_core);
}

void LaunchProgram(Device *device, Program &program) {
    {//Profiler scope start
    ZoneScoped;
    detail::DispatchStateCheck( false );
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("LaunchProgram");
    detail::CompileProgram(device, program);
    detail::WriteRuntimeArgsToDevice(device, program);
    detail::ConfigureDeviceWithProgram(device, program);
    auto device_id = device->id();

    tt::Cluster::instance().dram_barrier(device_id);

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox (eg, storage cores) have all landed
    tt::Cluster::instance().l1_barrier(device->id());

    std::vector<CoreCoord> logical_cores_used_in_program = program.logical_cores();
    for (const auto &logical_core : logical_cores_used_in_program) {
        launch_msg_t *msg = &program.kernels_on_core(logical_core)->launch_msg;
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        tt::llrt::write_launch_msg_to_core(device->id(), worker_core, msg);
    }

    // Wait for all cores to be done

    // get all the cores that need to be polled
    std::unordered_set<CoreCoord> not_done_cores(logical_cores_used_in_program.begin(), logical_cores_used_in_program.end());

    // poll the cores until the set of not done cores is empty
    while (!not_done_cores.empty()) {
        // Print not-done cores
        string not_done_cores_str = "Not done logical cores: ";
        for (const auto &core : not_done_cores) {
            not_done_cores_str += (core.str() + " ");
        }
        not_done_cores_str += "\n";
        log_debug(tt::LogMetal, not_done_cores_str.c_str());

        for (auto it = not_done_cores.begin(); it != not_done_cores.end(); ) {
            const auto &logical_core = *it;

            auto worker_core = device->worker_core_from_logical_core(logical_core);

            bool is_done = llrt::internal_::check_if_riscs_on_specified_core_done(device_id, worker_core);

            if (is_done) {
                log_debug(tt::LogMetal, "Logical core just done: {}", logical_core.str());
                it = not_done_cores.erase(it);
            } else {
                ++it;
            }
        }
    }

    }//Profiler scope end
    detail::DumpDeviceProfileResults(device,program);
}


}  // namespace tt_metal

}  // namespace tt
