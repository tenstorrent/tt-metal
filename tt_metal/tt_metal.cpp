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

namespace tt {

namespace tt_metal {

namespace {

void ConfigureKernelGroup(const Program &program, const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group.compute_id.has_value()) {
        detail::GetKernel(program, kernel_group.compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group.riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv1_id.value())->configure(device, logical_core);
    }
    detail::GetKernel(program, kernel_group.riscv0_id.value())->configure(device, logical_core);
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

    bool ConfigureDeviceWithProgram(Device *device, const Program &program) {
        ZoneScoped;
        bool pass = true;
        detail::DispatchStateCheck( false );
        detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ConfigureDeviceWithProgram");

        std::unordered_set<CoreCoord> worker_cores;
        auto cluster = device->cluster();
        auto device_id = device->id();

        for (auto &[logical_core, kernel_group] : program.core_to_kernel_group()) {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            worker_cores.insert(worker_core);

            if (program.circular_buffers_on_core(logical_core).size()) {
                detail::ValidateCircularBufferRegion(program, device, logical_core);
            }
            // CircularBufferConfigVec -- common across all kernels, so written once to the core
            llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

            // Load firmware into L1 of worker core
            llrt::disable_ncrisc(cluster, device_id, worker_core);     // PROF_BEGIN("CONF_DISABLE_NCTR")
            llrt::disable_triscs(cluster, device_id, worker_core);     // PROF_END("CONF_DISABLE_NCTR")

            ConfigureKernelGroup(program, kernel_group, device, logical_core); // PROF_BEGIN("CONF_KERN") PROF_END("CONF_KERN")

            auto cbs_on_core = program.circular_buffers_on_core(logical_core);         // PROF_BEGIN("CBS")
            for (auto circular_buffer : cbs_on_core) {
                for (auto buffer_index : circular_buffer.buffer_indices()) {
                    llrt::set_config_for_circular_buffer(
                        circular_buffer_config_vec,
                        buffer_index,
                        circular_buffer.address(),
                        circular_buffer.size(),
                        circular_buffer.num_tiles());
                }
            }  // PROF_END("CBS")

            if (cbs_on_core.size()) {
                llrt::write_circular_buffer_config_vector_to_core(
                    cluster,
                    device_id,
                    worker_core,
                    circular_buffer_config_vec);  // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")
            }

            program.init_semaphores(*device, logical_core);
        }

        // Skip loading of blank kernels to storage cores
        for (const CoreCoord &logical_storage_core : device->storage_only_cores()) {
            worker_cores.insert(device->worker_core_from_logical_core(logical_storage_core));
        }

        // Load blank kernel to all riscs of all cores excluding those in worker_cores
        const llrt::TensixRiscsOptions riscs_options = llrt::TensixRiscsOptions::ALL_RISCS;  // PROF_BEGIN("LOAD_BLANK")
        llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
            cluster, device_id, riscs_options, worker_cores);                                // PROF_END("LOAD_BLANK")

        return pass;
    }

    void WriteRuntimeArgsToDevice(Device *device, const Program &program) {
        ZoneScoped;
        auto cluster = device->cluster();
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
                tt::llrt::write_hex_vec_to_core(cluster, device_id, worker_core, rt_args, get_l1_arg_base_addr(processor));
            }
        }
    }

    void CompileProgram(Device *device, Program &program){
        ZoneScoped;
        program.compile(device);
    }
}

Device *CreateDevice(int device_id, const std::vector<uint32_t>& l1_bank_remap) {
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
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        // This triggers the firmware to do logic only needed for slow dispatch
        dm_config.defines["TT_METAL_SLOW_DISPATCH_MODE"] = "1";
    }
    auto kernel = new DataMovementKernel(file_name, core_ranges, dm_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

KernelID CreateComputeKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<ComputeConfig> &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto compute_config = config.has_value() ? config.value() : ComputeConfig{};
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        // This triggers the firmware to do logic only needed for fast dispatch
        compute_config.defines["TT_METAL_SLOW_DISPATCH_MODE"] = "1";
    }
    auto kernel = new ComputeKernel(file_name, core_ranges, compute_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

CircularBufferID CreateCircularBuffer(
    Program &program,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    CoreRange single_core_range = {.start = core, .end = core};
    return CreateCircularBuffers(
        program,
        std::set<u32>({buffer_index}),
        CoreRangeSet({single_core_range}),
        num_tiles,
        size_in_bytes,
        data_format,
        l1_address);
}

CircularBufferID CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(
        program,
        std::set<u32>({buffer_index}),
        CoreRangeSet({core_range}),
        num_tiles,
        size_in_bytes,
        data_format,
        l1_address);
}

CircularBufferID CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(
        program, std::set<u32>({buffer_index}), core_range_set, num_tiles, size_in_bytes, data_format, l1_address);
}

CircularBufferID CreateCircularBuffers(
    Program &program,
    const std::set<uint32_t> &buffer_indices,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return program.add_circular_buffer(
        core_range_set, buffer_indices, num_tiles, size_in_bytes, data_format, l1_address);
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

void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("WriteToDevice");

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
                device->cluster()->write_dram_vec(
                    page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address);
            } break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(
                    device->cluster(), device->id(), noc_coordinates, page, absolute_address);
            } break;
            default: TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}

void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Writing to host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ReadFromDevice");

    host_buffer.clear();  // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
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
                device->cluster()->read_dram_vec(
                    page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address, page_size);
            } break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(
                    device->cluster(), device->id(), noc_coordinates, absolute_address, page_size);
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

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    Device *device = buffer.device();
    tt_cluster *cluster = buffer.device()->cluster();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            if (buffer.buffer_type() == BufferType::DRAM) cluster->dram_barrier(device->id()); else cluster->l1_barrier(device->id());
            ReadFromDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Reading from host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

Buffer CreateBuffer(Device *device, std::uint64_t size, std::uint64_t page_size, const BufferType buffer_type)
{
    return Buffer(device, size, page_size, buffer_type);
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }


void ConfigureKernelGroup(const Program &program, const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group.compute_id.has_value()) {
        detail::GetKernel(program, kernel_group.compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group.riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv1_id.value())->configure(device, logical_core);
    }
    detail::GetKernel(program, kernel_group.riscv0_id.value())->configure(device, logical_core);
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
    auto cluster = device->cluster();
    auto device_id = device->id();

    cluster->dram_barrier(device_id);

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox (eg, storage cores) have all landed
    cluster->l1_barrier(device->id());

    std::vector<CoreCoord> logical_cores_used_in_program = program.logical_cores();
    std::vector<uint32_t> run_mailbox_go_val = {RUN_MSG_GO};
    for (const auto &logical_core : logical_cores_used_in_program) {
        // XXXX move this to llrt
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        tt::llrt::write_hex_vec_to_core(cluster, device_id, worker_core, run_mailbox_go_val, GET_MAILBOX_ADDRESS_HOST(launch.run));
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

            bool is_done = llrt::internal_::check_if_riscs_on_specified_core_done(cluster, device_id, worker_core);

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
