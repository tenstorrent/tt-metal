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


inline void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &c, const std::vector<uint32_t> &runtime_args)
{
    detail::GetKernel(program, kernel_id)->set_runtime_args(c, runtime_args);
}


inline void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args)
{
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            SetRuntimeArgs(program, kernel_id, CoreCoord(x,y), runtime_args);
        }
    }
}

}  // namespace

namespace detail {

    void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
        ZoneScoped;
        detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("WriteToDevice");

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

    void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
        ZoneScoped;
        detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ReadFromDevice");

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

    void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
        Device *device = buffer.device();
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1: {
                if (buffer.buffer_type() == BufferType::DRAM) {
                    tt::Cluster::instance().dram_barrier(device->id());
                } else {
                    tt::Cluster::instance().l1_barrier(device->id());
                }
                ReadFromDevice(buffer, host_buffer);
            } break;
            case BufferType::SYSTEM_MEMORY: {
                TT_FATAL(false && "Reading from host memory is unsupported!");
            } break;
            default: TT_FATAL(false && "Unsupported buffer type!");
        }
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
        std::unordered_set<CoreCoord> not_done_cores;
        for (const auto &logical_core : logical_cores_used_in_program) {
            launch_msg_t *msg = &program.kernels_on_core(logical_core)->launch_msg;
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            not_done_cores.insert(worker_core);
            tt::llrt::write_launch_msg_to_core(device->id(), worker_core, msg);
        }

        // Wait for all cores to be done
        llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);

        }//Profiler scope end
        DumpDeviceProfileResults(device, program);
    }


    bool ConfigureDeviceWithProgram(Device *device, Program &program) {
        ZoneScoped;
        bool pass = true;
        detail::DispatchStateCheck( false );
        detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ConfigureDeviceWithProgram");

        std::unordered_set<CoreCoord> worker_cores;
        auto device_id = device->id();

        program.allocate_circular_buffers();
        detail::ValidateCircularBufferRegion(program, device);

        std::vector<CoreCoord> logical_cores_used_in_program = program.logical_cores();
        for (const auto &logical_core : logical_cores_used_in_program) {
            KernelGroup *kernel_group = program.kernels_on_core(logical_core);
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            worker_cores.insert(worker_core);

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
                default: TT_THROW("Unsupported {} processor does not support runtime args", riscv);
            }
            return l1_arg_base;
        };

        for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
            const auto kernel = detail::GetKernel(program, kernel_id);
            auto processor = kernel->processor();
            for (const auto &logical_core : kernel->cores_with_runtime_args()) {
                auto worker_core = device->worker_core_from_logical_core(logical_core);
                const auto & rt_args = kernel->runtime_args(logical_core);
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

Program CreateProgram(){
    return Program();
}

KernelID CreateKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::variant<DataMovementConfig,ComputeConfig, EthernetConfig> &config) {
    return std::visit( [&](auto&& cfg) -> KernelID
                        {
                            CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
                            Kernel * kernel;
                            using T = std::decay_t<decltype(cfg)>;
                            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                                detail::CheckDataMovementConfig(program, file_name, core_ranges);
                                kernel = new DataMovementKernel(file_name, core_ranges, cfg);
                            }
                            else if constexpr (std::is_same_v<T, ComputeConfig>) {
                                kernel = new ComputeKernel(file_name, core_ranges, cfg);
                            }
                            else if constexpr (std::is_same_v<T, EthernetConfig>) {
                                kernel = new EthernetKernel(file_name, core_ranges, cfg);
                            }
                            return detail::AddKernel(program, kernel);
                        },
                        config
                    );
}

CircularBufferID CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return program.add_circular_buffer(core_ranges, config);
}

const CircularBufferConfig &GetCircularBufferConfig(Program &program, CircularBufferID cb_handle) {
    return detail::GetCircularBuffer(program, cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program &program, CircularBufferID cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBuffer> circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program &program, CircularBufferID cb_handle, uint8_t buffer_index, uint32_t page_size) {
    detail::GetCircularBuffer(program, cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program &program, CircularBufferID cb_handle, const Buffer &buffer) {
    if (buffer.buffer_type() != BufferType::L1) {
        TT_FATAL("Only L1 buffers can have an associated circular buffer!");
    }
    detail::GetCircularBuffer(program, cb_handle)->config().set_globally_allocated_address(buffer);
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
    if (kernel_group.riscv0_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv0_id.value())->configure(device, logical_core);
    }
}
void SetRuntimeArgs(const Program &program, KernelID kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &core_spec, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
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

void SetRuntimeArgs(const Program &program, KernelID kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)
{
    ZoneScoped;
    TT_FATAL( core_spec.size() == runtime_args.size(), "Mistmatch between number of cores {} and number of runtime args {} getting updated", core_spec.size(), runtime_args.size());
    Kernel * k = detail::GetKernel(program, kernel);
    for (size_t i = 0; i < core_spec.size(); i++)
        k->set_runtime_args(core_spec[i], runtime_args[i]);
}

std::vector<uint32_t> & GetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core) {
    return detail::GetKernel(program, kernel_id)->runtime_args(logical_core);
}



}  // namespace tt_metal

}  // namespace tt
