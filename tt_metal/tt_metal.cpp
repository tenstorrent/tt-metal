#include <algorithm>
#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <string>

#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "tools/cpuprof/cpuprof.h"

#include "common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/program.hpp"

//TODO(MO): hack until ticket #1184 is in
bool enable_fw_profile_hack = false;

namespace tt {

namespace tt_metal {

namespace {

    detail::CompilationReporter compilation_reporter = detail::CompilationReporter();
    detail::MemoryReporter memory_reporter = detail::MemoryReporter();

    void DownloadFirmware(Device *device, CoreCoord phys_core) {
        for (int riscv_id = 0; riscv_id < 5; riscv_id++)  {
            string fname;
            switch (riscv_id) {
            case 0:
                fname = "brisc/brisc.hex";
                tt::llrt::program_brisc_startup_addr(device->cluster(), device->pcie_slot(), phys_core);
                break;
            case 1: fname = "ncrisc/ncrisc.hex"; break;
            case 2: fname = "tensix_thread0/tensix_thread0.hex"; break;
            case 3: fname = "tensix_thread1/tensix_thread1.hex"; break;
            case 4: fname = "tensix_thread2/tensix_thread2.hex"; break;
            }
            tt::llrt::test_load_write_read_risc_binary(device->cluster(), fname, device->pcie_slot(),
                                                    phys_core, riscv_id, true);
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
                    TT_THROW("Cannot add semaphore on core " + logical_core.str() + ". Max number of semaphores (" + std::to_string(NUM_SEMAPHORES) + ") reached!");
                }
                uint32_t addr = num_semaphores == 0 ? SEMAPHORE_BASE : program.semaphore_address(num_semaphores-1) + ALIGNED_SIZE_PER_SEMAPHORE;
                if (!address.has_value()) {
                    address = addr;
                } else if (addr != address) {
                    TT_THROW("Expected semaphore on logical core " + logical_core.str() + " to be initialized at L1 address " + std::to_string(address.value()) + " but it is at " + std::to_string(addr));
                }
            }
        }
        return address;
    }
}

static Profiler tt_metal_profiler = Profiler();

bool enable_compile_cache = false;
void EnableCompileCache() { enable_compile_cache = true; }
void DisableCompileCache() { enable_compile_cache = false; }
void ClearCompileCache() { detail::HashLookup::inst().clear(); }
bool GetCompileCacheEnabled() { return enable_compile_cache; }

bool enable_compilation_reports = false;
void EnableCompilationReports() { enable_compilation_reports = true; }
void DisableCompilationReports() { enable_compilation_reports = false; }

bool enable_memory_reports = false;
void EnableMemoryReports() { enable_memory_reports = true; }
void DisableMemoryReports() { enable_memory_reports = false; }

void DumpDeviceMemoryState(const Device *device) {
    memory_reporter.dump_memory_usage_state(device);
}

void DumpHostProfileResults(std::string name_prepend){
    tt_metal_profiler.dumpHostResults(name_prepend);
}

void DumpDeviceProfileResults(Device *device, const Program &program) {
    tt_metal_profiler.markStart("DumpDeviceProfileResults");
    TT_ASSERT(tt_is_print_server_running() == false, "Debug print server is running, cannot dump device profiler data");
    auto worker_cores_used_in_program =\
        device->worker_cores_from_logical_cores(program.logical_cores());

    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    tt_metal_profiler.dumpDeviceResults(cluster, pcie_slot, worker_cores_used_in_program);
    tt_metal_profiler.markStop("DumpDeviceProfileResults");
}

void SetHostProfilerFlag(bool do_profile){
     tt_metal_profiler.setHostDoProfile(do_profile);
}

void SetProfilerDir(std::string output_dir){
     tt_metal_profiler.setOutputDir(output_dir);
}

void FreshProfilerHostLog(){
     tt_metal_profiler.setHostNewLogFlag(true);
}

void FreshProfilerDeviceLog(){
     tt_metal_profiler.setDeviceNewLogFlag(true);
}

Host *GetHost() {
    return new Host();
}

Device *CreateDevice(tt::ARCH arch, int pcie_slot) {
    return new Device(arch, pcie_slot);
}

bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator) {

    bool init;
    if (device->initialize(memory_allocator)) {
        static std::mutex build_mutex;
        static bool global_init_complete = false;

        {
            // Need a lock here to prevent the race of building mulitple times
            const std::lock_guard<std::mutex> lock(build_mutex);
            if (!global_init_complete) {
                build_kernel_for_riscv_options_t build_options(device->pcie_slot());
                detail::GenerateBankToNocCoordHeaders(device, &build_options, "");
                std::string arch_name = tt::get_string_lowercase(device->arch());
                generate_binaries_params_t default_params;
                generate_binaries_all_riscs(&build_options,
                                            "",
                                            arch_name,
                                            default_params,
                                            enable_fw_profile_hack);

                char *dbg_print = std::getenv("TT_KERNEL_DEBUG_PRINT");
                if (dbg_print != nullptr) {
                    uint32_t x, y;
                    sscanf(dbg_print, "%d,%d", &x, &y);
                    auto hart_mask = DPRINT_HART_BR;
                    CoreCoord coord = {x, y};
                    tt_start_debug_print_server(device->cluster(), {0}, {coord}, hart_mask);
                    log_debug(tt::LogMetal, "Started debug print server on core {}", coord.str());
                }

                global_init_complete = true;
            }
        }

        // Download to worker cores
        CoreCoord grid_size = device->logical_grid_size();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord phys_core = device->worker_core_from_logical_core(logical_core);
                DownloadFirmware(device, phys_core);

                // This is ugly
                // Enable ncrisc/trisc on worker cores since device dispatch
                // expects this.  Non device-dispatch will override and device
                // dispatch will set up the dispatch cores appropriately
                tt::llrt::enable_ncrisc(device->cluster(), device->pcie_slot(), phys_core);
                tt::llrt::enable_triscs(device->cluster(), device->pcie_slot(), phys_core);
            }
        }

        init = true;
    } else {
        init = false;
    }

    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        HACK_CQ = std::make_unique<CommandQueue>(device);
    }

    return init;
}

bool CloseDevice(Device *device) {

    // Needed to ensure that HACK_CQ doesn't contain a closed device
    if (HACK_CQ) {
        HACK_CQ.reset(nullptr);
    }

    return device->close();
}

void StartDebugPrintServer(Device* device) {
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});
}

void StartDebugPrintServerOnCores(Device* device, const vector<vector<int>>& in_cores) {
    vector<CoreCoord> cores;
    for (int j = 0; j < in_cores.size(); j++) {
        TT_ASSERT(in_cores[j].size() == 2);
        cores.push_back(CoreCoord{size_t(in_cores[j][0]), size_t(in_cores[j][1])});
    }
    tt_start_debug_print_server(device->cluster(), {0}, cores);
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, compile_args, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    TT_ASSERT(core_range.start == core_range.end or core_range.start < core_range.end && "Invalid core range!");
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, compile_args, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc) {
    TT_ASSERT(core_range.start == core_range.end or core_range.start < core_range.end && "Invalid core range!");
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range_set, compile_args, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range_set, processor_type, noc);
    detail::AddKernel( program, kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core,
        compile_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    detail::AddKernel( program, kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    TT_ASSERT(core_range.start == core_range.end or core_range.start < core_range.end && "Invalid core range!");
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core_range,
        compile_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    detail::AddKernel( program, kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core_range_set,
        compile_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    detail::AddKernel( program, kernel);
    return kernel;
}

uint32_t TileSize(const DataFormat &data_format) {
    return tt::tile_size(data_format);
}

const CircularBuffer &CreateCircularBuffer(
    Program &program,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    CoreRange single_core_range = {.start = core, .end = core};
    return CreateCircularBuffers(program, std::set<u32>({buffer_index}), CoreRangeSet({single_core_range}), num_tiles, size_in_bytes, data_format, l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(program, std::set<u32>({buffer_index}), CoreRangeSet({core_range}), num_tiles, size_in_bytes, data_format, l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(program, std::set<u32>({buffer_index}), core_range_set, num_tiles, size_in_bytes, data_format, l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    const std::set<uint32_t> &buffer_indices,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return program.add_circular_buffer(core_range_set, buffer_indices, num_tiles, size_in_bytes, data_format, l1_address);
}


uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value) {
    return CreateSemaphore ( program, CoreRangeSet({core_range}), initial_value );
}

uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value) {
    std::optional<uint32_t> address;
    TT_ASSERT( core_range_set.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
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
    TT_ASSERT( address.has_value(), "Expecting a valid Semaphore address!");

    program.add_semaphore(core_range_set, address.value(), initial_value);

    return address.value();
}

void WriteToDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("WriteToDevice");

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    uint32_t num_entries_per_page = buffer.size() / (num_pages * bytes_per_page_entry);

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());
    uint32_t bank_index = 0;
    int data_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        page.insert(page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                device->cluster()->write_dram_vec(page, tt_target_dram{device->pcie_slot(), dram_channel, 0}, absolute_address);
            }
            break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(device->cluster(), device->pcie_slot(), noc_coordinates, page, absolute_address);
            }
            break;
            default:
                TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
    tt_metal_profiler.markStop("WriteToDevice");
}

void WriteToBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            WriteToDevice(buffer, host_buffer);
        }
        break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Writing to host memory is unsupported!");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("ReadFromDevice");

    host_buffer.clear(); // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());

    uint32_t bank_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        switch (buffer.buffer_type()) {
            case BufferType::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                device->cluster()->read_dram_vec(page, tt_target_dram{device->pcie_slot(), dram_channel, 0}, absolute_address, page_size);
            }
            break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), noc_coordinates, absolute_address, page_size);
            }
            break;
            default:
                TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        // Copy page into host buffer
        for (uint32_t entry: page) {
            host_buffer.push_back(entry);
        }

        bank_index = (bank_index + 1) % num_banks;
    }

    tt_metal_profiler.markStop("ReadFromDevice");
}

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            ReadFromDevice(buffer, host_buffer);
        }
        break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Reading from host memory is unsupported!");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void DeallocateBuffer(Buffer &buffer) {
    buffer.deallocate();
}

bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("ReadFromDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->read_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address, size);
    tt_metal_profiler.markStop("ReadFromDeviceDRAMChannel");
    return pass;
}

bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("WriteToDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->write_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address);
    tt_metal_profiler.markStop("WriteToDeviceDRAMChannel");
    return pass;
}

bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("WriteToDeviceL1");
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    llrt::write_hex_vec_to_core(device->cluster(), device->pcie_slot(), worker_core, host_buffer, address);
    tt_metal_profiler.markStop("WriteToDeviceL1");
    return pass;
}

bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("ReadFromDeviceL1");
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    host_buffer = llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), worker_core, address, size);
    tt_metal_profiler.markStop("ReadFromDeviceL1");
    return pass;
}

bool GenerateBinaries(
    Device *device,
    build_kernel_for_riscv_options_t *build_options,
    const std::string &op_path_suffix,
    bool profile_kernel,
    Kernel *kernel)
{
    std::string arch_name = tt::get_string_lowercase(device->arch());

    generate_descriptors(build_options, op_path_suffix);
    try {
        if (auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
            detail::GenerateBankToNocCoordHeaders(device, build_options, op_path_suffix);
            switch (dm_kernel->data_movement_processor()) {
                case (DataMovementProcessor::RISCV_0): {
                    generate_binary_for_brisc(
                        build_options,
                        op_path_suffix,
                        arch_name,
                        dm_kernel->noc(),
                        dm_kernel->compile_time_args(),
                        profile_kernel);
                }
                break;
                case (DataMovementProcessor::RISCV_1): {
                    generate_binary_for_ncrisc(
                        build_options,
                        op_path_suffix,
                        arch_name,
                        dm_kernel->noc(),
                        dm_kernel->compile_time_args(),
                        profile_kernel);
                }
                break;
                default:
                    TT_ASSERT(false, "Unsupported data movement processor!");
            }
        } else if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
            generate_binaries_for_triscs(
                build_options, op_path_suffix, arch_name, compute_kernel->compile_time_args(), profile_kernel);
        }
    } catch (std::runtime_error &ex) {
        log_error(tt::LogMetal, "EXCEPTION in GenerateBinaries: ", ex.what());
    }
    return true;
}

void SetCircularBufferDataFormat(
    Device *device, const Program &program, Kernel *kernel, build_kernel_for_riscv_options_t &build_options) {
    for (auto logical_core : kernel->logical_cores()) {
        auto cbs_on_core = program.circular_buffers_on_core(logical_core);
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer.buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(static_cast<CB>(buffer_index), circular_buffer.data_format());
            }
        }
    }
}

#ifdef GENERATE_HASH_LOG
#include <fstream>
#endif

size_t KernelCompileHash(Kernel *kernel, build_kernel_for_riscv_options_t &build_options, const int &pcie_slot, bool profile_kernel) {
    string compile_hash_str = std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc));
    compile_hash_str += kernel->compile_time_args_hash();
    compile_hash_str += std::to_string(kernel->define_args_hash());
    compile_hash_str += std::to_string(std::hash<std::string>{}(kernel->name()));
    compile_hash_str += std::to_string(size_t(profile_kernel));

    if (kernel->kernel_type() == KernelType::DataMovement) {
        auto data_movement_kernel = dynamic_cast<DataMovementKernel *>(kernel);
        TT_ASSERT(data_movement_kernel != nullptr);
        compile_hash_str += std::to_string(size_t(data_movement_kernel->noc()));
    } else {
        auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel);
        TT_ASSERT(compute_kernel != nullptr);
        compile_hash_str += std::to_string(std::hash<MathFidelity>{}(compute_kernel->math_fidelity()));
        compile_hash_str += std::to_string(size_t(compute_kernel->fp32_dest_acc_en()));
        compile_hash_str += std::to_string(size_t(compute_kernel->math_approx_mode()));
    }

    size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

    #ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        unique_lock<mutex> lock;
        f << kernel->name() << " :: "
          << std::hash<tt_hlk_desc>{}(build_options.hlk_desc) << " :: "
          << kernel->compile_time_args_hash() << " :: "
          << kernel->define_args_hash() << " :: "
          << std::hash<std::string>{}(kernel->name()) << " :: ";
          << profile_kernel << " :: ";
        if (auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
            f << dm_kernel->noc() << " :: ";
        } else {
            auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel);
            f << compute_kernel->math_fidelity() << " :: "
              << compute_kernel->fp32_dest_acc_en() << " :: "
              << compute_kernel->math_approx_mode() << " :: ";
        }
        f << compile_hash_str
        f << compile_hash
          << std::endl << std::flush;
    }
    #endif
    return compile_hash;
}



void SetBuildKernelOptions(Kernel *kernel, build_kernel_for_riscv_options_t &build_options) {
    if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
        build_options.set_hlk_file_name_all_cores(compute_kernel->kernel_path_file_name());
        build_options.set_hlk_math_fidelity_all_cores(compute_kernel->math_fidelity());
        build_options.set_hlk_math_approx_mode_all_cores(compute_kernel->math_approx_mode());
        build_options.fp32_dest_acc_en = compute_kernel->fp32_dest_acc_en();
        build_options.hlk_defines = compute_kernel->defines();
    } else {
        auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel);
        switch (dm_kernel->data_movement_processor()) {
                case (DataMovementProcessor::RISCV_0): {
                    build_options.brisc_kernel_file_name = dm_kernel->kernel_path_file_name();
                    build_options.brisc_defines = dm_kernel->defines();
                }
                break;
                case (DataMovementProcessor::RISCV_1): {
                    build_options.ncrisc_kernel_file_name = dm_kernel->kernel_path_file_name();
                    build_options.ncrisc_defines = dm_kernel->defines();
                }
                break;
                default:
                    TT_ASSERT(false, "Unsupported data movement processor!");
        }
    }
}

void CompileKernel(Device *device, Program &program, Kernel *kernel, bool profile_kernel) {
    build_kernel_for_riscv_options_t build_options(device->pcie_slot(), kernel->name());

    SetBuildKernelOptions(kernel, build_options);
    SetCircularBufferDataFormat(device, program, kernel, build_options);

    auto kernel_hash = KernelCompileHash(kernel, build_options, device->pcie_slot(), profile_kernel);
    std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash);

    bool cache_hit = true;
    bool path_exists = std::filesystem::exists(build_options.outpath + kernel_path_suffix);
    if ( enable_compile_cache && path_exists ){
        TT_ASSERT ( detail::HashLookup::inst().exists(kernel_hash) );
    } else if ( detail::HashLookup::inst().add(kernel_hash) ) {
        cache_hit = false;
        GenerateBinaries(device, &build_options, kernel_path_suffix, profile_kernel, kernel);
    }

    if (enable_compilation_reports) {
        compilation_reporter.add_kernel_compile_stats(program, kernel, cache_hit, kernel_hash);
    }

    kernel->set_binary_path(kernel_path_suffix);
}

void AddBlankKernels(Device *device, Program &program, bool profile_kernel) {
    // This can be smarter by combining core ranges into maximal rectangles but this code can be removed once we load BRISC FW separately from the kernel binary
    std::set<CoreRange> unique_core_ranges_without_brisc_kernel,
                        unique_core_ranges_without_ncrisc_kernel,
                        unique_core_ranges_without_compute_kernel ;
    vector<Kernel *> blanks;
    for (auto &[logical_core, kernel_group] : program.core_to_kernel_group()) {
        CoreRange core_range = {.start = logical_core, .end = logical_core};
        if (kernel_group.riscv_0 == nullptr)
            unique_core_ranges_without_brisc_kernel.insert(core_range);
        if (kernel_group.riscv_1 == nullptr )
            unique_core_ranges_without_ncrisc_kernel.insert(core_range);
        if (kernel_group.compute == nullptr)
            unique_core_ranges_without_compute_kernel.insert(core_range);
    }

    if (not unique_core_ranges_without_brisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_brisc_kernel);
        auto blank_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_range_set, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
        blanks.push_back(blank_kernel);
    }

    if (not unique_core_ranges_without_ncrisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_ncrisc_kernel);
        auto blank_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_range_set, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

        blanks.push_back(blank_kernel);
    }
    if (not unique_core_ranges_without_compute_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_compute_kernel);
        auto blank_kernel = CreateComputeKernel(
            program, "tt_metal/kernels/compute/blank.cpp", core_range_set, {0}, MathFidelity::HiFi4, false, false);

        blanks.push_back(blank_kernel);
    }

    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        for (const auto &blank_kernel : blanks) {
            blank_kernel->add_define("TT_METAL_DEVICE_DISPATCH_MODE", 1);
        }
    }

    for (const auto &blank_kernel : blanks) {
        CompileKernel(device, program, blank_kernel, profile_kernel);
        blank_kernel->read_binaries(device->pcie_slot());
    }

}

bool CompileProgram(Device *device, Program &program, bool profile_kernel) {
    log_assert(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is dependent on information that is set during device initialization.", program.get_id()
    );

    bool pass = true;
    tt_metal_profiler.markStart("CompileProgram");
    std::vector< std::future<void> > events;
    log_assert(!(profile_kernel && tt_is_print_server_running()), "Debug print server is running, profiling is not allowed");
    tt_set_profiler_state_for_debug_print(profile_kernel);

    {
        tf::Taskflow tf;

        tf.emplace([device, &program, profile_kernel] { AddBlankKernels(device, program, profile_kernel); });

        // Currently we want to support both slow and fast dispatch until we
        // fully move over to fast, so using this env var method to set all
        // the kernels to using fast dispatch mode. Need to be done after adding
        // blanks to ensure that they get the define, too.
        const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");

        if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
            for (auto kernel : program.kernels()) {
                tf.emplace([kernel, device, &program, profile_kernel] {
                    kernel->add_define("TT_METAL_DEVICE_DISPATCH_MODE", 1);
                    CompileKernel(device, program, kernel, profile_kernel);
                });
            }
        } else {
            for (auto kernel : program.kernels()) {
                tf.emplace([kernel, device, &program, profile_kernel] {
                    CompileKernel(device, program, kernel, profile_kernel);
                });
            }
        }

        GetExecutor().run(tf).wait();
    }
    {
        tf::Taskflow tf;

        for (auto kernel : program.kernels()) {
            tf.emplace ( [kernel, device] { kernel->read_binaries(device->pcie_slot()); });
        }
        GetExecutor().run(tf).wait();
    }
    program.construct_core_range_set_for_worker_cores();

    if (enable_compilation_reports) {
        compilation_reporter.flush_program_entry(program, enable_compile_cache);
    }
    if (enable_memory_reports) {
        memory_reporter.flush_program_memory_usage(program, device);
    }

    tt_metal_profiler.markStop("CompileProgram");
    return pass;
}

void ValidateKernelGroup(const KernelGroup &kernel_group, const CoreCoord &logical_core) {
    if (kernel_group.riscv_0 != nullptr and kernel_group.riscv_1 != nullptr) {
        if (kernel_group.riscv_0->noc() == kernel_group.riscv_1->noc() and kernel_group.riscv_0->name() != "blank") {
            TT_THROW("Data movement kernels on RISCV_0 and RISCV_1 on core " + logical_core.str() + " cannot use the same NOC, doing so results in a hang!");
        }
    }
}

void ConfigureKernelGroup(const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group.compute != nullptr) {
        kernel_group.compute->configure(device, logical_core);
    }
    if (kernel_group.riscv_1 != nullptr) {
        kernel_group.riscv_1->configure(device, logical_core);
    }
    kernel_group.riscv_0->configure(device, logical_core);
}

bool ConfigureDeviceWithProgram(Device *device, const Program &program) {
    bool pass = true;

    tt_metal_profiler.markStart("ConfigureDeviceWithProgram");
    std::vector<CoreCoord> worker_cores;
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    for (auto &[logical_core, kernel_group] : program.core_to_kernel_group()) {
        ValidateKernelGroup(kernel_group, logical_core);
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        worker_cores.push_back(worker_core);

        detail::ValidateCircularBufferRegion(program, device, logical_core);
        // CircularBufferConfigVec -- common across all kernels, so written once to the core
        llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

        // Load firmware into L1 of worker core
        llrt::disable_ncrisc(cluster, pcie_slot, worker_core); // PROF_BEGIN("CONF_DISABLE_NCTR")
        llrt::disable_triscs(cluster, pcie_slot, worker_core); // PROF_END("CONF_DISABLE_NCTR")

        ConfigureKernelGroup(kernel_group, device, logical_core); // PROF_BEGIN("CONF_KERN") PROF_END("CONF_KERN")

        // Initialize registers to INVALID
        constexpr static uint32_t INVALID = 0x4321; // PROF_BEGIN("WRITE_HEX")
        uint32_t stream_register_address = STREAM_REG_ADDR(0, 24);
        llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, {INVALID}, stream_register_address); // PROF_END("WRITE_HEX")

        auto cbs_on_core = program.circular_buffers_on_core(logical_core); // PROF_BEGIN("CBS")
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer.buffer_indices()) {
                llrt::set_config_for_circular_buffer(
                    circular_buffer_config_vec,
                    buffer_index,
                    circular_buffer.address(),
                    circular_buffer.size(),
                    circular_buffer.num_tiles()
                );
            }
        } // PROF_END("CBS")

        llrt::write_circular_buffer_config_vector_to_core(cluster, pcie_slot, worker_core, circular_buffer_config_vec); // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")

        program.init_semaphores(*device, logical_core);
    }

    // Skip loading of blank kernels to storage cores when using L1 banking
    if (device->allocator_scheme() == MemoryAllocator::L1_BANKING) {
        for (const auto& core : device->cluster()->get_soc_desc(device->pcie_slot()).storage_cores) {
            const auto logical_coord = get_core_coord_from_relative(core, device->logical_grid_size());
            worker_cores.push_back(device->worker_core_from_logical_core(logical_coord));
        }
        std::sort(worker_cores.begin(), worker_cores.end());
    }

    // Load blank kernel to all riscs of all cores excluding those in worker_cores
    const llrt::TensixRiscsOptions riscs_options = llrt::TensixRiscsOptions::ALL_RISCS; // PROF_BEGIN("LOAD_BLANK")
    llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, pcie_slot, riscs_options, worker_cores);                               // PROF_END("LOAD_BLANK")

    tt_metal_profiler.markStop("ConfigureDeviceWithProgram");
    return pass;
}

void SetRuntimeArgs(Kernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    TT_ASSERT(kernel->kernel_type() != KernelType::Compute, "Compute kernels do not support runtime args");
    kernel->set_runtime_args(logical_core, runtime_args);
}

void SetRuntimeArgs(Kernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args) {
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            CoreCoord logical_core(x, y);
            SetRuntimeArgs(kernel, logical_core, runtime_args);
        }
    }
}

void SetRuntimeArgs(Kernel *kernel, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args) {
    for (auto core_range : core_range_set.ranges()) {
        SetRuntimeArgs(kernel, core_range, runtime_args);
    }
}

std::vector<uint32_t> GetRuntimeArgs(Kernel *kernel, const CoreCoord &logical_core) {
    return kernel->runtime_args(logical_core);
}

void WriteRuntimeArgsToDevice(Device *device, const Program &program) {
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    auto get_l1_arg_base_addr = [](const RISCV &riscv) {
        uint32_t l1_arg_base = 0;
        switch (riscv) {
            case RISCV::BRISC: {
                l1_arg_base = BRISC_L1_ARG_BASE;
            }
            break;
            case RISCV::NCRISC: {
                l1_arg_base = NCRISC_L1_ARG_BASE;
            }
            break;
            default:
                log_assert(false, "Unsupported {} processor does not support runtime args", riscv);
        }
        return l1_arg_base;
    };

    for (const auto &kernel : program.kernels()) {
        auto processor = kernel->processor();
        for (const auto &[logical_core, rt_args] : kernel->runtime_args()) {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            tt::llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, rt_args, get_l1_arg_base_addr(processor));
        }
    }
}

bool core_runs_ncrisc(const Program &program, const CoreCoord &logical_core) {
    auto kernel_group = program.kernels_on_core(logical_core);
    return kernel_group.riscv_1 != nullptr;
}

bool core_runs_triscs(const Program &program, const CoreCoord &logical_core) {
    auto kernel_group = program.kernels_on_core(logical_core);
    return kernel_group.compute != nullptr;
}

llrt::TensixRiscsOptions GetRiscOptionFromCoreConfig(bool core_runs_ncrisc, bool core_runs_triscs) {
    auto risc_option = llrt::TensixRiscsOptions::BRISC_ONLY;
    if (core_runs_ncrisc and not core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_NCRISC;
    } else if (not core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_TRISCS;
    } else if (core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::ALL_RISCS;
    }
    return risc_option;
}

bool LaunchKernels(Device *device, const Program &program, bool stagger_start) {

    tt_metal_profiler.markStart("LaunchKernels");
    bool pass = true;

    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    // Cores have to be enabled before BRISC reset is de-asserted
    auto logical_cores_used_in_program = program.logical_cores();
    auto worker_cores = device->worker_cores_from_logical_cores(logical_cores_used_in_program);

    llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster, stagger_start);

    bool riscs_are_done = false;
    while (not riscs_are_done) {
        riscs_are_done = true;
        for (const auto &logical_core : logical_cores_used_in_program) {
            // Check if all the riscs on the core are done
            bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
            bool triscs_run = core_runs_triscs(program, logical_core);
            auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            riscs_are_done &=
                llrt::internal_::check_if_riscs_on_specified_core_done(cluster, pcie_slot, risc_option, worker_core);
        }
    }

    // Reset the mailboxes on each core to enable multiple launches of the same program
    // without needing to re-configure the device
    for (const auto &logical_core : logical_cores_used_in_program) {
        bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
        bool triscs_run = core_runs_triscs(program, logical_core);
        auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        llrt::internal_::setup_riscs_on_specified_core(cluster, pcie_slot, risc_option, worker_core);
    }

    // Reset the device that was running
    cluster->broadcast_remote_tensix_risc_reset(pcie_slot, TENSIX_ASSERT_SOFT_RESET);


    tt_metal_profiler.markStop("LaunchKernels");
    return pass;
}

bool WriteToDeviceL1(
    Device *device,
    const CoreCoord &core,
    op_info_t op_info,
    int op_idx) {
    int pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    llrt::write_graph_interpreter_op_info_to_core(
        device->cluster(), device->pcie_slot(), worker_core, op_info, op_idx);
    return pass;
}

void Synchronize() {
    if (HACK_CQ) {
        Finish(*HACK_CQ);
    }
}

}  // namespace tt_metal

}  // namespace tt
