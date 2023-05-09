#include <algorithm>
#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <string>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/cpuprof/cpuprof.h"

using std::unique_lock;
using std::mutex;

namespace tt { namespace llrt {
extern bool llrt_enable_binary_cache;
} }
namespace tt {

namespace tt_metal {

static Profiler tt_metal_profiler = Profiler();

bool enable_compile_cache = false;
int force_recompiles = 0;
void EnableCompileCache() { enable_compile_cache = true; }
void DisableCompileCache() { enable_compile_cache = false; }
bool GetCompileCacheEnabled() { return enable_compile_cache; }
void SetForceRecompiles(int newval) { enable_compile_cache = true; force_recompiles = newval; }
int  GetForceRecompiles() { return force_recompiles; }
void EnableBinaryCache() { tt::llrt::llrt_enable_binary_cache = true; }
void DisableBinaryCache() { tt::llrt::llrt_enable_binary_cache = false; }
bool GetBinaryCacheEnabled() { return tt::llrt::llrt_enable_binary_cache; }

void DumpHostProfileResults(std::string name_prepend){
    tt_metal_profiler.dumpHostResults(name_prepend);
}

void DumpDeviceProfileResults(Device *device, Program *program) {
    auto worker_cores_used_in_program =\
        device->worker_cores_from_logical_cores(program->logical_cores());

    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    tt_metal_profiler.dumpDeviceResults(cluster, pcie_slot, worker_cores_used_in_program);
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
    TT_ASSERT(arch == tt::ARCH::GRAYSKULL, "Only Grayskull is supported!");
    return new Device(arch, pcie_slot);
}

bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator) {
    return device->initialize(memory_allocator);
}

bool CloseDevice(Device *device) { return device->close(); }

void StartDebugPrintServer(Device* device) {
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}}); // TODO(AP): temp, need to rethink
}

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const tt_xy_pair &logical_core, const std::vector<uint32_t> &compile_time_args) {
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(logical_core, compile_time_args);
    return kernel_args;
}

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreRange &core_range, const std::vector<uint32_t> &compile_time_args) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    CoreBlocks core_blocks = {core_range};
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(core_blocks, {compile_time_args});
    return kernel_args;
}

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec) {
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(core_blocks, compile_time_args_spec);
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const tt_xy_pair &logical_core, const vector<uint32_t> &compile_time_args) {
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(logical_core, compile_time_args);
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreRange &core_range, const vector<uint32_t> &compile_time_args) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    CoreBlocks core_blocks = {core_range};
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(core_blocks, {compile_time_args});
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec) {
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(core_blocks, compile_time_args_spec);
    return kernel_args;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    DataMovementProcessor processor_type,
    NOC noc) {
    auto kernel_args = new DataMovementKernelArgs();
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core,
        kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    auto kernel_args = new DataMovementKernelArgs();
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core_range,
        kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    program->add_kernel(kernel);
    return kernel;
}

uint32_t DatumSize(const DataFormat &data_format) {
    return tt::datum_size(data_format);
}

uint32_t TileSize(const DataFormat &data_format) {
    return tt::tile_size(data_format);
}

DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address) {
    TT_ASSERT(dram_channel >= 0 and dram_channel <= 7, "Valid range for DRAM channel is [0, 7]");
    DramBuffer *buffer = new DramBuffer(device, dram_channel, size_in_bytes, address);
    return buffer;
}

DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes) {
    TT_ASSERT(dram_channel >= 0 and dram_channel <= 7, "Valid range for DRAM channel is [0, 7]");
    DramBuffer *buffer = new DramBuffer(device, dram_channel, size_in_bytes);
    return buffer;
}

InterleavedDramBuffer *CreateInterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {
    InterleavedDramBuffer *interleaved_buffer = new InterleavedDramBuffer(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    return interleaved_buffer;
}

InterleavedL1Buffer *CreateInterleavedL1Buffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {
    InterleavedL1Buffer *interleaved_buffer = new InterleavedL1Buffer(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    return interleaved_buffer;
}

L1Buffer *CreateL1Buffer(Program *program, Device *device, const tt_xy_pair &core, uint32_t size_in_bytes, uint32_t address) {
    L1Buffer *l1_buffer = new L1Buffer(device, core, size_in_bytes, address);
    program->add_l1_buffer(l1_buffer);
    return l1_buffer;
}

L1Buffer *CreateL1Buffer(Program *program, Device *device, const tt_xy_pair &core, uint32_t size_in_bytes) {
    L1Buffer *l1_buffer = new L1Buffer(device, core, size_in_bytes);
    program->add_l1_buffer(l1_buffer);
    return l1_buffer;
}

CircularBuffer *CreateCircularBuffer(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const tt_xy_pair &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format) {
    CircularBuffer *circular_buffer =
        new CircularBuffer(device, core, buffer_index, num_tiles, size_in_bytes, l1_address, data_format);
    program->add_circular_buffer(circular_buffer);
    return circular_buffer;
}

CircularBuffer *CreateCircularBuffer(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const tt_xy_pair &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) {
    CircularBuffer *circular_buffer =
        new CircularBuffer(device, core, buffer_index, num_tiles, size_in_bytes, data_format);
    program->add_circular_buffer(circular_buffer);
    return circular_buffer;
}

std::vector<CircularBuffer *> CreateCircularBuffers(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format) {
    std::vector<CircularBuffer *> circular_buffers;
    auto start_core = core_range.first;
    auto end_core = core_range.second;
    TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = tt_xy_pair(x, y);
            CircularBuffer *circular_buffer =
                new CircularBuffer(device, core, buffer_index, num_tiles, size_in_bytes, l1_address, data_format);
            program->add_circular_buffer(circular_buffer);
            circular_buffers.push_back(circular_buffer);
        }
    }
    return circular_buffers;
}

std::vector<CircularBuffer *> CreateCircularBuffers(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) {
    uint32_t l1_address = device->address_for_circular_buffers_across_core_range(core_range, size_in_bytes);
    std::vector<CircularBuffer *> circular_buffers;
    auto start_core = core_range.first;
    auto end_core = core_range.second;
    TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            CircularBuffer *circular_buffer =
                new CircularBuffer(device, logical_core, buffer_index, num_tiles, size_in_bytes, l1_address, data_format);
            // CBs constructed with address don't invoke the allocator, so we need to manually invoke it by reserving space
            circular_buffer->reserve();
            program->add_circular_buffer(circular_buffer);
            circular_buffers.push_back(circular_buffer);
        }
    }
    return circular_buffers;
}

std::vector<Semaphore *> CreateSemaphores(Program *program, Device *device, const CoreRange &core_range, uint32_t initial_value) {
    std::vector<Semaphore *> semaphores;
    auto start_core = core_range.first;
    auto end_core = core_range.second;
    auto size_per_semaphore = SEMAPHORE_SIZE / NUM_SEMAPHORES;
    TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
    uint32_t address = -1;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            auto semaphores_on_core = program->semaphores_on_core(logical_core);
            if (semaphores_on_core.size() == NUM_SEMAPHORES) {
                TT_THROW("Cannot add semaphore on core " + logical_core.str() + ". Max number of semaphores (" + std::to_string(NUM_SEMAPHORES) + ") reached!");
            }
            uint32_t addr = semaphores_on_core.empty() ? SEMAPHORE_BASE : semaphores_on_core.back()->address() + size_per_semaphore;
            if (address == -1) {
                address = addr;
            } else {
                TT_ASSERT(addr == address);
            }
            Semaphore *semaphore = new Semaphore(device, logical_core, address, initial_value);
            program->add_semaphore(semaphore);
            semaphores.push_back(semaphore);
        }
    }
    return semaphores;
}

void DeallocateBuffer(Buffer *buffer) {
    buffer->free();
}

bool GenerateBinaries(
    Device *device,
    build_kernel_for_riscv_options_t *build_kernel_for_riscv_options,
    const std::string &op_path,
    bool profile_kernel,
    const KernelGroup &kernel_group,
    const tt_xy_pair &logical_core)
{
    std::string arch_name = tt::get_string_lowercase(device->arch());

    auto brisc_lambda = [=]() {
        generate_binary_for_brisc(
            build_kernel_for_riscv_options,
            op_path,
            arch_name,
            kernel_group.riscv_0->noc(),
            kernel_group.riscv_0->compile_time_args(logical_core),
            profile_kernel); };
    auto ncrisc_lambda = [=]() {
        generate_binary_for_ncrisc(
            build_kernel_for_riscv_options,
            op_path,
            arch_name,
            kernel_group.riscv_1->noc(),
            kernel_group.riscv_1->compile_time_args(logical_core),
            profile_kernel); };

    generate_descriptors(build_kernel_for_riscv_options, op_path);
    std::thread br_thread(brisc_lambda);
    std::thread nc_thread(ncrisc_lambda);
    if (kernel_group.compute != nullptr) {
        auto triscs_lambda = [=]() {
            generate_binaries_for_triscs(
                build_kernel_for_riscv_options, op_path, arch_name, kernel_group.compute->compile_time_args(logical_core), profile_kernel);
        };
        std::thread tr_thread(triscs_lambda);
        tr_thread.join();
    }
    nc_thread.join();
    br_thread.join();
    return true;
}

bool BlankKernelBinariesExist(const std::string &blank_op_path) {
    bool binaries_exist = std::filesystem::exists(blank_op_path + "/brisc/brisc.hex");
    binaries_exist &= std::filesystem::exists(blank_op_path + "/ncrisc/ncrisc.hex");
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        std::string trisc_id_str = std::to_string(trisc_id);
        std::string trisc_hex_name = blank_op_path + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
        binaries_exist &= std::filesystem::exists(trisc_hex_name);
    }
    return binaries_exist;
}

void CompileBlankKernel(Device *device, const std::string &out_dir_path) {
    build_kernel_for_riscv_options_t blank_build_options("blank_op", "blank_op");
    // Crude way to check if blank_op needs to be compiled or not
    if (BlankKernelBinariesExist(out_dir_path + blank_build_options.name)) {
        return;
    }
    struct hlk_args_t {
        std::int32_t dummy;
    };
    void *hlk_args = new hlk_args_t{
        .dummy = 0,
    };
    blank_build_options.set_hlk_args_all_cores(hlk_args, sizeof(hlk_args_t));
    blank_build_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
    blank_build_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    blank_build_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    std::string arch_name = tt::get_string_lowercase(device->arch());

    generate_binaries_params_t default_params;
    generate_binaries_all_riscs(&blank_build_options, out_dir_path + blank_build_options.name, arch_name, default_params);
}

void SetCircularBufferDataFormat(
    Program *program, const tt_xy_pair &logical_core, build_kernel_for_riscv_options_t *build_kernel_for_riscv_options, const std::string &op_path) {
    for (auto circular_buffer : program->circular_buffers_on_core(logical_core)) {
        build_kernel_for_riscv_options->set_cb_dataformat_all_cores(
            static_cast<CB>(circular_buffer->buffer_index()), circular_buffer->data_format());
    }
    std::filesystem::create_directories(op_path);
    generate_data_format_descriptors(build_kernel_for_riscv_options, op_path);
}

void ValidateL1Buffers(Device *device, Program *program, const tt_xy_pair &logical_core) {
    auto l1_buffers_on_core = program->l1_buffers_on_core(logical_core);
    auto cbs_on_core = program->circular_buffers_on_core(logical_core);
    std::unordered_set<uint32_t> buffer_addresses;
    uint32_t total_l1_buffer_size_in_bytes = 0;
    uint32_t max = device->l1_size() - UNRESERVED_BASE;
    for (auto l1_buffer : l1_buffers_on_core) {
        if (buffer_addresses.find(l1_buffer->address()) != buffer_addresses.end()) {
            continue;
        }
        buffer_addresses.insert(l1_buffer->address());
        total_l1_buffer_size_in_bytes += l1_buffer->size();
        if (total_l1_buffer_size_in_bytes > max) {
            TT_THROW("Size of L1 buffers on " + logical_core.str() + "is " + std::to_string(total_l1_buffer_size_in_bytes) + " bytes, which exceeds maximum size of " + std::to_string(max) + " bytes");
        }
    }
    for (auto circular_buffer : cbs_on_core) {
        if (buffer_addresses.find(circular_buffer->address()) != buffer_addresses.end()) {
            continue;
        }
        buffer_addresses.insert(circular_buffer->address());
        total_l1_buffer_size_in_bytes += circular_buffer->size();
        if (total_l1_buffer_size_in_bytes > max) {
            TT_THROW("Size of L1 buffers on " + logical_core.str() + "is " + std::to_string(total_l1_buffer_size_in_bytes) + " bytes, which exceeds maximum size of " + std::to_string(max) + " bytes");
        }
    }
}

void ValidateKernelGroup(const KernelGroup &kernel_group, const tt_xy_pair &logical_core) {
    if (kernel_group.riscv_0 != nullptr and kernel_group.riscv_1 != nullptr) {
        if (kernel_group.riscv_0->noc() == kernel_group.riscv_1->noc()) {
            TT_THROW("Data movement kernels on RISCV_0 and RISCV_1 on core " + logical_core.str() + " cannot use the same NOC, doing so results in a hang!");
        }
    }
}

// Gets all kernels running on a specific core and creates blank kernels for RISCV0 and RISCV1 if the data movement
// processors do not have a kernel
void PopulateKernelGroupWithDataMovementKernels(Program *program, KernelGroup &kernel_group, const tt_xy_pair &logical_core) {
    // Toggle NOC
    std::function<NOC(DataMovementKernel *, NOC)> get_noc_id = [&](DataMovementKernel *existing_dm_kernel, NOC default_noc) {
        if (existing_dm_kernel != nullptr) {
            uint8_t toggled_noc =  !(existing_dm_kernel->noc());
            return static_cast<NOC>(toggled_noc);
        }
        return default_noc;
    };

    DataMovementKernelArgs *empty_kernel_args = new DataMovementKernelArgs();
    if (kernel_group.riscv_0 == nullptr) {
        NOC riscv_0_noc = get_noc_id(kernel_group.riscv_1, NOC::RISCV_0_default);
        auto riscv_0_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            logical_core,
            empty_kernel_args,
            DataMovementProcessor::RISCV_0,
            riscv_0_noc);
        kernel_group.riscv_0 = riscv_0_kernel;
    }

    if (kernel_group.riscv_1 == nullptr) {
        NOC riscv_1_noc = get_noc_id(kernel_group.riscv_0, NOC::RISCV_1_default);
        auto riscv_1_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            logical_core,
            empty_kernel_args,
            DataMovementProcessor::RISCV_1,
            riscv_1_noc);
        kernel_group.riscv_1 = riscv_1_kernel;
    }
}

std::string GetOpName(const KernelGroup &kernel_group) {
    std::string dummy_op_name;
    std::vector<Kernel *> kernels = {kernel_group.compute, kernel_group.riscv_0, kernel_group.riscv_1};
    for (auto kernel_index = 0; kernel_index < kernels.size(); kernel_index++) {
        auto kernel = kernels.at(kernel_index);
        if (kernel == nullptr) {
            continue;
        }
        dummy_op_name += kernel->name();
        if (kernel_index != kernels.size() - 1) {
            dummy_op_name += "_";
        }
    }
    return dummy_op_name;
}

size_t KernelGroupCompileHash(const KernelGroup &kernel_group, const tt_xy_pair &logical_core, const std::string &op_name) {
    size_t kg_compile_hash = 0;
    if (kernel_group.compute != nullptr) {
        tt::utils::hash_combine(kg_compile_hash, kernel_group.compute->compile_time_args_hash(logical_core));
        tt::utils::hash_combine(kg_compile_hash, kernel_group.compute->define_args_hash(logical_core));
    }
    tt::utils::hash_combine(kg_compile_hash, kernel_group.riscv_0->compile_time_args_hash(logical_core));
    tt::utils::hash_combine(kg_compile_hash, kernel_group.riscv_1->compile_time_args_hash(logical_core));
    tt::utils::hash_combine(kg_compile_hash, std::hash<std::string>{}(op_name));
    return kg_compile_hash;
}

struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        return hashes_.find(khash) != hashes_.end();
    }
    void add(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        hashes_.insert(khash);
    }

private:
    std::mutex mutex_;
    std::unordered_set<size_t> hashes_;
};

bool CompileProgram(Device *device, Program *program, bool profile_kernel) {
    bool pass = true;
    tt_metal_profiler.markStart("CompileProgram");

    std::string out_dir_path = tt::utils::get_root_dir() + "/built_kernels/";
    CompileBlankKernel(device, out_dir_path); // PROF_BEGIN("CCBLANK") PROF_END("CCBLANK")

    // Compute kernels generate dependencies for data movement kernels
    // Kernels running on a core need to be grouped together for compilation
    // The same group of kernels shouldn't be compiled multiple times
    auto op_idx = 0;
    for (auto &[logical_core, kernel_group] : program->core_to_kernel_group()) {
        ValidateL1Buffers(device, program, logical_core); // PROF_BEGIN("CCGEN_PREAMBLE")

        ValidateKernelGroup(kernel_group, logical_core);

        // Modifies kernel_group to have blank data movement kernels if they are not present
	    PopulateKernelGroupWithDataMovementKernels(program, kernel_group, logical_core);

        auto dummy_op_name = GetOpName(kernel_group);
        build_kernel_for_riscv_options_t dummy_op("dummy_type", dummy_op_name + std::to_string(op_idx++));

        auto kernel_group_hash = KernelGroupCompileHash(kernel_group, logical_core, dummy_op_name);
        std::string op_path = out_dir_path + dummy_op_name + "/" + std::to_string(kernel_group_hash);

        SetCircularBufferDataFormat(program, logical_core, &dummy_op, op_path);

        string root_dir = tt::utils::get_root_dir();
        ConfigureForCompilation(kernel_group.compute, &dummy_op, logical_core, op_path);
        ConfigureForCompilation(kernel_group.riscv_0, &dummy_op, logical_core, op_path);
        ConfigureForCompilation(kernel_group.riscv_1, &dummy_op, logical_core, op_path);

        if (HashLookup::inst().exists(kernel_group_hash)) {
            //std::cout << "--- Kernel Cache hit" << std::endl;
            continue;
        }

        // TODO(AP): this is a hack to speed up the debugging process
        bool path_exists = false;
        if (enable_compile_cache)
            path_exists = std::filesystem::exists(op_path); // PROF_END("CCGEN_PREAMBLE")
        if (!path_exists || force_recompiles > 0) {
            //if (enable_compile_cache)
            //    cout << "======= Compiling" << std::endl;
            // PROF_BEGIN("CCGEN_BIN")
            GenerateBinaries(device, &dummy_op, op_path, profile_kernel, kernel_group, logical_core);
            // PROF_END("CCGEN_BIN")
            force_recompiles = std::max(0, force_recompiles-1);
        } else {
            //if (enable_compile_cache)
            //    cout << "======= Skipping compiling..." << std::endl;
        }
        HashLookup::inst().add(kernel_group_hash);
    }

    tt_metal_profiler.markStop("CompileProgram");
    return pass;
}

void ConfigureKernelGroup(const KernelGroup &kernel_group, Device *device, const tt_xy_pair &logical_core) {
    // No need to check if kernel_group.riscv_0 and kernel_group.riscv_1 are null because compilation
    // creates blank data movement kernels for riscs0/1 if there is no kernel on them
    if (kernel_group.compute != nullptr) {
        kernel_group.compute->configure(device, logical_core);
    }
    kernel_group.riscv_1->configure(device, logical_core);
    kernel_group.riscv_0->configure(device, logical_core);
}



bool ConfigureDeviceWithProgram(Device *device, Program *program) {
    bool pass = true;

    tt_metal_profiler.markStart("ConfigureDeviceWithProgram");
    std::vector<tt_xy_pair> worker_cores;
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    for (auto &[logical_core, kernel_group] : program->core_to_kernel_group()) {
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        worker_cores.push_back(worker_core);

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

        auto cbs_on_core = program->circular_buffers_on_core(logical_core); // PROF_BEGIN("CBS")
        for (auto circular_buffer : cbs_on_core) {
            llrt::set_config_for_circular_buffer(
                circular_buffer_config_vec,
                circular_buffer->buffer_index(),
                circular_buffer->address(),
                circular_buffer->size(),
                circular_buffer->num_tiles());
        } // PROF_END("CBS")

        llrt::write_circular_buffer_config_vector_to_core(cluster, pcie_slot, worker_core, circular_buffer_config_vec); // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")

        auto semaphores_on_core = program->semaphores_on_core(logical_core);
        for (auto semaphore : semaphores_on_core) {
            llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, {semaphore->initial_value()}, semaphore->address());
        }
    }

    // Load blank kernel to all riscs of all cores excluding those in worker_cores
    const llrt::TensixRiscsOptions riscs_options = llrt::TensixRiscsOptions::ALL_RISCS; // PROF_BEGIN("LOAD_BLANK")
    llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, pcie_slot, riscs_options, worker_cores);                               // PROF_END("LOAD_BLANK")

    tt_metal_profiler.markStop("ConfigureDeviceWithProgram");
    return pass;
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const tt_xy_pair &logical_core, const std::vector<uint32_t> &runtime_args) {
    bool pass = true;
    kernel->kernel_args()->set_runtime_args(logical_core, runtime_args);
    kernel->write_runtime_args_to_device(device, logical_core);
    return pass;
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args) {
    TT_ASSERT(core_range.first == core_range.second or core_range.first < core_range.second && "Invalid core range!");
    CoreBlocks core_blocks = {core_range};
    return WriteRuntimeArgsToDevice(device, kernel, core_blocks, {runtime_args});
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &runtime_args_spec) {
    bool pass = true;
    TT_ASSERT(core_blocks.size() == runtime_args_spec.size());
    for (auto index = 0; index < core_blocks.size(); index++) {
        auto core = core_blocks.at(index);
        auto runtime_args = runtime_args_spec.at(index);
        std::visit(overloaded_core {
            [device, kernel, runtime_args](tt_xy_pair single_core) {
                WriteRuntimeArgsToDevice(device, kernel, single_core, runtime_args);
            },
            [device, kernel, runtime_args](CoreRange core_range) {
                auto start_core = core_range.first;
                auto end_core = core_range.second;
                TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
                for (auto x = start_core.x; x <= end_core.x; x++) {
                    for (auto y = start_core.y; y <= end_core.y; y++) {
                        auto core_in_range = tt_xy_pair(x, y);
                        WriteRuntimeArgsToDevice(device, kernel, core_in_range, runtime_args);
                    }
                }
            }
        }, core);
    }
    return pass;
}

bool core_runs_ncrisc(Program *program, const tt_xy_pair &logical_core) {
    auto kernel_group = program->kernels_on_core(logical_core);
    return kernel_group.riscv_1 != nullptr;
}

bool core_runs_triscs(Program *program, const tt_xy_pair &logical_core) {
    auto kernel_group = program->kernels_on_core(logical_core);
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

bool LaunchKernels(Device *device, Program *program, bool stagger_start) {

    tt_metal_profiler.markStart("LaunchKernels");
    bool pass = true;

    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    // Cores have to be enabled before BRISC reset is de-asserted
    auto logical_cores_used_in_program = program->logical_cores();
    auto worker_cores = device->worker_cores_from_logical_cores(logical_cores_used_in_program);
    llrt::internal_::enable_cores(cluster, pcie_slot, worker_cores);  // BRISC FW waits for this enable to run

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

// Copies data from a host buffer into a buffer within the device DRAM channel
bool WriteToDeviceDRAM(DramBuffer *dram_buffer, std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("WriteToDeviceDRAM");
    bool pass = true;
    dram_buffer->device()->cluster()->write_dram_vec(
        host_buffer, tt_target_dram{dram_buffer->device()->pcie_slot(), dram_buffer->dram_channel(), 0}, dram_buffer->address());
    tt_metal_profiler.markStop("WriteToDeviceDRAM");
    return pass;
}

// Copy data from a device DRAM channel to a host buffer
bool ReadFromDeviceDRAM(
    DramBuffer *dram_buffer,
    std::vector<uint32_t> &host_buffer) {
    tt_metal_profiler.markStart("ReadFromDeviceDRAM");
    bool pass = true;
    dram_buffer->device()->cluster()->read_dram_vec(
        host_buffer, tt_target_dram{dram_buffer->device()->pcie_slot(), dram_buffer->dram_channel(), 0}, dram_buffer->address(), dram_buffer->size());
    tt_metal_profiler.markStop("ReadFromDeviceDRAM");
    return pass;
}

// Copies data from a host buffer into the device DRAM channel
bool WriteToDeviceDRAMChannel(
    Device *device, int dram_channel, std::vector<uint32_t> &host_buffer, uint32_t dram_address) {
    tt_metal_profiler.markStart("WriteToDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->write_dram_vec(
        host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, dram_address);
    tt_metal_profiler.markStop("WriteToDeviceDRAMChannel");
    return pass;
}

// Copy data from a device DRAM channel to a host buffer
bool ReadFromDeviceDRAMChannel(
    Device *device,
    int dram_channel,
    uint32_t device_address,
    std::vector<uint32_t> &host_buffer,
    uint32_t size) {
    tt_metal_profiler.markStart("ReadFromDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->read_dram_vec(
        host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, device_address, size);
    tt_metal_profiler.markStop("ReadFromDeviceDRAMChannel");
    return pass;
}

bool ReadFromDeviceDRAMChannelsInterleaved(
    Device *device, std::vector<uint32_t> &host_buffer, uint32_t start_dram_buffer_address,
    int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {

    /*
        Reads interleaved bank units into a host buffer vector. A bank unit is just the unit of data
        that we store in banks round-robbin.
    */
    int dram_channel = 0;
    int dram_addr = start_dram_buffer_address;
    int tensor_index = 0;
    int bank_unit_size = num_entries_per_bank_unit * num_bytes_per_entry;

    bool pass = true;
    for (int s = 0; s < num_bank_units; s++) {
        std::vector<uint32_t> bank_unit;
        pass &= tt_metal::ReadFromDeviceDRAMChannel(device, dram_channel, dram_addr, bank_unit, bank_unit_size);

        // Copy bank unit into vector
        for (uint32_t el: bank_unit) {
            host_buffer.push_back(el);
        }

        dram_channel++;
        if (dram_channel == device->num_dram_channels()) {
            dram_channel = 0;
            dram_addr += bank_unit_size;
        }

        tensor_index += num_entries_per_bank_unit;
    }

    return pass;
}

bool WriteToDeviceDRAMChannelsInterleaved(
    Device *device, std::vector<uint32_t> &host_buffer, uint32_t start_dram_buffer_address,
    int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {

    /*
        Writes a vector into device DRAM interleaved, where the vector is broken up into
        bank units and written to DRAM round-robbin.
    */
    int dram_channel = 0;
    int dram_addr = start_dram_buffer_address;
    int tensor_index = 0;

    bool pass = true;
    for (int s = 0; s < num_bank_units; s++) {
        std::vector<uint32_t> bank_unit;
        bank_unit.insert(bank_unit.end(), host_buffer.begin() + tensor_index, host_buffer.begin() + tensor_index + num_entries_per_bank_unit);

        pass &= tt_metal::WriteToDeviceDRAMChannel(device, dram_channel, bank_unit, dram_addr);

        dram_channel++;
        if (dram_channel == device->num_dram_channels()) {
            dram_channel = 0;
            dram_addr += num_entries_per_bank_unit * num_bytes_per_entry;
        }

        tensor_index += num_entries_per_bank_unit;
    }

    return pass;
}

// Read from interleave tiles from 8 banks starting at a given address (same starting address for each bank)
// See comments for the Write version of this function
bool ReadFromDeviceDRAMChannelsInterleavedTiles(
    Device *device,
    uint32_t device_dram_address,
    std::vector<uint32_t> &dst_host_buffer,
    uint32_t size_bytes) {
    using std::vector;

    const uint32_t tile_bytes = 2048; // TODO(AP): magic for bf16
    dst_host_buffer.resize(0);
    TT_ASSERT(size_bytes % tile_bytes == 0);
    int num_tiles = size_bytes/tile_bytes;
    uint32_t src_address = device_dram_address;
    for (int i_tile = 0; i_tile < num_tiles; i_tile++) {
        uint32_t dram_channel = (i_tile % 8);
        if ((i_tile > 0) && ((i_tile % 8) == 0))
            src_address += tile_bytes;
        vector<uint32_t> onetile;
        device->cluster()->read_dram_vec(
            onetile, tt_target_dram{device->pcie_slot(), dram_channel, 0}, src_address, tile_bytes);
        dst_host_buffer.insert(dst_host_buffer.end(), onetile.begin(), onetile.end());
    }
    TT_ASSERT(size_bytes == dst_host_buffer.size()*sizeof(dst_host_buffer[0]));
    return true;
}

// Interleave tiles into 8 banks starting at a given address (same starting address for each bank)
// Each write is tile-sized, so performance is probably not ideal.
// This can probably be made more optimal with strided or chain or async DMAs or whatnot
bool WriteToDeviceDRAMChannelsInterleavedTiles(
    Device *device,
    std::vector<uint32_t> &host_buffer,
    uint32_t dram_address) {

    using std::vector;

    const uint32_t tile_bytes = 2048; // TODO(AP): magic for bf16
    const uint32_t tile_u32s = tile_bytes/sizeof(uint32_t);
    int num_tiles = host_buffer.size()*sizeof(uint32_t);
    TT_ASSERT(num_tiles % tile_bytes == 0);
    num_tiles /= tile_bytes;
    uint32_t dst_address = dram_address;
    for (int i_tile = 0; i_tile < num_tiles; i_tile++) {
        vector<uint32_t> tile = vector<uint32_t>(
            host_buffer.begin()+(i_tile+0)*tile_u32s,
            host_buffer.begin()+(i_tile+1)*tile_u32s
        );
        uint32_t dram_channel = (i_tile % 8);
        if ((i_tile > 0) && ((i_tile % 8) == 0))
            dst_address += tile_bytes;
        device->cluster()->write_dram_vec(
            tile, tt_target_dram{device->pcie_slot(), dram_channel, 0}, dst_address);
    }
    return true;
}

void ReadFromDeviceDRAMChannelsInterleaved(InterleavedDramBuffer *buffer, std::vector<uint32_t> &host_buffer) {
    uint32_t bank_unit_size = buffer->bank_unit_size();
    TT_ASSERT(buffer->size() % bank_unit_size == 0);
    uint32_t num_bank_units = buffer->size() / bank_unit_size;

    int bank_index = 0;
    for (int bank_unit_index = 0; bank_unit_index < num_bank_units; bank_unit_index++) {
        auto dram_bank = buffer->bank(bank_index);
        auto absolute_address = buffer->address_of_bank_unit(bank_index, bank_unit_index);
        std::vector<uint32_t> bank_unit;
        tt_metal::ReadFromDeviceDRAMChannel(buffer->device(), dram_bank.channel, absolute_address, bank_unit, bank_unit_size);

        // Copy bank unit into vector
        for (uint32_t el: bank_unit) {
            host_buffer.push_back(el);
        }

        bank_index = (bank_index + 1) %  buffer->num_banks();
    }
}

void WriteToDeviceDRAMChannelsInterleaved(InterleavedDramBuffer *buffer, std::vector<uint32_t> &host_buffer) {
    uint32_t bank_unit_size = buffer->bank_unit_size();
    TT_ASSERT(buffer->size() % bank_unit_size == 0);
    uint32_t num_bank_units = buffer->size() / bank_unit_size;
    uint32_t num_entries_per_bank_unit = buffer->num_entries_per_bank_unit();

    int bank_index = 0;
    int data_index = 0;
    for (int bank_unit_index = 0; bank_unit_index < num_bank_units; bank_unit_index++) {
        auto dram_bank = buffer->bank(bank_index);
        auto absolute_address = buffer->address_of_bank_unit(bank_index, bank_unit_index);
        std::vector<uint32_t> bank_unit;
        bank_unit.insert(bank_unit.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_bank_unit);

        tt_metal::WriteToDeviceDRAMChannel(buffer->device(), dram_bank.channel, bank_unit, absolute_address);

        bank_index = (bank_index + 1) %  buffer->num_banks();
        data_index += num_entries_per_bank_unit;
    }
}

void ReadFromDeviceL1Interleaved(InterleavedL1Buffer *buffer, std::vector<uint32_t> &host_buffer) {
    uint32_t bank_unit_size = buffer->bank_unit_size();
    TT_ASSERT(buffer->size() % bank_unit_size == 0);
    uint32_t num_bank_units = buffer->size() / bank_unit_size;

    int bank_index = 0;
    for (int bank_unit_index = 0; bank_unit_index < num_bank_units; bank_unit_index++) {
        auto l1_bank = buffer->bank(bank_index);
        auto absolute_address = buffer->address_of_bank_unit(bank_index, bank_unit_index);

        std::vector<uint32_t> bank_unit;
        tt_metal::ReadFromDeviceL1(buffer->device(), l1_bank.logical_core, absolute_address, bank_unit, bank_unit_size);

        // Copy bank unit into vector
        for (uint32_t el: bank_unit) {
            host_buffer.push_back(el);
        }

        bank_index = (bank_index + 1) %  buffer->num_banks();
    }
}

void WriteToDeviceL1Interleaved(InterleavedL1Buffer *buffer, std::vector<uint32_t> &host_buffer) {
    auto bank_unit_size = buffer->bank_unit_size();
    TT_ASSERT(buffer->size() % bank_unit_size == 0);
    auto num_bank_units = buffer->size() / bank_unit_size;
    auto num_entries_per_bank_unit = buffer->num_entries_per_bank_unit();

    int bank_index = 0;
    int data_index = 0;
    for (int bank_unit_index = 0; bank_unit_index < num_bank_units; bank_unit_index++) {
        auto l1_bank = buffer->bank(bank_index);
        auto absolute_address = buffer->address_of_bank_unit(bank_index, bank_unit_index);

        std::vector<uint32_t> bank_unit;
        bank_unit.insert(bank_unit.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_bank_unit);

        tt_metal::WriteToDeviceL1(buffer->device(), l1_bank.logical_core, bank_unit, absolute_address);

        bank_index = (bank_index + 1) %  buffer->num_banks();
        data_index += num_entries_per_bank_unit;
    }
}

bool WriteToDeviceL1(
    Device *device,
    const tt_xy_pair &core,
    std::vector<uint32_t> &host_buffer,
    uint32_t buffer_address) {
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    llrt::write_hex_vec_to_core(
        device->cluster(), device->pcie_slot(), worker_core, host_buffer, buffer_address);
    return pass;
}

bool WriteToDeviceL1(L1Buffer *l1_buffer, std::vector<uint32_t> &host_buffer) {
    bool pass = true;
    llrt::write_hex_vec_to_core(
        l1_buffer->device()->cluster(), l1_buffer->device()->pcie_slot(), l1_buffer->noc_coordinates(), host_buffer, l1_buffer->address());
    return pass;
}

bool WriteToDeviceL1(
    Device *device,
    const tt_xy_pair &core,
    op_info_t op_info,
    int op_idx) {
    int pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    llrt::write_graph_interpreter_op_info_to_core(
        device->cluster(), device->pcie_slot(), worker_core, op_info, op_idx);
    return pass;
}

bool ReadFromDeviceL1(
    Device *device, const tt_xy_pair &core, int device_buffer_addess, std::vector<uint32_t> &host_buffer, int size) {
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    host_buffer =
        llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), worker_core, device_buffer_addess, size);
    return pass;
}

bool ReadFromDeviceL1(L1Buffer *l1_buffer, std::vector<uint32_t> &host_buffer) {
    bool pass = true;
    host_buffer = llrt::read_hex_vec_from_core(
            l1_buffer->device()->cluster(), l1_buffer->device()->pcie_slot(), l1_buffer->noc_coordinates(), l1_buffer->address(), l1_buffer->size());
    return pass;
}

}  // namespace tt_metal

}  // namespace tt
