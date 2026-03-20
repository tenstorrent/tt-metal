// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/types.hpp"
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cluster.hpp"
#include "ttnn/reports.hpp"
#include <boost/algorithm/string/replace.hpp>
#include <cstdlib>

#include <fstream>
#include <sstream>
#include <enchantum/enchantum.hpp>
#include <memory>
#include <string>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <unordered_map>

using namespace tt::tt_metal;

namespace {

std::string tensorMemoryLayoutToString(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return "INTERLEAVED";
        case TensorMemoryLayout::HEIGHT_SHARDED: return "HEIGHT_SHARDED";
        case TensorMemoryLayout::WIDTH_SHARDED: return "WIDTH_SHARDED";
        case TensorMemoryLayout::BLOCK_SHARDED: return "BLOCK_SHARDED";
        default: return "UNKNOWN";  // Handle unexpected values
    }
}

nlohmann::json to_json(const ttnn::graph::GraphProcessor::Vertex& data) {
    nlohmann::json j;
    j[ttnn::graph::kCounter] = data.counter;
    j[ttnn::graph::kNodeType] = data.node_type;

    static const std::unordered_set<std::string> integer_params = {
        ttnn::graph::kSize,
        ttnn::graph::kAddress,
        ttnn::graph::kTensorId,
        ttnn::graph::kDeviceId,
        ttnn::graph::kBufferTypeValue,
        ttnn::graph::kPageSize,
        ttnn::graph::kNumCores,
        ttnn::graph::kMaxSizePerBank,
    };
    nlohmann::json params_json;
    for (const auto& [key, value] : data.params) {
        if (integer_params.contains(key)) {
            params_json[key] = std::stoll(value);
        } else {
            params_json[key] = value;
        }
    }
    j[ttnn::graph::kParams] = std::move(params_json);

    j[ttnn::graph::kArguments] = data.arguments;
    j[ttnn::graph::kConnections] = data.connections;
    j[ttnn::graph::kInputTensors] = data.input_tensors;
    j[ttnn::graph::kStackingLevel] = data.stacking_level;

    // Include duration_ns for function_end and capture_end nodes
    if ((data.node_type == ttnn::graph::kNodeFunctionEnd || data.node_type == ttnn::graph::kNodeCaptureEnd) &&
        data.duration_ns > 0) {
        j[ttnn::graph::kDurationNs] = data.duration_ns;
    }

    return j;
}

nlohmann::json to_json(const std::vector<ttnn::graph::GraphProcessor::Vertex>& data) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& item : data) {
        j.push_back(to_json(item));
    }
    return j;
}

uint64_t current_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Get cluster descriptor YAML content
// Note: Only called during report generation (offline), not on hot path
std::string get_cluster_descriptor_content() {
    std::string descriptor_path = ttnn::cluster::serialize_cluster_descriptor();
    if (descriptor_path.empty()) {
        return "";
    }
    std::ifstream file(descriptor_path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Get mesh coordinate mapping from generated/fabric/ directory
// This matches the old behavior of save_mesh_descriptor() which copied these YAML files
// Note: Only called during report generation (offline), not on hot path
std::string get_mesh_coordinate_mapping_content() {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    if (tt_metal_home == nullptr) {
        log_warning(tt::LogAlways, "TT_METAL_HOME not set, mesh coordinate mapping will be missing from report");
        return "";
    }

    std::filesystem::path fabric_dir = std::filesystem::path(tt_metal_home) / "generated" / "fabric";
    if (!std::filesystem::exists(fabric_dir)) {
        return "";
    }

    // Collect all physical_chip_mesh_coordinate_mapping*.yaml files
    std::string combined_content;
    for (const auto& entry : std::filesystem::directory_iterator(fabric_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find("physical_chip_mesh_coordinate_mapping") != std::string::npos &&
                filename.find(".yaml") != std::string::npos) {
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    if (!combined_content.empty()) {
                        combined_content += "\n---\n";  // YAML document separator
                    }
                    combined_content += "# " + filename + "\n";
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    combined_content += buffer.str();
                }
            }
        }
    }
    return combined_content;
}

}  // namespace

namespace ttnn::graph {

std::atomic<bool> GraphProcessor::capture_detailed_buffer_tracing_{false};

void GraphProcessor::enable_detailed_buffer_tracing() { capture_detailed_buffer_tracing_ = true; }

void GraphProcessor::disable_detailed_buffer_tracing() { capture_detailed_buffer_tracing_ = false; }

bool GraphProcessor::is_detailed_buffer_tracing_enabled() { return capture_detailed_buffer_tracing_; }

GraphProcessor::GraphProcessor(RunMode mode) : run_mode(mode) { GraphProcessor::begin_capture(mode); }

void GraphProcessor::track_device(const tt::tt_metal::IDevice* device) {
    if (device == nullptr) {
        return;
    }
    auto device_id = device->id();
    if (captured_device_info.contains(device_id)) {
        return;  // Already captured
    }

    // Capture device info now - the device may not be available when report is generated
    const auto& allocator = device->allocator();
    auto compute_grid = device->compute_with_storage_grid_size();
    auto logical_grid = device->logical_grid_size();

    size_t num_x_compute_cores = compute_grid.x;
    size_t num_y_compute_cores = compute_grid.y;
    size_t num_compute_cores = num_x_compute_cores * num_y_compute_cores;
    size_t num_storage_cores = device->storage_only_cores().size();
    size_t worker_l1_size = allocator->get_worker_l1_size();
    size_t l1_num_banks = allocator->get_num_banks(tt::tt_metal::BufferType::L1);
    size_t l1_bank_size = allocator->get_bank_size(tt::tt_metal::BufferType::L1);

    nlohmann::json info;
    info[kDeviceId] = device_id;
    info[kDeviceNumYCores] = logical_grid.y;
    info[kDeviceNumXCores] = logical_grid.x;
    info[kDeviceNumYComputeCores] = num_y_compute_cores;
    info[kDeviceNumXComputeCores] = num_x_compute_cores;
    info[kDeviceWorkerL1Size] = worker_l1_size;
    info[kDeviceL1NumBanks] = l1_num_banks;
    info[kDeviceL1BankSize] = l1_bank_size;
    info[kDeviceAddressAtFirstL1Bank] = allocator->get_bank_offset(tt::tt_metal::BufferType::L1, 0);
    info[kDeviceAddressAtFirstL1CbBuffer] = allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    info[kDeviceNumBanksPerStorageCore] = l1_bank_size > 0 ? worker_l1_size / l1_bank_size : 0;
    info[kDeviceNumComputeCores] = num_compute_cores;
    info[kDeviceNumStorageCores] = num_storage_cores;
    info[kDeviceTotalL1Memory] = (num_storage_cores + num_compute_cores) * worker_l1_size;
    info[kDeviceTotalL1ForTensors] = 0;  // Not computed in original implementation
    info[kDeviceTotalL1ForInterleavedBuffers] =
        (num_storage_cores + num_compute_cores +
         (l1_bank_size > 0 ? (worker_l1_size / l1_bank_size) * num_storage_cores : 0)) *
        l1_bank_size;
    info[kDeviceTotalL1ForShardedBuffers] = num_compute_cores * l1_bank_size;
    info[kDeviceCbLimit] = worker_l1_size - allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Extra useful info
    info["arch"] = enchantum::to_string(device->arch());
    info["num_dram_channels"] = device->num_dram_channels();
    info["dram_size_per_channel"] = device->dram_size_per_channel();

    captured_device_info[device_id] = std::move(info);

    // Store device pointer for buffer page capture (MeshDevice extends IDevice)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* mesh_device = const_cast<tt::tt_metal::distributed::MeshDevice*>(
        dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device));
    if (mesh_device != nullptr) {
        captured_mesh_devices.push_back(mesh_device);
    }
}

void GraphProcessor::track_allocate(const tt::tt_metal::Buffer* buffer) {
    const std::lock_guard<std::mutex> lock(mutex);

    // Track the device for later device info collection
    track_device(buffer->device());

    node_id buffer_node_id = add_buffer(buffer);

    node_id counter = graph.size();
    int stacking_level = static_cast<int>(current_op_id.size()) - 1;

    // Compute max_size_per_bank: the maximum allocation footprint per bank for this
    // buffer.  For interleaved layouts we use the real bank count from the allocator
    // (important for L1_SMALL which has fewer banks than L1).  For sharded layouts
    // we use the per-core page spread which matches the allocator's distribution for
    // all standard shard specs.
    uint32_t num_pages = buffer->num_pages();
    uint32_t page_sz = buffer->page_size();
    uint32_t max_size_per_bank;
    if (tt::tt_metal::is_sharded(buffer->buffer_layout())) {
        uint32_t nc = buffer->num_cores().value_or(1);
        uint32_t pages_per_core = nc > 0 ? (num_pages + nc - 1) / nc : num_pages;
        max_size_per_bank = pages_per_core * page_sz;
    } else {
        uint32_t num_banks = buffer->device()->allocator()->get_num_banks(buffer->buffer_type());
        uint32_t pages_per_bank = num_banks > 0 ? (num_pages + num_banks - 1) / num_banks : num_pages;
        max_size_per_bank = pages_per_bank * page_sz;
    }

    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kAddress, std::to_string(buffer->address())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kExactBufferType, std::string(enchantum::to_string(buffer->buffer_type()))},
        {kBufferTypeValue, std::to_string(static_cast<int>(buffer->buffer_type()))},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kPageSize, std::to_string(buffer->page_size())},
        {kNumCores, std::to_string(buffer->num_cores().value_or(0))},  // use 0 for interleaved
        {kDeviceId, std::to_string(buffer->device()->id())},
        {kMaxSizePerBank, std::to_string(max_size_per_bank)}};
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeBufferAllocate,
            .params = std::move(params),
            .connections = {buffer_node_id},
            .stacking_level = stacking_level});
        graph[current_op_id.top()].connections.push_back(counter);
    }

    // Capture buffer pages for this address on every allocation.
    // The same address may be re-allocated with a different page configuration
    // (e.g. fold at ps=448 then max_pool2d at ps=128).  Each allocation is
    // recorded with its graph counter so the importer can pick the right version.
    uint64_t addr = buffer->address();
    if (capture_detailed_buffer_tracing_ && !captured_mesh_devices.empty()) {
        auto all_pages = ttnn::reports::get_buffer_pages(captured_mesh_devices);
        std::vector<ttnn::reports::BufferPageInfo> pages_for_addr;
        for (const auto& page : all_pages) {
            if (page.address == addr) {
                pages_for_addr.push_back(page);
            }
        }
        if (!pages_for_addr.empty()) {
            buffer_pages_by_address_[addr].emplace_back(counter, std::move(pages_for_addr));
        }
    }
}

void GraphProcessor::track_deallocate(tt::tt_metal::Buffer* buffer) {
    const std::lock_guard<std::mutex> lock(mutex);

    // Track the device for later device info collection
    track_device(buffer->device());

    node_id buffer_node_id = add_buffer(buffer);
    node_id counter = graph.size();
    int stacking_level = static_cast<int>(current_op_id.size()) - 1;
    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kAddress, std::to_string(buffer->address())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kExactBufferType, std::string(enchantum::to_string(buffer->buffer_type()))},
        {kBufferTypeValue, std::to_string(static_cast<int>(buffer->buffer_type()))},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kPageSize, std::to_string(buffer->page_size())},
        {kNumCores, std::to_string(buffer->num_cores().value_or(0))},  // use 0 for interleaved
        {kDeviceId, std::to_string(buffer->device()->id())}};
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeBufferDeallocate,
            .params = std::move(params),
            .connections = {buffer_node_id},
            .stacking_level = stacking_level});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_allocate_cb(
    const CoreRangeSet& core_range_set,
    uint64_t addr,
    uint64_t size,
    bool is_globally_allocated,
    const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);
    const std::lock_guard<std::mutex> lock(mutex);

    // Track the device for later device info collection
    track_device(device);

    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(size)},
        {kAddress, std::to_string(addr)},
        {kCoreRangeSet, core_range_set.str()},
        {kGloballyAllocated, std::to_string(is_globally_allocated)},
        {kDeviceId, std::to_string(device->id())}};
    node_id counter = graph.size();
    int stacking_level = static_cast<int>(current_op_id.size()) - 1;
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeCBAllocate,
            .params = std::move(params),
            .connections = {},
            .stacking_level = stacking_level});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_deallocate_cb(const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);
    const std::lock_guard<std::mutex> lock(mutex);

    // Track the device for later device info collection
    track_device(device);

    node_id counter = graph.size();
    int stacking_level = static_cast<int>(current_op_id.size()) - 1;
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeCBDeallocateAll,
            .params = {{kDeviceId, std::to_string(device->id())}},
            .connections = {current_op_id.top()},
            .stacking_level = stacking_level});
        graph[current_op_id.top()].connections.push_back(counter);
    }
}

void GraphProcessor::track_program(tt::tt_metal::Program* program, const tt::tt_metal::IDevice* device) {
    TT_ASSERT(device);

    // All previous CBs are deallocated before a new program run
    track_deallocate_cb(device);

    if (run_mode == RunMode::NORMAL) {
        // we will track real buffer allocations during program run
        return;
    }

    for (const auto& cb : program->circular_buffers()) {
        track_allocate_cb(cb->core_ranges(), 0, cb->size(), cb->globally_allocated(), device);
    }
}

template <typename T>
using ProcessFunc = void (GraphProcessor::*)(const T&);

template <typename T, ProcessFunc<T> Process>
static void process(GraphProcessor& self, const std::any& any_val) {
    (self.*Process)(std::any_cast<std::reference_wrapper<T>>(any_val).get());
}

template <typename T, ProcessFunc<T> Process>
consteval std::pair<const std::type_info&, void (*)(GraphProcessor&, const std::any&)> make_process() {
    return {typeid(std::reference_wrapper<T>), &process<T, Process>};
}

void GraphProcessor::track_function_start(
    std::string_view function_name, std::span<tt::tt_metal::TrackedArgument> input_parameters) {
    static constexpr std::array begin_function_any_map = {
        make_process<std::vector<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<std::vector<std::optional<Tensor>>, &GraphProcessor::begin_function_process>(),
        make_process<std::vector<std::optional<const Tensor>>, &GraphProcessor::begin_function_process>(),
        make_process<std::vector<std::reference_wrapper<const Tensor>>, &GraphProcessor::begin_function_process>(),
        make_process<Tensor, &GraphProcessor::begin_function_process>(),
        make_process<const Tensor, &GraphProcessor::begin_function_process>(),
        make_process<std::optional<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<const std::optional<Tensor>, &GraphProcessor::begin_function_process>(),
        make_process<std::optional<const Tensor>, &GraphProcessor::begin_function_process>(),
    };

    // Record start time BEFORE acquiring lock to minimize overhead in timing measurement
    auto start_time = std::chrono::steady_clock::now();

    const std::lock_guard<std::mutex> lock(mutex);
    log_debug(tt::LogAlways, "Begin op: {}", function_name);

    // Push start time onto stack for duration calculation
    function_start_times.push(start_time);

    // Clear the input tensor list for this new operation
    current_input_tensors.clear();

    std::unordered_map<std::string, std::string> params = {
        {kInputs, std::to_string(input_parameters.size())},
        {kName, std::string(function_name)},
    };

    std::vector<std::string> serialized_arguments;
    serialized_arguments.reserve(input_parameters.size());
    for (const auto& arg : input_parameters) {
        auto str = arg.to_string_fn(arg.value);
        boost::algorithm::replace_all(str, "__1::", "");
        serialized_arguments.push_back(std::move(str));
    }

    node_id counter = graph.size();
    // Track stacking level: current stack depth (before pushing this operation)
    int stacking_level = static_cast<int>(current_op_id.size());

    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeFunctionStart,
            .params = std::move(params),
            .arguments = serialized_arguments,
            .connections = {/*current_op_id.top()*/},
            .input_tensors = {},
            .stacking_level = stacking_level,
            .duration_ns = 0});
        if (last_finished_op_id != -1) {
            graph[last_finished_op_id].connections.push_back(counter);
            last_finished_op_id = -1;
        }
        graph[current_op_id.top()].connections.push_back(counter);
        current_op_id.push(counter);
    }

    for (const auto& tracked_arg : input_parameters) {
        const auto* const it =
            std::ranges::find(begin_function_any_map, tracked_arg.value.type(), [](const auto& pair) -> const auto& {
                return pair.first;
            });

        if (it != begin_function_any_map.end()) {
            it->second(*this, tracked_arg.value);
        } else {
            log_debug(tt::LogAlways, "input any type name ignored: {}", tracked_arg.value.type().name());
        }
    }

    // Populate the input_tensors field of the function_start vertex
    graph[counter].input_tensors = current_input_tensors;
}

void GraphProcessor::track_function_end_impl() {
    // Calculate duration - get end time first for accuracy
    uint64_t duration_ns = 0;
    if (!function_start_times.empty()) {
        auto end_time = std::chrono::steady_clock::now();
        auto start_time = function_start_times.top();
        function_start_times.pop();
        duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

    auto name = graph[current_op_id.top()].params[kName];
    log_debug(tt::LogAlways, "End op: {} (duration: {} ns)", name, duration_ns);

    node_id function_start_id = current_op_id.top();
    int stacking_level = graph[function_start_id].stacking_level;

    node_id counter = graph.size();
    {
        graph.push_back(Vertex{
            .counter = counter,
            .node_type = kNodeFunctionEnd,
            .params = {{kName, name}},
            .connections = {},
            .stacking_level = stacking_level,
            .duration_ns = duration_ns});
        graph[current_op_id.top()].connections.push_back(counter);
    }
    last_finished_op_id = counter;

    // Snapshot live buffer state after each top-level operation completes
    if (stacking_level == 1 && !captured_mesh_devices.empty()) {
        per_op_buffers_[function_start_id] = ttnn::reports::get_buffers(captured_mesh_devices);
    }
}

void GraphProcessor::track_function_end() {
    const std::lock_guard<std::mutex> lock(mutex);
    this->track_function_end_impl();
    TT_ASSERT(!current_op_id.empty());  // we should always have capture_start on top
    current_op_id.pop();
}

void GraphProcessor::track_function_end(const std::any& output_tensors) {
    static constexpr std::array end_function_any_map{
        make_process<std::vector<Tensor>, &GraphProcessor::end_function_process>(),
        make_process<std::vector<std::optional<Tensor>>, &GraphProcessor::end_function_process>(),
        make_process<std::vector<std::optional<const Tensor>>, &GraphProcessor::end_function_process>(),
        make_process<Tensor, &GraphProcessor::end_function_process>(),
    };

    const std::lock_guard<std::mutex> lock(mutex);
    this->track_function_end_impl();

    const auto* const it = std::ranges::find(
        end_function_any_map, output_tensors.type(), [](const auto& pair) -> const auto& { return pair.first; });

    if (it != end_function_any_map.end()) {
        it->second(*this, output_tensors);
    } else {
        log_debug(tt::LogAlways, "output any type name ignored: {}", output_tensors.type().name());
    }
    TT_ASSERT(!current_op_id.empty());  // we should always have capture_start on top
    current_op_id.pop();
}

node_id GraphProcessor::add_tensor(const Tensor& t) {
    tt::tt_metal::Buffer* buffer = nullptr;
    nlohmann::json device_tensors_json = nlohmann::json::array();
    if (is_device_tensor(t) && t.is_allocated()) {
        const auto& mesh_buffer = t.mesh_buffer();
            // `t.buffers()` returns a reference buffer allocated on first device in a mesh.
            // It has an ID different from the "backing" buffer that was used to perform the allocation.
            // To deduplicate an entry for this buffer, captured during its allocation, use the "backing"
            // buffer.
        buffer = mesh_buffer->get_backing_buffer();
        for (const auto& coord : t.device_storage().get_coords()) {
            auto* device_buffer = mesh_buffer->get_device_buffer(coord);
            if (device_buffer != nullptr) {
                device_tensors_json.push_back(
                    {{"device_id", device_buffer->device()->id()},
                     {kMeshDeviceId, buffer != nullptr ? buffer->device()->id() : device_buffer->device()->id()},
                     {"address", device_buffer->address()}});
            }
        }
    }

    node_id tensor_counter = graph.size();
    auto shape = t.logical_shape();

    std::unordered_map<std::string, std::string> params = {
        {kShape, fmt::format("{}", shape)},
        {kTensorId, fmt::format("{}", t.tensor_id)},
        {kDtype, fmt::format("{}", t.dtype())},
        {kLayout, fmt::format("{}", t.layout())},
        {kSize, std::to_string(t.logical_volume() * t.element_size())},
    };

    // Add memory config if tensor is on device
    if (t.is_allocated() && t.storage_type() == StorageType::DEVICE) {
        params[kMemoryConfig] = fmt::format("{}", t.memory_config());
    }

    // Add buffer-related info (primary device for single-device tensors)
    if (buffer != nullptr) {
        params[kDeviceId] = std::to_string(buffer->device()->id());
        params[kAddress] = std::to_string(buffer->address());
        params[kBufferType] = fmt::format("{}", buffer->buffer_type());
        params[kBufferTypeValue] = std::to_string(static_cast<int>(buffer->buffer_type()));
    }

    if (!device_tensors_json.empty()) {
        params[kDeviceTensors] = device_tensors_json.dump();
    }

    int stacking_level = static_cast<int>(current_op_id.size()) - 1;
    graph.push_back(Vertex{
        .counter = tensor_counter,
        .node_type = kNodeTensor,
        .params = std::move(params),
        .connections = {},
        .stacking_level = stacking_level});

    if (buffer == nullptr) {
        log_debug(tt::LogAlways, "Tensor doesn't have buffer, but storage is {}", t.storage_type());
    } else {
        node_id buffer_node_id = add_buffer(buffer);
        graph[buffer_node_id].connections.push_back(tensor_counter);
    }

    return tensor_counter;
}

node_id GraphProcessor::add_buffer(const tt::tt_metal::Buffer* buffer) {
    const auto buffer_unique_id = buffer->unique_id();

    if (const auto it = buffer_id_to_counter.find(buffer_unique_id); it != buffer_id_to_counter.end()) {
        return it->second;
    }

    const node_id counter = graph.size();
    int stacking_level = static_cast<int>(current_op_id.size()) - 1;
    std::unordered_map<std::string, std::string> params = {
        {kSize, std::to_string(buffer->size())},
        {kType, buffer->is_dram() ? "DRAM" : "L1"},
        {kExactBufferType, std::string(enchantum::to_string(buffer->buffer_type()))},
        {kBufferTypeValue, std::to_string(static_cast<int>(buffer->buffer_type()))},
        {kLayout, tensorMemoryLayoutToString(buffer->buffer_layout())},
        {kDeviceId, std::to_string(buffer->device()->id())}};

    graph.push_back(Vertex{
        .counter = counter,
        .node_type = kNodeBuffer,
        .params = std::move(params),
        .connections = {},
        .stacking_level = stacking_level});
    graph[current_op_id.top()].connections.push_back(counter);
    buffer_id_to_counter.emplace(buffer_unique_id, counter);
    return counter;
}

void GraphProcessor::begin_function_process(const Tensor& tensor) {
    node_id tensor_node_id = add_tensor(tensor);
    graph[tensor_node_id].connections.push_back(current_op_id.top());
    current_input_tensors.push_back(tensor_node_id);
}

void GraphProcessor::begin_function_process(const std::reference_wrapper<const Tensor>& tensor_ref) {
    begin_function_process(tensor_ref.get());
}

template <typename T>
void GraphProcessor::begin_function_process(const std::optional<T>& tensor_opt) {
    if (tensor_opt.has_value()) {
        begin_function_process(*tensor_opt);
    }
}

template <typename T>
void GraphProcessor::begin_function_process(const std::vector<T>& tensor_vec) {
    for (const auto& it : tensor_vec) {
        begin_function_process(it);
    }
}

void GraphProcessor::end_function_process(const Tensor& tensor) {
    node_id tensor_node_id = add_tensor(tensor);
    graph[last_finished_op_id].connections.push_back(tensor_node_id);
}

template <typename T>
void GraphProcessor::end_function_process(const std::optional<T>& tensor_opt) {
    if (tensor_opt.has_value()) {
        end_function_process(*tensor_opt);
    }
}

template <typename T>
void GraphProcessor::end_function_process(const std::vector<T>& tensor_vec) {
    for (const auto& it : tensor_vec) {
        end_function_process(it);
    }
}

void GraphProcessor::begin_capture(RunMode mode) {
    const std::lock_guard<std::mutex> lock(mutex);
    graph.clear();
    buffer_id_to_counter.clear();
    captured_device_info.clear();
    captured_mesh_devices.clear();
    per_op_buffers_.clear();
    buffer_pages_by_address_.clear();

    function_start_times = {};

    // Record capture start time
    capture_start_time = std::chrono::steady_clock::now();
    capture_start_timestamp_ns = current_time_ns();

    graph.push_back(
        Vertex{.counter = 0, .node_type = kNodeCaptureStart, .params = {}, .connections = {}, .stacking_level = 0});

    if (!tt::tt_metal::GraphTracker::instance().get_hook()) {
        hook = std::make_shared<ProcessorHooks>();
        tt::tt_metal::GraphTracker::instance().add_hook(hook);
        hook->set_block(mode == RunMode::NO_DISPATCH);
    }
    current_op_id.push(0);
}
nlohmann::json GraphProcessor::end_capture() {
    const std::lock_guard<std::mutex> lock(mutex);

    // Calculate total capture duration
    auto capture_end_time = std::chrono::steady_clock::now();
    uint64_t total_duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(capture_end_time - capture_start_time).count();

    node_id counter = graph.size();
    graph.push_back(Vertex{
        .counter = counter,
        .node_type = kNodeCaptureEnd,
        .params = {},
        .connections = {},
        .stacking_level = 0,
        .duration_ns = total_duration_ns});

    if (last_finished_op_id != -1) {
        graph[last_finished_op_id].connections.push_back(counter);
    } else {
        // lets connect capture_start with capture_end
        // it means we didn't capture any functions
        TT_ASSERT(
            !current_op_id.empty(),
            "Graph size cannot be 0. This means that track_function_end was called more than begin.");
        graph[0].connections.push_back(counter);
    }
    clean_hook();
    return to_json(graph);
}

nlohmann::json GraphProcessor::get_report() const {
    nlohmann::json report;

    // Version for forward compatibility
    report[kReportVersion] = kCurrentReportVersion;

    // Graph trace
    report[kReportGraph] = to_json(graph);

    // Device info captured during trace
    nlohmann::json devices = nlohmann::json::array();
    for (const auto& [device_id, device_info] : captured_device_info) {
        devices.push_back(device_info);
    }
    report[kReportDevices] = devices;

    // Metadata
    nlohmann::json metadata;
    metadata[kReportTimestampNs] = capture_start_timestamp_ns;

    // Calculate total duration if capture has ended (capture_end node exists)
    if (!graph.empty() && graph.back().node_type == kNodeCaptureEnd) {
        metadata[kReportTotalDurationNs] = graph.back().duration_ns;
    }

    report[kReportMetadata] = metadata;

    // Cluster descriptor (YAML content) - always try, returns empty if unavailable
    std::string cluster_desc = get_cluster_descriptor_content();
    if (!cluster_desc.empty()) {
        report["cluster_descriptor"] = cluster_desc;
    }

    // Mesh coordinate mapping (from generated/fabric/*.yaml) - matches old behavior
    std::string mesh_mapping = get_mesh_coordinate_mapping_content();
    if (!mesh_mapping.empty()) {
        report["mesh_coordinate_mapping"] = mesh_mapping;
    }

    // Buffer pages (when detailed buffer report is enabled)
    if (capture_detailed_buffer_tracing_ && !captured_mesh_devices.empty()) {
        auto buffer_pages = ttnn::reports::get_buffer_pages(captured_mesh_devices);
        nlohmann::json buffer_pages_json = nlohmann::json::array();
        for (const auto& page : buffer_pages) {
            buffer_pages_json.push_back(
                {{"device_id", page.device_id},
                 {"address", page.address},
                 {"core_y", page.core_y},
                 {"core_x", page.core_x},
                 {"bank_id", page.bank_id},
                 {"page_index", page.page_index},
                 {"page_address", page.page_address},
                 {"page_size", page.page_size},
                 {"buffer_type", static_cast<int>(page.buffer_type)}});
        }
        report["buffer_pages"] = buffer_pages_json;
    }

    // Per-operation buffer snapshots from get_buffers()
    if (!per_op_buffers_.empty()) {
        nlohmann::json per_op_json = nlohmann::json::object();
        for (const auto& [op_counter, buffers] : per_op_buffers_) {
            nlohmann::json bufs_json = nlohmann::json::array();
            for (const auto& buf : buffers) {
                bufs_json.push_back(
                    {{"device_id", buf.device_id},
                     {"address", buf.address},
                     {"max_size_per_bank", buf.max_size_per_bank},
                     {"buffer_type", static_cast<int>(buf.buffer_type)},
                     {"buffer_layout", static_cast<int>(buf.buffer_layout)}});
            }
            per_op_json[std::to_string(op_counter)] = std::move(bufs_json);
        }
        report["per_operation_buffers"] = std::move(per_op_json);
    }

    // Buffer pages keyed by address, versioned by allocation counter.
    if (!buffer_pages_by_address_.empty()) {
        nlohmann::json bp_json = nlohmann::json::object();
        for (const auto& [addr, snapshots] : buffer_pages_by_address_) {
            nlohmann::json snaps_json = nlohmann::json::array();
            for (const auto& [alloc_counter, pages] : snapshots) {
                nlohmann::json pages_json = nlohmann::json::array();
                for (const auto& page : pages) {
                    pages_json.push_back(
                        {{"device_id", page.device_id},
                         {"address", page.address},
                         {"core_y", page.core_y},
                         {"core_x", page.core_x},
                         {"bank_id", page.bank_id},
                         {"page_index", page.page_index},
                         {"page_address", page.page_address},
                         {"page_size", page.page_size},
                         {"buffer_type", static_cast<int>(page.buffer_type)}});
                }
                snaps_json.push_back({{"alloc_counter", alloc_counter}, {"pages", std::move(pages_json)}});
            }
            bp_json[std::to_string(addr)] = std::move(snaps_json);
        }
        report["buffer_pages_by_address"] = std::move(bp_json);
    }

    return report;
}

void GraphProcessor::clean_hook() {
    if (hook) {
        // If we installed hooks then we must clean
        hook = nullptr;
        tt::tt_metal::GraphTracker::instance().clear_hook();
    }
}

GraphProcessor::~GraphProcessor() { clean_hook(); }

void GraphProcessor::begin_graph_capture(RunMode mode = RunMode::NORMAL) {
    tt::tt_metal::GraphTracker::instance().push_processor(std::make_shared<GraphProcessor>(mode));
}

nlohmann::json GraphProcessor::end_graph_capture() {
    auto res = tt::tt_metal::GraphTracker::instance().get_processors().back()->end_capture();
    tt::tt_metal::GraphTracker::instance().pop_processor();
    return res;
}

nlohmann::json GraphProcessor::end_graph_capture_to_file(const std::filesystem::path& report_path) {
    const auto& processors = tt::tt_metal::GraphTracker::instance().get_processors();
    TT_ASSERT(!processors.empty(), "No active graph capture to end");

    auto* processor = dynamic_cast<GraphProcessor*>(processors.back().get());
    TT_ASSERT(processor != nullptr, "Current processor is not a GraphProcessor");

    // Finalize the graph, then build the report via the shared get_report() path
    auto graph_json = processor->end_capture();
    nlohmann::json report = processor->get_report();

    // Write to file
    if (report_path.has_parent_path()) {
        std::filesystem::create_directories(report_path.parent_path());
    }
    std::ofstream file(report_path);
    if (!file.is_open()) {
        TT_THROW("Failed to open graph report file for writing: {}", report_path.string());
    }
    file << report.dump(2);
    if (!file) {
        TT_THROW("Failed to write graph report to file: {}", report_path.string());
    }
    file.close();
    log_info(tt::LogAlways, "Graph report written to: {}", report_path.string());

    tt::tt_metal::GraphTracker::instance().pop_processor();
    return graph_json;
}

bool ProcessorHooks::hook_allocate(const tt::tt_metal::Buffer* /*buffer*/) { return do_block; }

bool ProcessorHooks::hook_deallocate(tt::tt_metal::Buffer* /*buffer*/) { return do_block; }

bool ProcessorHooks::hook_write_to_device(const tt::tt_metal::Buffer* /*buffer*/) { return do_block; }

bool ProcessorHooks::hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* /*mesh_buffer*/) {
    return do_block;
}

bool ProcessorHooks::hook_read_from_device(tt::tt_metal::Buffer* /*buffer*/) { return do_block; }

bool ProcessorHooks::hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* /*mesh_buffer*/) {
    return do_block;
}

bool ProcessorHooks::hook_program(tt::tt_metal::Program*) { return do_block; }

void ProcessorHooks::set_block(bool block) { do_block = block; }
bool ProcessorHooks::get_block() const { return do_block; }

ScopedGraphCapture::ScopedGraphCapture(GraphProcessor::RunMode mode) : is_active(true) {
    GraphProcessor::begin_graph_capture(mode);
}

ScopedGraphCapture::ScopedGraphCapture(GraphProcessor::RunMode mode, std::filesystem::path report_path) :
    is_active(true), auto_report_path(std::move(report_path)) {
    GraphProcessor::begin_graph_capture(mode);
}

ScopedGraphCapture::~ScopedGraphCapture() {
    if (!is_active) {
        return;
    }
    try {
        if (!auto_report_path.empty()) {
            GraphProcessor::end_graph_capture_to_file(auto_report_path);
        } else {
            GraphProcessor::end_graph_capture();
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogAlways, "Exception during graph capture teardown: {}", e.what());
    }
}

nlohmann::json ScopedGraphCapture::end_graph_capture() {
    is_active = false;
    return GraphProcessor::end_graph_capture();
}

nlohmann::json ScopedGraphCapture::end_graph_capture_to_file(const std::filesystem::path& report_path) {
    is_active = false;
    return GraphProcessor::end_graph_capture_to_file(report_path);
}

}  // namespace ttnn::graph
