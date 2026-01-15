// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <limits>
#include <mutex>
#include <reflect>
#include <stack>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

using json = nlohmann::json;

namespace tt::tt_metal::op_profiler {

enum class OpType { python_fallback, tt_dnn_cpu, tt_dnn_device, unknown };

#if defined(TRACY_ENABLE)
class thread_safe_cached_ops_map {
    using OP_INFO_MAP = std::unordered_map<tt::tt_metal::operation::Hash, std::string>;
    using DEVICE_OP_MAP = std::unordered_map<uint32_t, OP_INFO_MAP>;

public:
    DEVICE_OP_MAP::iterator find(uint32_t device_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.find(device_id);
    }
    DEVICE_OP_MAP::iterator end() {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.end();
    }
    OP_INFO_MAP& at(uint32_t device_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.at(device_id);
    }
    void emplace(uint32_t device_id, OP_INFO_MAP&& device_op_entry) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        map.emplace(device_id, device_op_entry);
    }

private:
    std::mutex map_mutex;
    DEVICE_OP_MAP map;
};

class thread_safe_call_stack {
public:
    void push(const TracyCZoneCtx& ctx) {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        call_stack.push(ctx);
    }
    bool empty() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        return call_stack.empty();
    }
    void pop() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        call_stack.pop();
    }
    TracyCZoneCtx& top() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        return call_stack.top();
    }

private:
    std::mutex stack_mutex;
    std::stack<TracyCZoneCtx> call_stack;
};

inline thread_safe_cached_ops_map cached_ops{};
inline thread_safe_call_stack call_stack;
inline bool op_profiler_is_enabled = false;

template <typename device_operation_t>
inline auto compute_program_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (requires(
                      const typename device_operation_t::operation_attributes_t& operation_attributes,
                      const typename device_operation_t::tensor_args_t& tensor_args) {
                      {
                          device_operation_t::compute_program_hash(operation_attributes, tensor_args)
                      } -> std::convertible_to<tt::stl::hash::hash_t>;
                  }) {
        ZoneScopedN("Op profiler Compute custom program hash");
        return device_operation_t::compute_program_hash(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Op profiler Compute default program hash");
        return tt::stl::hash::hash_objects_with_default_seed(
            tt::stl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    }
}

inline bool is_op_profiler_env_var_set() {
    const char* op_profiler_enable_str = std::getenv("TTNN_OP_PROFILER");
    if (op_profiler_enable_str != nullptr && op_profiler_enable_str[0] == '1') {
        op_profiler_is_enabled = true;
    }
    return op_profiler_is_enabled;
}
#endif

class RuntimeIDToOpName {
    using RuntimeID = uint32_t;
    using KeyType = std::pair<ChipId, RuntimeID>;
    using MapType = std::map<KeyType, std::string>;

public:
    MapType::iterator find(ChipId device_id, RuntimeID runtime_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.find({device_id, runtime_id});
    }
    std::string at(ChipId device_id, RuntimeID runtime_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.at({device_id, runtime_id});
    }
    void insert(KeyType key, std::string opname) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        map.emplace(key, std::move(opname));
    }
    MapType export_map() {
        // thread-safe copy of internal map contents
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map;
    }

private:
    std::mutex map_mutex;
    MapType map;
};

inline RuntimeIDToOpName runtime_id_to_opname_{};

class ProgramHashToOpName {
    using KeyType = std::pair<ChipId, tt::stl::hash::hash_t>;

public:
    std::string find_if_exists(const KeyType& key) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        auto it = map.find(key);
        if (it != map.end()) {
            return it->second;
        }
        return "";
    }
    void insert(const KeyType& key, std::string opname) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        map.emplace(key, std::move(opname));
    }

private:
    std::mutex map_mutex;
    std::map<KeyType, std::string> map;
};

inline ProgramHashToOpName program_hash_to_opname_{};

inline void start_tracy_zone(
    [[maybe_unused]] const std::string& source,
    [[maybe_unused]] const std::string& functName,
    [[maybe_unused]] uint32_t lineNum,
    [[maybe_unused]] uint32_t color = 0) {
#if defined(TRACY_ENABLE)
    auto tracySrcLoc =
        ___tracy_alloc_srcloc(lineNum, source.c_str(), source.length(), functName.c_str(), functName.length());
    TracyCZoneCtx ctx = ___tracy_emit_zone_begin_alloc(tracySrcLoc, 1);
    if (color != 0) {
        TracyCZoneColor(ctx, color);
    }

    call_stack.push(ctx);
#endif
}

inline bool stop_tracy_zone([[maybe_unused]] const std::string& name = "", [[maybe_unused]] uint32_t color = 0) {
    bool callStackWasEmpty = true;
#if defined(TRACY_ENABLE)
    if (!call_stack.empty()) {
        callStackWasEmpty = false;
        TracyCZoneCtx ctx = call_stack.top();
        if (!name.empty()) {
            TracyCZoneName(ctx, name.c_str(), name.length());
        }
        if (color != 0) {
            TracyCZoneColor(ctx, color);
        }
        TracyCZoneEnd(ctx);
        call_stack.pop();
    }
#endif
    return callStackWasEmpty;
}

constexpr auto tracy_max_message_length =
    static_cast<size_t>(std::numeric_limits<uint16_t>::max());  // Tracy hard limit is 64KiB including null terminator

inline void tracy_message([[maybe_unused]] const std::string& source, [[maybe_unused]] uint32_t color = 0xf0f8ff) {
#if defined(TRACY_ENABLE)
    const auto truncated_size = std::min(source.size(), tracy_max_message_length - 1);
    if (source.size() > truncated_size) {
        log_warning(
            tt::LogMetal,
            "Tracy profiler message truncated from {} to {} bytes to honor tracy_max_message_length. Perf op report "
            "generation might break due to corrupted json message data",
            source.size(),
            truncated_size);
    }
    TracyMessageC(source.c_str(), truncated_size, color);
#endif
}

inline void tracy_frame() {
#if defined(TRACY_ENABLE)
    FrameMark;
#endif
}

#if defined(TRACY_ENABLE)
static inline json get_kernels_json(ChipId device_id, const Program& program) {
    std::vector<json> computeKernels;
    std::vector<json> datamovementKernels;

    IDevice* device = tt::tt_metal::detail::GetActiveDevice(device_id);

    json kernelSizes;
    // TODO(HalProcessorClassType): all the combinations can be queried from HAL instead of hardcoded here, but
    // currently HAL does not correctly report the number of processors under DM.
    // It should report (DM, 0) and (DM, 1), but instead it currently reports (DM, 0) and (DM+1, 0).
    // So hardcode for now, this is on par with previously hardcoded brisc, ncrisc, etc.
    kernelSizes["TENSIX_DM_0_max_kernel_size"] = 0;
    kernelSizes["TENSIX_DM_1_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_0_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_1_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_2_max_kernel_size"] = 0;
    kernelSizes["ACTIVE_ETH_DM_0_max_kernel_size"] = 0;
    kernelSizes["ACTIVE_ETH_DM_1_max_kernel_size"] = 0;
    kernelSizes["IDLE_ETH_DM_0_max_kernel_size"] = 0;
    kernelSizes["IDLE_ETH_DM_1_max_kernel_size"] = 0;

    for (const auto& kernel : detail::collect_kernel_meta(program, device)) {
        json kernelObj;
        kernelObj["source"] = kernel.source;
        kernelObj["name"] = kernel.name;

        auto processor_class = kernel.processor_class;
        if (processor_class == HalProcessorClassType::COMPUTE) {
            MathFidelity mathFidelity = kernel.math_fidelity.value();
            kernelObj["math_fidelity"] = enchantum::to_string(mathFidelity);
            computeKernels.push_back(std::move(kernelObj));
        } else {
            datamovementKernels.push_back(std::move(kernelObj));
        }

        auto core_type = kernel.programmable_core_type;
        auto core_type_name = enchantum::to_string(core_type);
        auto processor_class_name = enchantum::to_string(kernel.processor_class);

        for (const auto& binary_meta : kernel.binary_meta) {
            auto key = fmt::format(
                "{}_{}_{}_max_kernel_size", core_type_name, processor_class_name, binary_meta.processor_type);
            if (kernelSizes.value(key, 0) < binary_meta.packed_size) {
                kernelSizes[key] = binary_meta.packed_size;
            }
        }
    }

    json ret;
    ret["compute_kernels"] = std::move(computeKernels);
    ret["datamovement_kernels"] = std::move(datamovementKernels);
    ret["kernel_sizes"] = std::move(kernelSizes);
    return ret;
}

static inline json get_tensor_json(const Tensor& tensor) {
    json ret;
    if (tensor.storage_type() == StorageType::DEVICE) {
        ret["storage_type"]["device_id"] = tensor.device()->id();
        ret["storage_type"]["memory_config"]["buffer_type"] =
            enchantum::to_string(tensor.memory_config().buffer_type());
        ret["storage_type"]["memory_config"]["memory_layout"] =
            enchantum::to_string(tensor.memory_config().memory_layout());
    } else {
        ret["storage_type"] = fmt::format("{}", enchantum::to_string(tensor.storage_type()));
    }

    auto tensor_shape_padded = tensor.padded_shape();
    auto tensor_shape_logical = tensor.logical_shape();
    ret["shape"]["W"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 4 ? tensor_shape_padded[-4] : 1,
        tensor_shape_logical.rank() >= 4 ? tensor_shape_logical[-4] : 1);
    ret["shape"]["Z"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 3 ? tensor_shape_padded[-3] : 1,
        tensor_shape_logical.rank() >= 3 ? tensor_shape_logical[-3] : 1);
    ret["shape"]["Y"] = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 2 ? tensor_shape_padded[-2] : 1,
        tensor_shape_logical.rank() >= 2 ? tensor_shape_logical[-2] : 1);
    ret["shape"]["X"] = fmt::format("{}[{}]", tensor_shape_padded[-1], tensor_shape_logical[-1]);
    ret["layout"] = fmt::format("{}", enchantum::to_string(tensor.layout()));
    ret["dtype"] = fmt::format("{}", enchantum::to_string(tensor.dtype()));

    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<Tensor>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    ret.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        ret.push_back(get_tensor_json(tensor));
    }
    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<std::optional<const Tensor>>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    for (const auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

static inline std::vector<json> get_tensors_json(const std::vector<std::optional<Tensor>>& tensors) {
    ZoneScoped;
    std::vector<json> ret;
    for (const auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

template <bool IsExternal = false, typename Operation>
inline json get_base_json(
    uint32_t opID,
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    std::optional<std::reference_wrapper<typename Operation::OutputTensors>> output_tensors = std::nullopt) {
    ZoneScoped;
    json j;
    j["global_call_count"] = opID;

    std::string opName = op.get_type_name();

    if constexpr (!IsExternal) {
        auto profiler_info = op.create_profiler_info(input_tensors);
        if (profiler_info.preferred_name.has_value()) {
            j["op_code"] = profiler_info.preferred_name.value();
        }

        if (profiler_info.parallelization_strategy.has_value()) {
            j["parallelization_strategy"] = profiler_info.parallelization_strategy.value();
        }
    }

    std::replace(opName.begin(), opName.end(), ',', ';');
    j["op_code"] = opName;

    json attributesObj;
    auto attributes = op.attributes();
    if (not attributes.empty()) {
        ZoneScopedN("get_attributes_json");
        for (auto&& [name, value] : attributes) {
            std::string nameStr;
            nameStr = fmt::format("{}", name);
            attributesObj[nameStr] = fmt::format("{}", value);
        }
    }

    j["attributes"] = attributesObj;

    j["input_tensors"] = get_tensors_json(input_tensors);

    if (output_tensors.has_value()) {
        j["output_tensors"] = get_tensors_json(output_tensors.value());
    }
    return j;
}

template <typename device_operation_t>
inline json get_base_json(
    uint32_t operation_id,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
#if defined(TRACY_ENABLE)
    if (!is_op_profiler_env_var_set()) {
        return json{};
    }
    ZoneScoped;
    json j;
    j["global_call_count"] = operation_id;

    auto as_string = [](std::string_view v) -> std::string { return {v.data(), v.size()}; };
    std::string opName = as_string(tt::stl::get_type_name<device_operation_t>());
    std::replace(opName.begin(), opName.end(), ',', ';');
    j["op_code"] = opName;

    json attributesObj;
    for (auto&& [name, value] : tt::stl::reflection::get_attributes(operation_attributes)) {
        std::string nameStr;
        nameStr = fmt::format("{}", name);
        attributesObj[nameStr] = fmt::format("{}", value);
    }
    j["attributes"] = attributesObj;

    std::vector<json> input_tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](auto&& tensor) { input_tensors.push_back(get_tensor_json(tensor)); }, tensor_args);
    j["input_tensors"] = input_tensors;

    std::vector<json> output_tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&output_tensors](auto&& tensor) { output_tensors.push_back(get_tensor_json(tensor)); }, tensor_return_value);
    j["output_tensors"] = output_tensors;

    return j;
#else
    return json{};
#endif
}

inline std::string op_meta_data_serialized_json(
    uint32_t opID, const tt::tt_metal::operation::ExternalOperation& op, const std::vector<Tensor>& input_tensors) {
#if defined(TRACY_ENABLE)
    if (!is_op_profiler_env_var_set()) {
        return {};
    }
    auto j = get_base_json<true>(opID, op, input_tensors);
    j["op_type"] = enchantum::to_string(OpType::python_fallback);
    std::string ser = j.dump(4);
    return fmt::format("`TT_DNN_FALL_BACK_OP:{} ->\n{}`", j["op_code"].dump(), ser);
#else
    return {};
#endif
}

template <typename device_operation_t>
inline std::string op_meta_data_serialized_json(
    const device_operation_t& /*operation*/,
    uint32_t operation_id,
    auto device_id,
    const auto& program,
    const auto& operation_attributes,
    const auto& tensor_args,
    auto& tensor_return_value) {
#if defined(TRACY_ENABLE)
    if (!is_op_profiler_env_var_set()) {
        return {};
    }
    const bool useCachedOps = std::getenv("TT_METAL_PROFILER_NO_CACHE_OP_INFO") == nullptr;
    auto program_hash = compute_program_hash<device_operation_t>(operation_attributes, tensor_args);

    if (!useCachedOps || (cached_ops.find(device_id) == cached_ops.end()) ||
        (cached_ops.at(device_id).find(program_hash) == cached_ops.at(device_id).end())) {
        auto j =
            get_base_json<device_operation_t>(operation_id, operation_attributes, tensor_args, tensor_return_value);
        j["op_type"] = enchantum::to_string(OpType::tt_dnn_device);
        j["device_id"] = device_id;
        j["op_hash"] = program_hash;
        j["kernel_info"] = get_kernels_json(device_id, program);

        auto opname = j["op_code"].template get<std::string>();
        runtime_id_to_opname_.insert({device_id, program.get_runtime_id()}, opname);
        program_hash_to_opname_.insert({device_id, program_hash}, opname);

        j["optional_input_tensors"] = std::vector<json>{};

        auto perfModel = [&]() {
            if constexpr (requires { device_operation_t::create_op_performance_model; }) {
                return device_operation_t::create_op_performance_model(
                    operation_attributes, tensor_args, tensor_return_value);
            } else {
                return tt::tt_metal::operation::OpPerformanceModel{};
            }
        }();
        j["performance_model"]["compute_ns"] = perfModel.get_compute_ns();
        j["performance_model"]["ideal_ns"] = perfModel.get_ideal_ns();
        j["performance_model"]["bandwidth_ns"] = perfModel.get_bandwidth_ns();
        j["performance_model"]["input_bws"] = perfModel.get_input_bws();
        j["performance_model"]["output_bws"] = perfModel.get_output_bws();

        std::string short_str =
            fmt::format("`TT_DNN_DEVICE_OP: {}, {}, {}, ", j["op_code"].dump(), program_hash, device_id);
        if (cached_ops.find(device_id) == cached_ops.end()) {
            cached_ops.emplace(
                device_id, (std::unordered_map<tt::tt_metal::operation::Hash, std::string>){{program_hash, short_str}});
        } else {
            cached_ops.at(device_id).emplace(program_hash, short_str);
        }

        auto msg = fmt::format("{}{} ->\n{}`", short_str, operation_id, j.dump(4));
        if (msg.size() >= std::numeric_limits<uint16_t>::max()) {
            msg = fmt::format("{}{} ->\n{}`", short_str, operation_id, j.dump(-1));
        }
        return msg;
    }
    auto opname = program_hash_to_opname_.find_if_exists({device_id, program_hash});
    runtime_id_to_opname_.insert({device_id, program.get_runtime_id()}, std::move(opname));
    return fmt::format("{}{}`", cached_ops.at(device_id).at(program_hash), operation_id);
#else
    return {};
#endif
}

#define TracyOpTTNNDevice(                                                                                        \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value)          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                \
        std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                         \
            operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value); \
        std::string op_text = fmt::format("id:{}", operation_id);                                                 \
        ZoneText(op_text.c_str(), op_text.size());                                                                \
        tt::tt_metal::op_profiler::tracy_message(op_message);                                                     \
    }

#define TracyOpTTNNExternal(op, input_tensors, base_op_id)                                                          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                  \
        /* This op runs entirely on host, but its ID must be generated using the same data-path as device-side */   \
        /* ops, for accurate reporting by the performance post-processor. */                                        \
        auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(base_op_id, 0, true);                           \
        std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(op_id, op, input_tensors); \
        std::string op_text = fmt::format("id:{}", op_id);                                                          \
        ZoneText(op_text.c_str(), op_text.size());                                                                  \
        tt::tt_metal::op_profiler::tracy_message(op_message);                                                       \
    }

#define TracyOpMeshWorkload(                                                                                       \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value)                 \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {                                                 \
        for (const auto& [range, program] : (mesh_workload).get_programs()) {                                      \
            auto base_program_id = program.get_runtime_id();                                                       \
            for (auto coord : range) {                                                                             \
                /* Important! `TT_DNN_DEVICE_OP` must be used in conjunction with `TracyOpMeshWorkload` to feed */ \
                /* regression tests well-formed data. */                                                           \
                /* TODO: (Issue #20233): Move the zone below outside TracyOpMeshWorkload. */                       \
                if (!(mesh_device)->is_local(coord)) {                                                             \
                    continue;                                                                                      \
                }                                                                                                  \
                ZoneScopedN("TT_DNN_DEVICE_OP");                                                                   \
                auto device_id = (mesh_device)->get_device(coord)->id();                                           \
                auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(base_program_id, device_id);           \
                std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                  \
                    operation, op_id, device_id, program, operation_attributes, tensor_args, tensor_return_value); \
                std::string op_text = fmt::format("id:{}", op_id);                                                 \
                ZoneText(op_text.c_str(), op_text.size());                                                         \
                tt::tt_metal::op_profiler::tracy_message(op_message);                                              \
            }                                                                                                      \
        }                                                                                                          \
    }

#else

#define TracyOpTTNNDevice( \
    operation, operation_id, device_id, program, operation_attributes, tensor_args, tensor_return_value)
#define TracyOpTTNNExternal(op, input_tensors, base_op_id)
#define TracyOpMeshWorkload( \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value)

#endif
}  // namespace tt::tt_metal::op_profiler
