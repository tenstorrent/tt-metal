// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lightweight header for op profiler template shims.
// Does NOT include nlohmann/json.hpp, tt-metalium/tt_metal.hpp, or ttnn/operation.hpp.
// Heavy JSON assembly lives in op_profiler_json.cpp.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/type_name.hpp>
#include "ttnn/tensor/tensor.hpp"

// Forward declarations — avoid pulling in heavy headers for 1000+ TU include chain.
namespace tt::tt_metal {
class Program;
namespace detail {
// Note: no default argument here — redefinition of default argument is ill-formed when tt_metal.hpp is also included.
// Callers in this header always pass all three arguments explicitly.
uint32_t EncodePerDeviceProgramID(uint32_t base_program_id, uint32_t device_id, bool is_host_fallback_op);
}  // namespace detail
}  // namespace tt::tt_metal

namespace tt::tt_metal::op_profiler {

enum class OpType { python_fallback, tt_dnn_cpu, tt_dnn_device, unknown };

// ---------------------------------------------------------------------------
// Plain data structs — no JSON types, safe to include in hot include chains.
// ---------------------------------------------------------------------------

struct TensorMeta {
    bool is_device = false;
    int device_id = -1;
    std::string buffer_type;
    std::string memory_layout;
    std::string storage_type_str;
    std::string shape_W, shape_Z, shape_Y, shape_X;
    std::string layout;
    std::string dtype;
};

struct OpProfileData {
    uint32_t operation_id = 0;
    std::string op_name;
    std::vector<std::pair<std::string, std::string>> attributes;  // (name, fmt-formatted value)
    std::vector<TensorMeta> input_tensors;
    std::vector<TensorMeta> output_tensors;
    // Performance model fields — default matches OpPerformanceModel default constructor
    int perf_compute_ns = 1;
    int perf_ideal_ns = 1;
    int perf_bandwidth_ns = 1;
    std::vector<float> perf_input_bws;
    std::vector<float> perf_output_bws;
};

// ---------------------------------------------------------------------------
// Non-template assembly function — defined in op_profiler_json.cpp.
// Builds JSON, updates caches, returns the formatted Tracy message string.
// ---------------------------------------------------------------------------
std::string assemble_device_op_json(
    const OpProfileData& data,
    ttsl::hash::hash_t program_hash,
    ChipId device_id,
    bool program_cache_hit,
    const tt::tt_metal::Program& program);

#if defined(TRACY_ENABLE)

class thread_safe_cached_ops_map {
    using OP_INFO_MAP = std::unordered_map<ttsl::hash::hash_t, std::string>;
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

#endif  // TRACY_ENABLE

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
    using KeyType = std::pair<ChipId, ttsl::hash::hash_t>;

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

// ---------------------------------------------------------------------------
// Tracy zone helpers (inline, no JSON)
// ---------------------------------------------------------------------------

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

inline bool is_op_profiler_env_var_set() {
    const char* op_profiler_enable_str = std::getenv("TTNN_OP_PROFILER");
    if (op_profiler_enable_str != nullptr && op_profiler_enable_str[0] == '1') {
        op_profiler_is_enabled = true;
    }
    return op_profiler_is_enabled;
}

// ---------------------------------------------------------------------------
// compute_program_hash — template, stays in header
// ---------------------------------------------------------------------------

template <typename device_operation_t>
inline auto compute_program_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (requires(
                      const typename device_operation_t::operation_attributes_t& operation_attributes,
                      const typename device_operation_t::tensor_args_t& tensor_args) {
                      {
                          device_operation_t::compute_program_hash(operation_attributes, tensor_args)
                      } -> std::convertible_to<ttsl::hash::hash_t>;
                  }) {
        ZoneScopedN("Op profiler Compute custom program hash");
        return device_operation_t::compute_program_hash(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Op profiler Compute default program hash");
        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    }
}

// ---------------------------------------------------------------------------
// make_tensor_meta — extract TensorMeta from a Tensor (no JSON)
// ---------------------------------------------------------------------------

static inline TensorMeta make_tensor_meta(const Tensor& tensor) {
    TensorMeta m;
    if (tensor.storage_type() == StorageType::DEVICE) {
        m.is_device = true;
        m.device_id = tensor.device()->id();
        m.buffer_type = std::string(enchantum::to_string(tensor.memory_config().buffer_type()));
        m.memory_layout = std::string(enchantum::to_string(tensor.memory_config().memory_layout()));
    } else {
        m.storage_type_str = fmt::format("{}", enchantum::to_string(tensor.storage_type()));
    }

    auto tensor_shape_padded = tensor.padded_shape();
    auto tensor_shape_logical = tensor.logical_shape();
    m.shape_W = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 4 ? tensor_shape_padded[-4] : 1,
        tensor_shape_logical.rank() >= 4 ? tensor_shape_logical[-4] : 1);
    m.shape_Z = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 3 ? tensor_shape_padded[-3] : 1,
        tensor_shape_logical.rank() >= 3 ? tensor_shape_logical[-3] : 1);
    m.shape_Y = fmt::format(
        "{}[{}]",
        tensor_shape_padded.rank() >= 2 ? tensor_shape_padded[-2] : 1,
        tensor_shape_logical.rank() >= 2 ? tensor_shape_logical[-2] : 1);
    m.shape_X = fmt::format("{}[{}]", tensor_shape_padded[-1], tensor_shape_logical[-1]);
    m.layout = fmt::format("{}", enchantum::to_string(tensor.layout()));
    m.dtype = fmt::format("{}", enchantum::to_string(tensor.dtype()));

    return m;
}

// ---------------------------------------------------------------------------
// op_meta_data_serialized_json<device_operation_t>
// Template shim: extracts plain data, delegates JSON assembly to .cpp
// ---------------------------------------------------------------------------

template <typename device_operation_t>
inline std::string op_meta_data_serialized_json(
    const device_operation_t& /*operation*/,
    uint32_t operation_id,
    auto device_id,
    const auto& program,
    const auto& operation_attributes,
    const auto& tensor_args,
    auto& tensor_return_value,
    bool program_cache_hit = false) {
    if (!is_op_profiler_env_var_set()) {
        return {};
    }
    const bool useCachedOps = std::getenv("TT_METAL_PROFILER_NO_CACHE_OP_INFO") == nullptr;
    auto program_hash = compute_program_hash<device_operation_t>(operation_attributes, tensor_args);

    if (!useCachedOps || (cached_ops.find(device_id) == cached_ops.end()) ||
        (cached_ops.at(device_id).find(program_hash) == cached_ops.at(device_id).end())) {
        // --- Cache miss: build OpProfileData and delegate JSON assembly ---

        OpProfileData data;
        data.operation_id = operation_id;

        // Op name
        auto as_string = [](std::string_view v) -> std::string { return {v.data(), v.size()}; };
        data.op_name = as_string(ttsl::get_type_name<device_operation_t>());
        if constexpr (requires { device_operation_t::get_type_name(operation_attributes); }) {
            data.op_name = device_operation_t::get_type_name(operation_attributes);
        }
        std::replace(data.op_name.begin(), data.op_name.end(), ',', ';');

        // Attributes as string key-value pairs
        for (auto&& [name, value] : ttsl::reflection::get_attributes(operation_attributes)) {
            data.attributes.emplace_back(fmt::format("{}", name), fmt::format("{}", value));
        }

        // Input tensors → TensorMeta (no JSON)
        ttsl::reflection::visit_object_of_type<Tensor>(
            [&data](auto&& tensor) { data.input_tensors.push_back(make_tensor_meta(tensor)); }, tensor_args);

        // Output tensors → TensorMeta (no JSON)
        ttsl::reflection::visit_object_of_type<Tensor>(
            [&data](auto&& tensor) { data.output_tensors.push_back(make_tensor_meta(tensor)); }, tensor_return_value);

        // Performance model — use if constexpr to avoid depending on OpPerformanceModel type
        if constexpr (requires { device_operation_t::create_op_performance_model; }) {
            auto perfModel =
                device_operation_t::create_op_performance_model(operation_attributes, tensor_args, tensor_return_value);
            data.perf_compute_ns = perfModel.get_compute_ns();
            data.perf_ideal_ns = perfModel.get_ideal_ns();
            data.perf_bandwidth_ns = perfModel.get_bandwidth_ns();
            data.perf_input_bws = perfModel.get_input_bws();
            data.perf_output_bws = perfModel.get_output_bws();
        }
        // else: default values (1, 1, 1, {}, {}) match OpPerformanceModel default constructor

        return assemble_device_op_json(data, program_hash, device_id, program_cache_hit, program);
    }

    // --- Cache hit: fast path, no JSON needed ---
    auto opname = program_hash_to_opname_.find_if_exists({device_id, program_hash});
    runtime_id_to_opname_.insert({device_id, program.get_runtime_id()}, std::move(opname));
    return fmt::format("{}{}`", cached_ops.at(device_id).at(program_hash), operation_id);
}

// ---------------------------------------------------------------------------
// TracyOpMeshWorkload macro — uses the template shim above
// ---------------------------------------------------------------------------

#define TracyOpMeshWorkload(                                                                                       \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value, program_cache_hit)                 \
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
                auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(base_program_id, device_id, false);     \
                std::string op_message = tt::tt_metal::op_profiler::op_meta_data_serialized_json(                  \
                    operation, op_id, device_id, program, operation_attributes, tensor_args, tensor_return_value, program_cache_hit); \
                std::string op_text = fmt::format("id:{}", op_id);                                                 \
                ZoneText(op_text.c_str(), op_text.size());                                                         \
                tt::tt_metal::op_profiler::tracy_message(op_message);                                              \
            }                                                                                                      \
        }                                                                                                          \
    }

#else  // !TRACY_ENABLE

#define TracyOpMeshWorkload( \
    mesh_device, mesh_workload, operation, operation_attributes, tensor_args, tensor_return_value, program_cache_hit)

#endif  // TRACY_ENABLE

}  // namespace tt::tt_metal::op_profiler
