// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/reshape_policy.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement::detail {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* kPolicyEnvVar = "TT_METAL_EXECUTION_POLICY_RESHAPE";
constexpr size_t kMaxReportedMismatches = 16;
constexpr int kMaxSpecDifferenceWarnings = 16;

std::atomic<int> g_spec_difference_warnings{0};

// -1: no override; otherwise the ReshapeExecutionPolicy value.
std::atomic<int> g_process_override{-1};

std::optional<ReshapeExecutionPolicy>& thread_policy() {
    thread_local std::optional<ReshapeExecutionPolicy> policy;
    return policy;
}

ReshapeExecutionPolicy environment_policy() {
    static const ReshapeExecutionPolicy policy = [] {
        const char* value = std::getenv(kPolicyEnvVar);
        if (value == nullptr || *value == '\0') {
            return ReshapeExecutionPolicy::New;
        }
        return parse_reshape_execution_policy(value);
    }();
    return policy;
}

std::string topology_string(const tt::tt_metal::TensorTopology& topology) {
    std::ostringstream os;
    os << topology;
    return os.str();
}

// Logical elements of one shard as raw bit patterns, in row-major logical order. Block-float dtypes have
// no per-element storage; their decoded float32 values are compared instead, which is what any consumer
// observes and is still exact (the decode is deterministic). std::nullopt for dtypes without a host readback.
std::optional<std::vector<uint32_t>> logical_bits(const ttnn::Tensor& shard) {
    std::vector<uint32_t> bits;
    auto widen = [&bits](const auto& values, auto to_bits) {
        bits.resize(values.size());
        std::transform(values.begin(), values.end(), bits.begin(), to_bits);
    };
    switch (shard.dtype()) {
        case DataType::BFLOAT16:
            widen(shard.to_vector<bfloat16>(), [](bfloat16 v) { return uint32_t{std::bit_cast<uint16_t>(v)}; });
            break;
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            widen(shard.to_vector<float>(), [](float v) { return std::bit_cast<uint32_t>(v); });
            break;
        case DataType::INT32:
            widen(shard.to_vector<int32_t>(), [](int32_t v) { return std::bit_cast<uint32_t>(v); });
            break;
        case DataType::UINT32: widen(shard.to_vector<uint32_t>(), [](uint32_t v) { return v; }); break;
        case DataType::UINT16: widen(shard.to_vector<uint16_t>(), [](uint16_t v) { return uint32_t{v}; }); break;
        case DataType::UINT8: widen(shard.to_vector<uint8_t>(), [](uint8_t v) { return uint32_t{v}; }); break;
        case DataType::INT8:
            widen(shard.to_vector<int8_t>(), [](int8_t v) { return uint32_t{std::bit_cast<uint8_t>(v)}; });
            break;
        default: return std::nullopt;
    }
    return bits;
}

std::string decode_bits(DataType dtype, uint32_t bits) {
    switch (dtype) {
        case DataType::BFLOAT16: return fmt::format("{}", std::bit_cast<float>(bits << 16));
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return fmt::format("{}", std::bit_cast<float>(bits));
        case DataType::INT32: return fmt::format("{}", std::bit_cast<int32_t>(bits));
        default: return fmt::format("{}", bits);
    }
}

std::string nd_index(const ttnn::Shape& shape, uint64_t flat) {
    std::vector<uint64_t> index(shape.rank());
    for (int dim = static_cast<int>(shape.rank()) - 1; dim >= 0; --dim) {
        index[dim] = flat % shape[dim];
        flat /= shape[dim];
    }
    return fmt::format("[{}]", fmt::join(index, ", "));
}

struct Shard {
    std::string label;
    ttnn::Tensor tensor;
};

std::vector<Shard> shards_of(const ttnn::Tensor& tensor) {
    std::vector<Shard> shards;
    auto parts = ttnn::distributed::get_device_tensors(tensor);
    shards.reserve(parts.size());
    if (is_device_tensor(tensor)) {
        const auto coords = tensor.device_storage().get_coords();
        for (size_t i = 0; i < parts.size(); ++i) {
            shards.push_back({fmt::format("mesh coordinate {}", coords[i]), std::move(parts[i])});
        }
    } else {
        for (size_t i = 0; i < parts.size(); ++i) {
            shards.push_back({fmt::format("host shard {}", i), std::move(parts[i])});
        }
    }
    return shards;
}

bool readback_is_illegal(const ttnn::Tensor& tensor) {
    if (tt::tt_metal::GraphTracker::instance().get_hook() != nullptr) {
        return true;
    }
    if (!is_device_tensor(tensor)) {
        return false;
    }
    auto* device = tensor.device();
    for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
        if (device->mesh_command_queue(cq_id).trace_id().has_value()) {
            return true;
        }
    }
    return false;
}

std::string describe_request(const ReshapeRequest& request) {
    return fmt::format(
        "input spec: {}\n"
        "requested logical shape: {}, requested padded shape: {}\n"
        "requested memory_config: {}, pad_value: {}, sub_core_grid: {}, skip_padding_fill: {}",
        request.input.tensor_spec(),
        request.logical_shape,
        request.padded_shape,
        request.memory_config.has_value() ? fmt::format("{}", *request.memory_config) : std::string("(inherited)"),
        request.pad_value.has_value()
            ? std::visit([](auto value) { return fmt::format("{}", value); }, *request.pad_value)
            : std::string("(default)"),
        request.sub_core_grid.has_value() ? fmt::format("{}", *request.sub_core_grid) : std::string("(none)"),
        request.skip_padding_fill);
}

nlohmann::json shape_json(const ttnn::Shape& shape) {
    nlohmann::json dims = nlohmann::json::array();
    for (size_t i = 0; i < shape.rank(); ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

nlohmann::json topology_json(const tt::tt_metal::TensorTopology& topology) {
    using tt::tt_metal::distributed::MeshMapperConfig;
    nlohmann::json distribution_shape = nlohmann::json::array();
    for (size_t i = 0; i < topology.distribution_shape().dims(); ++i) {
        distribution_shape.push_back(topology.distribution_shape()[i]);
    }
    nlohmann::json placements = nlohmann::json::array();
    for (const auto& placement : topology.placements()) {
        if (const auto* shard = std::get_if<MeshMapperConfig::Shard>(&placement)) {
            placements.push_back({{"shard", shard->dim}});
        } else {
            placements.push_back("replicate");
        }
    }
    return {{"distribution_shape", std::move(distribution_shape)}, {"placements", std::move(placements)}};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::ostream& operator<<(std::ostream& os, const ReshapeCallRecord& record) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& request = record.request;
    const auto& input = request.input;
    const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();

    nlohmann::json input_json{
        {"logical_shape", shape_json(input.logical_shape())},
        {"padded_shape", shape_json(input.padded_shape())},
        {"dtype", enchantum::to_string(input.dtype())},
        {"layout", enchantum::to_string(input.layout())},
        {"tile", {tile_shape[0], tile_shape[1]}},
        {"storage", is_device_tensor(input) ? "device" : "host"},
        {"memory_config", ttsl::json::to_json(input.memory_config())},
        {"topology", topology_json(input.tensor_topology())},
    };

    nlohmann::json pad_value = nullptr;
    if (request.pad_value.has_value()) {
        pad_value = std::visit(
            [](auto value) -> nlohmann::json {
                if constexpr (std::is_same_v<decltype(value), float>) {
                    return {{"float", value}};
                } else {
                    return {{"uint", value}};
                }
            },
            *request.pad_value);
    }

    nlohmann::json request_json{
        {"logical_shape", shape_json(request.logical_shape)},
        {"padded_shape", shape_json(request.padded_shape)},
        {"memory_config",
         request.memory_config.has_value() ? ttsl::json::to_json(*request.memory_config) : nlohmann::json(nullptr)},
        {"pad_value", std::move(pad_value)},
        {"sub_core_grid",
         request.sub_core_grid.has_value() ? ttsl::json::to_json(*request.sub_core_grid) : nlohmann::json(nullptr)},
        {"skip_padding_fill", request.skip_padding_fill},
    };

    return os << nlohmann::json{{"input", std::move(input_json)}, {"request", std::move(request_json)}}.dump();
}

ReshapeExecutionPolicy parse_reshape_execution_policy(std::string_view value) {
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "new") {
        return ReshapeExecutionPolicy::New;
    }
    if (lowered == "legacy") {
        return ReshapeExecutionPolicy::Legacy;
    }
    if (lowered == "compare") {
        return ReshapeExecutionPolicy::Compare;
    }
    TT_THROW(
        "{}='{}' is not a reshape execution policy; expected one of 'new', 'legacy', 'compare'",
        CMAKE_UNIQUE_NAMESPACE::kPolicyEnvVar,
        value);
}

ReshapeExecutionPolicy reshape_execution_policy() {
    if (const auto& scoped = CMAKE_UNIQUE_NAMESPACE::thread_policy(); scoped.has_value()) {
        return *scoped;
    }
    if (const int process_override = CMAKE_UNIQUE_NAMESPACE::g_process_override.load(std::memory_order_relaxed);
        process_override >= 0) {
        return static_cast<ReshapeExecutionPolicy>(process_override);
    }
    return CMAKE_UNIQUE_NAMESPACE::environment_policy();
}

void set_reshape_execution_policy_override(std::optional<ReshapeExecutionPolicy> policy) {
    CMAKE_UNIQUE_NAMESPACE::g_process_override.store(
        policy.has_value() ? static_cast<int>(*policy) : -1, std::memory_order_relaxed);
}

ScopedReshapeExecutionPolicy::ScopedReshapeExecutionPolicy(ReshapeExecutionPolicy policy) :
    previous_(std::exchange(CMAKE_UNIQUE_NAMESPACE::thread_policy(), policy)) {}

ScopedReshapeExecutionPolicy::~ScopedReshapeExecutionPolicy() { CMAKE_UNIQUE_NAMESPACE::thread_policy() = previous_; }

void compare_reshape_results(const ReshapeRequest& request, const ttnn::Tensor& legacy, const ttnn::Tensor& candidate) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    if (readback_is_illegal(candidate)) {
        static std::once_flag skipped_warned;
        std::call_once(skipped_warned, [] {
            log_warning(
                tt::LogOp,
                "{}=compare: outputs cannot be read back during trace capture or NO_DISPATCH graph capture; "
                "the comparison is skipped there and the new result is returned. Emitted once per process.",
                kPolicyEnvVar);
        });
        return;
    }

    TT_FATAL(
        legacy.logical_shape() == candidate.logical_shape() && legacy.dtype() == candidate.dtype(),
        "{}=compare: reshape outputs disagree on logical shape or dtype.\n"
        "legacy output spec: {}\nnew output spec: {}\n{}",
        kPolicyEnvVar,
        legacy.tensor_spec(),
        candidate.tensor_spec(),
        describe_request(request));

    // Expected where legacy is wrong (e.g. #19272 shard specs, #64322 topology labels), so only logged, and
    // only for the first few calls: models reshape in per-step loops.
    if ((legacy.tensor_spec() != candidate.tensor_spec() || legacy.tensor_topology() != candidate.tensor_topology()) &&
        g_spec_difference_warnings.fetch_add(1, std::memory_order_relaxed) < kMaxSpecDifferenceWarnings) {
        log_warning(
            tt::LogOp,
            "{}=compare: output specs or topologies differ; data is still compared bit for bit. "
            "(At most {} of these warnings are emitted per process.)\n"
            "legacy: {} / {}\nnew: {} / {}\n{}",
            kPolicyEnvVar,
            kMaxSpecDifferenceWarnings,
            legacy.tensor_spec(),
            topology_string(legacy.tensor_topology()),
            candidate.tensor_spec(),
            topology_string(candidate.tensor_topology()),
            describe_request(request));
    }

    const auto legacy_shards = shards_of(legacy);
    const auto candidate_shards = shards_of(candidate);
    TT_FATAL(
        legacy_shards.size() == candidate_shards.size(),
        "{}=compare: legacy produced {} shards, new produced {}.\n{}",
        kPolicyEnvVar,
        legacy_shards.size(),
        candidate_shards.size(),
        describe_request(request));

    const auto& shape = candidate.logical_shape();
    const DataType dtype = candidate.dtype();
    for (size_t s = 0; s < candidate_shards.size(); ++s) {
        const auto expected_bits = logical_bits(legacy_shards[s].tensor);
        const auto actual_bits = logical_bits(candidate_shards[s].tensor);
        if (!expected_bits.has_value() || !actual_bits.has_value()) {
            static std::once_flag unsupported_warned;
            std::call_once(unsupported_warned, [&] {
                log_warning(
                    tt::LogOp,
                    "{}=compare: {} outputs cannot be read back on host, so they are not compared. Emitted once per "
                    "process.",
                    kPolicyEnvVar,
                    dtype);
            });
            return;
        }
        const auto& expected = *expected_bits;
        const auto& actual = *actual_bits;
        TT_FATAL(
            expected.size() == actual.size(),
            "{}=compare: {} holds {} logical elements in legacy and {} in new.\n{}",
            kPolicyEnvVar,
            candidate_shards[s].label,
            expected.size(),
            actual.size(),
            describe_request(request));

        size_t mismatches = 0;
        std::string report;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (expected[i] == actual[i]) {
                continue;
            }
            if (mismatches < kMaxReportedMismatches) {
                report += fmt::format(
                    "  at {}: legacy 0x{:08x} ({}), new 0x{:08x} ({})\n",
                    nd_index(shape, i),
                    expected[i],
                    decode_bits(dtype, expected[i]),
                    actual[i],
                    decode_bits(dtype, actual[i]));
            }
            ++mismatches;
        }
        if (mismatches != 0) {
            TT_THROW(
                "{}=compare: reshape outputs differ in {} of {} logical elements on {}{}:\n{}"
                "legacy output spec: {}\nnew output spec: {}\n{}",
                kPolicyEnvVar,
                mismatches,
                actual.size(),
                candidate_shards[s].label,
                mismatches > kMaxReportedMismatches ? fmt::format(" (first {} shown)", kMaxReportedMismatches) : "",
                report,
                legacy.tensor_spec(),
                candidate.tensor_spec(),
                describe_request(request));
        }
    }
}

}  // namespace ttnn::operations::data_movement::detail
