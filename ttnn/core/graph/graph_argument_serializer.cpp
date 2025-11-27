// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_argument_serializer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/types.hpp"
#include <boost/algorithm/string/replace.hpp>
#include <tt_stl/small_vector.hpp>

namespace ttnn::graph {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tile& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tensor& tensor) {
    tt::stl::reflection::operator<<(os, tensor);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>& kernel_config) {
    tt::stl::reflection::operator<<(os, kernel_config);
    return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::variant<T, U>& variant) {
    tt::stl::reflection::operator<<(os, variant);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::variant<
        ttnn::operations::matmul::MatmulMultiCoreProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>& program_config) {
    tt::stl::reflection::operator<<(os, program_config);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<uint8_t, tt::tt_metal::QueueIdTag>& h) {
    return os << *h;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional_value) {
    if (optional_value.has_value()) {
        os << optional_value.value();
    } else {
        os << "nullopt";
    }
    return os;
}

std::string graph_demangle(const std::string_view name) {
    int status = -4;
    char* res = abi::__cxa_demangle(name.data(), nullptr, nullptr, &status);
    const char* const demangled_name = (status == 0) ? res : name.data();
    std::string ret_val(demangled_name);
    free(res);  // NOLINT(cppcoreguidelines-no-malloc)
    return ret_val;
}

// Generic helper to serialize any iterable container
template <typename Container>
std::string serialize_container(const Container& container) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < container.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << container[i];
    }
    oss << "]";
    return oss.str();
}

// Helper for serializing containers of arrays (nested structure)
template <typename Container, typename T, std::size_t N>
std::string serialize_nested_array_container(const Container& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << serialize_container(vec[i]);
    }
    oss << "]";
    return oss.str();
}

// Specialization for SmallVector with nested arrays (e.g., SmallVector<std::array<uint, 2>, 8>)
template <typename T, std::size_t N, std::size_t M>
std::string serialize_container(const ttsl::SmallVector<std::array<T, N>, M>& vec) {
    return serialize_nested_array_container<ttsl::SmallVector<std::array<T, N>, M>, T, N>(vec);
}

// Helper function to register both const and non-const reference wrappers
// The lambda should use 'const T' for any_cast since it works for both const and non-const reference_wrappers
template <typename T>
void register_reference_wrapper_pair(
    std::unordered_map<std::type_index, GraphArgumentSerializer::ConversionFunction>& registry,
    std::function<std::string(const T&)> serializer_func) {
    auto wrapper = [serializer_func](const std::any& value) -> std::string {
        // Try non-const first
        try {
            auto ref = std::any_cast<std::reference_wrapper<T>>(value);
            return serializer_func(ref.get());
        } catch (const std::bad_any_cast&) {
            // Try const
            auto ref = std::any_cast<std::reference_wrapper<const T>>(value);
            return serializer_func(ref.get());
        }
    };

    registry[typeid(std::reference_wrapper<T>)] = wrapper;
    registry[typeid(std::reference_wrapper<const T>)] = wrapper;
}

GraphArgumentSerializer::GraphArgumentSerializer() { initialize(); }

GraphArgumentSerializer& GraphArgumentSerializer::instance() {
    static GraphArgumentSerializer new_instance;
    return new_instance;
}

std::unordered_map<std::type_index, GraphArgumentSerializer::ConversionFunction>& GraphArgumentSerializer::registry() {
    return map;
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_small_vector() {
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<T, N>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<T, N>>>(value);
        oss << referenced_value.get();
        return oss.str();
    };
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_array() {
    register_reference_wrapper_pair<std::array<T, N>>(registry(), serialize_container<std::array<T, N>>);
}

template <typename OptionalT>
void GraphArgumentSerializer::register_optional_type() {
    registry()[typeid(std::reference_wrapper<OptionalT>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<OptionalT>>(value);
        auto& referenced_optional = referenced_value.get();
        if (referenced_optional.has_value()) {
            oss << *referenced_optional;
        } else {
            oss << "nullopt";
        }

        return oss.str();
    };
}

template <typename T>
void GraphArgumentSerializer::register_type() {
    GraphArgumentSerializer::ConversionFunction regular_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        if (value.type() == typeid(std::reference_wrapper<T>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<T>>(value);
            oss << referenced_value.get();
        } else if (value.type() == typeid(std::reference_wrapper<const T>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const T>>(value);
            oss << referenced_value.get();
        } else {
            oss << "Unable to parse";
        }

        return oss.str();
    };

    // regular cases
    registry()[typeid(std::reference_wrapper<T>)] = regular_function;
    registry()[typeid(std::reference_wrapper<const T>)] = regular_function;
    registry()[typeid(const std::reference_wrapper<T>)] = regular_function;

    // Particular cases for optional
    register_optional_type<std::optional<T>>();
    register_optional_type<const std::optional<T>>();
    register_optional_type<std::optional<const T>>();
    register_optional_type<const std::optional<const T>>();
    // Handle complex types (feel free to add more in the future)
    // Small vector
    register_small_vector<T, 2>();
    register_small_vector<T, 4>();
    register_small_vector<T, 8>();
    register_small_vector<T, 16>();
    // std::array
    register_array<T, 2>();
    register_array<T, 4>();
    register_array<T, 8>();
    register_array<T, 16>();
}

std::vector<std::string> GraphArgumentSerializer::to_list(const std::span<std::any>& span) {
    std::vector<std::string> result;
    for (const auto& element : span) {
        if (!element.has_value()) {
            result.push_back("[any, empty]");
            continue;
        }

        auto it = registry().find(element.type());
        if (it != registry().end()) {
            auto str_result = it->second(element);
            boost::algorithm::replace_all(str_result, "__1::", "");
            result.push_back(str_result);
        } else {
            // for debugging reasons, I want to report the type that is not managed
            std::ostringstream oss;
            oss << "[ unsupported type"
                << " , ";
            auto demangled_name = graph_demangle(element.type().name());
            boost::algorithm::replace_all(demangled_name, "__1::", "");
            oss << demangled_name;
            oss << "]";
            result.push_back(oss.str());
        }
    }

    return result;
}

void GraphArgumentSerializer::initialize() {
    GraphArgumentSerializer::register_type<bool>();
    GraphArgumentSerializer::register_type<int>();
    GraphArgumentSerializer::register_type<uint>();
    GraphArgumentSerializer::register_type<long>();
    GraphArgumentSerializer::register_type<float>();
    GraphArgumentSerializer::register_type<uint8_t>();
    GraphArgumentSerializer::register_type<uint16_t>();
    GraphArgumentSerializer::register_type<std::string>();
    GraphArgumentSerializer::register_type<tt::tt_metal::DataType>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Layout>();
    GraphArgumentSerializer::register_type<tt::tt_metal::MemoryConfig>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Shape>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tensor>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tile>();
    GraphArgumentSerializer::register_type<tt::stl::StrongType<uint8_t, tt::tt_metal::QueueIdTag>>();
    GraphArgumentSerializer::register_type<ttnn::types::CoreGrid>();
    GraphArgumentSerializer::register_type<
        std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>>();
    GraphArgumentSerializer::register_type<std::variant<
        ttnn::operations::matmul::MatmulMultiCoreProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>>();
    GraphArgumentSerializer::register_type<std::variant<float, int>>();
    GraphArgumentSerializer::register_type<std::variant<int, float>>();
    GraphArgumentSerializer::register_type<std::variant<unsigned int, float>>();
    GraphArgumentSerializer::register_type<std::variant<float, unsigned int>>();

    // Register SmallVector types for various operations using generic serialize_container
    // Note: std::array<unsigned int, N> is already registered via register_type<uint>() above
    register_reference_wrapper_pair<ttsl::SmallVector<std::array<unsigned int, 2>, 8>>(
        registry(), [](const ttsl::SmallVector<std::array<unsigned int, 2>, 8>& vec) -> std::string {
            return serialize_container(vec);
        });
    register_reference_wrapper_pair<ttsl::SmallVector<long, 8>>(
        registry(), [](const ttsl::SmallVector<long, 8>& vec) -> std::string { return serialize_container(vec); });
    register_reference_wrapper_pair<ttsl::SmallVector<int, 8>>(
        registry(), [](const ttsl::SmallVector<int, 8>& vec) -> std::string { return serialize_container(vec); });

    // Register std::variant<std::array<uint, 2>, std::array<uint, 4>> for conv2d stride/dilation
    register_reference_wrapper_pair<std::variant<std::array<unsigned int, 2>, std::array<unsigned int, 4>>>(
        registry(),
        [](const std::variant<std::array<unsigned int, 2>, std::array<unsigned int, 4>>& var) -> std::string {
            return std::holds_alternative<std::array<unsigned int, 2>>(var)
                       ? serialize_container(std::get<std::array<unsigned int, 2>>(var))
                       : serialize_container(std::get<std::array<unsigned int, 4>>(var));
        });

    // Register std::nullopt_t for optional parameters
    register_reference_wrapper_pair<std::nullopt_t>(
        registry(), [](const std::nullopt_t&) -> std::string { return "nullopt"; });
}
}  // namespace ttnn::graph
