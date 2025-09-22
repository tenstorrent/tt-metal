// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_argument_serializer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <boost/algorithm/string/replace.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& value) {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << (value[i] ? "true" : "false");
        if (i < value.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

namespace ttnn::graph {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tensor& tensor) {
    tt::stl::reflection::operator<<(os, tensor);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os, const tt::stl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const tt::stl::StrongType<unsigned long, tt::tt_metal::experimental::GlobalCircularBuffer>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::variant<
        ttnn::operations::matmul::MatmulMultiCoreProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::variant<
        ttnn::operations::matmul::MatmulMultiCoreProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig,
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::variant<float, int>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::variant<int, float>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::variant<unsigned int, float>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::variant<float, unsigned int>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::variant<int, ttsl::SmallVector<int, 8ul>>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::set<tt::tt_metal::distributed::MeshCoordinate, std::less<tt::tt_metal::distributed::MeshCoordinate>>&
        value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const ttnn::types::CoreRangeSet& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::GlobalSemaphore& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& value) {
    if (value.has_value()) {
        os << *value;
    } else {
        os << "<nullopt>";
    }
    return os;
}

// Helper function to serialize a value
template <typename T>
void serialize_value(std::ostringstream& oss, const T& value) {
    oss << value;
}

std::string graph_demangle(const std::string_view name) {
    int status = -4;
    char* res = abi::__cxa_demangle(name.data(), nullptr, nullptr, &status);
    const char* const demangled_name = (status == 0) ? res : name.data();
    std::string ret_val(demangled_name);
    free(res);  // NOLINT(cppcoreguidelines-no-malloc)
    return ret_val;
}

GraphArgumentSerializer::GraphArgumentSerializer() { initialize(); }

GraphArgumentSerializer& GraphArgumentSerializer::instance() {
    static GraphArgumentSerializer new_instance;
    return new_instance;
}

std::unordered_map<std::type_index, GraphArgumentSerializer::ConvertionFunction>& GraphArgumentSerializer::registry() {
    return map;
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_small_vector() {
    auto conversion_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<T, N>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            serialize_value(oss, vec[i]);
            if (i < vec.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<T, N>>)] = conversion_function;
    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<T, N>>)] = conversion_function;
    registry()[typeid(const std::reference_wrapper<ttsl::SmallVector<T, N>>)] = conversion_function;
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<const T, N>>)] = conversion_function;
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_array() {
    registry()[typeid(std::reference_wrapper<std::array<T, N>>)] = [](const std::any& value) -> std::string {
        auto referenced_value = std::any_cast<std::reference_wrapper<std::array<T, N>>>(value);
        const auto& arr = referenced_value.get();
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            serialize_value(oss, arr[i]);
            if (i < arr.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    };
}

template <typename T>
void GraphArgumentSerializer::register_vector() {
    registry()[typeid(std::reference_wrapper<std::vector<T, std::allocator<T>>>)] =
        [](const std::any& value) -> std::string {
        auto referenced_value = std::any_cast<std::reference_wrapper<std::vector<T, std::allocator<T>>>>(value);
        const auto& vec = referenced_value.get();
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            serialize_value(oss, vec[i]);
            if (i < vec.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    };
}

template <typename OptionalT>
void GraphArgumentSerializer::register_optional_type() {
    registry()[typeid(std::reference_wrapper<OptionalT>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<OptionalT>>(value);
        auto& referenced_optional = referenced_value.get();
        if (referenced_optional.has_value()) {
            serialize_value(oss, *referenced_optional);
        } else {
            oss << "nullopt";
        }

        return oss.str();
    };
}

template <typename T>
void GraphArgumentSerializer::register_type() {
    GraphArgumentSerializer::ConvertionFunction regular_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        if (value.type() == typeid(std::reference_wrapper<T>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<T>>(value);
            serialize_value(oss, referenced_value.get());
        } else if (value.type() == typeid(std::reference_wrapper<const T>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const T>>(value);
            serialize_value(oss, referenced_value.get());
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

    // Skip array registration for bool type due to std::array<bool, N> serialization issues
    if constexpr (!std::is_same_v<T, bool>) {
        register_array<T, 2>();
        register_array<T, 4>();
        register_array<T, 8>();
        register_array<T, 16>();
        register_vector<T>();
    }

    register_vector<std::optional<T>>();
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
    GraphArgumentSerializer::register_type<unsigned int>();
    GraphArgumentSerializer::register_type<unsigned long>();
    GraphArgumentSerializer::register_type<uint>();
    GraphArgumentSerializer::register_type<long>();
    GraphArgumentSerializer::register_type<float>();
    GraphArgumentSerializer::register_type<uint8_t>();
    GraphArgumentSerializer::register_type<uint16_t>();
    GraphArgumentSerializer::register_type<uint32_t>();
    GraphArgumentSerializer::register_type<uint64_t>();
    GraphArgumentSerializer::register_type<int32_t>();
    GraphArgumentSerializer::register_type<std::string>();
    GraphArgumentSerializer::register_type<tt::tt_metal::DataType>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Layout>();
    GraphArgumentSerializer::register_type<tt::tt_metal::MemoryConfig>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Shape>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tensor>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tile>();
    GraphArgumentSerializer::register_type<tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>>();
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
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 8ul>>>();
    GraphArgumentSerializer::register_type<
        std::set<tt::tt_metal::distributed::MeshCoordinate, std::less<tt::tt_metal::distributed::MeshCoordinate>>>();
    GraphArgumentSerializer::register_type<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>();
    GraphArgumentSerializer::register_type<ttnn::types::CoreRangeSet>();
    GraphArgumentSerializer::register_type<signed char>();
    // std::nullopt_t cannot be used as a template parameter for std::optional
    GraphArgumentSerializer::register_type<tt::tt_metal::distributed::MeshDevice>();
    GraphArgumentSerializer::register_type<tt::tt_metal::distributed::MeshCoordinate>();
    GraphArgumentSerializer::register_type<tt::tt_metal::GlobalSemaphore>();
    GraphArgumentSerializer::register_type<tt::tt_fabric::Topology>();
    GraphArgumentSerializer::register_type<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>();
    GraphArgumentSerializer::register_type<
        ttsl::StrongType<unsigned long, tt::tt_metal::experimental::GlobalCircularBuffer>>();
}

}  // namespace ttnn::graph
