// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_argument_serializer.hpp"
#include "ttnn/graph/graph_registration.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <boost/algorithm/string/replace.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt_stl/optional_reference.hpp>
#include <type_traits>

// Include headers for additional types
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <umd/device/types/xy_pair.hpp>

// Include operation-specific serialization (moving towards decoupled architecture per review comment)
#include "ttnn/operations/eltwise/graph_serialization.hpp"
#include "ttnn/operations/matmul/graph_serialization.hpp"
#include "ttnn/operations/normalization/graph_serialization.hpp"
#include "ttnn/operations/core/graph_serialization.hpp"
#include "ttnn/operations/transformer/graph_serialization.hpp"
#include "ttnn/operations/conv/graph_serialization.hpp"
#include "ttnn/operations/data_movement/graph_serialization.hpp"
#include "ttnn/operations/sliding_window/graph_serialization.hpp"
#include "ttnn/operations/embedding/graph_serialization.hpp"

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

// Helper function to sanitize strings for JSON serialization
// Replaces invalid UTF-8 sequences with replacement character (U+FFFD)
std::string sanitize_utf8(const std::string& str) {
    std::string result;
    result.reserve(str.size());

    for (size_t i = 0; i < str.size();) {
        unsigned char c = str[i];

        // ASCII character (0x00-0x7F)
        if (c < 0x80) {
            result.push_back(c);
            i++;
        }
        // 2-byte UTF-8 (0xC0-0xDF)
        else if ((c >= 0xC0) && (c < 0xE0)) {
            if (i + 1 < str.size() && (str[i + 1] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                i += 2;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // 3-byte UTF-8 (0xE0-0xEF)
        else if ((c >= 0xE0) && (c < 0xF0)) {
            if (i + 2 < str.size() && (str[i + 1] & 0xC0) == 0x80 && (str[i + 2] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                result.push_back(str[i + 2]);
                i += 3;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // 4-byte UTF-8 (0xF0-0xF7)
        else if ((c >= 0xF0) && (c < 0xF8)) {
            if (i + 3 < str.size() && (str[i + 1] & 0xC0) == 0x80 && (str[i + 2] & 0xC0) == 0x80 &&
                (str[i + 3] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                result.push_back(str[i + 2]);
                result.push_back(str[i + 3]);
                i += 4;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // Invalid UTF-8 start byte (0x80-0xBF, 0xF8-0xFF)
        else {
            // Replace with hex escape sequence for debugging
            char hex[8];
            snprintf(hex, sizeof(hex), "\\x%02X", c);
            result.append(hex);
            i++;
        }
    }

    return result;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tensor& tensor) {
    tt::stl::reflection::operator<<(os, tensor);
    // Append layout and storage_type information after the standard reflection output
    os << ",layout=" << tensor.layout();
    os << ",storage_type=";
    switch (tensor.storage_type()) {
        case tt::tt_metal::StorageType::HOST: os << "StorageType::HOST"; break;
        case tt::tt_metal::StorageType::DEVICE: os << "StorageType::DEVICE"; break;
        default: os << "StorageType::UNKNOWN"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::IDevice* device) {
    if (device) {
        os << device->id();
    } else {
        os << "nullptr";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<unsigned char, tt::tt_metal::QueueIdTag>& value) {
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

template <typename... Types>
std::ostream& operator<<(std::ostream& os, const std::variant<Types...>& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
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

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::experimental::GlobalCircularBuffer& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::xy_pair& value) {
    os << value.str();
    return os;
}

// TileReshapeMapMode operator moved to ttnn/operations/data_movement/graph_serialization.hpp
// EmbeddingsType operator moved to ttnn/operations/embedding/graph_serialization.hpp

// LayerNorm and Softmax operators moved to ttnn/operations/normalization/graph_serialization.hpp

// SDPA, Conv2d, OpSlicing, ComputeKernel operators moved to respective graph_serialization.hpp files

// Matmul program config operators moved to ttnn/operations/matmul/graph_serialization.hpp

// BasicUnaryWithParam operators moved to ttnn/operations/eltwise/graph_serialization.hpp

// Variant operators - must be defined after all element type operators
std::ostream& operator<<(std::ostream& os, const std::variant<float, tt::tt_metal::Tensor>& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

std::ostream& operator<<(
    std::ostream& os, const std::variant<std::string, ttnn::operations::unary::BasicUnaryWithParam<float>>& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

// LayerNorm and Softmax variant operators moved to ttnn/operations/normalization/graph_serialization.hpp

// Compute kernel config variant operator moved to ttnn/operations/core/graph_serialization.hpp

// Matmul program config variant operator moved to ttnn/operations/matmul/graph_serialization.hpp

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
    // Try to use operator<<, but fallback to type name if it doesn't exist
    if constexpr (requires { oss << value; }) {
        oss << value;
    } else {
        oss << "<" << graph_demangle(typeid(value).name()) << ">";
    }
}

// Specialization for ttsl::optional_reference
template <typename T>
void serialize_value(std::ostringstream& oss, const ttsl::optional_reference<T>& value) {
    if (value.has_value()) {
        serialize_value(oss, value.value());
    } else {
        oss << "nullopt";
    }
}

template <typename... Types>
void serialize_value(std::ostringstream& oss, const std::variant<Types...>& value) {
    std::visit([&oss](const auto& v) { serialize_value(oss, v); }, value);
}

// Helper function to serialize a SmallVector
template <typename T, std::size_t N>
void serialize_small_vector(std::ostringstream& oss, const ttsl::SmallVector<T, N>& vec) {
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        serialize_value(oss, vec[i]);
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
}

std::string graph_demangle(const std::string_view name) {
    int status = -4;
    char* res = abi::__cxa_demangle(name.data(), nullptr, nullptr, &status);
    const char* const demangled_name = (status == 0) ? res : name.data();
    std::string ret_val(demangled_name);
    free(res);  // NOLINT(cppcoreguidelines-no-malloc)
    return ret_val;
}

GraphArgumentSerializer::GraphArgumentSerializer() {
    // DO NOT call initialize() here!
    // initialize() must be called explicitly at runtime, after all static objects are constructed
}

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
        if (value.type() == typeid(std::reference_wrapper<ttsl::SmallVector<T, N>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<T, N>>>(value);
            const auto& vec = referenced_value.get();
            serialize_small_vector(oss, vec);
        } else if (value.type() == typeid(std::reference_wrapper<const ttsl::SmallVector<T, N>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const ttsl::SmallVector<T, N>>>(value);
            const auto& vec = referenced_value.get();
            serialize_small_vector(oss, vec);
        }

        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<T, N>>)] = conversion_function;
    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<T, N>>)] = conversion_function;
    registry()[typeid(const std::reference_wrapper<ttsl::SmallVector<T, N>>)] = conversion_function;
    // Skip SmallVector<const T, N> registration to avoid compilation errors with std::reference_wrapper
    // when used with certain types (e.g., std::string, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig,
    // tt::tt_metal::Shape, tt::tt_metal::DataType) due to protected member access issues in llvm_small_vector.hpp
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_array() {
    auto array_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;

        const void* arr = nullptr;
        size_t arr_size = 0;
        if (value.type() == typeid(std::reference_wrapper<std::array<T, N>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<std::array<T, N>>>(value);
            arr = &referenced_value.get();
            arr_size = referenced_value.get().size();
        } else if (value.type() == typeid(std::reference_wrapper<const std::array<T, N>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const std::array<T, N>>>(value);
            arr = &referenced_value.get();
            arr_size = referenced_value.get().size();
        } else if (value.type() == typeid(std::reference_wrapper<std::array<const T, N>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<std::array<const T, N>>>(value);
            arr = &referenced_value.get();
            arr_size = referenced_value.get().size();
        } else {
            oss << "Unable to parse" << graph_demangle(value.type().name());
            return oss.str();
        }

        oss << "[";
        for (size_t i = 0; i < arr_size; ++i) {
            if (value.type() == typeid(std::reference_wrapper<std::array<const T, N>>)) {
                auto const_arr = static_cast<const std::array<const T, N>*>(arr);
                serialize_value(oss, (*const_arr)[i]);
            } else {
                auto const_arr = static_cast<const std::array<T, N>*>(arr);
                serialize_value(oss, (*const_arr)[i]);
            }
            if (i < arr_size - 1) {
                oss << ", ";
            }
        }
        oss << "]";

        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<std::array<T, N>>)] = array_function;
    registry()[typeid(std::reference_wrapper<const std::array<T, N>>)] = array_function;
    registry()[typeid(std::reference_wrapper<std::array<const T, N>>)] = array_function;
}

template <typename T>
void GraphArgumentSerializer::register_vector() {
    auto register_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto serialize_vector_contents = [&](const std::vector<T>& vec) {
            oss << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                serialize_value(oss, vec[i]);
                if (i < vec.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
        };

        if (value.type() == typeid(std::reference_wrapper<std::vector<T>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<std::vector<T>>>(value);
            serialize_vector_contents(referenced_value.get());
        } else if (value.type() == typeid(std::reference_wrapper<const std::vector<T>>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const std::vector<T>>>(value);
            serialize_vector_contents(referenced_value.get());
        } else {
            oss << "Unable to parse" << graph_demangle(value.type().name());
            return oss.str();
        }

        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<std::vector<T>>)] = register_function;
    registry()[typeid(std::reference_wrapper<const std::vector<T>>)] = register_function;
}

template <typename OptionalT>
void GraphArgumentSerializer::register_optional_type() {
    auto register_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        // Handle basic optional
        if (value.type() == typeid(std::reference_wrapper<OptionalT>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<OptionalT>>(value);
            auto& referenced_optional = referenced_value.get();
            if (referenced_optional.has_value()) {
                serialize_value(oss, *referenced_optional);
            } else {
                oss << "nullopt";
            }
        } else {
            auto handle_optional_array = [&]<std::size_t N>() -> bool {
                if (value.type() == typeid(std::reference_wrapper<std::array<OptionalT, N>>)) {
                    auto referenced_value = std::any_cast<std::reference_wrapper<std::array<OptionalT, N>>>(value);
                    auto& referenced_optional = referenced_value.get();
                    oss << "[";
                    for (std::size_t i = 0; i < N; ++i) {
                        if (i > 0) {
                            oss << ", ";
                        }
                        if (referenced_optional[i].has_value()) {
                            serialize_value(oss, *referenced_optional[i]);
                        } else {
                            oss << "nullopt";
                        }
                    }
                    oss << "]";
                    return true;
                }
                return false;
            };

            // This is not the most elegant solution in the world, but since the api
            // is flexible enough to allow things like this:
            // std::reference_wrapper<std::optional<
            // std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > >
            // > It gets tricky to do a generic solution.
            // Try all array sizes from 1 to 16
            bool handled = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return (handle_optional_array.template operator()<(Is + 1)>() || ...);
            }(std::make_index_sequence<16>{});

            if (!handled) {
                oss << "Unable to parse" << graph_demangle(value.type().name());
            }
        }

        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<OptionalT>)] = register_function;

    // Register array types from 1 to 16
    [this, &register_function]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((registry()[typeid(std::reference_wrapper<std::array<OptionalT, (Is + 1)>>)] = register_function), ...);
    }(std::make_index_sequence<16>{});
}

template <typename T>
void GraphArgumentSerializer::register_optional_reference_type() {
    registry()[typeid(std::reference_wrapper<ttsl::optional_reference<T>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::optional_reference<T>>>(value);
        auto& referenced_optional_reference = referenced_value.get();
        if (referenced_optional_reference.has_value()) {
            serialize_value(oss, referenced_optional_reference.value());
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
        } else if (value.type() == typeid(std::reference_wrapper<const T>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const T>>(value);
            serialize_value(oss, referenced_value.get());
        } else if (value.type() == typeid(std::reference_wrapper<T*>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<T*>>(value);
            serialize_value(oss, referenced_value.get());
        } else if (value.type() == typeid(std::reference_wrapper<const T*>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const T*>>(value);
            serialize_value(oss, referenced_value.get());
        } else {
            oss << "Unable to parse" << graph_demangle(value.type().name());
        }

        return oss.str();
    };

    // regular cases
    registry()[typeid(std::reference_wrapper<T>)] = regular_function;
    registry()[typeid(std::reference_wrapper<const T>)] = regular_function;
    registry()[typeid(std::reference_wrapper<const T>)] = regular_function;
    registry()[typeid(const std::reference_wrapper<T>)] = regular_function;
    registry()[typeid(std::reference_wrapper<T*>)] = regular_function;
    registry()[typeid(std::reference_wrapper<const T*>)] = regular_function;
    registry()[typeid(const std::reference_wrapper<T*>)] = regular_function;
    registry()[typeid(const std::reference_wrapper<const T*>)] = regular_function;

    // Particular cases for optional (only if T is not abstract)
    if constexpr (!std::is_abstract_v<T>) {
        register_optional_type<std::optional<T>>();
    }
    // register optional types (only if T is not abstract)
    if constexpr (!std::is_abstract_v<T>) {
        register_optional_type<const std::optional<T>>();
        register_optional_type<std::optional<const T>>();
        register_optional_type<const std::optional<const T>>();
    }

    register_optional_reference_type<ttsl::optional_reference<T>>();
    register_optional_reference_type<ttsl::optional_reference<const T>>();

    // Handle complex types (feel free to add more in the future)
    // Register small vectors from 2 to 16
    [this]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((register_small_vector<T, (Is + 1)>()), ...);
    }(std::make_index_sequence<15>{});  // 15 elements: 2,3,4,...,16

    // Skip array registration for bool type due to std::array<bool, N> serialization issues
    if constexpr (!std::is_same_v<T, bool> && !std::is_abstract_v<T>) {
        // Register arrays for all sizes from 2 to 16
        [this]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((register_array<T, (Is + 1)>()), ...);
        }(std::make_index_sequence<15>{});  // 15 elements: 2,3,4,...,16

        register_vector<T>();

        // Register optional arrays for all sizes from 2 to 16
        [this]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((register_optional_type<std::optional<std::array<T, (Is + 1)>>>()), ...);
        }(std::make_index_sequence<15>{});  // 15 elements: 2,3,4,...,16
    }

    // register vector of optional (only if T is not abstract)
    if constexpr (!std::is_abstract_v<T>) {
        register_vector<std::optional<T>>();
    }
}

std::vector<std::string> GraphArgumentSerializer::to_list(const std::span<std::any>& span) {
    // By the time to_list() is called, all static objects have been constructed
    instance().initialize();

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
            oss << "[ unsupported type" << " , ";
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
    // Make initialize() idempotent - safe to call multiple times
    if (initialized_) {
        return;  // Already initialized, nothing to do
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // CRITICAL: Execute all queued operation-specific registrations FIRST
    // This must happen before any graph capture can start
    // Operations self-register via TTNN_REGISTER_GRAPH_ARG macro in their headers
    // ═══════════════════════════════════════════════════════════════════════════════
    GraphArgumentRegistrationQueue::instance().execute_all();

    // Now register core types
    GraphArgumentSerializer::register_type<bool>();
    GraphArgumentSerializer::register_type<int>();
    GraphArgumentSerializer::register_type<unsigned int>();
    GraphArgumentSerializer::register_type<unsigned long>();
    GraphArgumentSerializer::register_type<uint>();
    GraphArgumentSerializer::register_type<long>();
    GraphArgumentSerializer::register_type<float>();
    GraphArgumentSerializer::register_type<uint8_t>();
    GraphArgumentSerializer::register_type<uint16_t>();
    GraphArgumentSerializer::register_type<double>();
    GraphArgumentSerializer::register_type<uint32_t>();
    GraphArgumentSerializer::register_type<uint64_t>();
    GraphArgumentSerializer::register_type<int32_t>();

    // Custom handler for std::string to sanitize UTF-8
    GraphArgumentSerializer::ConvertionFunction string_function = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        if (value.type() == typeid(std::reference_wrapper<std::string>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<std::string>>(value);
            oss << sanitize_utf8(referenced_value.get());
        } else if (value.type() == typeid(std::reference_wrapper<const std::string>)) {
            auto referenced_value = std::any_cast<std::reference_wrapper<const std::string>>(value);
            oss << sanitize_utf8(referenced_value.get());
        } else {
            oss << "Unable to parse string";
        }
        return oss.str();
    };
    registry()[typeid(std::reference_wrapper<std::string>)] = string_function;
    registry()[typeid(std::reference_wrapper<const std::string>)] = string_function;
    registry()[typeid(const std::reference_wrapper<std::string>)] = string_function;
    GraphArgumentSerializer::register_type<tt::tt_metal::DataType>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Layout>();
    GraphArgumentSerializer::register_type<tt::tt_metal::MemoryConfig>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Shape>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tensor>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tile>();
    GraphArgumentSerializer::register_type<tt::tt_metal::QueueId>();
    GraphArgumentSerializer::register_type<ttnn::types::CoreGrid>();
    GraphArgumentSerializer::register_type<std::variant<float, int>>();
    GraphArgumentSerializer::register_type<std::variant<int, float>>();
    GraphArgumentSerializer::register_type<std::variant<unsigned int, float>>();
    GraphArgumentSerializer::register_type<std::variant<float, unsigned int>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 2>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 4>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 8>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 16>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 8ul>>>();
    GraphArgumentSerializer::register_type<std::variant<float, tt::tt_metal::Tensor>>();
    GraphArgumentSerializer::register_type<
        std::set<tt::tt_metal::distributed::MeshCoordinate, std::less<tt::tt_metal::distributed::MeshCoordinate>>>();
    GraphArgumentSerializer::register_type<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>();
    GraphArgumentSerializer::register_type<ttnn::types::CoreRangeSet>();
    GraphArgumentSerializer::register_type<signed char>();
    // std::nullopt_t cannot be used as a template parameter for std::optional
    GraphArgumentSerializer::register_type<tt::tt_metal::distributed::MeshDevice>();
    GraphArgumentSerializer::register_type<tt::tt_metal::distributed::MeshCoordinate>();
    GraphArgumentSerializer::register_type<tt::tt_metal::GlobalSemaphore>();
    GraphArgumentSerializer::register_type<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>();
    GraphArgumentSerializer::register_type<
        ttsl::StrongType<unsigned long, tt::tt_metal::experimental::GlobalCircularBuffer>>();

    GraphArgumentSerializer::register_type<tt::tt_metal::experimental::GlobalCircularBuffer>();
    GraphArgumentSerializer::register_type<tt::tt_metal::IDevice>();

    // Fabric topology
    GraphArgumentSerializer::register_type<tt::tt_fabric::Topology>();

    // xy_pair
    GraphArgumentSerializer::register_type<tt::xy_pair>();

    // Note: std::nullopt_t is already handled specially and cannot be registered as a template parameter
    // Note: Operation-specific types are auto-registered at the START of initialize() via execute_all()

    // Mark as initialized
    initialized_ = true;
}

}  // namespace ttnn::graph
