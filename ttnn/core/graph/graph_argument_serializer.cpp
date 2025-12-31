// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_argument_serializer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <tt_stl/small_vector.hpp>
#include <array>
#include <span>
#include <sstream>
#include <string>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer_types.hpp>

namespace ttnn::graph {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tile& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::StorageType& storage_type) {
    switch (storage_type) {
        case tt::tt_metal::StorageType::HOST: os << "StorageType::HOST"; break;
        case tt::tt_metal::StorageType::DEVICE: os << "StorageType::DEVICE"; break;
        default: os << "StorageType::UNKNOWN"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tensor& tensor) {
    // First serialize using reflection to get all tensor fields
    // The reflection output format is: Tensor(storage=..., tensor_spec=...)
    // We need to capture it, find the last ')', and insert storage_type and layout before it
    std::ostringstream reflection_stream;
    tt::stl::reflection::operator<<(reflection_stream, tensor);
    std::string reflection_output = reflection_stream.str();

    // Find the last closing parenthesis (the end of the Tensor(...) structure)
    size_t last_paren = reflection_output.rfind(')');
    if (last_paren != std::string::npos) {
        // Build the additional fields to insert
        std::ostringstream additional_fields;

        // Add storage_type
        tt::tt_metal::StorageType storage_type = tensor.storage_type();
        additional_fields << ", storage_type=";
        switch (storage_type) {
            case tt::tt_metal::StorageType::HOST: additional_fields << "StorageType::HOST"; break;
            case tt::tt_metal::StorageType::DEVICE: additional_fields << "StorageType::DEVICE"; break;
            default: additional_fields << "StorageType::UNKNOWN"; break;
        }

        // Add layout (TILE or ROW_MAJOR)
        tt::tt_metal::Layout layout = tensor.layout();
        additional_fields << ", layout=";
        switch (layout) {
            case tt::tt_metal::Layout::TILE: additional_fields << "Layout::TILE"; break;
            case tt::tt_metal::Layout::ROW_MAJOR: additional_fields << "Layout::ROW_MAJOR"; break;
            case tt::tt_metal::Layout::INVALID: additional_fields << "Layout::INVALID"; break;
            default: additional_fields << "Layout::UNKNOWN"; break;
        }

        reflection_output.insert(last_paren, additional_fields.str());
    } else {
        // Fallback: if no closing paren found, just append
        tt::tt_metal::StorageType storage_type = tensor.storage_type();
        tt::tt_metal::Layout layout = tensor.layout();

        reflection_output += ", storage_type=";
        switch (storage_type) {
            case tt::tt_metal::StorageType::HOST: reflection_output += "StorageType::HOST"; break;
            case tt::tt_metal::StorageType::DEVICE: reflection_output += "StorageType::DEVICE"; break;
            default: reflection_output += "StorageType::UNKNOWN"; break;
        }

        reflection_output += ", layout=";
        switch (layout) {
            case tt::tt_metal::Layout::TILE: reflection_output += "Layout::TILE"; break;
            case tt::tt_metal::Layout::ROW_MAJOR: reflection_output += "Layout::ROW_MAJOR"; break;
            case tt::tt_metal::Layout::INVALID: reflection_output += "Layout::INVALID"; break;
            default: reflection_output += "Layout::UNKNOWN"; break;
        }
    }

    os << reflection_output;
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

template <typename T, typename U, typename V>
std::ostream& operator<<(std::ostream& os, const std::variant<T, U, V>& variant) {
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

std::ostream& operator<<(std::ostream& os, const ttnn::operations::embedding::EmbeddingsType& embeddings_type) {
    switch (embeddings_type) {
        case ttnn::operations::embedding::EmbeddingsType::GENERIC: os << "GENERIC"; break;
        case ttnn::operations::embedding::EmbeddingsType::PADDED: os << "PADDED"; break;
        case ttnn::operations::embedding::EmbeddingsType::BINARY: os << "BINARY"; break;
        default: os << "UNKNOWN"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ttnn::TileReshapeMapMode& mode) {
    switch (mode) {
        case ttnn::TileReshapeMapMode::CACHE: os << "CACHE"; break;
        case ttnn::TileReshapeMapMode::RECREATE: os << "RECREATE"; break;
        default: os << "UNKNOWN"; break;
    }
    return os;
}

template <typename T, typename Tag>
std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<T, Tag>& strong_type) {
    os << *strong_type;
    return os;
}

std::ostream& operator<<(std::ostream& os, const ttnn::operations::transformer::SDPAProgramConfig& config) {
    os << "SDPAProgramConfig{compute_grid=" << config.compute_with_storage_grid_size.x << ","
       << config.compute_with_storage_grid_size.y << ", q_chunk_size=" << config.q_chunk_size
       << ", k_chunk_size=" << config.k_chunk_size << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::GlobalSemaphore& semaphore) {
    os << "GlobalSemaphore(...)";  // Simplified representation
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_fabric::Topology& topology) {
    os << "Topology(...)";  // Simplified representation
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::CoreRangeSet& core_range_set) {
    os << "CoreRangeSet(...)";  // Simplified representation
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::experimental::GlobalCircularBuffer& gcb) {
    os << "GlobalCircularBuffer(...)";  // Simplified representation
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::TensorMemoryLayout& layout) {
    switch (layout) {
        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: os << "TensorMemoryLayout::INTERLEAVED"; break;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: os << "TensorMemoryLayout::WIDTH_SHARDED"; break;
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: os << "TensorMemoryLayout::HEIGHT_SHARDED"; break;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: os << "TensorMemoryLayout::BLOCK_SHARDED"; break;
        default: os << "TensorMemoryLayout::UNKNOWN"; break;
    }
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::span<T>& span) {
    os << "[";
    for (size_t i = 0; i < span.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << span[i];
    }
    os << "]";
    return os;
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
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<T, N>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<T, N>>>(value);
        oss << referenced_value.get();
        return oss.str();
    };
}

template <typename T, std::size_t N>
void GraphArgumentSerializer::register_array() {
    registry()[typeid(std::reference_wrapper<std::array<T, N>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::array<T, N>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::array<T, N>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::array<T, N>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
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
            oss << *referenced_optional;
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
    GraphArgumentSerializer::register_type<double>();
    GraphArgumentSerializer::register_type<std::string>();
    GraphArgumentSerializer::register_type<tt::tt_metal::DataType>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Layout>();
    GraphArgumentSerializer::register_type<tt::tt_metal::MemoryConfig>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Shape>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tensor>();
    GraphArgumentSerializer::register_type<tt::tt_metal::Tile>();
    GraphArgumentSerializer::register_type<tt::stl::StrongType<uint8_t, tt::tt_metal::QueueIdTag>>();
    GraphArgumentSerializer::register_type<ttnn::types::CoreGrid>();
    GraphArgumentSerializer::register_type<ttnn::operations::embedding::EmbeddingsType>();
    GraphArgumentSerializer::register_type<ttnn::TileReshapeMapMode>();
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
    GraphArgumentSerializer::register_type<std::variant<float, int, tt::tt_metal::Tensor>>();
    GraphArgumentSerializer::register_type<std::variant<int, float, tt::tt_metal::Tensor>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 2>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 4>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 8>>>();
    GraphArgumentSerializer::register_type<std::variant<int, ttsl::SmallVector<int, 16>>>();

    // Register std::array types commonly used for pad operations
    // We register these directly without going through register_type to avoid recursive template expansion
    registry()[typeid(std::reference_wrapper<std::array<unsigned int, 2>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::array<unsigned int, 2>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < 2; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::array<unsigned int, 2>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::array<unsigned int, 2>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < 2; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<std::array<unsigned int, 4>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::array<unsigned int, 4>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < 4; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::array<unsigned int, 4>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::array<unsigned int, 4>>>(value);
        const auto& arr = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < 4; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    };

    // Register SmallVector<std::array<unsigned int, 2>, 8> for pad operations
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<std::array<unsigned int, 2>, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<ttsl::SmallVector<std::array<unsigned int, 2>, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "[" << vec[i][0] << ", " << vec[i][1] << "]";
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<std::array<unsigned int, 2>, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const ttsl::SmallVector<std::array<unsigned int, 2>, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "[" << vec[i][0] << ", " << vec[i][1] << "]";
        }
        oss << "]";
        return oss.str();
    };

    // Register SmallVector<long, 8> for permute operations (dims parameter)
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<long, 8>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<long, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<long, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const ttsl::SmallVector<long, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    // Register SmallVector<BasicUnaryWithParam<float, int, unsigned int>, 8> for unary operations
    registry()[typeid(std::reference_wrapper<
                      ttsl::SmallVector<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int>, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<
            ttsl::SmallVector<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int>, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "BasicUnaryWithParam(...)";
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(
        std::reference_wrapper<
            const ttsl::SmallVector<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int>, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<
            const ttsl::SmallVector<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int>, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "BasicUnaryWithParam(...)";
        }
        oss << "]";
        return oss.str();
    };

    // Register SmallVector<int, 8> for reshape operations (shape parameter)
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<int, 8>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<int, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<int, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const ttsl::SmallVector<int, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    // Register std::variant<std::array<uint, 2>, std::array<uint, 4>> for conv2d stride/dilation
    registry()[typeid(std::reference_wrapper<std::variant<std::array<unsigned int, 2>, std::array<unsigned int, 4>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<std::variant<std::array<unsigned int, 2>, std::array<unsigned int, 4>>>>(value);
        const auto& var = referenced_value.get();

        if (std::holds_alternative<std::array<unsigned int, 2>>(var)) {
            const auto& arr = std::get<std::array<unsigned int, 2>>(var);
            oss << "[" << arr[0] << ", " << arr[1] << "]";
        } else {
            const auto& arr = std::get<std::array<unsigned int, 4>>(var);
            oss << "[" << arr[0] << ", " << arr[1] << ", " << arr[2] << ", " << arr[3] << "]";
        }
        return oss.str();
    };

    // Register std::nullopt_t for optional parameters
    registry()[typeid(std::reference_wrapper<std::nullopt_t>)] = [](const std::any&) -> std::string {
        return "nullopt";
    };

    registry()[typeid(std::reference_wrapper<const std::nullopt_t>)] = [](const std::any&) -> std::string {
        return "nullopt";
    };

    // Register TensorMemoryLayout type
    GraphArgumentSerializer::register_type<tt::tt_metal::TensorMemoryLayout>();

    // Register std::reference_wrapper<std::optional<tt::tt_metal::TensorMemoryLayout const> const> for max_pool2d
    // applied_shard_scheme
    registry()[typeid(std::reference_wrapper<const std::optional<const tt::tt_metal::TensorMemoryLayout>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::optional<const tt::tt_metal::TensorMemoryLayout>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            const auto& layout = opt.value();
            switch (layout) {
                case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: oss << "TensorMemoryLayout::INTERLEAVED"; break;
                case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: oss << "TensorMemoryLayout::WIDTH_SHARDED"; break;
                case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
                    oss << "TensorMemoryLayout::HEIGHT_SHARDED";
                    break;
                case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: oss << "TensorMemoryLayout::BLOCK_SHARDED"; break;
                default: oss << "TensorMemoryLayout::UNKNOWN"; break;
            }
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // Register std::reference_wrapper<std::variant<int, std::array<unsigned int, 2>>> for upsample scale_factor
    registry()[typeid(std::reference_wrapper<std::variant<int, std::array<unsigned int, 2>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::variant<int, std::array<unsigned int, 2>>>>(value);
        const auto& var = referenced_value.get();

        if (std::holds_alternative<int>(var)) {
            oss << std::get<int>(var);
        } else {
            const auto& arr = std::get<std::array<unsigned int, 2>>(var);
            oss << "[" << arr[0] << ", " << arr[1] << "]";
        }
        return oss.str();
    };

    // Register const version for upsample scale_factor
    registry()[typeid(std::reference_wrapper<const std::variant<int, std::array<unsigned int, 2>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::variant<int, std::array<unsigned int, 2>>>>(value);
        const auto& var = referenced_value.get();

        if (std::holds_alternative<int>(var)) {
            oss << std::get<int>(var);
        } else {
            const auto& arr = std::get<std::array<unsigned int, 2>>(var);
            oss << "[" << arr[0] << ", " << arr[1] << "]";
        }
        return oss.str();
    };

    // Register additional types for graph tracing
    GraphArgumentSerializer::register_type<ttnn::operations::transformer::SDPAProgramConfig>();
    GraphArgumentSerializer::register_type<tt::tt_metal::GlobalSemaphore>();
    GraphArgumentSerializer::register_type<tt::tt_fabric::Topology>();

    // Register LayerNorm program config variants
    GraphArgumentSerializer::register_type<std::variant<
        ttnn::operations::normalization::LayerNormDefaultProgramConfig,
        ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig>>();

    // Register optional types for various configurations

    // Register set types
    registry()[typeid(std::reference_wrapper<std::set<tt::tt_metal::distributed::MeshCoordinate>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::set<tt::tt_metal::distributed::MeshCoordinate>>>(value);
        const auto& set = referenced_value.get();
        oss << "{";
        bool first = true;
        for ([[maybe_unused]] const auto& coord : set) {
            if (!first) {
                oss << ", ";
            }
            oss << "MeshCoordinate(...)";  // Simplified representation
            first = false;
        }
        oss << "}";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::set<tt::tt_metal::distributed::MeshCoordinate>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::set<tt::tt_metal::distributed::MeshCoordinate>>>(value);
        const auto& set = referenced_value.get();
        oss << "{";
        bool first = true;
        for ([[maybe_unused]] const auto& coord : set) {
            if (!first) {
                oss << ", ";
            }
            oss << "MeshCoordinate(...)";  // Simplified representation
            first = false;
        }
        oss << "}";
        return oss.str();
    };

    // Register vector types commonly used in operations
    registry()[typeid(std::reference_wrapper<std::vector<unsigned int>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::vector<unsigned int>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::vector<unsigned int>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::vector<unsigned int>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    // Register std::vector<tt::tt_metal::Tensor> for operations that take multiple tensors
    // Properly serialize each tensor using reflection to capture full tensor information
    registry()[typeid(std::reference_wrapper<std::vector<tt::tt_metal::Tensor>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::vector<tt::tt_metal::Tensor>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            // Serialize each tensor using reflection to get full tensor information (shape, dtype, layout,
            // memory_config)
            tt::stl::reflection::operator<<(oss, vec[i]);
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::vector<tt::tt_metal::Tensor>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::vector<tt::tt_metal::Tensor>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            // Serialize each tensor using reflection to get full tensor information (shape, dtype, layout,
            // memory_config)
            tt::stl::reflection::operator<<(oss, vec[i]);
        }
        oss << "]";
        return oss.str();
    };

    // Register std::vector<tt::tt_metal::GlobalSemaphore> for operations that use global semaphores
    registry()[typeid(std::reference_wrapper<std::vector<tt::tt_metal::GlobalSemaphore>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::vector<tt::tt_metal::GlobalSemaphore>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "GlobalSemaphore(...)";  // Simplified representation
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::vector<tt::tt_metal::GlobalSemaphore>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::vector<tt::tt_metal::GlobalSemaphore>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "GlobalSemaphore(...)";  // Simplified representation
        }
        oss << "]";
        return oss.str();
    };

    // Register additional complex types
    registry()[typeid(std::reference_wrapper<signed char>)] = [](const std::any& value) -> std::string {
        auto referenced_value = std::any_cast<std::reference_wrapper<signed char>>(value);
        return std::to_string(static_cast<int>(referenced_value.get()));
    };

    registry()[typeid(std::reference_wrapper<const signed char>)] = [](const std::any& value) -> std::string {
        auto referenced_value = std::any_cast<std::reference_wrapper<const signed char>>(value);
        return std::to_string(static_cast<int>(referenced_value.get()));
    };

    // Register tuple types (simplified)
    registry()[typeid(std::reference_wrapper<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>)] =
        [](const std::any& value) -> std::string {
        return "(Tensor, Tensor)";  // Simplified representation for tensor tuple
    };

    registry()[typeid(std::reference_wrapper<const std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>)] =
        [](const std::any& value) -> std::string {
        return "(Tensor, Tensor)";  // Simplified representation for tensor tuple
    };

    // Register array types
    registry()[typeid(std::reference_wrapper<std::array<tt::tt_metal::Tensor, 3>>)] =
        [](const std::any& value) -> std::string {
        return "[Tensor, Tensor, Tensor]";  // Simplified representation for tensor array
    };

    registry()[typeid(std::reference_wrapper<const std::array<tt::tt_metal::Tensor, 3>>)] =
        [](const std::any& value) -> std::string {
        return "[Tensor, Tensor, Tensor]";  // Simplified representation for tensor array
    };

    // Register std::span types for slice operations
    registry()[typeid(std::reference_wrapper<std::span<const int>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::span<const int>>>(value);
        const auto& span = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < span.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << span[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const std::span<const int>>)] = [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::span<const int>>>(value);
        const auto& span = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < span.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << span[i];
        }
        oss << "]";
        return oss.str();
    };

    // Register optional types wrapped in reference_wrapper
    // std::optional<CoreRangeSet> const
    registry()[typeid(std::reference_wrapper<const std::optional<tt::tt_metal::CoreRangeSet>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::optional<tt::tt_metal::CoreRangeSet>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "CoreRangeSet(...)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<GlobalCircularBuffer const> const
    registry()[typeid(
        std::reference_wrapper<const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "GlobalCircularBuffer(...)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<StrongType<unsigned char, SubDeviceIdTag>> const
    registry()[typeid(
        std::reference_wrapper<const std::optional<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<const std::optional<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>>>(
            value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << static_cast<int>(*opt.value());
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::vector<std::optional<Tensor>>>
    registry()[typeid(std::reference_wrapper<std::optional<std::vector<std::optional<tt::tt_metal::Tensor>>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::optional<std::vector<std::optional<tt::tt_metal::Tensor>>>>>(
                value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "[";
            const auto& vec = opt.value();
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                if (vec[i].has_value()) {
                    oss << "Tensor(...)";
                } else {
                    oss << "nullopt";
                }
            }
            oss << "]";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::set<MeshCoordinate, ...>> const
    registry()[typeid(std::reference_wrapper<const std::optional<const std::set<
                          tt::tt_metal::distributed::MeshCoordinate,
                          std::less<tt::tt_metal::distributed::MeshCoordinate>,
                          std::allocator<tt::tt_metal::distributed::MeshCoordinate>>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const std::optional<const std::set<
            tt::tt_metal::distributed::MeshCoordinate,
            std::less<tt::tt_metal::distributed::MeshCoordinate>,
            std::allocator<tt::tt_metal::distributed::MeshCoordinate>>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "{MeshCoordinate(...)}";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::set<MeshCoordinate, ...>> const (non-const reference_wrapper)
    registry()[typeid(std::reference_wrapper<std::optional<const std::set<
                          tt::tt_metal::distributed::MeshCoordinate,
                          std::less<tt::tt_metal::distributed::MeshCoordinate>,
                          std::allocator<tt::tt_metal::distributed::MeshCoordinate>>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<std::optional<const std::set<
            tt::tt_metal::distributed::MeshCoordinate,
            std::less<tt::tt_metal::distributed::MeshCoordinate>,
            std::allocator<tt::tt_metal::distributed::MeshCoordinate>>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "{MeshCoordinate(...)}";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::array<Tensor, 3>>
    registry()[typeid(std::reference_wrapper<std::optional<std::array<tt::tt_metal::Tensor, 3>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::optional<std::array<tt::tt_metal::Tensor, 3>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "[Tensor, Tensor, Tensor]";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::tuple<Tensor, Tensor>>
    registry()[typeid(std::reference_wrapper<std::optional<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<std::optional<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "(Tensor, Tensor)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::tuple<Tensor, Tensor>> const
    registry()[typeid(
        std::reference_wrapper<const std::optional<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<const std::optional<std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "(Tensor, Tensor)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<std::variant<std::string, BasicUnaryWithParam<float>> const> const
    registry()[typeid(std::reference_wrapper<const std::optional<
                          const std::variant<std::string, ttnn::operations::unary::BasicUnaryWithParam<float>>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<
            const std::optional<const std::variant<std::string, ttnn::operations::unary::BasicUnaryWithParam<float>>>>>(
            value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            const auto& var = opt.value();
            if (std::holds_alternative<std::string>(var)) {
                oss << "\"" << std::get<std::string>(var) << "\"";
            } else {
                oss << "BasicUnaryWithParam(...)";
            }
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<BasicUnaryWithParam<float>>
    registry()[typeid(std::reference_wrapper<std::optional<ttnn::operations::unary::BasicUnaryWithParam<float>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<std::optional<ttnn::operations::unary::BasicUnaryWithParam<float>>>>(
                value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "BasicUnaryWithParam(...)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<StrongType<unsigned char, SubDeviceIdTag>>
    registry()[typeid(
        std::reference_wrapper<std::optional<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<std::optional<ttsl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag>>>>(
            value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << static_cast<int>(*opt.value());
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // MeshDevice* pointer type
    registry()[typeid(std::reference_wrapper<tt::tt_metal::distributed::MeshDevice*>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice*>>(value);
        auto ptr = referenced_value.get();
        if (ptr) {
            oss << "MeshDevice(ptr)";
        } else {
            oss << "nullptr";
        }
        return oss.str();
    };

    // std::optional<Conv2dConfig const> const
    registry()[typeid(
        std::reference_wrapper<const std::optional<const ttnn::operations::conv::conv2d::Conv2dConfig>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<const std::optional<const ttnn::operations::conv::conv2d::Conv2dConfig>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "Conv2dConfig(...)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // std::optional<Op2DSliceConfig const> const
    registry()[typeid(
        std::reference_wrapper<const std::optional<const ttnn::operations::op_slicing::Op2DSliceConfig>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<
            std::reference_wrapper<const std::optional<const ttnn::operations::op_slicing::Op2DSliceConfig>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            oss << "Op2DSliceConfig(...)";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };

    // LayerNormDistributedDefaultProgramConfig const
    registry()[typeid(
        std::reference_wrapper<const ttnn::operations::normalization::LayerNormDistributedDefaultProgramConfig>)] =
        [](const std::any&) -> std::string { return "LayerNormDistributedDefaultProgramConfig(...)"; };

    registry()[typeid(
        std::reference_wrapper<ttnn::operations::normalization::LayerNormDistributedDefaultProgramConfig>)] =
        [](const std::any&) -> std::string { return "LayerNormDistributedDefaultProgramConfig(...)"; };

    // std::variant<float, int, Tensor> const
    registry()[typeid(std::reference_wrapper<const std::variant<float, int, tt::tt_metal::Tensor>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::variant<float, int, tt::tt_metal::Tensor>>>(value);
        const auto& var = referenced_value.get();
        if (std::holds_alternative<float>(var)) {
            oss << std::get<float>(var);
        } else if (std::holds_alternative<int>(var)) {
            oss << std::get<int>(var);
        } else {
            oss << "Tensor(...)";
        }
        return oss.str();
    };

    // SmallVector<unsigned int, 8>
    registry()[typeid(std::reference_wrapper<ttsl::SmallVector<unsigned int, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<ttsl::SmallVector<unsigned int, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    registry()[typeid(std::reference_wrapper<const ttsl::SmallVector<unsigned int, 8>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value = std::any_cast<std::reference_wrapper<const ttsl::SmallVector<unsigned int, 8>>>(value);
        const auto& vec = referenced_value.get();
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    };

    // std::optional<SmallVector<unsigned int, 8>> const
    registry()[typeid(std::reference_wrapper<const std::optional<ttsl::SmallVector<unsigned int, 8>>>)] =
        [](const std::any& value) -> std::string {
        std::ostringstream oss;
        auto referenced_value =
            std::any_cast<std::reference_wrapper<const std::optional<ttsl::SmallVector<unsigned int, 8>>>>(value);
        const auto& opt = referenced_value.get();
        if (opt.has_value()) {
            const auto& vec = opt.value();
            oss << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << vec[i];
            }
            oss << "]";
        } else {
            oss << "nullopt";
        }
        return oss.str();
    };
}
}  // namespace ttnn::graph
