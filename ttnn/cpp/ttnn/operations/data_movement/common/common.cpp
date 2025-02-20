// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include "cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "cpp/ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

ttnn::Shape squeeze_shape_to_4D(ttnn::Shape shape) {
    if (shape.rank() <= 4) {
        return shape;
    }
    std::array<uint32_t, 4> shape_4d;
    shape_4d[0] = 1;
    int extra_rank = shape.rank() - 4;
    for (int i = extra_rank; i >= 0; i--) {
        shape_4d[0] *= shape[i];
    }
    shape_4d[1] = shape[1 + extra_rank];
    shape_4d[2] = shape[2 + extra_rank];
    shape_4d[3] = shape[3 + extra_rank];
    return ttnn::Shape(shape_4d);
}

ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor) {
    auto shape = tensor.get_logical_shape();
    auto rank = shape.rank();
    TT_FATAL(shape.rank() >= 4, "Tensor has to be of rank larger than 4! Instead is {}", shape.rank());
    if (rank == 4) {
        return tensor;
    }
    int i = 0;
    // This is a workaround for now, it will be fixed in another PR
    if (shape[i] == 1) {
        auto squeezed = tensor;
        while (rank > 4 && shape[i] == 1) {
            squeezed = ttnn::squeeze(squeezed, 0);
            rank = squeezed.get_logical_shape().rank();
            i++;
        }
        if (rank <= 4) {
            return squeezed;
        }
        return ttnn::reshape(squeezed, squeeze_shape_to_4D(shape));
    }
    return ttnn::reshape(tensor, squeeze_shape_to_4D(shape));
}

ttnn::Tensor pad_to_tile_vol(
    QueueId queue_id,
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config) {
    auto logical_shape = tensor.get_logical_shape();
    auto padded_shape = tensor.get_padded_shape();
    auto rank = logical_shape.rank();
    if (padded_shape[-1] % tt::constants::TILE_WIDTH != 0 || padded_shape[-2] % tt::constants::TILE_HEIGHT != 0) {
        TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

        auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
        auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
        uint32_t num_non_hw_dims = rank - 2u;
        auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0, 0});
        padding_vec.reserve(rank);
        padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
        padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

        constexpr bool pad_use_multicore = true;
        auto padded_output = ttnn::pad(queue_id, tensor, padding_vec, value, use_multicore, memory_config);
        TT_FATAL(
            padded_output.get_padded_shape()[-1] % tt::constants::TILE_WIDTH == 0 &&
                padded_output.get_padded_shape()[-2] % tt::constants::TILE_HEIGHT == 0,
            "pad_to_tile_vol: output tensor must be divisible by tile size");
        return padded_output;
    }
    return tensor;
}
uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

std::array<uint32_t, 2> compute_block_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const tt::tt_metal::Layout& layout,
                                                          const tt::tt_metal::CoreCoord& grid_size,
                                                          const tt::tt_metal::ShardOrientation& orientation,
                                                          const uint32_t total_num_cores) {
    TT_FATAL(grid_size.y * grid_size.x == total_num_cores, "compute_block_sharded_shard_shape received a core grid shape that does not match the total number of cores");
    auto adjusted_grid_size = grid_size;
    if (orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
        // for col major, we partition the width of the tensor along the height of the core grid
        std::swap(adjusted_grid_size.x, adjusted_grid_size.y);
    }

    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto tensor_height_padded_to_tile =
        layout == tt::tt_metal::Layout::TILE
            ? tt::round_up(tensor_height, adjusted_grid_size.y * tt::constants::TILE_HEIGHT)
            : tensor_height;
    std::array<uint32_t, 2> shard_shape = {tt::div_up(tensor_height_padded_to_tile, adjusted_grid_size.y),
                                           tt::div_up(tensor_width, adjusted_grid_size.x)};

    return shard_shape;
}

std::array<uint32_t, 2> compute_width_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const uint32_t total_num_cores) {
    return {squeezed_tensor_hw[0], tt::div_up(squeezed_tensor_hw[1], total_num_cores)};
}

std::array<uint32_t, 2> compute_height_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                           const tt::tt_metal::Layout& layout,
                                                           const uint32_t total_num_cores) {
    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto squeezed_height_padded_to_tile = layout == tt::tt_metal::Layout::TILE
                                                    ? tt::round_up(tensor_height, total_num_cores)
                                                    : tensor_height;
    return {tt::div_up(squeezed_height_padded_to_tile, total_num_cores), tensor_width};
}

ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape,
    const tt::tt_metal::Layout& layout) {
    auto rank = logical_shape.rank();
    TT_FATAL(rank >= 2, "rank of tensor to shard must be at least 2.");

    ttnn::TensorMemoryLayout tensor_memory_layout;
    if (strategy == ShardStrategy::BLOCK) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    } else if (strategy == ShardStrategy::WIDTH) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    } else if (strategy == ShardStrategy::HEIGHT) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
    }

    auto height = logical_shape[-2];
    auto width = logical_shape[-1];
    std::array<uint32_t, 2> computed_shard_shape;

    if (shard_shape.has_value()) {
        computed_shard_shape = shard_shape.value();
    } else {
        uint32_t batch_size = 1;
        for (int i = 0; i < rank - 2; i++) {
            batch_size *= logical_shape[i];
        }

        auto tensor_height = batch_size * height;
        auto tensor_width = width;
        std::array<uint32_t, 2> squeezed_tensor_hw{tensor_height, tensor_width};
        auto total_num_cores = core_grid.num_cores();
        CoreCoord grid_size = core_grid.bounding_box().grid_size();

        switch (strategy) {
            case ShardStrategy::BLOCK:
                computed_shard_shape = compute_block_sharded_shard_shape(squeezed_tensor_hw, layout, grid_size, orientation, total_num_cores);
                break;
            case ShardStrategy::WIDTH:
                computed_shard_shape = compute_width_sharded_shard_shape(squeezed_tensor_hw, total_num_cores);
                break;
            case ShardStrategy::HEIGHT:
                computed_shard_shape = compute_height_sharded_shard_shape(squeezed_tensor_hw, layout, total_num_cores);
                break;
            default:
                TT_ASSERT(false, "Invalid shard strategy");
        }
    }

    if (layout == tt::tt_metal::Layout::TILE) {
        auto [shard_height, shard_width] = computed_shard_shape;
        auto tile_divides_shard_height = shard_height % tt::constants::TILE_HEIGHT == 0;
        auto tile_divides_shard_width = shard_width % tt::constants::TILE_WIDTH == 0;
        TT_FATAL(tile_divides_shard_width && tile_divides_shard_height,
                 "For sharding tiled tensors, the shard shape must fit neatly into tiles but "
                 "create_sharded_memory_config got shard width {} and shard height {} while "
                 "on this architecture we have tile width {} and tile height {}",
                 computed_shard_shape[0], computed_shard_shape[1], tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT);
    }

    auto shard_spec = tt::tt_metal::ShardSpec(core_grid, computed_shard_shape, orientation);
    return ttnn::MemoryConfig(tensor_memory_layout, ttnn::BufferType::L1, shard_spec);
}

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord) {
    std::array<uint32_t, 2> tensor_shape_2d{0, 0};
    for (size_t i = 0; i < tensor_shape.size(); i++) {
        if (i == tensor_shape.size() - 1) {
            // width dimension, goes unmodified
            tensor_shape_2d[1] = tensor_shape[i];
        } else {
            // height dimension, squeeze into 2D shape
            if (tensor_shape_2d[0] == 0) {
                // first time we've seen this dimension
                tensor_shape_2d[0] = tensor_shape[i];
            } else {
                tensor_shape_2d[0] *= tensor_shape[i];
            }
        }
    }

    std::array<uint32_t, 2> tensor_coord_2d{0, tensor_coord.back()};
    uint32_t height_2d = 0;
    for (size_t i = 0; i < tensor_coord.size() - 1; i++) {
        std::vector<uint32_t> page_shapes(tensor_shape.begin() + i + 1, tensor_shape.end() - 1);
        auto component_sum =
            tensor_coord[i] * std::accumulate(page_shapes.begin(), page_shapes.end(), 1, std::multiplies<uint32_t>());
        height_2d += component_sum;
    }
    tensor_coord_2d[0] = height_2d;

    uint32_t shard_height = shard_shape[0];
    uint32_t w_in_shard = tensor_coord_2d[1];
    uint32_t h_in_shard = height_2d % shard_height;
    uint32_t which_shard = height_2d / shard_height;

    std::array<uint32_t, 2> shard_coord{h_in_shard, w_in_shard};
    return std::make_pair(which_shard, shard_coord);
}

uint32_t l1_space_post_allocation(const TensorSpec& tensor_spec, IDevice* device) {
    auto buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    auto page_size_bytes = tensor_spec.compute_page_size_bytes();
    auto shard_spec_buffer = tensor_spec.compute_shard_spec_buffer();
    auto memory_config = tensor_spec.tensor_layout().get_memory_config();

    // get current device address
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    // return current device address if DRAM
    if (memory_config.buffer_type == BufferType::DRAM) {
        return max_l1_space;
    }

    uint32_t num_banks = device->allocator()->get_num_banks(BufferType::L1);
    if (memory_config.shard_spec.has_value()) {
        num_banks = memory_config.shard_spec.value().num_cores();
    }

    uint32_t alignment_bytes = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::L1);
    uint32_t num_pages = page_size_bytes == 0 ? 0 : buffer_size_bytes / page_size_bytes;
    uint32_t num_equally_distributed_pages = num_pages == 0 ? 0 : 1 + ((num_pages - 1) / num_banks);
    uint32_t buffer_addr_add = num_equally_distributed_pages * tt::round_up(page_size_bytes, alignment_bytes);

    uint32_t final_space = max_l1_space - buffer_addr_add;
    return final_space;
}

//----------------------------------------------------------------------
// to_string_custom overloads for converting values to strings
// (These functions use std::string and are not constexpr.)

// For integral types (except char)
template <typename T>
std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, char>, std::string> to_string_custom(const T& value) {
    return std::to_string(value);
}

// For floating point types
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, std::string> to_string_custom(const T& value) {
    return std::to_string(value);
}

// For char: show it in single quotes.
std::string to_string_custom(char value) { return std::string("'") + value + "'"; }

// New overload for const char*
std::string to_string_custom(const char* s) { return std::string(s); }

// For std::string
std::string to_string_custom(const std::string& s) { return s; }

// For data_format
std::string to_string_custom(tt::DataFormat data_format) {
    switch (data_format) {
        case tt::DataFormat::Float32: return "float";
        case tt::DataFormat::UInt8: return "uint8_t";
        case tt::DataFormat::UInt16: return "uint16_t";
        case tt::DataFormat::UInt32: return "uint32_t";
        case tt::DataFormat::Int8: return "int8_t";
        case tt::DataFormat::Int32: return "int32_t";
        default: {
            TT_FATAL(false, "Unsupported data format for string conversion");
        }
    }
}

// For data type
std::string to_string_custom(DataType data_type) {
    switch (data_type) {
        case DataType::FLOAT32: return "float";
        case DataType::UINT32: return "uint32_t";
        case DataType::UINT8: return "uint8_t";
        case DataType::UINT16: return "uint16_t";
        case DataType::INT32: return "int32_t";
        default: {
            TT_FATAL(false, "Unsupported data type for string conversion");
        }
    }
}

// For primitive types
std::string to_string_custom(PrimType prim_type) {
    switch (prim_type) {
        case PrimType::INT8: return "int8_t";
        case PrimType::INT16: return "int16_t";
        case PrimType::INT32: return "int32_t";
        case PrimType::INT64: return "int64_t";
        case PrimType::UINT8: return "uint8_t";
        case PrimType::UINT16: return "uint16_t";
        case PrimType::UINT32: return "uint32_t";
        case PrimType::UINT64: return "uint64_t";
        case PrimType::FLOAT16: return "float16_t";
        case PrimType::FLOAT32: return "float";
        case PrimType::BOOL: return "bool";
        case PrimType::CHAR: return "char";
        case PrimType::VOID: return "void";
        default: {
            TT_FATAL(false, "Unsupported primitive type for string conversion");
        }
    }
}

// For C-style arrays: produces a single pair of braces.
template <typename T, std::size_t N>
std::string to_string_custom(const T (&arr)[N]) {
    std::string result = "{";
    for (std::size_t i = 0; i < N; ++i) {
        result += to_string_custom(arr[i]);
        if (i != N - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

// For std::array (treated like a C-style array)
template <typename T, std::size_t N>
std::string to_string_custom(const std::array<T, N>& arr) {
    std::string result = "{";
    for (std::size_t i = 0; i < arr.size(); ++i) {
        result += to_string_custom(arr[i]);
        if (i != arr.size() - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

//----------------------------------------------------------------------
// Specialization for std::vector: output double braces.
// This makes a temporary array literal that binds to a parameter of type const T(&)[N].
template <typename T, typename Allocator>
std::string to_string_custom(const std::vector<T, Allocator>& vec) {
    std::string result = "{{";
    bool first = true;
    for (const auto& elem : vec) {
        if (!first) {
            result += ", ";
        }
        result += to_string_custom(elem);
        first = false;
    }
    result += "}}";
    return result;
}

//----------------------------------------------------------------------
// For generic containers (other than std::vector) that are not strings or C-style arrays.
template <typename Container, typename Enable>
std::string to_string_custom(const Container& container) {
    std::string result = "{";
    bool first = true;
    for (const auto& elem : container) {
        if (!first) {
            result += ", ";
        }
        result += to_string_custom(elem);
        first = false;
    }
    result += "}";
    return result;
}

//----------------------------------------------------------------------
// Tuple-to-vector conversion for template arguments.
template <typename T>
std::string to_template_arg_string(const T& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
        return arg;
    } else {
        return to_string_custom(arg);
    }
}

template <typename Tuple, std::size_t... I>
std::vector<std::string> tuple_to_vector_of_strings_impl(const Tuple& tup, std::index_sequence<I...>) {
    return {to_template_arg_string(std::get<I>(tup))...};
}

template <typename Tuple>
std::vector<std::string> tuple_to_vector_of_strings(const Tuple& tup) {
    return tuple_to_vector_of_strings_impl(tup, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

//----------------------------------------------------------------------
// A simple validator for the instantiation string.
// Checks that parentheses, angle brackets, and curly braces are balanced.
bool validate_instantiation_string(const std::string& s) {
    int paren = 0, angle = 0, curly = 0;
    for (char c : s) {
        if (c == '(') {
            paren++;
        }
        if (c == ')') {
            paren--;
        }
        if (c == '<') {
            angle++;
        }
        if (c == '>') {
            angle--;
        }
        if (c == '{') {
            curly++;
        }
        if (c == '}') {
            curly--;
        }
        if (paren < 0 || angle < 0 || curly < 0) {
            return false;
        }
    }
    return (paren == 0 && angle == 0 && curly == 0);
}

//----------------------------------------------------------------------
// The instantiate function.
// It takes the struct name as a string, a tuple of template arguments,
// and a variadic list of constructor arguments (converted to strings).
template <typename Tuple, typename... Args>
std::string instantiate(const std::string& structName, const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    std::vector<std::string> templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
    for (const auto& arg : templateArgs) {
        TT_FATAL(!arg.empty(), "Template argument string should not be empty");
    }

    std::string result = structName;
    if (!templateArgs.empty()) {
        result += "<";
        for (size_t i = 0; i < templateArgs.size(); ++i) {
            result += templateArgs[i];
            if (i != templateArgs.size() - 1) {
                result += ", ";
            }
        }
        result += ">";
    }
    result += "(";
    bool first = true;
    ((result += (first ? "" : ", ") + to_string_custom(constructorArgs), first = false), ...);
    result += ")";

    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");

    return result;
}

// We then allow our non-template overload of instantiate to use TemplateNameTTHelper.

//----------------------------------------------------------------------
// instantiate overload for template types.
// (For types that are templates)
template <template <typename, auto...> class Struct, typename Tuple, typename... Args>
std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    std::vector<std::string> templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
    for (const auto& arg : templateArgs) {
        TT_FATAL(!arg.empty(), "Template argument string should not be empty");
    }

    std::string result = TemplateNameTT<Struct>::name;
    if (!templateArgs.empty()) {
        result += "<";
        for (size_t i = 0; i < templateArgs.size(); ++i) {
            result += templateArgs[i];
            if (i != templateArgs.size() - 1) {
                result += ", ";
            }
        }
        result += ">";
    }
    result += "(";
    bool first = true;
    ((result += (first ? "" : ", ") + to_string_custom(constructorArgs), first = false), ...);
    result += ")";

    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");

    return result;
}

//----------------------------------------------------------------------
// instantiate overload for non-template types (like CompositeStruct).
template <typename Struct, typename Tuple, typename... Args>
std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    std::vector<std::string> templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
    for (const auto& arg : templateArgs) {
        TT_FATAL(!arg.empty(), "Template argument string should not be empty");
    }

    std::string result = TemplateNameTTHelper<Struct>::name;
    if (!templateArgs.empty()) {
        result += "<";
        for (size_t i = 0; i < templateArgs.size(); ++i) {
            result += templateArgs[i];
            if (i != templateArgs.size() - 1) {
                result += ", ";
            }
        }
        result += ">";
    }
    result += "(";
    bool first = true;
    ((result += (first ? "" : ", ") + to_string_custom(constructorArgs), first = false), ...);
    result += ")";

    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");

    return result;
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
