// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"

#include "cpp/ttnn/tensor/types.hpp"
#include "cpp/ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

ttnn::Shape squeeze_shape_to_4D(ttnn::Shape output_shape);
ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor);

ttnn::Tensor pad_to_tile_vol(
    QueueId queue_id,
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config);

uint32_t wrap_index(int index, int size);

template <typename OpOutputType, typename... OpInputTypes>
struct MassagedOperationParams {
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    PredicateFunc predicate;           // Function to determine if formatting should be applied
    PreTransformFunc pre_transform;    // Function to pre-process input arguments
    PostTransformFunc post_transform;  // Function to post-process the operation output
    OpType operation;                  // The main operation to be performed
};

template <typename OpOutputType, typename... OpInputTypes>
class MassagedOperation {
public:
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    // post transform takes the output and optionally the args; it may use
    // the args in order to know if it needs to post process the output.
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...> params) :
        predicate_(params.predicate),
        pre_transform_(params.pre_transform),
        post_transform_(params.post_transform),
        operation_(params.operation) {}

    inline bool should_format(OpInputTypes... args) const { return predicate_(args...); }

    inline OwnedArgsType pre_format(OpInputTypes... args) const { return pre_transform_(args...); }

    inline OpOutputType post_format(OpOutputType output) const { return post_transform_(output); }

    inline OpOutputType operator()(OpInputTypes... args) const {
        if (should_format(args...)) {
            auto formatted_input = pre_format(args...);
            auto op_output = std::apply(operation_, formatted_input);
            return post_format(op_output);
        }
        return operation_(args...);
    }

    MassagedOperation sequence(const MassagedOperation& other) {
        std::shared_ptr<bool> t1_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t2_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t1_then_t2_required = std::make_shared<bool>(false);

        auto merged_predicate =
            [p1 = this->predicate_, p2 = other.predicate_, t1_required, t2_required](OpInputTypes... args) -> bool {
            if (p1(args...)) {
                *t1_required = true;
            }
            if (p2(args...)) {
                *t2_required = true;
            }
            return *t1_required or *t2_required;
        };

        auto merged_pre_transform = [t1 = this->pre_transform_,
                                     t2 = other.pre_transform_,
                                     p1 = this->predicate_,
                                     p2 = other.predicate_,
                                     t1_required,
                                     t2_required,
                                     t1_then_t2_required](OpInputTypes... args) -> OwnedArgsType {
            if (*t1_required) {
                auto transformed_args = t1(args...);
                if (std::apply(p2, transformed_args)) {
                    *t1_then_t2_required = true;
                    return std::apply(t2, transformed_args);
                }
                return transformed_args;
            } else if (*t2_required) {
                return t2(args...);
            } else {
                return std::make_tuple(args...);
            }
        };

        auto merged_post_transform =
            [t1 = this->post_transform_, t2 = other.post_transform_, t1_then_t2_required, t1_required, t2_required](
                OpOutputType output) -> OpOutputType {
            if (*t1_then_t2_required) {
                // we go backwards for post-transform
                auto t2_output = t2(output);
                auto t1_output = t1(t2_output);
                return t1_output;
            } else if (*t1_required) {
                return t1(output);
            } else if (*t2_required) {
                return t2(output);
            } else {
                return output;
            }
        };

        return MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...>{
            .predicate = merged_predicate,
            .pre_transform = merged_pre_transform,
            .post_transform = merged_post_transform,
            .operation = this->operation_});
    }

    // getters for all private members
    PredicateFunc get_predicate() const { return predicate_; }
    PreTransformFunc get_pre_transform() const { return pre_transform_; }
    PostTransformFunc get_post_transform() const { return post_transform_; }
    OpType get_operation() const { return operation_; }

    // setters for all private members
    void set_predicate(PredicateFunc predicate) { predicate_ = predicate; }
    void set_pre_transform(PreTransformFunc pre_transform) { pre_transform_ = pre_transform; }
    void set_post_transform(PostTransformFunc post_transform) { post_transform_ = post_transform; }
    void set_operation(OpType operation) { operation_ = operation; }

private:
    PredicateFunc predicate_;
    PreTransformFunc pre_transform_;
    PostTransformFunc post_transform_;
    OpType operation_;
};

ttnn::Tensor pad_to_tile_vol(
    QueueId queue_id,
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config);

enum class ShardStrategy { BLOCK, HEIGHT, WIDTH };

// Helper function for creating a sharded memory configuration for a tensor
// based on its logical shape, a shard strategy and orientation, and a core
// grid. Optionally, you may pass a preferred shard shape to use. If not
// provided, the shard shape will be inferred from the tensor shape and the
// shard strategy.
ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape = std::nullopt,
    const tt::tt_metal::Layout& layout = tt::tt_metal::Layout::ROW_MAJOR);

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord);

uint32_t l1_space_post_allocation(const TensorSpec& tensor_spec, IDevice* device);

//----------------------------------------------------------------------
// Non-templated function definitions (all inline)

inline std::string stringify(char value) { return std::string("'") + value + "'"; }

inline std::string stringify(const char* s) { return std::string(s); }

inline std::string stringify(const std::string& s) { return s; }

inline std::string stringify(tt::DataFormat data_format) {
    switch (data_format) {
        case tt::DataFormat::Float32: return "float";
        case tt::DataFormat::UInt8: return "uint8_t";
        case tt::DataFormat::UInt16: return "uint16_t";
        case tt::DataFormat::UInt32: return "uint32_t";
        case tt::DataFormat::Int8: return "int8_t";
        case tt::DataFormat::Int32: return "int32_t";
        default: TT_FATAL(false, "Unsupported data format for string conversion"); return "";
    }
}

inline std::string stringify(DataType data_type) {
    switch (data_type) {
        case DataType::FLOAT32: return "float";
        case DataType::UINT32: return "uint32_t";
        case DataType::UINT8: return "uint8_t";
        case DataType::UINT16: return "uint16_t";
        case DataType::INT32: return "int32_t";
        default: TT_FATAL(false, "Unsupported data type for string conversion"); return "";
    }
}

inline bool validate_instantiation_string(const std::string& s) {
    int paren = 0, angle = 0, curly = 0;
    for (char c : s) {
        if (c == '(') {
            ++paren;
        } else if (c == ')') {
            --paren;
        } else if (c == '<') {
            ++angle;
        } else if (c == '>') {
            --angle;
        } else if (c == '{') {
            ++curly;
        } else if (c == '}') {
            --curly;
        }
        if (paren < 0 || angle < 0 || curly < 0) {
            return false;
        }
    }
    return (paren == 0 && angle == 0 && curly == 0);
}

//----------------------------------------------------------------------
// Template function definitions

// For integral types (except char)
template <typename T>
inline std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, char>, std::string> stringify(const T& value) {
    return std::to_string(value);
}

// For floating point types
template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T>, std::string> stringify(const T& value) {
    return std::to_string(value);
}

// For C-style arrays.
template <typename T, std::size_t N>
inline std::string stringify(const T (&arr)[N]) {
    std::string result = "{";
    for (std::size_t i = 0; i < N; ++i) {
        result += stringify(arr[i]);
        if (i != N - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

// For std::array.
template <typename T, std::size_t N>
inline std::string stringify(const std::array<T, N>& arr) {
    std::string result = "{";
    for (std::size_t i = 0; i < arr.size(); ++i) {
        result += stringify(arr[i]);
        if (i != arr.size() - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

// Specialization for std::vector (output with double braces).
template <typename T, typename Allocator>
inline std::string stringify(const std::vector<T, Allocator>& vec) {
    std::string result = "{{";
    bool first = true;
    for (const auto& elem : vec) {
        if (!first) {
            result += ", ";
        }
        result += stringify(elem);
        first = false;
    }
    result += "}}";
    return result;
}

// For generic containers (excluding std::string and C-style arrays).
template <
    typename Container,
    typename = std::enable_if_t<
        std::
            is_same_v<decltype(std::begin(std::declval<Container>())), decltype(std::end(std::declval<Container>()))> &&
        !std::is_same_v<Container, std::string> && !std::is_array_v<Container>>>
inline std::string stringify(const Container& container) {
    std::string result = "{";
    bool first = true;
    for (const auto& elem : container) {
        if (!first) {
            result += ", ";
        }
        result += stringify(elem);
        first = false;
    }
    result += "}";
    return result;
}

//----------------------------------------------------------------------
// Tuple-to-vector conversion for template arguments.

template <typename T>
inline std::string to_template_arg_string(const T& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
        return arg;
    } else {
        return stringify(arg);
    }
}

template <typename Tuple, std::size_t... I>
inline std::vector<std::string> tuple_to_vector_of_strings_impl(const Tuple& tup, std::index_sequence<I...>) {
    return {to_template_arg_string(std::get<I>(tup))...};
}

template <typename Tuple>
inline std::vector<std::string> tuple_to_vector_of_strings(const Tuple& tup) {
    return tuple_to_vector_of_strings_impl(tup, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

//----------------------------------------------------------------------
// Helper traits for retrieving a type’s name.
// Users must specialize these for their types.

template <template <typename, auto...> class T>
struct TemplateNameTT;  // No default definition

template <typename T>
struct TemplateNameTTHelper;  // No default definition

//----------------------------------------------------------------------
// The instantiate API.
// There are three overloads provided:

// 1. Instantiate using an explicit struct name.
template <typename Tuple, typename... Args>
inline std::string instantiate(
    const std::string& structName, const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    auto templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
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
    ((result += (first ? "" : ", ") + stringify(constructorArgs), first = false), ...);
    result += ")";
    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");
    return result;
}

// 2. Overload for template types (the user must specialize TemplateNameTT).
template <template <typename, auto...> class Struct, typename Tuple, typename... Args>
inline std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    auto templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
    for (const auto& arg : templateArgs) {
        TT_FATAL(!arg.empty(), "Template argument string should not be empty");
    }
    std::string result = TemplateNameTT<Struct>::value;
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
    ((result += (first ? "" : ", ") + stringify(constructorArgs), first = false), ...);
    result += ")";
    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");
    return result;
}

// 3. Overload for non-template types (users must specialize TemplateNameTTHelper).
template <typename Struct, typename Tuple, typename... Args>
    requires requires { Struct{std::declval<Args>()...}; }
inline std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs) {
    auto templateArgs = tuple_to_vector_of_strings(templateArgsTuple);
    for (const auto& arg : templateArgs) {
        TT_FATAL(!arg.empty(), "Template argument string should not be empty");
    }
    std::string result = TemplateNameTTHelper<Struct>::value;
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
    ((result += (first ? "" : ", ") + stringify(constructorArgs), first = false), ...);
    result += ")";
    TT_FATAL(validate_instantiation_string(result), "Generated instantiation string is not syntactically valid");
    return result;
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
