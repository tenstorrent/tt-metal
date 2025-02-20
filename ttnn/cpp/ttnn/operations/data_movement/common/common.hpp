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

// ----------------------------------------------------------------------
// Primitive types for template argument conversion
enum class PrimType { INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT16, FLOAT32, BOOL, CHAR, VOID };

// ----------------------------------------------------------------------
// Non-templated function declarations (implementations in instantiation_utils.cpp)
std::string to_string_custom(char value);
std::string to_string_custom(const char* s);
std::string to_string_custom(const std::string& s);
std::string to_string_custom(tt::DataFormat data_format);
std::string to_string_custom(DataType data_type);

bool validate_instantiation_string(const std::string& s);

// ----------------------------------------------------------------------
// Template function declarations
// (These functions must be explicitly instantiated in the source file if you wish
// to keep their definitions out of the header.)

// For integral types (except char)
template <typename T>
std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, char>, std::string> to_string_custom(const T& value);

// For floating point types
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, std::string> to_string_custom(const T& value);

// For C-style arrays
template <typename T, std::size_t N>
std::string to_string_custom(const T (&arr)[N]);

// For std::array
template <typename T, std::size_t N>
std::string to_string_custom(const std::array<T, N>& arr);

// Specialization for std::vector
template <typename T, typename Allocator>
std::string to_string_custom(const std::vector<T, Allocator>& vec);

// A trait to check if T is iterable (has valid begin and end)
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

// For generic containers (excluding std::string and C-style arrays)
template <
    typename Container,
    typename = std::enable_if_t<
        is_iterable<Container>::value &&            // Only iterable types…
        !std::is_same_v<Container, std::string> &&  // …excluding std::string…
        !std::is_array_v<Container> &&              // …and C-style arrays…
        !std::is_arithmetic_v<Container>            // …and scalar arithmetic types.
        >>
std::string to_string_custom(const Container& container);

// Tuple-to-vector conversion for template arguments.
template <typename T>
std::string to_template_arg_string(const T& arg);

template <typename Tuple, std::size_t... I>
std::vector<std::string> tuple_to_vector_of_strings_impl(const Tuple& tup, std::index_sequence<I...>);

template <typename Tuple>
std::vector<std::string> tuple_to_vector_of_strings(const Tuple& tup);

//----------------------------------------------------------------------
// TemplateNameTT: the single helper trait for retrieving a type’s name,
// For template types (passed as a template template parameter) the user specializes TemplateNameTT;
// for non-template types the specialization is similar.
template <template <typename, auto...> class T>
struct TemplateNameTT;  // no default definition

// For non-template types, we provide a separate specialization.
template <typename T>
struct TemplateNameTTHelper;  // primary template not defined

// The instantiate function: constructs an instantiation string from a struct name,
// a tuple of template arguments, and a variadic list of constructor arguments.
template <typename Tuple, typename... Args>
std::string instantiate(const std::string& structName, const Tuple& templateArgsTuple, const Args&... constructorArgs);

// Example usage:
//----------------------------------------------------------------------
// Template struct definitions (with constexpr constructors)

// template<typename T, std::size_t N>
// struct MyStruct {
//     T myArray[N];
//     constexpr MyStruct(const T (&arr)[N]) : myArray{} {
//         for (std::size_t i = 0; i < N; ++i)
//             myArray[i] = arr[i];
//     }
// };
// instantiate the struct with a string and an array of ints
// constexpr int arr[4] = {1, 2, 3, 4};
// auto tupleArgs = std::make_tuple("int", 4);
// std::string inst = instantiate("MyStruct", tupleArgs, arr);  // MyStruct<int, 4>({1, 2, 3, 4})

//----------------------------------------------------------------------
// instantiate overload for template types with TemplateNameTTHelper::name defines as a string.
template <template <typename, auto...> class Struct, typename Tuple, typename... Args>
std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs);

// Example usage:
//----------------------------------------------------------------------
// Specializations for template types:
// template<>
// struct TemplateNameTT<MyStruct> {
//     static constexpr const char* value = "MyStruct";
// };
// constexpr int arr1[4] = {1, 2, 3, 4};
// auto tupleArgs1 = std::make_tuple("int", 4);
// std::string inst1 = instantiate<MyStruct>(tupleArgs1, arr1); // MyStruct<int, 4>({1, 2, 3, 4});

//----------------------------------------------------------------------
// instantiate overload for non-template types (like CompositeStruct).
template <typename Struct, typename Tuple, typename... Args>
std::string instantiate(const Tuple& templateArgsTuple, const Args&... constructorArgs);

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
