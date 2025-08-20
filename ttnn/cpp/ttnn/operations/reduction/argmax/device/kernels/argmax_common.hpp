// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "utils/bfloat16.h"
#include "utils/float32.h"
#include "utils/int32.h"

#include <cstdint>

inline constexpr uint32_t MIN_UINT32 = 0x00000000;  // Representation of minimum uint32 value
inline constexpr uint32_t MAX_UINT32 = 0xFFFFFFFF;  // Representation of maximum uint32 value
inline constexpr uint16_t MIN_UINT16 = 0x0000;      // Representation of minimum uint16 value
inline constexpr uint16_t MAX_UINT16 = 0xFFFF;      // Representation of maximum uint16 value

/**
 * @brief Returns a typed volatile L1 memory pointer based on the specified data format.
 *
 * This template function converts a raw memory address to a typed volatile pointer
 * that corresponds to the underlying data type of the specified DataFormat enum.
 * The returned pointer is marked as volatile and uses the tt_l1_ptr qualifier,
 * indicating it points to L1 cache memory that may be modified by hardware.
 *
 * @tparam data_format The DataFormat enum value that determines the return type
 * @param addr The raw memory address to be cast to the appropriate pointer type
 *
 * @return A volatile tt_l1_ptr pointer of the appropriate type:
 *         - uint16_t* for Float16_b and UInt16 formats
 *         - uint32_t* for Float32 and UInt32 formats
 *         - int32_t* for Int32 format
 *
 * @note This function uses compile-time template specialization to ensure
 *       type safety and optimal performance. Unsupported data formats will
 *       trigger a compile-time assertion error.
 */
template <DataFormat data_format>
auto get_tt_l1_ptr_based_on_data_format(const uint32_t addr) {
    if constexpr (data_format == DataFormat::Float16_b) {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    } else if constexpr (data_format == DataFormat::UInt16) {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    } else if constexpr (data_format == DataFormat::Float32) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    } else if constexpr (data_format == DataFormat::Int32) {
        return reinterpret_cast<volatile tt_l1_ptr int32_t*>(addr);
    } else if constexpr (data_format == DataFormat::UInt32) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    } else {
        // We need a value-dependent expression (gcc-12) that is not
        // tautologically false (gcc-15)
        static_assert(
            data_format == DataFormat::Float16_b, "Unsupported data format in get_tt_l1_ptr_based_on_data_format");
    }
}

/**
 * @brief Returns the default minimum value for argmax operations based on the specified data format.
 *
 * This template function provides the appropriate negative infinity or minimum value
 * for different data formats used in argmax operations. The returned value serves as
 * the initial comparison value when finding the maximum element and its index.
 *
 * @tparam data_format The data format enum specifying the type of data being processed
 *
 * @return The default minimum value appropriate for the specified data format:
 *         - Float16_b: Returns NEG_INF_BFLOAT16 as uint16_t
 *         - UInt16: Returns NEG_INF_BFLOAT16 as uint16_t
 *         - Float32: Returns NEG_INF_FLOAT32 as uint32_t
 *         - Int32: Returns NEG_INF_INT32 as int32_t
 *         - UInt32: Returns MIN_UINT32 as uint32_t
 *
 * @note This function uses compile-time evaluation with constexpr if statements
 * @throws Compile-time error for unsupported data formats via static_assert
 */
template <DataFormat data_format>
auto get_default_value() {
    // Check for supported datatypes
    if constexpr (data_format == DataFormat::Float16_b) {
        return uint16_t{NEG_INF_BFLOAT16};
    } else if constexpr (data_format == DataFormat::UInt16) {
        return uint16_t{MIN_UINT16};
    } else if constexpr (data_format == DataFormat::Float32) {
        return uint32_t{NEG_INF_FLOAT32};
    } else if constexpr (data_format == DataFormat::Int32) {
        return int32_t{NEG_INF_INT32};
    } else if constexpr (data_format == DataFormat::UInt32) {
        return uint32_t{MIN_UINT32};
    } else {
        // We need a value-dependent expression (gcc-12) that is not
        // tautologically false (gcc-15)
        static_assert(data_format == DataFormat::Float16_b, "Unsupported data format");
    }
}

/**
 * @brief Helper function to calculate the index for argmax operations.
 *
 * @param reduce_all Flag indicating whether to reduce across all dimensions
 * @param k Index within the outer dimension
 * @param j Index within the middle dimension
 * @param i Index within the reduction dimension (inner-most index)
 * @param inner_dim_units Number of units in the inner dimension
 * @param red_dim_units Number of units in the reduction dimension
 * @return The calculated index based on the reduce_all flag
 */
inline uint32_t calculate_argmax_index(
    bool reduce_all,
    const uint32_t k,
    const uint32_t j,
    const uint32_t i,
    const uint32_t inner_dim_units,
    const uint32_t red_dim_units) {
    return reduce_all ? (k * inner_dim_units * red_dim_units + j * red_dim_units + i) : i;
}

/**
 * @brief Template helper function to perform value comparison and update max value/index.
 *
 * @tparam ValueType The underlying data type (uint16_t, uint32_t, int32_t)
 * @tparam CompareFunc The comparison function type
 * @param in_vals Pointer to the input values array
 * @param max_val Reference to the current maximum value
 * @param max_idx Reference to the current maximum index
 * @param val The value to compare
 * @param index The index of the value
 * @param compare_func The comparison function to use
 */
template <typename ValueType, typename CompareFunc>
inline void update_max_if_greater(
    ValueType& max_val, uint32_t& max_idx, const ValueType val, const uint32_t index, CompareFunc compare_func) {
    if (compare_func(val, max_val)) {
        max_idx = index;
        max_val = val;
    }
}

/**
 * @brief Compares values from a circular buffer and updates the maximum value and its index for argmax operation.
 *
 * This template function reads a value from the specified circular buffer address and compares it with the current
 * maximum value. If the new value is greater, it updates both the maximum value and its corresponding index.
 * The comparison logic is specialized for different data formats using compile-time branching.
 *
 * @tparam data_format The data format type (Float16_b, Float32, UInt16, Int32, UInt32) that determines
 *                     the value type and comparison method to use
 *
 * @param src_cb_addr Address of the source circular buffer containing the input values
 * @param max_val Reference to the current maximum value that may be updated if a larger value is found
 * @param max_idx Reference to the index of the current maximum value that may be updated
 * @param i Index within the reduction dimension (inner-most index)
 * @param j Index within the middle dimension
 * @param k Index within the outer dimension
 * @param red_dim_units Number of units in the reduction dimension
 * @param reduce_all Flag indicating whether to reduce across all dimensions (affects index calculation)
 * @param inner_dim_units Number of units in the inner dimension (used for multi-dimensional index calculation)
 *
 * @note The function uses specialized comparison functions (bfloat16_greater, float32_greater, etc.) for
 *       floating-point and signed integer types to handle special cases like NaN values properly.
 * @note When reduce_all is true, the index is calculated as a flattened multi-dimensional index,
 *       otherwise only the reduction dimension index (i) is used.
 */
template <DataFormat data_format>
void compare_values(
    const uint32_t src_cb_addr,
    decltype(get_default_value<data_format>())& max_val,
    uint32_t& max_idx,
    const uint32_t i,
    const uint32_t j,
    const uint32_t k,
    const uint32_t red_dim_units,
    bool reduce_all,
    uint32_t inner_dim_units) {
    auto in_vals = get_tt_l1_ptr_based_on_data_format<data_format>(src_cb_addr);
    auto val = in_vals[i];
    const uint32_t index = calculate_argmax_index(reduce_all, k, j, i, inner_dim_units, red_dim_units);

    if constexpr (data_format == DataFormat::Float16_b) {
        update_max_if_greater(max_val, max_idx, val, index, bfloat16_greater);
    } else if constexpr (data_format == DataFormat::Float32) {
        update_max_if_greater(max_val, max_idx, val, index, float32_greater);
    } else if constexpr (data_format == DataFormat::UInt16) {
        update_max_if_greater(max_val, max_idx, val, index, [](auto a, auto b) { return a > b; });
    } else if constexpr (data_format == DataFormat::Int32) {
        update_max_if_greater(max_val, max_idx, val, index, int32_greater);
    } else if constexpr (data_format == DataFormat::UInt32) {
        update_max_if_greater(max_val, max_idx, val, index, [](auto a, auto b) { return a > b; });
    } else {
        // We need a value-dependent expression (gcc-12) that is not
        // tautologically false (gcc-15)
        static_assert(data_format == DataFormat::Float16_b, "Unsupported data format in compare_values");
    }
}

/**
 * @brief Helper function template for processing a single core's data in argmax reduction.
 *
 * This function template provides a unified interface for comparing values from different
 * data formats by using appropriate comparison functions and pointer types.
 *
 * @tparam data_format Data format of the values being compared
 * @tparam ValueType The underlying C++ type for the data format (e.g., uint16_t for Float16_b)
 * @tparam CompareFunc Function type for value comparison
 *
 * @param inner_idx Index within the core's output buffer
 * @param i_red_vals Pointer to the core's reduction values
 * @param i_red_idxs Pointer to the core's reduction indices
 * @param max_val Reference to the current maximum value
 * @param max_idx Reference to the current maximum index
 * @param compare_func Function to compare two values of the data format
 */
template <DataFormat data_format, typename ValueType, typename CompareFunc>
inline void process_core_data(
    const uint32_t inner_idx,
    volatile tt_l1_ptr ValueType* i_red_vals,
    volatile tt_l1_ptr uint32_t* i_red_idxs,
    decltype(get_default_value<data_format>())& max_val,
    uint32_t& max_idx,
    CompareFunc compare_func) {
    ValueType val = i_red_vals[inner_idx];

    if (compare_func(val, max_val)) {
        max_idx = i_red_idxs[inner_idx];
        max_val = val;
    } else if ((val == max_val) && (i_red_idxs[inner_idx] < max_idx)) {
        max_idx = i_red_idxs[inner_idx];
    }
}

/**
 * @brief Helper function template for processing value comparison in find_argmax_for_core.
 *
 * This function template provides a unified interface for comparing values and updating
 * max_val and max_idx based on different data formats and reduction modes.
 *
 * @tparam data_format Data format of the values being compared
 * @tparam ValueType The underlying C++ type for the data format
 * @tparam CompareFunc Function type for value comparison
 * @tparam reduce_all Boolean flag indicating reduction mode
 *
 * @param val The current value being compared
 * @param max_val Reference to the current maximum value
 * @param max_idx Reference to the current maximum index
 * @param i Current element index in reduction dimension
 * @param outer_idx Current outer dimension index
 * @param j Current inner dimension index
 * @param inner_dim_units Number of inner dimension units
 * @param red_dim_units Total reduction dimension units
 * @param compare_func Function to compare two values of the data format
 */
template <DataFormat data_format, typename ValueType, bool reduce_all, typename CompareFunc>
inline void process_value_comparison(
    ValueType val,
    decltype(get_default_value<data_format>())& max_val,
    uint32_t& max_idx,
    const uint32_t i,
    const uint32_t outer_idx,
    const uint32_t j,
    const uint32_t inner_dim_units,
    const uint32_t red_dim_units,
    CompareFunc compare_func) {
    if (compare_func(val, max_val)) {
        auto full_idx = outer_idx * inner_dim_units * red_dim_units + j * red_dim_units + i;
        max_idx = reduce_all ? full_idx : i;
        max_val = val;
    } else if (val == max_val) {
        auto full_idx = outer_idx * inner_dim_units * red_dim_units + j * red_dim_units + i;
        max_idx = reduce_all ? std::min(max_idx, full_idx) : std::min(max_idx, i);
    }
}
