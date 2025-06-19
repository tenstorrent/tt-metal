// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The intent is to merge this file into dataflow_api.h and then refactor it into multiple files.
// It is currently here while its reliability is proven

#pragma once
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include "dataflow_api.h"

#endif

#include "ttnn/operations/ccl/common/types/sharding_common.hpp"

using mapping_table_t = uint32_t;

template <
    uint32_t SHARD_TYPE,
    uint32_t NUMBER_OF_CORES,
    uint32_t PAGE_SIZE_JUMP,
    uint32_t PAGES_PER_TENSOR_ROW,
    uint32_t CONTIGUITY,
    uint32_t PAGES_PER_SHARD_WIDTH,
    uint32_t ROWS_PER_SHARD_HEIGHT>
struct ShardedInfo {
public:
    // The isX types are correctly templated shard_grid_info class objects containing the information of the respective
    // grid
    constexpr static tt::tt_metal::TensorMemoryLayout shard_type =
        static_cast<tt::tt_metal::TensorMemoryLayout>(SHARD_TYPE);
    constexpr static uint32_t number_of_cores = NUMBER_OF_CORES;
    constexpr static uint32_t page_size_jump = PAGE_SIZE_JUMP;
    constexpr static uint32_t pages_per_tensor_row = PAGES_PER_TENSOR_ROW;
    constexpr static shard_addr_gen_consts::ContiguityType contiguity =
        static_cast<shard_addr_gen_consts::ContiguityType>(CONTIGUITY);
    constexpr static uint32_t pages_per_shard_width = PAGES_PER_SHARD_WIDTH;
    constexpr static uint32_t rows_per_shard_height = ROWS_PER_SHARD_HEIGHT;
};
namespace experimental {
namespace shard_addr_gen_utils {

struct ShardCoordInfo {
    uint32_t core_num;
    uint32_t page_num;
    uint32_t num_contiguous_pages;
};

template <uint32_t columns_per_shard, uint32_t total_pages_last_dim, shard_addr_gen_consts::ContiguityType contiguity>
struct ShardCoordInfo get_width_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    struct ShardCoordInfo coord_info;
    uint32_t page_row = page_num / total_pages_last_dim;
    uint32_t page_col = page_num - page_row * total_pages_last_dim;
    uint32_t w_core_id = page_col / columns_per_shard;
    uint32_t w_offset = page_col - w_core_id * columns_per_shard;
    coord_info.core_num = w_core_id;
    coord_info.page_num = page_row * columns_per_shard + w_offset;
    if constexpr (
        contiguity != shard_addr_gen_consts::ContiguityType::L1_PADDING_BETWEEN_PAGES &&
        contiguity != shard_addr_gen_consts::ContiguityType::DRAM_PADDING_BETWEEN_PAGES) {
        uint32_t space_left_in_shard = columns_per_shard - w_offset;
        uint32_t space_left_in_tensor = total_pages_last_dim - page_col;
        coord_info.num_contiguous_pages =
            space_left_in_shard < space_left_in_tensor ? space_left_in_shard : space_left_in_tensor;
    } else {
        coord_info.num_contiguous_pages = 1;
    }
    return coord_info;
}

template <uint32_t rows_per_shard, uint32_t total_pages_last_dim, shard_addr_gen_consts::ContiguityType contiguity>
struct ShardCoordInfo get_height_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    struct ShardCoordInfo coord_info;
    constexpr uint32_t num_pages_per_core = total_pages_last_dim * rows_per_shard;
    coord_info.core_num = page_num / num_pages_per_core;
    coord_info.page_num = page_num - coord_info.core_num * num_pages_per_core;
    if constexpr (
        contiguity == shard_addr_gen_consts::ContiguityType::L1_PADDING_BETWEEN_PAGES ||
        contiguity == shard_addr_gen_consts::ContiguityType::DRAM_PADDING_BETWEEN_PAGES) {
        coord_info.num_contiguous_pages = 1;
    } else if constexpr (
        contiguity == shard_addr_gen_consts::ContiguityType::L1_PADDING_IN_RIGHTMOST_SHARD ||
        contiguity == shard_addr_gen_consts::ContiguityType::DRAM_PADDING_IN_RIGHTMOST_SHARD) {
        coord_info.num_contiguous_pages = total_pages_last_dim - page_num % total_pages_last_dim;
    } else {
        coord_info.num_contiguous_pages = num_pages_per_core - coord_info.page_num;
    }
    return coord_info;
}

template <
    uint32_t columns_per_shard,
    uint32_t rows_per_shard,
    uint32_t total_pages_last_dim,
    shard_addr_gen_consts::ContiguityType contiguity>
experimental::shard_addr_gen_utils::ShardCoordInfo get_block_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    // Calculate how many cores are in the sharding grid
    constexpr uint32_t cores_per_block_row = (total_pages_last_dim - 1) / columns_per_shard + 1;
    experimental::shard_addr_gen_utils::ShardCoordInfo coord_info;
    // Get row and column ID of this page
    uint32_t page_row = page_num / total_pages_last_dim;
    uint32_t page_col = page_num - page_row * total_pages_last_dim;
    // Find the w direction core and the offset within it
    uint32_t w_core_id = page_col / columns_per_shard;
    uint32_t w_offset = page_col - w_core_id * columns_per_shard;
    // Find the h direction core and the offset within it
    uint32_t h_core_id = page_row / rows_per_shard;
    uint32_t h_offset = page_row - h_core_id * rows_per_shard;
    // Find the coord_info
    coord_info.core_num = w_core_id + h_core_id * cores_per_block_row;
    coord_info.page_num = w_offset + h_offset * columns_per_shard;
    if constexpr (
        contiguity != shard_addr_gen_consts::ContiguityType::L1_PADDING_BETWEEN_PAGES &&
        contiguity != shard_addr_gen_consts::ContiguityType::DRAM_PADDING_BETWEEN_PAGES) {
        uint32_t space_left_in_shard = columns_per_shard - w_offset;
        uint32_t space_left_in_tensor = total_pages_last_dim - page_col;
        coord_info.num_contiguous_pages =
            space_left_in_shard < space_left_in_tensor ? space_left_in_shard : space_left_in_tensor;
    } else {
        coord_info.num_contiguous_pages = 1;
    }
    return coord_info;
}

/*
 * Returns a 16 bit compressed representation of the core number for each core in the shard grid
 * ShardedAddrGen::get_sharded_addr can extract the noc address from this compressed core representation
 * Representation is placed in a uint32_t array in a big endian ordering
 */

template <typename ShardingInfoType>
std::pair<const mapping_table_t* const, uint32_t> get_shard_map(uint32_t L1_address) {
    // Gets the shard_array from the runtime arguments
    // returns a pair where .first holds the shard array map
    // and .second holds the size of the map
    constexpr ShardingInfoType CONSTANT_ARGS{};
    const mapping_table_t* const map = reinterpret_cast<const mapping_table_t* const>(L1_address);
    constexpr uint32_t incrementation = (CONSTANT_ARGS.number_of_cores - 1) / 2 + 1;
    return std::pair<const mapping_table_t* const, uint32_t>(map, incrementation);
}

}  // namespace shard_addr_gen_utils

/*
* ShardedAddrGen requires the type definition of a ShardedInfo class object whose templates hold the CT information
    ex.
    using tensor_1_shard_info = ShardedInfo <
    SHARD_TYPE,
    NUMBER_OF_CORES,
    PAGE_SIZE_JUMP,
    PAGES_PER_TENSOR_ROW,
    CONTIGUITY,
    PAGES_PER_SHARD_WIDTH,
    ROWS_PER_SHARD_HEIGHT>;

    The above parameters are usually obtained using get_compile_time_arg_val.
    In the program factory you can create an vector containing the above parameters in order using the function
    shard_builder:generate_compile_time_args(const tt::tt_metal::Tensor& t)
    defined in ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp

    It also needs a shard array map which can be extracted from the RT args using shard_addr_gen_utils::get_shard_map
function which requires the ShardedInfo class object ex. auto mapping =
experimental::shard_addr_gen_utils::get_shard_map<ShardingInfoType>(get_arg_addr(rt_index)); const mapping_table_t*
const shard_array_map = mapping.first;
//Contains the shard array map
rt_index += mapping.second;//contains the size of the map hence how much to increment the rt values

In the program factory you can create an vector containing the runtime arguments extracted by this function using the
function shard_builder:generate_run_time_args(const tt::tt_metal::Tensor& t)
    defined in ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp



    Lastly you need the bank_base_address from the Tensor object just like interleaved addr gen

    You can then create a sharded addrgen as follows:
    s = ShardedAddrGen <tensor_1_shard_info> {.bank_base_address = bank_base_address, .shard_array=shard_array_map};
    This object can then be used by the get_noc_addr api.
*/
template <typename ShardingInfoType>
struct ShardedAddrGen {
    // Use this address generator for sharded tensors

    constexpr static ShardingInfoType CONSTANT_ARGS{};
    // Sharded Info Class is a ShardedInfo class object that is appropriately templated
    // including all the compile time parameters
    uint32_t bank_base_address;
    const mapping_table_t* const shard_array;

    FORCE_INLINE
    std::uint64_t get_sharded_addr(
        const uint32_t l1_addr, const uint32_t sharding_coordinates, const uint32_t noc = noc_index) const {
        constexpr uint32_t SHIFT_BITS = 8;
        constexpr uint32_t MASK = 0xFF;
        constexpr shard_addr_gen_consts::ContiguityType contiguity = CONSTANT_ARGS.contiguity;
        constexpr bool is_dram = contiguity == shard_addr_gen_consts::ContiguityType::DRAM_PADDING_BETWEEN_PAGES ||
                                 contiguity == shard_addr_gen_consts::ContiguityType::DRAM_PADDING_IN_RIGHTMOST_SHARD ||
                                 contiguity == shard_addr_gen_consts::ContiguityType::DRAM_NO_SHARD_PADDING;
        if constexpr (is_dram) {
            uint32_t bank_id = (sharding_coordinates >> SHIFT_BITS) & MASK;
            uint32_t src_addr = l1_addr + bank_to_dram_offset[bank_id];
            uint32_t src_noc_xy = dram_bank_to_noc_xy[noc][bank_id];
            return ((uint64_t)(src_noc_xy) << NOC_ADDR_COORD_SHIFT) | src_addr;
        } else {
            // Extracts the X and Y value and using the l1 address gets the noc address
            return NOC_XY_ADDR(
                DYNAMIC_NOC_X(noc, ((sharding_coordinates >> SHIFT_BITS) & MASK)),
                DYNAMIC_NOC_Y(noc, (sharding_coordinates & MASK)),
                l1_addr);
        }
    }

    std::uint32_t get_sharded_l1_addr(const uint32_t core_page, const uint32_t offset = 0) const {
        // Get the L1 address
        return this->bank_base_address + (core_page * CONSTANT_ARGS.page_size_jump) + offset;
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return this->get_contiguous_noc_addr(id, offset, noc).first;
    }
    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        // Returns the noc address AND the number of contiguous pages after.

        // Resolve linear core id/bank address, the page offset in the core,
        // and the number of contiguous pages within that core
        experimental::shard_addr_gen_utils::ShardCoordInfo sharding_coordinates{};
        if constexpr (CONSTANT_ARGS.shard_type == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
            sharding_coordinates = experimental::shard_addr_gen_utils::get_width_sharded_coordinates<
                CONSTANT_ARGS.pages_per_shard_width,
                CONSTANT_ARGS.pages_per_tensor_row,
                CONSTANT_ARGS.contiguity>(id);
        } else if constexpr (CONSTANT_ARGS.shard_type == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
            sharding_coordinates = experimental::shard_addr_gen_utils::get_height_sharded_coordinates<
                CONSTANT_ARGS.rows_per_shard_height,
                CONSTANT_ARGS.pages_per_tensor_row,
                CONSTANT_ARGS.contiguity>(id);
        } else {
            sharding_coordinates = experimental::shard_addr_gen_utils::get_block_sharded_coordinates<
                CONSTANT_ARGS.pages_per_shard_width,
                CONSTANT_ARGS.rows_per_shard_height,
                CONSTANT_ARGS.pages_per_tensor_row,
                CONSTANT_ARGS.contiguity>(id);
        }
        // Get the value from the resolved core location containing the core x and y each 8 bits
        // Note we are stripping this from a 32 bit array hence the floor division by 2 and in
        // odd numbered cores a right shift by 16 and a masking
        uint32_t sharding_coordinate_value =
            (shard_array[(sharding_coordinates.core_num) >> 1] >> ((sharding_coordinates.core_num & 1) == 1 ? 0 : 16)) &
            0xFFFF;
        // Find the L1 address within the resolved core
        auto resolved_l1_addr = get_sharded_l1_addr(sharding_coordinates.page_num, offset);
        // Find the noc address using the x,y core information
        auto resolved_sharded_addr = get_sharded_addr(resolved_l1_addr, sharding_coordinate_value, noc);
        // Return the core info and the number of contiguous cores
        std::pair<uint64_t, uint32_t> return_val(resolved_sharded_addr, sharding_coordinates.num_contiguous_pages);
        return return_val;
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(this->get_noc_addr(id, offset, noc), dest_addr, CONSTANT_ARGS.page_size_jump, noc);
    }
};
}  // namespace experimental

/**
 * gets the noc address from the addrgen object for a given page.
 * This tells the user  the address of the given page
 * Can accept ShardedAddrGen, InterleavedAddrGen, InterleavedPow2AddrGen,
 *       InterleavedAddrGenFast, or InterleavedPow2AddrGenFast objects
 * Return value: A uint64_t object with the noc address of the object
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|----------------------------------------------------------------|----------|
 * | id                | The page or tile number to be accessed                  | uint32_t | 0..1MB      | True     | |
 * | AddrGenObj        | The address generator object to use                     | see above| N/A         | True     | |
 * | offset            | The offset within the page or tile to access            | uint32_t | 0..page size| False    | |
 * | noc               | Which noc to use, defaults to noc_index                 | uint8_t  | 0 or 1      | False    | |
 */

template <typename SIC>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const experimental::ShardedAddrGen<SIC>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    return s.get_noc_addr(id, offset, noc);
}

// Interleaved versions of get_noc_addr are implemented in dataflow_api.h

/**
 * gets the contiguous noc address from the addrgen object for a given page.
 * This tells the user both the address of the given page and how many subsequent
 * pages are contiguously located sequentially in the same memory.
 * Can accept ShardedAddrGen, InterleavedAddrGen, InterleavedPow2AddrGen,
 *       InterleavedAddrGenFast, or InterleavedPow2AddrGenFast objects
 * Return value: An std::pair object where the .first value is the noc address of the object
 *       and .second is the number of sequentially contiguous pages starting at id.
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|----------------------------------------------------------------|----------|
 * | id                | The page or tile number to be accessed                  | uint32_t | 0..1MB      | True     | |
 * | AddrGenObj        | The address generator object to use                     | see above| N/A         | True     | |
 * | offset            | The offset within the page or tile to access            | uint32_t | 0..page size| False    | |
 * | noc               | Which noc to use, defaults to noc_index                 | uint8_t  | 0 or 1      | False    | |
 */

template <typename SIC>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const experimental::ShardedAddrGen<SIC>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    return s.get_contiguous_noc_addr(id, offset, noc);
}

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    std::pair<uint64_t, uint32_t> ret_val(s.get_noc_addr(id, offset, noc), 1);
    return ret_val;
}

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    std::pair<uint64_t, uint32_t> ret_val(s.get_noc_addr(id, offset, noc), 1);
    return ret_val;
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    std::pair<uint64_t, uint32_t> ret_val(s.get_noc_addr(id, offset, noc), 1);
    return ret_val;
}

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGenFast<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    std::pair<uint64_t, uint32_t> ret_val(s.get_noc_addr(id, offset, noc), 1);
    return ret_val;
}

#endif

template <typename SIC>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const experimental::ShardedAddrGen<SIC>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_page(id, dst_local_l1_addr, offset, noc);
}
