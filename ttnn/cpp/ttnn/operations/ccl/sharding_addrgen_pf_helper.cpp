// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/tensor/tensor.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_pf_helper.hpp"

enum class Contiguity_types {
    PADDING_BETWEEN_PAGES = 0,
    PADDING_IN_RIGHTMOST_SHARD,
    NO_SHARD_PADDING,
};

namespace shard_pf_builder {

uint32_t get_sharding_core_count(const tt::tt_metal::Tensor& t) {
    uint32_t core_count = 0;
    const auto core_ranges = t.buffer()->shard_spec().grid().ranges();
    for (uint32_t cr = 0; cr < core_ranges.size(); cr++) {
        TT_FATAL(
            core_ranges.at(cr).start_coord.x <= core_ranges.at(cr).end_coord.x,
            "end coordinates left of start coordinates in shard");
        TT_FATAL(
            core_ranges.at(cr).start_coord.y <= core_ranges.at(cr).end_coord.y,
            "end coordinates above of start coordinates in shard");
        core_count += (core_ranges.at(cr).end_coord.x - core_ranges.at(cr).start_coord.x + 1) *
                      (core_ranges.at(cr).end_coord.y - core_ranges.at(cr).start_coord.y + 1);
    }
    return core_count;
}

std::vector<CoreCoord> get_vector_of_shard_cores(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t) {
    std::vector<CoreCoord> coordinates;
    struct ShardSpec shard_spec = t.shard_spec().value();
    const auto core_ranges = t.buffer()->shard_spec().grid().ranges();
    bool shard_grid_transposed =
        ((t.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
          shard_spec.orientation == ShardOrientation::ROW_MAJOR) ||
         ((t.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
           t.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) &&
          shard_spec.orientation == ShardOrientation::COL_MAJOR));
    bool last = false;
    uint32_t held_value = 0;
    uint32_t concatenated_core = 0;
    for (uint32_t cr = 0; cr < core_ranges.size(); cr++) {
        TT_FATAL(
            core_ranges.at(cr).start_coord.x <= core_ranges.at(cr).end_coord.x,
            "end coordinates left of start coordinates in shard");
        TT_FATAL(core_ranges.at(cr).end_coord.x <= 0xFF, "sharding coordinates out of range");
        TT_FATAL(
            core_ranges.at(cr).start_coord.y <= core_ranges.at(cr).end_coord.y,
            "end coordinates above of start coordinates in shard");
        TT_FATAL(core_ranges.at(cr).end_coord.y <= 0xFF, "sharding coordinates out of range");
        if (shard_grid_transposed) {
            for (uint32_t x_index = core_ranges.at(cr).start_coord.x; x_index <= core_ranges.at(cr).end_coord.x;
                 x_index++) {
                for (uint32_t y_index = core_ranges.at(cr).start_coord.y; y_index <= core_ranges.at(cr).end_coord.y;
                     y_index++) {
                    CoreCoord noc_core = device->worker_core_from_logical_core(CoreCoord(x_index, y_index));
                    coordinates.push_back(noc_core);
                }
            }
        } else {
            for (uint32_t y_index = core_ranges.at(cr).start_coord.y; y_index <= core_ranges.at(cr).end_coord.y;
                 y_index++) {
                for (uint32_t x_index = core_ranges.at(cr).start_coord.x; x_index <= core_ranges.at(cr).end_coord.x;
                     x_index++) {
                    CoreCoord noc_core = device->worker_core_from_logical_core(CoreCoord(x_index, y_index));
                    coordinates.push_back(noc_core);
                }
            }
        }
    }
    return coordinates;
}

std::vector<uint32_t> get_linear_shard_list(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t) {
    std::vector<uint32_t> args;
    struct ShardSpec shard_spec = t.shard_spec().value();
    const auto core_ranges = t.buffer()->shard_spec().grid().ranges();
    bool shard_grid_transposed =
        ((t.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
          shard_spec.orientation == ShardOrientation::ROW_MAJOR) ||
         ((t.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
           t.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) &&
          shard_spec.orientation == ShardOrientation::COL_MAJOR));
    bool last = false;
    uint32_t held_value = 0;
    uint32_t concatenated_core = 0;
    for (uint32_t cr = 0; cr < core_ranges.size(); cr++) {
        TT_FATAL(
            core_ranges.at(cr).start_coord.x <= core_ranges.at(cr).end_coord.x,
            "end coordinates left of start coordinates in shard");
        TT_FATAL(core_ranges.at(cr).end_coord.x <= 0xFF, "sharding coordinates out of range");
        TT_FATAL(
            core_ranges.at(cr).start_coord.y <= core_ranges.at(cr).end_coord.y,
            "end coordinates above of start coordinates in shard");
        TT_FATAL(core_ranges.at(cr).end_coord.y <= 0xFF, "sharding coordinates out of range");
        if (shard_grid_transposed) {
            for (uint32_t x_index = core_ranges.at(cr).start_coord.x; x_index <= core_ranges.at(cr).end_coord.x;
                 x_index++) {
                for (uint32_t y_index = core_ranges.at(cr).start_coord.y; y_index <= core_ranges.at(cr).end_coord.y;
                     y_index++) {
                    CoreCoord noc_core = device->worker_core_from_logical_core(CoreCoord(x_index, y_index));
                    concatenated_core = (noc_core.x & 0xFF) << 8 | (noc_core.y & 0xFF);
                    if (last) {
                        args.push_back(concatenated_core | (held_value << 16));
                    } else {
                        held_value = concatenated_core;
                    }
                    last = !last;
                }
            }
        } else {
            for (uint32_t y_index = core_ranges.at(cr).start_coord.y; y_index <= core_ranges.at(cr).end_coord.y;
                 y_index++) {
                for (uint32_t x_index = core_ranges.at(cr).start_coord.x; x_index <= core_ranges.at(cr).end_coord.x;
                     x_index++) {
                    CoreCoord noc_core = device->worker_core_from_logical_core(CoreCoord(x_index, y_index));
                    concatenated_core = (noc_core.x & 0xFF) << 8 | (noc_core.y & 0xFF);
                    if (last) {
                        args.push_back(concatenated_core | (held_value << 16));
                    } else {
                        held_value = concatenated_core;
                    }
                    last = !last;
                }
            }
        }
    }
    if (last) {
        args.push_back((held_value << 16));
    }
    return args;
}

static void add_sharding_rt_to_existing_rt(
    const IDevice* d, const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args) {
    const auto& new_args = get_linear_shard_list(d, t);
    std::copy(std::begin(new_args), std::end(new_args), std::back_inserter(args));
}

std::vector<uint32_t> sharding_ct_table_builder(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t) {
    std::vector<uint32_t> args;
    TT_ASSERT(t.is_sharded());
    TT_FATAL(
        t.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            t.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            t.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "ShardedAddrGenArgBuilder::emit_ct_args was invoked with a tensor containing an unsupported (Sharded) Tensor "
        "Memory Layout: {}",
        t.memory_config().memory_layout);
    struct ShardSpec shard_spec = t.shard_spec().value();
    struct ShardSpecBuffer buf_shard_spec = t.buffer()->shard_spec();
    const auto& [pages_per_shard_y, pages_per_shard_x] = buf_shard_spec.shape_in_pages();
    // contiguity is 0 if there is padding between unaligned page, 1 if there is padding in the rightmost shard, and 2
    // otherwise
    Contiguity_types contiguity =
        (t.buffer()->aligned_page_size() != t.buffer()->page_size()) ? Contiguity_types::PADDING_BETWEEN_PAGES
        : (buf_shard_spec.tensor2d_shape[1] == (pages_per_shard_x * get_sharding_core_count(t)))
            ? Contiguity_types::NO_SHARD_PADDING
            : Contiguity_types::PADDING_IN_RIGHTMOST_SHARD;
    args.push_back(static_cast<uint32_t>(t.memory_config().memory_layout));  // Memory layout
    args.push_back(static_cast<uint32_t>(get_sharding_core_count(t)));       // The number of sharding cores
    args.push_back(static_cast<uint32_t>(t.buffer()->aligned_page_size()));  // The page size we offset each write to
    TT_FATAL(t.buffer()->aligned_page_size() > 0, "algined page size is 0");
    args.push_back(static_cast<uint32_t>(
        buf_shard_spec.tensor2d_shape[1]));  // The number of pages in each sharding row not including padding pages
    TT_FATAL(buf_shard_spec.tensor2d_shape[1], "the page is empty");
    args.push_back(static_cast<uint32_t>(contiguity));  // This defines times when contiguous pages can't be calculated
    args.push_back(pages_per_shard_x);
    args.push_back(pages_per_shard_y);
    return args;
}

static void add_sharding_ct_to_existing_ct(
    const IDevice* d, const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args) {
    const auto& new_args = sharding_ct_table_builder(d, t);
    std::copy(std::begin(new_args), std::end(new_args), std::back_inserter(args));
}

}  // namespace shard_pf_builder
