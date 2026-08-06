// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_generic.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// (Names are prefixed to avoid unity-build collisions with the sibling reshard factories.)
constexpr const char* kGenReaderKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp";
constexpr const char* kGenReaderDiffWidthKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader_diff_width.cpp";

// Resource / parameter names referenced by the kernel sources (tensor:: / dfb:: accessors).
constexpr const char* kGenInputTensorParam = "input";
constexpr const char* kGenOutputTensorParam = "output";
constexpr const char* kGenOutputShardDfbName = "output_shard";

constexpr const char* kGenReaderKernel = "reader";
constexpr const char* kGenWriterKernel = "writer";

}  // namespace

namespace detail {
// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct Stride {
    CoreCoord core;
    uint32_t data{};
};

struct PageStride {
    CoreCoord start_core;
    uint32_t start_data{};
    uint32_t stride_size{};  // number of pages per stride
    Stride stride;
    uint32_t num_strides{};
    bool skip{};
};

struct CompressedStrideBlock {
    std::vector<PageStride> base_pattern;
    std::vector<Stride> meta_strides;
    uint32_t num_repeats = 0;
};

enum class ReshardStridesInRange { ALL_STRIDES, FIRST_HALF };

std::unordered_map<CoreCoord, std::vector<detail::PageStride>> create_map_for_reshard(
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page,
    Buffer* input_buffer,
    Buffer* output_buffer) {
    std::unordered_map<CoreCoord, std::vector<detail::PageStride>> ret_map;
    auto output_cores = output_buffer->get_buffer_page_mapping()->all_cores;
    ret_map.reserve(output_cores.size());

    auto* device = input_buffer->device();
    auto full_grid = device->compute_with_storage_grid_size();
    uint32_t output_core_id = 0;
    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<PageStride>{});

        const auto& input_cores_with_pages = output_core_to_vector_input_core_page[output_core_id];
        auto it = input_cores_with_pages.begin();
        const auto end = input_cores_with_pages.end();

        while (it != end) {
            // hit padding, will see how many consecutive pages has padding to make a padded range
            if (!it->has_value()) {
                auto consecutive_it = it + 1;
                auto last_it_consec = it;
                while (consecutive_it != end) {
                    if (consecutive_it->has_value()) {
                        break;
                    }
                    last_it_consec = consecutive_it;
                    consecutive_it = consecutive_it + 1;
                }
                uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                ret_map[output_core].push_back(PageStride{
                    .start_core = output_core,
                    .start_data = 0,
                    .stride_size = stride_size,
                    .stride = Stride{.core = {0, 0}, .data = 0},
                    .num_strides = 1,
                    .skip = true});
                it += stride_size;
            } else {
                const auto start_core = it->value().first;
                Stride stride = Stride{.core = {0, 0}, .data = 0};
                if ((it + 1) == end) {
                    ret_map[output_core].push_back(PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = 1,
                        .stride = stride,
                        .num_strides = 1,
                        .skip = false});
                    it = end;
                } else {
                    // first get a single stride, go through the number of consecutive pages in the same core
                    auto consecutive_it = it + 1;
                    auto last_it_consec = it;
                    while (consecutive_it != end and consecutive_it->has_value()) {
                        auto next_input_page = *(consecutive_it);
                        auto curr_input_page = *(last_it_consec);
                        // diff core or not consecutive
                        if (curr_input_page.value().first != next_input_page.value().first ||
                            (curr_input_page.value().second + 1) != next_input_page.value().second) {
                            break;
                        }
                        // next page is padding
                        last_it_consec = consecutive_it;
                        consecutive_it = consecutive_it + 1;
                    }
                    uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                    auto stride_it = it + stride_size;
                    auto last_it_stride = it;

                    // TT_ASSERT((stride_it == end) or stride_it->has_value());
                    TT_ASSERT(last_it_stride->has_value());
                    // if stride_range is within same core
                    // the jump in data is end of curr - end last stride
                    // if stride range is in diff core
                    // jump in data is curr - beginning of last stride
                    uint32_t data_stride;
                    if ((stride_it != end) and (stride_it != it) and stride_it->has_value()) {
                        // data stride within core
                        if (stride_it->has_value() and stride_it->value().first == last_it_stride->value().first and
                            (stride_it->value().second > last_it_stride->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *(last_it_stride);
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = next_input_page.value().second - prev_input_page.value().second - stride_size;
                            stride = Stride{.core = {0, 0}, .data = data_stride};
                        }
                        // strided core but same data
                        // currently only handling increasing cores within same stride
                        // TODO : negative strides for cores
                        else if (
                            stride_it->has_value() and (stride_it->value().first != last_it_stride->value().first) and
                            (stride_it->value().first.x >= it->value().first.x and
                             stride_it->value().first.y >= it->value().first.y) and
                            (stride_it->value().second == it->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *it;
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = 0;
                            stride = Stride{
                                .core =
                                    {next_input_page.value().first.x - prev_input_page.value().first.x,
                                     next_input_page.value().first.y - prev_input_page.value().first.y},
                                .data = data_stride};
                        }
                        // diff data and diff core, not handled yet
                        else {
                            TT_ASSERT(it->has_value());
                            ret_map[output_core].push_back(PageStride{
                                .start_core = start_core,
                                .start_data = it->value().second,
                                .stride_size = stride_size,
                                .stride = stride,
                                .num_strides = 1,
                                .skip = false});
                            it = stride_it;
                            continue;
                        }
                        // TODO add stride of data and core
                    }
                    // only single stride
                    else {
                        data_stride = 0;
                    }

                    TT_ASSERT(stride.core.x < full_grid.x and stride.core.y < full_grid.y);
                    TT_ASSERT(data_stride < output_buffer->num_pages());
                    uint32_t num_strides = 1;
                    while (stride_it != end and stride_it->has_value()) {
                        bool stride_not_complete = false;
                        auto stride_it_inner = stride_it + 1;
                        auto last_it_stride_inner = stride_it;
                        for (uint32_t i = 0; i < stride_size - 1; i++) {
                            auto next_input_page = *(stride_it_inner);
                            auto curr_input_page = *(last_it_stride_inner);
                            TT_ASSERT(curr_input_page.has_value());
                            int increment = 1;
                            if (!(next_input_page.has_value()) or
                                (next_input_page.value().first != curr_input_page.value().first) or
                                ((int)next_input_page.value().second !=
                                 (int)(curr_input_page.value().second) + (int)increment)) {
                                stride_not_complete = true;
                                break;
                            }
                            last_it_stride_inner = stride_it_inner;
                            stride_it_inner = stride_it_inner + 1;
                        }
                        if (stride_not_complete) {
                            break;
                        }
                        num_strides++;
                        last_it_stride = stride_it_inner - 1;
                        stride_it = stride_it_inner;
                        if (stride_it == end or !stride_it->has_value()) {
                            break;
                        }
                        auto next_input_page = *(stride_it);
                        auto curr_input_page = *(last_it_stride);
                        // TT_ASSERT(curr_input_page.has_value());
                        if (!curr_input_page.has_value() or !next_input_page.has_value() or
                            (next_input_page.value().first.x - curr_input_page.value().first.x != stride.core.x) or
                            (next_input_page.value().first.y - curr_input_page.value().first.y != stride.core.y) or
                            (abs((int)next_input_page.value().second - (int)curr_input_page.value().second) !=
                             (int)stride.data)) {
                            break;
                        }
                    }
                    TT_ASSERT(it->has_value());
                    ret_map[output_core].push_back(PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = stride_size,
                        .stride = stride,
                        .num_strides = num_strides,
                        .skip = false});
                    it = stride_it;
                }
            }
        }
        output_core_id++;
    }
    return ret_map;
}

bool are_strides_structurally_equal(const PageStride& a, const PageStride& b) {
    return a.stride_size == b.stride_size && a.stride.core == b.stride.core && a.stride.data == b.stride.data &&
           a.num_strides == b.num_strides && a.skip == b.skip;
}

std::unordered_map<CoreCoord, std::vector<detail::CompressedStrideBlock>> create_stride_of_strides_ret_map(
    const std::unordered_map<CoreCoord, std::vector<detail::PageStride>>& input_map) {
    std::unordered_map<CoreCoord, std::vector<detail::CompressedStrideBlock>> ret_map;
    ret_map.reserve(input_map.size());

    for (const auto& [core, page_strides] : input_map) {
        if (page_strides.empty()) {
            continue;
        }
        std::vector<detail::CompressedStrideBlock> compressed_blocks;
        compressed_blocks.reserve(page_strides.size());
        auto it = page_strides.cbegin();
        while (it != page_strides.cend()) {
            size_t best_pattern_len = 0;
            uint32_t best_num_repeats = 1;
            std::vector<Stride> best_meta_strides;

            // Find the longest repeating pattern starting at the current position `it`
            for (size_t pattern_len = 1; it + pattern_len <= page_strides.cend(); ++pattern_len) {
                auto pattern_begin = it;
                uint32_t num_repeats = 1;

                // First, find how many times the pattern is structurally repeated
                while (true) {
                    auto next_block_start = it + num_repeats * pattern_len;
                    if (next_block_start + pattern_len > page_strides.cend()) {
                        break;  // Not enough elements for another full repetition
                    }

                    bool structurally_equal = true;
                    for (size_t i = 0; i < pattern_len; ++i) {
                        if (!are_strides_structurally_equal(*(pattern_begin + i), *(next_block_start + i))) {
                            structurally_equal = false;
                            break;
                        }
                    }

                    if (!structurally_equal) {
                        break;
                    }
                    num_repeats++;
                }

                if (num_repeats <= 1) {
                    continue;
                }

                // Now, validate that the start coords/data progress with a consistent stride
                auto first_repeat_begin = it + pattern_len;
                std::vector<Stride> meta_strides;
                meta_strides.reserve(pattern_len);
                for (size_t i = 0; i < pattern_len; i++) {
                    const auto& base_ps = *(pattern_begin + i);
                    const auto& repeat_ps = *(first_repeat_begin + i);
                    meta_strides.push_back(
                        {.core =
                             {repeat_ps.start_core.x - base_ps.start_core.x,
                              repeat_ps.start_core.y - base_ps.start_core.y},
                         .data = repeat_ps.start_data - base_ps.start_data});
                }

                bool all_repeats_valid = true;
                for (uint32_t r = 1; r < num_repeats; ++r) {
                    auto current_block_start = it + r * pattern_len;
                    for (size_t i = 0; i < pattern_len; ++i) {
                        const auto& original_page_stride = *(pattern_begin + i);
                        const auto& current_page_stride = *(current_block_start + i);
                        const auto& pattern_meta_stride = meta_strides[i];

                        CoreCoord expected_core = {
                            original_page_stride.start_core.x + (r * pattern_meta_stride.core.x),
                            original_page_stride.start_core.y + (r * pattern_meta_stride.core.y)};
                        uint32_t expected_data = original_page_stride.start_data + (r * pattern_meta_stride.data);

                        if (current_page_stride.start_core != expected_core ||
                            current_page_stride.start_data != expected_data) {
                            all_repeats_valid = false;
                            num_repeats = r;  // This pattern is only valid for r repetitions
                            break;
                        }
                    }
                    if (!all_repeats_valid) {
                        break;
                    }
                }

                if (num_repeats > 1 && (pattern_len * num_repeats) > (best_pattern_len * best_num_repeats)) {
                    best_pattern_len = pattern_len;
                    best_num_repeats = num_repeats;
                    best_meta_strides = meta_strides;
                }
            }

            if (best_pattern_len > 0) {
                // A compressible pattern was found.
                std::vector<PageStride> base_pattern(it, it + best_pattern_len);
                compressed_blocks.push_back(
                    {.base_pattern = std::move(base_pattern),
                     .meta_strides = std::move(best_meta_strides),
                     .num_repeats = best_num_repeats});
                it += best_pattern_len * best_num_repeats;
            } else {
                // No repeating pattern found, treat as a block of 1.
                compressed_blocks.push_back({.base_pattern = {*it}, .meta_strides = {{}}, .num_repeats = 1});
                it++;
            }
        }
        ret_map.try_emplace(core, std::move(compressed_blocks));
    }
    return ret_map;
}

std::unordered_map<CoreCoord, std::vector<detail::PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    const auto& output_buffer_page_mapping = *output_buffer->get_buffer_page_mapping();
    const auto& input_buffer_page_mapping = *input_buffer->get_buffer_page_mapping();

    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_input_core_mapping(input_buffer->num_pages());
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        host_page_to_input_core_mapping[mapped_page.host_page] = {core, mapped_page.device_page};
    }

    auto output_cores = output_buffer_page_mapping.all_cores;

    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (auto mapped_page : output_buffer_page_mapping) {
        auto& cur_output_core_to_vector_input_core_page = output_core_to_vector_input_core_page[mapped_page.core_id];
        auto [input_core, input_core_page] = host_page_to_input_core_mapping[mapped_page.host_page];
        if (cur_output_core_to_vector_input_core_page.size() <= mapped_page.device_page) {
            cur_output_core_to_vector_input_core_page.resize(mapped_page.device_page + 1);
        }
        cur_output_core_to_vector_input_core_page[mapped_page.device_page] = {input_core, input_core_page};
    }
    auto ret_map = create_map_for_reshard(output_core_to_vector_input_core_page, input_buffer, output_buffer);
    return ret_map;
}

std::unordered_map<CoreCoord, std::vector<detail::CompressedStrideBlock>> get_core_page_ranges_diff_width(
    Buffer* input_buffer, Buffer* output_buffer, const Tensor& input) {
    const auto& output_buffer_page_mapping = *output_buffer->get_buffer_page_mapping();
    const auto& input_buffer_page_mapping = *input_buffer->get_buffer_page_mapping();
    uint32_t num_rows = 1;
    for (uint32_t i = 0; i < input.logical_shape().rank() - 1; i++) {
        num_rows *= input.logical_shape()[i];
    }
    // Find GCD of page sizes to use as the new base page size
    uint32_t input_page_size = input_buffer->page_size();
    uint32_t output_page_size = output_buffer->page_size();
    uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

    // Calculate how many base pages make up an input/output page
    uint32_t input_pages_per_original = input_page_size / base_page_size;
    uint32_t output_pages_per_original = output_page_size / base_page_size;

    auto input_width = input_buffer->shard_spec().shape()[1];
    auto output_width = output_buffer->shard_spec().shape()[1];
    auto total_width = input.logical_shape()[-1];

    uint32_t total_page_number =
        (input.logical_shape()[-1] * input.element_size() + base_page_size - 1) / base_page_size;
    uint32_t num_input_pages_per_row = input_pages_per_original * ((total_width + input_width - 1) / input_width);
    uint32_t num_output_pages_per_row = output_pages_per_original * ((total_width + output_width - 1) / output_width);

    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_input_core_mapping(total_page_number * num_rows);

    // data structure to account for padded base pages in the mapping
    std::unordered_map<uint32_t, uint32_t> invalid_mapping_input;
    std::unordered_map<uint32_t, uint32_t> invalid_mapping_output;
    uint32_t num_invalid_pages_input = 0;
    uint32_t num_invalid_pages_output = 0;
    for (uint32_t i = 1; i <= num_rows; i++) {
        invalid_mapping_input[total_page_number * i] = 0;
        invalid_mapping_output[total_page_number * i] = 0;
    }

    // find input invalid base pages if applicable
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        CoreCoord shard_grid = input_buffer->shard_spec().grid().ranges()[0].grid_size();
        bool is_last_in_row = (core.x == shard_grid.x - 1);
        if (input_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            is_last_in_row = (core.y == shard_grid.y - 1);
        }
        uint32_t base_start_page = mapped_page.host_page * input_pages_per_original;
        uint32_t valid_pages = input_pages_per_original;
        if (is_last_in_row) {
            uint32_t next_total =
                ((base_start_page + num_input_pages_per_row) / num_input_pages_per_row) * total_page_number;
            next_total = std::max(next_total, total_page_number);
            valid_pages = std::min(next_total - base_start_page, input_pages_per_original);
        }
        if (input_pages_per_original - valid_pages > 0) {
            num_invalid_pages_input = input_pages_per_original - valid_pages;
            break;
        }
    }

    // find output invalid base pages if applicable
    for (auto mapped_page : output_buffer_page_mapping) {
        auto core = output_buffer_page_mapping.all_cores[mapped_page.core_id];
        CoreCoord shard_grid = output_buffer->shard_spec().grid().ranges()[0].grid_size();
        bool is_last_in_row = (core.x == shard_grid.x - 1);
        if (output_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            is_last_in_row = (core.y == shard_grid.y - 1);
        }
        uint32_t base_start_page = mapped_page.host_page * output_pages_per_original;
        uint32_t valid_pages = output_pages_per_original;
        if (is_last_in_row) {
            uint32_t next_total =
                ((base_start_page + num_output_pages_per_row) / num_output_pages_per_row) * total_page_number;
            valid_pages = std::min(next_total - base_start_page, output_pages_per_original);
        }
        if (output_pages_per_original - valid_pages > 0) {
            num_invalid_pages_output = output_pages_per_original - valid_pages;
            break;
        }
    }

    for (uint32_t i = 1; i <= num_rows; i++) {
        invalid_mapping_input[total_page_number * i] = (i - 1) * num_invalid_pages_input;
        invalid_mapping_output[total_page_number * i] = (i - 1) * num_invalid_pages_output;
    }

    // Create mapping of input base host pages to their cores
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        uint32_t base_start_page = mapped_page.host_page * input_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * input_pages_per_original;
        uint32_t next_total =
            ((base_start_page + num_input_pages_per_row) / num_input_pages_per_row) * total_page_number;
        next_total = std::max(next_total, total_page_number);
        base_start_page = base_start_page - invalid_mapping_input[next_total];
        uint32_t valid_pages = std::min(next_total - base_start_page, input_pages_per_original);
        for (uint32_t i = 0; i < valid_pages; i++) {
            host_page_to_input_core_mapping[base_start_page + i] = {core, device_base_start + i};
        }
    }

    // Create similar mapping for output pages to their cores
    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_output_core_mapping(total_page_number * num_rows);

    for (auto mapped_page : output_buffer_page_mapping) {
        auto core = output_buffer_page_mapping.all_cores[mapped_page.core_id];
        uint32_t base_start_page = mapped_page.host_page * output_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * output_pages_per_original;

        uint32_t next_total =
            ((base_start_page + num_output_pages_per_row) / num_output_pages_per_row) * total_page_number;
        next_total = std::max(next_total, total_page_number);
        base_start_page = base_start_page - invalid_mapping_output[next_total];
        uint32_t valid_pages = std::min(next_total - base_start_page, output_pages_per_original);
        for (uint32_t i = 0; i < valid_pages; i++) {
            host_page_to_output_core_mapping[base_start_page + i] = {core, device_base_start + i};
        }
    }
    // Create final mapping of output cores to input pages they need
    auto output_cores = output_buffer_page_mapping.all_cores;
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (uint32_t core_id = 0; core_id < output_cores.size(); core_id++) {
        auto& cur_output_core_pages = output_core_to_vector_input_core_page[core_id];

        // Find all host pages that map to this output core
        for (uint32_t host_page = 0; host_page < host_page_to_output_core_mapping.size(); host_page++) {
            if (host_page_to_output_core_mapping[host_page].first == output_cores[core_id]) {
                // This host page belongs to current output core
                // Get corresponding input core and page
                auto input_mapping = host_page_to_input_core_mapping[host_page];

                // Add to vector if needed
                uint32_t device_page = host_page_to_output_core_mapping[host_page].second;
                if (cur_output_core_pages.size() <= device_page) {
                    cur_output_core_pages.resize(device_page + 1);
                }
                cur_output_core_pages[device_page] = input_mapping;
            }
        }
    }

    auto ret_map = create_map_for_reshard(output_core_to_vector_input_core_page, input_buffer, output_buffer);

    auto processed_ret_map = create_stride_of_strides_ret_map(ret_map);
    return processed_ret_map;
}

// Per-(kernel, node) runtime arguments for one half of a core's range list.
//
// The input-buffer base address the legacy helpers emitted (and the factory then back-patched to a
// Buffer* binding) is gone: it now reaches the kernel through tensor::input. What remains splits in
// two: three scalars that become named RTAs, and the per-range / per-block words that become
// positional varargs. The physical-core-coordinate table stays a vararg prefix, prepended by the
// caller, so the kernel's data-selected indexing into it is unchanged.
struct ReshardRangeArgs {
    uint32_t num_output_pages = 0;
    uint32_t num_ranges = 0;  // num_blocks for the diff-width source
    uint32_t output_page_offset = 0;
    std::vector<uint32_t> payload;
};

ReshardRangeArgs get_runtime_args_for_given_ranges_diff_width(
    const std::vector<detail::CompressedStrideBlock>& compressed_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t starting_range,
    const uint32_t ending_range) {
    ReshardRangeArgs out;
    out.num_ranges = ending_range - starting_range;
    out.output_page_offset = output_page_offset;

    for (uint32_t block_id = starting_range; block_id < ending_range; block_id++) {
        const auto& block = compressed_stride_vector[block_id];

        out.payload.push_back(block.num_repeats);
        out.payload.push_back(block.base_pattern.size());

        uint32_t pages_in_base_pattern = 0;
        for (size_t i = 0; i < block.base_pattern.size(); ++i) {
            const auto& ps = block.base_pattern[i];
            const auto& ms = block.meta_strides[i];

            // Pack meta stride
            uint32_t meta_stride_core = (static_cast<uint32_t>(ms.core.x) << 16) | static_cast<uint32_t>(ms.core.y);
            out.payload.push_back(meta_stride_core);
            out.payload.push_back(ms.data);

            // Pack page stride
            uint32_t core_start_stride =
                (ps.start_core.x << 24) | (ps.start_core.y << 16) | (ps.stride.core.x << 8) | ps.stride.core.y;
            out.payload.push_back(core_start_stride);
            uint32_t stride_data_start = (ps.stride.data << 16) | (ps.start_data);
            out.payload.push_back(stride_data_start);
            uint32_t stride_size_num_strides = (ps.stride_size << 16) | (ps.num_strides << 8) | ((uint32_t)ps.skip);
            out.payload.push_back(stride_size_num_strides);

            pages_in_base_pattern += ps.stride_size * ps.num_strides;
        }
        out.num_output_pages += pages_in_base_pattern * block.num_repeats;
    }
    return out;
}

ReshardRangeArgs get_runtime_args_for_given_ranges(
    const std::vector<PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const ReshardStridesInRange reshard_strides_in_range = ReshardStridesInRange::ALL_STRIDES) {
    ReshardRangeArgs out;
    out.num_ranges = ending_range - starting_range;
    out.output_page_offset = output_page_offset;
    uint32_t num_output_pages = 0;
    for (uint32_t range_id = starting_range; range_id < ending_range; range_id++) {
        PageStride ps = page_stride_vector[range_id];
        uint32_t num_strides;
        uint32_t start_core_x;
        uint32_t start_core_y;
        uint32_t start_data;
        if (reshard_strides_in_range == ReshardStridesInRange::ALL_STRIDES) {
            num_strides = ps.num_strides;
            start_core_x = ps.start_core.x;
            start_core_y = ps.start_core.y;
            start_data = ps.start_data;
        } else {
            if (reshard_strides_in_range == ReshardStridesInRange::FIRST_HALF) {
                num_strides = ps.num_strides / 2;
                start_core_x = ps.start_core.x;
                start_core_y = ps.start_core.y;
                start_data = ps.start_data;
            } else {
                uint32_t strides_in_first_half = ps.num_strides / 2;
                num_strides = ps.num_strides - (strides_in_first_half);
                start_core_x = ps.start_core.x + (strides_in_first_half * ps.stride.core.x);
                start_core_y = ps.start_core.y + (strides_in_first_half * ps.stride.core.y);
                start_data = ps.start_data + (strides_in_first_half * ps.start_data);
            }
        }
        if (num_strides > 0) {
            uint32_t core_start_stride =
                (start_core_x << 24) | (start_core_y << 16) | (ps.stride.core.x << 8) | ps.stride.core.y;
            out.payload.push_back((uint32_t)core_start_stride);  // start_x
            uint32_t stride_data_start = (ps.stride.data << 16) | (start_data);
            out.payload.push_back((uint32_t)stride_data_start);  // stride_data
            uint32_t stride_size_num_strides = (ps.stride_size << 16) | (num_strides << 8) | ((uint32_t)ps.skip);
            out.payload.push_back((uint32_t)stride_size_num_strides);  // stride_size
            num_output_pages += ps.stride_size * num_strides;
        }
    }
    out.num_output_pages = num_output_pages;
    return out;
}

}  // namespace detail

ttnn::device_operation::ProgramArtifacts ReshardGenericFactory::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* device = input.device();
    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    auto grid = input_buffer->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                : device->compute_with_storage_grid_size();
    auto input_core_type = input_buffer->core_type();
    auto cores = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));

    uint32_t total_size = 0;
    uint32_t page_size = 0;
    uint32_t unit_size = 0;
    auto output_shard_shape = output.shard_spec().value().shape;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    if (input.layout() == Layout::TILE) {
        page_size = tt::tile_size(data_format);
        unit_size = page_size;
        total_size = static_cast<uint32_t>(output.shard_spec().value().numel() / TILE_HW) * unit_size;
    } else {
        // For ROW_MAJOR, use base page size from GCD calculation
        uint32_t input_page_size = input_buffer->page_size();
        uint32_t output_page_size = output_buffer->page_size();
        uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

        unit_size = base_page_size;
        page_size = base_page_size;
        total_size = static_cast<uint32_t>(output_shard_shape[0] * output_shard_shape[1] * output.element_size());
    }

    const bool diff_width = input_buffer->page_size() != output_buffer->page_size();
    const char* kernel_source = diff_width ? kGenReaderDiffWidthKernelPath : kGenReaderKernelPath;

    std::vector<uint32_t> physical_core_coords;
    physical_core_coords.reserve(grid.x * grid.y);
    for (uint32_t i = 0; i < grid.x; i++) {
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(i, 0), input_core_type);
        physical_core_coords.push_back(physical_input_core.x);
    }
    for (uint32_t i = 0; i < grid.y; i++) {
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(0, i), input_core_type);
        physical_core_coords.push_back(physical_input_core.y);
    }

    // ------------------------------------------------------------------
    // Per-core runtime argument generation. Unchanged from the legacy factory apart from the
    // packing: the input base address is no longer emitted (it arrives via tensor::input), the
    // three scalars become named RTAs, and the range payload becomes varargs.
    // ------------------------------------------------------------------
    std::vector<detail::ReshardRangeArgs> reader_args;
    std::vector<detail::ReshardRangeArgs> writer_args;
    reader_args.reserve(cores.size());
    writer_args.reserve(cores.size());

    for (const auto& core : cores) {
        if (diff_width) {
            auto output_core_to_page_range_pair =
                detail::get_core_page_ranges_diff_width(input_buffer, output_buffer, input);
            const auto& page_stride_vector = output_core_to_page_range_pair.at(core);
            auto args_0 = detail::get_runtime_args_for_given_ranges_diff_width(
                page_stride_vector, 0, 0, tt::div_up(static_cast<uint32_t>(page_stride_vector.size()), 2u));
            auto args_1 = detail::get_runtime_args_for_given_ranges_diff_width(
                page_stride_vector,
                args_0.num_output_pages,
                tt::div_up(static_cast<uint32_t>(page_stride_vector.size()), 2u),
                page_stride_vector.size());
            reader_args.push_back(std::move(args_0));
            writer_args.push_back(std::move(args_1));
        } else {
            auto output_core_to_page_range_pair = detail::get_core_page_ranges(input_buffer, output_buffer);
            const auto& page_stride_vector = output_core_to_page_range_pair.at(core);
            auto args_0 = detail::get_runtime_args_for_given_ranges(
                page_stride_vector, 0, 0, tt::div_up(static_cast<uint32_t>(page_stride_vector.size()), 2u));
            // offset is equivalent to number of pages output in previous risc core
            auto args_1 = detail::get_runtime_args_for_given_ranges(
                page_stride_vector,
                args_0.num_output_pages,
                tt::div_up(static_cast<uint32_t>(page_stride_vector.size()), 2u),
                page_stride_vector.size());
            reader_args.push_back(std::move(args_0));
            writer_args.push_back(std::move(args_1));
        }
    }

    // Vararg layout, per node: [physical_core_coords ... | range payload ... | zero padding].
    // The coordinate table keeps its leading position so the kernel's data-selected reads
    // (get_vararg(start_x_index), get_vararg(y_offset + start_y_index)) and its payload cursor
    // (arg_index = num_x_cores + num_y_cores) are both unchanged. A single per-kernel vararg count
    // covers every node, so each node's payload is zero-padded up to the longest one; the kernel's
    // loop is bounded by the named range count, so padding words are never read.
    const uint32_t num_coords = static_cast<uint32_t>(physical_core_coords.size());
    uint32_t max_payload = 0;
    for (const auto& args : reader_args) {
        max_payload = std::max(max_payload, static_cast<uint32_t>(args.payload.size()));
    }
    for (const auto& args : writer_args) {
        max_payload = std::max(max_payload, static_cast<uint32_t>(args.payload.size()));
    }
    const uint32_t num_varargs = num_coords + max_payload;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "reshard_generic";

    // The two sources name the outer loop bound differently (ranges vs compressed blocks).
    const char* count_arg_name = diff_width ? "num_blocks" : "num_ranges";

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"num_x_cores", static_cast<uint32_t>(grid.x)},
        {"num_y_cores", static_cast<uint32_t>(grid.y)},
        {"page_size", page_size},
        {"unit_size", unit_size},
    };

    const auto make_worker = [&](const char* name, DataMovementHardwareConfig hw_config, DFBEndpointType endpoint) {
        return KernelSpec{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_source),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kGenOutputShardDfbName},
                .accessor_name = kGenOutputShardDfbName,
                .endpoint_type = endpoint,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{kGenInputTensorParam}, .accessor_name = kGenInputTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema = {.runtime_arg_names = {"num_output_pages", count_arg_name, "output_page_offset"}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
    };

    // The two same-source instances raw-write disjoint output page ranges; the output shard is
    // resident and nothing drains it. Two role-free touchers over one grid -> assign 1P + 1C.
    spec.kernels = {
        make_worker(
            kGenReaderKernel, ttnn::create_reader_datamovement_config(device->arch()), DFBEndpointType::PRODUCER),
        make_worker(
            kGenWriterKernel, ttnn::create_writer_datamovement_config(device->arch()), DFBEndpointType::CONSUMER),
    };

    // Output sharded DFB, built on the output buffer's borrowed memory so its backing L1 address is
    // refreshed from the tensor argument on every enqueue.
    //
    // The DFB is only an address source: the kernel writes through get_write_ptr() + offset and never
    // touches the FIFO, so entry_size / num_entries do not bound its accesses. That matters because a
    // padded shard shape can make the shard-derived total_size exceed the output tensor's packed size
    // (a tiled [32, 96] tensor with a (32, 128) shard is 4 tiles of shard against 3 tiles of packed
    // tensor), and Metal 2.0's spec-time borrowed-DFB check compares against exactly that packed size
    // (program_spec.cpp:1541-1580) — it has no Buffer at spec time, so it cannot see the larger real
    // per-bank allocation the legacy dynamic CB was checked against. Clamp so the DFB never claims
    // more L1 than the TensorSpec reports; behaviour is unchanged because nothing reads the size.
    const uint32_t dfb_entry_size = static_cast<uint32_t>(output_buffer->page_size());
    const uint32_t output_packed_bytes = static_cast<uint32_t>(output.tensor_spec().compute_packed_buffer_size_bytes());
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = DFBSpecName{kGenOutputShardDfbName},
        .entry_size = dfb_entry_size,
        .num_entries = std::min(total_size, output_packed_bytes) / dfb_entry_size,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kGenOutputTensorParam},
    }};

    // `output` is a borrowed-only TensorParameter: it backs the output shard DFB and is never bound
    // by a kernel. The spec validator counts a borrowed_from reference as a use.
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kGenInputTensorParam}, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kGenOutputTensorParam}, .spec = output.tensor_spec()},
    };

    spec.work_units = {WorkUnitSpec{
        .name = "reshard_generic_work_unit",
        .kernels = {KernelSpecName{kGenReaderKernel}, KernelSpecName{kGenWriterKernel}},
        .target_nodes = all_cores,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    const auto build_kernel_run_args = [&](const char* name, const std::vector<detail::ReshardRangeArgs>& per_core) {
        KernelRunArgs run_args{.kernel = KernelSpecName{name}};
        for (size_t core_idx = 0; core_idx < cores.size(); ++core_idx) {
            const auto& node = cores[core_idx];
            AddRuntimeArgsForNode(
                run_args.runtime_arg_values,
                node,
                {{"num_output_pages", per_core[core_idx].num_output_pages},
                 {count_arg_name, per_core[core_idx].num_ranges},
                 {"output_page_offset", per_core[core_idx].output_page_offset}});
            AdvancedKernelRunArgs::Varargs varargs(num_varargs, 0u);
            std::copy(physical_core_coords.begin(), physical_core_coords.end(), varargs.begin());
            std::copy(
                per_core[core_idx].payload.begin(), per_core[core_idx].payload.end(), varargs.begin() + num_coords);
            run_args.advanced_options.runtime_varargs.emplace(node, std::move(varargs));
        }
        return run_args;
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        build_kernel_run_args(kGenReaderKernel, reader_args),
        build_kernel_run_args(kGenWriterKernel, writer_args),
    };
    run_params.tensor_args = {
        {TensorParamName{kGenInputTensorParam}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{kGenOutputTensorParam}, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
