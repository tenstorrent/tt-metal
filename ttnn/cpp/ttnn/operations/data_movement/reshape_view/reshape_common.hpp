// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include <cstdint>
#include "reshape_kernel_common.hpp"
#include <tt-metalium/work_split.hpp>

#pragma once
using PadValue = std::variant<uint32_t, float>;

namespace ttnn::operations::data_movement::reshape {
namespace detail {

struct Dims {
    Dims(const Shape& shape, const std::array<uint32_t, 2>& tile_shape) :
        w(tt::div_up(shape[-1], tile_shape[1])),
        h(tt::div_up(shape[-2], tile_shape[0])),
        c(w * h),
        total(shape[-3] * c) {}

    const uint32_t w;
    const uint32_t h;
    const uint32_t c;
    const uint32_t total;
};

inline std::tuple<uint32_t, uint32_t, uint32_t> page_index_to_tensor_idxs(
    const uint32_t& page_index, const std::array<uint32_t, 2>& tile_shape, const Dims& tile_dims) {
    const uint32_t c = page_index / tile_dims.c;

    const uint32_t hw_tile_offset = page_index % tile_dims.c;
    const uint32_t h_tile = hw_tile_offset / tile_dims.w;
    const uint32_t w_tile = hw_tile_offset % tile_dims.w;

    return std::make_tuple(c, h_tile * tile_shape[0], w_tile * tile_shape[1]);
}

inline auto idxs_to_reshaped_idxs(
    const uint32_t c1, const uint32_t h1, const uint32_t w1, const Shape& shape1, const Shape& shape2) {
    const uint32_t flat_offset = (c1 * shape1[-2] * shape1[-1]) + (h1 * shape1[-1]) + w1;

    const uint32_t c2 = flat_offset / (shape2[-2] * shape2[-1]);
    const uint32_t hw2 = flat_offset % (shape2[-2] * shape2[-1]);

    const uint32_t h2 = hw2 / shape2[-1];
    const uint32_t w2 = hw2 % shape2[-1];

    return std::make_tuple(c2, h2, w2);
}

inline uint32_t tensor_idxs_to_page_idx(
    const uint32_t c,
    const uint32_t h,
    const uint32_t w,
    const Shape& shape,
    const std::array<uint32_t, 2>& tile_shape,
    const Dims& tile_dims) {
    return (c * tile_dims.c) + (h / tile_shape[0] * tile_dims.w) + (w / tile_shape[1]);
}

inline uint32_t tensor_idxs_to_faced_tile_offset(
    const uint32_t h,
    const uint32_t w,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape) {
    const uint32_t face_dim_w = tile_shape[1] / face_shape[1];

    const uint32_t intra_tile_h = h % tile_shape[0];
    const uint32_t intra_tile_w = w % tile_shape[1];

    const uint32_t hf = intra_tile_h / face_shape[0];
    const uint32_t wf = intra_tile_w / face_shape[1];
    const uint32_t intra_face_h = intra_tile_h % face_shape[0];
    const uint32_t intra_face_w = intra_tile_w % face_shape[1];

    const uint32_t faceoffset = (hf * face_dim_w) + wf;

    return (faceoffset * (face_shape[0] * face_shape[1])) + (intra_face_h * face_shape[1]) + intra_face_w;
}

struct TileIterator {
    TileIterator(
        const uint32_t& in_start_h, const uint32_t& in_start_w, const uint32_t& in_end_h, const uint32_t& in_end_w) :
        start_h(in_start_h),
        start_w(in_start_w),
        tile_idx_h(0),
        tile_idx_w(0),
        tile_end_h(in_end_h - 1),
        tile_end_w(in_end_w - 1),
        first(true) {};

    bool next() {
        if (first) {
            first = false;
            return true;
        }
        if (tile_idx_w < tile_end_w) {
            ++tile_idx_w;
            return true;
        } else if (tile_idx_h < tile_end_h) {
            tile_idx_w = 0;
            ++tile_idx_h;
            return true;
        } else {
            return false;
        }
    }

    auto operator*() { return std::make_tuple(this->h(), this->w()); }

    uint32_t size() { return (tile_end_h + 1) * (tile_end_w + 1); }

protected:
    const uint32_t& start_h;
    const uint32_t& start_w;
    uint32_t tile_idx_h;
    uint32_t tile_idx_w;
    const uint32_t tile_end_h;
    const uint32_t tile_end_w;
    bool first;

    uint32_t h() { return start_h + tile_idx_h; }

    uint32_t w() { return start_w + tile_idx_w; }
};

inline std::vector<SegmentMapData> reshape_map_output_page(
    const uint32_t output_page_index,
    const Shape& input_shape,
    const Shape& output_shape,
    const Dims& tile_dims_input,
    const Dims& tile_dims_output,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape) {
    TT_ASSERT(input_shape.rank() == 3);
    TT_ASSERT(output_shape.rank() == 3);

    std::map<uint32_t, std::vector<SegmentMapData>> map_data;

    const auto [co_0, ho_0, wo_0] = page_index_to_tensor_idxs(output_page_index, tile_shape, tile_dims_output);
    const uint32_t ho_sz = std::min(output_shape[-2] - ho_0, tile_shape[0]);
    const uint32_t wo_sz = std::min(output_shape[-1] - wo_0, tile_shape[1]);

    TT_ASSERT(co_0 < output_shape[0]);

    TileIterator output_tile_iterator(ho_0, wo_0, ho_sz, wo_sz);

    uint32_t prev_offset_i{}, prev_offset_o{}, prev_page_idx_i{};

    // TODO there are properties of the mapping we could take advantage of to avoid some computation.
    while (output_tile_iterator.next()) {
        const auto [ho, wo] = *output_tile_iterator;
        const auto offset_o = tensor_idxs_to_faced_tile_offset(ho, wo, tile_shape, face_shape);

        TT_ASSERT(ho < output_shape[1], "{} {}", ho, output_shape[1]);
        TT_ASSERT(wo < output_shape[2], "{} {}", wo, output_shape[2]);
        TT_ASSERT(offset_o < tile_shape[0] * tile_shape[1]);

        const auto [ci, hi, wi] = idxs_to_reshaped_idxs(co_0, ho, wo, output_shape, input_shape);
        const auto page_idx_i = tensor_idxs_to_page_idx(ci, hi, wi, input_shape, tile_shape, tile_dims_input);
        const auto offset_i = tensor_idxs_to_faced_tile_offset(hi, wi, tile_shape, face_shape);

        TT_ASSERT(ci < input_shape[0]);
        TT_ASSERT(hi < input_shape[1], "hi: {} input_shape[1]: {} ", hi, input_shape[1]);
        TT_ASSERT(wi < input_shape[2], "wi: {} input_shape[2]: {} ", wi, input_shape[2]);
        TT_ASSERT(offset_i < tile_shape[0] * tile_shape[1]);

        if (map_data.count(page_idx_i)) {
            if (page_idx_i == prev_page_idx_i && offset_i - prev_offset_i == 1 && offset_o - prev_offset_o == 1) {
                ++map_data[page_idx_i].back().num_elements;
            } else {
                map_data[page_idx_i].emplace_back(page_idx_i, offset_i, offset_o, 1);
            }
        } else {
            map_data[page_idx_i].emplace_back(page_idx_i, offset_i, offset_o, 1);
        }

        prev_offset_o = offset_o;
        prev_offset_i = offset_i;
        prev_page_idx_i = page_idx_i;
    }

    auto total_num_elements = std::accumulate(map_data.begin(), map_data.end(), 0, [](auto acc, auto& v) {
        return acc + std::accumulate(
                         v.second.begin(), v.second.end(), 0, [](auto acc2, auto& d) { return acc2 + d.num_elements; });
    });

    TT_ASSERT(output_tile_iterator.size() == total_num_elements);

    // flatten map
    uint32_t max_input_segments_page = std::max_element(map_data.begin(), map_data.end(), [](auto& a, auto& b) {
                                           return a.second.size() < b.second.size();
                                       })->second.size();

    std::vector<SegmentMapData> flat_map_data(max_input_segments_page * map_data.size());
    auto it = flat_map_data.begin();
    for (const auto& m : map_data) {
        std::copy(m.second.begin(), m.second.end(), it);

        it += max_input_segments_page;
    }
    return flat_map_data;
}

// Pattern template: describes a stride run structure
struct PatternTemplate {
    int32_t input_page_stride;
    int32_t input_offset_stride;
    int32_t output_offset_stride;
    uint32_t num_elements;

    bool operator==(const PatternTemplate& other) const {
        return input_page_stride == other.input_page_stride && input_offset_stride == other.input_offset_stride &&
               output_offset_stride == other.output_offset_stride && num_elements == other.num_elements;
    }
    bool operator<(const PatternTemplate& other) const {
        if (input_page_stride != other.input_page_stride) {
            return input_page_stride < other.input_page_stride;
        }
        if (input_offset_stride != other.input_offset_stride) {
            return input_offset_stride < other.input_offset_stride;
        }
        if (output_offset_stride != other.output_offset_stride) {
            return output_offset_stride < other.output_offset_stride;
        }
        return num_elements < other.num_elements;
    }
};

// Instance of a pattern template for a specific output page
struct PagePatternInstance {
    uint32_t output_page_index;
    uint32_t input_page_index;
    uint32_t input_offset_start;
    uint32_t output_offset_start;
    uint32_t run_length;
    uint32_t pattern_template_index;  // Index into shared pattern table

    bool operator==(const PagePatternInstance& other) const {
        return output_page_index == other.output_page_index && input_page_index == other.input_page_index &&
               input_offset_start == other.input_offset_start && output_offset_start == other.output_offset_start &&
               run_length == other.run_length && pattern_template_index == other.pattern_template_index;
    }
};

struct PagePatternRun {
    uint32_t output_page_index_start;
    uint32_t output_page_index_end;  // inclusive
    uint32_t input_page_index_start;
    uint32_t input_offset_start;
    uint32_t output_offset_start;
    uint32_t run_length;
    uint32_t pattern_template_index;
    int32_t input_page_index_stride;  // stride between input_page_index for each output page
    int32_t input_offset_stride;      // stride between input_offset_start for each output page
    int32_t output_offset_stride;     // stride between output_offset_start for each output page
};

// Global compressed mapping: shared templates, per-page instances, and irregulars
struct GlobalCompressedReshapeMap {
    std::vector<PatternTemplate> pattern_templates;
    std::vector<PagePatternInstance> page_pattern_instances;
    std::vector<PagePatternRun> page_pattern_runs;
};

inline uint32_t pack_rt_short(uint16_t val1, uint16_t val2) { return (val1 << 16) | val2; }

inline size_t detect_stride_run(
    const std::vector<SegmentMapData>& segs,
    size_t start,
    PatternTemplate& tmpl,
    PagePatternInstance& instance,
    uint32_t output_page_index) {
    if (segs[start].num_elements == 0) {
        return 0;
    }
    auto& base = segs[start];
    size_t len = 1;
    int32_t input_page_stride = 0, input_offset_stride = 0, output_stride = 0;

    // Try to find stride by looking at consecutive segments
    if (start + 1 < segs.size() && segs[start + 1].num_elements == base.num_elements) {
        input_page_stride =
            static_cast<int32_t>(segs[start + 1].input_page_index) - static_cast<int32_t>(base.input_page_index);
        input_offset_stride =
            static_cast<int32_t>(segs[start + 1].input_page_offset) - static_cast<int32_t>(base.input_page_offset);
        output_stride =
            static_cast<int32_t>(segs[start + 1].output_page_offset) - static_cast<int32_t>(base.output_page_offset);
        len = 2;

        for (size_t i = start + 2; i < segs.size(); ++i) {
            if (segs[i].num_elements == 0 || segs[i].num_elements != base.num_elements) {
                break;
            }

            int32_t curr_input_page_stride =
                static_cast<int32_t>(segs[i].input_page_index) - static_cast<int32_t>(segs[i - 1].input_page_index);
            int32_t curr_input_offset_stride =
                static_cast<int32_t>(segs[i].input_page_offset) - static_cast<int32_t>(segs[i - 1].input_page_offset);
            int32_t curr_output_stride =
                static_cast<int32_t>(segs[i].output_page_offset) - static_cast<int32_t>(segs[i - 1].output_page_offset);

            if (curr_input_page_stride != input_page_stride || curr_input_offset_stride != input_offset_stride ||
                curr_output_stride != output_stride) {
                break;
            }
            len++;
        }

        tmpl = {input_page_stride, input_offset_stride, output_stride, base.num_elements};
        instance = {
            output_page_index,
            base.input_page_index,
            base.input_page_offset,
            base.output_page_offset,
            static_cast<uint32_t>(len),
            0};
        return len;
    }

    // Fall back to single segment
    tmpl = {0, 0, 0, base.num_elements};
    instance = {output_page_index, base.input_page_index, base.input_page_offset, base.output_page_offset, 1, 0};
    return 1;
}

inline std::vector<PagePatternRun> compress_page_pattern_instances(const std::vector<PagePatternInstance>& instances) {
    std::vector<PagePatternRun> runs;
    if (instances.empty()) {
        return runs;
    }

    size_t i = 0;
    while (i < instances.size()) {
        size_t j = i + 1;
        // Try to find a run
        int32_t input_page_index_stride = 0, input_offset_stride = 0, output_offset_stride = 0;
        if (j < instances.size()) {
            input_page_index_stride = static_cast<int32_t>(instances[j].input_page_index) -
                                      static_cast<int32_t>(instances[i].input_page_index);
            input_offset_stride = static_cast<int32_t>(instances[j].input_offset_start) -
                                  static_cast<int32_t>(instances[i].input_offset_start);
            output_offset_stride = static_cast<int32_t>(instances[j].output_offset_start) -
                                   static_cast<int32_t>(instances[i].output_offset_start);
        }
        while (j < instances.size() && instances[j].pattern_template_index == instances[i].pattern_template_index &&
               (instances[j].output_page_index == instances[i].output_page_index) &&
               (instances[j].output_page_index - instances[j - 1].output_page_index == 1) &&
               (instances[j].input_page_index - instances[j - 1].input_page_index == input_page_index_stride) &&
               (instances[j].input_offset_start - instances[j - 1].input_offset_start == input_offset_stride) &&
               (instances[j].output_offset_start - instances[j - 1].output_offset_start == output_offset_stride) &&
               (instances[j].run_length == instances[i].run_length)) {
            ++j;
        }
        runs.push_back(PagePatternRun{
            instances[i].output_page_index,
            instances[j - 1].output_page_index,
            instances[i].input_page_index,
            instances[i].input_offset_start,
            instances[i].output_offset_start,
            instances[i].run_length,
            instances[i].pattern_template_index,
            input_page_index_stride,
            input_offset_stride,
            output_offset_stride});
        i = j;
    }
    return runs;
}

inline GlobalCompressedReshapeMap compress_mapping_global(
    const std::vector<std::vector<SegmentMapData>>& mapping_vector) {
    std::vector<PatternTemplate> pattern_templates;
    std::vector<PagePatternInstance> page_pattern_instances;
    std::map<PatternTemplate, uint32_t> template_to_index;

    for (uint32_t output_page_idx = 0; output_page_idx < mapping_vector.size(); ++output_page_idx) {
        const auto& segments = mapping_vector[output_page_idx];
        size_t i = 0;
        while (i < segments.size()) {
            PatternTemplate tmpl;
            PagePatternInstance instance;
            size_t run_len = detect_stride_run(segments, i, tmpl, instance, output_page_idx);
            if (run_len >= 2) {
                uint32_t tmpl_idx;
                auto it = template_to_index.find(tmpl);
                if (it == template_to_index.end()) {
                    tmpl_idx = pattern_templates.size();
                    pattern_templates.push_back(tmpl);
                    template_to_index[tmpl] = tmpl_idx;
                } else {
                    tmpl_idx = it->second;
                }
                instance.pattern_template_index = tmpl_idx;
                page_pattern_instances.push_back(instance);
                i += run_len;
                continue;
            }
            // Treat irregular segment as a run of length 1 with its own template
            const auto& irr = segments[i];
            PatternTemplate irr_tmpl = {0, 0, 0, irr.num_elements};  // stride 0, size = num_elements
            uint32_t irr_tmpl_idx;
            auto it = template_to_index.find(irr_tmpl);
            if (it == template_to_index.end()) {
                irr_tmpl_idx = pattern_templates.size();
                pattern_templates.push_back(irr_tmpl);
                template_to_index[irr_tmpl] = irr_tmpl_idx;
            } else {
                irr_tmpl_idx = it->second;
            }
            PagePatternInstance irr_instance = {
                output_page_idx, irr.input_page_index, irr.input_page_offset, irr.output_page_offset, 1, irr_tmpl_idx};
            page_pattern_instances.push_back(irr_instance);
            ++i;
        }
    }

    // Compress page pattern instances into runs
    std::vector<PagePatternRun> page_pattern_runs = compress_page_pattern_instances(page_pattern_instances);

    return {pattern_templates, page_pattern_instances, page_pattern_runs};
}

inline GlobalCompressedReshapeMap compute_reshape_map(
    const uint32_t num_input_pages,
    const uint32_t num_output_pages,
    const Shape& input_shape,
    const Shape& output_shape,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape) {
    Dims tile_dims_input(input_shape, tile_shape), tile_dims_output(output_shape, tile_shape);

    std::vector<std::vector<SegmentMapData>> mapping_vector;
    mapping_vector.reserve(num_output_pages);

    for (uint32_t output_page_idx = 0; output_page_idx < num_output_pages; ++output_page_idx) {
        mapping_vector.emplace_back(reshape_map_output_page(
            output_page_idx, input_shape, output_shape, tile_dims_input, tile_dims_output, tile_shape, face_shape));
    }
    auto compressed_map = compress_mapping_global(mapping_vector);
    return compressed_map;
}

struct ReshapeRTArgsEstimate {
    uint32_t max_reader_args_per_core;
    uint32_t max_writer_args_per_core;
    uint32_t total_cores_used;
    bool exceeds_limit;

    bool can_fit_in_rt_args(uint32_t limit = 341) const {
        return !exceeds_limit && max_reader_args_per_core < limit && max_writer_args_per_core < limit;
    }
};

inline ReshapeRTArgsEstimate estimate_reshape_rt_args(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_shape,
    const ttnn::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& memory_config) {
    ReshapeRTArgsEstimate estimate{0, 0, 0, false};

    const auto input_shape = input_tensor.logical_shape();
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    const uint32_t num_input_pages = tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
    const uint32_t num_output_pages = tt::div_up(padded_output_shape.volume(), tile_shape[0] * tile_shape[1]);

    auto compressed_map = reshape::detail::compute_reshape_map(
        num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape);

    auto device = input_tensor.device();
    const auto grid = device->compute_with_storage_grid_size();

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_output_pages);

    estimate.total_cores_used = num_cores;

    uint32_t page_idx_start = 0;
    uint32_t max_reader_args = 0, max_writer_args = 0;

    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            increment = num_tiles_per_core_group_2;
        } else {
            continue;
        }

        uint32_t page_idx_end = page_idx_start + increment;

        size_t num_short_runs = 0, num_long_runs = 0;
        std::set<uint32_t> used_template_indices;

        for (const auto& run : compressed_map.page_pattern_runs) {
            if (run.output_page_index_end < page_idx_start || run.output_page_index_start >= page_idx_end) {
                continue;
            }
            used_template_indices.insert(run.pattern_template_index);
            if (run.run_length == 1) {
                num_short_runs++;
            } else {
                num_long_runs++;
            }
        }

        uint32_t base_args = 4;  // num_templates, num_short_runs, num_long_runs, buffer_addr
        uint32_t template_args = used_template_indices.size() * 4;  // 4 args per template
        uint32_t short_run_args = num_short_runs * 3;
        uint32_t long_run_args = num_long_runs * 5;

        uint32_t core_reader_args = base_args + template_args + short_run_args + long_run_args;
        uint32_t core_writer_args = core_reader_args;  // Writer uses same args structure

        max_reader_args = std::max(max_reader_args, core_reader_args);
        max_writer_args = std::max(max_writer_args, core_writer_args);

        page_idx_start += increment;
    }

    estimate.max_reader_args_per_core = max_reader_args;
    estimate.max_writer_args_per_core = max_writer_args;

    // Check if we exceed limits
    estimate.exceeds_limit = (estimate.max_reader_args_per_core >= 341) || (estimate.max_writer_args_per_core >= 341);

    return estimate;
}
}  // namespace detail
}  // namespace ttnn::operations::data_movement::reshape
