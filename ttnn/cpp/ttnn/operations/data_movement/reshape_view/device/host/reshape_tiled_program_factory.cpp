// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"
#include "reshape_program_factory.hpp"

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

std::tuple<uint32_t, uint32_t, uint32_t> page_index_to_tensor_idxs(
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

uint32_t tensor_idxs_to_page_idx(
    const uint32_t c,
    const uint32_t h,
    const uint32_t w,
    const Shape& shape,
    const std::array<uint32_t, 2>& tile_shape,
    const Dims& tile_dims) {
    return (c * tile_dims.c) + (h / tile_shape[0] * tile_dims.w) + (w / tile_shape[1]);
}

uint32_t tensor_idxs_to_faced_tile_offset(
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

std::vector<SegmentMapData> reshape_map_output_page(
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
    int32_t input_offset_stride;
    int32_t output_offset_stride;
    uint32_t num_elements;

    bool operator==(const PatternTemplate& other) const {
        return input_offset_stride == other.input_offset_stride && output_offset_stride == other.output_offset_stride &&
               num_elements == other.num_elements;
    }
    bool operator<(const PatternTemplate& other) const {
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
    std::vector<SegmentMapData> irregular_segments;
};

// Detects the longest stride run starting at 'start'
size_t detect_stride_run(
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
    int32_t input_stride = 0, output_stride = 0;
    if (start + 1 < segs.size() && segs[start + 1].num_elements == base.num_elements &&
        segs[start + 1].input_page_index == base.input_page_index) {
        input_stride =
            static_cast<int32_t>(segs[start + 1].input_page_offset) - static_cast<int32_t>(base.input_page_offset);
        output_stride =
            static_cast<int32_t>(segs[start + 1].output_page_offset) - static_cast<int32_t>(base.output_page_offset);
        len = 2;
        for (size_t i = start + 2; i < segs.size(); ++i) {
            if (segs[i].num_elements == 0) {
                break;
            }
            if (segs[i].input_page_index != base.input_page_index || segs[i].num_elements != base.num_elements) {
                break;
            }
            int32_t curr_input_stride =
                static_cast<int32_t>(segs[i].input_page_offset) - static_cast<int32_t>(segs[i - 1].input_page_offset);
            int32_t curr_output_stride =
                static_cast<int32_t>(segs[i].output_page_offset) - static_cast<int32_t>(segs[i - 1].output_page_offset);
            if (curr_input_stride != input_stride || curr_output_stride != output_stride) {
                break;
            }
            len++;
        }
        tmpl = {input_stride, output_stride, base.num_elements};
        instance = {
            output_page_index,
            base.input_page_index,
            base.input_page_offset,
            base.output_page_offset,
            static_cast<uint32_t>(len),
            0};  // pattern_template_index to be filled later
        return len;
    }
    return 0;
}

std::vector<PagePatternRun> compress_page_pattern_instances(const std::vector<PagePatternInstance>& instances) {
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

GlobalCompressedReshapeMap compress_mapping_global(const std::vector<std::vector<SegmentMapData>>& mapping_vector) {
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
            PatternTemplate irr_tmpl = {0, 0, irr.num_elements};  // stride 0, size = num_elements
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

    // No need for a separate irregular_segments vector anymore
    return {pattern_templates, page_pattern_instances, page_pattern_runs, {}};
}

inline uint32_t pack4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) { return (a << 24) | (b << 16) | (c << 8) | d; }

std::tuple<Tensor, GlobalCompressedReshapeMap> compute_reshape_mapping_host_tensor(
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
    printf("initial mapping\n");
    for (uint32_t output_page_idx = 0; output_page_idx < num_output_pages; ++output_page_idx) {
        printf("Output Page %u:\n", output_page_idx);
        for (const auto& seg : mapping_vector[output_page_idx]) {
            printf(
                "  Segment: InputPage %u, InOffset %u, OutOffset %u, NumElem %u\n",
                seg.input_page_index,
                seg.input_page_offset,
                seg.output_page_offset,
                seg.num_elements);
        }
    }

    printf("general compressed mapping\n");
    auto compressed_map = compress_mapping_global(mapping_vector);
    printf("Check here\n");
    for (const auto& run : compressed_map.page_pattern_runs) {
        printf(
            "Run: out_page_start=%u, out_page_end=%u, run_length=%u, tmpl_idx=%u\n",
            run.output_page_index_start,
            run.output_page_index_end,
            run.run_length,
            run.pattern_template_index);
    }

    std::vector<uint32_t> rt_args;
    for (const auto& run : compressed_map.page_pattern_runs) {
        // Example: pack output_page_index_start, output_page_index_end, input_page_index_start, pattern_template_index
        rt_args.push_back(pack4(
            run.output_page_index_start & 0xFF,
            run.output_page_index_end & 0xFF,
            run.input_page_index_start & 0xFF,
            run.pattern_template_index & 0xFF));

        // Pack input_offset_start, output_offset_start, run_length, input_page_index_stride
        rt_args.push_back(pack4(
            run.input_offset_start & 0xFF,
            run.output_offset_start & 0xFF,
            run.run_length & 0xFF,
            run.input_page_index_stride & 0xFF));

        // Pack input_offset_stride, output_offset_stride, (pad with zeros)
        rt_args.push_back(pack4(run.input_offset_stride & 0xFF, run.output_offset_stride & 0xFF, 0, 0));
    }
    printf("len of rt_args (page pattern runs): %zu\n", rt_args.size());

    printf("Pattern Templates (%zu):\n", compressed_map.pattern_templates.size());
    for (size_t i = 0; i < compressed_map.pattern_templates.size(); ++i) {
        const auto& pt = compressed_map.pattern_templates[i];
        printf(
            "  Template %zu: InOffsetStride %d, OutOffsetStride %d, NumElem %u\n",
            i,
            pt.input_offset_stride,
            pt.output_offset_stride,
            pt.num_elements);
    }
    printf("Page Pattern Instances (%zu):\n", compressed_map.page_pattern_instances.size());
    for (const auto& pi : compressed_map.page_pattern_instances) {
        printf(
            "  Instance: OutPage %u, InPage %u, InOffsetStart %u, OutOffsetStart %u, RunLen %u, TemplateIdx %u\n",
            pi.output_page_index,
            pi.input_page_index,
            pi.input_offset_start,
            pi.output_offset_start,
            pi.run_length,
            pi.pattern_template_index);
    }
    printf("Page Pattern Runs (%zu):\n", compressed_map.page_pattern_runs.size());
    for (const auto& pr : compressed_map.page_pattern_runs) {
        printf(
            "  Run: OutPageStart %u, OutPageEnd %u, InPageStart %u, InOffsetStart %u, OutOffsetStart %u, RunLen %u, "
            "TemplateIdx %u, InPageStride %d, InOffsetStride %d, OutOffsetStride %d\n",
            pr.output_page_index_start,
            pr.output_page_index_end,
            pr.input_page_index_start,
            pr.input_offset_start,
            pr.output_offset_start,
            pr.run_length,
            pr.pattern_template_index,
            pr.input_page_index_stride,
            pr.input_offset_stride,
            pr.output_offset_stride);
    }
    printf("Irregular Segments (%zu):\n", compressed_map.irregular_segments.size());
    for (const auto& seg : compressed_map.irregular_segments) {
        printf(
            "  Segment: InputPage %u, InOffset %u, OutOffset %u, NumElem %u\n",
            seg.input_page_index,
            seg.input_page_offset,
            seg.output_page_offset,
            seg.num_elements);
    }

    // flatten again
    uint32_t max_input_segments =
        std::max_element(mapping_vector.begin(), mapping_vector.end(), [](const auto& a, const auto& b) {
            return a.size() < b.size();
        })->size();

    // Ensure that map data is always aligned
    max_input_segments += max_input_segments % (tt::tt_metal::hal::get_l1_alignment());

    // initialize to 0 because that will be checked by the kernel as a stopping condition
    std::vector<uint32_t> flat_mapping_vector(SegmentMapData::size * num_output_pages * max_input_segments, 0);
    auto it = flat_mapping_vector.begin();
    for (const auto& v : mapping_vector) {
        auto map_ptr = reinterpret_cast<SegmentMapData*>(&(*it));
        std::copy(v.begin(), v.end(), map_ptr);

        it += max_input_segments * SegmentMapData::size;
    }

    const std::array<uint32_t, 2> mapping_shape_vector = {num_output_pages, SegmentMapData::size * max_input_segments};
    const Shape mapping_shape(mapping_shape_vector);
    const tt::tt_metal::TensorLayout mapping_layout(
        tt::tt_metal::convert_to_data_type<decltype(flat_mapping_vector)::value_type>(),
        ttnn::ROW_MAJOR_LAYOUT,
        MemoryConfig());

    return {Tensor::from_vector(flat_mapping_vector, TensorSpec(mapping_shape, mapping_layout)), compressed_map};
}
}  // namespace detail

// Algorithm overview:
// The host computes the mapping between input shape and the output shapes as a series of data segments that are
// contiguous for both input and output tensors. The mapping data is stored as 4 integers per segment: input page index,
// offset of the segment in the input page, offset of the segment in the output page, number of elements in the segment;
// the ordering of the segments in the map are concomitant with the ordering of the output tensor pages. The mapping
// data is stored as an auxiliary integer tensor where each page corresponds to a page of the output tensor.

// The device operation is parallelized over output tensor pages, where each core operates on a range of pages.

// The reader kernel loads the mapping tensor page that corresponds to the current output tensor page on which it is
// operating and pushes it on to the circular buffer. The reader kernel loops over all of the data segments represented
// by the map and loads the specified input pages, avoiding redundant loads of pages for segments that come from the
// same input page, and pushes them to the circular buffer.

// The writer kernel pops mapping pages off the circular buffer, corresponding to the current page. It loops through
// the input tensor pages specified by the map and, as necessary, pops input pages off the circular buffer, again
// accounting for consecutive segments that come from the same input page. Using the offsets and size supplied by the
// map, the reader copies the segment from the input page to a scratch page stored in L1. When all segments are written,
// the scratch page is copied to its output destination.

tt::tt_metal::operation::ProgramWithCallbacks reshape_tiled_program_factory(
    const Tensor& input_tensor, const Tensor& output_tensor) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_tensor.logical_shape();

    TT_FATAL(input_shape.volume() == output_shape.volume(), "Requested shapes are not of equal volume");

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    TT_ASSERT(input_shape.size() == 3 && output_shape.size() == 3, "Kernel designed for rank 3 tensors");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::distributed::MeshDevice* device = input_tensor.device();

    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    TT_ASSERT(input_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_input_pages = tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
    const uint32_t num_output_pages = tt::div_up(output_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);

    auto [mapping_tensor_0, compressed_map] = detail::compute_reshape_mapping_host_tensor(
        num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape);
    auto mapping_tensor = mapping_tensor_0.to_device(device);

    tt::tt_metal::Buffer* mapping_buffer = mapping_tensor.buffer();
    const auto grid = device->compute_with_storage_grid_size();

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // set up CB for mapping metadata

    // PCC fails when this is greater than 1. TODO figure out why.
    constexpr auto reader_cb_len = 1;

    auto mapping_page_size = mapping_tensor.logical_shape()[-1];
    auto mapping_dataformat = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    auto mapping_page_size_bytes = mapping_page_size * mapping_tensor.element_size();
    constexpr auto mapping_cb_idx = tt::CBIndex::c_0;

    const tt::tt_metal::CircularBufferConfig cb_mapping_config =
        tt::tt_metal::CircularBufferConfig(
            mapping_page_size_bytes * reader_cb_len, {{mapping_cb_idx, mapping_dataformat}})
            .set_page_size(mapping_cb_idx, mapping_page_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_mapping_config);

    // set up CB for input tiles
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto input_tile_size_bytes = tt::tile_size(input_cb_data_format);
    constexpr auto input_cb_idx = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            input_tile_size_bytes * reader_cb_len, {{input_cb_idx, input_cb_data_format}})
            .set_page_size(input_cb_idx, input_tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_input_config);

    // TODO assert output tile size and data format same as input
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const auto output_tile_size_bytes = tt::tile_size(output_cb_data_format);
    constexpr auto output_cb_idx = tt::CBIndex::c_2;

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(output_tile_size_bytes, {{output_cb_idx, output_cb_data_format}})
            .set_page_size(output_cb_idx, output_tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_output_pages);

    TT_ASSERT(num_cores <= num_output_pages);

    std::vector<uint32_t> reader_compile_time_args = {
        mapping_page_size_bytes, input_tile_size_bytes, mapping_cb_idx, input_cb_idx};
    tt::tt_metal::TensorAccessorArgs(*mapping_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const uint32_t max_map_entries = mapping_page_size / detail::SegmentMapData::size;
    std::vector<uint32_t> writer_compile_time_args = {
        input_tile_size_bytes,
        max_map_entries,
        tt::datum_size(output_cb_data_format),
        mapping_cb_idx,
        input_cb_idx,
        output_cb_idx};
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/"
        "writer_reshape_tiled.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> utilized_cores;
    /*
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            increment = num_tiles_per_core_group_2;
        } else {
            continue;
        }
        page_idx_end += increment;

        const std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(), mapping_buffer->address(), page_idx_start, page_idx_end};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args = {output_buffer->address(), page_idx_start, page_idx_end};

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
        utilized_cores.push_back(c);
    }
    */

    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            increment = num_tiles_per_core_group_2;
        } else {
            continue;
        }
        page_idx_end += increment;

        // Build per-core rt_args from compressed_map.page_pattern_runs
        std::vector<uint32_t> core_rt_args;

        // Pack pattern templates
        for (const auto& tmpl : compressed_map.pattern_templates) {
            core_rt_args.push_back(tmpl.input_offset_stride);
            core_rt_args.push_back(tmpl.output_offset_stride);
            core_rt_args.push_back(tmpl.num_elements);
        }

        // Pack runs
        size_t num_runs = 0;
        printf("size of compressed_map.page_pattern_runs: %zu\n", compressed_map.page_pattern_runs.size());
        for (const auto& run : compressed_map.page_pattern_runs) {
            if (run.output_page_index_end < page_idx_start || run.output_page_index_start >= page_idx_end) {
                continue;
            }
            uint32_t start = std::max(run.output_page_index_start, page_idx_start);
            uint32_t end = std::min(run.output_page_index_end, page_idx_end - 1);

            core_rt_args.push_back(detail::pack4(
                start & 0xFF, end & 0xFF, run.input_page_index_start & 0xFF, run.pattern_template_index & 0xFF));
            core_rt_args.push_back(run.input_offset_start);   // 32 bits
            core_rt_args.push_back(run.output_offset_start);  // 32 bits
            core_rt_args.push_back(detail::pack4(
                run.run_length & 0xFF,
                run.input_page_index_stride & 0xFF,
                run.input_offset_stride & 0xFF,
                run.output_offset_stride & 0xFF));
            ++num_runs;
        }

        // Build final RT args vector
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.push_back(compressed_map.pattern_templates.size());  // num_templates
        reader_runtime_args.push_back(num_runs);                                 // num_runs
        reader_runtime_args.push_back(input_buffer->address());                  // buffer_addr
        for (auto k : core_rt_args) {
            reader_runtime_args.push_back(k);
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        std::vector<uint32_t> writer_runtime_args;
        writer_runtime_args.push_back(compressed_map.pattern_templates.size());
        writer_runtime_args.push_back(num_runs);
        writer_runtime_args.push_back(output_buffer->address());
        for (auto k : core_rt_args) {
            writer_runtime_args.push_back(k);
        }

        printf("writer rt_args (core %zu,%zu): ", c.x, c.y);
        for (auto k : writer_runtime_args) {
            printf("%u ", k);
        }
        printf("\n");
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
        utilized_cores.push_back(c);
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id,
         writer_kernel_id,
         utilized_cores,
         // capture this to cache the computed mapping tensor. Cheap copy since data is on device.
         mapping_tensor](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) mutable {
            const auto& input_tensor = input_tensors.at(0);
            const auto& output_tensor = output_tensors.at(0);

            const auto& op = *reinterpret_cast<const ttnn::ReshapeDeviceOperation*>(operation);
            if (op.recreate_mapping_tensor) {
                const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
                const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();
                const uint32_t num_input_pages =
                    tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
                const uint32_t num_output_pages =
                    tt::div_up(output_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);

                auto mapping_tuple = detail::compute_reshape_mapping_host_tensor(
                    num_input_pages,
                    num_output_pages,
                    input_tensor.logical_shape(),
                    output_tensor.logical_shape(),
                    tile_shape,
                    face_shape);
                mapping_tensor = std::get<0>(mapping_tuple).to_device(input_tensor.device());
            }

            const auto input_buffer_addr = input_tensor.buffer()->address();
            const auto output_buffer_addr = output_tensor.buffer()->address();

            for (const auto& core : utilized_cores) {
                auto& reader_runtime_args_core = GetRuntimeArgs(program, reader_kernel_id, core);
                reader_runtime_args_core.at(1) = input_buffer_addr;
                if (op.recreate_mapping_tensor) {
                    reader_runtime_args_core.at(2) = mapping_tensor.buffer()->address();
                }

                auto& writer_runtime_args_core = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args_core.at(2) = output_buffer_addr;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

};  // namespace ttnn::operations::data_movement::reshape
