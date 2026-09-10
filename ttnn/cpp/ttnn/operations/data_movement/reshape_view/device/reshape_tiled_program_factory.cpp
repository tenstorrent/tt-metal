// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_tiled_program_factory.hpp"

#include <cmath>
#include <numeric>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

namespace ttnn::prim {

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
    const Shape& /*shape*/,
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
        start_h(in_start_h), start_w(in_start_w), tile_end_h(in_end_h - 1), tile_end_w(in_end_w - 1) {};

    bool next() {
        if (first) {
            first = false;
            return true;
        }
        if (tile_idx_w < tile_end_w) {
            ++tile_idx_w;
            return true;
        }
        if (tile_idx_h < tile_end_h) {
            tile_idx_w = 0;
            ++tile_idx_h;
            return true;
        }
        return false;
    }

    auto operator*() { return std::make_tuple(this->h(), this->w()); }

    uint32_t size() const { return (tile_end_h + 1) * (tile_end_w + 1); }

protected:
    const uint32_t& start_h;
    const uint32_t& start_w;
    uint32_t tile_idx_h{0};
    uint32_t tile_idx_w{0};
    const uint32_t tile_end_h;
    const uint32_t tile_end_w;
    bool first{true};

    uint32_t h() const { return start_h + tile_idx_h; }

    uint32_t w() const { return start_w + tile_idx_w; }
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

        if (map_data.contains(page_idx_i)) {
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

Tensor compute_reshape_mapping_host_tensor(
    const uint32_t /*num_input_pages*/,
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
        auto* map_ptr = reinterpret_cast<SegmentMapData*>(&(*it));
        std::copy(v.begin(), v.end(), map_ptr);

        it += max_input_segments * SegmentMapData::size;
    }

    const std::array<uint32_t, 2> mapping_shape_vector = {num_output_pages, SegmentMapData::size * max_input_segments};
    const Shape mapping_shape(mapping_shape_vector);
    const tt::tt_metal::TensorLayout mapping_layout(
        tt::tt_metal::convert_to_data_type<decltype(flat_mapping_vector)::value_type>(),
        ttnn::ROW_MAJOR_LAYOUT,
        MemoryConfig());

    return Tensor::from_vector(flat_mapping_vector, tt::tt_metal::TensorSpec(mapping_shape, mapping_layout));
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
// operating and pushes it on to the dataflow buffer. The reader kernel loops over all of the data segments represented
// by the map and loads the specified input pages, avoiding redundant loads of pages for segments that come from the
// same input page, and pushes them to the dataflow buffer.

// The writer kernel pops mapping pages off the dataflow buffer, corresponding to the current page. It loops through
// the input tensor pages specified by the map and, as necessary, pops input pages off the dataflow buffer, again
// accounting for consecutive segments that come from the same input page. Using the offsets and size supplied by the
// map, the writer copies the segment from the input page to a scratch page stored in L1. When all segments are written,
// the scratch page is copied to its output destination.

ttnn::device_operation::ProgramArtifacts ReshapeViewTiledProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;
    const auto& input_mt = input_tensor.mesh_tensor();
    const auto& output_mt = output_tensor.mesh_tensor();

    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_tensor.logical_shape();

    TT_FATAL(input_shape.volume() == output_shape.volume(), "Requested shapes are not of equal volume");

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    TT_ASSERT(input_shape.size() == 3 && output_shape.size() == 3, "Kernel designed for rank 3 tensors");

    tt::tt_metal::distributed::MeshDevice* device = input_tensor.device();

    const uint32_t num_input_pages = tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
    const uint32_t num_output_pages = tt::div_up(output_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);

    // Host-compute and upload the input->output page-mapping tensor.
    Tensor mapping_tensor = detail::compute_reshape_mapping_host_tensor(
                                num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape)
                                .to_device(device);

    const uint32_t mapping_page_size = mapping_tensor.logical_shape()[-1];
    const auto mapping_dataformat = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    const uint32_t mapping_page_size_bytes = mapping_page_size * mapping_tensor.element_size();

    // Op-owned tensor: move the owning MeshTensor out of the built Tensor into the artifact so its
    // device allocation outlives the cached program (~Tensor would otherwise force-deallocate it).
    // The mapping depends only on the hashed input/output shapes, so on a cache hit the cached
    // mapping is always valid; operation_attributes.recreate_mapping_tensor stays ignored (it is
    // excluded from the program hash).
    std::vector<tt::tt_metal::MeshTensor> op_owned;
    op_owned.reserve(1);
    op_owned.push_back(mapping_tensor.device_storage().release_mesh_tensor());
    const auto& mapping_owned = op_owned.back();

    const auto grid = device->compute_with_storage_grid_size();

    // PCC fails when this is greater than 1. TODO figure out why.
    constexpr auto reader_dfb_len = 1;

    // input tile DFB / output tile DFB format metadata
    const auto input_dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto input_tile_size_bytes = tt::tile_size(input_dfb_data_format);
    // TODO assert output tile size and data format same as input
    const auto output_dfb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const auto output_tile_size_bytes = tt::tile_size(output_dfb_data_format);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            operation_attributes.sub_core_grid.has_value()
                ? split_work_to_cores(operation_attributes.sub_core_grid.value(), num_output_pages)
                : split_work_to_cores(grid, num_output_pages);

    TT_ASSERT(num_cores <= num_output_pages);

    const uint32_t max_map_entries = mapping_page_size / detail::SegmentMapData::size;

    // Resource names.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName MAPPING{"mapping"};  // reader-produces / writer-consumes mapping-page FIFO
    const DFBSpecName IN_TILES{"in_tiles"};  // reader-produces / writer-consumes input-tile FIFO
                                             // (distinct from tensor::input, the input tensor)
    const DFBSpecName WORKING{"working"};  // writer-only L1 scratch page (self-loop)
    const TensorParamName INPUT_T{"input"};
    const TensorParamName MAPPING_T{"map"};  // the op-owned page-mapping tensor
    const TensorParamName OUTPUT_T{"output"};

    ProgramSpec spec;
    spec.name = "reshape_view_tiled";

    // DFBs (placement derived from bindings; no core_ranges). The mapping metadata DFB stages
    // mapping pages on the fly; the input DFB stages input tiles; the output/working DFB is an L1
    // scratch page the writer both fills and drains (self-loop).
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = MAPPING,
            .entry_size = mapping_page_size_bytes,
            .num_entries = reader_dfb_len,
            .data_format_metadata = mapping_dataformat,
        },
        DataflowBufferSpec{
            .unique_id = IN_TILES,
            .entry_size = input_tile_size_bytes,
            .num_entries = reader_dfb_len,
            .data_format_metadata = input_dfb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = WORKING,
            .entry_size = output_tile_size_bytes,
            .num_entries = 1,
            .data_format_metadata = output_dfb_data_format,
        },
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = MAPPING, .accessor_name = "mapping", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = IN_TILES, .accessor_name = "in_tiles", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = MAPPING_T, .accessor_name = "map"},
            },
        .compile_time_args =
            {
                {"Max_Map_Size_Bytes", mapping_page_size_bytes},
                {"Tile_size_bytes", input_tile_size_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_page_idx", "end_output_page_idx"}},
        .hw_config = create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/writer_reshape_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = MAPPING, .accessor_name = "mapping", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = IN_TILES, .accessor_name = "in_tiles", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = WORKING, .accessor_name = "working", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = WORKING, .accessor_name = "working", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_T, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"Tile_size_bytes", input_tile_size_bytes},
                {"Max_Map_Entries", max_map_entries},
                {"element_sz_bytes", tt::datum_size(output_dfb_data_format)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_page", "end_output_page"}},
        .hw_config = create_writer_datamovement_config(),
    };

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT_T, .spec = input_mt.tensor_spec()},
        TensorParameter{.unique_id = MAPPING_T, .spec = mapping_owned.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT_T, .spec = output_mt.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}};

    // Per-node runtime args. The run-args table is keyed name-first (name -> node -> value);
    // AddRuntimeArgsForNode builds that from the per-core loop below.
    KernelRunArgs reader_kra{.kernel = READER};
    KernelRunArgs writer_kra{.kernel = WRITER};

    uint32_t page_idx_start = 0, page_idx_end = 0;
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

        AddRuntimeArgsForNode(
            reader_kra.runtime_arg_values,
            c,
            {{"start_output_page_idx", page_idx_start}, {"end_output_page_idx", page_idx_end}});
        AddRuntimeArgsForNode(
            writer_kra.runtime_arg_values,
            c,
            {{"start_output_page", page_idx_start}, {"end_output_page", page_idx_end}});

        page_idx_start += increment;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_kra), std::move(writer_kra)};
    run_args.tensor_args = {
        {INPUT_T, TensorArgument{input_mt}},
        {MAPPING_T, TensorArgument{mapping_owned}},
        {OUTPUT_T, TensorArgument{output_mt}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned),
    };
}

}  // namespace ttnn::prim
