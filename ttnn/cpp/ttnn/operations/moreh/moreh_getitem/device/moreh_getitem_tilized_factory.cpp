// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "moreh_getitem_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
struct IndexInfo {
    bool is_defined{};
    tt::tt_metal::TensorAccessorArgs args;
    tt::tt_metal::Buffer* buffer{};
    uint32_t unit_size{};
};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::moreh::moreh_getitem {

tt::tt_metal::ProgramDescriptor MorehGetItemOperation::MorehGetItemTilizedFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& input = tensor_args.input;
    const auto& index_tensors = tensor_args.index_tensors;
    const auto& output = output_tensor;
    auto index_dims = operation_attributes.index_dims;
    auto TILE_HEIGHT = constants::TILE_HEIGHT;
    auto TILE_WIDTH = constants::TILE_WIDTH;
    auto* device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange allCores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    auto core_range = allCores;

    auto input_shape = input.padded_shape();
    auto input_shape_without_padding = input.logical_shape();
    auto output_shape = output.padded_shape();
    auto output_shape_without_padding = output.logical_shape();

    std::array<uint32_t, 5> new_input_shape{};
    std::array<uint32_t, 5> new_output_shape{};
    std::array<uint32_t, 5> new_input_padded_shape{};
    std::array<uint32_t, 5> new_output_padded_shape{};

    new_input_shape.fill(1);
    new_input_padded_shape.fill(1);
    auto input_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < input_shape.rank(); index++) {
        new_input_shape[index + input_dim_offset] = input_shape_without_padding[index];
        new_input_padded_shape[index + input_dim_offset] = input_shape[index];
    }

    new_output_shape.fill(1);
    new_output_padded_shape.fill(1);
    auto output_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < output_shape.rank(); index++) {
        new_output_shape[index + output_dim_offset] = output_shape_without_padding[index];
        new_output_padded_shape[index + output_dim_offset] = output_shape[index];
    }

    ttnn::Shape input_5d_shape(new_input_padded_shape);
    ttnn::Shape output_5d_shape(new_output_padded_shape);

    bool is_w_index_exist = false;
    for (auto dim : index_dims) {
        if (dim + input_dim_offset == 4) {
            is_w_index_exist = true;
        }
    }

    ttnn::Shape input_5d_shape_without_padding(new_input_shape);
    ttnn::Shape output_5d_shape_without_padding(new_output_shape);

    auto index_layout = index_tensors.front().layout();
    bool is_row_major_index = (index_layout == Layout::ROW_MAJOR);

    if (is_w_index_exist) {
        // compute index info
        IndexInfo index_info[5] = {{false}};

        for (uint32_t i = 0; i < index_tensors.size(); i++) {
            auto dim = index_dims[i] + input_dim_offset;
            const auto& index = index_tensors[i];

            index_info[dim].is_defined = true;
            index_info[dim].buffer = index.buffer();
            index_info[dim].args = tt::tt_metal::TensorAccessorArgs(index.buffer());
            index_info[dim].unit_size = index.element_size();
        }

        uint32_t index_size = index_tensors[0].logical_shape()[-1];

        uint32_t input_unit_size = input.element_size();
        uint32_t output_unit_size = output.element_size();

        uint32_t alignment_size = 32;
        uint32_t num_elements_per_alignment = alignment_size / output_unit_size;
        uint32_t num_units =
            output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
            output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
            ((output_5d_shape_without_padding[4] + num_elements_per_alignment - 1) / num_elements_per_alignment);

        uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
                split_work_to_cores_wt_core_range(core_range, num_units);

        ProgramDescriptor desc;

        // create circular buffers
        auto src_cb_data_format = datatype_to_dataformat_converter(input.dtype());
        auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].dtype());
        auto output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

        auto src_cb_index = CBIndex::c_0;
        auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
        desc.cbs.push_back(CBDescriptor{
            .total_size = rounded_input_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src_cb_index),
                .data_format = src_cb_data_format,
                .page_size = rounded_input_page_size,
            }}},
        });

        for (uint32_t dim = 0; dim < 5; dim++) {
            if (!index_info[dim].is_defined) {
                continue;
            }

            auto src1_cb_index = CBIndex::c_1 + dim;
            auto index_page_size = 1024 * 4;
            desc.cbs.push_back(CBDescriptor{
                .total_size = static_cast<uint32_t>(index_page_size),
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(src1_cb_index),
                    .data_format = index_cb_data_format,
                    .page_size = static_cast<uint32_t>(index_page_size),
                }}},
            });
        }

        auto out_cb0_index = CBIndex::c_16;
        auto rounded_output_page_size = round_up_to_mul32(output_unit_size);
        desc.cbs.push_back(CBDescriptor{
            .total_size = rounded_output_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb0_index),
                .data_format = output_cb_data_format,
                .page_size = rounded_output_page_size,
            }}},
        });

        auto out_cb1_index = CBIndex::c_17;
        desc.cbs.push_back(CBDescriptor{
            .total_size = rounded_output_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb1_index),
                .data_format = output_cb_data_format,
                .page_size = rounded_output_page_size,
            }}},
        });

        // create read/write kernel
        KernelDescriptor::Defines reader_defines;
        KernelDescriptor::Defines writer_defines;

        if (is_row_major_index) {
            reader_defines.emplace_back("ROW_MAJOR_INDEX", "1");
        } else {
            reader_defines.emplace_back("TILIZE_INDEX", "1");
        }

        KernelDescriptor::CompileTimeArgs reader_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
        for (const auto& info : index_info) {
            info.args.append_to(reader_compile_time_args);
        }

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "reader_moreh_getitem_tilize_w.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.defines = std::move(reader_defines);
        reader_desc.config = ReaderConfigDescriptor{};

        KernelDescriptor::CompileTimeArgs writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "writer_moreh_getitem_tilize_w.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_compile_time_args);
        writer_desc.defines = std::move(writer_defines);
        writer_desc.config = WriterConfigDescriptor{};

        uint32_t face_width = 16;
        uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
        uint32_t num_alignment_width = div_up(output_5d_shape_without_padding[4], num_elements_per_alignment);
        uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

        uint32_t input_num_tile_c = input_5d_shape[1];
        uint32_t input_num_tile_d = input_5d_shape[2];
        uint32_t input_num_tile_height = input_5d_shape[3] / TILE_HEIGHT;
        uint32_t input_num_tile_width = input_5d_shape[4] / TILE_WIDTH;
        uint32_t input_noc_id_stride_h = input_num_tile_width;
        uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
        uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
        uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

        uint32_t output_num_tile_c = output_5d_shape[1];
        uint32_t output_num_tile_d = output_5d_shape[2];
        uint32_t output_num_tile_height = output_5d_shape[3] / TILE_HEIGHT;
        uint32_t output_num_tile_width = output_5d_shape[4] / TILE_WIDTH;

        uint32_t output_noc_id_stride_h = output_num_tile_width;
        uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
        uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
        uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

        uint32_t input_stick_idx_stride_w = 1;
        uint32_t input_stick_idx_stride_h = input_num_stick_width;
        uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape_without_padding[3];
        uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape_without_padding[2];
        uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape_without_padding[1];

        // Set Runtime Args
        auto core_x_offset = core_range.start_coord.x;
        auto core_y_offset = core_range.start_coord.y;

        uint32_t g1_numcores = core_group_1.num_cores();

        uint32_t start_id = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
            uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

            reader_desc.emplace_runtime_args(
                core,
                {
                    // buffers
                    input.buffer(),
                    index_info[0].buffer,
                    index_info[1].buffer,
                    index_info[2].buffer,
                    index_info[3].buffer,
                    index_info[4].buffer,

                    // input
                    input_stick_idx_stride_n,
                    input_stick_idx_stride_c,
                    input_stick_idx_stride_d,
                    input_stick_idx_stride_h,
                    input_stick_idx_stride_w,
                    input_5d_shape_without_padding[1],
                    input_5d_shape_without_padding[2],
                    input_5d_shape_without_padding[3],
                    input_num_stick_width,
                    input_noc_id_stride_n,
                    input_noc_id_stride_c,
                    input_noc_id_stride_d,
                    input_noc_id_stride_h,

                    input_5d_shape_without_padding[0],
                    input_5d_shape_without_padding[1],
                    input_5d_shape_without_padding[2],
                    input_5d_shape_without_padding[3],
                    input_5d_shape_without_padding[4],

                    // index
                    index_info[0].is_defined,
                    index_info[1].is_defined,
                    index_info[2].is_defined,
                    index_info[3].is_defined,
                    index_info[4].is_defined,
                    index_info[0].unit_size,
                    index_info[1].unit_size,
                    index_info[2].unit_size,
                    index_info[3].unit_size,
                    index_info[4].unit_size,
                    index_size,

                    // output
                    output_5d_shape_without_padding[0],
                    output_5d_shape_without_padding[1],
                    output_5d_shape_without_padding[2],
                    output_5d_shape_without_padding[3],
                    output_5d_shape_without_padding[4],
                    output_num_stick_width,

                    // etc
                    start_id,
                    num_units_per_core,
                    input.element_size(),
                    num_elements_per_alignment,
                    num_alignment_width,
                });

            writer_desc.emplace_runtime_args(
                core,
                {
                    // buffers
                    output.buffer(),

                    // output
                    output_5d_shape_without_padding[1],
                    output_5d_shape_without_padding[2],
                    output_5d_shape_without_padding[3],
                    output_5d_shape_without_padding[4],
                    output_noc_id_stride_n,
                    output_noc_id_stride_c,
                    output_noc_id_stride_d,
                    output_noc_id_stride_h,
                    output_num_stick_width,

                    // etc
                    start_id,
                    num_units_per_core,
                    output_unit_size,
                    output.element_size(),
                    num_elements_per_alignment,
                    num_alignment_width,
                });

            start_id += num_units_per_core;
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));

        return desc;

    }  // compute index info

    IndexInfo index_info[5] = {{false}};

    for (uint32_t i = 0; i < index_tensors.size(); i++) {
        auto dim = index_dims[i] + input_dim_offset;
        const auto& index = index_tensors[i];

        index_info[dim].is_defined = true;
        index_info[dim].buffer = index_tensors[i].buffer();
        index_info[dim].args = tt::tt_metal::TensorAccessorArgs(index_tensors[i].buffer());
        index_info[dim].unit_size = index.padded_shape()[-1] * index.element_size();
    }
    uint32_t index_size = index_tensors[0].logical_shape()[-1];

    uint32_t input_unit_size = 16 * input.element_size();
    uint32_t output_unit_size = 16 * output.element_size();

    uint32_t num_units = output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
                         output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
                         ((output_5d_shape_without_padding[4] + 15) / 16);

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_units);

    ProgramDescriptor desc;

    // create circular buffers
    auto src_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].dtype());
    auto output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_cb_index = CBIndex::c_0;
    auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
    desc.cbs.push_back(CBDescriptor{
        .total_size = rounded_input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb_index),
            .data_format = src_cb_data_format,
            .page_size = rounded_input_page_size,
        }}},
    });

    for (uint32_t dim = 0; dim < 5; dim++) {
        if (!index_info[dim].is_defined) {
            continue;
        }

        auto src1_cb_index = CBIndex::c_1 + dim;
        auto index_page_size = 1024 * 4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = static_cast<uint32_t>(index_page_size),
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = index_cb_data_format,
                .page_size = static_cast<uint32_t>(index_page_size),
            }}},
        });
    }

    auto out_cb_index = CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rounded_input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_index),
            .data_format = output_cb_data_format,
            .page_size = rounded_input_page_size,
        }}},
    });

    // create read/write kernel
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (is_row_major_index) {
        reader_defines.emplace_back("ROW_MAJOR_INDEX", "1");
    } else {
        reader_defines.emplace_back("TILIZE_INDEX", "1");
    }

    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    for (const auto& info : index_info) {
        info.args.append_to(reader_compile_time_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
        "reader_moreh_getitem_tilize.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
        "writer_moreh_getitem_tilize.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t face_width = 16;
    uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
    uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

    uint32_t input_num_tile_c = input_5d_shape[1];
    uint32_t input_num_tile_d = input_5d_shape[2];
    uint32_t input_num_tile_height = input_5d_shape[3] / TILE_HEIGHT;
    uint32_t input_num_tile_width = input_5d_shape[4] / TILE_WIDTH;
    uint32_t input_noc_id_stride_h = input_num_tile_width;
    uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
    uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
    uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

    uint32_t output_num_tile_c = output_5d_shape[1];
    uint32_t output_num_tile_d = output_5d_shape[2];
    uint32_t output_num_tile_height = output_5d_shape[3] / TILE_HEIGHT;
    uint32_t output_num_tile_width = output_5d_shape[4] / TILE_WIDTH;

    uint32_t output_noc_id_stride_h = output_num_tile_width;
    uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
    uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
    uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

    uint32_t input_stick_idx_stride_w = 1;
    uint32_t input_stick_idx_stride_h = input_num_stick_width;
    uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape_without_padding[3];
    uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape_without_padding[2];
    uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape_without_padding[1];

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    uint32_t g1_numcores = core_group_1.num_cores();

    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        reader_desc.emplace_runtime_args(
            core,
            {
                // buffers
                input.buffer(),
                index_info[0].buffer,
                index_info[1].buffer,
                index_info[2].buffer,
                index_info[3].buffer,
                index_info[4].buffer,

                // input
                input_stick_idx_stride_n,
                input_stick_idx_stride_c,
                input_stick_idx_stride_d,
                input_stick_idx_stride_h,
                input_stick_idx_stride_w,
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_noc_id_stride_n,
                input_noc_id_stride_c,
                input_noc_id_stride_d,
                input_noc_id_stride_h,
                input_num_stick_width,

                input_5d_shape_without_padding[0],
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_5d_shape_without_padding[4],

                // index
                index_info[0].is_defined,
                index_info[1].is_defined,
                index_info[2].is_defined,
                index_info[3].is_defined,
                index_info[4].is_defined,
                index_info[0].unit_size,
                index_info[1].unit_size,
                index_info[2].unit_size,
                index_info[3].unit_size,
                index_info[4].unit_size,
                index_size,

                // output
                output_5d_shape[0],
                output_5d_shape[1],
                output_5d_shape[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                input_unit_size,
                input.element_size(),
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                // buffers
                output.buffer(),

                // output
                output_5d_shape_without_padding[1],
                output_5d_shape_without_padding[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_noc_id_stride_n,
                output_noc_id_stride_c,
                output_noc_id_stride_d,
                output_noc_id_stride_h,
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                output_unit_size,
                output.element_size(),
            });

        start_id += num_units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_getitem
