// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/moe/device/moe_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <cmath>

using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor MoeProgramFactory::create_descriptor(
    const MoeParams& operation_attributes, const MoeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    const auto& expert_mask_tensor = tensor_args.expert_mask;
    const auto& topk_mask_tensor = tensor_args.topk_mask;

    const auto k = operation_attributes.k;

    CoreRange core({0, 0}, {0, 0});
    CoreRangeSet core_ranges{core};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat topk_mask_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(topk_mask_tensor.dtype());
    tt::DataFormat expert_mask_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(expert_mask_tensor.dtype());
    tt::DataFormat out_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;
    tt::DataFormat value_cb_data_format = tt::DataFormat::Float16_b;

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t topk_mask_tile_size = tile_size(topk_mask_cb_data_format);
    uint32_t expert_mask_tile_size = tile_size(expert_mask_cb_data_format);
    uint32_t out_tile_size = tile_size(out_cb_data_format);
    uint32_t scalar_tile_size = tile_size(scalar_df);
    uint32_t index_tile_size = tile_size(index_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);

    auto* input_buffer = input_tensor.buffer();
    auto* topk_mask_buffer = topk_mask_tensor.buffer();
    auto* expert_mask_buffer = expert_mask_tensor.buffer();
    auto* out_buffer = output_tensor.buffer();

    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_hw = input_tensor.tensor_spec().tile().get_tile_hw();
    uint32_t num_out_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t scale_tiles = 1;

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    uint32_t Wt = input_shape[3] / tile_width;
    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    ProgramDescriptor desc;

    // INPUT CBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    uint32_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * input_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_tile_size,
        }}},
    });

    uint32_t expert_mask_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * expert_mask_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(expert_mask_cb_index),
            .data_format = expert_mask_cb_data_format,
            .page_size = expert_mask_tile_size,
        }}},
    });

    uint32_t topk_mask_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * topk_mask_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(topk_mask_cb_index),
            .data_format = topk_mask_cb_data_format,
            .page_size = topk_mask_tile_size,
        }}},
    });

    // identity scale input
    uint32_t scale_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scale_cb_index),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // TOP K CBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * index_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * value_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_transposed_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_tile_size,
        }}},
    });

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * index_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_transposed_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // topk values
    uint32_t values_cb_index = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * value_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(values_cb_index),
            .data_format = value_cb_data_format,
            .page_size = value_tile_size,
        }}},
    });

    // topk indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * index_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_ind_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    uint32_t cb_cur_max_index = tt::CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * out_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_cur_max_index),
            .data_format = out_cb_data_format,
            .page_size = out_tile_size,
        }}},
    });

    uint32_t cb_cur_sum_index = tt::CBIndex::c_10;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * out_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_cur_sum_index),
            .data_format = out_cb_data_format,
            .page_size = out_tile_size,
        }}},
    });

    // OUTPUT CBs
    uint32_t out_cb_index = tt::CBIndex::c_11;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * out_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_index),
            .data_format = out_cb_data_format,
            .page_size = out_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index, index_cb_index, topk_mask_cb_index, expert_mask_cb_index, Ht, Wt, k};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(topk_mask_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_mask_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_runtime_args(
        core.start_coord,
        {
            input_buffer,
            topk_mask_buffer,
            expert_mask_buffer,
        });

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index, Ht, k};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(
        core.start_coord,
        {
            out_buffer,
        });

    std::vector<uint32_t> compute_args = {
        input_cb_index,
        topk_mask_cb_index,
        expert_mask_cb_index,
        scale_cb_index,
        index_cb_index,
        input_transposed_cb_index,
        index_transposed_cb_index,
        values_cb_index,
        output_ind_cb_index,
        out_cb_index,
        Ht,
        Wt,
        k,
        static_cast<uint32_t>(std::log2(k)),
        static_cast<uint32_t>(std::log2(Wt)),
        cb_cur_max_index,
        cb_cur_sum_index,
        tile_width};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_ranges;
    compute_desc.compile_time_args = compute_args;
    compute_desc.config = ComputeConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
