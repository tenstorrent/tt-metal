// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor DeepseekMoEFastReduceNCProgramFactory::create_descriptor(
    const DeepseekMoEFastReduceNCParams& operation_attributes,
    const DeepseekMoEFastReduceNCInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input_tensor.device();
    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const std::vector<ttnn::Tensor>& output_tensors = tensor_return_value;

    const uint32_t reduction_dim = operation_attributes.dim;

    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
    const uint32_t input_tensor_Wt = input_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t slice_Wt = input_tensor_Wt / output_tensors.size();

    uint32_t inner_dims_product = 1;
    for (uint32_t dim = reduction_dim + 1; dim < input_rank; ++dim) {
        inner_dims_product *= input_shape[dim];
    }

    const uint32_t reduction_dim_size = input_shape[reduction_dim];
    const uint32_t inner_num_tiles = inner_dims_product / num_tile_elements;
    const uint32_t reduction_num_tiles = inner_num_tiles * reduction_dim_size;

    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (reduction_dim_size % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_output_tiles = input_tensor.physical_volume() / num_tile_elements / reduction_dim_size;

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const auto compute_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);

    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_page_size = output_tensors.at(0).buffer()->page_size();
    const uint32_t compute_page_size = tt::tile_size(compute_data_format);

    const uint32_t input_tensor_buffer_factor = input_granularity * 2;
    const uint32_t compute_buffer_factor = 1;
    const uint32_t output_tensor_buffer_factor = 2;

    uint32_t compute_input_cb_id_0 = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_tensor_buffer_factor * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_input_cb_id_0),
            .data_format = input_data_format,
            .page_size = input_page_size,
        }}},
    });

    uint32_t compute_input_cb_id_1 = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = compute_buffer_factor * compute_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_input_cb_id_1),
            .data_format = compute_data_format,
            .page_size = compute_page_size,
        }}},
    });

    uint32_t compute_output_cb_id = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tensor_buffer_factor * output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_output_cb_id),
            .data_format = output_data_format,
            .page_size = output_page_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_ct_args = {
        compute_input_cb_id_0,
        compute_input_cb_id_1,
        input_page_size,
        num_cores_to_be_used,
        input_granularity,
        reduction_dim,
        reduction_dim_size,
        inner_num_tiles,
        reduction_num_tiles};
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        compute_output_cb_id, output_page_size, num_cores_to_be_used, input_tensor_Wt, slice_Wt, output_tensors.size()};
    for (uint32_t i = 0; i < output_tensors.size(); ++i) {
        writer_ct_args.push_back(output_page_size);
    }
    for (const ttnn::Tensor& output_tensor : output_tensors) {
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_reader.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_writer.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), operation_attributes.compute_kernel_config);

    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_reduce.cpp";

    std::vector<uint32_t> compute_ct_args_group_1 = {
        num_cols_per_core_group_1,
        reduction_dim_size,
        input_granularity,
        compute_input_cb_id_0,
        compute_input_cb_id_1,
        compute_output_cb_id,
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = std::move(compute_ct_args_group_1);
    compute_desc_1.defines = {compute_defines.begin(), compute_defines.end()};
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    std::optional<KernelDescriptor> compute_desc_2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_ct_args_group_2 = {
            num_cols_per_core_group_2,
            reduction_dim_size,
            input_granularity,
            compute_input_cb_id_0,
            compute_input_cb_id_1,
            compute_output_cb_id,
        };
        compute_desc_2.emplace();
        compute_desc_2->kernel_source = compute_kernel_file;
        compute_desc_2->source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2->core_ranges = core_group_2;
        compute_desc_2->compile_time_args = std::move(compute_ct_args_group_2);
        compute_desc_2->defines = {compute_defines.begin(), compute_defines.end()};
        compute_desc_2->config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indices are increments of num_cores_to_be_used. The total number of
    // units is num_tiles_per_group times num_cores_to_be_used.
    // For example, with 130 output tiles to be processed on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indices is
    // the size of the inner dimensions in tiles (inner_num_tiles). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduction_num_tiles).
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        uint32_t page_id_range_length = num_tiles_per_core * num_cores_to_be_used;

        uint32_t start_tiles_read = i;
        uint32_t start_tiles_to_read = start_tiles_read + page_id_range_length;

        uint32_t start_slice_row_offset = (start_tiles_read / input_tensor_Wt) * slice_Wt;
        uint32_t start_pages_read_in_row = start_tiles_read % input_tensor_Wt;

        reader_desc.emplace_runtime_args(core, {input_tensor.buffer(), start_tiles_read, start_tiles_to_read});

        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(start_tiles_read);
        writer_rt_args.push_back(start_tiles_to_read);
        writer_rt_args.push_back(start_slice_row_offset);
        writer_rt_args.push_back(start_pages_read_in_row);
        for (const ttnn::Tensor& output_tensor : output_tensors) {
            writer_rt_args.push_back(output_tensor.buffer());
        }
        writer_desc.emplace_runtime_args(core, writer_rt_args);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (compute_desc_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
