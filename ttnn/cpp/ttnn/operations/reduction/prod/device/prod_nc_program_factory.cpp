// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <string>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor ProdNcDeviceOperation::ProdNcProgramFactory::create_descriptor(
    const ProdNcParams& operation_attributes, const ProdNcInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const int64_t dim = operation_attributes.dim;

    TT_FATAL(dim == 0 || dim == 1, "Dimension ({}) must be either 0 or 1", dim);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t single_tile_size = tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_hw = input.tensor_spec().tile().get_tile_hw();

    [[maybe_unused]] const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto Ht = input_shape[2] / tile_height;
    const auto Wt = input_shape[3] / tile_width;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={})", Ht, Wt);

    const auto HtWt = Ht * Wt;
    const auto CHtWt = C * Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];
    const auto input_tile_offset = (dim == 0) ? (CHtWt) : (HtWt);
    const auto num_output_tiles = output.physical_volume() / tile_hw;

    log_debug(tt::LogOp, "N {} C {} Ht {} Wt {}", N, C, Ht, Wt);
    log_debug(
        tt::LogOp,
        "dim {} num_reduce_input_tile {} input_tile_offset {}, num_output_tiles {}",
        dim,
        num_reduce_input_tile,
        input_tile_offset,
        num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    TT_FATAL(num_cores_y != 0, "Compute grid y-dimension must be non-zero");

    const uint32_t in0_t = 2;        // input
    const uint32_t in1_t = 1;        // zero
    const uint32_t intermed0_t = 1;  // accumulated sum
    const uint32_t out0_t = 2;       // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),  // input
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),  // zero
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed0_t * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),  // accumulated sum
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),  // output
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(dim)};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_3;
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(cb_id_out)};
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp";
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = compute_args_group_1;
    compute_desc_1.config = ComputeConfigDescriptor{
        .dst_full_sync_en = false,
    };

    std::optional<KernelDescriptor> compute_desc_2;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        KernelDescriptor cd2;
        cd2.kernel_source = "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp";
        cd2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd2.core_ranges = core_group_2;
        cd2.compile_time_args = compute_args_group_2;
        cd2.config = ComputeConfigDescriptor{
            .dst_full_sync_en = false,
        };
        compute_desc_2 = std::move(cd2);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {input.buffer(),
             num_reduce_input_tile,
             num_tiles_per_core,
             input_tile_offset,
             tile_offset,
             HtWt,
             CHtWt,
             static_cast<uint32_t>(dim)});

        writer_desc.emplace_runtime_args(
            core,
            {output.buffer(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(ttnn::operations::is_dram(output))});

        if (core_group_1.contains(core)) {
            compute_desc_1.emplace_runtime_args(core, {num_reduce_input_tile, num_tiles_per_core});
        } else if (core_group_2.contains(core)) {
            TT_FATAL(compute_desc_2.has_value(), "compute_desc_2 needs to have a value");
            compute_desc_2->emplace_runtime_args(core, {num_reduce_input_tile, num_tiles_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (compute_desc_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim
