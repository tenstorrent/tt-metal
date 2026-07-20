// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_tilize_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

ProgramDescriptor DispatchTilizeProgramFactory::create_descriptor(
    const DispatchTilizeParams& operation_attributes,
    const DispatchTilizeInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    // Stage 1 tilizes the full buffer; the routing metadata in operation_attributes /
    // tensor_args drives the region-aware skip added in stage 2.
    (void)operation_attributes;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_block = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_block);

    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet available_grid(default_cores);

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = ntiles_per_block * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = ntiles_per_block * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // reader
    uint32_t page_size = src0_buffer->page_size();
    uint32_t aligned_page_size = src0_buffer->aligned_page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;

    std::vector<uint32_t> reader_ct_args = {
        aligned_page_size, num_pages_in_row, size_of_valid_data_in_last_page_in_row};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_tilize/device/kernels/"
        "dispatch_tilize_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // writer
    std::vector<uint32_t> writer_ct_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_tilize/device/kernels/"
        "dispatch_tilize_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // compute
    std::vector<uint32_t> compute_args = {nblocks_per_core, ntiles_per_block};
    std::vector<uint32_t> compute_args_cliff = {nblocks_per_core_cliff, ntiles_per_block};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    const std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_tilize/device/kernels/"
        "dispatch_tilize_compute.cpp";

    std::optional<KernelDescriptor> compute_desc;
    if (!core_range.ranges().empty()) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range;
        cd.compile_time_args = std::move(compute_args);
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_llk_acc,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        compute_desc = std::move(cd);
    }

    std::optional<KernelDescriptor> compute_cliff_desc;
    if (!core_range_cliff.empty()) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range_cliff;
        cd.compile_time_args = std::move(compute_args_cliff);
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_llk_acc,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        compute_cliff_desc = std::move(cd);
    }

    bool has_cliff = !core_range_cliff.empty();
    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             nblocks_per_core * TILE_HEIGHT,
             page_size,
             ntiles_per_block,
             page_size,
             std::uint32_t{1},
             std::uint32_t{0},
             std::uint32_t{0},
             page_start_id});
        writer_desc.emplace_runtime_args(core, {dst_buffer, ntiles_per_block * nblocks_per_core, tile_start_id});
        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             nblocks_per_core_cliff * TILE_HEIGHT,
             page_size,
             ntiles_per_block,
             page_size,
             std::uint32_t{1},
             std::uint32_t{0},
             std::uint32_t{0},
             page_start_id});
        writer_desc.emplace_runtime_args(core, {dst_buffer, ntiles_per_block * nblocks_per_core_cliff, tile_start_id});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (compute_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc));
    }
    if (compute_cliff_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_cliff_desc));
    }

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
