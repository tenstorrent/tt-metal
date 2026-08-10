// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_scores/device/attn_res_scores_program_factory.hpp"

#include <bit>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kAttnResScoresKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/reduction/attn_res_scores/device/kernels/";

// The sum of squares and the dot that share a candidate, interleaved into one CB.
constexpr uint32_t kStatsPerCandidate = 2;

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResScoresProgramFactory::create_descriptor(
    const AttnResScoresParams& operation_attributes,
    const AttnResScoresInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.stats.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto stats_data_format = datatype_to_dataformat_converter(tensor_args.stats.dtype());
    const auto stats_tile_size = tt::tile_size(stats_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
    const auto num_partials = operation_attributes.num_partials;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    auto [num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_tiles_group_1, num_tiles_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // One output tile consumes one tile of each statistic per rank, pushed as
    // pairs. Double-buffered so the reader runs a candidate ahead of compute.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kStatsPerCandidate * num_partials * 2 * stats_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = stats_data_format,
            .page_size = stats_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = output_data_format,
            .page_size = output_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Candidate c occupies output page c and input pages c and c + C: dim 1 is
    // the stacked pair and everything below it is identical in both halves, so
    // the stride between a candidate's two statistics is the whole output. A
    // gathering collective repeats that whole pair per rank, one output's worth
    // of pages for each of the two statistics.
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(num_output_tiles),
        num_partials,
        static_cast<uint32_t>(kStatsPerCandidate * num_output_tiles)};
    TensorAccessorArgs(*tensor_args.stats.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = std::string(kAttnResScoresKernelDir) + "reader_attn_res_scores.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = std::string(kAttnResScoresKernelDir) + "writer_attn_res_scores.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The SFPU scalar binops take an fp32 value as its bit pattern.
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = std::string(kAttnResScoresKernelDir) + "attn_res_scores.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = {
        std::bit_cast<uint32_t>(operation_attributes.inv_hidden_size),
        std::bit_cast<uint32_t>(operation_attributes.eps),
        num_partials};
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const stats_buffer = tensor_args.stats.buffer();
    auto* const output_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0, start_id = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core{i % num_cores_x, i / num_cores_x};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_kernel_desc.emplace_runtime_args(core, {stats_buffer, num_tiles_per_core, start_id});
        writer_kernel_desc.emplace_runtime_args(core, {output_buffer, num_tiles_per_core, start_id});
        compute_kernel_desc.emplace_runtime_args(core, {num_tiles_per_core});

        start_id += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
