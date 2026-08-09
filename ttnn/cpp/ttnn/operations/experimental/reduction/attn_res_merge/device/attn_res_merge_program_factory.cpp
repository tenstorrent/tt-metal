// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_merge/device/attn_res_merge_program_factory.hpp"

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

constexpr auto kAttnResMergeKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/reduction/attn_res_merge/device/kernels/";

// partial and prefix_sum, interleaved into one CB so both broadcast MACs read
// the same CB pair and the compute kernel needs a single `init_bcast`.
constexpr uint32_t kOperands = 2;
// The row weights `a` and `b`, and the scalars they are derived from.
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t kScalars = 3;

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResMergeProgramFactory::create_descriptor(
    const AttnResMergeParams& operation_attributes,
    const AttnResMergeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.partial.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto wide_data_format = datatype_to_dataformat_converter(tensor_args.partial.dtype());
    const auto wide_tile_size = tt::tile_size(wide_data_format);
    const auto scalar_data_format = datatype_to_dataformat_converter(tensor_args.shift.dtype());
    const auto scalar_tile_size = tt::tile_size(scalar_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    const auto& input_shape = tensor_args.partial.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;

    // A scalar operand is one tile column wide, so its dim-0 plane is Ht pages and
    // selecting a read site is a page offset. Resolved here rather than in the
    // kernel because a shared operand ignores the site (see the operand comment on
    // AttnResMergeParams), and that test belongs where the shapes are.
    const auto scalar_page_offset = [&](const Tensor& scalar) {
        return scalar.padded_shape()[0] == 1 ? 0u : operation_attributes.site * Ht;
    };
    const uint32_t shift_page_offset = scalar_page_offset(tensor_args.shift);
    const uint32_t mass_page_offset = scalar_page_offset(tensor_args.mass);
    const uint32_t live_scores_page_offset = scalar_page_offset(tensor_args.live_scores);

    // The partial is full width, so its dim-0 plane is a whole Ht*Wt block. Same
    // selection as the scalars, different stride: a caller that reduced every read
    // site in one dispatch hands the batch over whole and names its site here.
    const uint32_t partial_page_offset =
        tensor_args.partial.padded_shape()[0] == 1 ? 0u : operation_attributes.site * Ht * Wt;

    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
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
    // One output tile needs one tile of each full-width operand, pushed as a
    // pair. Double-buffered so the reader runs a pair ahead of compute.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kOperands * 2 * wide_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = wide_data_format,
            .page_size = wide_tile_size,
        }}},
    });

    // The row weights `a` and `b`, produced by compute rather than read: the
    // scalar chain that derives them runs in dst and packs the two results here,
    // where the broadcast MAC unpacks them alongside c_0. Their format follows
    // the output, since that is what the packer emits.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kRowWeights * 2 * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = output_data_format,
            .page_size = output_tile_size,
        }}},
    });

    // shift, mass, live_scores — one tile each per token row, in one CB so the
    // derivation reads all three through a single unpack configuration.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kScalars * 2 * scalar_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scalar_data_format,
            .page_size = scalar_tile_size,
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
    // Wt is a compile-time arg because the reader divides by it once per output
    // tile to find the token row, and RISC-V has no divide instruction.
    std::vector<uint32_t> reader_compile_time_args = {Wt};
    TensorAccessorArgs(*tensor_args.partial.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.prefix_sum.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.shift.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.mass.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.live_scores.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = std::string(kAttnResMergeKernelDir) + "reader_attn_res_merge.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = std::string(kAttnResMergeKernelDir) + "writer_attn_res_merge.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = std::string(kAttnResMergeKernelDir) + "attn_res_merge.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = {Wt};
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Contiguous, not round-robin: the Wt tiles of one token row share a single
    // scalar set, and only a contiguous run walks whole rows. Strides of
    // num_cores would land on a different row almost every tile and re-derive
    // the row weights per output tile.
    auto* const partial_buffer = tensor_args.partial.buffer();
    auto* const prefix_sum_buffer = tensor_args.prefix_sum.buffer();
    auto* const shift_buffer = tensor_args.shift.buffer();
    auto* const mass_buffer = tensor_args.mass.buffer();
    auto* const live_scores_buffer = tensor_args.live_scores.buffer();
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

        reader_kernel_desc.emplace_runtime_args(
            core,
            {partial_buffer,
             prefix_sum_buffer,
             shift_buffer,
             mass_buffer,
             live_scores_buffer,
             num_tiles_per_core,
             start_id,
             shift_page_offset,
             mass_page_offset,
             live_scores_page_offset,
             partial_page_offset});
        writer_kernel_desc.emplace_runtime_args(core, {output_buffer, num_tiles_per_core, start_id});
        compute_kernel_desc.emplace_runtime_args(core, {num_tiles_per_core, start_id});

        start_id += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
