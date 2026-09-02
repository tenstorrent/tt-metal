// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/device/attn_res_weighted_reduce_nc_program_factory.hpp"

#include <algorithm>

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

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/device/kernels/";

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResWeightedReduceNCProgramFactory::create_descriptor(
    const AttnResWeightedReduceNCParams& operation_attributes,
    const AttnResWeightedReduceNCInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(tensor_args.input.dtype());
    const auto input_tile_size = tt::tile_size(input_data_format);
    const auto weight_data_format = datatype_to_dataformat_converter(tensor_args.weight.dtype());
    const auto weight_tile_size = tt::tile_size(weight_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    // Validation pins rank to 4 and dim to 1, so there are no dims between the
    // reduced axis and the trailing two: the inner block is exactly Ht*Wt tiles.
    const auto& input_shape = tensor_args.input.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_candidates = input_shape[operation_attributes.dim];
    const uint32_t inner_tile_size = Ht * Wt;

    // The weight is [R, C, H, 1] in TILE layout, i.e. one tile column wide, so
    // its inner block is Ht tiles against the input's Ht*Wt. Every tile in an
    // input row of Wt shares one weight tile — that sharing is what the work
    // split below is arranged to exploit.
    const uint32_t weight_inner_tile_size = Ht;
    const uint32_t weight_reduce_tile_size = num_candidates * Ht;
    const uint32_t num_sites = tensor_args.weight.padded_shape()[0];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // Sites are reduced in groups, one DEST tile per site in the group, so a
    // group's worth of accumulators has to fit in DEST — which the sync mode and
    // fp32 accumulation each halve. Exceeding it does not fault, it corrupts the
    // tiles past the end, so the group size is derived rather than chosen.
    const uint32_t dest_capacity = get_dest_reg_count(operation_attributes.compute_kernel_config);
    const uint32_t sites_per_group = std::min(num_sites, dest_capacity);
    const uint32_t num_groups = (num_sites + sites_per_group - 1) / sites_per_group;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // Split the input's tile positions, not the output's tiles. Every site reads
    // the same input, so a core that owns a position produces that position for
    // all R sites and reads the candidates for it once per group. Splitting the
    // output instead would scatter one position across cores and each of them
    // would re-read the whole reduction.
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    auto [num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_tiles_group_1, num_tiles_group_2] =
        tt::tt_metal::split_work_to_cores(grid, inner_tile_size, /*row_wise=*/true);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // No intermediate CB: the running sums live in DEST for the whole reduction,
    // which is what lets one pass over the input do the multiply and the add.
    // A whole reduction's candidates are one CB unit, so the group loop can walk
    // them repeatedly without the reader re-fetching between sites.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_candidates * 2 * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = input_data_format,
            .page_size = input_tile_size,
        }}},
    });

    // One group's weight sets, single-buffered. The set turns over once every Wt
    // output tiles, so a prefetch buffer would idle through nearly the whole row
    // while doubling what is already the largest CB here: sites_per_group * C
    // tiles, over a hundred kilobytes at the shape AttnRes asks for.
    desc.cbs.push_back(CBDescriptor{
        .total_size = sites_per_group * num_candidates * weight_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = weight_data_format,
            .page_size = weight_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = sites_per_group * 2 * output_tile_size,
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
    // Every stride is a compile-time arg. The reader divides by inner_tile_size
    // and by Wt once per output tile, and RISC-V has no divide instruction —
    // constants let the compiler emit multiply-shift instead of a libcall.
    std::vector<uint32_t> reader_compile_time_args = {
        num_candidates,
        inner_tile_size,
        Wt,
        weight_inner_tile_size,
        weight_reduce_tile_size,
        num_sites,
        sites_per_group,
        num_groups};
    TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.weight.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {inner_tile_size, num_sites, sites_per_group, num_groups};
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = std::string(kKernelDir) + "reader_weighted_reduce_nc.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = std::string(kKernelDir) + "writer_weighted_reduce_nc.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // One descriptor over all cores, with the per-core tile count as a runtime
    // arg. fast_reduce_nc splits into two compute kernels to keep that count
    // compile-time; here the loop that matters — the MAC over candidates — is
    // already compile-time bounded, so the second kernel would buy nothing.
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = std::string(kKernelDir) + "weighted_reduce_nc.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = {num_candidates, Wt, num_sites, sites_per_group, num_groups};
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Contiguous, not round-robin. fast_reduce_nc hands core i tiles
    // i, i+num_cores, i+2*num_cores, ... which is free when every output tile
    // reads a disjoint input column. It is not free here: consecutive strides of
    // num_cores land on a different token row almost every time, so the weight
    // set would be refetched per position — G times a pass over the weight.
    // Contiguous ranges give each core ~num_tiles/Wt distinct rows, so the
    // weights are read about once per group.
    auto* const input_buffer = tensor_args.input.buffer();
    auto* const weight_buffer = tensor_args.weight.buffer();
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

        reader_kernel_desc.emplace_runtime_args(core, {input_buffer, weight_buffer, num_tiles_per_core, start_id});
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
