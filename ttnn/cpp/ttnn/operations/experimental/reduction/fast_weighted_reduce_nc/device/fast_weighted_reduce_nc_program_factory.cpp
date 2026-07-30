// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/device/fast_weighted_reduce_nc_program_factory.hpp"

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

constexpr auto kKernelDir = "ttnn/cpp/ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/device/kernels/";

}  // namespace

tt::tt_metal::ProgramDescriptor FastWeightedReduceNCProgramFactory::create_descriptor(
    const FastWeightedReduceNCParams& operation_attributes,
    const FastWeightedReduceNCInputs& tensor_args,
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
    const uint32_t reduce_tile_size = num_candidates * inner_tile_size;

    // The weight is [B, C, H, 1] in TILE layout, i.e. one tile column wide, so
    // its inner block is Ht tiles against the input's Ht*Wt. Every tile in an
    // input row of Wt shares one weight tile — that sharing is what the work
    // split below is arranged to exploit.
    const uint32_t weight_inner_tile_size = Ht;
    const uint32_t weight_reduce_tile_size = num_candidates * Ht;

    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // Largest factor of num_candidates that is <= 8. Divisibility is required,
    // not just preferred: the compute kernel derives a candidate's weight index
    // from its position in the granule as `j * granularity + k`, which is only
    // the true candidate index when the granule tiles a whole reduction.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_candidates % input_granularity == 0) {
            break;
        }
    }

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
    // No intermediate CB: the running sum lives in dst0 for the whole reduction,
    // which is what lets one pass over the input do the multiply and the add.
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_granularity * 2 * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = input_data_format,
            .page_size = input_tile_size,
        }}},
    });

    // Two full candidate sets: the reader prefetches the next row's weights
    // while compute is still consuming the current row's. Sized in tiles, so a
    // large reduction dim costs L1 here; C is 9 for the case this was built for.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_candidates * 2 * weight_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = weight_data_format,
            .page_size = weight_tile_size,
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
    // Every stride is a compile-time arg. The reader divides by inner_tile_size
    // and by Wt once per output tile, and RISC-V has no divide instruction —
    // constants let the compiler emit multiply-shift instead of a libcall.
    std::vector<uint32_t> reader_compile_time_args = {
        input_granularity,
        num_candidates,
        inner_tile_size,
        reduce_tile_size,
        Wt,
        weight_inner_tile_size,
        weight_reduce_tile_size};
    TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*tensor_args.weight.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
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
    compute_kernel_desc.compile_time_args = {num_candidates, input_granularity, Wt};
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Contiguous, not round-robin. fast_reduce_nc hands core i output tiles
    // i, i+num_cores, i+2*num_cores, ... which is free when every output tile
    // reads a disjoint input column. It is not free here: consecutive strides of
    // num_cores land on a different token row almost every time, so the weight
    // set would be refetched per output tile — one extra pass over a tensor the
    // size of the input. Contiguous ranges give each core ~num_tiles/Wt distinct
    // rows, so the weights are read about once.
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
