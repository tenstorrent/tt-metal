// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nc_device_operation.hpp"

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <utility>
#include <vector>

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

// Returns {inner_tile_size, reduce_tile_size} for a TILE-layout input reducing along `dim`.
//   inner_tile_size  = product of tile counts of dims strictly inside `dim`
//                      (i.e. dims dim+1 .. rank-3) * Ht * Wt
//                    = number of output tiles per "slab" (one value of dim[dim])
//   reduce_tile_size = shape[dim] * inner_tile_size
//                    = stride between input tiles along the reduced dim,
//                      for the base `output_tile_id` inside the slab.
template <typename ShapeT>
std::pair<uint32_t, uint32_t> extract_nc_strides(const ShapeT& padded_shape, int32_t dim) {
    const int32_t rank = static_cast<int32_t>(padded_shape.rank());
    const uint32_t Wt = padded_shape[rank - 1] / TILE_WIDTH;
    const uint32_t Ht = padded_shape[rank - 2] / TILE_HEIGHT;

    uint32_t inner_dims_product = 1;
    for (int32_t i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= padded_shape[i];
    }

    const uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    const uint32_t reduce_dim = padded_shape[dim];
    const uint32_t reduce_tile_size = reduce_dim * inner_tile_size;
    return {inner_tile_size, reduce_tile_size};
}

}  // namespace

ProgramDescriptor ArgMaxNCDeviceOperation::create_descriptor(
    const ArgMaxNCParams& operation_attributes, const ArgMaxNCInputs& tensor_args, Tensor& tensor_return_value) {
    auto* device = tensor_args.input.device();

    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    const auto& input_shape = input.padded_shape();
    const int32_t rank = static_cast<int32_t>(input_shape.rank());
    const int32_t normalized_dim =
        operation_attributes.dim < 0 ? operation_attributes.dim + rank : operation_attributes.dim;

    const auto [inner_tile_size, reduce_tile_size] = extract_nc_strides(input_shape, normalized_dim);
    const uint32_t num_reduce_tiles = input_shape[normalized_dim];
    const uint32_t num_output_tiles = output.physical_volume() / TILE_HW;

    const DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    // Indices live entirely in DST: the compute kernel materializes them as
    // uint32 via fill_tile_int<UInt32> rather than staging them through a CB.
    // The uint32 result is packed straight to the UInt32 output CB (no typecast).
    const DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tile_size(input_data_format);
    const uint32_t output_tile_size = tile_size(output_data_format);

    // packer_l1_acc is irrelevant here: argmax packs a single index result per
    // output tile, there is nothing to accumulate across passes.
    // ComputeConfigDescriptor has no packer_l1_acc field, and the previous factory
    // did not forward the knob either.
    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    // We need 32-bit DST to hold uint32 indices in registers.
    fp32_dest_acc_en = true;

    // Split work across cores.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = grid.x;
    const bool use_sub_core_grids = operation_attributes.sub_core_grids.has_value();
    auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] =
            use_sub_core_grids ? split_work_to_cores(*operation_attributes.sub_core_grids, num_output_tiles)
                               : split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    // Circular buffers (double-buffered input so the compute kernel can overlap with reader).
    constexpr uint32_t input_cb_depth = 2;
    constexpr uint32_t output_cb_depth = 2;

    constexpr auto src_cb = CBIndex::c_0;
    constexpr auto out_cb = CBIndex::c_16;

    ProgramDescriptor desc;
    desc.cbs.reserve(2);
    desc.kernels.reserve(4);

    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_depth * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb),
            .data_format = input_data_format,
            .page_size = input_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_depth * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb),
            .data_format = output_data_format,
            .page_size = output_tile_size,
        }}},
    });

    // Kernels
    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/writer_argmax_nc.cpp";
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_nc_compute.cpp";

    Buffer* input_buffer = input.buffer();
    Buffer* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "argmax_nc input tensor has no device buffer");
    TT_FATAL(output_buffer != nullptr, "argmax_nc output tensor has no device buffer");

    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = reader_kernel_file;
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_compile_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = writer_kernel_file;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    // Route the unpacker directly to DST for the fp32 value CB so 32-bit precision
    // is preserved end-to-end. Without this, `copy_tile` funnels data through SrcA
    // (bf16 precision) and the argmax picks a different index than torch whenever
    // two bf16-rounded values tie but their fp32 originals differ. bf16 inputs
    // don't need the override — bf16 unpack widens into the full 32-bit DST slot
    // on its own.
    // Size must match host JIT expectation (NUM_CIRCULAR_BUFFERS = 64 on Blackhole / host; WH device uses 32).
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (input_data_format == DataFormat::Float32) {
        unpack_to_dest_mode[static_cast<uint32_t>(src_cb)] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto make_compute_kernel = [&](const CoreRangeSet& cores, uint32_t ntiles_per_core) {
        KernelDescriptor compute_kernel;
        compute_kernel.kernel_source = compute_kernel_file;
        compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_kernel.core_ranges = cores;
        compute_kernel.compile_time_args = {ntiles_per_core, num_reduce_tiles};
        compute_kernel.defines = compute_defines;
        compute_kernel.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
        return compute_kernel;
    };

    // Runtime args per core.
    std::vector<CoreCoord> ordered_cores;
    ordered_cores.reserve(num_cores_to_be_used);
    if (use_sub_core_grids) {
        for (const auto& range : all_cores.ranges()) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    ordered_cores.emplace_back(x, y);
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            ordered_cores.emplace_back(i % num_cores_x, i / num_cores_x);
        }
    }

    const uint32_t dim_is_zero = (normalized_dim == 0) ? 1u : 0u;

    reader_kernel.runtime_args.reserve(num_cores_to_be_used);
    reader_kernel.buffer_bindings.reserve(num_cores_to_be_used);
    writer_kernel.runtime_args.reserve(num_cores_to_be_used);
    writer_kernel.buffer_bindings.reserve(num_cores_to_be_used);

    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        const CoreCoord& core = ordered_cores[i];
        uint32_t num_tiles_this_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_this_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_this_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("argmax_nc: core not in any core group");
        }

        reader_kernel.emplace_runtime_args(
            core,
            {input_buffer,
             num_tiles_this_core,
             tile_offset,
             num_reduce_tiles,
             reduce_tile_size,
             inner_tile_size,
             dim_is_zero});

        writer_kernel.emplace_runtime_args(core, {output_buffer, num_tiles_this_core, tile_offset});

        tile_offset += num_tiles_this_core;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    if (!core_group_1.ranges().empty()) {
        desc.kernels.push_back(make_compute_kernel(core_group_1, num_tiles_per_core_group_1));
    }
    if (!core_group_2.ranges().empty()) {
        desc.kernels.push_back(make_compute_kernel(core_group_2, num_tiles_per_core_group_2));
    }

    return desc;
}

}  // namespace ttnn::prim
