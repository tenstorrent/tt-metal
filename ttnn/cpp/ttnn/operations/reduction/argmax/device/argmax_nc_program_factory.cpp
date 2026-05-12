// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nc_device_operation.hpp"

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

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

ArgMaxNCProgramFactory::cached_program_t ArgMaxNCProgramFactory::create(
    const ArgMaxNCParams& operation_attributes, const ArgMaxNCInputs& tensor_args, Tensor& tensor_return_value) {
    auto* device = tensor_args.input.device();
    Program program{};

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
    // Indices are staged through L1 as fp32 scalars (the reader writes (float)k
    // into every element of an idx tile). The compute kernel accumulates argmax
    // arithmetically in fp32 and typecasts to UInt32 before packing to the
    // UInt32 output CB.
    const DataFormat index_data_format = DataFormat::Float32;
    const DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tile_size(input_data_format);
    const uint32_t index_tile_size = tile_size(index_data_format);
    const uint32_t output_tile_size = tile_size(output_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    // We need 32-bit DST to hold uint32 indices in registers.
    fp32_dest_acc_en = true;

    // Split work across cores.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = grid.x;
    auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    // Circular buffers (double-buffered input so the compute kernel can overlap with reader).
    constexpr uint32_t input_cb_depth = 2;
    constexpr uint32_t index_cb_depth = 2;
    constexpr uint32_t output_cb_depth = 2;

    constexpr auto src_cb = CBIndex::c_0;
    constexpr auto idx_cb = CBIndex::c_1;
    constexpr auto out_cb = CBIndex::c_16;

    CircularBufferConfig src_cb_config =
        CircularBufferConfig(input_cb_depth * input_tile_size, {{src_cb, input_data_format}})
            .set_page_size(src_cb, input_tile_size);
    CreateCircularBuffer(program, all_cores, src_cb_config);

    CircularBufferConfig idx_cb_config =
        CircularBufferConfig(index_cb_depth * index_tile_size, {{idx_cb, index_data_format}})
            .set_page_size(idx_cb, index_tile_size);
    CreateCircularBuffer(program, all_cores, idx_cb_config);

    CircularBufferConfig out_cb_config =
        CircularBufferConfig(output_cb_depth * output_tile_size, {{out_cb, output_data_format}})
            .set_page_size(out_cb, output_tile_size);
    CreateCircularBuffer(program, all_cores, out_cb_config);

    // Kernels
    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/writer_argmax_nc.cpp";
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_nc_compute.cpp";

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_args);

    const KernelHandle reader_kernel_id =
        CreateKernel(program, reader_kernel_file, all_cores, ReaderDataMovementConfig(reader_compile_args));

    const KernelHandle writer_kernel_id =
        CreateKernel(program, writer_kernel_file, all_cores, WriterDataMovementConfig(writer_compile_args));

    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Route the unpacker directly to DST for fp32 CBs so 32-bit precision is
    // preserved end-to-end. Without this, `copy_tile` funnels data through
    // SrcA (bf16 precision) and the argmax picks a different index than torch
    // whenever two bf16-rounded values tie but their fp32 originals differ.
    // - `src_cb` is fp32 only when the user input is fp32.
    // - `idx_cb` is always fp32 so (float)k is exactly representable for all k.
    // Size must match host JIT expectation (NUM_CIRCULAR_BUFFERS = 64 on Blackhole / host; WH device uses 32).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (input_data_format == DataFormat::Float32) {
        unpack_to_dest_mode[static_cast<uint32_t>(src_cb)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    unpack_to_dest_mode[static_cast<uint32_t>(idx_cb)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    auto make_compute_config = [&](uint32_t ntiles_per_core) {
        return ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = std::vector<uint32_t>{ntiles_per_core, num_reduce_tiles},
            .defines = compute_defines,
        };
    };

    if (!core_group_1.ranges().empty()) {
        CreateKernel(program, compute_kernel_file, core_group_1, make_compute_config(num_tiles_per_core_group_1));
    }
    if (!core_group_2.ranges().empty()) {
        CreateKernel(program, compute_kernel_file, core_group_2, make_compute_config(num_tiles_per_core_group_2));
    }

    // Runtime args per core.
    std::vector<CoreCoord> ordered_cores;
    ordered_cores.reserve(num_cores_to_be_used);
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        ordered_cores.emplace_back(i % num_cores_x, i / num_cores_x);
    }

    const uint32_t dim_is_zero = (normalized_dim == 0) ? 1u : 0u;

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

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_tiles_this_core,
             tile_offset,
             num_reduce_tiles,
             reduce_tile_size,
             inner_tile_size,
             dim_is_zero});

        SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), num_tiles_this_core, tile_offset});

        tile_offset += num_tiles_this_core;
    }

    return cached_program_t{
        std::move(program),
        {reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_x, std::move(ordered_cores)},
    };
}

void ArgMaxNCProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ArgMaxNCParams&,
    const ArgMaxNCInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& ordered_cores = cached_program.shared_variables.ordered_cores;

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (const auto& core : ordered_cores) {
        reader_args_by_core[core.x][core.y][0] = input_buffer->address();
        writer_args_by_core[core.x][core.y][0] = output_buffer->address();
    }
}

}  // namespace ttnn::prim
