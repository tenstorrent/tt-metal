// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

// TEMPLATE / SKELETON ONLY.
//
// Mirrors the single-core skeleton but shows how output work would be split
// across the device's compute grid. The actual kernels and per-core runtime
// args are left as TODOs, so this is not functional.
ProgramDescriptor MatmulDecodeDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t in0_tile_size = tt::tile_size(in0_data_format);
    const uint32_t in1_tile_size = tt::tile_size(in1_data_format);
    const uint32_t out_tile_size = tt::tile_size(out_data_format);

    // Number of output tiles to distribute across cores.
    const uint32_t num_output_tiles_total = output_tensor.physical_volume() / constants::TILE_HW;

    IDevice* device = input_tensor_a.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // TEMPLATE: split the output tiles across the available cores. The returned
    // core groups / per-core counts would feed the per-core runtime args below.
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    ProgramDescriptor desc;

    // ---- Circular buffers (allocated on every participating core) ----
    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    constexpr uint32_t in1_cb_index = CBIndex::c_1;
    constexpr uint32_t out_cb_index = CBIndex::c_16;
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in0_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in1_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * out_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
        }}},
    });

    // ---- TODO: kernels + per-core runtime args ----
    // For each core in all_cores, look up its tile count (core_group_1 vs
    // core_group_2) and emit reader/compute/writer runtime args describing the
    // tile range that core is responsible for.
    (void)num_cores;
    (void)core_group_1;
    (void)core_group_2;
    (void)num_tiles_per_core_group_1;
    (void)num_tiles_per_core_group_2;

    return desc;
}

}  // namespace ttnn::operations::experimental::matmul_decode
