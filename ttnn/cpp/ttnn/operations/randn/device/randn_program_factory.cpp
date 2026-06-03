// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <ctime>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "randn_device_operation.hpp"

namespace ttnn::operations::randn {

using namespace tt;
using namespace tt::tt_metal;

std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed(std::mt19937& rng) -> uint32_t { return distribution(rng); }

tt::tt_metal::ProgramDescriptor RandnDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ProgramDescriptor desc;

    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr uint32_t in_out_num_tiles = 2;

    constexpr uint32_t dst_cb_id = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_out_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dst_cb_id),
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/randn/device/kernels/";
    std::vector<uint32_t> writer_compile_time_args{dst_cb_id};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
    const std::string writer_file_path = kernels_dir_path + "writer_standard_normal.cpp";
    std::vector<uint32_t> compute_compile_time_args{dst_cb_id};
    const std::string compute_file_path = kernels_dir_path + "compute_standard_normal.cpp";

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_file_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor::Defines compute_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: compute_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        default: break;
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_file_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        // if fp32_dest_acc_en set to false a precision error may occur which makes generated number out of range
        // [from, to)
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    std::mt19937 rng = operation_attributes.seed.has_value() ? std::mt19937(*operation_attributes.seed)
                                                             : std::mt19937(std::time(nullptr));

    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());

    auto* output_buffer = output.buffer();
    uint32_t tile_offset = 0;
    for (auto core : cores) {
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t seed = get_random_seed(rng);

        compute_desc.emplace_runtime_args(core, {seed, units_per_core});
        writer_desc.emplace_runtime_args(core, {output_buffer, tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::randn
