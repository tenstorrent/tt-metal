// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "randn_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::randn {

using namespace tt;
using namespace tt::tt_metal;

std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

namespace {
// Persistent generator for the unseeded path, so consecutive calls advance instead of repeating.
std::mt19937 default_rng(std::random_device{}());

// Kernel push order in create_descriptor; override_runtime_arguments indexes the cached program by these.
constexpr uint32_t writer_kernel_idx = 0;
constexpr uint32_t compute_kernel_idx = 1;

// Core layout shared by create_descriptor and override_runtime_arguments. It mirrors
// split_work_to_cores (num_cores = min(tiles, grid)) + column-major grid_to_cores, so the cache-hit
// path can walk the cores without re-running the full work split.
uint32_t randn_num_cores(uint32_t units_to_divide, const CoreCoord& grid) {
    return std::min<uint32_t>(units_to_divide, grid.x * grid.y);
}
CoreCoord randn_core(uint32_t i, const CoreCoord& grid) { return {i / grid.y, i % grid.y}; }

}  // namespace

auto get_random_seed(std::mt19937& rng) -> uint32_t { return distribution(rng); }

ProgramDescriptor RandnDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    TT_FATAL(
        num_cores == randn_num_cores(units_to_divide, grid),
        "randn core count {} diverged from split_work_to_cores ({}); override_runtime_arguments would patch the wrong "
        "cores",
        randn_num_cores(units_to_divide, grid),
        num_cores);

    ProgramDescriptor desc;

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr uint32_t in_out_num_tiles = 2;

    constexpr uint32_t dst_cb_id = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_out_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = dst_cb_id,
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/randn/device/kernels/";
    std::vector<uint32_t> writer_compile_time_args{dst_cb_id};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
    const std::string writer_file_path = kernels_dir_path + "writer_standard_normal.cpp";
    const std::string compute_file_path = kernels_dir_path + "compute_standard_normal.cpp";

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_file_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

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
    compute_desc.compile_time_args = {dst_cb_id};
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                   // generated number out of range [from, to)
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };
    compute_desc.runtime_args.reserve(num_cores);

    std::mt19937 seeded_rng(operation_attributes.seed.value_or(0));
    std::mt19937& rng = operation_attributes.seed.has_value() ? seeded_rng : default_rng;

    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = randn_core(i, grid);
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // The per-core seed is hash-excluded (see compute_program_hash): baked here for the cache-miss
        // build and redrawn on every cache hit by override_runtime_arguments.
        uint32_t seed = get_random_seed(rng);
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{seed, units_per_core});

        writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void RandnDeviceOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Only the per-core seeds (hash-excluded) and the output address change between hits; tile
    // offsets and counts derive from the hashed shape and are left as built.
    const auto grid = output.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = randn_num_cores(output.physical_volume() / constants::TILE_HW, grid);
    const uint32_t output_addr = output.buffer()->address();

    std::mt19937 seeded_rng(operation_attributes.seed.value_or(0));
    std::mt19937& rng = operation_attributes.seed.has_value() ? seeded_rng : default_rng;

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = randn_core(i, grid);
        {
            auto& runtime_args = GetRuntimeArgs(program, compute_kernel_idx, core);
            runtime_args[0] = get_random_seed(rng);
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_idx, core);
            runtime_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::randn
