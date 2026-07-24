// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <ctime>
#include <limits>
#include <random>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "uniform_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::uniform {

using namespace tt;
using namespace tt::tt_metal;

namespace {
std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution<int32_t> distribution(1, std::numeric_limits<int32_t>::max());

uint32_t get_random_seed() { return distribution(rng); }

// Work split used by create_descriptor on both cache miss and hit (via override_runtime_arguments)
// so both derive the identical core list.
struct UniformWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

UniformWorkSplit uniform_work_split(Tensor& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores)};
}
}  // namespace

static constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp";

ProgramDescriptor UniformDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    IDevice* device = output.device();
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2, cores] =
        uniform_work_split(output);
    const auto num_cores_total = cores.size();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    constexpr uint32_t intermed_cb_id = CBIndex::c_24;
    constexpr uint32_t dst_cb_id = CBIndex::c_0;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    ProgramDescriptor desc;

    // Intermediate CB (Float32)
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_num_tiles * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed_cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = intermed_tile_size,
        }}},
    });

    // Output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_out_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = dst_cb_id,
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    // Writer kernel
    KernelDescriptor::Defines writer_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    KernelDescriptor::CompileTimeArgs writer_ct_args{intermed_cb_id, dst_cb_id};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_total);

    // Compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = std::move(all_cores);
    compute_desc.compile_time_args = {intermed_cb_id};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                   // generated number out of range [from, to)
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };
    compute_desc.runtime_args.reserve(num_cores_total);

    // Runtime args per core
    const float eps = 1e-6f;
    const uint32_t f2u_from = std::bit_cast<uint32_t>(operation_attributes.from);
    // -eps make sure that generated number is < operation_attributes.to
    const uint32_t f2u_to = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    uint32_t tile_offset = 0;
    for (int i = 0; i < static_cast<int>(cores.size()); ++i) {
        const auto& core = cores[i];
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // Each core has its own seed to increase the number of generated random numbers
        uint32_t seed = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();

        // seed/from/to are DYNAMIC (excluded from compute_program_hash): derived here from attrs on
        // both the cache-miss build and every cache hit (via override_runtime_arguments).
        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{seed, f2u_from, f2u_to, tile_offset, units_per_core});

        // Register the (in-place) output address as a Buffer* binding so uniform takes the fast
        // cache-hit path; the framework allows the input==output alias (see resolve_bindings).
        writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void UniformDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-derive seed/from/to (hash-excluded) and every other per-dispatch arg from the single source of
    // truth (create_descriptor) and re-apply to the cached program. No rebuild; supersedes get_dynamic.
    auto desc = create_descriptor(operation_attributes, tensor_args, output);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}

}  // namespace ttnn::operations::uniform
