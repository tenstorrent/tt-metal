// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "bernoulli_new_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::bernoulli_new {

using namespace tt;
using namespace tt::tt_metal;

namespace {
std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution d(1, 1 << 20);
uint32_t get_random_seed() { return d(rng); }
}  // namespace

// ---------------------------------------------------------------
// The ONLY method an OP writer implements.
//
// Returns a ProgramDescriptor that fully describes:
//   - Circular buffers  (compile-time, hashed)
//   - Kernels with compile args & defines  (compile-time, hashed)
//   - Runtime args per core  (runtime, NOT hashed, memcpy'd on cache hit)
//
// No create().  No override_runtime_arguments().  No shared_variables_t.
// ---------------------------------------------------------------
ProgramDescriptor BernoulliNewDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;

    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    // ---- Circular buffers ----

    ProgramDescriptor desc;

    constexpr uint32_t num_tiles = 2;
    auto in_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t in_dtype_tile_size = tile_size(in_data_format);
    constexpr uint32_t in_cb_id = CBIndex::c_0;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * in_dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb_id,
            .data_format = in_data_format,
            .page_size = in_dtype_tile_size,
        }}},
    });

    const uint32_t float32_tile_size = tile_size(tt::DataFormat::Float32);
    constexpr uint32_t intermed_cb_id = CBIndex::c_24;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * float32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed_cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = float32_tile_size,
        }}},
    });

    auto out_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_dtype_tile_size = tile_size(out_data_format);
    constexpr uint32_t intermed1_cb_id = CBIndex::c_25;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * out_dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed1_cb_id,
            .data_format = out_data_format,
            .page_size = out_dtype_tile_size,
        }}},
    });

    // ---- Kernels ----

    // NOTE: We reuse the same kernel source files as original Bernoulli.
    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/bernoulli/device/kernels/";

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args{in_cb_id};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernels_dir_path + "reader_bernoulli.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args{in_cb_id, intermed_cb_id, intermed1_cb_id};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines writer_defines;
    switch (input.dtype()) {
        case DataType::BFLOAT16: writer_defines.emplace_back("INPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("INPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }
    switch (output.dtype()) {
        case DataType::BFLOAT16: writer_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kernels_dir_path + "writer_bernoulli.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = writer_defines;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kernels_dir_path + "compute_bernoulli.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,  // must always be true, otherwise generated floats are in [0.4, 0.5]
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // ---- Runtime args per core ----

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

        reader_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{input.buffer()->address(), tile_offset, units_per_core});

        uint32_t seed = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();
        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{seed, tile_offset, units_per_core});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output.buffer()->address(), tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

// ---------------------------------------------------------------
// Called by the framework AFTER patching buffer addresses on cache hit.
// Only updates the random seed in the compute kernel's runtime args.
// ---------------------------------------------------------------
void BernoulliNewDeviceOperation::ProgramFactory::override_runtime_arguments(
    Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    // Compute kernel is index 2 (reader=0, writer=1, compute=2 per create_descriptor).
    constexpr uint32_t compute_kernel_handle = 2;
    for (int i = 0; i < static_cast<int>(cores.size()); ++i) {
        auto& runtime_args = GetRuntimeArgs(program, compute_kernel_handle, cores[i]);
        runtime_args[0] = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();
    }
}

}  // namespace ttnn::operations::bernoulli_new
