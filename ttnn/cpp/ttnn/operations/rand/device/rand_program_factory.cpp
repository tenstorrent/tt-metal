// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <cstring>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

using Factory = RandDeviceOperation::RandMeshWorkloadFactory;

Factory::cached_mesh_workload_t Factory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, output);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

Factory::cached_program_t Factory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    Program program = Program();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    constexpr uint32_t intermed_cb_id = CBIndex::c_24;
    CircularBufferConfig cb_intermed_config =
        CircularBufferConfig(intermed_num_tiles * intermed_tile_size, {{intermed_cb_id, tt::DataFormat::Float32}})
            .set_page_size(intermed_cb_id, intermed_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed_config);

    constexpr uint32_t dst_cb_id = CBIndex::c_0;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(in_out_num_tiles * dtype_tile_size, {{dst_cb_id, out_data_format}})
            .set_page_size(dst_cb_id, dtype_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/rand/device/kernels/";
    std::vector<uint32_t> writer_compile_time_args{intermed_cb_id, dst_cb_id};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
    const std::string writer_file_path = kernels_dir_path + "writer_uniform.cpp";
    const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};
    const std::string compute_file_path = kernels_dir_path + "compute_uniform.cpp";

    std::map<std::string, std::string> writer_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_file_path, all_cores, WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_file_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                       // generated number out of range [from, to)
            .dst_full_sync_en = false,
            .math_approx_mode = true,
            .compile_args = compute_compile_time_args,
        });

    // Derive a per-device seed offset so that devices holding different shards
    // generate distinct random sequences, while replicas share the same seed.
    // Only mesh dimensions marked as sharded contribute to the index; replicate
    // dimensions are ignored so that all replicas of the same shard are identical.
    uint32_t device_seed_offset = 0;
    const auto& shard_mask = operation_attributes.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        const auto& mesh_shape = operation_attributes.device->shape();
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += mesh_coordinate[i] * shard_stride;
                shard_stride *= mesh_shape[i];
            }
        }
        device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(cores.size());
    }

    uint32_t tile_offset = 0;
    for (int i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const float eps = 1e-6f;
        const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
        const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

        // Each core gets its own seed to increase entropy across the output tensor.
        // With a user-supplied seed (!=0) the value is deterministic; with seed==0
        // a fresh random seed is drawn from the host RNG for every core invocation.
        uint32_t seed =
            operation_attributes.seed != 0 ? operation_attributes.seed + i + device_seed_offset : get_random_seed();

        std::vector<uint32_t> compute_runtime_args = {seed, from_bits, to_bits, tile_offset, units_per_core};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {output.buffer()->address(), tile_offset, units_per_core};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.compute_kernel_id = compute_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void Factory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    const auto& shard_mask = operation_attributes.mesh_dim_is_sharded;
    const auto& mesh_shape = operation_attributes.device->shape();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        auto& cores = shared_vars.cores;

        const uint32_t output_addr = output.buffer()->address();

        uint32_t device_seed_offset = 0;
        if (!shard_mask.empty()) {
            const auto& coord = coordinate_range.start_coord();
            size_t shard_linear_idx = 0;
            size_t shard_stride = 1;
            for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
                if (shard_mask[i]) {
                    shard_linear_idx += coord[i] * shard_stride;
                    shard_stride *= mesh_shape[i];
                }
            }
            device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(cores.size());
        }

        for (int i = 0; i < cores.size(); ++i) {
            {
                auto& runtime_args = GetRuntimeArgs(program, shared_vars.compute_kernel_id, cores[i]);
                runtime_args[0] = operation_attributes.seed != 0 ? operation_attributes.seed + i + device_seed_offset
                                                                 : get_random_seed();
                runtime_args[1] = from_bits;
                runtime_args[2] = to_bits;
            }
            {
                auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, cores[i]);
                runtime_args[0] = output_addr;
            }
        }
    }
}

}  // namespace ttnn::operations::rand
