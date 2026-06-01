// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <ctime>
#include <limits>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp";

}  // namespace

ProgramDescriptor RandDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    const auto num_cores_total = cores.size();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    constexpr uint32_t intermed_cb_id = CBIndex::c_24;
    constexpr uint32_t dst_cb_id = CBIndex::c_0;

    const ttnn::MeshCoordinate mesh_coordinate = mesh_dispatch_coordinate.value_or(
        ttnn::MeshCoordinate::zero_coordinate(operation_attributes.device->shape().dims()));

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

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_num_tiles * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed_cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = intermed_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = in_out_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = dst_cb_id,
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    writer_ct_args.reserve(8);
    writer_ct_args.push_back(intermed_cb_id);
    writer_ct_args.push_back(dst_cb_id);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_desc.defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_desc.defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default:
            // The writer kernel only implements float32 and bfloat16 output paths.
            // Fail fast here so we never instantiate a program that can hang at runtime.
            TT_THROW("RandDeviceOperation: unsupported output dtype for writer kernel");
    }
    writer_desc.runtime_args.reserve(num_cores_total);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {intermed_cb_id};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                   // generated number out of range [from, to)
        .dst_full_sync_en = false,
        .math_approx_mode = true,
    };
    compute_desc.runtime_args.reserve(num_cores_total);

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

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

        uint32_t seed =
            operation_attributes.seed != 0 ? operation_attributes.seed + i + device_seed_offset : get_random_seed();

        // seed/from/to are DYNAMIC (excluded from compute_program_hash): baked here for the
        // cache-miss build, and re-applied on every cache hit via get_dynamic_runtime_args().
        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{seed, from_bits, to_bits, tile_offset, units_per_core});

        // Register the output address as a Buffer* binding so rand takes the fast cache-hit path
        // (real program caching) with the address correctly re-patched each dispatch.
        writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

std::vector<tt::tt_metal::DynamicRuntimeArg> RandDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // MUST mirror the seed/from/to runtime args produced in create_descriptor() above:
    //   - writer is pushed as kernel index 0, compute as kernel index 1
    //   - compute runtime args layout is {seed, from_bits, to_bits, tile_offset, units_per_core}
    // Only the per-call values (seed, from, to) are re-applied here; the structural args
    // (tile_offset, units_per_core) are part of the program identity and never change for a given
    // cache entry.  The work-split is recomputed (cheap host-side integer math, no kernel rebuild)
    // so the per-core seed offsets match create_descriptor() exactly.  The
    // test_rand_different_seed_values regression test enforces this mirror.
    constexpr uint32_t kComputeKernelIdx = 1;
    constexpr uint32_t kSeedArgIdx = 0;
    constexpr uint32_t kFromArgIdx = 1;
    constexpr uint32_t kToArgIdx = 2;

    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    // Only the core count/list matter here (per-core seed offsets); the work-split groups and
    // per-group tile counts are part of the static program identity and are not re-applied.
    [[maybe_unused]] auto
        [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
            split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    const ttnn::MeshCoordinate mesh_coordinate = mesh_dispatch_coordinate.value_or(
        ttnn::MeshCoordinate::zero_coordinate(operation_attributes.device->shape().dims()));

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

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(cores.size() * 3);
    for (int i = 0; i < static_cast<int>(cores.size()); ++i) {
        const auto& core = cores[i];
        const uint32_t seed =
            operation_attributes.seed != 0 ? operation_attributes.seed + i + device_seed_offset : get_random_seed();
        dynamic_args.push_back({kComputeKernelIdx, core, kSeedArgIdx, seed});
        dynamic_args.push_back({kComputeKernelIdx, core, kFromArgIdx, from_bits});
        dynamic_args.push_back({kComputeKernelIdx, core, kToArgIdx, to_bits});
    }
    return dynamic_args;
}

}  // namespace ttnn::operations::rand
